namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// Individual-investor tax ledger — the tax-law state object that replaces the
/// bare G_YTD scalar (v0.25, issue #23).
///
/// Encodes the Schedule D mechanics that matter for tax-loss harvesting in the
/// individual (indefinite-carryforward) regime:
///
///   1. RealizedGainsYTD — signed net realized P&amp;L this calendar year
///      (external gains positive, harvested losses negative). Resets at
///      year-end. Identical semantics to the pre-v0.25 G_YTD scalar.
///   2. Up to $3,000/yr of net capital loss offsets ordinary income
///      (26 USC §1211(b)); the unused remainder is OrdinaryOffsetBudget.
///   3. Net loss beyond the ordinary allowance carries forward indefinitely
///      (26 USC §1212(b)) — LossCarryforward survives year-end. This is the
///      mechanic the old constant re-seed got wrong: a loss harvested with no
///      gains available is banked, never wasted.
///
/// The ledger is deterministic bookkeeping over already-simulated events —
/// a state-transition function (state, decision) → state', structurally
/// identical to the wash-clock evolution. It is NOT an estimator and holds
/// no latent parameters.
///
/// v0.25 simplification (deliberate, documented): short- and long-term
/// realizations share one blended pool with a single τ(h) applied at harvest.
/// Real Schedule D runs two typed pools with cross-netting and
/// character-preserving carryforward; splitting them is a bounded future
/// extension.
/// </summary>
public sealed class TaxLedger
{
    // ── Tax-law constants ────────────────────────────────────────────────────

    /// <summary>Annual cap on net capital loss deducted against ordinary income (26 USC §1211(b)).</summary>
    public const decimal AnnualOrdinaryOffsetCap = 3_000m;

    /// <summary>τ short-term — ordinary marginal rate applied when h &lt; 365 days.</summary>
    public const decimal TauShortTerm = 0.37m;

    /// <summary>τ long-term — preferential rate applied when h ≥ 365 days.</summary>
    public const decimal TauLongTerm = 0.20m;

    /// <summary>
    /// Rate applied to the banked slice of a harvest (losses used in a future
    /// tax year). Blended-pool simplification: future absorption is assumed to
    /// offset long-term-rate gains.
    /// </summary>
    public const decimal TauFuture = 0.20m;

    /// <summary>
    /// δ — discount on the banked slice of a harvest's tax value. Constant in
    /// v0.25, but conceptually a hazard-rate object:
    ///   δ ≈ Pr(loss absorbed by a gain before death) × time-value to absorption.
    /// It must never be silently assumed → 1 (carryforward is NOT worth face
    /// value for a low-outside-activity client; see Rev. Rul. 74-175 —
    /// carryforward dies with the taxpayer).
    /// </summary>
    public const decimal CarryforwardDiscount = 0.5m;

    // ── State ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Signed net realized P&amp;L this calendar year. External/seeded gains push
    /// it up, harvested losses push it down — exactly the pre-v0.25 G_YTD.
    /// </summary>
    public decimal RealizedGainsYTD { get; private set; }

    /// <summary>
    /// Accumulated net losses beyond each year's ordinary allowance.
    /// SURVIVES year-end — the field the old constant re-seed got wrong.
    /// </summary>
    public decimal LossCarryforward { get; private set; }

    // ── Derived quantities ───────────────────────────────────────────────────

    /// <summary>
    /// Remaining ordinary-income offset allowance this year:
    /// max(0, $3,000 − net loss already realized). Resets implicitly at
    /// year-end because it is derived from RealizedGainsYTD.
    /// </summary>
    public decimal OrdinaryOffsetBudget =>
        Math.Max(0m, AnnualOrdinaryOffsetCap - Math.Max(0m, -RealizedGainsYTD));

    /// <summary>
    /// offsetCapacity_t — dollars of a NEW harvested loss usable this tax year:
    /// net gains still un-offset plus the remaining ordinary allowance.
    /// </summary>
    public decimal OffsetCapacity =>
        Math.Max(0m, RealizedGainsYTD) + OrdinaryOffsetBudget;

    // ── Transitions ──────────────────────────────────────────────────────────

    /// <summary>Realized P&amp;L from a harvest/sale: ΔG = q_k · (P_t − p_k), negative for a loss.</summary>
    public void RecordRealized(decimal delta) => RealizedGainsYTD += delta;

    /// <summary>
    /// External/exogenous gains — the legacy G_YTD "seed" representing client
    /// activity outside the simulated book. Deliberately does NOT net against
    /// LossCarryforward, so gated-mode behaviour stays bit-identical to the
    /// pre-ledger engine; carryforward netting against endogenous gains arrives
    /// with the sell-winner trim process (v0.3+).
    /// </summary>
    public void RecordExternalGains(decimal amount) => RealizedGainsYTD += amount;

    /// <summary>
    /// Year-end (Jan 1) roll: net loss beyond the ordinary allowance banks into
    /// LossCarryforward; the annual accumulator resets. Mirrors Schedule D
    /// year-boundary netting under the blended-pool simplification.
    /// </summary>
    public void RollYearEnd()
    {
        decimal netLoss = Math.Max(0m, -RealizedGainsYTD);
        LossCarryforward += Math.Max(0m, netLoss - AnnualOrdinaryOffsetCap);
        RealizedGainsYTD  = 0m;
    }

    // ── Valuation ────────────────────────────────────────────────────────────

    /// <summary>
    /// taxValue_k — dollar value of harvesting a loss of <paramref name="lossDollars"/>
    /// right now, given current ledger state:
    ///
    ///   τ(h)·min(loss, offsetCapacity)              — usable THIS year, full rate
    /// + τ_future·max(loss − offsetCapacity, 0)·δ    — banked, discounted
    ///
    /// Supersedes the v0.2 TaxAlpha = τ(h)·|P&amp;L|·𝟙[G_YTD&gt;0], which (a) counted
    /// winners' |gains| as if they were harvestable losses and (b) valued every
    /// loss dollar at the full current-year rate regardless of offset capacity.
    /// </summary>
    /// <param name="lossDollars">Unrealized loss in dollars, ≥ 0 (0 for lots not at a loss).</param>
    /// <param name="holdingDays">Holding period h — selects the short/long-term rate τ(h).</param>
    public decimal ComputeTaxValue(decimal lossDollars, int holdingDays) =>
        ComputeTaxValue(lossDollars, holdingDays, OffsetCapacity);

    /// <summary>
    /// Static pure form — used by the soft-label forward closures, which freeze
    /// offsetCapacity at the snapshot and re-value the loss along future price
    /// paths without holding a ledger reference.
    /// </summary>
    public static decimal ComputeTaxValue(decimal lossDollars, int holdingDays, decimal offsetCapacity)
    {
        if (lossDollars <= 0m) return 0m;

        decimal tau      = holdingDays >= 365 ? TauLongTerm : TauShortTerm;
        decimal usedNow  = Math.Min(lossDollars, offsetCapacity);
        decimal banked   = Math.Max(lossDollars - offsetCapacity, 0m);

        return tau * usedNow + TauFuture * banked * CarryforwardDiscount;
    }
}
