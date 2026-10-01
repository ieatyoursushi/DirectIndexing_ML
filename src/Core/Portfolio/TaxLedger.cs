namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// Individual-investor tax ledger — the Schedule D state object (v0.25, issue #23;
/// character pools and carryforward consumption since v0.3-3, ROADMAP F8).
///
/// State (all $, this calendar year unless noted):
///   G^ST, G^LT  — signed net realized P&amp;L by §1222 character (gains +, losses −)
///   C^ST, C^LT  — capital-loss carryforward from PRIOR years, by character (≥ 0;
///                 §1212(b) preserves character)
///
/// Year-end netting S(ledger) — the law as Schedule D + the carryover worksheet run it:
///   1. carryforward enters as a loss of its own character:  n_S = G^ST − C^ST,  n_L = G^LT − C^LT
///   2. opposite-signed nets cross-net:                      n_S, n_L ← netted toward 0
///   3. a remaining net loss deducts up to $3,000 from ordinary income (§1211(b)),
///      taken from the short-term loss first
///   4. what is left carries forward with its character (§1212(b))
///   tax T = τ_ST·n_S⁺ + τ_LT·n_L⁺ − τ_ord·deduction        (τ_ord = τ_ST: ST gains are ordinary)
///
/// Carryforward is CONSUMED in step 1 — the year's gains absorb it before a new loss
/// can — which is the F8 bug of the v0.25 blended pool (banked, never used; the $3k line
/// valued at full rate even when carryforward already claimed it).
///
/// The ledger is deterministic bookkeeping — a state-transition function, structurally
/// identical to the wash-clock evolution. It is NOT an estimator and holds no latent
/// parameters. Valuation (<see cref="ComputeTaxValue(decimal,bool)"/>) is a
/// counterfactual difference of S, so the rate a loss earns is the rate of whatever it
/// actually offsets, never the harvested lot's own τ by fiat.
/// </summary>
// [math:ledger] — DataMemo/spec/SymbolTable.md
public sealed class TaxLedger
{
    // ── Tax-law constants ────────────────────────────────────────────────────

    /// <summary>Annual cap on net capital loss deducted against ordinary income (26 USC §1211(b)).</summary>
    public const decimal AnnualOrdinaryOffsetCap = 3_000m;

    /// <summary>τ_ST — ordinary marginal rate: short-term gains and the §1211(b) deduction (held ≤ 1 calendar year, §1222).</summary>
    public const decimal TauShortTerm = 0.37m;

    /// <summary>τ_LT — preferential rate on net long-term gains (held &gt; 1 calendar year).</summary>
    public const decimal TauLongTerm = 0.20m;

    /// <summary>τ_ord — rate of the §1211(b) ordinary-income deduction (= τ_ST for this client).</summary>
    public const decimal TauOrdinary = TauShortTerm;

    /// <summary>
    /// Rate applied to newly BANKED carryforward (losses used in a future tax year).
    /// Character-blind simplification: future absorption is assumed to offset
    /// long-term-rate gains.
    /// </summary>
    public const decimal TauFuture = 0.20m;

    /// <summary>
    /// δ — discount on the banked slice of a harvest's tax value. Constant, but
    /// conceptually a hazard-rate object:
    ///   δ ≈ Pr(loss absorbed by a gain before death) × time-value to absorption.
    /// It must never be silently assumed → 1 (carryforward is NOT worth face
    /// value for a low-outside-activity client; see Rev. Rul. 74-175 —
    /// carryforward dies with the taxpayer).
    /// </summary>
    public const decimal CarryforwardDiscount = 0.5m;

    // ── State ────────────────────────────────────────────────────────────────

    /// <summary>G^ST — signed net short-term realized P&amp;L this year.</summary>
    public decimal NetShortTerm { get; private set; }

    /// <summary>G^LT — signed net long-term realized P&amp;L this year.</summary>
    public decimal NetLongTerm { get; private set; }

    /// <summary>C^ST — short-term capital-loss carryforward from prior years (≥ 0).</summary>
    public decimal CarryShortTerm { get; private set; }

    /// <summary>C^LT — long-term capital-loss carryforward from prior years (≥ 0).</summary>
    public decimal CarryLongTerm { get; private set; }

    /// <summary>Σ of closed years' tax T (negative years = a net ordinary-income deduction).</summary>
    public decimal TaxPaid { get; private set; }

    /// <summary>
    /// W_tax = −Paid − T(open year) + τ_fut·δ·ΣC' — the tax-position potential
    /// (DataMemo/decisions/PolicyLayer_v04.md §4). Changes at trades AND at <see cref="RollYearEnd"/>:
    /// after the roll the new year's netting applies up to $3k of carried loss to ordinary income at
    /// once, so W jumps by (τ_ord − τ_fut·δ)·min(O_max, C) (with no gains) — the yearly conversion of
    /// banked carryforward into the full-rate ordinary deduction. ΔW over one loss harvest is
    /// exactly that harvest's <see cref="ComputeTaxValue(decimal,bool)"/>; over a gain sale it is
    /// the (negative) value of the tax the gain creates.
    /// </summary>
    // [math:tax_position] — DataMemo/spec/SymbolTable.md
    public decimal TaxPosition
    {
        get
        {
            var y = State.Close();
            return -TaxPaid - y.Tax + TauFuture * CarryforwardDiscount * (y.CarryShortTerm + y.CarryLongTerm);
        }
    }

    /// <summary>The ledger's four numbers as a value — what a frozen-state valuation needs.</summary>
    public LedgerState State => new(NetShortTerm, NetLongTerm, CarryShortTerm, CarryLongTerm);

    // ── Derived quantities ───────────────────────────────────────────────────

    /// <summary>G^net = G^ST + G^LT — signed net realized P&amp;L this year, both characters.</summary>
    public decimal RealizedGainsYTD => NetShortTerm + NetLongTerm;

    /// <summary>C = C^ST + C^LT — total carryforward on the books.</summary>
    public decimal LossCarryforward => CarryShortTerm + CarryLongTerm;

    /// <summary>
    /// O_t — the §1211(b) allowance NOT yet claimed if the year closed today:
    /// 3000 − deduction(S(ledger)). Carryforward claims it before a new harvest can.
    /// </summary>
    // [math:offset_budget] — DataMemo/spec/SymbolTable.md
    public decimal OrdinaryOffsetBudget => State.OrdinaryOffsetBudget;

    /// <summary>
    /// cap_t — dollars of a NEW harvested loss usable this tax year (either character):
    /// net gains still un-offset after carryforward and cross-netting, plus O_t.
    /// </summary>
    // [math:offset_capacity] — DataMemo/spec/SymbolTable.md
    public decimal OffsetCapacity => State.OffsetCapacity;

    // ── Transitions ──────────────────────────────────────────────────────────

    /// <summary>Realized P&amp;L of a sale: ΔG = q_k·(P_t − p_k) into the pool of its §1222 character.</summary>
    public void RecordRealized(decimal delta, bool isLongTerm)
    {
        if (isLongTerm) NetLongTerm  += delta;
        else            NetShortTerm += delta;
    }

    /// <summary>
    /// External/exogenous gains — client activity outside the simulated book (the hook
    /// for the v0.5 outside-gains client personas). Same pools as the book's own sales.
    /// </summary>
    public void RecordExternalGains(decimal amount, bool isLongTerm) => RecordRealized(amount, isLongTerm);

    /// <summary>
    /// Year-end (Jan 1) roll: run Schedule D netting, keep only the character-split
    /// carryforward, reset the annual pools.
    /// </summary>
    // [math:year_end_roll] — DataMemo/spec/SymbolTable.md
    public void RollYearEnd()
    {
        var y = State.Close();
        TaxPaid       += y.Tax;
        CarryShortTerm = y.CarryShortTerm;
        CarryLongTerm  = y.CarryLongTerm;
        NetShortTerm   = 0m;
        NetLongTerm    = 0m;
    }

    // ── Valuation ────────────────────────────────────────────────────────────

    /// <summary>
    /// taxValue_k — dollar value of harvesting a loss of <paramref name="lossDollars"/>
    /// right now, as a counterfactual difference of the year-end netting S:
    ///
    ///   [T(ledger) − T(ledger ⊕ loss)]                        — tax saved THIS year
    /// + τ_fut·δ·[C(ledger ⊕ loss) − C(ledger)]                — newly banked, discounted
    ///
    /// with "⊕ loss" = record −lossDollars in the lot's character pool. The rate the
    /// current-year slice earns is whatever it displaces: a short-term gain (τ_ST), a
    /// long-term gain (τ_LT), or the ordinary deduction (τ_ord) — and nothing at all if
    /// carryforward already absorbs those (F8).
    /// </summary>
    /// <param name="lossDollars">Unrealized loss in dollars, ≥ 0 (0 for lots not at a loss).</param>
    /// <param name="isLongTerm">§1222 character of the lot (see <see cref="IsLongTerm"/>).</param>
    public decimal ComputeTaxValue(decimal lossDollars, bool isLongTerm) =>
        ComputeTaxValue(lossDollars, isLongTerm, State);

    /// <summary>
    /// 26 USC §1222: a holding is long-term iff held MORE than one year, measured on the
    /// calendar (the holding period starts the day after acquisition, so a sale on the
    /// anniversary is still short-term). v0.3-2 (ROADMAP F6): previously a trading-day count
    /// compared with 365, i.e. ≈1.45 calendar years.
    /// </summary>
    // [math:lt_flag] — DataMemo/spec/SymbolTable.md
    public static bool IsLongTerm(DateOnly purchaseDate, DateOnly date) =>
        date > purchaseDate.AddYears(1);

    /// <summary>
    /// Static pure form — used by the soft-label forward closures, which freeze the
    /// ledger at the snapshot and re-value the loss along future price paths.
    /// </summary>
    // [math:g_tax] — DataMemo/spec/SymbolTable.md
    public static decimal ComputeTaxValue(decimal lossDollars, bool isLongTerm, LedgerState ledger)
    {
        if (lossDollars <= 0m) return 0m;

        var before = ledger.Close();
        var after  = ledger.With(-lossDollars, isLongTerm).Close();

        decimal savedNow = before.Tax - after.Tax;
        decimal banked   = (after.CarryShortTerm + after.CarryLongTerm)
                         - (before.CarryShortTerm + before.CarryLongTerm);
        return savedNow + TauFuture * banked * CarryforwardDiscount;
    }
}

/// <summary>
/// The ledger's four numbers as an immutable value: (G^ST, G^LT, C^ST, C^LT).
/// <see cref="Close"/> is the Schedule D year-end netting S — a pure function.
/// </summary>
public readonly record struct LedgerState(
    decimal NetShortTerm, decimal NetLongTerm, decimal CarryShortTerm, decimal CarryLongTerm)
{
    public LedgerState With(decimal delta, bool isLongTerm) => isLongTerm
        ? this with { NetLongTerm  = NetLongTerm  + delta }
        : this with { NetShortTerm = NetShortTerm + delta };

    /// <summary>
    /// S(ledger): Schedule D netting as if the year closed now →
    /// (tax, §1211(b) deduction, next year's character-split carryforward, the
    /// post-netting gains n_S⁺ + n_L⁺).
    /// </summary>
    // [math:schedule_d] — DataMemo/spec/SymbolTable.md
    public YearClose Close()
    {
        // 1. prior carryforward enters as a loss of its own character
        decimal nS = NetShortTerm - CarryShortTerm;
        decimal nL = NetLongTerm  - CarryLongTerm;

        // 2. cross-netting of opposite-signed character nets
        if (nS < 0m && nL > 0m) { decimal x = Math.Min(-nS, nL); nS += x; nL -= x; }
        if (nL < 0m && nS > 0m) { decimal x = Math.Min(-nL, nS); nL += x; nS -= x; }

        // 3. §1211(b) deduction, short-term loss first
        decimal lossS = Math.Max(0m, -nS), lossL = Math.Max(0m, -nL);
        decimal deduction = Math.Min(TaxLedger.AnnualOrdinaryOffsetCap, lossS + lossL);
        decimal fromS = Math.Min(deduction, lossS);

        // 4. character-preserving carryforward (§1212(b))
        decimal gains = Math.Max(0m, nS) + Math.Max(0m, nL);
        decimal tax   = TaxLedger.TauShortTerm * Math.Max(0m, nS)
                      + TaxLedger.TauLongTerm  * Math.Max(0m, nL)
                      - TaxLedger.TauOrdinary  * deduction;
        return new YearClose(tax, deduction, lossS - fromS, lossL - (deduction - fromS), gains);
    }

    public decimal OrdinaryOffsetBudget => TaxLedger.AnnualOrdinaryOffsetCap - Close().Deduction;

    public decimal OffsetCapacity
    {
        get { var y = Close(); return y.GainsAfterNetting + (TaxLedger.AnnualOrdinaryOffsetCap - y.Deduction); }
    }
}

/// <summary>The result of <see cref="LedgerState.Close"/>.</summary>
public readonly record struct YearClose(
    decimal Tax, decimal Deduction, decimal CarryShortTerm, decimal CarryLongTerm, decimal GainsAfterNetting);
