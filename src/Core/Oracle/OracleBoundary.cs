using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Oracle;

/// <summary>
/// The mechanistic oracle  f* : X → {0,1}  — the tax-loss harvesting decision rule.
///
/// v0.25 (issue #23): two modes, selected by <see cref="OracleConfig.Mode"/>.
///
/// GATED (v0.2-legacy, ablation baseline):
///   f*(x) = 𝟙[ℓ ≤ −θ₁] · 𝟙[σ_TE ≤ θ₂] · 𝟙[G_YTD &gt; 0] · 𝟙[𝒲 ≥ 30]
///
/// SCALARIZED (industry-faithful composite — Wealthfront/Betterment form):
///   f*(x) = 𝟙[ℓ ≤ −θ₁] · 𝟙[𝒲 ≥ 30] · 𝟙[σ_TE ≤ θ_max] · 𝟙[U(x) &gt; 0]
///   U(x)  = taxValueₖ(ledgerₜ, hₖ, ℓₖ) − λ·σ_TE² − c_trade
///
/// Hard gates survive only where they encode a genuine legal rule or threshold
/// fact (loss depth, IRS §1091 wash clock, tail-risk TE ceiling); the gains
/// gate is removed — its information lives inside taxValue's offset-capacity
/// split (see TaxLedger). The boundary ∂Ω is the level set {x : U(x) = 0},
/// not the corner of an axis-aligned box.
///
/// Note on notation: taxValueₖ already carries the tax rates τ(h)/τ_future
/// internally (TaxLedger.ComputeTaxValue), so U applies no further rate factor.
///
/// This class is STATELESS — pure functions over lot geometry + config.
/// </summary>
public static class OracleBoundary
{
    // ── Legacy named thresholds ──────────────────────────────────────────────
    // Kept as the canonical v0.2 constants: they are the defaults inside
    // OracleConfig and the parameters of the legacy overload below.

    /// <summary>θ₁ — minimum unrealized loss to justify harvesting.</summary>
    public const decimal LossThreshold    = 0.02m;

    /// <summary>θ₂ — the v0.2 fine-grained TE cap (gated mode only).</summary>
    public const decimal TrackingErrorCap = 0.05m;

    /// <summary>IRS wash-sale rule: 30 calendar days.</summary>
    public const int WashSaleDays = 30;

    // ── Core predicate (config-driven) ───────────────────────────────────────

    /// <summary>
    /// Full oracle over explicit scalars. In gated mode <paramref name="taxValue"/>
    /// is ignored; in scalarized mode <paramref name="netRealizedYtd"/> is ignored
    /// (the ledger's information enters through taxValue instead).
    /// </summary>
    /// <param name="unrealizedReturn">ℓ = (P_t − p_k)/p_k — negative for a loss</param>
    /// <param name="sigmaTE">σ_TE — current annualised tracking error vs benchmark</param>
    /// <param name="netRealizedYtd">ledger net realized P&amp;L YTD (legacy G_YTD; gated gate 3)</param>
    /// <param name="washClock">𝒲_t^{A_i} — days since last harvest of this ticker</param>
    /// <param name="taxValue">taxValueₖ — capacity-aware harvest value in dollars (scalarized)</param>
    /// <param name="config">threshold/λ/mode configuration</param>
    public static int Label(
        decimal unrealizedReturn,
        float   sigmaTE,
        decimal netRealizedYtd,
        int     washClock,
        decimal taxValue,
        OracleConfig config)
    {
        bool lossDeepEnough = unrealizedReturn <= -config.LossThreshold;
        bool washSaleClear  = washClock        >=  config.WashSaleDays;

        if (config.Mode == OracleMode.Gated)
        {
            bool teWithinBudget = (decimal)sigmaTE <= config.LegacyTrackingErrorCap;
            bool gainsToOffset  = netRealizedYtd   >  0m;
            return (lossDeepEnough && teWithinBudget && gainsToOffset && washSaleClear) ? 1 : 0;
        }

        bool teBelowCeiling = (decimal)sigmaTE <= config.TrackingErrorCeiling;
        bool netBenefit     = Utility(taxValue, sigmaTE, config) > 0m;
        return (lossDeepEnough && washSaleClear && teBelowCeiling && netBenefit) ? 1 : 0;
    }

    /// <summary>
    /// U(x) = taxValue − λσ_TE² − c_trade. Exported as the Y_Utility label —
    /// a genuine intermediate object (the v0.4 RL per-decision reward), not
    /// just plumbing. f* = 𝟙[U &gt; 0] keeps the codomain {0,1}.
    /// </summary>
    public static decimal Utility(decimal taxValue, float sigmaTE, OracleConfig config)
    {
        decimal s = (decimal)sigmaTE;
        return taxValue - config.Lambda * s * s - config.CTrade;
    }

    // ── Legacy 4-scalar overload (gated semantics, v0.2 signature) ───────────

    /// <summary>
    /// v0.2-compatible gated oracle. Used by the engines' spectator-label
    /// bookkeeping and by the legacy tests; behaviour is bit-identical to the
    /// pre-v0.25 oracle.
    /// </summary>
    public static int Label(decimal unrealizedReturn, float sigmaTE, decimal gYtd, int washClock) =>
        Label(unrealizedReturn, sigmaTE, gYtd, washClock, taxValue: 0m, OracleConfig.Gated);

    // ── Snapshot overloads ───────────────────────────────────────────────────

    /// <summary>
    /// Config-driven convenience overload — the canonical call site coupling
    /// (GYTD_Redesign_Plan.md v2 §5.3): new oracle inputs ride as snapshot
    /// columns, so callers routed through here never change signature again.
    /// </summary>
    public static int Label(LotStateVector snapshot, OracleConfig config) =>
        Label(
            unrealizedReturn: (decimal)snapshot.L,
            sigmaTE:          snapshot.Sigma_TE,
            netRealizedYtd:   (decimal)snapshot.RealizedGainsYTD,
            washClock:        snapshot.WashClock,
            taxValue:         (decimal)snapshot.TaxValue,
            config:           config);

    /// <summary>Legacy snapshot overload — gated semantics (v0.2).</summary>
    public static int Label(LotStateVector snapshot) =>
        Label(snapshot, OracleConfig.Gated);
}
