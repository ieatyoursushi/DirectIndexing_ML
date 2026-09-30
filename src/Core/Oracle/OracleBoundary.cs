using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Oracle;

/// <summary>
/// The mechanistic oracle  f* : X → {0,1}  — the tax-loss harvesting decision rule
/// (the scalarized, industry-faithful composite of v0.25 / issue #23 — the
/// Wealthfront/Betterment form):
///
///   f*(x) = 𝟙[ℓ ≤ −θ₁] · 𝟙[𝒲 ≥ 30] · 𝟙[σ_TE ≤ θ_max] · 𝟙[U(x) &gt; 0]
///   U(x)  = taxValueₖ(ledgerₜ, hₖ, ℓₖ) − λ·σ_TE² − c_trade
///
/// Hard gates survive only where they encode a genuine legal rule or threshold
/// fact (loss depth, IRS §1091 wash clock, tail-risk TE ceiling). Realized gains
/// are not a gate — their information lives inside taxValue's offset-capacity
/// split (see TaxLedger). The boundary ∂Ω is the level set {x : U(x) = 0}, not
/// the corner of an axis-aligned box. (The v0.2 four-gate "gated" oracle with its
/// G_YTD &gt; 0 gate was retired in the pre-v0.3 downsizing — see
/// DataMemo/archive/RetiredComponents.md §6.)
///
/// Note on notation: taxValueₖ already carries the tax rates τ(h)/τ_future
/// internally (TaxLedger.ComputeTaxValue), so U applies no further rate factor.
///
/// This class is STATELESS — pure functions over lot geometry + config.
/// </summary>
public static class OracleBoundary
{
    /// <summary>θ₁ — minimum unrealized loss to justify harvesting (default of <see cref="OracleConfig.LossThreshold"/>).</summary>
    public const decimal LossThreshold = 0.02m;

    /// <summary>
    /// IRS §1091 wash-sale window, in the units of the simulator's wash clock
    /// (one tick per simulated trading day — see ROADMAP finding F7 on units).
    /// </summary>
    public const int WashSaleDays = 30;

    /// <summary>
    /// The oracle over explicit scalars.
    /// </summary>
    /// <param name="unrealizedReturn">ℓ = (P_t − p_k)/p_k — negative for a loss</param>
    /// <param name="sigmaTE">σ_TE — current annualised tracking error vs benchmark</param>
    /// <param name="washClock">𝒲_t^{A_i} — days since last harvest of this ticker</param>
    /// <param name="taxValue">taxValueₖ — capacity-aware harvest value in dollars</param>
    /// <param name="config">thresholds and the economic terms of U</param>
    public static int Label(
        decimal unrealizedReturn,
        float   sigmaTE,
        int     washClock,
        decimal taxValue,
        OracleConfig config)
    {
        bool lossDeepEnough = unrealizedReturn <= -config.LossThreshold;
        bool washSaleClear  = washClock        >=  config.WashSaleDays;
        bool teBelowCeiling = (decimal)sigmaTE <= config.TrackingErrorCeiling;
        bool netBenefit     = Utility(taxValue, sigmaTE, config) > 0m;
        return (lossDeepEnough && washSaleClear && teBelowCeiling && netBenefit) ? 1 : 0;
    }

    /// <summary>
    /// U(x) = taxValue − λσ_TE² − c_trade. Exported as the Y_Utility label.
    /// Note (ROADMAP finding F3): U is a per-lot, one-step score. Summing it across
    /// lots charges the shared σ_TE once per harvested lot, so it is not by itself
    /// the v0.4 RL reward — that is pinned at portfolio level in the SymbolTable.
    /// f* = 𝟙[U &gt; 0] keeps the codomain {0,1}.
    /// </summary>
    public static decimal Utility(decimal taxValue, float sigmaTE, OracleConfig config)
    {
        decimal s = (decimal)sigmaTE;
        return taxValue - config.Lambda * s * s - config.CTrade;
    }

    /// <summary>
    /// Snapshot overload — the canonical call-site coupling (GYTD_Redesign_Plan.md
    /// v2 §5.3): new oracle inputs ride as snapshot columns, so callers routed
    /// through here never change signature again.
    /// </summary>
    public static int Label(LotStateVector snapshot, OracleConfig config) =>
        Label(
            unrealizedReturn: (decimal)snapshot.L,
            sigmaTE:          snapshot.Sigma_TE,
            washClock:        snapshot.WashClock,
            taxValue:         (decimal)snapshot.TaxValue,
            config:           config);
}
