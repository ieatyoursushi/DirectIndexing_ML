using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Oracle;

/// <summary>
/// The mechanistic oracle  f* : X → {0,1}  — the tax-loss harvesting decision rule.
///
/// Formally this is the indicator of the harvest region Ω ⊂ X, and its threshold
/// hypersurfaces collectively define the boundary ∂Ω  (see PortfolioMath.md §4).
/// The oracle fires (returns 1) when the feature vector x sits strictly inside Ω,
/// i.e. when all four gate conditions hold simultaneously:
///
///   f*(x) = 𝟙[ℓ ≤ −θ₁]  ·  𝟙[σ_TE ≤ θ₂]  ·  𝟙[G_YTD > 0]  ·  𝟙[𝒲 ≥ 30]
///
/// This class is STATELESS — a pure function over lot geometry.
/// No fields, no constructors, no dependency injection.
/// </summary>
public static class OracleBoundary
{
    // ── Named thresholds (never magic numbers) ───────────────────────────────

    /// <summary>
    /// θ₁ — minimum unrealized loss to justify harvesting.
    /// Harvest fires when ℓ ≤ −LossThreshold (e.g. −0.02 = −2%).
    /// </summary>
    public const decimal LossThreshold    = 0.02m;

    /// <summary>
    /// θ₂ — maximum tolerated annualised tracking error.
    /// Harvest is blocked when σ_TE > TrackingErrorCap to prevent benchmark drift.
    /// </summary>
    public const decimal TrackingErrorCap = 0.05m;

    /// <summary>
    /// IRS wash-sale rule: cannot claim a loss if a substantially identical asset
    /// is purchased within 30 calendar days before or after the sale.
    /// </summary>
    public const int WashSaleDays = 30;

    // ── Core predicate ───────────────────────────────────────────────────────

    /// <summary>
    /// f*(x) = 𝟙[ℓ ≤ −θ₁]  ·  𝟙[σ_TE ≤ θ₂]  ·  𝟙[G_YTD &gt; 0]  ·  𝟙[𝒲 ≥ 30]
    ///
    /// During simulation, the engine extracts these four scalars from
    /// (Lot, PortfolioState, currentPrice, sigmaTE) before calling Label().
    /// </summary>
    /// <param name="unrealizedReturn">ℓ = (P_t − p_k)/p_k — negative for a loss</param>
    /// <param name="sigmaTE">σ_TE — current annualised tracking error vs benchmark</param>
    /// <param name="gYtd">G_YTD — net realised gain this calendar year; oracle blocked when ≤ 0</param>
    /// <param name="washClock">𝒲_t^{A_i} — days since last harvest of this ticker</param>
    /// <returns>1 if all gates open, 0 otherwise</returns>
    public static int Label(decimal unrealizedReturn, float sigmaTE, decimal gYtd, int washClock)
    {
        bool lossDeepEnough = unrealizedReturn  <= -LossThreshold;
        bool teWithinBudget = (decimal)sigmaTE  <= TrackingErrorCap;
        bool gainsToOffset  = gYtd             >   0m;
        bool washSaleClear  = washClock         >=  WashSaleDays;

        return (lossDeepEnough && teWithinBudget && gainsToOffset && washSaleClear) ? 1 : 0;
    }

    /// <summary>
    /// Convenience overload — works directly on a populated LotSnapshot.
    /// Useful for validation passes and tests; the simulation calls the
    /// scalar overload above since it constructs the snapshot in the same pass.
    /// </summary>
    public static int Label(LotSnapshot snapshot) =>
        Label(
            unrealizedReturn: (decimal)snapshot.L,
            sigmaTE:          snapshot.Sigma_TE,
            gYtd:             (decimal)snapshot.G_YTD,
            washClock:        snapshot.WashClock
        );
}
