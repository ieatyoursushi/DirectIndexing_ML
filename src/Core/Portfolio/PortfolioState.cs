namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// The complete portfolio state triple  𝒮_t = (μ_t, ledger_t, 𝒲_t):
///
///   μ_t      = OpenLots              — full lot measure across all assets
///   ledger_t = Ledger                — TaxLedger: net realized P&amp;L, loss
///                                      carryforward, ordinary-offset budget
///   𝒲_t      = _washClocks           — function  ticker → days since last harvest
///
/// v0.25: the bare G_YTD scalar became the TaxLedger (issue #23). The G_YTD
/// property survives as a read alias of Ledger.RealizedGainsYTD — identical
/// semantics and values to the pre-ledger engine, so the gated oracle and all
/// logging are unchanged.
///
/// Sign convention for G_YTD / Ledger.RealizedGainsYTD:
///   Harvesting a LOSING lot contributes a NEGATIVE delta (currentPrice &lt; CostBasis).
///   The gated oracle only fires when G_YTD &gt; 0, i.e. there are net realized
///   gains available to offset.  It oscillates throughout the year as gains
///   are realised and losses are harvested against them.
///
/// AdvanceDay() implements the time evolution of 𝒲_t (increment every clock by 1).
/// HarvestLot() implements the state transition:
///   remove the atom from μ_t, record realized P&amp;L in the ledger, reset 𝒲_t^{A_i} ← 0.
/// </summary>
public class PortfolioState
{
    // ledger_t — deterministic Schedule D bookkeeping (see TaxLedger)
    public TaxLedger Ledger { get; } = new();

    // Legacy alias: G_t^YTD ∈ ℝ (negative after net-loss harvests, positive when gains dominate)
    public decimal G_YTD => Ledger.RealizedGainsYTD;

    // 𝒲_t : S → ℤ_{≥0}   (days since last harvest per ticker; 999 = never harvested)
    private readonly Dictionary<string, int> _washClocks = new();

    // μ_t = { atoms currently open }
    public List<Lot> OpenLots { get; } = new();

    // ─── Wash-sale helpers ───────────────────────────────────────────────────

    public int GetWashClock(string symbol) =>
        _washClocks.GetValueOrDefault(symbol, 999);

    // Blocks harvest when 𝒲_t^{A_i} < 30 (IRS 30-day wash-sale window)
    public bool IsWashSaleBlocked(string symbol) =>
        GetWashClock(symbol) < 30;

    // ─── Time evolution ──────────────────────────────────────────────────────

    /// <summary>Advance every wash-sale clock by one trading day. Days should be 1 unless testing</summary>
    public void AdvanceDay(int days = 1)
    {
        foreach (var key in _washClocks.Keys.ToList())
            _washClocks[key] += days;
    }

    // ─── State transitions ───────────────────────────────────────────────────

    public void OpenLot(Lot lot) =>
        OpenLots.Add(lot);

    /// <summary>
    /// Realise the P&amp;L of a lot and remove it from the measure.
    /// ΔG = q_k · (P_t − p_k)  — negative when harvesting a loss.
    /// secondary goal: two TE reductable routes of either repurchasing back after the 30d wash sale rule timer or replacing the harvested with a colinear asset that meets as many colinearity conditions as possible (like simlar sector, weight, covariance, etc.). The reducable can be chosen from the lots that got reduced away in the step 0 PCA/K-means dimensionality reduction.
    /// </summary>
    public void HarvestLot(Lot lot, decimal currentPrice)
    {
        var gain = (currentPrice - lot.CostBasis) * lot.Shares;
        Ledger.RecordRealized(gain);   // negative delta for a loss — sign is self-consistent
        lot.IsOpen    = false;
        OpenLots.Remove(lot);
        _washClocks[lot.Symbol] = 0;   // reset 𝒲_t^{A_i} ← 0
    }

    // ─── Derived quantities ──────────────────────────────────────────────────

    public decimal PortfolioValue(Dictionary<string, decimal> currentPrices) =>
        OpenLots.Sum(lot => lot.Shares * currentPrices[lot.Symbol]);

    /// <summary>
    /// Year boundary (Jan 1): roll the ledger — net loss beyond the $3k
    /// ordinary allowance banks into LossCarryforward (which survives),
    /// the annual accumulator resets to 0.
    /// Wash-sale clocks intentionally persist — the IRS window crosses year-end.
    /// </summary>
    public void ResetForNewYear() =>
        Ledger.RollYearEnd();

    /// <summary>
    /// Seed the ledger with an external gain amount — used at simulation start
    /// and after year-end resets to represent gains from other client activity
    /// (dividends, rebalancing, other account sales) that are not modelled
    /// explicitly. Without this, the gated oracle's G_YTD &gt; 0 gate is
    /// permanently closed. The scalarized oracle (v0.25+) runs with this OFF.
    /// </summary>
    public void SeedGYTD(decimal amount) => Ledger.RecordExternalGains(amount);
}
