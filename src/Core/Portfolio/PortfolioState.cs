namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// The complete portfolio state triple  𝒮_t = (μ_t, G_t^YTD, 𝒲_t):
///
///   μ_t     = OpenLots              — full lot measure across all assets
///   G_t^YTD = G_YTD                 — signed scalar: net realized P&amp;L this calendar year
///   𝒲_t     = _washClocks           — function  ticker → days since last harvest
///
/// Sign convention for G_YTD:
///   Harvesting a LOSING lot contributes a NEGATIVE delta (currentPrice &lt; CostBasis).
///   The oracle only fires when G_YTD &gt; 0, i.e. there are net realized gains
///   available to offset.  G_YTD therefore oscillates throughout the year as gains
///   are realised and losses are harvested against them.
///
/// AdvanceDay() implements the time evolution of 𝒲_t (increment every clock by 1).
/// HarvestLot() implements the state transition:
///   remove the atom from μ_t, update G_YTD, reset 𝒲_t^{A_i} ← 0.
/// </summary>
public class PortfolioState
{
    // G_t^YTD ∈ ℝ  (negative after net-loss harvests, positive when gains dominate)
    public decimal G_YTD { get; private set; } = 0m;

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

    /// <summary>Advance every wash-sale clock by one trading day.</summary>
    public void AdvanceDay()
    {
        foreach (var key in _washClocks.Keys.ToList())
            _washClocks[key]++;
    }

    // ─── State transitions ───────────────────────────────────────────────────

    public void OpenLot(Lot lot) =>
        OpenLots.Add(lot);

    /// <summary>
    /// Realise the P&amp;L of a lot and remove it from the measure.
    /// ΔG = q_k · (P_t − p_k)  — negative when harvesting a loss.
    /// </summary>
    public void HarvestLot(Lot lot, decimal currentPrice)
    {
        var gain = (currentPrice - lot.CostBasis) * lot.Shares;
        G_YTD        += gain;   // negative delta for a loss — sign is self-consistent
        lot.IsOpen    = false;
        OpenLots.Remove(lot);
        _washClocks[lot.Symbol] = 0;   // reset 𝒲_t^{A_i} ← 0
    }

    // ─── Derived quantities ──────────────────────────────────────────────────

    public decimal PortfolioValue(Dictionary<string, decimal> currentPrices) =>
        OpenLots.Sum(lot => lot.Shares * currentPrices[lot.Symbol]);

    /// <summary>
    /// Reset G_YTD at year boundary (Jan 1).
    /// Wash-sale clocks intentionally persist — the IRS window crosses year-end.
    /// Production would carry forward unused losses; simulation resets cleanly.
    /// </summary>
    public void ResetForNewYear() =>
        G_YTD = 0m;
}
