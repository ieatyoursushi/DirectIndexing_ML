namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// Represents a single tax lot — one Dirac atom in the measure-valued
/// portfolio representation.  Formally, each open lot for asset A_i is an
/// atom q_k · δ_(p_k, s_k) in the measure μ_t^{A_i}, where:
///   q_k  = Shares           (quantity / mass of the atom)
///   p_k  = CostBasis        (purchase price — the basis of the atom)
///   s_k  = PurchaseDayIndex (purchase day — the time support point)
/// </summary>
public class Lot
{
    public string  Symbol          { get; init; }
    public string  Sector          { get; init; }
    public decimal CostBasis       { get; init; }   // price per share at purchase
    public int     Shares          { get; init; }
    public int     PurchaseDayIndex { get; init; }  // simulation day index (not DateTime)
    public bool    IsOpen          { get; set; } = true;

    public Lot(string symbol, string sector,
               decimal costBasis, int shares, int purchaseDayIndex)
    {
        Symbol           = symbol;
        Sector           = sector;
        CostBasis        = costBasis;
        Shares           = shares;
        PurchaseDayIndex = purchaseDayIndex;
    }

    // ℓ = (P_t − p_k) / p_k  ∈ (−1, ∞)
    // Negative for a harvestable lot (price has fallen below cost basis).
    public decimal UnrealizedReturn(decimal currentPrice) =>
        (currentPrice - CostBasis) / CostBasis;

    // h = t − s_k  ∈ ℤ_{≥0}
    public int HoldingPeriod(int currentDay) =>
        currentDay - PurchaseDayIndex;

    // s = 𝟙[h ≥ 365]  ∈ {0, 1}
    public bool IsLongTerm(int currentDay) =>
        HoldingPeriod(currentDay) >= 365;
}
