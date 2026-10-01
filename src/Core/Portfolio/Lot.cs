namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// Represents a single tax lot — one Dirac atom in the measure-valued
/// portfolio representation.  Formally, each open lot for asset A_i is an
/// atom q_k · δ_(p_k, s_k) in the measure μ_t^{A_i}, where:
///   q_k  = Shares           (quantity / mass of the atom)
///   p_k  = CostBasis        (purchase price — the basis of the atom)
///   s_k  = PurchaseDayIndex (purchase day — the time support point)
/// </summary>
// [math:lot] — DataMemo/spec/SymbolTable.md
public class Lot
{
    public string  Symbol          { get; init; }
    public string  Sector          { get; init; }
    public decimal CostBasis       { get; init; }   // price per share at purchase
    public int     Shares          { get; init; }
    public int     PurchaseDayIndex { get; init; }  // simulation day index (trading days)
    /// <summary>Calendar date of acquisition — what §1091 and §1222 are measured in.</summary>
    public DateOnly PurchaseDate    { get; init; }
    public bool    IsOpen          { get; set; } = true;

    public Lot(string symbol, string sector,
               decimal costBasis, int shares, int purchaseDayIndex,
               DateOnly purchaseDate = default)
    {
        Symbol           = symbol;
        Sector           = sector;
        CostBasis        = costBasis;
        Shares           = shares;
        PurchaseDayIndex = purchaseDayIndex;
        PurchaseDate     = purchaseDate;
    }

    // ℓ = (P_t − p_k) / p_k  ∈ (−1, ∞)
    // Negative for a harvestable lot (price has fallen below cost basis).
    public decimal UnrealizedReturn(decimal currentPrice) =>
        (currentPrice - CostBasis) / CostBasis;

    // h = t − s_k  ∈ ℤ_{≥0}  in TRADING days (a feature/age coordinate — NOT the §1222
    // holding period, which is calendar-based: IsLongTerm below)
    // [math:holding_period] — DataMemo/spec/SymbolTable.md
    public int HoldingPeriod(int currentDay) =>
        currentDay - PurchaseDayIndex;

    // s = 𝟙[date > purchase date + 1 calendar year]  ∈ {0, 1}  (§1222, see TaxLedger.IsLongTerm)
    public bool IsLongTerm(DateOnly today) =>
        TaxLedger.IsLongTerm(PurchaseDate, today);
}
