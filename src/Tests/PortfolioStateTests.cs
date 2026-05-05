// Tests/PortfolioStateTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Oracle;
using DirectIndexing.Core.Portfolio;

public class PortfolioStateTests
{
// Test 1: G_YTD sign convention
public void Test_HarvestLoss_DecreasesGYTD()
{
    var state = new PortfolioState();
    var lot = new Lot("AAPL", "Tech", 
                       costBasis: 100m, shares: 10, purchaseDayIndex: 0);
    var lot2 = new Lot("MSFT", "Tech", 400m, 100, 0);
    state.OpenLot(lot);
    state.OpenLot(lot2);
    
    state.HarvestLot(lot, currentPrice: 90m);  // $10 loss × 10 shares = -$100
    state.HarvestLot(lot2, currentPrice: 331m);  // $69 loss × 100 shares = -$6900
    
    Debug.Assert(state.G_YTD == -7000m, 
        $"Expected -7000, got {state.G_YTD}");
    Console.WriteLine("Test 1 passed: G_YTD = -7000 after harvesting loss");
}

// Test 2: wash-sale clock ordering
public void Test_WashSaleClock_StartsAtZeroAfterHarvest()
{
    var state = new PortfolioState();
    var lot = new Lot("AAPL", "Tech", 100m, 10, 0);
    state.OpenLot(lot);
    state.HarvestLot(lot, 90m);
    
    Debug.Assert(state.GetWashClock("AAPL") == 0,
        $"Expected 0, got {state.GetWashClock("AAPL")}");
    
    state.AdvanceDay();
    Debug.Assert(state.GetWashClock("AAPL") == 1,
        $"Expected 1 after advance, got {state.GetWashClock("AAPL")}");
    
    state.AdvanceDay(30);
    //can buy (aka open lot) back after 30 days test
    Debug.Assert(state.GetWashClock("AAPL") == 31,
        $"Expected 31 after advance, got {state.GetWashClock("AAPL")}");
    Debug.Assert(!state.IsWashSaleBlocked("AAPL"),
        $"Expected false, got {state.IsWashSaleBlocked("AAPL")}");
    Console.WriteLine("Test 2 passed: wash-sale clock = 31 after advance, not blocking harvest");
}

// Test 3: G_YTD sign drives oracle gate — harvesting a loss pushes G_YTD negative,
//         which blocks the oracle from firing on any subsequent lot regardless of loss depth.
public void Test_OracleBlocked_WhenGYTD_IsNegative()
{
    var state = new PortfolioState();

    // Harvest a losing lot → G_YTD goes negative
    var lot = new Lot("MSFT", "Tech", 100m, 10, 0);
    state.OpenLot(lot);
    state.HarvestLot(lot, 80m);   // ΔG = (80−100)×10 = −200 → G_YTD = −200

    Debug.Assert(state.G_YTD == -200m,
        $"Expected G_YTD = -200, got {state.G_YTD}");

    // Right after harvest: wash-sale clock = 0 → IS blocked (expected)
    Debug.Assert(state.IsWashSaleBlocked("MSFT"),
        "Wash-sale should be blocking MSFT immediately after harvest (clock = 0)");

    // Advance past the wash-sale window so it's no longer the binding constraint
    state.AdvanceDay(30);
    Debug.Assert(!state.IsWashSaleBlocked("MSFT"),
        "Wash-sale should clear after 30 days");

    // Oracle must still return 0 because G_YTD < 0 blocks the gains-to-offset gate.
    // unrealizedReturn = −0.05 (5% loss — well past the 2% threshold)
    // sigmaTE = 0.01f (1% — well within the 5% budget)
    // G_YTD = −200 → oracle blocked on this gate
    int label = OracleBoundary.Label(
        unrealizedReturn: -0.05m,
        sigmaTE:           0.01f,
        gYtd:              state.G_YTD,
        washClock:         state.GetWashClock("MSFT")
    );

    Debug.Assert(label == 0,
        $"Oracle should return 0 when G_YTD < 0, got {label}");

    Console.WriteLine("Test 3 passed: oracle blocked (label=0) when G_YTD negative, even past wash-sale window");
}
}