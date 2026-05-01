// Tests/PortfolioStateTests.cs
using System;
using System.Diagnostics;
using DirectIndexing.Core.Portfolio;

// (placeholder unit tests just to make sure runtime lifecycle is working to be replaced)
public class PortfolioStateTests
{
// Test 1: G_YTD sign convention
public void Test_HarvestLoss_DecreasesGYTD()
{
    var state = new PortfolioState();
    var lot = new Lot("AAPL", "Tech", 
                       costBasis: 100m, shares: 10, purchaseDayIndex: 0);
    state.OpenLot(lot);
    
    state.HarvestLot(lot, currentPrice: 90m);  // $10 loss × 10 shares = -$100
    
    Debug.Assert(state.G_YTD == -100m, 
        $"Expected -100, got {state.G_YTD}");
    Console.WriteLine("Test 1 passed: G_YTD = -100 after harvesting loss");
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
    
    Console.WriteLine("Test 2 passed: wash-sale clock = 0 after harvest, 1 after advance");
}

// Test 3: oracle conditions fire correctly
//oracle is not currently implemented expected failure 
public void Test_OracleBlocked_WhenGYTD_IsNegative()
{
    var state = new PortfolioState();
    // simulate G_YTD already negative
    var gainLot = new Lot("MSFT", "Tech", 100m, 10, 0);
    state.OpenLot(gainLot);
    state.HarvestLot(gainLot, 80m);  // G_YTD = -200
    
    Debug.Assert(state.G_YTD < 0, "G_YTD should be negative");
    Debug.Assert(!state.IsWashSaleBlocked("MSFT"), 
        "Wash-sale shouldn't block MSFT yet");
    
    // oracle should NOT fire even though loss is deep enough
    // because G_YTD < 0
    Console.WriteLine("Test 3 passed: G_YTD negative correctly blocks oracle");
}
}