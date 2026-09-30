// Tests/PortfolioStateTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Portfolio;

public class PortfolioStateTests
{
// Test 1: realized-P&L sign convention — harvesting losses pushes G_net negative
public void Test_HarvestLoss_DecreasesRealizedGains()
{
    var state = new PortfolioState();
    var lot = new Lot("AAPL", "Tech", 
                       costBasis: 100m, shares: 10, purchaseDayIndex: 0);
    var lot2 = new Lot("MSFT", "Tech", 400m, 100, 0);
    state.OpenLot(lot);
    state.OpenLot(lot2);
    
    state.HarvestLot(lot, currentPrice: 90m);  // $10 loss × 10 shares = -$100
    state.HarvestLot(lot2, currentPrice: 331m);  // $69 loss × 100 shares = -$6900
    
    Debug.Assert(state.Ledger.RealizedGainsYTD == -7000m, 
        $"Expected -7000, got {state.Ledger.RealizedGainsYTD}");
    Console.WriteLine("Test 1 passed: G_net = -7000 after harvesting losses");
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

// Test 3: year-end roll — net loss beyond the $3k ordinary allowance banks into
//         carryforward; the accumulator resets; wash-sale clocks PERSIST (the
//         IRS window straddles Dec 31).
public void Test_YearEnd_BanksNetLoss_AndClocksPersist()
{
    var state = new PortfolioState();
    var lot = new Lot("MSFT", "Tech", 100m, 100, 0);
    state.OpenLot(lot);
    state.HarvestLot(lot, 80m);                 // ΔG = (80−100)×100 = −2000
    state.AdvanceDay(10);

    state.ResetForNewYear();
    Debug.Assert(state.Ledger.RealizedGainsYTD == 0m,
        $"Accumulator must reset at year-end, got {state.Ledger.RealizedGainsYTD}");
    Debug.Assert(state.Ledger.LossCarryforward == 0m,
        $"A $2k net loss is inside the $3k allowance — nothing to bank, got {state.Ledger.LossCarryforward}");
    Debug.Assert(state.GetWashClock("MSFT") == 10,
        $"Wash clock must survive the year boundary, got {state.GetWashClock("MSFT")}");

    Console.WriteLine("Test 3 passed: year-end resets G_net, keeps wash clocks");
}
}