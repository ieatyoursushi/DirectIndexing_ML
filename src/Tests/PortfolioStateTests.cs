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

// Test 2: the §1091 after-side on the calendar — a loss sale on D blocks buying the
//         ticker through D+30 inclusive; D+31 is the first clean day.
public void Test_WashSaleClock_StartsAtZeroAfterHarvest()
{
    var d0 = new DateOnly(2024, 3, 1);
    var state = new PortfolioState();
    state.SetDate(d0);
    var lot = new Lot("AAPL", "Tech", 100m, 10, 0, d0.AddDays(-400));
    state.OpenLot(lot);
    state.HarvestLot(lot, 90m);                       // a LOSS sale on d0

    Debug.Assert(state.DaysSinceLossSale("AAPL") == 0, $"Expected 0, got {state.DaysSinceLossSale("AAPL")}");
    Debug.Assert(!state.CanBuy("AAPL"), "no buy on the sale date");
    state.SetDate(d0.AddDays(30));
    Debug.Assert(!state.CanBuy("AAPL"), "day +30 is still inside the inclusive window");
    state.SetDate(d0.AddDays(31));
    Debug.Assert(state.CanBuy("AAPL"), "day +31 is the first clean day");
    Debug.Assert(state.EarliestBuyDate("AAPL") == d0.AddDays(31), "earliest buy date must be sale + 31");
    Console.WriteLine("Test 2 passed: after-side window is 30 calendar days inclusive (buy clean from day +31)");
}

// Test 3: year-end resets G_net; the §1091 window straddles Dec 31 (it is a date
//         difference, untouched by the ledger roll).
public void Test_YearEnd_BanksNetLoss_AndClocksPersist()
{
    var dec20 = new DateOnly(2024, 12, 20);
    var state = new PortfolioState();
    state.SetDate(dec20);
    var lot = new Lot("MSFT", "Tech", 100m, 100, 0, dec20.AddDays(-400));
    state.OpenLot(lot);
    state.HarvestLot(lot, 80m);                 // ΔG = (80−100)×100 = −2000

    state.ResetForNewYear();
    state.SetDate(new DateOnly(2024, 12, 30));
    Debug.Assert(state.Ledger.RealizedGainsYTD == 0m,
        $"Accumulator must reset at year-end, got {state.Ledger.RealizedGainsYTD}");
    Debug.Assert(state.Ledger.LossCarryforward == 0m,
        $"A $2k net loss is inside the $3k allowance — nothing to bank, got {state.Ledger.LossCarryforward}");
    Debug.Assert(state.DaysSinceLossSale("MSFT") == 10,
        $"Wash clock must survive the year boundary, got {state.DaysSinceLossSale("MSFT")}");

    Console.WriteLine("Test 3 passed: year-end resets G_net, keeps the §1091 window");
}
}