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

// Test 4: SeedGYTD transitions the oracle's G_YTD > 0 gate from blocked → open.
//
// Assumption: a real direct-indexing client already has realised gains elsewhere
// (dividends re-invested, cap-gains from other accounts, etc.) that are available
// to offset harvested losses.  The simulation has no such activity modelled
// explicitly, so the engine seeds G_YTD to a representative starting value.
// Without this seed, G_YTD = 0 and the third oracle gate (gYtd > 0) is
// permanently closed — no harvest would ever fire.
public void Test_SeedGYTD_EnablesOracleGate()
{
    var state = new PortfolioState();

    // Baseline: G_YTD = 0 → oracle blocked even with a deep loss and clear wash clock
    int before = OracleBoundary.Label(
        unrealizedReturn: -0.05m,
        sigmaTE:          0.01f,
        gYtd:             state.G_YTD,   // = 0
        washClock:        999);
    Debug.Assert(before == 0,
        "Oracle must be blocked when G_YTD = 0 (no gains to offset harvested losses)");

    // Seed simulates $50K of prior-year or external gains available for offset
    state.SeedGYTD(50_000m);
    Debug.Assert(state.G_YTD == 50_000m,
        $"G_YTD should be 50 000 after seed, got {state.G_YTD}");

    // G_YTD > 0 gate now passes → oracle fires
    int after = OracleBoundary.Label(
        unrealizedReturn: -0.05m,
        sigmaTE:          0.01f,
        gYtd:             state.G_YTD,
        washClock:        999);
    Debug.Assert(after == 1,
        "Oracle must fire once G_YTD is seeded positive");

    Console.WriteLine("Test 4 passed: SeedGYTD opens the G_YTD > 0 oracle gate");
}

// Test 5: ResetForNewYear clears G_YTD; the engine must re-seed for the new
//         tax year or the oracle goes permanently dark again.
//
// Assumption: IRS tracks gains/losses on a calendar-year basis, so G_YTD resets
// to 0 at Jan 1.  Wash-sale clocks intentionally persist across the reset
// (the 30-day window can straddle year-end).  SimulationEngine calls SeedGYTD
// immediately after ResetForNewYear to replicate the client's ongoing external
// gain activity in the new year.
public void Test_SeedGYTD_ReSeeds_AfterYearEndReset()
{
    var state = new PortfolioState();
    state.SeedGYTD(50_000m);

    // Year-end: G_YTD clears to 0
    state.ResetForNewYear();
    Debug.Assert(state.G_YTD == 0m,
        $"ResetForNewYear must zero G_YTD, got {state.G_YTD}");

    // Without re-seed: oracle immediately blocked again
    int blocked = OracleBoundary.Label(-0.05m, 0.01f, state.G_YTD, 999);
    Debug.Assert(blocked == 0,
        "Oracle should be blocked right after year-end reset, before re-seeding");

    // SimulationEngine re-seeds after the reset → oracle fires again
    state.SeedGYTD(50_000m);
    int fired = OracleBoundary.Label(-0.05m, 0.01f, state.G_YTD, 999);
    Debug.Assert(fired == 1,
        "Oracle must fire again after re-seeding for the new tax year");

    Console.WriteLine("Test 5 passed: SeedGYTD after year-end reset re-enables oracle gate");
}
}