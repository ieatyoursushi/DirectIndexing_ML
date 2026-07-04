// Tests/TaxLedgerTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Portfolio;

/// <summary>
/// Unit tests for the TaxLedger state transitions and the taxValue_k formula
/// (v0.25, issue #23). Style matches the existing Debug.Assert smoke runners.
/// </summary>
public class TaxLedgerTests
{
    // Test 1: legacy equivalence — seed + harvest deltas reproduce the old
    // G_YTD trajectory exactly (the byte-identity invariant for gated mode).
    public void Test_LedgerNet_MatchesLegacyGYTD()
    {
        var ledger = new TaxLedger();
        ledger.RecordExternalGains(1_000_000m);      // old SeedGYTD
        ledger.RecordRealized(-250_000m);            // harvest a loss
        ledger.RecordRealized(-100_000m);

        Debug.Assert(ledger.RealizedGainsYTD == 650_000m,
            $"Expected 650000, got {ledger.RealizedGainsYTD}");
        Console.WriteLine("TaxLedger Test 1 passed: net-realized matches legacy G_YTD arithmetic");
    }

    // Test 2: year-end roll — net loss beyond the $3k ordinary allowance banks
    // into LossCarryforward, which SURVIVES; the annual accumulator resets.
    public void Test_RollYearEnd_BanksExcessLoss()
    {
        var ledger = new TaxLedger();
        ledger.RecordRealized(-10_000m);             // net loss year: −10k

        ledger.RollYearEnd();
        Debug.Assert(ledger.RealizedGainsYTD == 0m,
            $"Net must reset at year-end, got {ledger.RealizedGainsYTD}");
        Debug.Assert(ledger.LossCarryforward == 7_000m,
            $"Carryforward must bank 10k − 3k = 7k, got {ledger.LossCarryforward}");

        // Second year: small net loss fully absorbed by the ordinary allowance
        ledger.RecordRealized(-2_000m);
        ledger.RollYearEnd();
        Debug.Assert(ledger.LossCarryforward == 7_000m,
            $"Loss within the $3k allowance must not grow carryforward, got {ledger.LossCarryforward}");

        Console.WriteLine("TaxLedger Test 2 passed: year-end roll banks only the excess beyond $3k");
    }

    // Test 3: derived allowance/capacity mechanics through a loss-only year.
    public void Test_OffsetBudget_And_Capacity_DrawDown()
    {
        var ledger = new TaxLedger();

        // Fresh year, no gains: capacity is exactly the $3k ordinary allowance
        Debug.Assert(ledger.OrdinaryOffsetBudget == 3_000m, "Fresh budget must be $3,000");
        Debug.Assert(ledger.OffsetCapacity == 3_000m, "Loss-only capacity must be $3,000");

        // Harvest a $1k loss → allowance partially consumed
        ledger.RecordRealized(-1_000m);
        Debug.Assert(ledger.OrdinaryOffsetBudget == 2_000m,
            $"Expected 2000 remaining, got {ledger.OrdinaryOffsetBudget}");

        // Harvest far beyond the allowance → budget floors at 0, capacity 0
        ledger.RecordRealized(-50_000m);
        Debug.Assert(ledger.OrdinaryOffsetBudget == 0m, "Budget must floor at 0");
        Debug.Assert(ledger.OffsetCapacity == 0m, "Capacity must floor at 0");

        // With gains present, capacity = net gains + full remaining allowance
        var gainLedger = new TaxLedger();
        gainLedger.RecordExternalGains(20_000m);
        Debug.Assert(gainLedger.OffsetCapacity == 23_000m,
            $"Expected 20k + 3k = 23k, got {gainLedger.OffsetCapacity}");

        Console.WriteLine("TaxLedger Test 3 passed: budget/capacity draw down and floor correctly");
    }

    // Test 4: taxValue_k — capacity split between full-rate current use and
    // discounted banked slice; τ(h) short/long selection; winners are worth 0.
    public void Test_ComputeTaxValue_CapacitySplit_And_Rates()
    {
        var ledger = new TaxLedger();
        ledger.RecordExternalGains(10_000m);         // capacity = 10k + 3k = 13k

        // Small short-term loss, fully within capacity: τ_short · loss
        decimal small = ledger.ComputeTaxValue(lossDollars: 1_000m, holdingDays: 100);
        Debug.Assert(small == 0.37m * 1_000m,
            $"Expected 370, got {small}");

        // Same loss held long-term: τ_long · loss
        decimal smallLT = ledger.ComputeTaxValue(lossDollars: 1_000m, holdingDays: 400);
        Debug.Assert(smallLT == 0.20m * 1_000m,
            $"Expected 200, got {smallLT}");

        // Loss exceeding capacity: full rate on 13k, discounted future rate on the rest
        decimal big      = ledger.ComputeTaxValue(lossDollars: 20_000m, holdingDays: 100);
        decimal expected = 0.37m * 13_000m
                         + TaxLedger.TauFuture * 7_000m * TaxLedger.CarryforwardDiscount;
        Debug.Assert(big == expected,
            $"Expected {expected}, got {big}");

        // Banked dollars are worth strictly less than current-use dollars
        Debug.Assert(TaxLedger.TauFuture * TaxLedger.CarryforwardDiscount < 0.20m,
            "Discounted future rate must be below the long-term rate");

        // A lot not at a loss has no harvestable tax value
        Debug.Assert(ledger.ComputeTaxValue(0m, 100) == 0m, "No loss → taxValue 0");
        Debug.Assert(ledger.ComputeTaxValue(-5m, 100) == 0m, "Negative input → taxValue 0");

        Console.WriteLine("TaxLedger Test 4 passed: taxValue splits at capacity with correct rates");
    }

    // Test 5: PortfolioState integration — HarvestLot routes P&L through the
    // ledger and the legacy G_YTD alias stays value-identical.
    public void Test_PortfolioState_RoutesThroughLedger()
    {
        var state = new PortfolioState();
        state.SeedGYTD(5_000m);

        var lot = new Lot("AAPL", "Tech", costBasis: 100m, shares: 10, purchaseDayIndex: 0);
        state.OpenLot(lot);
        state.HarvestLot(lot, currentPrice: 90m);    // ΔG = −100

        Debug.Assert(state.G_YTD == 4_900m,
            $"Legacy alias must track ledger net, got {state.G_YTD}");
        Debug.Assert(state.Ledger.RealizedGainsYTD == 4_900m,
            $"Ledger must record harvest P&L, got {state.Ledger.RealizedGainsYTD}");

        // Year boundary: engine calls ResetForNewYear() then SeedGYTD(seed) —
        // net resets (no carryforward: year ended net-positive), then re-seeds.
        state.ResetForNewYear();
        Debug.Assert(state.G_YTD == 0m, "Net must reset at year-end");
        Debug.Assert(state.Ledger.LossCarryforward == 0m,
            "Net-positive year must not create carryforward");
        state.SeedGYTD(5_000m);
        Debug.Assert(state.G_YTD == 5_000m, "Re-seed must restore the legacy trajectory");

        Console.WriteLine("TaxLedger Test 5 passed: PortfolioState routes P&L through the ledger");
    }
}
