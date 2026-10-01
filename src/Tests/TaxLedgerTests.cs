// Tests/TaxLedgerTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Portfolio;

/// <summary>
/// Unit tests for the TaxLedger state transitions and the taxValue_k formula
/// (v0.25, issue #23). Style matches the existing Debug.Assert smoke runners.
/// </summary>
public class TaxLedgerTests
{
    // Test 1: the pools accumulate by §1222 character; G^net is their signed sum.
    public void Test_LedgerNet_AccumulatesSignedRealized()
    {
        var ledger = new TaxLedger();
        ledger.RecordExternalGains(1_000_000m, isLongTerm: true);   // client gains realized elsewhere
        ledger.RecordRealized(-250_000m, isLongTerm: false);         // harvest a short-term loss
        ledger.RecordRealized(-100_000m, isLongTerm: true);

        Debug.Assert(ledger.NetShortTerm == -250_000m && ledger.NetLongTerm == 900_000m,
            $"pools: expected −250k / 900k, got {ledger.NetShortTerm} / {ledger.NetLongTerm}");
        Debug.Assert(ledger.RealizedGainsYTD == 650_000m,
            $"Expected 650000, got {ledger.RealizedGainsYTD}");
        Console.WriteLine("TaxLedger Test 1 passed: character pools; net realized = signed sum");
    }

    // Test 2: year-end roll — the $3k deduction comes out of the ST loss first, and the
    // remainder carries forward WITH character (§1212(b)).
    public void Test_RollYearEnd_BanksExcessLoss()
    {
        var ledger = new TaxLedger();
        ledger.RecordRealized(-2_000m, isLongTerm: false);
        ledger.RecordRealized(-8_000m, isLongTerm: true);           // net loss year: −10k
        ledger.RollYearEnd();
        Debug.Assert(ledger.RealizedGainsYTD == 0m, $"pools must reset, got {ledger.RealizedGainsYTD}");
        Debug.Assert(ledger.CarryShortTerm == 0m && ledger.CarryLongTerm == 7_000m,
            $"$3k = 2k ST + 1k LT → carry 0 ST / 7k LT, got {ledger.CarryShortTerm} / {ledger.CarryLongTerm}");

        // Year 2: a 5k LT gain CONSUMES 5k of the LT carryforward; the rest stays banked
        ledger.RecordRealized(5_000m, isLongTerm: true);
        ledger.RollYearEnd();
        Debug.Assert(ledger.CarryLongTerm == 0m,
            $"7k carry − 5k gain = 2k net loss ≤ $3k → fully deducted, carry 0, got {ledger.CarryLongTerm}");

        // Year 3: a ST gain cross-nets against LT carryforward (character is kept, not walled off)
        var l3 = new TaxLedger();
        l3.RecordRealized(-20_000m, isLongTerm: true); l3.RollYearEnd();       // carry LT 17k
        l3.RecordRealized( 10_000m, isLongTerm: false); l3.RollYearEnd();      // ST gain 10k absorbs 10k, $3k deducted
        Debug.Assert(l3.CarryShortTerm == 0m && l3.CarryLongTerm == 4_000m,
            $"17k LT carry − 10k ST gain − 3k = 4k LT, got {l3.CarryShortTerm} / {l3.CarryLongTerm}");
        Console.WriteLine("TaxLedger Test 2 passed: Schedule D roll — ST-first deduction, character carryforward, consumption, cross-netting");
    }

    // Test 3: derived allowance/capacity — carryforward claims the $3k line before new losses (F8).
    public void Test_OffsetBudget_And_Capacity_DrawDown()
    {
        var ledger = new TaxLedger();
        Debug.Assert(ledger.OrdinaryOffsetBudget == 3_000m, "Fresh budget must be $3,000");
        Debug.Assert(ledger.OffsetCapacity == 3_000m, "Loss-only capacity must be $3,000");
        ledger.RecordRealized(-1_000m, isLongTerm: false);
        Debug.Assert(ledger.OrdinaryOffsetBudget == 2_000m, $"Expected 2000 remaining, got {ledger.OrdinaryOffsetBudget}");
        ledger.RecordRealized(-50_000m, isLongTerm: false);
        Debug.Assert(ledger.OffsetCapacity == 0m, "Capacity must floor at 0");

        var gainLedger = new TaxLedger();
        gainLedger.RecordExternalGains(20_000m, isLongTerm: true);
        Debug.Assert(gainLedger.OffsetCapacity == 23_000m, $"Expected 20k + 3k = 23k, got {gainLedger.OffsetCapacity}");

        // F8: a 10k carryforward already uses this year's $3k line — and 10k of any gains
        var carried = new TaxLedger();
        carried.RecordRealized(-13_000m, isLongTerm: false); carried.RollYearEnd();   // carry ST 10k
        Debug.Assert(carried.OrdinaryOffsetBudget == 0m && carried.OffsetCapacity == 0m,
            $"carryforward must claim the $3k line, got budget {carried.OrdinaryOffsetBudget}");
        carried.RecordExternalGains(12_000m, isLongTerm: true);
        Debug.Assert(carried.OffsetCapacity == 2_000m + 3_000m,
            $"12k gain − 10k carry = 2k un-offset + $3k line, got {carried.OffsetCapacity}");
        Console.WriteLine("TaxLedger Test 3 passed: budget/capacity net of carryforward (F8)");
    }

    // Test 4: taxValue as a counterfactual difference — a loss earns the rate of what it
    // DISPLACES (not its own τ), capacity splits current/banked, carryforward crowds out.
    public void Test_ComputeTaxValue_CapacitySplit_And_Rates()
    {
        decimal banked = TaxLedger.TauFuture * TaxLedger.CarryforwardDiscount;   // 0.10 per banked $

        var stGains = new TaxLedger(); stGains.RecordExternalGains(10_000m, isLongTerm: false);
        var ltGains = new TaxLedger(); ltGains.RecordExternalGains(10_000m, isLongTerm: true);

        // offsetting a ST gain saves τ_ST, whatever the harvested lot's own character
        Debug.Assert(stGains.ComputeTaxValue(1_000m, isLongTerm: false) == 370m, "ST loss vs ST gain → 370");
        Debug.Assert(stGains.ComputeTaxValue(1_000m, isLongTerm: true)  == 370m, "LT loss cross-nets vs ST gain → 370");
        // offsetting a LT gain saves τ_LT
        Debug.Assert(ltGains.ComputeTaxValue(1_000m, isLongTerm: true)  == 200m, "LT loss vs LT gain → 200");
        Debug.Assert(ltGains.ComputeTaxValue(1_000m, isLongTerm: false) == 200m, "ST loss cross-nets vs LT gain → 200");

        // beyond the gains: the $3k ordinary line at τ_ord, then banked
        decimal big = stGains.ComputeTaxValue(20_000m, isLongTerm: false);
        decimal expected = 0.37m * 10_000m + TaxLedger.TauOrdinary * 3_000m + banked * 7_000m;
        Debug.Assert(big == expected, $"Expected {expected}, got {big}");

        // no gains, no carry: the $3k line, then banked
        var empty = new TaxLedger();
        Debug.Assert(empty.ComputeTaxValue(1_000m, false) == 370m, "inside the $3k line → τ_ord");
        Debug.Assert(empty.ComputeTaxValue(5_000m, true) == 0.37m * 3_000m + banked * 2_000m, "$3k line then banked");

        // F8: with carryforward on the books, the $3k line is taken — the loss is only banked
        var carried = new TaxLedger();
        carried.RecordRealized(-13_000m, isLongTerm: false); carried.RollYearEnd();
        Debug.Assert(carried.ComputeTaxValue(1_000m, false) == banked * 1_000m,
            $"F8: expected {banked * 1_000m}, got {carried.ComputeTaxValue(1_000m, false)}");

        Debug.Assert(banked < TaxLedger.TauLongTerm, "Discounted future rate must be below the long-term rate");
        Debug.Assert(empty.ComputeTaxValue(0m, false) == 0m && empty.ComputeTaxValue(-5m, false) == 0m,
            "No loss → taxValue 0");
        Console.WriteLine("TaxLedger Test 4 passed: taxValue = displaced-rate saving + discounted banked slice");
    }

    // Test 5: PortfolioState integration — HarvestLot routes P&L through the
    // ledger; a net-positive year resets without creating carryforward.
    public void Test_PortfolioState_RoutesThroughLedger()
    {
        var state = new PortfolioState();
        state.Ledger.RecordExternalGains(5_000m, isLongTerm: true);

        var lot = new Lot("AAPL", "Tech", costBasis: 100m, shares: 10, purchaseDayIndex: 0);
        state.OpenLot(lot);
        state.HarvestLot(lot, currentPrice: 90m);    // ΔG = −100

        Debug.Assert(state.Ledger.RealizedGainsYTD == 4_900m,
            $"Ledger must record harvest P&L, got {state.Ledger.RealizedGainsYTD}");

        state.ResetForNewYear();
        Debug.Assert(state.Ledger.RealizedGainsYTD == 0m, "Net must reset at year-end");
        Debug.Assert(state.Ledger.LossCarryforward == 0m,
            "Net-positive year must not create carryforward");

        Console.WriteLine("TaxLedger Test 5 passed: PortfolioState routes P&L through the ledger");
    }

    // Test 6: §1222 holding period on the CALENDAR — "more than one year", counted from the
    // day after acquisition: a sale on the anniversary is short-term, the next day long-term.
    // A Feb-29 purchase's anniversary is Feb 28 (Rev. Rul. 66-7 month counting), so LT from Mar 1.
    public void Test_IsLongTerm_CalendarEdges()
    {
        var buy = new DateOnly(2024, 1, 15);
        Debug.Assert(!TaxLedger.IsLongTerm(buy, new DateOnly(2025, 1, 15)), "anniversary is still short-term");
        Debug.Assert( TaxLedger.IsLongTerm(buy, new DateOnly(2025, 1, 16)), "day after anniversary is long-term");
        var leap = new DateOnly(2024, 2, 29);
        Debug.Assert(!TaxLedger.IsLongTerm(leap, new DateOnly(2025, 2, 28)), "Feb-29 lot: Feb 28 is the anniversary");
        Debug.Assert( TaxLedger.IsLongTerm(leap, new DateOnly(2025, 3, 1)),  "Feb-29 lot: long-term from Mar 1");
        // the v0.2 bug: 365 TRADING days ≈ 1.45 calendar years — a lot ~13 months old is long-term now
        Debug.Assert( TaxLedger.IsLongTerm(buy, buy.AddDays(400)), "400 calendar days is long-term");
        Console.WriteLine("TaxLedger Test 6 passed: §1222 calendar edges (anniversary, leap day)");
    }
}
