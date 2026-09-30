using System.Diagnostics;
using DirectIndexing.Core.Portfolio;
using DirectIndexing.Core.Simulation;

/// <summary>
/// §1091 — the independent audit (a restatement of the law over a trade log) and,
/// through it, the engine's wash-sale gating. SymbolTable rows `wash_audit`, `wash_clock`.
/// </summary>
public class WashSaleTests
{
    private static readonly DateOnly D0 = new(2024, 3, 1);

    private static TradeEvent Buy(Lot lot, int dayOffset) =>
        new(D0.AddDays(dayOffset), dayOffset, lot.Symbol, TradeKind.Buy, lot, lot.CostBasis, 0m);

    private static TradeEvent Sell(Lot lot, int dayOffset, decimal gain) =>
        new(D0.AddDays(dayOffset), dayOffset, lot.Symbol, TradeKind.Sell, lot, lot.CostBasis, gain);

    // The audit's definition, pinned at its edges: ±30 calendar days inclusive, a
    // DIFFERENT lot of the same ticker, loss sales only.
    public void Test_Audit_WindowEdges_SameLot_AndGains()
    {
        var old = new Lot("AAA", "X", 100m, 10, 0);
        var fresh = new Lot("AAA", "X", 90m, 10, 0);

        // replacement bought 30 days BEFORE a loss sale → violation (inside the window)
        int v1 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(fresh, 0), Sell(old, 30, -50m) }).Count;
        // …31 days before → clean
        int v2 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(fresh, 0), Sell(old, 31, -50m) }).Count;
        // replacement bought 30 days AFTER → violation; 31 days after → clean
        int v3 = WashSaleAudit.Violations(new[] { Buy(old, -400), Sell(old, 0, -50m), Buy(fresh, 30) }).Count;
        int v4 = WashSaleAudit.Violations(new[] { Buy(old, -400), Sell(old, 0, -50m), Buy(fresh, 31) }).Count;
        // selling the SAME lot you just bought, at a loss, is not a wash sale
        int v5 = WashSaleAudit.Violations(new[] { Buy(fresh, 0), Sell(fresh, 10, -50m) }).Count;
        // a GAIN sale next to a buy is not a wash sale
        int v6 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(fresh, 0), Sell(old, 5, +50m) }).Count;
        // another ticker's buy is irrelevant
        var other = new Lot("BBB", "X", 50m, 10, 0);
        int v7 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(other, 0), Sell(old, 5, -50m) }).Count;

        Debug.Assert(v1 == 1 && v2 == 0, $"before-side edge: expected 1/0, got {v1}/{v2}");
        Debug.Assert(v3 == 1 && v4 == 0, $"after-side edge: expected 1/0, got {v3}/{v4}");
        Debug.Assert(v5 == 0 && v6 == 0 && v7 == 0, $"exclusions: expected 0/0/0, got {v5}/{v6}/{v7}");
        Console.WriteLine("WashSale Test 1 passed: audit window is ±30 d_cal inclusive; same lot, gains, other tickers excluded");
    }
}
