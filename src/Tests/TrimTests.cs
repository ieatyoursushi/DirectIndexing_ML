using System.Diagnostics;
using DirectIndexing.Core.Simulation;

/// <summary>
/// v0.3-4 sell-winner trim — SymbolTable row `trim`. The trim must only ever sell gain
/// lots, stay §1091-clean, and actually feed realized gains into the ledger (so that
/// carryforward is consumed: lower final carryforward than the identical no-trim world).
/// </summary>
public class TrimTests
{
    public void Test_Trim_SellsOnlyGains_ConsumesCarryforward()
    {
        var world   = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(60, 0.35f), 1260, seed: 7);
        var contrib = ContributionPolicy.Off with { Enabled = true, IntervalDays = 21, NamesPerContribution = 5 };
        var trim    = TrimPolicy.Off with { Enabled = true };

        var off = new SimulationEngine(world, contributionPolicy: contrib);
        var on  = new SimulationEngine(world, contributionPolicy: contrib, trimPolicy: trim);
        var rowsOff = off.Run(10_000_000m);
        var rowsOn  = on.Run(10_000_000m);

        var gainSales = on.Trades.Where(x => x.Kind == TradeKind.Sell && x.RealizedGain >= 0m).ToList();
        int trimDayMisses = gainSales.Count(x => (x.Day - PriceLoader.WarmupDays) % trim.IntervalDays != 0);
        decimal carryOff = (decimal)(rowsOff[^1].CarryST + rowsOff[^1].CarryLT);
        decimal carryOn  = (decimal)(rowsOn[^1].CarryST + rowsOn[^1].CarryLT);
        int violations = WashSaleAudit.Violations(on.Trades).Count;

        Debug.Assert(off.Trades.All(x => x.Kind == TradeKind.Buy || x.RealizedGain < 0m),
            "without the trim the book never sells a winner");
        Debug.Assert(gainSales.Count > 20, $"trim must realize gains (got {gainSales.Count} gain sales)");
        Debug.Assert(trimDayMisses == 0, $"{trimDayMisses} gain sales off the trim schedule");
        Debug.Assert(violations == 0, $"trim arm: {violations} §1091 violations");
        Debug.Assert(carryOn < carryOff, $"realized gains must consume carryforward ({carryOn} vs {carryOff})");
        Console.WriteLine($"Trim Test 1 passed: {gainSales.Count} gain sales, 0 violations, " +
                          $"final carryforward {carryOff:N0} → {carryOn:N0}");
    }
}
