using System.Diagnostics;
using DirectIndexing.Core.Policy;
using DirectIndexing.Core.Simulation;

/// <summary>
/// v0.3-11 — the policy seam and RunMetrics (DataMemo/decisions/PolicyLayer_v04.md §4, §8).
/// SymbolTable rows `harvest_policy`, `tax_position`, `run_metrics`.
/// </summary>
public class PolicyTests
{
    private static PriceLoader World() => PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(40, 0.35f), 1000, seed: 17);
    private static readonly ContributionPolicy Contrib = ContributionPolicy.Off with { Enabled = true, IntervalDays = 21, NamesPerContribution = 5 };

    // The explicit oracle policy reproduces the default engine exactly; NeverHarvest never sells.
    public void Test_OraclePolicy_IsDefault_NeverHarvest_NeverSells()
    {
        var w = World();
        var a = new SimulationEngine(w, contributionPolicy: Contrib);
        var b = new SimulationEngine(w, contributionPolicy: Contrib, policy: new OraclePolicy());
        var never = new SimulationEngine(w, contributionPolicy: Contrib, policy: new NeverHarvestPolicy());
        var ra = a.Run(10_000_000m); var rb = b.Run(10_000_000m); never.Run(10_000_000m);
        bool same = ra.Count == rb.Count && ra.Zip(rb).All(p => p.First == p.Second);
        var m = never.Metrics();
        Debug.Assert(same, "explicit OraclePolicy must be byte-identical to the default engine");
        Debug.Assert(m.LossSales == 0 && m.GainSales == 0 && m.TaxPosition == 0m, $"never: {m.LossSales} sales, W={m.TaxPosition}");
        Console.WriteLine($"Policy Test 1 passed: oracle policy = default ({ra.Count} rows); never-harvest sells nothing");
    }

    // Accounting identities: W_T = ΣΔW_trades + ΣΔW_roll; ΣΔW_trades = used + banked − gain tax;
    // used + banked = Σ TaxValue of the harvested rows (the per-harvest g_tax); 0 wash violations.
    public void Test_RunMetrics_Identities()
    {
        var w = World();
        var e = new SimulationEngine(w, contributionPolicy: Contrib, trimPolicy: TrimPolicy.Off with { Enabled = true });
        var rows = e.Run(10_000_000m);
        var m = e.Metrics();
        decimal sumTaxValue = rows.Where(r => r.Y_Oracle == 1).Sum(r => (decimal)r.TaxValue);
        Debug.Assert(Math.Abs(m.TaxPosition - (m.SumTradeDeltaW + m.RollTrueUp)) < 0.01m,
            $"W_T {m.TaxPosition} ≠ trades {m.SumTradeDeltaW} + roll {m.RollTrueUp}");
        Debug.Assert(Math.Abs(m.SumTradeDeltaW - (m.BenefitUsedNow + m.BenefitBanked - m.GainTaxCost)) < 0.01m,
            "ΣΔW_trades ≠ used + banked − gain tax");
        Debug.Assert(Math.Abs(sumTaxValue - (m.BenefitUsedNow + m.BenefitBanked)) < 1e-4m * Math.Max(1m, sumTaxValue),
            $"Σ TaxValue of harvested rows {sumTaxValue:F2} ≠ used + banked {m.BenefitUsedNow + m.BenefitBanked:F2}");
        Debug.Assert(m.GainSales > 0 && m.RollTrueUp > 0m && m.WashViolations == 0,
            $"run must exercise gains ({m.GainSales}) and the roll ({m.RollTrueUp}); violations {m.WashViolations}");
        Console.WriteLine($"Policy Test 2 passed: W_T {m.TaxPosition:N0} = trades {m.SumTradeDeltaW:N0} + roll {m.RollTrueUp:N0}; " +
                          $"used {m.BenefitUsedNow:N0} + banked {m.BenefitBanked:N0} − gain tax {m.GainTaxCost:N0}");
    }

    // Conservation: on a flat world with one −10% step close to the end, the oracle harvests at the
    // bottom and its reopen falls PAST the last day. Prices never move again, so pre-tax wealth
    // (holdings + pending reopen cash + cash) must equal never-harvest's to the cent — the check that
    // caught proceeds being dropped when the reopen date ran off the calendar.
    public void Test_Wealth_IsConserved_WhenReopenRunsPastTheEnd()
    {
        const int T = 400, drop = 385;
        var d = new Dictionary<string, float[]>();
        for (int i = 0; i < 10; i++)
        {
            var r = new float[T]; r[0] = float.NaN;
            r[drop] = -0.10f;
            d[$"Z{i}"] = r;
        }
        var w = PriceLoader.CreateForTesting(d);
        var never  = new SimulationEngine(w, policy: new NeverHarvestPolicy());
        var oracle = new SimulationEngine(w, policy: new OraclePolicy());
        never.Run(1_000_000m); oracle.Run(1_000_000m);
        var a = never.Metrics(); var b = oracle.Metrics();
        decimal preA = a.TerminalHoldings + a.PendingReopenCash + a.Cash;
        decimal preB = b.TerminalHoldings + b.PendingReopenCash + b.Cash;
        Debug.Assert(b.LossSales > 0 && b.PendingReopenCash > 0m, $"oracle must harvest with reopens pending ({b.LossSales}, {b.PendingReopenCash})");
        Debug.Assert(Math.Abs(preA - preB) < 0.01m, $"pre-tax wealth not conserved: never {preA:N2} vs oracle {preB:N2}");
        Debug.Assert(b.AfterTaxWealth > a.AfterTaxWealth, "harvesting a pure loss must raise after-tax wealth");
        Console.WriteLine($"Policy Test 3 passed: pre-tax wealth conserved ({preA:N0}); after-tax +{b.AfterTaxWealth - a.AfterTaxWealth:N0} from {b.LossSales} harvests");
    }
}
