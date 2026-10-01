using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Policy;

/// <summary>
/// The policy seam (v0.3-11): WHO decides to harvest a gate-evaluated lot. The engine still
/// computes the oracle label Y_Oracle on every row (labels describe f*); the policy decides the
/// ACTION. Rungs 1–3 of the economic ladder are per-lot rules; v0.4's day-level executor
/// (DataMemo/decisions/PolicyLayer_v04.md §3) plugs in at the same call site.
/// </summary>
// [math:harvest_policy] — DataMemo/spec/SymbolTable.md
public interface IHarvestPolicy
{
    string Name { get; }
    bool ShouldHarvest(LotStateVector snapshot);
}

/// <summary>Rung 1 — never harvest (the buy-and-hold index).</summary>
public sealed class NeverHarvestPolicy : IHarvestPolicy
{
    public string Name => "never";
    public bool ShouldHarvest(LotStateVector s) => false;
}

/// <summary>
/// Rung 2 — the naive loss threshold: harvest any lot ≥ θ underwater that §1091 allows. No TE
/// term, no ledger value, no cost (the "harvest every dip" retail heuristic).
/// </summary>
public sealed class ThresholdHarvestPolicy : IHarvestPolicy
{
    public decimal Threshold { get; }
    public ThresholdHarvestPolicy(decimal threshold = 0.02m) => Threshold = threshold;
    public string Name => "threshold";
    public bool ShouldHarvest(LotStateVector s) =>
        (decimal)s.L <= -Threshold && s.WashClock > PortfolioState.WashWindowDays;
}

/// <summary>Rung 3 — the oracle f* (default; byte-identical to the pre-seam engine).</summary>
public sealed class OraclePolicy : IHarvestPolicy
{
    public string Name => "oracle";
    public bool ShouldHarvest(LotStateVector s) => s.Y_Oracle == 1;
}

public static class HarvestPolicies
{
    public static IHarvestPolicy Parse(string s) => s switch
    {
        "never"     => new NeverHarvestPolicy(),
        "threshold" => new ThresholdHarvestPolicy(),
        "oracle"    => new OraclePolicy(),
        _ => throw new ArgumentException($"--policy must be never|threshold|oracle, got '{s}'"),
    };
}
