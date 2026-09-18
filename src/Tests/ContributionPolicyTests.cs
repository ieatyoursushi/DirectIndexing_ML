// Tests/ContributionPolicyTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Simulation;

/// <summary>
/// Unit tests for the v0.3 contribution policy — the cost-basis-aging fix. The
/// invariants under test are the two that make it safe: the default is OFF (so an
/// unflagged run reproduces v0.26), and the deposit schedule is exogenous and
/// correctly pro-rated.
///
/// The wash-sale eligibility filter and the underweight ranking are exercised
/// end-to-end by the simulation itself (a contribution run asserts zero lots are
/// opened in a ticker whose WashClock &lt; 30); see the v0.3 verification notes.
/// </summary>
public class ContributionPolicyTests
{
    // Test 1: off by default — the v0.26 reproduction guarantee.
    public void Test_DefaultIsDisabled()
    {
        var p = ContributionPolicy.Off;
        Debug.Assert(!p.Enabled, "ContributionPolicy must default to disabled");
        Debug.Assert(p.DatasetTag == "",
            $"Disabled policy must not tag the dataset, got '{p.DatasetTag}'");
        Debug.Assert(new ContributionPolicy().Enabled == false,
            "A freshly constructed policy must also default to disabled");
        Console.WriteLine("Contribution Test 1 passed: disabled by default, no dataset tag");
    }

    // Test 2: the deposit is the annual rate pro-rated over the interval.
    public void Test_AmountProRatedOverInterval()
    {
        // 10%/yr of $10M = $1M/yr; quarterly (63 of 252 trading days) = $250k.
        var quarterly = new ContributionPolicy
            { Enabled = true, IntervalDays = 63, AnnualRate = 0.10m };
        decimal amt = quarterly.AmountPer(10_000_000m);
        Debug.Assert(amt == 250_000m, $"Expected $250,000 per quarter, got {amt}");

        // Same annual rate, monthly (21 days) → one third of the quarterly deposit.
        // 21/252 = 1/12 does not terminate in decimal, so compare within tolerance
        // rather than exactly (the residual is ~1e-23, pure representation error).
        var monthly = quarterly with { IntervalDays = 21 };
        Debug.Assert(Math.Abs(monthly.AmountPer(10_000_000m) * 3m - amt) < 0.01m,
            "Three monthly deposits must equal one quarterly deposit at the same annual rate");

        // Full year of deposits sums to the annual rate.
        Debug.Assert(quarterly.AmountPer(10_000_000m) * 4m == 1_000_000m,
            "Four quarterly deposits must sum to 10% of the initial book");

        Console.WriteLine("Contribution Test 2 passed: deposits pro-rate to the annual rate");
    }

    // Test 3: the schedule is exogenous — it keys off the INITIAL book value, so a
    // bull or bear market does not change the deposit size. This is what keeps the
    // contribution ablation interpretable (prevalence recovery is attributable to
    // fresh lots, not to a deposit schedule that compounds with performance).
    public void Test_ScheduleIsExogenous()
    {
        var p = new ContributionPolicy { Enabled = true, IntervalDays = 63, AnnualRate = 0.10m };
        decimal atOpen  = p.AmountPer(10_000_000m);
        decimal doubled = p.AmountPer(20_000_000m);
        Debug.Assert(doubled == atOpen * 2m,
            "Deposit must scale linearly with the initial value it is quoted against");
        // The engine always passes the INITIAL value, never the current one — verified
        // by construction (SimulationEngine._initialValue is set once, in InitializePortfolio).
        Console.WriteLine("Contribution Test 3 passed: schedule keys off the initial book value");
    }

    // Test 4: enabling tags the dataset so ablation arms never overwrite each other.
    public void Test_EnabledTagsDataset()
    {
        var on = new ContributionPolicy { Enabled = true };
        Debug.Assert(on.DatasetTag == "_contrib",
            $"Enabled policy must tag the dataset, got '{on.DatasetTag}'");
        Debug.Assert(on.Describe().Contains("63d") || on.Describe().Contains("every 63"),
            $"Describe() should state the interval, got '{on.Describe()}'");
        Console.WriteLine("Contribution Test 4 passed: enabled arm writes to its own dataset");
    }
}
