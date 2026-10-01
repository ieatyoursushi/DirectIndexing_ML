using System.Diagnostics;
using DirectIndexing.Core.Simulation;

/// <summary>
/// The synthetic GBM price source (<see cref="PriceLoader.FromGbm"/>) that replaced the
/// duplicated MonteCarloEngine: its world must be a well-formed trading calendar, a
/// deterministic function of the seed, carry the σ it was built with, and run through the
/// one canonical SimulationEngine.
/// </summary>
public class SyntheticWorldTests
{
    public void Test_Calendar_IsWeekdaysAndCrossesYearEnd()
    {
        var w = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(3), days: 504, seed: 1);
        var cal = w.Calendar;

        Debug.Assert(cal.Count == 504, $"expected 504 trading days, got {cal.Count}");
        Debug.Assert(cal.All(d => d.DayOfWeek is not (DayOfWeek.Saturday or DayOfWeek.Sunday)),
            "calendar must contain weekdays only");
        Debug.Assert(cal.Zip(cal.Skip(1)).All(p => p.First < p.Second), "calendar must be strictly increasing");
        Debug.Assert(cal[0].Year != cal[^1].Year, "a 504-day world must cross a year-end");
        Console.WriteLine($"SyntheticWorld Test 1 passed: {cal.Count} weekdays, {cal[0]} → {cal[^1]}");
    }

    public void Test_Deterministic_ForSeed()
    {
        var u = PriceLoader.UniformGbmUniverse(5);
        var a = PriceLoader.FromGbm(u, 300, seed: 7).GetCloseArray("SYN003");
        var b = PriceLoader.FromGbm(u, 300, seed: 7).GetCloseArray("SYN003");
        var c = PriceLoader.FromGbm(u, 300, seed: 8).GetCloseArray("SYN003");

        Debug.Assert(a.SequenceEqual(b), "same seed must give the same world");
        Debug.Assert(!a.SequenceEqual(c), "a different seed must give a different world");
        Debug.Assert(a.All(p => p > 0f && float.IsFinite(p)), "GBM prices must be positive and finite");
        Console.WriteLine("SyntheticWorld Test 2 passed: deterministic in the seed, prices positive");
    }

    public void Test_RealisedVol_MatchesSigma()
    {
        const float Sigma = 0.30f;
        const int Names = 30, Days = 1000;
        var w = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(Names, Sigma), Days, seed: 11);

        // Annualised sample std of simple returns, averaged over names. For one name the
        // standard error of σ̂ is ≈ σ/√(2(T−1)) ≈ 0.0067; averaged over 30 it is ≈ 0.0012,
        // so a 5% band (0.015) is a >10-SE test.
        double meanSigmaHat = w.Symbols.Average(sym =>
        {
            var r = w.GetReturnArray(sym).Skip(1).Select(x => (double)x).ToArray();
            double m = r.Average();
            return Math.Sqrt(r.Sum(x => (x - m) * (x - m)) / (r.Length - 1) * 252.0);
        });

        Debug.Assert(Math.Abs(meanSigmaHat - Sigma) < 0.05 * Sigma,
            $"realised σ {meanSigmaHat:F4} should be within 5% of {Sigma}");
        Console.WriteLine($"SyntheticWorld Test 3 passed: realised σ = {meanSigmaHat:F4} vs {Sigma}");
    }

    public void Test_CanonicalEngine_RunsOnSyntheticWorld()
    {
        const int Days = 504;
        var w = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(30, 0.35f), Days, seed: 42);
        var snaps = new SimulationEngine(w).Run(1_000_000m);
        new SoftLabelBuilder(w).Label(snaps);

        Debug.Assert(snaps.Count > 0, "engine produced no rows");
        Debug.Assert(snaps.All(s => s.Timestep >= PriceLoader.WarmupDays), "rows before warmup");
        // Y_Soft_BT is defined on synthetic paths wherever a full 30-day forward window exists.
        var early = snaps.Where(s => s.Timestep + 30 < Days).ToList();
        var tail  = snaps.Where(s => s.Timestep + 30 >= Days).ToList();
        Debug.Assert(early.All(s => !float.IsNaN(s.Y_Soft_BT)), "Y_Soft_BT must be defined before the tail");
        Debug.Assert(tail.All(s => float.IsNaN(s.Y_Soft_BT)), "Y_Soft_BT must be NaN in the last 30 days");
        // DaysToYE is calendar-based now (the old MC engine approximated it on a 252-step cycle).
        Debug.Assert(snaps.All(s => s.DaysToYE >= 0 && s.DaysToYE <= 365), "DaysToYE out of calendar range");
        Console.WriteLine($"SyntheticWorld Test 4 passed: {snaps.Count} rows, " +
                          $"{snaps.Count(s => s.Y_Oracle == 1)} harvests, Y_Soft_BT defined on {early.Count} rows");
    }
}
