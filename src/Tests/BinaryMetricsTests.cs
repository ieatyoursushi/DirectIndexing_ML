using System.Diagnostics;
using DirectIndexing.ML.MLNet.Metrics;

/// <summary>
/// Exact-value pins for the evaluation functionals (SymbolTable §F): PR-AUC is the
/// project's CV selection criterion, so its arithmetic — and the prevalence
/// asymmetry the v0.26 diagnosis rests on (MLDerivations §5.2) — is tested directly.
/// </summary>
public class BinaryMetricsTests
{
    private static ScoredRow R(bool y, float p) => new() { Label = y, Probability = p };

    // Hand-computed: ranking + − + −.
    //   AP  = Σ ΔRecall·Precision = 0.5·1 + 0.5·(2/3) = 5/6
    //   ROC = Mann–Whitney pair fraction = 3 of 4 (pos, neg) pairs ordered correctly = 3/4
    public void Test_AveragePrecision_And_Roc_HandComputed()
    {
        var m = BinaryMetrics.Compute(new[] { R(true, .9f), R(false, .8f), R(true, .7f), R(false, .6f) });
        Debug.Assert(Math.Abs(m.PrAuc - 5.0 / 6.0) < 1e-12, $"AP should be 5/6, got {m.PrAuc}");
        Debug.Assert(Math.Abs(m.RocAuc - 0.75) < 1e-12, $"ROC-AUC should be 3/4, got {m.RocAuc}");
        Console.WriteLine($"BinaryMetrics Test 1 passed: AP = {m.PrAuc:F4} (5/6), ROC = {m.RocAuc:F2} (3/4)");
    }

    // A perfect ranker scores 1 on BOTH functionals at any prevalence.
    public void Test_PerfectRanker_IsOne_AtAnyPrevalence()
    {
        var rows = Enumerable.Range(0, 100).Select(i => R(i < 5, 1f - i / 100f)).ToArray();  // p = 5%
        var m = BinaryMetrics.Compute(rows);
        Debug.Assert(Math.Abs(m.PrAuc - 1) < 1e-12 && Math.Abs(m.RocAuc - 1) < 1e-12,
            $"perfect ranker must give AP = ROC = 1, got {m.PrAuc}/{m.RocAuc}");
        Console.WriteLine("BinaryMetrics Test 2 passed: perfect ranker → AP = ROC = 1 at p = 5%");
    }

    // A no-skill (all-tied) scorer: ROC-AUC = 1/2 regardless of prevalence, but
    // PR-AUC = the prevalence p — the "0.5 = random" rule is ROC-only (MLDerivations §5.2).
    public void Test_NoSkill_Floors_RocHalf_PrEqualsPrevalence()
    {
        var rows = Enumerable.Range(0, 200).Select(i => R(i < 6, 0.3f)).ToArray();  // p = 3%
        var m = BinaryMetrics.Compute(rows);
        Debug.Assert(Math.Abs(m.RocAuc - 0.5) < 1e-12, $"no-skill ROC-AUC must be 1/2, got {m.RocAuc}");
        Debug.Assert(Math.Abs(m.PrAuc - 0.03) < 1e-12, $"no-skill PR-AUC must equal prevalence 0.03, got {m.PrAuc}");
        Console.WriteLine("BinaryMetrics Test 3 passed: no-skill → ROC = 0.5, PR-AUC = prevalence (0.03)");
    }
}
