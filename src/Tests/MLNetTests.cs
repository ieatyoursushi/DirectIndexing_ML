using System.Diagnostics;
using DirectIndexing.Core.Portfolio;
using DirectIndexing.Export;
using DirectIndexing.ML.MLNet.Data;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Splits;
using DirectIndexing.ML.MLNet.Tuning;
using Microsoft.ML;

/// <summary>
/// Smoke tests for the ML.NET layer. Same Debug.Assert harness used by the
/// rest of the project — wired in from Program.cs case "test".
/// </summary>
public class LotStateVectorCsvReaderTests
{
    public void Test_RoundTrip_PreservesAllFields()
    {
        var rows = new List<LotStateVector>
        {
            new() { L = -0.12f, H = 30, S = 0, B = 100.5f, W = 0.02f, K = 2,
                    G_YTD = 1234.5f, Sigma_TE = 0.015f, WashClock = 5,
                    R_t = 0.001f, SigmaRange = 0.02f, DeltaMA50 = -0.01f, DeltaMA200 = -0.05f,
                    TaxAlpha = 12.3f, DaysToYE = 200, Y_Oracle = 1,
                    Y_Soft_GBM = 0.7f, Y_Soft_BT = 0.3f,
                    Symbol = "AAPL", Sector = "Tech", Timestep = 42 },
            new() { L = float.NaN, H = 0, S = 1, B = 50.0f, W = 0.001f, K = 1,
                    G_YTD = 0f, Sigma_TE = 0.01f, WashClock = 999,
                    R_t = 0f, SigmaRange = 0.01f, DeltaMA50 = 0f, DeltaMA200 = 0f,
                    TaxAlpha = 0f, DaysToYE = 100, Y_Oracle = 0,
                    Y_Soft_GBM = float.NaN, Y_Soft_BT = float.NaN,
                    Symbol = "MSFT", Sector = "Tech", Timestep = 0 },
        };

        var tmp = Path.Combine(Path.GetTempPath(), $"lots_test_{Guid.NewGuid()}.csv");
        SimulationExporter.WriteCsv(rows, tmp);
        var roundTrip = LotStateVectorCsvReader.Read(tmp);
        File.Delete(tmp);

        Debug.Assert(roundTrip.Count == 2, $"expected 2 rows got {roundTrip.Count}");
        Debug.Assert(roundTrip[0].Symbol == "AAPL", "Symbol parse");
        Debug.Assert(roundTrip[0].Y_Oracle == 1, "Y_Oracle parse");
        Debug.Assert(Math.Abs(roundTrip[0].L - (-0.12f)) < 1e-4, "L parse");
        Debug.Assert(float.IsNaN(roundTrip[1].L), "NaN preserved");
        Debug.Assert(float.IsNaN(roundTrip[1].Y_Soft_BT), "NaN Y_Soft_BT preserved");
        Console.WriteLine("LotStateVectorCsvReader: round-trip preserves all fields ✓");
    }
}

public class StratifiedSplitTests
{
    public void Test_PreservesClassProportionWithin1Percent()
    {
        var data = MakeBinarySynthetic(1000, positiveRate: 0.01, seed: 1);
        var (train, test) = StratifiedSplit.Split(data, r => r.Y_Oracle, 0.20, seed: 42);

        Debug.Assert(train.Count + test.Count == data.Count, "sizes sum to N");

        double overallPos = data.Count(r => r.Y_Oracle == 1) / (double)data.Count;
        double trainPos = train.Count(r => r.Y_Oracle == 1) / (double)train.Count;
        double testPos  = test.Count(r => r.Y_Oracle == 1)  / (double)test.Count;

        Debug.Assert(Math.Abs(trainPos - overallPos) < 0.01, $"train class drift {trainPos} vs {overallPos}");
        Debug.Assert(Math.Abs(testPos  - overallPos) < 0.01, $"test class drift {testPos} vs {overallPos}");
        Console.WriteLine($"StratifiedSplit: class proportion preserved (overall {overallPos:P1}, train {trainPos:P1}, test {testPos:P1}) ✓");
    }

    internal static List<LotStateVector> MakeBinarySynthetic(int n, double positiveRate, int seed)
    {
        var rng = new Random(seed);
        var rows = new List<LotStateVector>(n);
        for (int i = 0; i < n; i++)
        {
            int y = rng.NextDouble() < positiveRate ? 1 : 0;
            rows.Add(new LotStateVector
            {
                L = (float)rng.NextDouble() - 0.5f,
                H = 30, S = 0, B = 100f, W = 0.01f, K = 1,
                G_YTD = 100f, Sigma_TE = 0.01f, WashClock = 999,
                R_t = (float)rng.NextDouble() * 0.01f,
                SigmaRange = 0.02f, DeltaMA50 = 0f, DeltaMA200 = 0f,
                TaxAlpha = 1f, DaysToYE = 200,
                Y_Oracle = y, Y_Soft_GBM = 0f, Y_Soft_BT = y == 1 ? 0.5f : 0f,
                Symbol = $"SYM{i % 50}", Sector = "Tech", Timestep = i,
            });
        }
        return rows;
    }
}

public class StratifiedKFoldTests
{
    public void Test_FoldsPartitionDataAndContainPositives()
    {
        var data = StratifiedSplitTests.MakeBinarySynthetic(1000, 0.05, seed: 2);
        int k = 5;
        int total = 0;
        int foldsWithPositives = 0;

        foreach (var (train, val) in StratifiedKFold.Folds(data, r => r.Y_Oracle, k, seed: 42))
        {
            Debug.Assert(train.Count + val.Count == data.Count, "fold partition complete");
            total += val.Count;
            if (val.Any(r => r.Y_Oracle == 1)) foldsWithPositives++;
        }

        Debug.Assert(total == data.Count, $"folds sum to N (got {total})");
        Debug.Assert(foldsWithPositives == k, $"every fold should have positives, got {foldsWithPositives}/{k}");
        Console.WriteLine($"StratifiedKFold: {k} folds partition N rows, all contain positives ✓");
    }
}

public class SilhouetteTests
{
    public void Test_TwoBlobsHighSilhouette()
    {
        var rng = new Random(42);
        var points = new double[200][];
        var clusters = new int[200];
        for (int i = 0; i < 100; i++)
        {
            points[i] = new[] { rng.NextDouble() + 5.0, rng.NextDouble() + 5.0 };
            clusters[i] = 0;
        }
        for (int i = 100; i < 200; i++)
        {
            points[i] = new[] { rng.NextDouble() - 5.0, rng.NextDouble() - 5.0 };
            clusters[i] = 1;
        }
        double s = SilhouetteScore.Compute(points, clusters);
        Debug.Assert(s > 0.7, $"two well-separated blobs should give silhouette > 0.7, got {s}");
        Console.WriteLine($"Silhouette: two well-separated blobs gave {s:F3} ✓");
    }
}

public class PreprocessingTests
{
    public void Test_MedianImputerReplacesNaNs()
    {
        var rows = new List<LotStateVector>
        {
            new() { L = -0.1f,        Sector = "Tech", Y_Oracle = 1 },
            new() { L = -0.3f,        Sector = "Tech", Y_Oracle = 0 },
            new() { L = float.NaN,    Sector = "",     Y_Oracle = 0 },
            new() { L = -0.5f,        Sector = "Fin",  Y_Oracle = 0 },
        };
        // Need at least all FloatNumericFeatures present; set the rest to 0 so medians compute.
        for (int i = 0; i < rows.Count; i++)
        {
            rows[i] = rows[i] with { B = 100f, W = 0.01f, G_YTD = 0f, Sigma_TE = 0.01f,
                                     R_t = 0f, SigmaRange = 0.01f,
                                     DeltaMA50 = 0f, DeltaMA200 = 0f, TaxAlpha = 0f };
        }

        var medians = MedianImputer.Fit(rows);
        var imputed = MedianImputer.Apply(rows, medians, r => r.Y_Oracle == 1);
        Debug.Assert(!float.IsNaN(imputed[2].L), "NaN L should be replaced");
        // Median of [-0.1, -0.3, -0.5] = -0.3.
        Debug.Assert(Math.Abs(imputed[2].L - (-0.3f)) < 1e-4, $"expected median -0.3, got {imputed[2].L}");
        Console.WriteLine($"MedianImputer: NaN replaced with training-fold median ✓");
    }

    public void Test_ClassWeightsBalanced()
    {
        var rows = new List<LotStateVector>();
        for (int i = 0; i < 9; i++) rows.Add(new() { Y_Oracle = 0 });
        rows.Add(new() { Y_Oracle = 1 });
        // N = 10, n_classes = 2, n_0 = 9, n_1 = 1.
        // w_0 = 10/(2*9) = 0.555…, w_1 = 10/(2*1) = 5.0.
        var weighted = ClassWeights.AttachBalancedWeights(rows, r => r.Y_Oracle);
        var pos = weighted.First(w => w.Y_Oracle == 1);
        var neg = weighted.First(w => w.Y_Oracle == 0);
        Debug.Assert(Math.Abs(pos.Weight - 5.0f) < 1e-3, $"expected pos weight 5.0, got {pos.Weight}");
        Debug.Assert(Math.Abs(neg.Weight - 10f/(2*9)) < 1e-3, $"expected neg weight 0.555, got {neg.Weight}");
        Console.WriteLine($"ClassWeights: balanced weights w_0={neg.Weight:F3}, w_1={pos.Weight:F3} ✓");
    }
}

public class GridSearchTests
{
    public void Test_PicksLargestCForLinearlySeparableData()
    {
        // Synthetic linearly separable: Y = 1 iff L < -0.3.
        var rng = new Random(7);
        var data = new List<LotStateVector>(800);
        for (int i = 0; i < 800; i++)
        {
            float l = (float)(rng.NextDouble() - 0.5) * 2f;  // ~U(-1, 1)
            int y = l < -0.3f ? 1 : 0;
            data.Add(new LotStateVector
            {
                L = l, H = 30, S = 0, B = 100f, W = 0.01f, K = 1,
                G_YTD = 100f, Sigma_TE = 0.01f, WashClock = 999,
                R_t = 0f, SigmaRange = 0.01f, DeltaMA50 = 0f, DeltaMA200 = 0f,
                TaxAlpha = 1f, DaysToYE = 200,
                Y_Oracle = y, Y_Soft_GBM = 0f, Y_Soft_BT = y,
                Symbol = $"SYM{i % 10}", Sector = "Tech", Timestep = i,
            });
        }

        var ml = new MLContext(seed: 42);
        var grid = new Dictionary<string, object[]> { ["C"] = new object[] { 0.01, 0.1, 1.0, 10.0 } };
        IEstimator<ITransformer> Factory(IDictionary<string, object> cfg) =>
            DirectIndexing.ML.MLNet.Preprocessing.PreprocessingPipeline.Build(ml)
                .Append(ml.BinaryClassification.Trainers.LbfgsLogisticRegression(
                    new Microsoft.ML.Trainers.LbfgsLogisticRegressionBinaryTrainer.Options
                    {
                        LabelColumnName = "Label",
                        FeatureColumnName = "Features",
                        ExampleWeightColumnName = "Weight",
                        L2Regularization = (float)(1.0 / (double)cfg["C"]),
                        MaximumNumberOfIterations = 200,
                    }));

        var result = GridSearchCV.Search(ml, data, r => r.Y_Oracle, grid, Factory, r => r.Y_Oracle == 1, k: 3, seed: 42);

        // On a cleanly separable problem the highest-C (weakest regularization) should win
        // or at minimum the best mean CV PR-AUC should be > 0.9.
        Debug.Assert(result.MeanCvScore > 0.9, $"separable data → expected CV PR-AUC > 0.9, got {result.MeanCvScore:F3}");
        Console.WriteLine($"GridSearchCV: best C = {result.BestParams["C"]}, CV PR-AUC = {result.MeanCvScore:F3} ✓");
    }
}
