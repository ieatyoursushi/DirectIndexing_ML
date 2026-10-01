using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Io;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Models;
using DirectIndexing.ML.MLNet.Tuning;
using Microsoft.ML;

namespace DirectIndexing.ML.MLNet;

/// <summary>
/// Top-level orchestration for the ML.NET pipeline. Each method consumes a
/// typed <c>List&lt;LotStateVector&gt;</c> and writes JSON / CSV / .zip
/// artifacts under <c>data/artifacts-mlnet/</c>. Rendering (PNGs, HTML) is
/// delegated to the Python renderer via <see cref="PythonRunner"/>.
/// </summary>
public static class MLnetPipeline
{
    // ── Shared artifact payload types ────────────────────────────────────────

    private record Confusion(int Tp, int Fp, int Tn, int Fn);
    private record CurvePointDto(double Threshold, double X, double Y);

    private record BaseMetrics(
        string Target,
        int    RowsTrain,
        int    RowsTest,
        double CvBestMeanPrAuc,
        double[] CvPerFold,
        double TestRocAuc,
        double TestPrAuc,
        double F1At05,
        double F1AtBest,
        double BestThreshold,
        Confusion ConfusionAt05,
        Confusion ConfusionAtBest,
        CurvePointDto[] RocCurve,
        CurvePointDto[] PrCurve,
        IReadOnlyList<StratumMetrics> StrataBySigmaMkt);

    private static BaseMetrics ToBase(
        BinaryMetricsResult m, string target, int train, int test, double[] cvFolds) =>
        new(target, train, test,
            cvFolds.Average(), cvFolds,
            m.RocAuc, m.PrAuc, m.F1At05, m.F1AtBest, m.BestThreshold,
            new(m.Tp05, m.Fp05, m.Tn05, m.Fn05),
            new(m.TpBest, m.FpBest, m.TnBest, m.FnBest),
            m.RocCurve.Select(p => new CurvePointDto(p.Threshold, p.X, p.Y)).ToArray(),
            m.PrCurve .Select(p => new CurvePointDto(p.Threshold, p.X, p.Y)).ToArray(),
            m.Strata);

    // ── Per-model public entry points (full CV + test eval) ──────────────────

    /// <summary>
    /// Run a specific model's full pipeline (CV grid search + test eval).
    /// Used by the individual <c>mlnet-gbt</c> / <c>mlnet-logistic</c> cases.
    /// </summary>
    public static void RunSupervisedModel(
        string modelName,
        IReadOnlyList<LotStateVector> data,
        string target,
        string artifactsDir)
    {
        Console.WriteLine($"[MLnetPipeline] {modelName} target={target} rows={data.Count}");
        Directory.CreateDirectory(artifactsDir);
        var ml = new MLContext(seed: 42);

        switch (modelName)
        {
            case "logistic": WriteLogistic(LogisticTrainer.Run(ml, data, target), target, artifactsDir, ml); break;
            case "gbt":      WriteGbt     (GradientBoostedTreesTrainer.Run(ml, data, target), target, artifactsDir, ml); break;
            default: throw new ArgumentException($"unknown model '{modelName}'");
        }
    }

    /// <summary>
    /// The two supervised models kept after the pre-v0.3 downsizing
    /// (<c>DataMemo/archive/RetiredComponents.md</c>): GBT, the champion and RL
    /// substrate, and L2 logistic, the linear control that measures how much
    /// non-linearity the target actually has.
    /// </summary>
    public static readonly IReadOnlyList<string> Models = new[] { "gbt", "logistic" };

    /// <summary>
    /// Champion = argmax of mean CV PR-AUC. A pure function of the CV results, so the
    /// selection rule is testable without touching any test set.
    /// </summary>
    // [math:champion] — DataMemo/spec/SymbolTable.md
    public static string SelectChampion(IEnumerable<CvResult> cvResults) =>
        cvResults.OrderByDescending(r => r.MeanCvScore).First().ModelName;

    /// <summary>
    /// Comparison run: CV-tunes both models, emits a leaderboard that names the
    /// champion, then runs the full test evaluation for both. They are a
    /// pre-registered comparison pair (champion vs linear control), so evaluating
    /// both on test selects nothing on test — the selection rule reads CV only, and
    /// the test set is touched only after the leaderboard is written.
    /// </summary>
    public static void RunAllSupervised(
        IReadOnlyList<LotStateVector> data,
        string target,
        string artifactsDir)
    {
        Console.WriteLine($"[MLnetPipeline] all-supervised target={target} rows={data.Count}");
        Directory.CreateDirectory(artifactsDir);
        var ml = new MLContext(seed: 42);

        // 1. CV phase — test set untouched.
        var cvResults = new[]
        {
            GradientBoostedTreesTrainer.RunCV(ml, data, target),
            LogisticTrainer            .RunCV(ml, data, target),
        };
        var champion = SelectChampion(cvResults);

        // 2. Emit CV leaderboard.
        var leaderboard = cvResults
            .OrderByDescending(r => r.MeanCvScore)
            .Select(r => new
            {
                r.ModelName,
                MeanCvPrAuc  = r.MeanCvScore,
                r.PerFoldScores,
                IsChampion   = r.ModelName == champion,
                Role         = r.ModelName == "gbt" ? "champion_candidate" : "linear_control",
                AllConfigs   = r.AllConfigs.Select(c => new
                {
                    Params   = c.Params,
                    MeanPrAuc = c.MeanScore,
                }),
            });
        Artifacts.WriteJson(leaderboard,
            Path.Combine(artifactsDir, $"{target}_cv_leaderboard.json"));

        // 3. Full test eval — only after the CV leaderboard exists.
        foreach (var name in Models)
            RunSupervisedModel(name, data, target, artifactsDir);
    }

    // ── Artifact writers ─────────────────────────────────────────────────────

    private static void WriteLogistic(
        LogisticTrainer.LogisticOutput r, string target, string dir, MLContext ml)
    {
        var name = $"logistic_{target}";
        var b    = ToBase(r.Metrics, target, r.RowsTrain, r.RowsTest, r.PerFoldCvScores);
        Artifacts.WriteJson(new
        {
            b.Target, b.RowsTrain, b.RowsTest,
            BestC     = r.BestC,
            L2Used    = r.L2Used,
            AllConfigs = r.AllConfigs.Select(c => new { C = c.C, MeanCvPrAuc = c.MeanScore }),
            b.CvBestMeanPrAuc, b.CvPerFold,
            b.TestRocAuc, b.TestPrAuc, b.F1At05, b.F1AtBest, b.BestThreshold,
            b.ConfusionAt05, b.ConfusionAtBest, b.RocCurve, b.PrCurve, b.StrataBySigmaMkt,
        }, Path.Combine(dir, $"{name}_metrics.json"));

        WriteCoefficients(r.Coefficients, dir, name);
        ml.Model.Save(r.Model, null, Path.Combine(dir, $"{name}_model.zip"));
    }

    private static void WriteGbt(
        GradientBoostedTreesTrainer.GbtOutput r, string target, string dir, MLContext ml)
    {
        var name = $"gbt_{target}";
        var b    = ToBase(r.Metrics, target, r.RowsTrain, r.RowsTest, r.PerFoldCvScores);
        Artifacts.WriteJson(new
        {
            b.Target, b.RowsTrain, b.RowsTest,
            BestNumberOfTrees  = r.BestNumberOfTrees,
            BestLearningRate   = r.BestLearningRate,
            BestNumberOfLeaves = r.BestNumberOfLeaves,
            AllConfigs = r.AllConfigs.Select(c => new
            {
                Params     = c.Params,
                MeanCvPrAuc = c.MeanScore,
            }),
            b.CvBestMeanPrAuc, b.CvPerFold,
            b.TestRocAuc, b.TestPrAuc, b.F1At05, b.F1AtBest, b.BestThreshold,
            b.ConfusionAt05, b.ConfusionAtBest, b.RocCurve, b.PrCurve, b.StrataBySigmaMkt,
            Note = "NormalizeMeanVariance applied for schema consistency; scale-invariant for trees",
        }, Path.Combine(dir, $"{name}_metrics.json"));

        ml.Model.Save(r.Model, null, Path.Combine(dir, $"{name}_model.zip"));
    }

    private static void WriteCoefficients(
        IReadOnlyList<(string Feature, double Coefficient)> coefficients,
        string dir, string name)
    {
        var rows = coefficients
            .OrderByDescending(c => Math.Abs(c.Coefficient))
            .Select(c => new object[] { c.Feature, c.Coefficient });
        Artifacts.WriteCsv(
            Path.Combine(dir, $"{name}_coefficients.csv"),
            new[] { "feature", "coefficient" },
            rows);
    }

    // ── Render (EDA + model plots via the Python renderer) ───────────────────

    public static int RunRender(
        string lotsCsv, string artifactsDir, string edaDir, string modelsDir)
    {
        int rc = PythonRunner.Run("scripts.eda",
            "--in",  lotsCsv,
            "--out", edaDir);
        if (rc != 0) return rc;
        return PythonRunner.Run("scripts.render",
            "--artifacts", artifactsDir,
            "--eda-out",   edaDir,
            "--models-out", modelsDir);
    }
}
