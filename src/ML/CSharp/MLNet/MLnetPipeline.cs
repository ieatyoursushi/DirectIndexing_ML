using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Io;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Models;
using DirectIndexing.ML.MLNet.Schema;
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
        CurvePointDto[] PrCurve);

    private static BaseMetrics ToBase(
        BinaryMetricsResult m, string target, int train, int test, double[] cvFolds) =>
        new(target, train, test,
            cvFolds.Average(), cvFolds,
            m.RocAuc, m.PrAuc, m.F1At05, m.F1AtBest, m.BestThreshold,
            new(m.Tp05, m.Fp05, m.Tn05, m.Fn05),
            new(m.TpBest, m.FpBest, m.TnBest, m.FnBest),
            m.RocCurve.Select(p => new CurvePointDto(p.Threshold, p.X, p.Y)).ToArray(),
            m.PrCurve .Select(p => new CurvePointDto(p.Threshold, p.X, p.Y)).ToArray());

    // ── Per-model public entry points (full CV + test eval) ──────────────────

    /// <summary>
    /// Run a specific model's full pipeline (CV grid search + champion test eval).
    /// Used by individual <c>mlnet-gbt</c>, <c>mlnet-rf</c>, etc. cases.
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
            case "rf":       WriteRf      (RandomForestTrainer.Run(ml, data, target), target, artifactsDir, ml); break;
            case "elnet":    WriteElnet   (ElasticNetTrainer.Run(ml, data, target), target, artifactsDir, ml); break;
            case "linreg":   WriteLinreg  (LinearRegressionTrainer.Run(ml, data, target), target, artifactsDir, ml); break;
            default: throw new ArgumentException($"unknown model '{modelName}'");
        }
    }

    /// <summary>
    /// Champion-selection run: CV-tunes all models, emits a leaderboard, then runs
    /// full test evaluation only for the top-2 classifiers + the linreg poor-fit
    /// demonstration. The test set is never touched until after the CV ranking.
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
            LogisticTrainer              .RunCV(ml, data, target),
            GradientBoostedTreesTrainer  .RunCV(ml, data, target),
            RandomForestTrainer          .RunCV(ml, data, target),
            ElasticNetTrainer            .RunCV(ml, data, target),
            LinearRegressionTrainer      .RunCV(ml, data, target),
        };

        // 2. Pick top-2 classifiers (exclude linreg — regression demonstration).
        var champions = cvResults
            .Where(r => r.ModelName != "linreg")
            .OrderByDescending(r => r.MeanCvScore)
            .Take(2)
            .Select(r => r.ModelName)
            .ToHashSet();

        // 3. Emit CV leaderboard.
        var leaderboard = cvResults
            .OrderByDescending(r => r.MeanCvScore)
            .Select(r => new
            {
                r.ModelName,
                MeanCvPrAuc  = r.MeanCvScore,
                r.PerFoldScores,
                IsChampion   = champions.Contains(r.ModelName),
                ModelType    = r.ModelName == "linreg" ? "regression_demonstration" : "classifier",
                AllConfigs   = r.AllConfigs.Select(c => new
                {
                    Params   = c.Params,
                    MeanPrAuc = c.MeanScore,
                }),
            });
        Artifacts.WriteJson(leaderboard,
            Path.Combine(artifactsDir, $"{target}_cv_leaderboard.json"));

        // 4. Full test eval — champions only + linreg poor-fit demonstration.
        foreach (var name in champions)
            RunSupervisedModel(name, data, target, artifactsDir);

        // Linreg always runs as the demonstration, regardless of CV rank.
        RunSupervisedModel("linreg", data, target, artifactsDir);
    }

    // ── Backward-compatible single-model entry point ─────────────────────────

    /// <summary>Runs logistic regression only. Kept for backward compatibility with
    /// <c>mlnet-supervised</c> and <c>mlnet-baseline</c> cases.</summary>
    public static void RunSupervised(
        IReadOnlyList<LotStateVector> data,
        string target,
        string artifactsDir)
    {
        Console.WriteLine($"[MLnetPipeline] supervised target={target} rows={data.Count}");
        Directory.CreateDirectory(artifactsDir);
        var ml = new MLContext(seed: 42);
        WriteLogistic(LogisticTrainer.Run(ml, data, target), target, artifactsDir, ml);
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
            b.ConfusionAt05, b.ConfusionAtBest, b.RocCurve, b.PrCurve,
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
            b.ConfusionAt05, b.ConfusionAtBest, b.RocCurve, b.PrCurve,
            Note = "NormalizeMeanVariance applied for schema consistency; scale-invariant for trees",
        }, Path.Combine(dir, $"{name}_metrics.json"));

        ml.Model.Save(r.Model, null, Path.Combine(dir, $"{name}_model.zip"));
    }

    private static void WriteRf(
        RandomForestTrainer.RfOutput r, string target, string dir, MLContext ml)
    {
        var name = $"rf_{target}";
        var b    = ToBase(r.Metrics, target, r.RowsTrain, r.RowsTest, r.PerFoldCvScores);
        Artifacts.WriteJson(new
        {
            b.Target, b.RowsTrain, b.RowsTest,
            BestNumberOfTrees  = r.BestNumberOfTrees,
            BestNumberOfLeaves = r.BestNumberOfLeaves,
            FeatureFraction    = r.FeatureFraction,
            AllConfigs = r.AllConfigs.Select(c => new
            {
                Params      = c.Params,
                MeanCvPrAuc = c.MeanScore,
            }),
            b.CvBestMeanPrAuc, b.CvPerFold,
            b.TestRocAuc, b.TestPrAuc, b.F1At05, b.F1AtBest, b.BestThreshold,
            b.ConfusionAt05, b.ConfusionAtBest, b.RocCurve, b.PrCurve,
        }, Path.Combine(dir, $"{name}_metrics.json"));

        ml.Model.Save(r.Model, null, Path.Combine(dir, $"{name}_model.zip"));
    }

    private static void WriteElnet(
        ElasticNetTrainer.ElasticNetOutput r, string target, string dir, MLContext ml)
    {
        var name = $"elnet_{target}";
        var b    = ToBase(r.Metrics, target, r.RowsTrain, r.RowsTest, r.PerFoldCvScores);
        Artifacts.WriteJson(new
        {
            b.Target, b.RowsTrain, b.RowsTest,
            BestL1 = r.BestL1,
            BestL2 = r.BestL2,
            AllConfigs = r.AllConfigs.Select(c => new
            {
                Params      = c.Params,
                MeanCvPrAuc = c.MeanScore,
            }),
            b.CvBestMeanPrAuc, b.CvPerFold,
            b.TestRocAuc, b.TestPrAuc, b.F1At05, b.F1AtBest, b.BestThreshold,
            b.ConfusionAt05, b.ConfusionAtBest, b.RocCurve, b.PrCurve,
        }, Path.Combine(dir, $"{name}_metrics.json"));

        WriteCoefficients(r.Coefficients, dir, name);
        ml.Model.Save(r.Model, null, Path.Combine(dir, $"{name}_model.zip"));
    }

    private static void WriteLinreg(
        LinearRegressionTrainer.LinRegOutput r, string target, string dir, MLContext ml)
    {
        var name = $"linreg_{target}";
        var b    = ToBase(r.Metrics, target, r.RowsTrain, r.RowsTest, r.PerFoldCvScores);
        Artifacts.WriteJson(new
        {
            b.Target, b.RowsTrain, b.RowsTest,
            ModelType        = "regression_demonstration",
            BestL2           = r.BestL2,
            Bias             = r.Bias,
            FractionOutsideUnit = r.FractionOutsideUnit,
            MeanPrediction   = r.MeanPrediction,
            AllConfigs = r.AllConfigs.Select(c => new
            {
                Params      = c.Params,
                MeanCvPrAuc = c.MeanScore,
            }),
            b.CvBestMeanPrAuc, b.CvPerFold,
            b.TestRocAuc, b.TestPrAuc, b.F1At05, b.F1AtBest, b.BestThreshold,
            b.ConfusionAt05, b.ConfusionAtBest, b.RocCurve, b.PrCurve,
            Note = "Score treated as probability proxy; predictions outside [0,1] are the primary failure signal",
        }, Path.Combine(dir, $"{name}_metrics.json"));

        WriteCoefficients(r.Coefficients, dir, name);
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

    // ── Unsupervised / render (unchanged) ────────────────────────────────────

    public static void RunUnsupervised(
        IReadOnlyList<LotStateVector> data,
        string artifactsDir)
    {
        Console.WriteLine($"[MLnetPipeline] unsupervised  rows={data.Count}");
        Directory.CreateDirectory(artifactsDir);
        var ml = new MLContext(seed: 42);

        var pca = PcaPipeline.Run(ml, data, cumulativeVarianceThreshold: 0.95);

        Artifacts.WriteJson(new
        {
            featureNames       = FeatureLists.NumericFeatures,
            explainedVariance  = pca.ExplainedVariance,
            cumulativeVariance = pca.CumulativeVariance,
            nKept              = pca.NKept,
        }, Path.Combine(artifactsDir, "pca_components.json"));

        Artifacts.WriteJson(new
        {
            k                  = Enumerable.Range(1, pca.ExplainedVariance.Length).ToArray(),
            explainedVariance  = pca.ExplainedVariance,
            cumulativeVariance = pca.CumulativeVariance,
            threshold          = 0.95,
            nKept              = pca.NKept,
        }, Path.Combine(artifactsDir, "pca_scree.json"));

        var loadingHeader = new[] { "component" }.Concat(FeatureLists.NumericFeatures);
        var loadingRows   = pca.Loadings.Select((row, i) =>
            new[] { (object)$"PC{i + 1}" }.Concat(row.Cast<object>()));
        Artifacts.WriteCsv(
            Path.Combine(artifactsDir, "pca_loadings.csv"),
            loadingHeader, loadingRows);

        ml.Model.Save(pca.Model, null, Path.Combine(artifactsDir, "pca_model.zip"));

        var km = KMeansPipeline.Run(ml, data);

        Artifacts.WriteJson(new
        {
            featureNames = KMeansPipeline.FeatureNames,
            bestK        = km.BestK,
            centers      = km.Centers,
        }, Path.Combine(artifactsDir, "kmeans_centers.json"));

        Artifacts.WriteJson(new
        {
            ks         = km.Elbow.Select(e => e.K).ToArray(),
            silhouette = km.Elbow.Select(e => e.Silhouette).ToArray(),
            inertia    = km.Elbow.Select(e => e.Inertia).ToArray(),
            bestK      = km.BestK,
        }, Path.Combine(artifactsDir, "kmeans_elbow.json"));

        Artifacts.WriteJson(
            km.Assignments.ToDictionary(a => a.Symbol, a => a.ClusterId),
            Path.Combine(artifactsDir, "cluster_assignments.json"));

        Artifacts.WriteCsv(
            Path.Combine(artifactsDir, "cluster_assignments.csv"),
            new[] { "symbol", "sector", "cluster_id" },
            km.Assignments.Select(a => new object[] { a.Symbol, a.Sector, a.ClusterId }));
    }

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
