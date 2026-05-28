using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Io;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Models;
using DirectIndexing.ML.MLNet.Schema;
using Microsoft.ML;

namespace DirectIndexing.ML.MLNet;

/// <summary>
/// Top-level orchestration for the ML.NET pipeline. Each method consumes a
/// typed <c>List&lt;LotStateVector&gt;</c> and writes the JSON / CSV / .zip
/// artifacts under <c>data/artifacts-mlnet/</c>. Rendering (PNGs, HTML) is
/// delegated to the Python renderer via <see cref="PythonRunner"/>.
/// </summary>
public static class MLnetPipeline
{
    private record MetricsPayload(
        string Target,
        int    RowsTrain,
        int    RowsTest,
        double BestC,
        double L2Used,
        double CvBestMeanPrAuc,
        double[] CvPerFold,
        IReadOnlyList<ConfigScore> AllConfigs,
        double TestRocAuc,
        double TestPrAuc,
        double F1At05,
        double F1AtBest,
        double BestThreshold,
        Confusion ConfusionAt05,
        Confusion ConfusionAtBest,
        CurvePoint[] RocCurve,
        CurvePoint[] PrCurve);

    private record ConfigScore(double C, double MeanCvPrAuc);
    private record Confusion(int Tp, int Fp, int Tn, int Fn);

    public static void RunSupervised(
        IReadOnlyList<LotStateVector> data,
        string target,
        string artifactsDir)
    {
        Console.WriteLine($"[MLnetPipeline] supervised target={target}  rows={data.Count}");
        var ml = new MLContext(seed: 42);

        var result = LogisticTrainer.Run(ml, data, target);

        var name = $"logistic_{target.ToLowerInvariant()}";

        var payload = new MetricsPayload(
            Target:            target,
            RowsTrain:         result.RowsTrain,
            RowsTest:          result.RowsTest,
            BestC:             result.BestC,
            L2Used:            result.L2Used,
            CvBestMeanPrAuc:   result.PerFoldCvScores.Average(),
            CvPerFold:         result.PerFoldCvScores,
            AllConfigs:        result.AllConfigs.Select(c => new ConfigScore(c.C, c.MeanScore)).ToList(),
            TestRocAuc:        result.Metrics.RocAuc,
            TestPrAuc:         result.Metrics.PrAuc,
            F1At05:            result.Metrics.F1At05,
            F1AtBest:          result.Metrics.F1AtBest,
            BestThreshold:     result.Metrics.BestThreshold,
            ConfusionAt05:     new Confusion(result.Metrics.Tp05, result.Metrics.Fp05, result.Metrics.Tn05, result.Metrics.Fn05),
            ConfusionAtBest:   new Confusion(result.Metrics.TpBest, result.Metrics.FpBest, result.Metrics.TnBest, result.Metrics.FnBest),
            RocCurve:          result.Metrics.RocCurve,
            PrCurve:           result.Metrics.PrCurve);

        Directory.CreateDirectory(artifactsDir);
        Artifacts.WriteJson(payload, Path.Combine(artifactsDir, $"{name}_metrics.json"));

        var coeffRows = result.Coefficients
            .OrderByDescending(c => Math.Abs(c.Coefficient))
            .Select(c => new object[] { c.Feature, c.Coefficient });
        Artifacts.WriteCsv(
            Path.Combine(artifactsDir, $"{name}_coefficients.csv"),
            new[] { "feature", "coefficient" },
            coeffRows);

        ml.Model.Save(result.Model, inputSchema: null,
            filePath: Path.Combine(artifactsDir, $"{name}_model.zip"));
    }

    public static void RunUnsupervised(
        IReadOnlyList<LotStateVector> data,
        string artifactsDir)
    {
        Console.WriteLine($"[MLnetPipeline] unsupervised  rows={data.Count}");
        Directory.CreateDirectory(artifactsDir);
        var ml = new MLContext(seed: 42);

        // ── PCA ──
        var pca = PcaPipeline.Run(ml, data, cumulativeVarianceThreshold: 0.95);

        Artifacts.WriteJson(new
        {
            featureNames        = FeatureLists.NumericFeatures,
            explainedVariance   = pca.ExplainedVariance,
            cumulativeVariance  = pca.CumulativeVariance,
            nKept               = pca.NKept,
        }, Path.Combine(artifactsDir, "pca_components.json"));

        // pca_scree.json mirrors pca_components but in a renderer-friendly shape.
        Artifacts.WriteJson(new
        {
            k                  = Enumerable.Range(1, pca.ExplainedVariance.Length).ToArray(),
            explainedVariance  = pca.ExplainedVariance,
            cumulativeVariance = pca.CumulativeVariance,
            threshold          = 0.95,
            nKept              = pca.NKept,
        }, Path.Combine(artifactsDir, "pca_scree.json"));

        // pca_loadings.csv: rows = principal components, cols = features.
        var loadingHeader = new[] { "component" }.Concat(FeatureLists.NumericFeatures);
        var loadingRows = pca.Loadings.Select((row, i) =>
            new[] { (object)$"PC{i + 1}" }.Concat(row.Cast<object>()));
        Artifacts.WriteCsv(
            Path.Combine(artifactsDir, "pca_loadings.csv"),
            loadingHeader,
            loadingRows);

        ml.Model.Save(pca.Model, inputSchema: null,
            filePath: Path.Combine(artifactsDir, "pca_model.zip"));

        // ── K-means ──
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

    public static int RunRender(string lotsCsv, string artifactsDir, string edaDir, string modelsDir)
    {
        // EDA (reads lots.csv directly) and report (reads artifacts JSON).
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
