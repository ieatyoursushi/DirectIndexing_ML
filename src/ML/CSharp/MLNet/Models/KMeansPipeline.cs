using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Schema;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace DirectIndexing.ML.MLNet.Models;

/// <summary>
/// K-means clustering on the per-symbol aggregate (mean of asset-level features
/// per Symbol — 503 rows, not 122K). Tunes k by maximising silhouette over a
/// candidate range. Mirrors Python <c>kmeans.py</c> in semantics; the
/// silhouette computation is O(n²) which is fine at n=503.
/// </summary>
public static class KMeansPipeline
{
    // Asset-level features only — the lot/portfolio-level features get averaged
    // across thousands of timesteps per symbol, which would just collapse to noise.
    private static readonly string[] ClusterFeatures =
    {
        "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200",
    };

    public record SymbolAggregate(string Symbol, string Sector, double[] Features);

    public record KMeansOutput(
        int BestK,
        IReadOnlyList<(int K, double Silhouette, double Inertia)> Elbow,
        IReadOnlyList<(string Symbol, string Sector, int ClusterId)> Assignments,
        double[][] Centers,
        ITransformer Model);

    public static KMeansOutput Run(
        MLContext ml,
        IReadOnlyList<LotStateVector> data,
        int[]? kCandidates = null)
    {
        kCandidates ??= new[] { 5, 10, 15, 20, 25 };

        var aggregates = BuildAggregates(data);
        var standardised = StandardiseAggregates(aggregates);

        var rowsForMLNet = standardised
            .Select(a => new ClusterRow
            {
                Features = a.Features.Select(d => (float)d).ToArray(),
                Symbol = a.Symbol,
                Sector = a.Sector,
            })
            .ToList();

        var view = ml.Data.LoadFromEnumerable(rowsForMLNet);

        var elbow = new List<(int K, double Silhouette, double Inertia)>();
        int bestK = kCandidates[0];
        double bestSilhouette = double.NegativeInfinity;
        ITransformer? bestModel = null;
        IReadOnlyList<int>? bestAssign = null;

        foreach (var k in kCandidates)
        {
            var trainer = ml.Clustering.Trainers.KMeans(new KMeansTrainer.Options
            {
                NumberOfClusters    = k,
                FeatureColumnName   = "Features",
                NumberOfThreads     = 1,
                MaximumNumberOfIterations = 200,
            });

            var model = trainer.Fit(view);
            var transformed = model.Transform(view);

            // ML.NET's KMeans labels clusters 1..k (uint); we want 0..k-1 for
            // zero-based array indexing throughout the rest of the pipeline.
            var assigns = ml.Data
                .CreateEnumerable<ClusterPrediction>(transformed, reuseRowObject: false)
                .Select(p => (int)p.PredictedLabel - 1)
                .ToArray();

            var sil = SilhouetteScore.Compute(
                standardised.Select(a => a.Features).ToArray(),
                assigns);

            double inertia = ComputeInertia(standardised, assigns, k);
            elbow.Add((k, sil, inertia));

            if (sil > bestSilhouette)
            {
                bestSilhouette = sil;
                bestK = k;
                bestModel = model;
                bestAssign = assigns;
            }
        }

        var assignments = standardised
            .Select((a, i) => (a.Symbol, a.Sector, bestAssign![i]))
            .ToList();

        // Centers: compute mean per cluster in standardised space.
        var centers = ComputeCenters(standardised, bestAssign!, bestK);

        return new KMeansOutput(bestK, elbow, assignments, centers, bestModel!);
    }

    private static IReadOnlyList<SymbolAggregate> BuildAggregates(IReadOnlyList<LotStateVector> data)
    {
        var groups = data
            .Where(r => !string.IsNullOrEmpty(r.Symbol))
            .Where(r =>
                !float.IsNaN(r.R_t) && !float.IsNaN(r.SigmaRange) &&
                !float.IsNaN(r.DeltaMA50) && !float.IsNaN(r.DeltaMA200))
            .GroupBy(r => r.Symbol);

        var result = new List<SymbolAggregate>();
        foreach (var g in groups)
        {
            var rows = g.ToList();
            double rt   = rows.Average(r => (double)r.R_t);
            double sr   = rows.Average(r => (double)r.SigmaRange);
            double m50  = rows.Average(r => (double)r.DeltaMA50);
            double m200 = rows.Average(r => (double)r.DeltaMA200);
            var sector  = rows[0].Sector ?? "Unknown";
            result.Add(new SymbolAggregate(g.Key, sector, new[] { rt, sr, m50, m200 }));
        }
        return result;
    }

    private static IReadOnlyList<SymbolAggregate> StandardiseAggregates(IReadOnlyList<SymbolAggregate> aggregates)
    {
        int p = ClusterFeatures.Length;
        var means = new double[p];
        var sds   = new double[p];

        for (int j = 0; j < p; j++)
        {
            means[j] = aggregates.Average(a => a.Features[j]);
        }
        for (int j = 0; j < p; j++)
        {
            double var = aggregates.Average(a =>
            {
                double d = a.Features[j] - means[j];
                return d * d;
            });
            sds[j] = Math.Sqrt(var);
        }

        return aggregates.Select(a =>
        {
            var f = new double[p];
            for (int j = 0; j < p; j++)
                f[j] = sds[j] > 1e-12 ? (a.Features[j] - means[j]) / sds[j] : 0.0;
            return new SymbolAggregate(a.Symbol, a.Sector, f);
        }).ToList();
    }

    private static double ComputeInertia(IReadOnlyList<SymbolAggregate> rows, IReadOnlyList<int> assigns, int k)
    {
        var centers = ComputeCenters(rows, assigns, k);
        double total = 0;
        for (int i = 0; i < rows.Count; i++)
        {
            var c = centers[assigns[i]];
            double sumSq = 0;
            for (int j = 0; j < c.Length; j++)
            {
                double d = rows[i].Features[j] - c[j];
                sumSq += d * d;
            }
            total += sumSq;
        }
        return total;
    }

    private static double[][] ComputeCenters(IReadOnlyList<SymbolAggregate> rows, IReadOnlyList<int> assigns, int k)
    {
        int p = ClusterFeatures.Length;
        var sums   = new double[k][];
        var counts = new int[k];
        for (int c = 0; c < k; c++) sums[c] = new double[p];

        for (int i = 0; i < rows.Count; i++)
        {
            int c = assigns[i];
            counts[c]++;
            for (int j = 0; j < p; j++) sums[c][j] += rows[i].Features[j];
        }

        var centers = new double[k][];
        for (int c = 0; c < k; c++)
        {
            centers[c] = new double[p];
            for (int j = 0; j < p; j++)
                centers[c][j] = counts[c] > 0 ? sums[c][j] / counts[c] : 0.0;
        }
        return centers;
    }

    private class ClusterRow
    {
        [VectorType(4)]
        public float[] Features { get; set; } = Array.Empty<float>();
        public string Symbol { get; set; } = "";
        public string Sector { get; set; } = "";
    }

    private class ClusterPrediction
    {
        [ColumnName("PredictedLabel")]
        public uint PredictedLabel { get; set; }
    }

    public static string[] FeatureNames => ClusterFeatures;
}
