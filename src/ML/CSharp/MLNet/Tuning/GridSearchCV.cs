using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Splits;
using Microsoft.ML;

namespace DirectIndexing.ML.MLNet.Tuning;

public record GridResult(
    IReadOnlyDictionary<string, object> BestParams,
    double MeanCvScore,
    double[] PerFoldScores,
    IReadOnlyList<(IReadOnlyDictionary<string, object> Params, double MeanScore, double[] FoldScores)> All);

/// <summary>
/// Generic grid search with stratified K-fold cross-validation, scored on
/// average-precision (PR-AUC) by default. Conceptually:
/// <code>
///   GridSearchCV : ParamGrid × (Params → IEstimator) × Folds × Scorer → BestParams
/// </code>
/// A hyperparameter configuration is just a function from a parameter dict to
/// an estimator — the factory makes that explicit. No black box.
///
/// <para>The class-weight invariant is enforced here: for each fold, weights
/// are computed on that fold's training rows via
/// <see cref="ClassWeights.AttachBalancedWeights"/> before fitting. The test
/// fold's labels never enter the weight calculation.</para>
/// </summary>
public static class GridSearchCV
{
    public static GridResult Search(
        MLContext ml,
        IReadOnlyList<LotStateVector> trainData,
        Func<LotStateVector, int> labelSelector,
        Dictionary<string, object[]> grid,
        Func<IDictionary<string, object>, IEstimator<ITransformer>> estimatorFactory,
        Func<LotStateVector, bool> binaryLabel,
        int k = 5,
        int seed = 42)
    {
        var configs = Expand(grid).ToArray();
        var folds   = StratifiedKFold.Folds(trainData, labelSelector, k, seed).ToArray();

        var all = new List<(IReadOnlyDictionary<string, object> Params, double MeanScore, double[] FoldScores)>();
        IReadOnlyDictionary<string, object>? bestParams = null;
        double bestMean = double.NegativeInfinity;
        double[] bestFolds = Array.Empty<double>();

        foreach (var cfg in configs)
        {
            var foldScores = new double[folds.Length];
            for (int f = 0; f < folds.Length; f++)
            {
                var (train, val) = folds[f];

                var medians = MedianImputer.Fit(train);
                var trainW  = ClassWeights.AttachBalancedWeights(train, labelSelector);
                var trainImputed = trainW.Select(w => WithImputed(w, medians, binaryLabel)).ToList();
                var valImputed   = MedianImputer.Apply(val, medians, binaryLabel);

                var trainView = ml.Data.LoadFromEnumerable(trainImputed);
                var valView   = ml.Data.LoadFromEnumerable(valImputed);

                var estimator = estimatorFactory(cfg);
                var model     = estimator.Fit(trainView);
                var scored    = model.Transform(valView);

                foldScores[f] = BinaryMetrics.Compute(ml, scored).PrAuc;
            }

            double mean = foldScores.Average();
            all.Add((cfg, mean, foldScores));
            if (mean > bestMean)
            {
                bestMean = mean;
                bestParams = cfg;
                bestFolds = foldScores;
            }
        }

        return new GridResult(bestParams!, bestMean, bestFolds, all);
    }

    private static MLReadyRow WithImputed(WeightedRow w, Dictionary<string, float> m, Func<LotStateVector, bool> binaryLabel)
    {
        // WeightedRow inherits LotStateVector; reuse MedianImputer logic by
        // calling it on a one-element list and copying Weight across.
        var imputed = MedianImputer.Apply(new[] { (LotStateVector)w }, m, binaryLabel)[0];
        return imputed with { Weight = w.Weight };
    }

    private static IEnumerable<Dictionary<string, object>> Expand(Dictionary<string, object[]> grid)
    {
        var keys = grid.Keys.ToArray();
        var lens = keys.Select(k => grid[k].Length).ToArray();
        int total = lens.Aggregate(1, (a, b) => a * b);

        for (int n = 0; n < total; n++)
        {
            var dict = new Dictionary<string, object>(keys.Length);
            int idx = n;
            for (int i = 0; i < keys.Length; i++)
            {
                dict[keys[i]] = grid[keys[i]][idx % lens[i]];
                idx /= lens[i];
            }
            yield return dict;
        }
    }
}
