using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Schema;
using DirectIndexing.ML.MLNet.Splits;
using DirectIndexing.ML.MLNet.Tuning;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace DirectIndexing.ML.MLNet.Models;

/// <summary>
/// Linear regression on a binary target — a deliberate "poor fit" baseline.
///
/// <b>Why this fails as a probabilistic classifier</b> (from the DataMemo theory):
/// The oracle label represents a geometric level-curve boundary in feature space.
/// A calibrated probabilistic classifier (logistic, GBT, RF) models P(y=1|x) via a
/// function bounded in [0,1] with a smooth decision surface. OLS regression fits a
/// hyperplane to {0,1} outcomes — predictions are unbounded reals, heteroskedastic,
/// and structurally uncalibrated as probabilities. The fraction of predictions
/// outside [0,1] (reported in <see cref="LinRegOutput.FractionOutsideUnit"/>)
/// quantifies the failure directly.
///
/// Grid (3 configs × 5 folds):
///   l2Regularization ∈ {0.0001, 0.001, 0.01}
///
/// <b>Important difference from binary trainers:</b>
/// ML.NET's <c>SdcaRegressionTrainer</c> does not accept a bool <c>Label</c>
/// column — it requires a float. We use the <c>FloatLabel</c> field on
/// <see cref="MLReadyRow"/> (set by <see cref="MedianImputer.Apply"/> as 1f/0f)
/// and pass <c>LabelColumnName = "FloatLabel"</c>.
/// The regression trainer also supports <c>ExampleWeightColumnName</c>, so
/// balanced class weights are applied for a fair comparison.
/// </summary>
public static class LinearRegressionTrainer
{
    private class RegressionScoredRow
    {
        public float FloatLabel { get; set; }
        public float Score      { get; set; }
    }

    public record LinRegOutput(
        BinaryMetricsResult Metrics,
        IReadOnlyList<(string Feature, double Coefficient)> Coefficients,
        double BestL2,
        double Bias,
        double[] PerFoldCvScores,
        IReadOnlyList<(IReadOnlyDictionary<string, object> Params, double MeanScore)> AllConfigs,
        double FractionOutsideUnit,
        double MeanPrediction,
        int    RowsTrain,
        int    RowsTest,
        ITransformer Model);

    public static CvResult RunCV(
        MLContext ml,
        IReadOnlyList<LotStateVector> data,
        string target)
    {
        var (filtered, label) = SelectTarget(data, target);
        var (train, _) = DataSplit.TrainTest(
            filtered, r => label(r) ? 1 : 0, testFraction: 0.20, seed: 42);

        var (best, foldScores, all) = InlineCvSearch(ml, train, label, k: 5);

        return new CvResult(
            ModelName:    "linreg",
            BestParams:   best,
            MeanCvScore:  foldScores.Average(),
            PerFoldScores: foldScores,
            AllConfigs:   all.Select(a => ((IReadOnlyDictionary<string, object>)a.Params, a.Mean)).ToList());
    }

    public static LinRegOutput Run(
        MLContext ml,
        IReadOnlyList<LotStateVector> data,
        string target)
    {
        var (filtered, label) = SelectTarget(data, target);
        var (train, test) = DataSplit.TrainTest(
            filtered, r => label(r) ? 1 : 0, testFraction: 0.20, seed: 42);

        var (bestParams, perFoldScores, all) = InlineCvSearch(ml, train, label, k: 5);
        double bestL2 = (double)bestParams["l2Regularization"];

        // Refit on full training set with best L2 and balanced weights.
        var medians    = MedianImputer.Fit(train);
        var trainW     = ClassWeights.AttachBalancedWeights(train, r => label(r) ? 1 : 0);
        var trainReady = trainW.Select(w =>
        {
            var imp = MedianImputer.Apply(new[] { (LotStateVector)w }, medians, label)[0];
            return imp with { Weight = w.Weight };
        }).ToList();
        var testReady = MedianImputer.Apply(test, medians, label);

        var trainView = ml.Data.LoadFromEnumerable(trainReady);
        var testView  = ml.Data.LoadFromEnumerable(testReady);

        var finalEstimator = BuildEstimator(ml, bestParams);
        var model  = finalEstimator.Fit(trainView);
        var scored = model.Transform(testView);

        var rows = ml.Data.CreateEnumerable<RegressionScoredRow>(scored, reuseRowObject: false).ToArray();

        // Compute fraction of predictions outside the unit interval [0,1].
        double fractionOutside = rows.Count(r => r.Score < 0f || r.Score > 1f) / (double)rows.Length;
        double meanPrediction  = rows.Average(r => r.Score);

        // Proxy binary metrics: treat Score directly as the "probability".
        var proxyRows = rows.Select(r => new ScoredRow
        {
            Label          = r.FloatLabel >= 0.5f,
            Probability    = r.Score,
            Score          = r.Score,
            PredictedLabel = r.Score >= 0.5f,
        }).ToList();
        var metrics = BinaryMetrics.Compute(proxyRows);

        var coefficients = ExtractCoefficients(model, out double bias);

        return new LinRegOutput(
            Metrics:            metrics,
            Coefficients:       coefficients,
            BestL2:             bestL2,
            Bias:               bias,
            PerFoldCvScores:    perFoldScores,
            AllConfigs:         all.Select(a => ((IReadOnlyDictionary<string, object>)a.Params, a.Mean)).ToList(),
            FractionOutsideUnit: fractionOutside,
            MeanPrediction:     meanPrediction,
            RowsTrain:          train.Count,
            RowsTest:           test.Count,
            Model:              model);
    }

    private static (Dictionary<string, object> Best, double[] FoldScores,
        List<(Dictionary<string, object> Params, double Mean)> All)
        InlineCvSearch(
            MLContext ml,
            IReadOnlyList<LotStateVector> trainData,
            Func<LotStateVector, bool> label,
            int k)
    {
        var grid  = BuildGrid();
        var folds = DataSplit.Folds(trainData, r => label(r) ? 1 : 0, k, seed: 42).ToArray();

        var all = new List<(Dictionary<string, object> Params, double Mean)>();
        Dictionary<string, object>? bestParams = null;
        double bestMean = double.NegativeInfinity;
        double[] bestFolds = Array.Empty<double>();

        foreach (var cfg in ExpandGrid(grid))
        {
            var foldScores = new double[folds.Length];
            for (int f = 0; f < folds.Length; f++)
            {
                var (tr, val) = folds[f];
                var medians  = MedianImputer.Fit(tr);
                var trW      = ClassWeights.AttachBalancedWeights(tr, r => label(r) ? 1 : 0);
                var trReady  = trW.Select(w =>
                {
                    var imp = MedianImputer.Apply(new[] { (LotStateVector)w }, medians, label)[0];
                    return imp with { Weight = w.Weight };
                }).ToList();
                var valReady = MedianImputer.Apply(val, medians, label);

                var trView   = ml.Data.LoadFromEnumerable(trReady);
                var valView  = ml.Data.LoadFromEnumerable(valReady);

                var est    = BuildEstimator(ml, cfg);
                var fitted = est.Fit(trView);
                var scored = fitted.Transform(valView);

                foldScores[f] = ComputeProxyPrAuc(ml, scored);
            }

            double mean = foldScores.Average();
            all.Add((cfg, mean));
            if (mean > bestMean) { bestMean = mean; bestParams = cfg; bestFolds = foldScores; }
        }

        return (bestParams!, bestFolds, all);
    }

    private static double ComputeProxyPrAuc(MLContext ml, IDataView scored)
    {
        var rows = ml.Data.CreateEnumerable<RegressionScoredRow>(scored, reuseRowObject: false).ToList();
        var proxy = rows.Select(r => new ScoredRow
        {
            Label          = r.FloatLabel >= 0.5f,
            Probability    = r.Score,
            Score          = r.Score,
            PredictedLabel = r.Score >= 0.5f,
        }).ToList();
        try { return BinaryMetrics.Compute(proxy).PrAuc; }
        catch { return 0.0; }
    }

    private static Dictionary<string, object[]> BuildGrid() => new()
    {
        ["l2Regularization"] = new object[] { 0.0001, 0.001, 0.01 },
    };

    private static IEstimator<ITransformer> BuildEstimator(
        MLContext ml, IDictionary<string, object> cfg)
    {
        var options = new SdcaRegressionTrainer.Options
        {
            LabelColumnName         = "FloatLabel",
            FeatureColumnName       = FeatureLists.FeaturesCol,
            ExampleWeightColumnName = FeatureLists.WeightCol,
            L2Regularization        = (float)(double)cfg["l2Regularization"],
        };
        return PreprocessingPipeline.Build(ml)
            .Append(ml.Regression.Trainers.Sdca(options));
    }

    private static (List<LotStateVector> Filtered, Func<LotStateVector, bool> Label)
        SelectTarget(IReadOnlyList<LotStateVector> data, string target) =>
        target.ToLowerInvariant() switch
        {
            "oracle"  => (data.ToList(), r => r.Y_Oracle == 1),
            "soft_bt" => (data.Where(r => !float.IsNaN(r.Y_Soft_BT)).ToList(),
                          r => r.Y_Soft_BT > 0f),
            _ => throw new ArgumentException($"unknown target '{target}'"),
        };

    private static IReadOnlyList<(string, double)> ExtractCoefficients(
        ITransformer model, out double bias)
    {
        bias = 0.0;
        if (model is not IEnumerable<ITransformer> chain) return Array.Empty<(string, double)>();
        foreach (var t in chain)
        {
            var prop = t.GetType().GetProperty("Model");
            if (prop?.GetValue(t) is LinearRegressionModelParameters lrm)
            {
                bias = lrm.Bias;
                var weights = lrm.Weights;  // IReadOnlyList<float> in ML.NET 3.x
                var names   = FeatureLists.NumericFeatures;
                return Enumerable.Range(0, weights.Count)
                    .Select(i => (i < names.Length ? names[i] : $"Sector_{i - names.Length}",
                                  (double)weights[i]))
                    .ToList();
            }
        }
        return Array.Empty<(string, double)>();
    }

    private static IEnumerable<Dictionary<string, object>> ExpandGrid(
        Dictionary<string, object[]> grid)
    {
        var keys  = grid.Keys.ToArray();
        var lens  = keys.Select(k => grid[k].Length).ToArray();
        int total = lens.Aggregate(1, (a, b) => a * b);
        for (int n = 0; n < total; n++)
        {
            var dict = new Dictionary<string, object>(keys.Length);
            int idx  = n;
            for (int i = 0; i < keys.Length; i++)
            {
                dict[keys[i]] = grid[keys[i]][idx % lens[i]];
                idx /= lens[i];
            }
            yield return dict;
        }
    }
}
