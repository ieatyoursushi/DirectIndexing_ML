using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Schema;
using DirectIndexing.ML.MLNet.Splits;
using DirectIndexing.ML.MLNet.Tuning;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;

namespace DirectIndexing.ML.MLNet.Models;

/// <summary>
/// L2-regularized logistic regression with stratified 5-fold CV grid search
/// over <c>C ∈ {0.01, 0.1, 1.0, 10.0}</c>. Mirrors Python's
/// <c>logistic.py</c>; differences are documented in
/// <c>DataMemo/MLNetSemanticReconciliation.md</c>.
///
/// Two targets supported (target derivation matches <c>targets.py</c>):
///   "oracle"  — Y_Oracle (deterministic gate; sanity baseline; expect ROC ≈ 1).
///   "soft_bt" — (Y_Soft_BT > 0)  i.e. "any oracle fire in next 30 days".
///               NaN Y_Soft_BT rows are dropped before splitting.
/// </summary>
public static class LogisticTrainer
{
    public record LogisticOutput(
        BinaryMetricsResult Metrics,
        IReadOnlyList<(string Feature, double Coefficient)> Coefficients,
        double BestC,
        double L2Used,
        double[] PerFoldCvScores,
        IReadOnlyList<(double C, double MeanScore)> AllConfigs,
        int RowsTrain,
        int RowsTest,
        ITransformer Model);

    /// <summary>
    /// CV-only path used by <c>RunAllSupervised</c> for champion selection.
    /// Does NOT touch the held-out test set.
    /// </summary>
    public static CvResult RunCV(
        MLContext ml,
        IReadOnlyList<LotStateVector> data,
        string target)
    {
        var (filtered, label) = SelectTarget(data, target);
        var (train, _) = StratifiedSplit.Split(
            filtered, r => label(r) ? 1 : 0, testFraction: 0.20, seed: 42);

        var grid = new Dictionary<string, object[]>
        {
            ["C"] = new object[] { 0.01, 0.1, 1.0, 10.0 },
        };
        var search = GridSearchCV.Search(
            ml, train, r => label(r) ? 1 : 0, grid,
            cfg => BuildEstimator(ml, (double)cfg["C"]),
            label, k: 5, seed: 42);

        return new CvResult(
            ModelName:   "logistic",
            BestParams:  search.BestParams,
            MeanCvScore: search.MeanCvScore,
            PerFoldScores: search.PerFoldScores,
            AllConfigs:  search.All.Select(a => (a.Params, a.MeanScore)).ToList());
    }

    public static LogisticOutput Run(
        MLContext ml,
        IReadOnlyList<LotStateVector> data,
        string target)
    {
        var (filtered, label) = SelectTarget(data, target);

        var (train, test) = StratifiedSplit.Split(
            filtered, r => label(r) ? 1 : 0, testFraction: 0.20, seed: 42);

        // Grid search on training fold only.
        var grid = new Dictionary<string, object[]>
        {
            ["C"] = new object[] { 0.01, 0.1, 1.0, 10.0 },
        };

        var search = GridSearchCV.Search(
            ml, train, r => label(r) ? 1 : 0,
            grid,
            cfg => BuildEstimator(ml, (double)cfg["C"]),
            label, k: 5, seed: 42);

        double bestC = (double)search.BestParams["C"];

        // Refit on FULL training set with best C and proper weights.
        var medians = MedianImputer.Fit(train);
        var trainW  = ClassWeights.AttachBalancedWeights(train, r => label(r) ? 1 : 0);
        var trainReady = trainW.Select(w =>
        {
            var imp = MedianImputer.Apply(new[] { (LotStateVector)w }, medians, label)[0];
            return imp with { Weight = w.Weight };
        }).ToList();
        var testReady  = MedianImputer.Apply(test, medians, label);

        var trainView = ml.Data.LoadFromEnumerable(trainReady);
        var testView  = ml.Data.LoadFromEnumerable(testReady);

        var finalEstimator = BuildEstimator(ml, bestC);
        var model  = finalEstimator.Fit(trainView);
        var scored = model.Transform(testView);

        var metrics = BinaryMetrics.Compute(ml, scored);

        // Extract per-feature coefficients from the trained LR predictor.
        var coefficients = ExtractCoefficients(model);

        var allConfigs = search.All
            .Select(c => ((double)c.Params["C"], c.MeanScore))
            .ToList();

        return new LogisticOutput(
            Metrics: metrics,
            Coefficients: coefficients,
            BestC: bestC,
            L2Used: 1.0 / bestC,
            PerFoldCvScores: search.PerFoldScores,
            AllConfigs: allConfigs,
            RowsTrain: train.Count,
            RowsTest: test.Count,
            Model: model);
    }

    private static IEstimator<ITransformer> BuildEstimator(MLContext ml, double C)
    {
        var options = new LbfgsLogisticRegressionBinaryTrainer.Options
        {
            LabelColumnName            = FeatureLists.LabelCol,
            FeatureColumnName          = FeatureLists.FeaturesCol,
            ExampleWeightColumnName    = FeatureLists.WeightCol,
            L2Regularization           = (float)(1.0 / C),  // see SemanticReconciliation §1
            L1Regularization           = 0f,
            MaximumNumberOfIterations  = 200,
        };

        return PreprocessingPipeline.Build(ml)
            .Append(ml.BinaryClassification.Trainers.LbfgsLogisticRegression(options));
    }

    private static (List<LotStateVector> Filtered, Func<LotStateVector, bool> Label)
        SelectTarget(IReadOnlyList<LotStateVector> data, string target)
    {
        return target.ToLowerInvariant() switch
        {
            "oracle" => (
                data.ToList(),
                r => r.Y_Oracle == 1),

            "soft_bt" => (
                data.Where(r => !float.IsNaN(r.Y_Soft_BT)).ToList(),
                r => r.Y_Soft_BT > 0f),

            _ => throw new ArgumentException($"unknown target '{target}'"),
        };
    }

    private static IReadOnlyList<(string Feature, double Coefficient)>
        ExtractCoefficients(ITransformer model)
    {
        // TransformerChain<T> is generic over the last transformer's concrete
        // type, so we can't pattern-match against TransformerChain<ITransformer>.
        // It does implement IEnumerable<ITransformer> however.
        if (model is not IEnumerable<ITransformer> chain)
        {
            var single = FindLinearPredictor(model);
            return single is null
                ? Array.Empty<(string, double)>()
                : BuildCoefficientList(single.Weights);
        }

        foreach (var t in chain)
        {
            var preds = FindLinearPredictor(t);
            if (preds is null) continue;
            return BuildCoefficientList(preds.Weights);
        }
        return Array.Empty<(string, double)>();
    }

    private static IReadOnlyList<(string Feature, double Coefficient)>
        BuildCoefficientList(IReadOnlyList<float> weights)
    {
        // After preprocessing, Features = [15 numeric] ++ [one-hot Sector].
        // We don't know one-hot vocabulary size statically — slice the numeric
        // prefix (which IS known) and label the rest by positional index.
        var names = FeatureLists.NumericFeatures;
        var result = new List<(string, double)>(weights.Count);
        for (int i = 0; i < weights.Count; i++)
        {
            var name = i < names.Length ? names[i] : $"Sector_{i - names.Length}";
            result.Add((name, weights[i]));
        }
        return result;
    }

    private static LinearBinaryModelParameters? FindLinearPredictor(object obj)
    {
        // Drill through CalibratedModelParametersBase + PredictionTransformer wrappers.
        switch (obj)
        {
            case LinearBinaryModelParameters lbm:
                return lbm;
            case Microsoft.ML.Calibrators.CalibratedModelParametersBase cmpb:
                return FindLinearPredictor(cmpb.SubModel);
            default:
                var prop = obj.GetType().GetProperty("Model");
                if (prop != null)
                {
                    var inner = prop.GetValue(obj);
                    if (inner != null) return FindLinearPredictor(inner);
                }
                return null;
        }
    }
}
