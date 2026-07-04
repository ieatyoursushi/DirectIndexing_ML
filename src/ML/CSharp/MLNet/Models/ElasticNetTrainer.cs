using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Schema;
using DirectIndexing.ML.MLNet.Splits;
using DirectIndexing.ML.MLNet.Tuning;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace DirectIndexing.ML.MLNet.Models;

/// <summary>
/// Elastic Net logistic regression via <c>SdcaLogisticRegressionBinaryTrainer</c>.
/// Elastic Net = logistic loss + L1 penalty (sparsity) + L2 penalty (shrinkage).
///
/// Grid (9 configs × 5 folds = 45 fits):
///   l1Regularization ∈ {0.001, 0.01, 0.1}
///   l2Regularization ∈ {0.001, 0.01, 0.1}
///
/// Semantic correspondence with sklearn
/// (see DataMemo/MLNetSemanticReconciliation.md §5):
///   sklearn <c>LogisticRegression(penalty='elasticnet', l1_ratio=ρ, C=c)</c>
///     → L1 = ρ/c, L2 = (1-ρ)/(2·c)
///   ML.NET <c>SdcaLogisticRegressionBinaryTrainer</c>
///     → L1Regularization (direct L1 weight), L2Regularization (direct L2 weight)
///
/// Coefficient extraction reuses the same reflection walk as LogisticTrainer
/// — SDCA produces the same <c>LinearBinaryModelParameters</c> base type.
/// </summary>
public static class ElasticNetTrainer
{
    public record ElasticNetOutput(
        BinaryMetricsResult Metrics,
        IReadOnlyList<(string Feature, double Coefficient)> Coefficients,
        double BestL1,
        double BestL2,
        double[] PerFoldCvScores,
        IReadOnlyList<(IReadOnlyDictionary<string, object> Params, double MeanScore)> AllConfigs,
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

        var search = GridSearchCV.Search(
            ml, train, r => label(r) ? 1 : 0, BuildGrid(),
            cfg => BuildEstimator(ml, cfg), label, k: 5, seed: 42);

        return new CvResult(
            ModelName:    "elnet",
            BestParams:   search.BestParams,
            MeanCvScore:  search.MeanCvScore,
            PerFoldScores: search.PerFoldScores,
            AllConfigs:   search.All.Select(a => (a.Params, a.MeanScore)).ToList());
    }

    public static ElasticNetOutput Run(
        MLContext ml,
        IReadOnlyList<LotStateVector> data,
        string target)
    {
        var (filtered, label) = SelectTarget(data, target);
        var (train, test) = DataSplit.TrainTest(
            filtered, r => label(r) ? 1 : 0, testFraction: 0.20, seed: 42);

        var search = GridSearchCV.Search(
            ml, train, r => label(r) ? 1 : 0, BuildGrid(),
            cfg => BuildEstimator(ml, cfg), label, k: 5, seed: 42);

        double bestL1 = (double)search.BestParams["l1Regularization"];
        double bestL2 = (double)search.BestParams["l2Regularization"];

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

        var finalEstimator = BuildEstimator(ml, (IDictionary<string, object>)search.BestParams);
        var model   = finalEstimator.Fit(trainView);
        var scored  = model.Transform(testView);
        var metrics = BinaryMetrics.Compute(ml, scored);
        var coefficients = ExtractCoefficients(model);

        return new ElasticNetOutput(
            Metrics:       metrics,
            Coefficients:  coefficients,
            BestL1:        bestL1,
            BestL2:        bestL2,
            PerFoldCvScores: search.PerFoldScores,
            AllConfigs:    search.All.Select(a => (a.Params, a.MeanScore)).ToList(),
            RowsTrain:     train.Count,
            RowsTest:      test.Count,
            Model:         model);
    }

    private static Dictionary<string, object[]> BuildGrid() => new()
    {
        ["l1Regularization"] = new object[] { 0.001, 0.01, 0.1 },
        ["l2Regularization"] = new object[] { 0.001, 0.01, 0.1 },
    };

    private static IEstimator<ITransformer> BuildEstimator(
        MLContext ml, IDictionary<string, object> cfg)
    {
        var options = new SdcaLogisticRegressionBinaryTrainer.Options
        {
            LabelColumnName         = FeatureLists.LabelCol,
            FeatureColumnName       = FeatureLists.FeaturesCol,
            ExampleWeightColumnName = FeatureLists.WeightCol,
            L1Regularization        = (float)(double)cfg["l1Regularization"],
            L2Regularization        = (float)(double)cfg["l2Regularization"],
            MaximumNumberOfIterations = 100,
        };
        return PreprocessingPipeline.Build(ml)
            .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(options));
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

    // Identical reflection walk to LogisticTrainer: SDCA returns the same
    // LinearBinaryModelParameters under CalibratedModelParametersBase.
    private static IReadOnlyList<(string Feature, double Coefficient)>
        ExtractCoefficients(ITransformer model)
    {
        if (model is IEnumerable<ITransformer> chain)
        {
            foreach (var t in chain)
            {
                var preds = FindLinearPredictor(t);
                if (preds is not null) return BuildCoefficientList(preds.Weights);
            }
        }
        else
        {
            var preds = FindLinearPredictor(model);
            if (preds is not null) return BuildCoefficientList(preds.Weights);
        }
        return Array.Empty<(string, double)>();
    }

    private static IReadOnlyList<(string Feature, double Coefficient)>
        BuildCoefficientList(IReadOnlyList<float> weights)
    {
        var names  = FeatureLists.NumericFeatures;
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
        switch (obj)
        {
            case LinearBinaryModelParameters lbm: return lbm;
            case Microsoft.ML.Calibrators.CalibratedModelParametersBase cmpb:
                return FindLinearPredictor(cmpb.SubModel);
            default:
                var prop = obj.GetType().GetProperty("Model");
                if (prop?.GetValue(obj) is { } inner)
                    return FindLinearPredictor(inner);
                return null;
        }
    }
}
