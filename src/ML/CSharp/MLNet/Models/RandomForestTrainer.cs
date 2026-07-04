using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Metrics;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Schema;
using DirectIndexing.ML.MLNet.Splits;
using DirectIndexing.ML.MLNet.Tuning;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;

namespace DirectIndexing.ML.MLNet.Models;

/// <summary>
/// Random Forest binary classifier via <c>FastForestBinaryTrainer</c>.
///
/// Grid (12 configs × 5 folds = 60 fits):
///   numberOfTrees   ∈ {100, 200}
///   numberOfLeaves  ∈ {20, 31}
///   featureFraction ∈ {0.3, 0.5, 0.7}   ← the ML.NET analogue of sklearn's
///                                          mtry / max_features (m), the fraction
///                                          of features sampled per split.
///
/// Tuning <c>featureFraction</c> is the decorrelation knob of a random forest:
/// a lower fraction forces trees to disagree (lower variance, possibly higher
/// bias). It is the single most important RF hyperparameter and is now searched
/// rather than held at the FastForest default.
///
/// <b>Preprocessing note:</b> NormalizeMeanVariance is scale-invariant for trees
/// and kept only for schema consistency with linear trainers.
/// </summary>
public static class RandomForestTrainer
{
    public record RfOutput(
        BinaryMetricsResult Metrics,
        int    BestNumberOfTrees,
        int    BestNumberOfLeaves,
        double FeatureFraction,
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
            ModelName:    "rf",
            BestParams:   search.BestParams,
            MeanCvScore:  search.MeanCvScore,
            PerFoldScores: search.PerFoldScores,
            AllConfigs:   search.All.Select(a => (a.Params, a.MeanScore)).ToList());
    }

    public static RfOutput Run(
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

        return new RfOutput(
            Metrics:           metrics,
            BestNumberOfTrees:  (int)search.BestParams["numberOfTrees"],
            BestNumberOfLeaves: (int)search.BestParams["numberOfLeaves"],
            FeatureFraction:   (double)search.BestParams["featureFraction"],
            PerFoldCvScores:   search.PerFoldScores,
            AllConfigs:        search.All.Select(a => (a.Params, a.MeanScore)).ToList(),
            RowsTrain:         train.Count,
            RowsTest:          test.Count,
            Model:             model);
    }

    private static Dictionary<string, object[]> BuildGrid() => new()
    {
        ["numberOfTrees"]   = new object[] { 100, 200 },
        ["numberOfLeaves"]  = new object[] { 20,  31  },
        ["featureFraction"] = new object[] { 0.3, 0.5, 0.7 },
    };

    private static IEstimator<ITransformer> BuildEstimator(
        MLContext ml, IDictionary<string, object> cfg)
    {
        var options = new FastForestBinaryTrainer.Options
        {
            LabelColumnName         = FeatureLists.LabelCol,
            FeatureColumnName       = FeatureLists.FeaturesCol,
            ExampleWeightColumnName = FeatureLists.WeightCol,
            NumberOfTrees           = (int)cfg["numberOfTrees"],
            NumberOfLeaves          = (int)cfg["numberOfLeaves"],
            FeatureFraction         = (double)cfg["featureFraction"],
        };
        return PreprocessingPipeline.Build(ml)
            .Append(ml.BinaryClassification.Trainers.FastForest(options));
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
}
