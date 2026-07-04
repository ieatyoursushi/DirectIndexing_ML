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
/// Gradient Boosted Trees binary classifier via <c>FastTreeBinaryTrainer</c>.
///
/// Grid (8 configs × 5 folds = 40 fits):
///   numberOfTrees  ∈ {100, 200}
///   learningRate   ∈ {0.10, 0.05}
///   numberOfLeaves ∈ {20, 31}
///
/// <b>Preprocessing note:</b> NormalizeMeanVariance is applied by
/// <see cref="PreprocessingPipeline"/> but is a semantic no-op for tree models
/// — splits are scale-invariant. It is kept in the chain for schema consistency
/// with the logistic and elastic-net trainers.
///
/// <b>Class balance:</b> balanced example weights are attached via
/// <c>ExampleWeightColumnName</c>, same pattern as the logistic trainer.
/// </summary>
public static class GradientBoostedTreesTrainer
{
    public record GbtOutput(
        BinaryMetricsResult Metrics,
        int    BestNumberOfTrees,
        double BestLearningRate,
        int    BestNumberOfLeaves,
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
            ModelName:    "gbt",
            BestParams:   search.BestParams,
            MeanCvScore:  search.MeanCvScore,
            PerFoldScores: search.PerFoldScores,
            AllConfigs:   search.All.Select(a => (a.Params, a.MeanScore)).ToList());
    }

    public static GbtOutput Run(
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

        // Refit on full training set with best params + balanced weights.
        var medians   = MedianImputer.Fit(train);
        var trainW    = ClassWeights.AttachBalancedWeights(train, r => label(r) ? 1 : 0);
        var trainReady = trainW.Select(w =>
        {
            var imp = MedianImputer.Apply(new[] { (LotStateVector)w }, medians, label)[0];
            return imp with { Weight = w.Weight };
        }).ToList();
        var testReady = MedianImputer.Apply(test, medians, label);

        var trainView = ml.Data.LoadFromEnumerable(trainReady);
        var testView  = ml.Data.LoadFromEnumerable(testReady);

        var finalEstimator = BuildEstimator(ml, (IDictionary<string, object>)search.BestParams);
        var model  = finalEstimator.Fit(trainView);
        var scored = model.Transform(testView);
        var metrics = BinaryMetrics.Compute(ml, scored);

        return new GbtOutput(
            Metrics:           metrics,
            BestNumberOfTrees:  (int)search.BestParams["numberOfTrees"],
            BestLearningRate:   (double)search.BestParams["learningRate"],
            BestNumberOfLeaves: (int)search.BestParams["numberOfLeaves"],
            PerFoldCvScores:   search.PerFoldScores,
            AllConfigs:        search.All.Select(a => (a.Params, a.MeanScore)).ToList(),
            RowsTrain:         train.Count,
            RowsTest:          test.Count,
            Model:             model);
    }

    private static Dictionary<string, object[]> BuildGrid() => new()
    {
        ["numberOfTrees"]  = new object[] { 100,  200  },
        ["learningRate"]   = new object[] { 0.10, 0.05 },
        ["numberOfLeaves"] = new object[] { 20,   31   },
    };

    private static IEstimator<ITransformer> BuildEstimator(
        MLContext ml, IDictionary<string, object> cfg)
    {
        var options = new FastTreeBinaryTrainer.Options
        {
            LabelColumnName         = FeatureLists.LabelCol,
            FeatureColumnName       = FeatureLists.FeaturesCol,
            ExampleWeightColumnName = FeatureLists.WeightCol,
            NumberOfTrees           = (int)cfg["numberOfTrees"],
            LearningRate            = (double)cfg["learningRate"],
            NumberOfLeaves          = (int)cfg["numberOfLeaves"],
        };
        return PreprocessingPipeline.Build(ml)
            .Append(ml.BinaryClassification.Trainers.FastTree(options));
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
