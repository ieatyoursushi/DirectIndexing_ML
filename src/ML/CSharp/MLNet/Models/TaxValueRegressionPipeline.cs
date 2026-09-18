using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Io;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Schema;
using DirectIndexing.ML.MLNet.Splits;
using Microsoft.ML;

namespace DirectIndexing.ML.MLNet.Models;

/// <summary>
/// Continuous regression on <c>Y_TaxValue</c> (v0.25, issue #17 family).
///
/// The target is the TaxLedger's capacity-aware harvest value
/// taxValue_k = τ(h)·min(loss, capacity) + τ_future·max(loss − capacity, 0)·δ.
/// Since the <c>TaxValue</c> FEATURE equals the target by construction, the
/// feature set is <see cref="FeatureLists.NumericFeaturesTaxValueRegression"/>
/// (everything except TaxValue): the task is recovering g(ledger, H, L) from
/// raw features — min/max kinks and a rate jump at H = 365 that a linear model
/// cannot represent and trees can. The regression analogue of the
/// classification story.
///
/// Two models, mirroring that contrast:
///   • SDCA linear regression (normalized features) — the linear baseline.
///   • FastTree regression — the tree recovery.
///
/// Notes:
///   • The target is zero-inflated (~98% of rows have no harvestable loss), so
///     alongside overall test RMSE/MAE/R² the artifact reports RMSE on the
///     TaxValue &gt; 0 subset, where the kinked structure actually lives.
///   • Split is stratified on Y_TaxValue &gt; 0 so the thin positive region is
///     represented in both folds. Medians are fit on the training fold only
///     (same leakage discipline as the classification pipeline).
///   • Fixed sensible hyperparameters; no CV grid — this is a diagnostic
///     target, not a champion contest. Tunable later if it earns a headline.
/// </summary>
public static class TaxValueRegressionPipeline
{
    public static void Run(
        IReadOnlyList<LotStateVector> data,
        string artifactsDir)
    {
        Console.WriteLine($"[TaxValueRegression] rows={data.Count} " +
                          $"features={FeatureLists.NumericFeaturesTaxValueRegression.Length} (TaxValue excluded)");
        Directory.CreateDirectory(artifactsDir);
        var ml = new MLContext(seed: 42);

        var (train, test) = DataSplit.TrainTest(
            data, r => r.Y_TaxValue > 0f ? 1 : 0, testFraction: 0.20, seed: 42);

        var medians   = MedianImputer.Fit(train);
        var trainRows = MedianImputer.Apply(train, medians, r => r.Y_TaxValue);
        var testRows  = MedianImputer.Apply(test,  medians, r => r.Y_TaxValue);

        var trainView = ml.Data.LoadFromEnumerable(trainRows);
        var testView  = ml.Data.LoadFromEnumerable(testRows);

        var results = new List<object>
        {
            RunOne(ml, "linreg_sdca", trainView, testView, testRows, normalize: true,
                p => ml.Regression.Trainers.Sdca(
                    labelColumnName: "FloatLabel", featureColumnName: p)),
            RunOne(ml, "gbt_fasttree", trainView, testView, testRows, normalize: false,
                p => ml.Regression.Trainers.FastTree(
                    labelColumnName: "FloatLabel", featureColumnName: p,
                    numberOfTrees: 100, numberOfLeaves: 20, learningRate: 0.1)),
        };

        Artifacts.WriteJson(new
        {
            Target          = FeatureLists.TargetTaxValue,
            Features        = FeatureLists.NumericFeaturesTaxValueRegression,
            RowsTrain       = trainRows.Count,
            RowsTest        = testRows.Count,
            PositiveRate    = data.Count(r => r.Y_TaxValue > 0f) / (double)data.Count,
            Models          = results,
        }, Path.Combine(artifactsDir, "tax_value_regression_metrics.json"));
    }

    private static object RunOne(
        MLContext ml,
        string name,
        IDataView trainView,
        IDataView testView,
        IReadOnlyList<MLReadyRow> testRows,
        bool normalize,
        Func<string, IEstimator<ITransformer>> makeTrainer)
    {
        var features = FeatureLists.NumericFeaturesTaxValueRegression;

        IEstimator<ITransformer> pipeline =
            ml.Transforms.Concatenate("Features", features);
        if (normalize)
            pipeline = pipeline.Append(ml.Transforms.NormalizeMeanVariance("Features"));
        var full = pipeline.Append(makeTrainer("Features"));

        Console.WriteLine($"[TaxValueRegression] training {name} …");
        var model  = full.Fit(trainView);
        var scored = model.Transform(testView);
        var m      = ml.Regression.Evaluate(scored, labelColumnName: "FloatLabel");

        // Positive-subset RMSE: where the kinked structure actually lives.
        var preds = ml.Data
            .CreateEnumerable<RegressionScoredRow>(scored, reuseRowObject: false)
            .ToList();
        double posSse = 0; int posN = 0;
        for (int i = 0; i < preds.Count; i++)
        {
            if (testRows[i].FloatLabel <= 0f) continue;
            double e = preds[i].Score - testRows[i].FloatLabel;
            posSse += e * e; posN++;
        }
        double posRmse = posN > 0 ? Math.Sqrt(posSse / posN) : double.NaN;

        Console.WriteLine($"[TaxValueRegression] {name}: " +
                          $"RMSE={m.RootMeanSquaredError:F2} MAE={m.MeanAbsoluteError:F2} " +
                          $"R²={m.RSquared:F4} posRMSE={posRmse:F2} (n={posN})");

        return new
        {
            ModelName    = name,
            TestRmse     = m.RootMeanSquaredError,
            TestMae      = m.MeanAbsoluteError,
            TestR2       = m.RSquared,
            TestPosRmse  = posRmse,
            TestPosCount = posN,
        };
    }

    private class RegressionScoredRow
    {
        public float Score { get; set; }
    }
}
