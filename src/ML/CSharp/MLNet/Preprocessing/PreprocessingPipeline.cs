using DirectIndexing.ML.MLNet.Schema;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace DirectIndexing.ML.MLNet.Preprocessing;

/// <summary>
/// Builds the typed <see cref="IEstimator{TTransformer}"/> chain that takes
/// <see cref="MLReadyRow"/> and produces a <c>Features</c> vector ready for the
/// trainer.
///
/// Chain (top-to-bottom):
///   Sector (NaN/empty → "Unknown") via CustomMapping
///   → OneHotEncoding(SectorClean)
///   → NormalizeMeanVariance(15 numeric features)
///   → Concatenate("Features", numericFeatures + SectorOneHot)
///
/// Each stage has a named input and named output column — there are no
/// positional indices. Fitting this chain on a training fold produces an
/// <see cref="ITransformer"/> that the trainer's outer chain prepends to its
/// own algorithm-specific stage.
/// </summary>
public static class PreprocessingPipeline
{
    [CustomMappingFactoryAttribute(nameof(SectorCleanFactory))]
    private class SectorCleanFactory : CustomMappingFactory<SectorIn, SectorOut>
    {
        public override Action<SectorIn, SectorOut> GetMapping() =>
            (input, output) => output.SectorClean =
                string.IsNullOrWhiteSpace(input.Sector) ? "Unknown" : input.Sector;
    }

    public class SectorIn  { public string Sector { get; set; } = ""; }
    public class SectorOut { public string SectorClean { get; set; } = ""; }

    public static IEstimator<ITransformer> Build(MLContext ml)
    {
        var numeric = FeatureLists.NumericFeatures;

        // Sector cleaning is the one place we genuinely need a CustomMapping
        // — everything else is built-in. The contract name lets a fitted model
        // be re-loaded from disk (ML.NET requires registering the factory).
        ml.ComponentCatalog.RegisterAssembly(typeof(SectorCleanFactory).Assembly);

        return ml.Transforms
            .CustomMapping(new SectorCleanFactory().GetMapping(), contractName: nameof(SectorCleanFactory))
            .Append(ml.Transforms.Categorical.OneHotEncoding(
                outputColumnName: FeatureLists.SectorOneHot,
                inputColumnName:  FeatureLists.SectorClean))
            .Append(ml.Transforms.NormalizeMeanVariance(numeric.Select(c =>
                new InputOutputColumnPair(c, c)).ToArray()))
            .Append(ml.Transforms.Concatenate(
                FeatureLists.FeaturesCol,
                numeric.Concat(new[] { FeatureLists.SectorOneHot }).ToArray()));
    }
}
