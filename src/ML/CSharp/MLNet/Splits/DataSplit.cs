using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.ML.MLNet.Splits;

/// <summary>
/// The single entry point trainers use for partitioning (v0.26). Dispatches on
/// <see cref="SplitPolicy"/>: legacy stratified-random by default, chronological
/// purged splits under <c>--split=temporal</c>. Signatures mirror the legacy
/// utilities exactly, so the 12 trainer call sites changed names only.
///
/// In temporal mode the label selector is unused (no stratification — regime
/// prevalence is the object of study) and the seed is irrelevant (deterministic
/// order); both parameters remain so call sites stay uniform.
/// </summary>
public static class DataSplit
{
    public static (List<LotStateVector> Train, List<LotStateVector> Test) TrainTest(
        IReadOnlyList<LotStateVector> data,
        Func<LotStateVector, int> labelSelector,
        double testFraction = 0.20,
        int seed = 42)
    {
        double frac = SplitPolicy.TestFractionOverride ?? testFraction;
        return SplitPolicy.Mode == SplitMode.TemporalPurged
            ? TemporalSplit.TrainTest(data, frac, SplitPolicy.EmbargoDays)
            : StratifiedSplit.Split(data, labelSelector, frac, seed);
    }

    public static IEnumerable<(List<LotStateVector> Train, List<LotStateVector> Val)> Folds(
        IReadOnlyList<LotStateVector> data,
        Func<LotStateVector, int> labelSelector,
        int k = 5,
        int seed = 42)
    {
        return SplitPolicy.Mode == SplitMode.TemporalPurged
            ? TemporalSplit.PurgedFolds(data, k, SplitPolicy.EmbargoDays)
            : StratifiedKFold.Folds(data, labelSelector, k, seed);
    }
}
