using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.ML.MLNet.Splits;

/// <summary>
/// Pure function: split <c>IReadOnlyList&lt;LotStateVector&gt;</c> into (train, test)
/// stratified by a binary label selector. Each class is independently shuffled
/// with a seeded RNG and its first <paramref name="testFraction"/> goes to test —
/// so the test set retains the population's class proportion exactly (rounded).
///
/// The partition logic is visible here, not hidden behind a constructor argument
/// — which is the point of writing this rather than calling a black-box utility.
/// </summary>
public static class StratifiedSplit
{
    public static (List<LotStateVector> Train, List<LotStateVector> Test) Split(
        IReadOnlyList<LotStateVector> data,
        Func<LotStateVector, int> labelSelector,
        double testFraction = 0.20,
        int seed = 42)
    {
        var byClass = new Dictionary<int, List<LotStateVector>>();
        foreach (var r in data)
        {
            int y = labelSelector(r);
            if (!byClass.TryGetValue(y, out var bucket))
                byClass[y] = bucket = new List<LotStateVector>();
            bucket.Add(r);
        }

        var rng = new Random(seed);
        var train = new List<LotStateVector>(capacity: data.Count);
        var test  = new List<LotStateVector>(capacity: (int)(data.Count * testFraction) + 1);

        foreach (var (_, bucket) in byClass)
        {
            Shuffle(bucket, rng);
            int nTest = (int)Math.Round(bucket.Count * testFraction);
            for (int i = 0; i < bucket.Count; i++)
                (i < nTest ? test : train).Add(bucket[i]);
        }

        // Re-shuffle the combined lists so order is not class-grouped.
        Shuffle(train, rng);
        Shuffle(test, rng);
        return (train, test);
    }

    private static void Shuffle<T>(List<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }
}
