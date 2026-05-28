using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.ML.MLNet.Splits;

/// <summary>
/// Pure function: partition <c>IReadOnlyList&lt;LotStateVector&gt;</c> into k folds
/// stratified by a binary label selector. Each fold inherits the population's
/// class proportion (within ±1 row per class).
///
/// Strategy: shuffle each class once with a seeded RNG, walk the shuffled list
/// assigning row i to fold (i mod k). This is the round-robin variant of
/// stratification — same final distribution as sklearn's, easier to read.
/// Yields k (train, val) pairs where val = fold i and train = everything else.
/// </summary>
public static class StratifiedKFold
{
    public static IEnumerable<(List<LotStateVector> Train, List<LotStateVector> Val)> Folds(
        IReadOnlyList<LotStateVector> data,
        Func<LotStateVector, int> labelSelector,
        int k = 5,
        int seed = 42)
    {
        if (k < 2) throw new ArgumentException("k must be >= 2", nameof(k));

        var foldAssignment = new int[data.Count];
        var classBuckets   = new Dictionary<int, List<int>>(); // class -> row indices

        for (int i = 0; i < data.Count; i++)
        {
            int y = labelSelector(data[i]);
            if (!classBuckets.TryGetValue(y, out var bucket))
                classBuckets[y] = bucket = new List<int>();
            bucket.Add(i);
        }

        var rng = new Random(seed);
        foreach (var (_, bucket) in classBuckets)
        {
            Shuffle(bucket, rng);
            for (int j = 0; j < bucket.Count; j++)
                foldAssignment[bucket[j]] = j % k;
        }

        for (int fold = 0; fold < k; fold++)
        {
            var train = new List<LotStateVector>(capacity: data.Count - data.Count / k);
            var val   = new List<LotStateVector>(capacity: data.Count / k + 1);
            for (int i = 0; i < data.Count; i++)
                (foldAssignment[i] == fold ? val : train).Add(data[i]);
            yield return (train, val);
        }
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
