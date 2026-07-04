using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.ML.MLNet.Splits;

/// <summary>
/// Pure functions: chronological train/test split and purged contiguous k-fold
/// on the Timestep axis (v0.26 validation hardening).
///
/// Why this exists: the soft labels are forward-looking — Y_Soft_BT(x_t) is a
/// deterministic function of prices over (t, t+30] — so rows within 30 trading
/// days of each other share future context. A stratified RANDOM split scatters
/// such neighbours across train and test, letting the model be scored on
/// futures it partially saw in training. The fix (López de Prado's purged
/// splits, specialised to this panel) is:
///
///   1. split chronologically on Timestep, and
///   2. PURGE every training row whose forward label window could overlap the
///      evaluation period: drop train rows with Timestep inside an embargo gap
///      of width E ≥ the label horizon before each evaluation block (and after
///      it, for interior CV blocks).
///
/// With E = 30, a training row at t ≤ T* − 31 has label window ⊆ (t, T* − 1],
/// strictly before an evaluation block starting at T*: zero overlap.
///
/// No shuffling, no stratification: class balance per period is whatever the
/// regime produced — that is the point of the exercise. Class weights are
/// computed per training fold downstream (ClassWeights), so imbalance is
/// handled where it always was.
/// </summary>
public static class TemporalSplit
{
    /// <summary>
    /// Chronological (train, test): the earliest ≈(1−testFraction) of rows by
    /// Timestep train; the latest ≈testFraction test; train rows within
    /// <paramref name="embargoDays"/> of the boundary are purged.
    /// </summary>
    public static (List<LotStateVector> Train, List<LotStateVector> Test) TrainTest(
        IReadOnlyList<LotStateVector> data,
        double testFraction = 0.20,
        int embargoDays = 30)
    {
        if (data.Count == 0) return (new List<LotStateVector>(), new List<LotStateVector>());

        int boundary = BoundaryTimestep(data, 1.0 - testFraction);

        var train = new List<LotStateVector>();
        var test  = new List<LotStateVector>(capacity: (int)(data.Count * testFraction) + 1);
        int purged = 0;

        foreach (var r in data)
        {
            if (r.Timestep >= boundary)              test.Add(r);
            else if (r.Timestep > boundary - embargoDays) purged++;   // embargo gap
            else                                     train.Add(r);
        }

        Console.WriteLine($"[TemporalSplit] boundary t={boundary}  " +
                          $"train={train.Count:N0}  test={test.Count:N0}  " +
                          $"purged={purged:N0} (embargo {embargoDays}d)");
        return (train, test);
    }

    /// <summary>
    /// Purged contiguous k-fold: the Timestep range is cut into k contiguous
    /// blocks (equal row mass, whole-Timestep boundaries); fold i uses block i
    /// as validation and everything OUTSIDE the block ± embargo as training —
    /// train data sits on both sides of interior blocks, purged on both edges
    /// (the standard purged k-fold of the finance-ML literature; it uses more
    /// data per fold than forward-chaining while giving the same guarantee).
    /// </summary>
    public static IEnumerable<(List<LotStateVector> Train, List<LotStateVector> Val)> PurgedFolds(
        IReadOnlyList<LotStateVector> data,
        int k = 5,
        int embargoDays = 30)
    {
        if (k < 2) throw new ArgumentException("k must be >= 2", nameof(k));

        // Block edges as Timestep values at row-mass quantiles.
        var edges = new int[k + 1];
        for (int i = 0; i <= k; i++)
            edges[i] = i == 0 ? int.MinValue
                     : i == k ? int.MaxValue
                     : BoundaryTimestep(data, (double)i / k);

        for (int fold = 0; fold < k; fold++)
        {
            int lo = edges[fold], hi = edges[fold + 1];   // val block = [lo, hi)
            var train = new List<LotStateVector>();
            var val   = new List<LotStateVector>();

            foreach (var r in data)
            {
                int t = r.Timestep;
                if (t >= lo && t < hi)
                    val.Add(r);
                else if (WithinEmbargo(t, lo, hi, embargoDays))
                    continue;                              // purged
                else
                    train.Add(r);
            }
            yield return (train, val);
        }
    }

    // Timestep below which ≈ `massFraction` of rows fall (never splits a day).
    private static int BoundaryTimestep(IReadOnlyList<LotStateVector> data, double massFraction)
    {
        var counts = new SortedDictionary<int, int>();
        foreach (var r in data)
            counts[r.Timestep] = counts.GetValueOrDefault(r.Timestep) + 1;

        long target = (long)(data.Count * massFraction);
        long seen = 0;
        foreach (var (t, n) in counts)
        {
            seen += n;
            if (seen >= target) return t + 1;   // boundary is exclusive of day t
        }
        return int.MaxValue;
    }

    // Train row at t is embargoed if its label window could touch [lo, hi):
    // within embargo BEFORE the block (window reaches forward into it) or
    // within embargo AFTER it (the block's own label windows reach forward
    // into the row's period — symmetric purge for interior CV blocks).
    private static bool WithinEmbargo(int t, int lo, int hi, int embargoDays)
    {
        bool before = lo != int.MinValue && t < lo && t > lo - embargoDays;
        bool after  = hi != int.MaxValue && t >= hi && t < hi + embargoDays;
        return before || after;
    }
}
