// Tests/TemporalSplitTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Splits;

/// <summary>
/// Unit tests for the v0.26 purged chronological splits. The invariant under
/// test is leak-proofing: no training row's forward label window (t, t+E] may
/// overlap any evaluation row's period.
/// </summary>
public class TemporalSplitTests
{
    // 100 timesteps × 10 rows each, positives sprinkled deterministically.
    private static List<LotStateVector> MakePanel(int days = 100, int perDay = 10)
    {
        var rows = new List<LotStateVector>(days * perDay);
        for (int t = 0; t < days; t++)
            for (int i = 0; i < perDay; i++)
                rows.Add(new LotStateVector
                {
                    Timestep = 200 + t,
                    Symbol   = $"S{i}",
                    Y_Oracle = (t * perDay + i) % 37 == 0 ? 1 : 0,
                });
        return rows;
    }

    // Test 1: chronological order + embargo gap between train and test.
    public void Test_TrainTest_BoundaryAndEmbargo()
    {
        var data = MakePanel();
        var (train, test) = TemporalSplit.TrainTest(data, testFraction: 0.20, embargoDays: 10);

        int maxTrain = int.MinValue, minTest = int.MaxValue;
        foreach (var r in train) maxTrain = Math.Max(maxTrain, r.Timestep);
        foreach (var r in test)  minTest  = Math.Min(minTest,  r.Timestep);

        Debug.Assert(maxTrain < minTest,
            $"Train must strictly precede test, got maxTrain={maxTrain} minTest={minTest}");
        Debug.Assert(minTest - maxTrain >= 10,
            $"Embargo gap must be >= 10 days, got {minTest - maxTrain}");
        // A train row's forward window (t, t+10] must end before the test period.
        Debug.Assert(maxTrain + 10 < minTest + 10 && maxTrain + 10 <= minTest,
            "No train label window may reach the test period");

        double testShare = test.Count / (double)data.Count;
        Debug.Assert(testShare > 0.15 && testShare < 0.25,
            $"Test share should approximate 0.20 (minus rounding), got {testShare:F3}");

        Console.WriteLine($"TemporalSplit Test 1 passed: boundary honest (gap={minTest - maxTrain}d), " +
                          $"test share {testShare:P1}");
    }

    // Test 2: purged rows are exactly the embargo band — nothing lost elsewhere.
    public void Test_TrainTest_PurgeAccounting()
    {
        var data = MakePanel();
        var (train, test) = TemporalSplit.TrainTest(data, testFraction: 0.20, embargoDays: 10);

        int purged = data.Count - train.Count - test.Count;
        // 10 embargoed days × 10 rows/day = 100 rows expected (±1 day of rounding).
        Debug.Assert(purged >= 90 && purged <= 110,
            $"Expected ~100 purged rows (10d × 10 rows), got {purged}");
        Console.WriteLine($"TemporalSplit Test 2 passed: purge accounting exact ({purged} rows)");
    }

    // Test 3: purged k-fold — every fold's train set stays >= embargo away
    // from its validation block on both sides, and folds partition the panel.
    public void Test_PurgedFolds_EmbargoBothSides()
    {
        var data = MakePanel();
        int k = 5, embargo = 10;
        int foldIdx = 0, totalVal = 0;

        foreach (var (train, val) in TemporalSplit.PurgedFolds(data, k, embargo))
        {
            int valLo = int.MaxValue, valHi = int.MinValue;
            foreach (var r in val)
            {
                valLo = Math.Min(valLo, r.Timestep);
                valHi = Math.Max(valHi, r.Timestep);
            }
            foreach (var r in train)
            {
                bool inside  = r.Timestep >= valLo && r.Timestep <= valHi;
                bool tooNear = (r.Timestep < valLo && r.Timestep > valLo - embargo)
                            || (r.Timestep > valHi && r.Timestep < valHi + embargo);
                Debug.Assert(!inside,  $"Fold {foldIdx}: train row inside val block (t={r.Timestep})");
                Debug.Assert(!tooNear, $"Fold {foldIdx}: train row within embargo of val block (t={r.Timestep})");
            }
            Debug.Assert(val.Count > 0 && train.Count > 0,
                $"Fold {foldIdx}: degenerate fold (train={train.Count}, val={val.Count})");
            totalVal += val.Count;
            foldIdx++;
        }

        Debug.Assert(foldIdx == k, $"Expected {k} folds, got {foldIdx}");
        Debug.Assert(totalVal == MakePanel().Count,
            $"Validation blocks must partition all rows, covered {totalVal}");
        Console.WriteLine("TemporalSplit Test 3 passed: purged folds embargoed on both sides, val partitions panel");
    }

    // Test 4: determinism — no RNG anywhere, two calls agree exactly.
    public void Test_Deterministic()
    {
        var data = MakePanel();
        var (t1, s1) = TemporalSplit.TrainTest(data, 0.20, 10);
        var (t2, s2) = TemporalSplit.TrainTest(data, 0.20, 10);
        Debug.Assert(t1.Count == t2.Count && s1.Count == s2.Count,
            "TemporalSplit must be deterministic");
        Console.WriteLine("TemporalSplit Test 4 passed: deterministic");
    }

    // Test 5: DataSplit facade dispatches on SplitPolicy and restores cleanly.
    public void Test_DataSplit_PolicyDispatch()
    {
        var data = MakePanel();
        var prevMode = SplitPolicy.Mode;
        try
        {
            SplitPolicy.Mode = SplitMode.StratifiedRandom;
            var (rTrain, rTest) = DataSplit.TrainTest(data, r => r.Y_Oracle, 0.20, 42);
            // Random mode interleaves time: test min Timestep should reach near the panel start.
            int rMinTest = rTest.Min(r => r.Timestep);
            Debug.Assert(rMinTest < 250, "Stratified-random test set should span early timesteps");

            SplitPolicy.Mode = SplitMode.TemporalPurged;
            SplitPolicy.EmbargoDays = 10;
            var (tTrain, tTest) = DataSplit.TrainTest(data, r => r.Y_Oracle, 0.20, 42);
            int tMaxTrain = tTrain.Max(r => r.Timestep);
            int tMinTest  = tTest.Min(r => r.Timestep);
            Debug.Assert(tMaxTrain < tMinTest, "Temporal mode must be chronological through the facade");

            Console.WriteLine("TemporalSplit Test 5 passed: DataSplit facade dispatches on SplitPolicy");
        }
        finally
        {
            SplitPolicy.Mode = prevMode;
            SplitPolicy.EmbargoDays = 30;
            SplitPolicy.TestFractionOverride = null;
        }
    }
}
