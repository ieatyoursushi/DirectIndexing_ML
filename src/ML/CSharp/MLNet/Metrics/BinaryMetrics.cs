using Microsoft.ML;
using Microsoft.ML.Data;

namespace DirectIndexing.ML.MLNet.Metrics;

/// <summary>
/// Per-row scored output of a binary classifier. Mirrors ML.NET's
/// <c>CalibratedBinaryClassificationMetrics</c> input rows.
/// </summary>
public class ScoredRow
{
    public bool  Label       { get; set; }
    public bool  PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score       { get; set; }
}

public record CurvePoint(double Threshold, double X, double Y);

public class BinaryMetricsResult
{
    public double RocAuc            { get; init; }
    public double PrAuc             { get; init; }
    public double F1At05            { get; init; }
    public double F1AtBest          { get; init; }
    public double BestThreshold     { get; init; }
    public int    Tp05, Fp05, Tn05, Fn05;
    public int    TpBest, FpBest, TnBest, FnBest;
    public CurvePoint[] RocCurve { get; init; } = Array.Empty<CurvePoint>();
    public CurvePoint[] PrCurve  { get; init; } = Array.Empty<CurvePoint>();
}

/// <summary>
/// Compute the full evaluation bundle:
///   ROC-AUC, PR-AUC (average precision), F1 at threshold 0.5,
///   F1-optimal threshold + confusion matrices at both 0.5 and best-F1,
///   ROC and PR curve point arrays (for the Python renderer).
///
/// Sweeps the (label, probability) array directly rather than calling
/// ML.NET's <c>BinaryClassification.Evaluate</c> because we want PR-AUC
/// (not exposed for non-calibrated trainers) and the curve points.
/// </summary>
public static class BinaryMetrics
{
    public static BinaryMetricsResult Compute(MLContext ml, IDataView scored)
    {
        // FastForest (and other uncalibrated trainers) produce "Score" but not
        // "Probability". Use Score as the probability proxy when Probability is absent.
        bool hasProb = scored.Schema.GetColumnOrNull("Probability") is not null;
        var rows = ml.Data.CreateEnumerable<ScoredRow>(
            scored, reuseRowObject: false, ignoreMissingColumns: true).ToArray();
        if (!hasProb)
        {
            for (int i = 0; i < rows.Length; i++)
                rows[i].Probability = rows[i].Score;
        }
        return Compute(rows);
    }

    public static BinaryMetricsResult Compute(IReadOnlyList<ScoredRow> rows)
    {
        // Sort descending by probability — once — and reuse for both curves.
        var sorted = rows.OrderByDescending(r => r.Probability).ToArray();
        int nPos = rows.Count(r => r.Label);
        int nNeg = rows.Count - nPos;
        if (nPos == 0 || nNeg == 0)
            throw new InvalidOperationException("Need both classes present to compute ROC/PR.");

        var roc = new List<CurvePoint>(sorted.Length + 2);
        var pr  = new List<CurvePoint>(sorted.Length + 2);

        roc.Add(new CurvePoint(double.PositiveInfinity, 0.0, 0.0));

        int tp = 0, fp = 0;
        double rocAuc = 0.0, prAuc = 0.0;
        double prevFpr = 0.0, prevRecall = 0.0, prevPrecision = 1.0;
        double bestF1 = -1, bestThresh = 0.5;

        for (int i = 0; i < sorted.Length; i++)
        {
            if (sorted[i].Label) tp++; else fp++;
            // Aggregate ties: emit a point only at threshold boundaries.
            if (i == sorted.Length - 1 || sorted[i].Probability != sorted[i + 1].Probability)
            {
                double tpr   = (double)tp / nPos;
                double fpr   = (double)fp / nNeg;
                double prec  = tp + fp == 0 ? 1.0 : (double)tp / (tp + fp);
                double recl  = tpr;
                double thresh = sorted[i].Probability;

                rocAuc += 0.5 * (fpr - prevFpr) * (tpr + prevRecall);
                // PR-AUC via "average precision" — area under step-function PR curve.
                prAuc += (recl - prevRecall) * prec;

                double f1 = (prec + recl) == 0 ? 0 : 2 * prec * recl / (prec + recl);
                if (f1 > bestF1) { bestF1 = f1; bestThresh = thresh; }

                roc.Add(new CurvePoint(thresh, fpr, tpr));
                pr.Add (new CurvePoint(thresh, recl, prec));

                prevFpr = fpr; prevRecall = recl; prevPrecision = prec;
            }
        }

        var (tp05, fp05, tn05, fn05) = Confusion(rows, 0.5);
        var (tpB,  fpB,  tnB,  fnB ) = Confusion(rows, bestThresh);
        double f1At05 = F1(tp05, fp05, fn05);

        return new BinaryMetricsResult
        {
            RocAuc        = rocAuc,
            PrAuc         = prAuc,
            F1At05        = f1At05,
            F1AtBest      = bestF1,
            BestThreshold = bestThresh,
            Tp05 = tp05, Fp05 = fp05, Tn05 = tn05, Fn05 = fn05,
            TpBest = tpB, FpBest = fpB, TnBest = tnB, FnBest = fnB,
            RocCurve = Downsample(roc),
            PrCurve  = Downsample(pr),
        };
    }

    // Curves carry one point per distinct score threshold, which is ~O(n) for a
    // model with continuous scores (e.g. linear regression on 1.8M rows → ~360k
    // points → an 80MB+ JSON that stalls serialization). The AUC/F1 scalars above
    // are computed over the FULL sweep and are unaffected by how many points we
    // *store*; downsampling the stored polyline to a fixed cap keeps the curve
    // visually identical (the Python renderer thins to ~500 anyway) while bounding
    // artifact size. First and last points are always retained.
    private const int MaxCurvePoints = 2000;

    private static CurvePoint[] Downsample(List<CurvePoint> pts)
    {
        if (pts.Count <= MaxCurvePoints) return pts.ToArray();
        var outp = new CurvePoint[MaxCurvePoints];
        double stride = (double)(pts.Count - 1) / (MaxCurvePoints - 1);
        for (int i = 0; i < MaxCurvePoints; i++)
            outp[i] = pts[(int)Math.Round(i * stride)];
        outp[MaxCurvePoints - 1] = pts[^1];   // guarantee the endpoint
        return outp;
    }

    private static (int tp, int fp, int tn, int fn) Confusion(
        IReadOnlyList<ScoredRow> rows, double threshold)
    {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        foreach (var r in rows)
        {
            bool pred = r.Probability >= threshold;
            if (r.Label && pred) tp++;
            else if (r.Label && !pred) fn++;
            else if (!r.Label && pred) fp++;
            else tn++;
        }
        return (tp, fp, tn, fn);
    }

    private static double F1(int tp, int fp, int fn)
    {
        double prec = (tp + fp == 0) ? 0 : (double)tp / (tp + fp);
        double recl = (tp + fn == 0) ? 0 : (double)tp / (tp + fn);
        return (prec + recl == 0) ? 0 : 2 * prec * recl / (prec + recl);
    }
}
