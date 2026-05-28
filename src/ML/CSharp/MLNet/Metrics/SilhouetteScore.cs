namespace DirectIndexing.ML.MLNet.Metrics;

/// <summary>
/// Mean silhouette coefficient on a small matrix (designed for the 503-row
/// per-symbol aggregate used by KMeansPipeline — not the 122K full dataset).
///
/// silhouette(i) = (b_i − a_i) / max(a_i, b_i)
///   a_i = mean distance from i to other points in its own cluster
///   b_i = min over other clusters c of (mean distance from i to points in c)
/// The mean across i is in [-1, 1]; higher is better.
/// </summary>
public static class SilhouetteScore
{
    public static double Compute(double[][] points, int[] clusterIds)
    {
        if (points.Length != clusterIds.Length)
            throw new ArgumentException("points and clusterIds length mismatch");

        int n = points.Length;
        int dim = points[0].Length;

        // Group indices by cluster for fast mean-distance computation.
        var byCluster = new Dictionary<int, List<int>>();
        foreach (var c in clusterIds.Distinct()) byCluster[c] = new List<int>();
        for (int i = 0; i < n; i++) byCluster[clusterIds[i]].Add(i);

        if (byCluster.Count < 2) return 0.0;  // degenerate: 1 cluster.

        double sum = 0.0;
        int counted = 0;

        for (int i = 0; i < n; i++)
        {
            int ci = clusterIds[i];
            var own = byCluster[ci];
            if (own.Count <= 1) continue;  // singleton — silhouette undefined; skip.

            double a = MeanDistanceToCluster(points, i, own, excludeSelf: true);
            double b = double.PositiveInfinity;
            foreach (var (cj, members) in byCluster)
            {
                if (cj == ci) continue;
                double m = MeanDistanceToCluster(points, i, members, excludeSelf: false);
                if (m < b) b = m;
            }

            double s = (b - a) / Math.Max(a, b);
            if (!double.IsNaN(s) && !double.IsInfinity(s))
            {
                sum += s; counted++;
            }
        }

        return counted == 0 ? 0.0 : sum / counted;
    }

    private static double MeanDistanceToCluster(double[][] points, int i, List<int> members, bool excludeSelf)
    {
        double total = 0;
        int count = 0;
        foreach (var j in members)
        {
            if (excludeSelf && j == i) continue;
            total += EuclideanDistance(points[i], points[j]);
            count++;
        }
        return count == 0 ? 0.0 : total / count;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sumSq = 0;
        for (int k = 0; k < a.Length; k++)
        {
            double d = a[k] - b[k];
            sumSq += d * d;
        }
        return Math.Sqrt(sumSq);
    }
}
