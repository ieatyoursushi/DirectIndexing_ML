using DirectIndexing.Core.Portfolio;
using DirectIndexing.ML.MLNet.Preprocessing;
using DirectIndexing.ML.MLNet.Schema;
using DirectIndexing.ML.MLNet.Splits;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace DirectIndexing.ML.MLNet.Models;

/// <summary>
/// PCA on the standardised numeric-feature matrix.
///
/// <para><b>Invariant:</b> the EstimatorChain (standardise → PCA) is
/// <c>.Fit(...)</c> on the <i>training fold only</i>; the MathNet
/// <see cref="Svd"/> call that recovers loadings + explained-variance ratios
/// runs on the same training-fold matrix. The fitted transformer is then
/// applied to the test fold via <c>.Transform(...)</c>. The Python version
/// calls <c>PCA().fit(df[NUMERIC_FEATURES].dropna().values)</c> on the full
/// dataset — a subtle leak. Here the EstimatorChain's shape makes the
/// training-fold restriction the natural path.</para>
/// </summary>
public static class PcaPipeline
{
    public record PcaOutput(
        double[] ExplainedVariance,
        double[] CumulativeVariance,
        int NKept,
        double[][] Loadings,
        ITransformer Model);

    public static PcaOutput Run(
        MLContext ml,
        IReadOnlyList<LotStateVector> data,
        double cumulativeVarianceThreshold = 0.95)
    {
        // Drop rows where ANY numeric feature is NaN — PCA can't tolerate them
        // and median-impute would distort the principal axes here.
        var clean = data.Where(NoNumericNaN).ToList();

        var (train, _) = StratifiedSplit.Split(
            clean, r => r.Y_Oracle, testFraction: 0.20, seed: 42);

        var medians = MedianImputer.Fit(train);
        var trainReady = MedianImputer.Apply(train, medians, r => r.Y_Oracle == 1);
        var trainView = ml.Data.LoadFromEnumerable(trainReady);

        // Build the chain: standardise the 15 numeric columns individually,
        // concat into "RawFeatures", then project to 15 principal components.
        var numeric = FeatureLists.NumericFeatures;
        var chain = ml.Transforms
            .NormalizeMeanVariance(numeric.Select(c => new InputOutputColumnPair(c, c)).ToArray())
            .Append(ml.Transforms.Concatenate("RawFeatures", numeric))
            .Append(ml.Transforms.ProjectToPrincipalComponents(
                outputColumnName: "PcaFeatures",
                inputColumnName:  "RawFeatures",
                rank: numeric.Length));

        var model = chain.Fit(trainView);

        // Recover loadings + explained-variance ratios via MathNet SVD on the
        // standardised training matrix. ML.NET's PCA transform doesn't expose
        // components/eigenvalues, so we compute them from the same matrix it
        // fit on — preserving the training-fold-only invariant.
        var stdMatrix = BuildStandardisedMatrix(trainReady);
        var (explainedVar, cumulative, loadings) = ComputeSvd(stdMatrix);

        int nKept = Array.FindIndex(cumulative, v => v >= cumulativeVarianceThreshold) + 1;
        if (nKept <= 0) nKept = numeric.Length;

        return new PcaOutput(
            ExplainedVariance: explainedVar,
            CumulativeVariance: cumulative,
            NKept: nKept,
            Loadings: loadings,
            Model: model);
    }

    private static bool NoNumericNaN(LotStateVector r) =>
        !(float.IsNaN(r.L) || float.IsNaN(r.B) || float.IsNaN(r.W) ||
          float.IsNaN(r.G_YTD) || float.IsNaN(r.Sigma_TE) ||
          float.IsNaN(r.R_t) || float.IsNaN(r.SigmaRange) ||
          float.IsNaN(r.DeltaMA50) || float.IsNaN(r.DeltaMA200) ||
          float.IsNaN(r.TaxAlpha));

    private static double[,] BuildStandardisedMatrix(IReadOnlyList<MLReadyRow> rows)
    {
        var numeric = FeatureLists.NumericFeatures;
        int n = rows.Count;
        int p = numeric.Length;
        var raw = new double[n, p];

        for (int i = 0; i < n; i++)
        {
            raw[i, 0]  = rows[i].L;
            raw[i, 1]  = rows[i].H;
            raw[i, 2]  = rows[i].S;
            raw[i, 3]  = rows[i].B;
            raw[i, 4]  = rows[i].W;
            raw[i, 5]  = rows[i].K;
            raw[i, 6]  = rows[i].G_YTD;
            raw[i, 7]  = rows[i].Sigma_TE;
            raw[i, 8]  = rows[i].WashClock;
            raw[i, 9]  = rows[i].R_t;
            raw[i, 10] = rows[i].SigmaRange;
            raw[i, 11] = rows[i].DeltaMA50;
            raw[i, 12] = rows[i].DeltaMA200;
            raw[i, 13] = rows[i].TaxAlpha;
            raw[i, 14] = rows[i].DaysToYE;
        }

        // Standardise per column (mean 0, variance 1). Same procedure ML.NET
        // applies in NormalizeMeanVariance — duplicated here so MathNet sees
        // the same matrix without round-tripping through IDataView.
        for (int j = 0; j < p; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += raw[i, j];
            mean /= n;

            double var = 0;
            for (int i = 0; i < n; i++)
            {
                double d = raw[i, j] - mean;
                var += d * d;
            }
            var /= n;
            double sd = Math.Sqrt(var);

            for (int i = 0; i < n; i++)
                raw[i, j] = sd > 1e-12 ? (raw[i, j] - mean) / sd : 0.0;
        }

        return raw;
    }

    private static (double[] ExplainedVar, double[] Cumulative, double[][] Loadings)
        ComputeSvd(double[,] standardisedMatrix)
    {
        int n = standardisedMatrix.GetLength(0);
        int p = standardisedMatrix.GetLength(1);

        // Build C = X^T X / (n-1), a tiny p × p matrix (15 × 15 here).
        // Eigendecomposing C gives:
        //   eigenvalues = explained variances (= σ_i² / (n-1) for the SVD of X),
        //   eigenvectors = principal axes (= V, the right singular vectors of X).
        // This avoids materialising the N × N U from a full SVD on X — which
        // would attempt to allocate ~10^10 doubles at N=122K.
        var cov = Matrix<double>.Build.Dense(p, p);
        for (int j = 0; j < p; j++)
        {
            for (int k = j; k < p; k++)
            {
                double s = 0;
                for (int i = 0; i < n; i++) s += standardisedMatrix[i, j] * standardisedMatrix[i, k];
                double v = s / (n - 1);
                cov[j, k] = v;
                cov[k, j] = v;
            }
        }

        var evd = cov.Evd(MathNet.Numerics.LinearAlgebra.Symmetricity.Symmetric);

        // EVD returns eigenvalues in ascending order; we want descending.
        var raw = evd.EigenValues.Select(c => c.Real).ToArray();
        var V   = evd.EigenVectors;  // columns are eigenvectors, p × p

        var indices = Enumerable.Range(0, p).OrderByDescending(i => raw[i]).ToArray();
        var eigen = indices.Select(i => raw[i]).ToArray();

        double total = eigen.Sum();
        var explained = eigen.Select(e => e / total).ToArray();

        var cumulative = new double[explained.Length];
        double running = 0;
        for (int i = 0; i < explained.Length; i++)
        {
            running += explained[i];
            cumulative[i] = running;
        }

        // Loadings[i] = column i of V (reordered) — the i-th principal axis in
        // feature space. Shape: [p components][p features].
        var loadings = new double[p][];
        for (int i = 0; i < p; i++)
        {
            loadings[i] = new double[p];
            int srcCol = indices[i];
            for (int j = 0; j < p; j++) loadings[i][j] = V[j, srcCol];
        }

        return (explained, cumulative, loadings);
    }
}
