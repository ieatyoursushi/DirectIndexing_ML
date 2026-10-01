namespace DirectIndexing.Core.Simulation.Covariance;

/// <summary>
/// Ledoit &amp; Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix": shrinkage of the
/// sample covariance S (1/T normalization) toward the constant-correlation target F, with the
/// closed-form intensity ρ* = clamp(κ̂/T, 0, 1), κ̂ = (π̂ − ψ̂)/γ̂
/// (DataMemo/decisions/VolatilityModel_v03.md §2). Input X is the CENTERED T×N window matrix.
///
/// Two of the terms collapse to cheap closed forms in X:
///   π̂ = Σ_ij (1/T)Σ_t (x_it x_jt − s_ij)² = (1/T)Σ_t (Σ_i x_it²)² − ‖S‖²_F      (O(NT))
///   ϑ̂_ii,ij = (1/T)Σ_t (x_it² − s_ii)(x_it x_jt − s_ij) = M3_ij − s_ii s_ij,
///   M3 = (1/T)(X∘X∘X)ᵀX                                                           (one O(N²T) product)
/// and by symmetry ψ̂ = Σ_i π̂_ii + ρ̄ Σ_{i≠j} √(s_jj/s_ii) ϑ̂_ii,ij.
/// </summary>
// [math:ledoit_wolf] — DataMemo/spec/SymbolTable.md
public static class LedoitWolf
{
    /// <summary>S = XᵀX / T.</summary>
    public static double[,] Sample(double[,] X)
    {
        int T = X.GetLength(0), N = X.GetLength(1);
        var S = new double[N, N];
        Parallel.For(0, N, i =>
        {
            for (int j = i; j < N; j++)
            {
                double acc = 0;
                for (int t = 0; t < T; t++) acc += X[t, i] * X[t, j];
                S[i, j] = acc / T;
            }
        });
        for (int i = 0; i < N; i++) for (int j = 0; j < i; j++) S[i, j] = S[j, i];
        return S;
    }

    /// <summary>(Σ̂, ρ*). Names flagged <paramref name="excluded"/> are left out of every average
    /// (the caller assigns their row/column).</summary>
    public static (double[,] Cov, double Intensity) Shrink(double[,] X, bool[]? excluded = null)
    {
        int T = X.GetLength(0), N = X.GetLength(1);
        excluded ??= new bool[N];
        var S  = Sample(X);
        var sd = new double[N];
        for (int i = 0; i < N; i++) sd[i] = Math.Sqrt(Math.Max(S[i, i], 0));
        bool Use(int i) => !excluded[i] && sd[i] > 0;

        // ρ̄ — average off-diagonal sample correlation over usable names
        double rsum = 0; long npair = 0;
        for (int i = 0; i < N; i++) if (Use(i))
            for (int j = i + 1; j < N; j++) if (Use(j)) { rsum += S[i, j] / (sd[i] * sd[j]); npair++; }
        double rbar = npair > 0 ? rsum / npair : 0;

        // F — constant-correlation target
        var F = new double[N, N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                F[i, j] = i == j ? S[i, i] : rbar * sd[i] * sd[j];

        // π̂ (usable block only)
        double piHat = 0, sNorm = 0;
        for (int t = 0; t < T; t++)
        {
            double q = 0;
            for (int i = 0; i < N; i++) if (Use(i)) q += X[t, i] * X[t, i];
            piHat += q * q;
        }
        piHat /= T;
        for (int i = 0; i < N; i++) if (Use(i)) for (int j = 0; j < N; j++) if (Use(j)) sNorm += S[i, j] * S[i, j];
        piHat -= sNorm;

        // π̂_ii and M3 = (1/T)(X∘³)ᵀX
        var piDiag = new double[N];
        var M3 = new double[N, N];
        Parallel.For(0, N, i =>
        {
            if (!Use(i)) return;
            double m4 = 0;
            for (int t = 0; t < T; t++) { double x = X[t, i]; m4 += x * x * x * x; }
            piDiag[i] = m4 / T - S[i, i] * S[i, i];
            for (int j = 0; j < N; j++)
            {
                if (j == i || !Use(j)) continue;
                double acc = 0;
                for (int t = 0; t < T; t++) { double x = X[t, i]; acc += x * x * x * X[t, j]; }
                M3[i, j] = acc / T;
            }
        });

        double psiHat = 0;
        for (int i = 0; i < N; i++) if (Use(i)) psiHat += piDiag[i];
        double off = 0;
        for (int i = 0; i < N; i++) if (Use(i))
            for (int j = 0; j < N; j++) if (j != i && Use(j))
                off += Math.Sqrt(S[j, j] / S[i, i]) * (M3[i, j] - S[i, i] * S[i, j]);
        psiHat += rbar * off;

        double gammaHat = 0;
        for (int i = 0; i < N; i++) if (Use(i)) for (int j = 0; j < N; j++) if (Use(j))
        { double d = F[i, j] - S[i, j]; gammaHat += d * d; }

        double kappa = gammaHat > 0 ? (piHat - psiHat) / gammaHat : 0;
        double rho   = Math.Clamp(kappa / T, 0.0, 1.0);

        var cov = new double[N, N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                cov[i, j] = rho * F[i, j] + (1 - rho) * S[i, j];
        return (cov, rho);
    }
}
