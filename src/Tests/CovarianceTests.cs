using System.Diagnostics;
using DirectIndexing.Core.Portfolio;
using DirectIndexing.Core.Simulation;
using DirectIndexing.Core.Simulation.Covariance;

/// <summary>
/// v0.3-6 — point-in-time Σ̂ (F1), Ledoit–Wolf shrinkage, dollar active weights (Q2).
/// SymbolTable rows `cov_hat_pit`, `ledoit_wolf`, `delta_w`.
/// </summary>
public class CovarianceTests
{
    // blocks = 1: constant correlation ρ (the LW target is the truth);
    // blocks = 2: two independent sectors with within-sector ρ (the target is misspecified).
    private static double[,] Gaussian(int T, int N, int seed, double rho = 0.3, int blocks = 1)
    {
        var rng = new Random(seed);
        double G() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        var X = new double[T, N];
        for (int t = 0; t < T; t++)
        {
            var f = new double[blocks];
            for (int b = 0; b < blocks; b++) f[b] = G();
            for (int i = 0; i < N; i++) X[t, i] = 0.01 * (Math.Sqrt(rho) * f[i * blocks / N] + Math.Sqrt(1 - rho) * G());
        }
        // center columns
        for (int i = 0; i < N; i++)
        {
            double m = 0; for (int t = 0; t < T; t++) m += X[t, i]; m /= T;
            for (int t = 0; t < T; t++) X[t, i] -= m;
        }
        return X;
    }

    private static bool IsPositiveDefinite(double[,] A)
    {
        int n = A.GetLength(0);
        var L = new double[n, n];
        for (int j = 0; j < n; j++)
        {
            double d = A[j, j];
            for (int k = 0; k < j; k++) d -= L[j, k] * L[j, k];
            if (d <= 1e-14) return false;
            L[j, j] = Math.Sqrt(d);
            for (int i = j + 1; i < n; i++)
            {
                double s = A[i, j];
                for (int k = 0; k < j; k++) s -= L[i, k] * L[j, k];
                L[i, j] = s / L[j, j];
            }
        }
        return true;
    }

    // T < N: the sample covariance is singular (rank ≤ T−1); Ledoit–Wolf is positive definite with
    // intensity in (0, 1]. When the target is the truth (constant correlation) ρ* → 1 (γ̂ → 0);
    // when it is misspecified (two sectors) ρ* falls as T grows (κ̂/T → 0) — the estimator trusts
    // the data more as there is more of it.
    public void Test_LedoitWolf_RepairsRank_IntensityShrinksWithT()
    {
        var Xs = Gaussian(T: 30, N: 40, seed: 1, blocks: 2);
        var S  = LedoitWolf.Sample(Xs);
        var (lw, rhoSmall) = LedoitWolf.Shrink(Xs);
        var (_,  rhoLarge) = LedoitWolf.Shrink(Gaussian(T: 2000, N: 40, seed: 2, blocks: 2));
        var (_,  rhoTrue)  = LedoitWolf.Shrink(Gaussian(T: 2000, N: 40, seed: 2, blocks: 1));

        Debug.Assert(!IsPositiveDefinite(S), "S must be singular when T < N");
        Debug.Assert(IsPositiveDefinite(lw), "Ledoit–Wolf Σ̂ must be positive definite");
        Debug.Assert(rhoSmall > 0 && rhoSmall <= 1, $"intensity in (0,1], got {rhoSmall}");
        Debug.Assert(rhoLarge < 0.5 * rhoSmall, $"misspecified target: intensity must fall with T ({rhoSmall:F3} → {rhoLarge:F3})");
        Debug.Assert(rhoTrue > 0.9, $"true target: intensity → 1, got {rhoTrue:F3}");
        Console.WriteLine($"Covariance Test 1 passed: S singular, LW PD; misspecified ρ* = {rhoSmall:F3} (T=30) → " +
                          $"{rhoLarge:F3} (T=2000); true-target ρ* = {rhoTrue:F3}");
    }

    // F1: the point-in-time Σ̂ at t must not change when returns AFTER t change; the full-sample one does.
    public void Test_PointInTime_IgnoresTheFuture()
    {
        const int T = 400, N = 5, t0 = 300;
        var rng = new Random(3);
        var a = new Dictionary<string, float[]>();
        var b = new Dictionary<string, float[]>();
        for (int i = 0; i < N; i++)
        {
            var r = new float[T]; r[0] = float.NaN;
            for (int s = 1; s < T; s++) r[s] = (float)(0.01 * (rng.NextDouble() - 0.5));
            var r2 = (float[])r.Clone();
            for (int s = t0 + 1; s < T; s++) r2[s] *= 10f;          // a future crash regime
            a[$"S{i}"] = r; b[$"S{i}"] = r2;
        }
        var pa = PriceLoader.CreateForTesting(a);
        var pb = PriceLoader.CreateForTesting(b);
        foreach (bool shrink in new[] { false, true })
        {
            var ca = new PointInTimeCovariance(pa, shrink).Fit(t0);
            var cb = new PointInTimeCovariance(pb, shrink).Fit(t0);
            for (int i = 0; i < N; i++) for (int j = 0; j < N; j++)
                Debug.Assert(ca[i, j] == cb[i, j], $"PIT (shrink={shrink}) Σ̂_t depends on r_(>t) at ({i},{j})");
        }
        var fa = new FullSampleCovariance(pa).At(t0);
        var fb = new FullSampleCovariance(pb).At(t0);
        Debug.Assert(fb[0, 0] > 2 * fa[0, 0], "full-sample Σ̂ must see the future (that is F1)");
        Console.WriteLine("Covariance Test 2 passed: Σ̂_t point-in-time (unchanged by r_(>t)); full-sample is not");
    }

    // Q2: dollar active weights. Holding every name in EQUAL dollars → σ_TE ≈ 0 under both forms;
    // holding every name but in UNEQUAL dollars → 0 under the legacy per-name form, > 0 under dollars.
    public void Test_DollarWeights_SeePositionSize()
    {
        const int T = 300;
        var rng = new Random(4);
        var rets = new Dictionary<string, float[]>();
        foreach (var s in new[] { "A", "B", "C" })
        {
            var r = new float[T]; r[0] = float.NaN;
            for (int k = 1; k < T; k++) r[k] = (float)(0.02 * (rng.NextDouble() - 0.5));
            rets[s] = r;
        }
        var prices = PriceLoader.CreateForTesting(rets);
        var closes = new Dictionary<string, decimal> { ["A"] = 100m, ["B"] = 100m, ["C"] = 100m };
        var equal   = new[] { new Lot("A", "X", 100m, 10, 0), new Lot("B", "X", 100m, 10, 0), new Lot("C", "X", 100m, 10, 0) };
        var unequal = new[] { new Lot("A", "X", 100m, 80, 0), new Lot("B", "X", 100m, 10, 0), new Lot("C", "X", 100m, 10, 0) };

        var cov     = new PointInTimeCovariance(prices, shrink: true);
        var dollars = new TrackingErrorProxy(prices, cov, TeWeighting.Dollars);
        var names   = new TrackingErrorProxy(prices, cov, TeWeighting.Names);
        float eqD = dollars.Update(T - 1, equal, closes), unD = dollars.Update(T - 1, unequal, closes);
        float unN = names.Update(T - 1, unequal, closes);

        Debug.Assert(eqD < 1e-5f, $"equal dollars in every name → σ_TE ≈ 0, got {eqD}");
        Debug.Assert(unN < 1e-5f, $"legacy per-name δw is blind to size → 0, got {unN}");
        Debug.Assert(unD > 0.01f, $"dollar δw must see the 80% position, got {unD}");
        Console.WriteLine($"Covariance Test 3 passed: dollar δw sees position size (σ_TE {unN:F4} → {unD:F4})");
    }
}
