using System.Diagnostics;
using DirectIndexing.Core.Simulation.Volatility;

/// <summary>
/// v0.3-7 σ̂ estimators and QLIKE (DataMemo/decisions/VolatilityModel_v03.md §3–§5).
/// SymbolTable rows `sigma_hat_ewma`, `sigma_hat_garch`, `vol_horizon`, `qlike`.
/// </summary>
public class VolatilityTests
{
    private static float[] SimulateGarch(int n, double omega, double alpha, double beta, int seed)
    {
        var rng = new Random(seed);
        double G() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        var r = new float[n]; r[0] = float.NaN;
        double h = omega / (1 - alpha - beta);
        for (int t = 1; t < n; t++)
        {
            double x = Math.Sqrt(h) * G();
            r[t] = (float)x;
            h = omega + alpha * x * x + beta * h;
        }
        return r;
    }

    // EWMA recursion pinned on a hand-checkable series; QLIKE minimized at the truth, with the
    // Gaussian floor E[Z² − ln Z² − 1] = γ_E + ln 2 ≈ 1.2704 for a one-day r² proxy.
    public void Test_Ewma_Recursion_And_QLike_Floor()
    {
        var r = new float[25];
        r[0] = float.NaN;
        for (int t = 1; t < 25; t++) r[t] = 0.01f;
        r[21] = 0.05f;                                             // the shock after 20 init returns
        var v = new EwmaVol().Path(r).Var1;
        double small = (double)0.01f * 0.01f, shock = (double)0.05f * 0.05f;   // float32 inputs
        double init = small;                                        // mean of twenty 0.01² (r[1..20])
        double expect = EwmaVol.Lambda * init + (1 - EwmaVol.Lambda) * shock;
        Debug.Assert(double.IsNaN(v[19]), "EWMA must be NaN before 20 returns");
        Debug.Assert(Math.Abs(v[20] - init) < 1e-12, $"init: expected {init}, got {v[20]}");
        Debug.Assert(Math.Abs(v[21] - expect) < 1e-12, $"recursion: expected {expect}, got {v[21]}");

        Debug.Assert(QLike.Loss(2.0, 2.0) == 0.0 && QLike.Loss(1.0, 2.0) > 0 && QLike.Loss(4.0, 2.0) > 0,
            "QLIKE must be 0 at the truth and positive elsewhere");
        var rng = new Random(9);
        double sum = 0; const int n = 200_000;
        for (int i = 0; i < n; i++)
        {
            double z = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
            sum += QLike.Loss(1.0, z * z);
        }
        double floor = 0.5772156649 + Math.Log(2);
        Debug.Assert(Math.Abs(sum / n - floor) < 0.02, $"QLIKE floor: expected ≈ {floor:F4}, got {sum / n:F4}");
        Console.WriteLine($"Volatility Test 1 passed: EWMA recursion exact; QLIKE floor {sum / n:F4} ≈ γ+ln2 = {floor:F4}");
    }

    // Walk-forward contract: out[t] never depends on r_(>t), for every estimator; Fit rejects look-ahead.
    public void Test_Estimators_AreCausal_FitRejectsLookAhead()
    {
        var a = SimulateGarch(1500, 2e-6, 0.08, 0.90, seed: 5);
        var b = (float[])a.Clone();
        const int t0 = 1100;
        for (int t = t0 + 1; t < b.Length; t++) b[t] *= 5f;
        foreach (IVolEstimator e in new IVolEstimator[] { new Trailing21Vol(), new EwmaVol(), new Garch11Vol() })
        {
            var pa = e.Path(a); var pb = e.Path(b);
            for (int t = 0; t <= t0; t++)
                Debug.Assert(pa.Var1[t].Equals(pb.Var1[t]) && pa.Phi[t].Equals(pb.Phi[t]),
                    $"{e.Name}: forecast at t={t} depends on r_(>{t0})");
        }
        bool threw = false;
        try { Garch11Vol.Fit(a, 0, t0 + 1, origin: t0); } catch (ArgumentException) { threw = true; }
        Debug.Assert(threw, "GARCH Fit must reject a window that reaches past its origin");
        Console.WriteLine("Volatility Test 2 passed: trailing21/EWMA/GARCH causal; Fit rejects look-ahead");
    }

    // On GARCH data the QMLE recovers (α, β) and the ranking flips vs the GBM control:
    // GARCH < EWMA < trailing-21 < constant in QLIKE at h = 1.
    public void Test_Garch_RecoversParams_AndWinsOnClusteredData()
    {
        var r = SimulateGarch(6000, 2e-6, 0.08, 0.90, seed: 11);
        var (p, _) = Garch11Vol.Fit(r, 1, 5999, origin: 5999);
        Debug.Assert(Math.Abs(p.Alpha - 0.08) < 0.03 && Math.Abs(p.Beta - 0.90) < 0.04,
            $"QMLE: expected α≈0.08 β≈0.90, got α={p.Alpha:F3} β={p.Beta:F3}");

        double Score(VolPath path)
        {
            double s = 0; int n = 0;
            for (int t = 600; t + 1 < r.Length; t++)
            {
                double f = path.Var1[t];
                if (double.IsNaN(f)) continue;
                s += QLike.Loss(f, (double)r[t + 1] * r[t + 1]); n++;
            }
            return s / n;
        }
        double g = Score(new Garch11Vol().Path(r)), e = Score(new EwmaVol().Path(r)), tr = Score(new Trailing21Vol().Path(r));
        var c = new double[r.Length]; double cs = 0; int cn = 0;
        for (int t = 0; t < r.Length; t++) { if (!float.IsNaN(r[t])) { cs += (double)r[t] * r[t]; cn++; } c[t] = cn > 20 ? cs / cn : double.NaN; }
        double k = Score(new VolPath(c, c, c));
        Debug.Assert(g < e && e < tr && tr < k, $"QLIKE ranking on GARCH data: garch {g:F4} ewma {e:F4} trailing {tr:F4} constant {k:F4}");
        Console.WriteLine($"Volatility Test 3 passed: α={p.Alpha:F3} β={p.Beta:F3}; QLIKE garch {g:F4} < ewma {e:F4} < trailing21 {tr:F4} < constant {k:F4}");
    }
}
