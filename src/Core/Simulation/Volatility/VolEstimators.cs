namespace DirectIndexing.Core.Simulation.Volatility;

/// <summary>
/// A volatility forecast path for one return series (daily units). Index t is the forecast made
/// at the CLOSE of day t (𝓕_t): <see cref="Var1"/>[t] ≈ Var(r_{t+1} | 𝓕_t). NaN where the estimator
/// has too little history. <see cref="Phi"/> and <see cref="LongRun"/> carry the term structure
/// E_t[σ²_{t+s}] = σ̄² + φ^{s−1}(σ²_{t+1} − σ̄²) (φ = 1: flat).
/// (DataMemo/decisions/VolatilityModel_v03.md §3.)
/// </summary>
public sealed record VolPath(double[] Var1, double[] Phi, double[] LongRun)
{
    /// <summary>V_{t,h} = Σ_{s=1}^{h} E_t[σ²_{t+s}] — the h-day variance forecast.</summary>
    // [math:vol_horizon] — DataMemo/spec/SymbolTable.md
    public double Horizon(int t, int h)
    {
        double v = Var1[t], phi = Phi[t], bar = LongRun[t];
        if (double.IsNaN(v)) return double.NaN;
        if (phi >= 1.0 - 1e-12) return h * v;
        return h * bar + (v - bar) * (1 - Math.Pow(phi, h)) / (1 - phi);
    }
}

/// <summary>σ̂ estimator: maps a daily return series to its 𝓕_t forecast path. Never reads r_{s&gt;t} for out[t].</summary>
public interface IVolEstimator
{
    string Name { get; }
    VolPath Path(float[] r);
}

public static class VolEstimators
{
    public static IVolEstimator Parse(string s) => s switch
    {
        "trailing21" => new Trailing21Vol(),
        "ewma"       => new EwmaVol(),
        "garch"      => new Garch11Vol(),
        _ => throw new ArgumentException($"vol estimator must be trailing21|ewma|garch, got '{s}'"),
    };

    internal static VolPath Flat(double[] v) =>
        new(v, Enumerable.Repeat(1.0, v.Length).ToArray(), (double[])v.Clone());
}

/// <summary>
/// Legacy: population variance of r_{t−21..t−1} (needs ≥ 5 returns) — exactly
/// SoftLabelBuilder's historical trailing estimate, which EXCLUDES r_t. Flat term structure.
/// </summary>
// [math:sigma_hat_trailing] — DataMemo/spec/SymbolTable.md
public sealed class Trailing21Vol : IVolEstimator
{
    public const int Window = 21;
    public string Name => "trailing21";

    public VolPath Path(float[] r)
    {
        var v = new double[r.Length];
        for (int t = 0; t < r.Length; t++)
        {
            int start = Math.Max(0, t - Window), n = 0;
            double sum = 0, sumSq = 0;
            for (int i = start; i < t; i++)
            {
                float x = r[i];
                if (float.IsNaN(x)) continue;
                sum += x; sumSq += (double)x * x; n++;
            }
            v[t] = n < 5 ? double.NaN : Math.Max(sumSq / n - (sum / n) * (sum / n), 0);
        }
        return VolEstimators.Flat(v);
    }
}

/// <summary>
/// RiskMetrics EWMA: σ̂²_t = λσ̂²_{t−1} + (1−λ)r_t², λ = 0.94, initialized with the mean of the
/// first <see cref="InitCount"/> squared returns (NaN before that). Zero-mean daily returns.
/// No fitted parameters → no leakage surface. IGARCH: flat term structure.
/// </summary>
// [math:sigma_hat_ewma] — DataMemo/spec/SymbolTable.md
public sealed class EwmaVol : IVolEstimator
{
    public const double Lambda    = 0.94;
    public const int    InitCount = 20;
    public string Name => "ewma";

    public VolPath Path(float[] r)
    {
        var v = new double[r.Length];
        Array.Fill(v, double.NaN);
        double s2 = double.NaN, initSum = 0;
        int seen = 0;
        for (int t = 0; t < r.Length; t++)
        {
            float x = r[t];
            if (!float.IsNaN(x))
            {
                double x2 = (double)x * x;
                if (seen < InitCount) { initSum += x2; seen++; if (seen == InitCount) s2 = initSum / InitCount; }
                else s2 = Lambda * s2 + (1 - Lambda) * x2;
            }
            v[t] = s2;   // stale-forward over missing days
        }
        return VolEstimators.Flat(v);
    }
}

/// <summary>
/// GARCH(1,1) σ²_{t+1} = ω + α r_t² + β σ²_t, Gaussian QMLE with variance targeting
/// (ω = s²(1 − α − β), s² the window's mean square) — fitted WALK-FORWARD only: every
/// <see cref="RefitDays"/> trading days on the trailing ≤ <see cref="MaxWindow"/> days ending
/// at the refit day (<see cref="Fit"/> rejects any window past its origin). Before
/// <see cref="MinObs"/> returns exist the path falls back to EWMA. Mean-reverting term
/// structure φ = α + β, σ̄² = s².
/// </summary>
// [math:sigma_hat_garch] — DataMemo/spec/SymbolTable.md
public sealed class Garch11Vol : IVolEstimator
{
    public const int RefitDays = 63;
    public const int MinObs    = 500;
    public const int MaxWindow = 2000;
    public string Name => "garch";

    public readonly record struct Params(double Omega, double Alpha, double Beta, double LongRun)
    {
        public double Persistence => Alpha + Beta;
    }

    public VolPath Path(float[] r)
    {
        var ewma = new EwmaVol().Path(r);
        var v   = (double[])ewma.Var1.Clone();
        var phi = Enumerable.Repeat(1.0, r.Length).ToArray();
        var bar = (double[])ewma.Var1.Clone();

        // valid-return index (GARCH runs on the observed sequence; missing days are skipped)
        int firstFit = -1, nValid = 0;
        for (int t = 0; t < r.Length; t++)
            if (!float.IsNaN(r[t]) && ++nValid == MinObs) { firstFit = t; break; }
        if (firstFit < 0) return new VolPath(v, phi, bar);

        Params p = default;
        double h = double.NaN;
        for (int t = firstFit; t < r.Length; t++)
        {
            if ((t - firstFit) % RefitDays == 0)
            {
                (p, h) = Fit(r, Math.Max(0, t - MaxWindow + 1), t, origin: t);
            }
            else if (!float.IsNaN(r[t]))
                h = p.Omega + p.Alpha * (double)r[t] * r[t] + p.Beta * h;
            v[t] = h; phi[t] = p.Persistence; bar[t] = p.LongRun;
        }
        return new VolPath(v, phi, bar);
    }

    /// <summary>
    /// QMLE on r[from..to] (inclusive, NaN skipped). Returns the parameters and the filtered
    /// one-step forecast σ²_{to+1|to}. Throws if the window reaches past <paramref name="origin"/> —
    /// the walk-forward contract (VolatilityModel_v03 §5, hazard 1).
    /// </summary>
    public static (Params P, double NextVar) Fit(float[] r, int from, int to, int origin)
    {
        if (to > origin)
            throw new ArgumentException($"GARCH Fit window ends at {to}, past the forecast origin {origin} (look-ahead)");
        var x2 = new List<double>(to - from + 1);
        for (int t = from; t <= to; t++) if (!float.IsNaN(r[t])) x2.Add((double)r[t] * r[t]);
        double s2 = x2.Count > 0 ? x2.Average() : 1e-4;

        double best = double.PositiveInfinity, bA = 0.05, bB = 0.90;
        for (double a = 0.01; a <= 0.30; a += 0.02)
            for (double b = 0.50; b < 0.995 - a; b += 0.03)
            {
                double nll = Nll(x2, s2, a, b, out _);
                if (nll < best) { best = nll; bA = a; bB = b; }
            }
        // pattern-search refinement
        for (double step = 0.01; step >= 0.000_5; step /= 2)
        {
            bool moved = true;
            while (moved)
            {
                moved = false;
                foreach (var (da, db) in new[] { (step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step) })
                {
                    double a = bA + da, b = bB + db;
                    if (a < 1e-4 || b < 0 || a + b >= 0.999) continue;
                    double nll = Nll(x2, s2, a, b, out _);
                    if (nll < best - 1e-12) { best = nll; bA = a; bB = b; moved = true; }
                }
            }
        }
        Nll(x2, s2, bA, bB, out double next);
        return (new Params(s2 * (1 - bA - bB), bA, bB, s2), next);
    }

    private static double Nll(List<double> x2, double s2, double a, double b, out double next)
    {
        double omega = s2 * (1 - a - b), h = s2, nll = 0;
        foreach (var e in x2)
        {
            nll += Math.Log(h) + e / h;
            h = omega + a * e + b * h;
        }
        next = h;
        return nll;
    }
}
