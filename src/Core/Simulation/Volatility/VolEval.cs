using System.Text.Json;

namespace DirectIndexing.Core.Simulation.Volatility;

/// <summary>
/// `vol-eval` (v0.3-7): out-of-sample QLIKE of each σ̂ estimator against the realized-variance
/// proxy σ̃²_{t,h} = (1/h)Σ_{s=1..h} r²_{t+s}, per horizon h ∈ {1, 5, 21}, overall and by tercile
/// of the market EWMA σ̂_m (the reporting stratification of VolatilityModel_v03 §4). Every
/// forecast is the 𝓕_t one (VolPath contract); only the PROXY looks forward — it is the target.
/// Baselines: constant (expanding mean of r²), trailing-21 (legacy), and the one-day Parkinson
/// range estimator SigmaRange²/(4 ln 2) where the world has intraday ranges.
/// </summary>
public static class VolEval
{
    public static readonly int[] Horizons = { 1, 5, 21 };

    public sealed record Cell(double Mean, double Low, double Mid, double High, long N);

    public static Dictionary<int, Dictionary<string, Cell>> Run(PriceLoader prices)
    {
        var symbols = prices.Symbols.OrderBy(s => s).ToList();
        int T = prices.DayCount;

        // market return r_m = mean of available r_i, and its EWMA → tercile cutoffs (reporting only)
        var rm = new float[T];
        for (int t = 0; t < T; t++)
        {
            double s = 0; int n = 0;
            foreach (var sym in symbols) { float x = prices.DailyReturn(sym, t); if (!float.IsNaN(x)) { s += x; n++; } }
            rm[t] = n > 0 ? (float)(s / n) : float.NaN;
        }
        var mkt = new EwmaVol().Path(rm).Var1;
        var valid = mkt.Where(x => !double.IsNaN(x)).OrderBy(x => x).ToArray();
        double q1 = valid.Length > 0 ? valid[valid.Length / 3] : 0, q2 = valid.Length > 0 ? valid[2 * valid.Length / 3] : 0;
        int Regime(int t) => double.IsNaN(mkt[t]) ? -1 : mkt[t] <= q1 ? 0 : mkt[t] <= q2 ? 1 : 2;

        var estimators = new (string Name, Func<string, float[], VolPath> Path)[]
        {
            ("constant",   (_, r) => ExpandingMean(r)),
            ("trailing21", (_, r) => new Trailing21Vol().Path(r)),
            ("ewma",       (_, r) => new EwmaVol().Path(r)),
            ("garch",      (_, r) => new Garch11Vol().Path(r)),
            ("range",      (sym, r) => Parkinson(prices, sym, r.Length)),
        };

        // accumulators[h][estimator] = (sum, n) per regime 0..2
        var acc = Horizons.ToDictionary(h => h, _ => estimators.ToDictionary(e => e.Name, _ => new double[6]));
        var gate = new object();
        Parallel.ForEach(symbols, sym =>
        {
            var r = prices.GetReturnArray(sym);
            var local = Horizons.ToDictionary(h => h, _ => estimators.ToDictionary(e => e.Name, _ => new double[6]));
            // all paths first; an estimator that is NaN everywhere for this name (no ranges in the
            // world) sits out, every other one must be valid at t — one common scoring set
            var paths = estimators.Select(e => (e.Name, P: e.Path(sym, r)))
                                  .Where(x => x.P.Var1.Any(v => !double.IsNaN(v))).ToList();
            var f = new double[paths.Count];
            foreach (int h in Horizons)
                for (int t = 0; t + h < T; t++)
                {
                    int g = Regime(t);
                    if (g < 0) continue;
                    bool ok = true;
                    for (int k = 0; k < paths.Count && ok; k++)
                    {
                        f[k] = paths[k].P.Horizon(t, h) / h;
                        ok = !double.IsNaN(f[k]) && f[k] > 0;
                    }
                    if (!ok) continue;
                    double proxy = 0;
                    for (int s = 1; s <= h && ok; s++) { float x = r[t + s]; if (float.IsNaN(x)) ok = false; else proxy += (double)x * x; }
                    if (!ok) continue;
                    for (int k = 0; k < paths.Count; k++)
                    {
                        var a = local[h][paths[k].Name];
                        a[2 * g]     += QLike.Loss(f[k], proxy / h);
                        a[2 * g + 1] += 1;
                    }
                }
            lock (gate)
                foreach (int h in Horizons)
                    foreach (var e in estimators)
                        for (int k = 0; k < 6; k++) acc[h][e.Name][k] += local[h][e.Name][k];
        });

        // every estimator present for a name is scored on the same (name, day) set, so cells are comparable
        return acc.ToDictionary(kv => kv.Key, kv => kv.Value
            .Where(e => e.Value[1] + e.Value[3] + e.Value[5] > 0)
            .ToDictionary(e => e.Key, e =>
            {
                var a = e.Value;
                double n = a[1] + a[3] + a[5];
                return new Cell((a[0] + a[2] + a[4]) / n, a[0] / Math.Max(a[1], 1), a[2] / Math.Max(a[3], 1), a[4] / Math.Max(a[5], 1), (long)n);
            }));
    }

    public static void Write(Dictionary<int, Dictionary<string, Cell>> result, string path, string world)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var doc = new
        {
            world,
            loss = "QLIKE = p/f - ln(p/f) - 1, p = mean r^2 over the next h days, f = V_{t,h}/h",
            regimes = "terciles of the market EWMA variance (low / mid / high)",
            horizons = result.ToDictionary(kv => $"h{kv.Key}", kv => kv.Value),
        };
        File.WriteAllText(path, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static VolPath ExpandingMean(float[] r)
    {
        var v = new double[r.Length];
        double s = 0; int n = 0;
        for (int t = 0; t < r.Length; t++)
        {
            if (!float.IsNaN(r[t])) { s += (double)r[t] * r[t]; n++; }
            v[t] = n >= 20 ? s / n : double.NaN;
        }
        return VolEstimators.Flat(v);
    }

    private static VolPath Parkinson(PriceLoader prices, string sym, int T)
    {
        var v = new double[T];
        for (int t = 0; t < T; t++)
        {
            float rv = prices.RangeVol(sym, t);
            v[t] = float.IsNaN(rv) || rv <= 0 ? double.NaN : (double)rv * rv / (4 * Math.Log(2));
        }
        return VolEstimators.Flat(v);
    }
}
