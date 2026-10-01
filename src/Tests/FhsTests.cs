using System.Diagnostics;
using DirectIndexing.Core.Simulation;
using DirectIndexing.Core.Simulation.Volatility;

/// <summary>
/// v0.3-9 / v0.3-10 — the FHS label simulator and the FHS world
/// (DataMemo/decisions/VolatilityModel_v03.md §7–§8). SymbolTable rows `y_soft_fhs`, `fhs_world`.
/// </summary>
public class FhsTests
{
    internal static PriceLoader ClusteredSource() => PriceLoader.GarchFactorPanel(20, 3000, seed: 21);

    private static double AbsAcf(float[] r, int lag)
    {
        var a = r.Skip(1).Select(x => Math.Abs((double)x)).ToArray();
        double m = a.Average(), num = 0, den = 0;
        for (int t = 0; t < a.Length; t++) den += (a[t] - m) * (a[t] - m);
        for (int t = lag; t < a.Length; t++) num += (a[t] - m) * (a[t - lag] - m);
        return num / den;
    }

    private static double ExcessKurtosis(IEnumerable<double> xs)
    {
        var a = xs.ToArray(); double m = a.Average();
        double m2 = a.Average(x => (x - m) * (x - m)), m4 = a.Average(x => Math.Pow(x - m, 4));
        return m4 / (m2 * m2) - 3;
    }

    private static double MeanPairCorr(PriceLoader w, int from)
    {
        var syms = w.Symbols.OrderBy(s => s).ToList();
        var R = syms.Select(s => w.GetReturnArray(s).Skip(from).Select(x => (double)x).ToArray()).ToList();
        double sum = 0; int k = 0;
        for (int i = 0; i < R.Count; i++) for (int j = i + 1; j < R.Count; j++)
        {
            double mi = R[i].Average(), mj = R[j].Average(), c = 0, vi = 0, vj = 0;
            for (int t = 0; t < R[i].Length; t++) { c += (R[i][t] - mi) * (R[j][t] - mj); vi += (R[i][t] - mi) * (R[i][t] - mi); vj += (R[j][t] - mj) * (R[j][t] - mj); }
            sum += c / Math.Sqrt(vi * vj); k++;
        }
        return sum / k;
    }

    // The FHS world clusters (ACF of |r| > 0, vs ≈ 0 for GBM), has fat tails, keeps the source's
    // cross-sectional correlation, and is a deterministic function of its seed.
    public void Test_FhsWorld_Clusters_FatTails_Correlation_Deterministic()
    {
        var source = ClusteredSource();
        var w  = PriceLoader.FromFhs(source, 2500, seed: 3);
        var w2 = PriceLoader.FromFhs(source, 2500, seed: 3);
        var g  = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(20, 0.25f), 2500, seed: 3);

        var sym = w.Symbols.OrderBy(s => s).First();
        double acfFhs = Enumerable.Range(1, 5).Average(l => AbsAcf(w.GetReturnArray(sym), l));
        double acfGbm = Enumerable.Range(1, 5).Average(l => AbsAcf(g.GetReturnArray(g.Symbols.First()), l));
        double kurt   = ExcessKurtosis(w.GetReturnArray(sym).Skip(1).Select(x => (double)x));
        double cSrc = MeanPairCorr(source, 1), cFhs = MeanPairCorr(w, 1);
        bool same = w.Symbols.All(s => w.GetReturnArray(s).SequenceEqual(w2.GetReturnArray(s)));

        Debug.Assert(acfFhs > 0.05 && Math.Abs(acfGbm) < 0.05, $"|r| ACF: FHS {acfFhs:F3} must exceed GBM {acfGbm:F3}");
        Debug.Assert(kurt > 0.5, $"FHS excess kurtosis must be > 0.5, got {kurt:F2}");
        Debug.Assert(Math.Abs(cSrc - cFhs) < 0.08, $"mean pairwise correlation: source {cSrc:F3} vs FHS {cFhs:F3}");
        Debug.Assert(same, "same seed must give the same world");
        Console.WriteLine($"FHS Test 1 passed: |r| ACF {acfFhs:F3} (GBM {acfGbm:F3}), excess kurtosis {kurt:F2}, " +
                          $"corr {cSrc:F3} → {cFhs:F3}, deterministic");
    }

    // FHS labels are causal (the pool and σ̂ at t ignore r_(>t)) and, on a constant-σ GBM world,
    // agree with the GBM label's firing probability within Monte Carlo + σ̂-noise tolerance.
    public void Test_FhsLabels_Causal_And_AgreeWithGbmOnGbm()
    {
        var gbmWorld = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(5, 0.30f), 900, seed: 8);
        var fhs = new FhsSimulator(gbmWorld, paths: 4000);
        var sym = gbmWorld.Symbols.OrderBy(s => s).First();
        const int t0 = 600;
        float p0 = gbmWorld.GetClose(sym, t0);
        // fire when the price is ≥ 5% below today's close (a pure loss-gate first passage)
        bool Fires(float price, int s) => price <= 0.95f * p0;
        float f = fhs.FractionFiring(sym, t0, p0, Fires, new Random(1));
        // GBM at the SAME σ̂ FHS starts from, so only the path model differs (4000 paths: MC s.e. ≈ 0.008)
        float sigHat = (float)Math.Sqrt(252 * new EwmaVol().Path(gbmWorld.GetReturnArray(sym)).Var1[t0]);
        float gb = new GbmSimulator(paths: 4000, horizon: 30).FractionFiring(p0, sigHat, Fires, new Random(1));
        float closed = (float)LossBarrier.TouchProbability(LossBarrier.Z(-Math.Log(0.95), 0.30 * 0.30 * 30 / 252));
        // both simulators monitor DAILY, so both sit below the continuous-time closed form (an upper
        // bound); on a constant-σ world FHS (unit-variance pool) must agree with GBM
        Debug.Assert(Math.Abs(f - gb) < 0.04, $"touch probability: FHS {f:F3} vs GBM {gb:F3}");
        Debug.Assert(f <= closed + 0.03 && gb <= closed + 0.03, $"discrete monitoring must not exceed 2Φ(−z) = {closed:F3}");

        // causality: perturbing returns after t0 leaves the t0 label unchanged
        var bumped = gbmWorld.Symbols.ToDictionary(s => s, s =>
        {
            var r = (float[])gbmWorld.GetReturnArray(s).Clone();
            for (int t = t0 + 1; t < r.Length; t++) r[t] *= 3f;
            return r;
        });
        var fhs2 = new FhsSimulator(PriceLoader.CreateForTesting(bumped));
        var fhs1 = new FhsSimulator(PriceLoader.CreateForTesting(gbmWorld.Symbols.ToDictionary(s => s, s => gbmWorld.GetReturnArray(s))));
        float a = fhs1.FractionFiring(sym, t0, 100f, (px, s) => px <= 95f, new Random(4));
        float b = fhs2.FractionFiring(sym, t0, 100f, (px, s) => px <= 95f, new Random(4));
        Debug.Assert(a == b, $"FHS label at t0 depends on r_(>t0): {a} vs {b}");
        Console.WriteLine($"FHS Test 2 passed: GBM world touch prob FHS {f:F3} / GBM {gb:F3} / closed {closed:F3}; causal");
    }
}
