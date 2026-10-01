namespace DirectIndexing.Core.Simulation.Covariance;

/// <summary>
/// Σ̂_t — the return covariance the tracking-error quadratic form uses on day t
/// (DataMemo/decisions/VolatilityModel_v03.md §2). Daily units, symmetric PSD, indexed
/// by <see cref="Symbols"/> (sorted). The measurability contract: <see cref="At"/>(t) may
/// use returns r_s only for s ≤ t — except the legacy <see cref="FullSampleCovariance"/>,
/// which is kept precisely so its look-ahead (ROADMAP F1) can be measured.
/// </summary>
public interface ICovarianceEstimator
{
    string Name { get; }
    IReadOnlyList<string> Symbols { get; }
    float[,] At(int t);
}

public enum CovarianceMode { FullSample, PitSample, PitLedoitWolf }

public static class CovarianceFactory
{
    public static ICovarianceEstimator Create(PriceLoader prices, CovarianceMode mode) => mode switch
    {
        CovarianceMode.FullSample    => new FullSampleCovariance(prices),
        CovarianceMode.PitSample     => new PointInTimeCovariance(prices, shrink: false),
        CovarianceMode.PitLedoitWolf => new PointInTimeCovariance(prices, shrink: true),
        _ => throw new ArgumentOutOfRangeException(nameof(mode)),
    };

    public static CovarianceMode Parse(string s) => s switch
    {
        "fullsample" => CovarianceMode.FullSample,
        "pit"        => CovarianceMode.PitSample,
        "pit-lw"     => CovarianceMode.PitLedoitWolf,
        _ => throw new ArgumentException($"--cov must be fullsample|pit|pit-lw, got '{s}'"),
    };

    public static string Tag(CovarianceMode m) => m switch
    {
        CovarianceMode.FullSample => "fullsample", CovarianceMode.PitSample => "pit", _ => "pit-lw",
    };
}

/// <summary>
/// The legacy v0.2 estimator: pairwise available-case sample covariance over the loader's
/// ENTIRE history, computed once. Contains r_{t+1..T} on day t — a look-ahead (F1). Arm only.
/// </summary>
// [math:cov_hat] — DataMemo/spec/SymbolTable.md
public sealed class FullSampleCovariance : ICovarianceEstimator
{
    private readonly float[,] _cov;
    public string Name => "fullsample";
    public IReadOnlyList<string> Symbols { get; }

    public FullSampleCovariance(PriceLoader prices)
    {
        var symbols = prices.Symbols.OrderBy(s => s).ToList();
        Symbols = symbols;
        _cov = Compute(prices, symbols);
    }

    public float[,] At(int t) => _cov;

    internal static float[,] Compute(PriceLoader prices, List<string> symbols)
    {
        int N = symbols.Count;
        var retArrays = symbols.Select(s => prices.GetReturnArray(s)).ToArray();
        int T         = retArrays[0].Length;

        var means  = new double[N];
        var counts = new int[N];
        for (int i = 0; i < N; i++)
        {
            for (int t = 0; t < T; t++)
            {
                float r = retArrays[i][t];
                if (!float.IsNaN(r)) { means[i] += r; counts[i]++; }
            }
            if (counts[i] > 0) means[i] /= counts[i];
        }

        var cov = new float[N, N];
        for (int i = 0; i < N; i++)
        {
            for (int j = i; j < N; j++)
            {
                double sumCov = 0;
                int    cnt    = 0;
                for (int t = 0; t < T; t++)
                {
                    float ri = retArrays[i][t];
                    float rj = retArrays[j][t];
                    if (!float.IsNaN(ri) && !float.IsNaN(rj))
                    {
                        sumCov += (ri - means[i]) * (rj - means[j]);
                        cnt++;
                    }
                }
                float c = cnt > 1 ? (float)(sumCov / (cnt - 1)) : 0f;
                cov[i, j] = c;
                cov[j, i] = c;
            }
        }
        return cov;
    }
}

/// <summary>
/// Point-in-time Σ̂_t from the trailing window r_{t−L+1..t} (L = 252; expanding from the first
/// return while fewer exist), refit every <see cref="RefitDays"/> trading days and held in
/// between — F_t-measurable by construction. <c>shrink</c> selects Ledoit–Wolf toward the
/// constant-correlation target (<see cref="LedoitWolf"/>); otherwise the raw sample S_t, which is
/// singular when L ≤ N (rank ≤ L − 1).
/// </summary>
// [math:cov_hat_pit] — DataMemo/spec/SymbolTable.md
public sealed class PointInTimeCovariance : ICovarianceEstimator
{
    public const int Window    = 252;
    public const int RefitDays = 21;
    public const int MinWindow = 60;

    private readonly float[][] _returns;   // by symbol index
    private readonly bool      _shrink;
    private int       _fitDay = int.MinValue;
    private float[,]? _cov;

    public string Name => _shrink ? "pit-lw" : "pit";
    public IReadOnlyList<string> Symbols { get; }

    /// <summary>ρ* of the most recent refit (0 for the unshrunk arm).</summary>
    public double LastIntensity { get; private set; }

    public PointInTimeCovariance(PriceLoader prices, bool shrink)
    {
        var symbols = prices.Symbols.OrderBy(s => s).ToList();
        Symbols  = symbols;
        _returns = symbols.Select(s => prices.GetReturnArray(s)).ToArray();
        _shrink  = shrink;
    }

    public float[,] At(int t)
    {
        if (_cov is null || t < _fitDay || t - _fitDay >= RefitDays)
        {
            _cov    = Fit(t);
            _fitDay = t;
        }
        return _cov;
    }

    /// <summary>Σ̂ from returns r_s, s ∈ [max(1, t−L+1), t] — never beyond t.</summary>
    public float[,] Fit(int t)
    {
        int N     = _returns.Length;
        int end   = Math.Min(t, _returns[0].Length - 1);
        int start = Math.Max(1, end - Window + 1);
        int T     = end - start + 1;
        if (T < 2) return new float[N, N];

        // centered window matrix; a name needs ≥ max(T/2, MinWindow∧T) observations, else it is "thin"
        var X    = new double[T, N];
        var thin = new bool[N];
        int need = Math.Max(T / 2, Math.Min(MinWindow, T));
        for (int i = 0; i < N; i++)
        {
            double sum = 0; int n = 0;
            for (int s = 0; s < T; s++)
            {
                float r = _returns[i][start + s];
                if (!float.IsNaN(r)) { sum += r; n++; }
            }
            if (n < need) { thin[i] = true; continue; }
            double mean = sum / n;
            for (int s = 0; s < T; s++)
            {
                float r = _returns[i][start + s];
                X[s, i] = float.IsNaN(r) ? 0.0 : r - mean;   // missing → window mean (biases s_ij toward 0; recorded)
            }
        }

        double[,] cov;
        if (_shrink) { (cov, double rho) = LedoitWolf.Shrink(X, thin); LastIntensity = rho; }
        else         { cov = LedoitWolf.Sample(X); LastIntensity = 0; }

        // thin names (too few returns in the window): median variance of the rest, zero covariance
        var vars = Enumerable.Range(0, N).Where(i => !thin[i]).Select(i => cov[i, i]).OrderBy(v => v).ToList();
        double medVar = vars.Count > 0 ? vars[vars.Count / 2] : 0.0;
        var outCov = new float[N, N];
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                outCov[i, j] = (float)(thin[i] || thin[j] ? (i == j ? medVar : 0.0) : cov[i, j]);
        return outCov;
    }
}

/// <summary>
/// The engine's risk-model configuration: which Σ̂ and which δw form σ_TE. Default (v0.3-6) is
/// the point-in-time Ledoit–Wolf Σ̂ with dollar active weights; <see cref="Legacy"/> reproduces
/// the v0.2–v0.3-5 numbers (full-history Σ̂, equal-per-name δw) for the F1 ablation.
/// </summary>
public sealed record RiskModel
{
    public CovarianceMode Covariance { get; init; } = CovarianceMode.PitLedoitWolf;
    public TeWeighting    Weighting  { get; init; } = TeWeighting.Dollars;

    public static RiskModel Default { get; } = new();
    public static RiskModel Legacy  { get; } = new() { Covariance = CovarianceMode.FullSample, Weighting = TeWeighting.Names };

    /// <summary>Dataset suffix for non-default arms, so they never overwrite the default dataset.</summary>
    public string DatasetTag =>
        (Covariance == CovarianceMode.PitLedoitWolf ? "" : $"_cov-{CovarianceFactory.Tag(Covariance)}") +
        (Weighting  == TeWeighting.Dollars          ? "" : "_te-names");
}
