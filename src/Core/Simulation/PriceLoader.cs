using System.Text.Json;
using DirectIndexing.DataCollection;

namespace DirectIndexing.Core.Simulation;

/// <summary>
/// Loads all S&amp;P 500 price series from data/raw/*.json, aligns them to a shared
/// trading calendar, and pre-computes feature arrays so the simulation engine
/// can do O(1) lookups at each (symbol, dayIndex) during the day loop.
///
/// Memory layout: one float[] of length DayCount per feature per ticker.
/// Missing data (ticker not listed on a given trading date) → float.NaN.
/// Pre-computed features: close, daily return, range volatility, MA-50, MA-200.
/// </summary>
public sealed class PriceLoader
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true
    };

    // ── Aligned price arrays [dayIndex] ─────────────────────────────────────
    private readonly Dictionary<string, float[]> _close    = new();
    private readonly Dictionary<string, float[]> _high     = new();
    private readonly Dictionary<string, float[]> _low      = new();
    private readonly Dictionary<string, float[]> _return   = new();   // r_t
    private readonly Dictionary<string, float[]> _rangeVol = new();   // (H-L)/P_{t-1}
    private readonly Dictionary<string, float[]> _ma50     = new();
    private readonly Dictionary<string, float[]> _ma200    = new();

    // ── Calendar ─────────────────────────────────────────────────────────────
    private readonly List<DateOnly> _calendar = new();
    private readonly Dictionary<DateOnly, int> _dateToIndex = new();

    // ── Constituent metadata (symbol → sector) ───────────────────────────────
    private readonly Dictionary<string, string> _sector = new();
    private readonly Dictionary<string, decimal> _weight = new();

    /// <summary>Number of trading days in the warm-up period (MA_200 requirement).</summary>
    public const int WarmupDays = 200;

    public IReadOnlyList<DateOnly>       Calendar   => _calendar;
    public int                           DayCount   => _calendar.Count;
    public IReadOnlyCollection<string>   Symbols    => _close.Keys;

    // ── Load ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Loads all JSON price files from rawDataDir and the optional constituents file.
    /// Call once before RunAsync().
    /// </summary>
    // [math:price_world] — DataMemo/spec/SymbolTable.md
    public void Load(string rawDataDir, string? constituentsFile = null)
    {
        LoadConstituents(constituentsFile);
        LoadPrices(rawDataDir);
    }

    private void LoadConstituents(string? path)
    {
        if (path is null || !File.Exists(path)) return;
        var json   = File.ReadAllText(path);
        var items  = JsonSerializer.Deserialize<SP500Constituent[]>(json, JsonOpts) ?? [];
        foreach (var c in items)
        {
            _sector[c.Symbol] = c.Sector;
            _weight[c.Symbol] = c.Weight;
        }
    }

    private void LoadPrices(string rawDataDir)
    {
        // ── Phase 1: deserialise all JSON files ──────────────────────────────
        var allPrices = new Dictionary<string, DailyPrice[]>(600);

        foreach (var filePath in Directory.EnumerateFiles(rawDataDir, "*.json"))
        {
            var symbol = Path.GetFileNameWithoutExtension(filePath);
            DailyPrice[]? arr;
            try
            {
                var json = File.ReadAllText(filePath);
                arr = JsonSerializer.Deserialize<DailyPrice[]>(json, JsonOpts);
            }
            catch (JsonException ex)
            {
                Console.WriteLine($"[WARN] Skipping corrupted price file {symbol}.json: {ex.Message}");
                continue;
            }

            if (arr is null || arr.Length == 0) continue;
            // FMP returns newest-first; sort ascending for array index alignment
            Array.Sort(arr, (a, b) => a.Date.CompareTo(b.Date));
            allPrices[symbol] = arr;
        }

        // ── Phase 2: build shared trading calendar ────────────────────────────
        var dateSet = new SortedSet<DateOnly>();
        foreach (var arr in allPrices.Values)
            foreach (var p in arr)
                dateSet.Add(p.Date);

        _calendar.AddRange(dateSet);
        for (int i = 0; i < _calendar.Count; i++)
            _dateToIndex[_calendar[i]] = i;

        int N = _calendar.Count;

        // ── Phase 3: fill aligned arrays and pre-compute features ─────────────
        foreach (var (symbol, prices) in allPrices)
        {
            var closes = new float[N]; Array.Fill(closes, float.NaN);
            var highs  = new float[N]; Array.Fill(highs,  float.NaN);
            var lows   = new float[N]; Array.Fill(lows,   float.NaN);

            foreach (var p in prices)
            {
                int idx  = _dateToIndex[p.Date];
                closes[idx] = (float)p.Close;
                highs[idx]  = (float)p.High;
                lows[idx]   = (float)p.Low;
            }

            // Daily return r_t = (close_t − close_{t−1}) / close_{t−1}
            var ret = new float[N];
            ret[0] = float.NaN;
            for (int t = 1; t < N; t++)
                ret[t] = IsValid(closes[t]) && IsValid(closes[t - 1]) && closes[t - 1] > 0f
                    ? (closes[t] - closes[t - 1]) / closes[t - 1]
                    : float.NaN;

            // Range volatility (H_t − L_t) / close_{t−1}
            var rv = new float[N];
            rv[0] = float.NaN;
            for (int t = 1; t < N; t++)
                rv[t] = IsValid(highs[t]) && IsValid(lows[t]) && IsValid(closes[t - 1]) && closes[t - 1] > 0f
                    ? (highs[t] - lows[t]) / closes[t - 1]
                    : float.NaN;

            _close[symbol]    = closes;
            _high[symbol]     = highs;
            _low[symbol]      = lows;
            _return[symbol]   = ret;
            _rangeVol[symbol] = rv;
            _ma50[symbol]     = ComputeMA(closes, 50);
            _ma200[symbol]    = ComputeMA(closes, 200);
        }

        if (N <= WarmupDays)
            throw new InvalidOperationException(
                $"[PriceLoader] Only {N} calendar days loaded but {WarmupDays} are needed " +
                $"for warmup alone (MA-200). Download a wider date range — the simulation " +
                $"needs at least {WarmupDays + 1} trading days to produce any snapshots.");

        Console.WriteLine(
            $"[PriceLoader] {_close.Count} tickers loaded, {N} calendar days. " +
            $"Simulation starts at day {WarmupDays} ({GetDate(WarmupDays)}).");
    }

    // ── Test factory ─────────────────────────────────────────────────────────

    /// <summary>
    /// Creates a PriceLoader pre-populated with synthetic return data — for unit tests only.
    ///
    /// Only <c>_close</c> and <c>_return</c> are populated (TrackingErrorProxy and
    /// SoftLabelBuilder only need <see cref="DailyReturn"/> and <see cref="Symbols"/>).
    /// The calendar is a sequential run of dates starting 2020-01-01.
    ///
    /// Convention: <c>dailyReturns[sym][0]</c> should be <c>float.NaN</c> (no return
    /// on day 0, same as production data where r_0 is undefined).
    /// </summary>
    public static PriceLoader CreateForTesting(Dictionary<string, float[]> dailyReturns)
    {
        var loader = new PriceLoader();
        if (dailyReturns.Count == 0) return loader;

        int N    = dailyReturns.Values.First().Length;
        var date = new DateOnly(2020, 1, 1);
        for (int i = 0; i < N; i++)
        {
            loader._calendar.Add(date);
            loader._dateToIndex[date] = i;
            date = date.AddDays(1);
        }

        foreach (var (sym, ret) in dailyReturns)
        {
            // Reconstruct close prices from returns (base = 100)
            var close = new float[N];
            close[0] = 100f;
            for (int t = 1; t < N; t++)
                close[t] = float.IsNaN(ret[t]) ? close[t - 1] : close[t - 1] * (1f + ret[t]);

            loader._close[sym]    = close;
            loader._return[sym]   = (float[])ret.Clone();   // defensive copy
            loader._high[sym]     = (float[])close.Clone();
            loader._low[sym]      = (float[])close.Clone();
            loader._rangeVol[sym] = new float[N];
            loader._ma50[sym]     = new float[N];
            loader._ma200[sym]    = new float[N];
        }
        return loader;
    }

    // ── Synthetic GBM world (the second price source) ─────────────────────────

    /// <summary>
    /// Synthetic FHS world (v0.3-10, the σ̂ environment role R3 of
    /// DataMemo/decisions/VolatilityModel_v03.md §8) — the RL episode generator with volatility
    /// clustering. From a SOURCE history (the real cache, or any world):
    ///   1. per name, GARCH(1,1) by QMLE on the source's full history (allowed here: this is an
    ///      environment, not a feature — nothing downstream is evaluated on the source), falling
    ///      back to (α, β) = (0.08, 0.90) and the sample variance below 500 returns;
    ///   2. standardized residuals ε_{i,s} = r_{i,s}/σ_{i,s} through that filter, centered per name;
    ///   3. each simulated day draws ONE historical date τ and uses the whole cross-sectional
    ///      vector ε_{·,τ} (a name missing on τ draws from its own pool) — this preserves the
    ///      contemporaneous correlation without estimating Σ̂;
    ///   4. each name's σ evolves by its GARCH recursion driven by its own simulated return, and
    ///      the log-price steps σε − σ²/2.
    /// Deterministic in <paramref name="seed"/>. RangeVol is the current σ·√(4/π) (no intraday path).
    /// </summary>
    // [math:fhs_world] — DataMemo/spec/SymbolTable.md
    public static PriceLoader FromFhs(PriceLoader source, int days, int seed, DateOnly? start = null)
    {
        if (days <= WarmupDays)
            throw new ArgumentException($"days must exceed the {WarmupDays}-day warmup.", nameof(days));

        var symbols = source.Symbols.OrderBy(s => s).ToList();
        int N = symbols.Count, T = source.DayCount;
        var omega = new double[N]; var alpha = new double[N]; var beta = new double[N]; var h0 = new double[N];
        var eps = new double[N][];
        var pools = new List<double>[N];
        for (int i = 0; i < N; i++)
        {
            var r = source.GetReturnArray(symbols[i]);
            var valid = r.Where(x => !float.IsNaN(x)).Select(x => (double)x).ToList();
            double s2 = valid.Count > 1 ? valid.Average(x => x * x) : 1e-4;
            double a = 0.08, b = 0.90;
            if (valid.Count >= Volatility.Garch11Vol.MinObs)
            {
                var (p, _) = Volatility.Garch11Vol.Fit(r, 0, T - 1, origin: T - 1);
                a = p.Alpha; b = p.Beta; s2 = p.LongRun;
            }
            omega[i] = s2 * (1 - a - b); alpha[i] = a; beta[i] = b; h0[i] = s2;

            eps[i] = new double[T];
            pools[i] = new List<double>();
            double h = s2;
            for (int s = 0; s < T; s++)
            {
                float x = r[s];
                if (float.IsNaN(x)) { eps[i][s] = double.NaN; continue; }
                eps[i][s] = x / Math.Sqrt(h);
                pools[i].Add(eps[i][s]);
                h = omega[i] + a * (double)x * x + b * h;
            }
            double m = pools[i].Count > 0 ? pools[i].Average() : 0;
            for (int s = 0; s < T; s++) if (!double.IsNaN(eps[i][s])) eps[i][s] -= m;
            for (int k = 0; k < pools[i].Count; k++) pools[i][k] -= m;
        }
        // candidate dates: at least half the names observed
        var dates = Enumerable.Range(0, T)
            .Where(s => Enumerable.Range(0, N).Count(i => !double.IsNaN(eps[i][s])) * 2 >= N).ToArray();
        if (dates.Length == 0) throw new InvalidOperationException("FHS source has no usable dates");

        var loader = new PriceLoader();
        var date = start ?? new DateOnly(2000, 1, 3);
        while (loader._calendar.Count < days)
        {
            if (date.DayOfWeek is not (DayOfWeek.Saturday or DayOfWeek.Sunday))
            {
                loader._dateToIndex[date] = loader._calendar.Count;
                loader._calendar.Add(date);
            }
            date = date.AddDays(1);
        }

        const float RangeScale = 1.1284f;   // √(4/π)
        var rng = new Random(seed);
        var close = new float[N][]; var ret = new float[N][]; var rv = new float[N][];
        var hNow = (double[])h0.Clone();
        for (int i = 0; i < N; i++)
        {
            close[i] = new float[days]; ret[i] = new float[days]; rv[i] = new float[days];
            close[i][0] = 100f; ret[i][0] = rv[i][0] = float.NaN;
        }
        for (int t = 1; t < days; t++)
        {
            int tau = dates[rng.Next(dates.Length)];
            for (int i = 0; i < N; i++)
            {
                double e = eps[i][tau];
                if (double.IsNaN(e)) e = pools[i].Count > 0 ? pools[i][rng.Next(pools[i].Count)] : GbmSimulator.NextGaussian(rng);
                double sig = Math.Sqrt(hNow[i]), rStar = sig * e;
                close[i][t] = (float)(close[i][t - 1] * Math.Exp(rStar - 0.5 * hNow[i]));
                ret[i][t]   = (close[i][t] - close[i][t - 1]) / close[i][t - 1];
                rv[i][t]    = (float)sig * RangeScale;
                hNow[i]     = omega[i] + alpha[i] * rStar * rStar + beta[i] * hNow[i];
            }
        }
        for (int i = 0; i < N; i++)
        {
            var sym = symbols[i];
            loader._close[sym]    = close[i];
            loader._high[sym]     = (float[])close[i].Clone();
            loader._low[sym]      = (float[])close[i].Clone();
            loader._return[sym]   = ret[i];
            loader._rangeVol[sym] = rv[i];
            loader._ma50[sym]     = ComputeMA(close[i], 50);
            loader._ma200[sym]    = ComputeMA(close[i], 200);
            loader._sector[sym]   = source.GetSector(sym);
        }
        Console.WriteLine($"[PriceLoader] synthetic FHS world: {N} names, {days} trading days, " +
                          $"{dates.Length} source dates, seed={seed}.");
        return loader;
    }

    /// <summary>
    /// A data-free CLUSTERED source for <see cref="FromFhs"/> (smoke tests, standalone runs): a common
    /// GARCH(1,1) market factor (ω = 2e-6, α = 0.10, β = 0.88) loading 0.9 on every name, plus
    /// idiosyncratic Gaussian noise (σ = 0.8%/day). Deterministic in the seed. Not a world to train
    /// on — a stand-in for the real cache when it is absent.
    /// </summary>
    public static PriceLoader GarchFactorPanel(int names, int days, int seed)
    {
        var rng = new Random(seed);
        var f = new double[days];
        double h = 1e-4;
        for (int t = 1; t < days; t++)
        {
            f[t] = Math.Sqrt(h) * GbmSimulator.NextGaussian(rng);
            h = 2e-6 + 0.10 * f[t] * f[t] + 0.88 * h;
        }
        var d = new Dictionary<string, float[]>();
        for (int i = 0; i < names; i++)
        {
            var r = new float[days]; r[0] = float.NaN;
            for (int t = 1; t < days; t++) r[t] = (float)(0.9 * f[t] + 0.008 * GbmSimulator.NextGaussian(rng));
            d[$"F{i:D3}"] = r;
        }
        return CreateForTesting(d);
    }


    /// <summary>
    /// Builds a fully synthetic price world: each name follows an independent GBM
    /// S_{t+1} = S_t · exp((μ − σ²/2)Δ + σ√Δ Z), Δ = 1/252, S_0 = 100, on a weekday
    /// calendar. The result is an ordinary <see cref="PriceLoader"/>, so the one
    /// canonical <see cref="SimulationEngine"/> runs on it unchanged — replacing the
    /// former MonteCarloEngine, a duplicated day loop that had drifted from the real
    /// engine (see <c>DataMemo/archive/RetiredComponents.md</c> §8).
    ///
    /// Features: returns and MA-50/200 are computed from the synthetic closes exactly
    /// as for real data. There is no intraday path, so RangeVol uses the proxy
    /// σ_daily·√(4/π) (twice E|Z| under a Brownian-bridge approximation), and
    /// high = low = close. Because the forward prices are real prices *of this world*,
    /// Y_Soft_BT is defined here too.
    /// </summary>
    /// <param name="universe">Names, sectors and annualised σ per name (see <see cref="CalibrateGbmUniverse"/>).</param>
    /// <param name="days">Calendar length in trading days; the engine starts at <see cref="WarmupDays"/>.</param>
    /// <param name="seed">RNG seed — the world is a deterministic function of (universe, days, seed, drift).</param>
    /// <param name="annualDrift">μ, annualised (default 0).</param>
    /// <param name="start">First calendar date (default 2000-01-03, a Monday).</param>
    // [math:price_world] — DataMemo/spec/SymbolTable.md
    public static PriceLoader FromGbm(
        IReadOnlyList<(string Symbol, string Sector, float AnnualSigma)> universe,
        int days, int seed, float annualDrift = 0f, DateOnly? start = null)
    {
        if (days <= WarmupDays)
            throw new ArgumentException($"days must exceed the {WarmupDays}-day warmup.", nameof(days));

        var loader = new PriceLoader();

        var date = start ?? new DateOnly(2000, 1, 3);
        while (loader._calendar.Count < days)
        {
            if (date.DayOfWeek is not (DayOfWeek.Saturday or DayOfWeek.Sunday))
            {
                loader._dateToIndex[date] = loader._calendar.Count;
                loader._calendar.Add(date);
            }
            date = date.AddDays(1);
        }

        const float Dt = 1f / 252f;
        const float RangeScale = 1.1284f;   // √(4/π)
        var rng = new Random(seed);

        foreach (var (sym, sector, sigma) in universe)
        {
            float ds    = sigma * MathF.Sqrt(Dt);                       // σ√Δ
            float drift = (annualDrift - 0.5f * sigma * sigma) * Dt;    // (μ − σ²/2)Δ

            var close = new float[days];
            close[0] = 100f;
            for (int t = 1; t < days; t++)
                close[t] = close[t - 1] * MathF.Exp(drift + ds * GbmSimulator.NextGaussian(rng));

            var ret = new float[days];
            var rv  = new float[days];
            ret[0] = rv[0] = float.NaN;
            float dailySigma = sigma * MathF.Sqrt(Dt);
            for (int t = 1; t < days; t++)
            {
                ret[t] = (close[t] - close[t - 1]) / close[t - 1];
                rv[t]  = dailySigma * RangeScale;
            }

            loader._close[sym]    = close;
            loader._high[sym]     = (float[])close.Clone();
            loader._low[sym]      = (float[])close.Clone();
            loader._return[sym]   = ret;
            loader._rangeVol[sym] = rv;
            loader._ma50[sym]     = ComputeMA(close, 50);
            loader._ma200[sym]    = ComputeMA(close, 200);
            loader._sector[sym]   = sector;
        }

        Console.WriteLine(
            $"[PriceLoader] synthetic GBM world: {universe.Count} names, {days} trading days " +
            $"({loader._calendar[0]} → {loader._calendar[^1]}), seed={seed}, μ={annualDrift}.");
        return loader;
    }

    /// <summary>
    /// Per-name σ for a synthetic world, calibrated from a real loader: annualised
    /// trailing-<paramref name="window"/>-day realised volatility at the last available
    /// day, falling back to <paramref name="fallbackSigma"/> with fewer than 5 valid returns.
    /// </summary>
    public static List<(string Symbol, string Sector, float AnnualSigma)> CalibrateGbmUniverse(
        PriceLoader real, int window = 60, float fallbackSigma = 0.20f)
    {
        int tLast = real.DayCount - 1;
        var universe = new List<(string, string, float)>();
        foreach (var sym in real.Symbols.OrderBy(s => s, StringComparer.Ordinal))
        {
            var ret = real.GetReturnArray(sym);
            int count = 0; double sum = 0, sumSq = 0;
            for (int i = Math.Max(0, tLast - window); i <= tLast; i++)
            {
                if (float.IsNaN(ret[i])) continue;
                sum += ret[i]; sumSq += (double)ret[i] * ret[i]; count++;
            }
            float sigma = fallbackSigma;
            if (count >= 5)
            {
                double mean = sum / count;
                double s    = Math.Sqrt(Math.Max(sumSq / count - mean * mean, 0.0) * 252.0);
                if (s > 0 && !double.IsNaN(s)) sigma = (float)s;
            }
            universe.Add((sym, real.GetSector(sym), sigma));
        }
        return universe;
    }

    /// <summary>A uniform synthetic universe — no real data required (smoke tests, stress runs).</summary>
    public static List<(string Symbol, string Sector, float AnnualSigma)> UniformGbmUniverse(
        int names, float annualSigma = 0.25f) =>
        Enumerable.Range(0, names)
            .Select(i => ($"SYN{i:D3}", $"Sector{i % 11:D2}", annualSigma))
            .ToList();

    // ── Lookup API ────────────────────────────────────────────────────────────

    public float   GetClose(string symbol, int t)     => _close[symbol][t];
    public float   DailyReturn(string symbol, int t)  => _return[symbol][t];
    public float   RangeVol(string symbol, int t)     => _rangeVol[symbol][t];

    public float DeviationFromMA(string symbol, int t, int period)
    {
        var ma = period switch { 50 => _ma50[symbol][t], 200 => _ma200[symbol][t],
                                 _ => throw new ArgumentOutOfRangeException(nameof(period)) };
        return float.IsNaN(ma) || ma == 0f ? float.NaN : (_close[symbol][t] - ma) / ma;
    }

    public bool HasData(string symbol, int t) =>
        _close.TryGetValue(symbol, out var arr) && t < arr.Length && IsValid(arr[t]);

    public DateOnly  GetDate(int t)          => _calendar[t];

    /// <summary>
    /// The first trading-day index whose date is on or after <paramref name="date"/>
    /// (<see cref="DayCount"/> if none) — how calendar-day rules such as §1091's
    /// 31-day reopen are mapped onto the trading calendar.
    /// </summary>
    public int FirstIndexOnOrAfter(DateOnly date)
    {
        int lo = 0, hi = _calendar.Count;
        while (lo < hi)
        {
            int mid = (lo + hi) >>> 1;
            if (_calendar[mid] < date) lo = mid + 1; else hi = mid;
        }
        return lo;
    }
    public string    GetSector(string symbol) => _sector.GetValueOrDefault(symbol, "");
    public decimal   GetWeight(string symbol) => _weight.GetValueOrDefault(symbol, 0m);

    /// <summary>All close prices at dayIndex as a decimal map — used for portfolio valuation.</summary>
    public Dictionary<string, decimal> GetClosesDecimal(int t)
    {
        var result = new Dictionary<string, decimal>(_close.Count);
        foreach (var (sym, arr) in _close)
            if (IsValid(arr[t]))
                result[sym] = (decimal)arr[t];
        return result;
    }

    /// <summary>Raw return array — used by SoftLabelBuilder and TrackingErrorProxy.</summary>
    public float[] GetReturnArray(string symbol)  => _return[symbol];
    public float[] GetCloseArray(string symbol)   => _close[symbol];

    // ── Private helpers ───────────────────────────────────────────────────────

    private static bool IsValid(float v) => !float.IsNaN(v) && !float.IsInfinity(v);

    private static float[] ComputeMA(float[] closes, int period)
    {
        int     N       = closes.Length;
        float[] ma      = new float[N];
        Array.Fill(ma, float.NaN);
        double  sum     = 0;
        int     valid   = 0;

        for (int t = 0; t < N; t++)
        {
            if (IsValid(closes[t])) { sum += closes[t]; valid++; }

            if (t >= period && IsValid(closes[t - period]))
            {
                sum -= closes[t - period];
                valid--;
            }

            if (t >= period - 1 && valid == period)
                ma[t] = (float)(sum / period);
        }
        return ma;
    }
}
