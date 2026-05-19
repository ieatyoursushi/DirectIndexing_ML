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
            var json   = File.ReadAllText(filePath);
            var arr    = JsonSerializer.Deserialize<DailyPrice[]>(json, JsonOpts);
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
