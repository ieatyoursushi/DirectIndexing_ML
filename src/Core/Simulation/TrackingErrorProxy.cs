namespace DirectIndexing.Core.Simulation;

/// <summary>
/// v0.1 proxy for annualised tracking error σ_TE.
///
/// σ_TE ≈ std(r_portfolio − r_benchmark) over trailing <see cref="Window"/> days × √252
///
/// Benchmark: equal-weighted daily return across all S&amp;P 500 tickers in PriceLoader.
/// Full quadratic form σ_TE = √(δw⊤ Σ δw) deferred to v0.2 (needs 503×503 covariance).
/// </summary>
public sealed class TrackingErrorProxy
{
    private readonly PriceLoader _prices;

    private const int Window = 30;

    private readonly float[] _portHistory  = new float[Window];
    private readonly float[] _benchHistory = new float[Window];
    private int  _head    = 0;
    private int  _filled  = 0;

    public TrackingErrorProxy(PriceLoader prices) => _prices = prices;

    /// <summary>
    /// Call once per day with the symbols of currently open lots.
    /// Returns annualised tracking error in [0, ∞).
    ///
    /// Portfolio return = equal-weighted return over open-lot symbols.
    /// This avoids spurious jumps from lots being added/removed at structural events
    /// (harvest + wash-sale reopen), which would inflate σ_TE by orders of magnitude
    /// if computed from raw dollar-value changes.
    /// </summary>
    public float Update(IEnumerable<string> openSymbols, int dayIndex)
    {
        // Portfolio: equal-weighted return of currently open lots
        float portSum = 0f;
        int   portN   = 0;
        foreach (var sym in openSymbols)
        {
            float r = _prices.DailyReturn(sym, dayIndex);
            if (!float.IsNaN(r)) { portSum += r; portN++; }
        }
        float portRet = portN > 0 ? portSum / portN : 0f;

        // Benchmark: equal-weighted average of all available tickers at this day
        float benchSum = 0f;
        int   benchN   = 0;
        foreach (var sym in _prices.Symbols)
        {
            float r = _prices.DailyReturn(sym, dayIndex);
            if (!float.IsNaN(r)) { benchSum += r; benchN++; }
        }
        float benchRet = benchN > 0 ? benchSum / benchN : 0f;

        // Rolling window update (circular buffer)
        _portHistory[_head]  = portRet;
        _benchHistory[_head] = benchRet;
        _head = (_head + 1) % Window;
        if (_filled < Window) _filled++;

        if (_filled < 5) return 0f;   // not enough history yet

        // std(r_port − r_bench) annualised
        float diffSum = 0f, diffSumSq = 0f;
        for (int i = 0; i < _filled; i++)
        {
            float d = _portHistory[i] - _benchHistory[i];
            diffSum   += d;
            diffSumSq += d * d;
        }
        float mean     = diffSum   / _filled;
        float variance = diffSumSq / _filled - mean * mean;
        float dailyStd = MathF.Sqrt(MathF.Max(variance, 0f));

        return dailyStd * MathF.Sqrt(252f);   // annualise
    }
}
