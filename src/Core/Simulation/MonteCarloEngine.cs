using DirectIndexing.Core.Oracle;
using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Simulation;

/// <summary>
/// A fully synthetic simulation engine that replaces historical prices with
/// per-stock GBM price paths.
///
/// When to use Monte Carlo vs backtesting:
///
///   • Backtesting (<see cref="SimulationEngine"/>) is the primary training data
///     source.  It uses real historical S&amp;P 500 prices (2020-2024) so the feature
///     distributions are empirically grounded.  Use it whenever price data is available.
///
///   • Monte Carlo (<see cref="MonteCarloEngine"/>) is useful for:
///       – Data augmentation: generate additional training rows with different
///         volatility regimes or starting portfolio configurations.
///       – Stress testing: run with inflated σ to see oracle behaviour in
///         high-volatility environments not covered by the historical window.
///       – Unit testing: exercise the full pipeline deterministically with a
///         fixed RNG seed without a dependency on the data/ directory.
///       – Forward-looking scenario analysis: simulate a future period with
///         calibrated but not yet realised σ estimates.
///
///   Note: Y_Soft_BT is always NaN in Monte Carlo mode because there are no
///   real forward prices to evaluate the backtesting soft label against.
///
/// Price generation:
///   Each ticker is simulated as an independent GBM from a starting price of 100.
///   When constructed from a <see cref="PriceLoader"/>, per-stock σ is calibrated
///   from the trailing 60-day realised volatility.  In the standalone constructor,
///   all tickers share <paramref name="globalSigma"/>.
///
/// Day loop structure mirrors <see cref="SimulationEngine"/>:
///   warmupDays … simDays−1: extract features, call oracle, harvest, reopen, advance.
///   Year-end reset every 252 days.  G_YTD seeded at 5% of initial portfolio value.
/// </summary>
public sealed class MonteCarloEngine
{
    // ── Universe and vol surface ──────────────────────────────────────────────

    private readonly List<(string Symbol, string Sector)> _universe;
    private readonly Dictionary<string, float>            _annualSigma = new();
    private readonly float                                _annualDrift;

    // ── GBM simulator for Y_Soft_GBM labels ──────────────────────────────────

    private static readonly GbmSimulator _labelSim = new(paths: 200, horizon: 30);

    // ── Constants ─────────────────────────────────────────────────────────────

    private const float StartPrice   = 100f;
    private const float TradingDays  = 252f;
    private const int   CalVolWindow = 60;      // trailing days for σ calibration
    private const float DefaultSigma = 0.20f;   // fallback when calibration unavailable
    private const float SeedFraction = 0.05f;   // G_YTD seed as fraction of portfolio value

    // ── Standalone constructor ─────────────────────────────────────────────────

    /// <summary>
    /// Standalone constructor — no real price data required.
    /// All tickers are simulated with the same <paramref name="globalSigma"/>.
    /// </summary>
    /// <param name="universe">
    /// The set of (Symbol, Sector) pairs that make up the simulated universe.
    /// </param>
    /// <param name="globalSigma">
    /// Annualised volatility applied to every ticker (default 0.20 = 20%).
    /// </param>
    /// <param name="annualDrift">
    /// Annualised drift μ passed through to <see cref="GbmSimulator"/>
    /// (default 0 = risk-neutral / zero-mean log returns).
    /// </param>
    public MonteCarloEngine(
        IEnumerable<(string Symbol, string Sector)> universe,
        float globalSigma = DefaultSigma,
        float annualDrift = 0f)
    {
        _universe    = universe.ToList();
        _annualDrift = annualDrift;

        foreach (var (sym, _) in _universe)
            _annualSigma[sym] = globalSigma;
    }

    // ── Calibrated constructor ────────────────────────────────────────────────

    /// <summary>
    /// Calibrated constructor — derives per-stock σ from the trailing
    /// 60-day realised volatility in <paramref name="prices"/>.
    ///
    /// The calibration reads from the last available day index in
    /// <paramref name="prices"/> (i.e. <c>prices.DayCount − 1</c>).
    /// For tickers with fewer than 5 valid returns in the window, σ falls
    /// back to 0.20.
    ///
    /// The universe is derived from <c>prices.Symbols</c> with sector metadata
    /// from <c>prices.GetSector</c>.
    /// </summary>
    /// <param name="prices">Loaded <see cref="PriceLoader"/> instance.</param>
    /// <param name="annualDrift">Annualised drift μ (default 0).</param>
    public MonteCarloEngine(PriceLoader prices, float annualDrift = 0f)
    {
        _annualDrift = annualDrift;

        _universe = prices.Symbols
            .Select(s => (s, prices.GetSector(s)))
            .ToList();

        int tLast = prices.DayCount - 1;

        foreach (var (sym, _) in _universe)
        {
            var    ret   = prices.GetReturnArray(sym);
            int    start = Math.Max(0, tLast - CalVolWindow);
            int    count = 0;
            double sum   = 0, sumSq = 0;

            for (int i = start; i <= tLast; i++)
            {
                float r = ret[i];
                if (float.IsNaN(r)) continue;
                sum   += r;
                sumSq += (double)r * r;
                count++;
            }

            float annSig = DefaultSigma;
            if (count >= 5)
            {
                double mean     = sum / count;
                double variance = sumSq / count - mean * mean;
                double dailyStd = Math.Sqrt(Math.Max(variance, 0.0));
                annSig = (float)(dailyStd * Math.Sqrt(TradingDays));
                if (annSig <= 0f || float.IsNaN(annSig)) annSig = DefaultSigma;
            }

            _annualSigma[sym] = annSig;
        }
    }

    // ── Public entry point ────────────────────────────────────────────────────

    /// <summary>
    /// Runs the full Monte Carlo simulation and returns a list of
    /// <see cref="LotStateVector"/> snapshots, one per (lot, day) observation.
    ///
    /// Day loop structure:
    ///   1. Look up simulated close prices at day t.
    ///   2. Compute portfolio value and σ_TE proxy.
    ///   3. For each open lot: extract features → call OracleBoundary → if fires, harvest.
    ///   4. Process reopen queue.
    ///   5. Advance wash-sale clocks; year-end reset every 252 days.
    /// </summary>
    /// <param name="initialPortfolioValue">Total dollars invested at simulation start.</param>
    /// <param name="simDays">
    /// Total length of each simulated price series in trading days.
    /// Must be greater than <paramref name="warmupDays"/>.
    /// </param>
    /// <param name="warmupDays">
    /// Number of leading days used to warm up MA windows; the day loop starts
    /// after this point so the MA-200 feature is available from day 1.
    /// </param>
    /// <param name="seed">RNG seed for reproducibility (default 42).</param>
    /// <returns>List of feature snapshots with Y_Soft_GBM filled and Y_Soft_BT = NaN.</returns>
    public List<LotStateVector> Run(
        decimal initialPortfolioValue = 10_000_000m,
        int     simDays               = 504,
        int     warmupDays            = 200,
        int     seed                  = 42)
    {
        if (simDays <= warmupDays)
            throw new ArgumentException("simDays must be greater than warmupDays.");

        var rng = new Random(seed);

        // ── Step 1: generate synthetic price matrix ───────────────────────────
        Console.WriteLine($"[MonteCarloEngine] Generating {_universe.Count} GBM price series " +
                          $"({simDays} days, seed={seed}) …");

        var closes = GeneratePrices(simDays, rng);   // sym → float[simDays]

        // ── Step 2: pre-compute derived feature arrays ────────────────────────
        var returns  = ComputeReturns(closes, simDays);
        var rangeVol = ComputeRangeVol(simDays);        // E|N(0,σ)| proxy, no intraday sim needed
        var ma50     = ComputeMA(closes, 50,  simDays);
        var ma200    = ComputeMA(closes, 200, simDays);

        // ── Step 3: initialise portfolio state ────────────────────────────────
        var state       = new PortfolioState();
        var lotCount    = new Dictionary<string, int>();
        var reopenQueue = new Dictionary<int, List<(string Symbol, string Sector, decimal Dollars)>>();
        var snapshots   = new List<LotStateVector>(128_000);

        decimal seedAmount = InitialisePortfolio(
            state, lotCount, closes, warmupDays, initialPortfolioValue);

        // Rolling σ_TE (30-day circular buffer; no PriceLoader dependency)
        var te = new SigmaTeBuffer();
        var allSymbols = _universe.Select(u => u.Symbol).ToList();

        // ── Step 4: day loop ──────────────────────────────────────────────────
        for (int t = warmupDays; t < simDays; t++)
        {
            ProcessDay(
                t, state, lotCount, reopenQueue, snapshots,
                closes, returns, rangeVol, ma50, ma200,
                te, allSymbols, simDays);

            // Year-end reset every 252 trading days counted from warmupDays
            if ((t - warmupDays + 1) % (int)TradingDays == 0)
            {
                state.ResetForNewYear();
                state.SeedGYTD(seedAmount);
                Console.WriteLine($"  [MC] Year-end reset at day {t} — G_YTD re-seeded to {seedAmount:C0}");
            }

            if (t % 50 == 0)
                Console.WriteLine($"  [MC] Day {t}/{simDays - 1}  " +
                                  $"open={state.OpenLots.Count}  " +
                                  $"snapshots={snapshots.Count}  " +
                                  $"G_YTD={state.G_YTD:F0}");
        }

        Console.WriteLine($"[MonteCarloEngine] Complete. Total snapshots: {snapshots.Count}");
        return snapshots;
    }

    // ── Private: price generation ─────────────────────────────────────────────

    private Dictionary<string, float[]> GeneratePrices(int simDays, Random rng)
    {
        var result = new Dictionary<string, float[]>(_universe.Count);

        foreach (var (sym, _) in _universe)
        {
            float sigma = _annualSigma.GetValueOrDefault(sym, DefaultSigma);
            float dt    = 1f / TradingDays;
            float ds    = sigma * MathF.Sqrt(dt);                          // σ√Δ
            float drift = (_annualDrift - 0.5f * sigma * sigma) * dt;     // (μ−σ²/2)Δ

            var arr = new float[simDays];
            arr[0]  = StartPrice;

            for (int t = 1; t < simDays; t++)
            {
                float z = GbmSimulator.NextGaussian(rng);
                arr[t]  = arr[t - 1] * MathF.Exp(drift + ds * z);
            }

            result[sym] = arr;
        }
        return result;
    }

    // ── Private: feature pre-computation ─────────────────────────────────────

    private static Dictionary<string, float[]> ComputeReturns(
        Dictionary<string, float[]> closes, int n)
    {
        var ret = new Dictionary<string, float[]>(closes.Count);
        foreach (var (sym, c) in closes)
        {
            var r = new float[n];
            r[0] = float.NaN;
            for (int t = 1; t < n; t++)
                r[t] = c[t - 1] > 0f ? (c[t] - c[t - 1]) / c[t - 1] : float.NaN;
            ret[sym] = r;
        }
        return ret;
    }

    /// <summary>
    /// Range vol proxy: σ_daily × √(4/π) ≈ 1.1284 × σ_daily.
    ///
    /// For a one-step N(0, σ_daily) draw the expected absolute value is
    /// E|Z| = σ_daily × √(2/π).  The high-low range of a Brownian bridge
    /// over a day is approximately twice that, giving the scale factor √(4/π).
    /// This avoids simulating explicit intraday paths.
    /// </summary>
    private Dictionary<string, float[]> ComputeRangeVol(int n)
    {
        const float ScaleToExpAbsNormal = 1.1284f;   // √(4/π)

        var rv = new Dictionary<string, float[]>(_universe.Count);
        foreach (var (sym, _) in _universe)
        {
            float sigma    = _annualSigma.GetValueOrDefault(sym, DefaultSigma);
            float dailySig = sigma / MathF.Sqrt(TradingDays);

            var arr = new float[n];
            arr[0]  = float.NaN;
            for (int t = 1; t < n; t++)
                arr[t] = dailySig * ScaleToExpAbsNormal;
            rv[sym] = arr;
        }
        return rv;
    }

    private static Dictionary<string, float[]> ComputeMA(
        Dictionary<string, float[]> closes, int period, int n)
    {
        var result = new Dictionary<string, float[]>(closes.Count);
        foreach (var (sym, c) in closes)
        {
            var    ma    = new float[n];
            Array.Fill(ma, float.NaN);
            double sum   = 0;
            int    valid = 0;

            for (int t = 0; t < n; t++)
            {
                if (!float.IsNaN(c[t])) { sum += c[t]; valid++; }
                if (t >= period && !float.IsNaN(c[t - period]))
                {
                    sum -= c[t - period];
                    valid--;
                }
                if (t >= period - 1 && valid == period)
                    ma[t] = (float)(sum / period);
            }
            result[sym] = ma;
        }
        return result;
    }

    // ── Private: portfolio initialisation ────────────────────────────────────

    private decimal InitialisePortfolio(
        PortfolioState          state,
        Dictionary<string, int> lotCount,
        Dictionary<string, float[]> closes,
        int                     day0,
        decimal                 totalValue)
    {
        int n = _universe.Count;
        if (n == 0) throw new InvalidOperationException("Universe is empty.");
        decimal perLot = totalValue / n;

        foreach (var (sym, sector) in _universe)
        {
            float pF = closes[sym][day0];
            if (pF <= 0f || float.IsNaN(pF)) continue;

            decimal dPrice = (decimal)pF;
            int     shares = (int)(perLot / dPrice);
            if (shares == 0) shares = 1;

            state.OpenLot(new Lot(sym, sector, dPrice, shares, day0));
            lotCount[sym] = 1;
        }

        decimal seedAmount = totalValue * (decimal)SeedFraction;
        state.SeedGYTD(seedAmount);

        Console.WriteLine(
            $"[MonteCarloEngine] Portfolio initialised: {state.OpenLots.Count} lots " +
            $"on day {day0}, value ≈ {totalValue:C0}, G_YTD seeded to {seedAmount:C0}");

        return seedAmount;
    }

    // ── Private: day loop ─────────────────────────────────────────────────────

    private void ProcessDay(
        int                     t,
        PortfolioState          state,
        Dictionary<string, int> lotCount,
        Dictionary<int, List<(string Symbol, string Sector, decimal Dollars)>> reopenQueue,
        List<LotStateVector>    snapshots,
        Dictionary<string, float[]> closes,
        Dictionary<string, float[]> returns,
        Dictionary<string, float[]> rangeVol,
        Dictionary<string, float[]> ma50,
        Dictionary<string, float[]> ma200,
        SigmaTeBuffer           te,
        List<string>            allSymbols,
        int                     simDays)
    {
        // Current portfolio value (using simulated closes)
        decimal portValue = state.OpenLots.Sum(lot =>
        {
            float pF = closes[lot.Symbol][t];
            return pF > 0f && !float.IsNaN(pF) ? lot.Shares * (decimal)pF : 0m;
        });
        if (portValue <= 0m) portValue = 1m;

        // σ_TE: rolling 30-day std(r_port − r_bench) × √252
        float sigmaTE = te.Update(
            state.OpenLots.Select(l => l.Symbol).ToList(),
            returns, t, allSymbols);

        // Snapshot + oracle for every open lot (iterate over a copy; harvests mutate the list)
        foreach (var lot in state.OpenLots.ToList())
        {
            float pF = closes[lot.Symbol][t];
            if (float.IsNaN(pF) || pF <= 0f) continue;
            decimal price = (decimal)pF;

            // Pull live state values for this lot before extracting the snapshot
            decimal gYtd      = state.G_YTD;
            int     washClock = state.GetWashClock(lot.Symbol);

            var snap = ExtractSnapshot(
                lot, t, price, portValue, sigmaTE,
                gYtd, washClock,
                returns, rangeVol, ma50, ma200, closes);

            snapshots.Add(snap);

            if (snap.Y_Oracle == 1)
                HarvestLot(lot, price, t, state, lotCount, reopenQueue, simDays);
        }

        // Re-open lots whose 30-day wash-sale window cleared exactly on day t
        if (reopenQueue.TryGetValue(t, out var toReopen))
        {
            foreach (var (sym, sector, dollars) in toReopen)
            {
                float pF = closes[sym][t];
                if (float.IsNaN(pF) || pF <= 0f) continue;
                decimal p  = (decimal)pF;
                int  shares = (int)(dollars / p);
                if (shares == 0) continue;
                state.OpenLot(new Lot(sym, sector, p, shares, t));
                lotCount[sym] = lotCount.GetValueOrDefault(sym) + 1;
            }
            reopenQueue.Remove(t);
        }

        state.AdvanceDay();
    }

    private LotStateVector ExtractSnapshot(
        Lot     lot,
        int     t,
        decimal price,
        decimal portValue,
        float   sigmaTE,
        decimal gYtd,
        int     washClock,
        Dictionary<string, float[]> returns,
        Dictionary<string, float[]> rangeVol,
        Dictionary<string, float[]> ma50,
        Dictionary<string, float[]> ma200,
        Dictionary<string, float[]> closes)
    {
        int     holdingDays = lot.HoldingPeriod(t);
        decimal unrealized  = lot.UnrealizedReturn(price);

        int yOracle = OracleBoundary.Label(
            unrealizedReturn: unrealized,
            sigmaTE:          sigmaTE,
            gYtd:             gYtd,
            washClock:        washClock);

        // Feature scalars
        float rT     = returns[lot.Symbol][t];
        float rvT    = rangeVol[lot.Symbol][t];
        float dMA50  = DeviationFromMA(closes[lot.Symbol][t], ma50[lot.Symbol][t]);
        float dMA200 = DeviationFromMA(closes[lot.Symbol][t], ma200[lot.Symbol][t]);

        // Y_Soft_GBM: fraction of GBM forward paths where OracleBoundary fires first-passage.
        // Frozen state: G_YTD, Sigma_TE, WashClock, and CostBasis are held constant from snapshot.
        float sigma     = _annualSigma.GetValueOrDefault(lot.Symbol, DefaultSigma);
        float costF     = (float)lot.CostBasis;
        float gYtdF     = (float)gYtd;
        float sigTEF    = sigmaTE;
        int   initWash  = washClock;

        float ySoftGbm = _labelSim.FractionFiring(
            startPrice:  closes[lot.Symbol][t],
            annualSigma: sigma,
            firesOnStep: (p, s) =>
            {
                float ell  = costF > 0f ? (p - costF) / costF : 0f;
                int   wash = initWash + s;
                return OracleBoundary.Label((decimal)ell, sigTEF, (decimal)gYtdF, wash) == 1;
            });

        // Approximate days-to-year-end: step within the current 252-day cycle
        int stepInYear = (t % (int)TradingDays);
        int daysToYE   = (int)TradingDays - stepInYear;

        int lotK = 1;   // MC mode: one lot per ticker (no lot-count aggregation)

        return new LotStateVector
        {
            // Lot-level
            L          = (float)unrealized,
            H          = holdingDays,
            S          = lot.IsLongTerm(t) ? 1 : 0,
            B          = (float)lot.CostBasis,
            W          = portValue > 0m ? (float)(lot.Shares * price / portValue) : 0f,
            K          = lotK,

            // Portfolio-level
            G_YTD      = gYtdF,
            Sigma_TE   = sigmaTE,
            WashClock  = washClock,

            // Asset-level
            R_t        = rT,
            SigmaRange = rvT,
            DeltaMA50  = dMA50,
            DeltaMA200 = dMA200,

            // Derived
            TaxAlpha   = ComputeTaxAlpha(lot, price, gYtd, holdingDays),
            DaysToYE   = daysToYE,

            // Labels
            Y_Oracle   = yOracle,
            Y_Soft_GBM = ySoftGbm,
            Y_Soft_BT  = float.NaN,   // not available in MC mode — requires real forward prices

            // Metadata
            Symbol   = lot.Symbol,
            Sector   = lot.Sector,
            Timestep = t
        };
    }

    private static void HarvestLot(
        Lot     lot,
        decimal price,
        int     t,
        PortfolioState          state,
        Dictionary<string, int> lotCount,
        Dictionary<int, List<(string Symbol, string Sector, decimal Dollars)>> reopenQueue,
        int     simDays)
    {
        decimal dollars    = lot.Shares * price;
        int     lotsBefore = lotCount.GetValueOrDefault(lot.Symbol, 1);

        state.HarvestLot(lot, price);
        lotCount[lot.Symbol] = Math.Max(0, lotsBefore - 1);

        int reopenDay = t + OracleBoundary.WashSaleDays;
        if (reopenDay < simDays)
        {
            if (!reopenQueue.TryGetValue(reopenDay, out var list))
                reopenQueue[reopenDay] = list = new();
            list.Add((lot.Symbol, lot.Sector, dollars));
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private static float DeviationFromMA(float price, float ma) =>
        float.IsNaN(ma) || ma == 0f ? float.NaN : (price - ma) / ma;

    private static float ComputeTaxAlpha(Lot lot, decimal price, decimal gYtd, int holdingDays)
    {
        if (gYtd <= 0m) return 0f;
        decimal absLoss = Math.Abs((price - lot.CostBasis) * lot.Shares);
        decimal taxRate = holdingDays >= 365 ? 0.20m : 0.37m;
        return (float)(taxRate * absLoss);
    }

    // ── Private: rolling σ_TE ─────────────────────────────────────────────────

    /// <summary>
    /// Stateful rolling-window σ_TE estimator for MC mode.
    ///
    /// Mirrors <see cref="TrackingErrorProxy"/> but operates on in-memory float[]
    /// arrays rather than a <see cref="PriceLoader"/>, so <see cref="MonteCarloEngine"/>
    /// has no dependency on real price data once the price arrays are generated.
    ///
    /// σ_TE = std(r_portfolio − r_benchmark, window=30) × √252,
    /// where r_benchmark = equal-weight return of all tickers at each step.
    /// </summary>
    private sealed class SigmaTeBuffer
    {
        private const int Window = 30;

        private readonly float[] _portH  = new float[Window];
        private readonly float[] _benchH = new float[Window];
        private int _head   = 0;
        private int _filled = 0;

        /// <summary>
        /// Record returns at timestep <paramref name="t"/> and return the
        /// current annualised tracking-error estimate.
        /// </summary>
        public float Update(
            List<string>                openSymbols,
            Dictionary<string, float[]> returns,
            int                         t,
            List<string>                allSymbols)
        {
            // Equal-weighted portfolio return (open lots only)
            float pSum = 0f; int pN = 0;
            foreach (var sym in openSymbols)
            {
                float r = returns[sym][t];
                if (!float.IsNaN(r)) { pSum += r; pN++; }
            }
            float portRet = pN > 0 ? pSum / pN : 0f;

            // Equal-weighted benchmark return (all tickers in universe)
            float bSum = 0f; int bN = 0;
            foreach (var sym in allSymbols)
            {
                float r = returns[sym][t];
                if (!float.IsNaN(r)) { bSum += r; bN++; }
            }
            float benchRet = bN > 0 ? bSum / bN : 0f;

            // Circular buffer update
            _portH[_head]  = portRet;
            _benchH[_head] = benchRet;
            _head = (_head + 1) % Window;
            if (_filled < Window) _filled++;

            if (_filled < 5) return 0f;   // insufficient history

            // Annualised std of the active difference buffer
            float diffSum = 0f, diffSumSq = 0f;
            for (int i = 0; i < _filled; i++)
            {
                float d = _portH[i] - _benchH[i];
                diffSum   += d;
                diffSumSq += d * d;
            }
            float mean     = diffSum   / _filled;
            float variance = diffSumSq / _filled - mean * mean;
            float dailyStd = MathF.Sqrt(MathF.Max(variance, 0f));
            return dailyStd * MathF.Sqrt(252f);
        }
    }
}
