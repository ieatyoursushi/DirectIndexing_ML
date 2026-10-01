using DirectIndexing.Core.Oracle;
using DirectIndexing.Core.Portfolio;

using DirectIndexing.Core.Simulation.Volatility;

namespace DirectIndexing.Core.Simulation;

/// <summary>Which model-based soft label fills Y_Soft_GBM (v0.3-9): constant-σ GBM paths or FHS paths.</summary>
public enum SoftGbmMode { Gbm, Fhs }

/// <summary>
/// Second-pass labeller — fills Y_Soft_GBM and Y_Soft_BT on each LotStateVector
/// after the main backtesting day loop has set Y_Oracle.
///
/// Both strategies freeze the portfolio state at the snapshot's timestep:
///   - The TaxLedger (G^ST, G^LT, C^ST, C^LT — snapshot.Ledger)
///     and Sigma_TE are held constant (from the snapshot fields).
///   - The wash-sale clock advances by the number of days into the window.
///   - The §1222 holding period is evaluated on the step's CALENDAR date, so τ can
///     flip short→long mid-window (TaxLedger.IsLongTerm).
///   - The cost basis p_k and share count q_k are held constant, so the loss is
///     re-dollarized at each forward price for the scalarized taxValue term.
///
/// The oracle itself stays a black box 𝒳 → {0,1}: swapping its internals
/// (4-gate AND → gates·𝟙[U&gt;0]) changes nothing about the Cesàro-averaging
/// machinery here (GYTD_Redesign_Plan.md v2 §5.1).
///
/// Y_Soft_GBM  — stochastic: 200 GBM paths, σ calibrated from trailing 21-day vol.
/// Y_Soft_BT   — deterministic: fraction of next 30 actual days oracle would fire.
/// </summary>
public sealed class SoftLabelBuilder
{
    private readonly PriceLoader  _prices;
    private readonly OracleConfig _oracle;

    public  const int Window    = 30;   // forward simulation window (trading days) — T_fwd
    private const int VolWindow = 21;   // trailing days for σ estimate

    // Shared GbmSimulator instance — same Paths/Horizon as the old inline constants.
    // Static so all parallel Parallel.For workers share one simulator configuration
    // while each supplies its own thread-local Random instance.
    private static readonly GbmSimulator _gbm = new(paths: 200, horizon: Window);

    private readonly FhsSimulator? _fhs;

    /// <param name="softGbm">Gbm (default, byte-identical) or Fhs — filtered historical simulation
    /// (DataMemo/decisions/VolatilityModel_v03.md §7); FHS falls back to GBM for a snapshot whose
    /// name has no σ̂ forecast or fewer than 20 residuals yet.</param>
    public SoftLabelBuilder(PriceLoader prices, OracleConfig? oracleConfig = null, SoftGbmMode softGbm = SoftGbmMode.Gbm)
    {
        _prices = prices;
        _oracle = oracleConfig ?? OracleConfig.Default;
        if (softGbm == SoftGbmMode.Fhs) _fhs = new FhsSimulator(prices, paths: 200, horizon: Window);
    }

    // ── Public entry point ───────────────────────────────────────────────────

    /// <summary>
    /// Fills Y_Soft_GBM and Y_Soft_BT on every snapshot in place (via record with-expression).
    /// Runs in parallel over snapshots for performance.
    /// </summary>
    public void Label(List<LotStateVector> snapshots)
    {
        Console.WriteLine($"[SoftLabelBuilder] Labelling {snapshots.Count} snapshots " +
                          $"(λ={_oracle.Lambda}, c_trade={_oracle.CTrade}) …");

        Parallel.For(0, snapshots.Count, i =>
        {
            var snap = snapshots[i];
            var (bt30, bt90, taxWeighted) = ComputeBT(snap);
            snapshots[i] = snap with
            {
                Y_Soft_GBM    = ComputeGBM(snap),
                Y_Soft_BT     = bt30,
                Y_Soft_BT_90  = bt90,
                Y_TaxWeighted = taxWeighted,
            };
        });

        Console.WriteLine("[SoftLabelBuilder] Done.");
    }

    // ── Frozen-state oracle step (shared by both strategies) ─────────────────

    /// <summary>
    /// Evaluates the oracle at forward step s under frozen portfolio state:
    /// price is the step's (simulated or historical) close; everything ledger-
    /// and TE-shaped comes from the snapshot; the holding period advances by s
    /// trading days and the §1091 wash clock by the CALENDAR days between t and t+s.
    /// </summary>
    // [math:soft_step] — DataMemo/spec/SymbolTable.md
    private int StepLabel(
        float price, int s, int calendarAhead,
        float costBasis, float shares, DateOnly purchaseDate, DateOnly stepDate, int initClock,
        float sigmaTE, LedgerState frozenLedger) =>
        StepLabel(price, s, calendarAhead, costBasis, shares, purchaseDate, stepDate, initClock,
                  sigmaTE, frozenLedger, out _);

    /// <summary>The step label, also returning the step's TaxValue (for Y_TaxWeighted).</summary>
    private int StepLabel(
        float price, int s, int calendarAhead,
        float costBasis, float shares, DateOnly purchaseDate, DateOnly stepDate, int initClock,
        float sigmaTE, LedgerState frozenLedger, out decimal taxValue)
    {
        float ell = costBasis > 0f ? (price - costBasis) / costBasis : 0f;

        decimal lossDollars = ell < 0f
            ? (decimal)(costBasis - price) * (decimal)shares
            : 0m;
        taxValue = TaxLedger.ComputeTaxValue(
            lossDollars, TaxLedger.IsLongTerm(purchaseDate, stepDate), frozenLedger);

        return OracleBoundary.Label(
            unrealizedReturn: (decimal)ell,
            sigmaTE:          sigmaTE,
            washClock:        initClock + calendarAhead,
            taxValue:         taxValue,
            config:           _oracle);
    }

    // ── GBM soft label ────────────────────────────────────────────────────────

    // [math:y_soft_gbm] — DataMemo/spec/SymbolTable.md
    private float ComputeGBM(LotStateVector snap)
    {
        if (!_prices.HasData(snap.Symbol, snap.Timestep)) return float.NaN;

        float currentClose = _prices.GetClose(snap.Symbol, snap.Timestep);
        if (float.IsNaN(currentClose) || currentClose <= 0f) return float.NaN;

        float annualSigma = EstimateVol(snap.Symbol, snap.Timestep);
        if (float.IsNaN(annualSigma) || annualSigma <= 0f)
            annualSigma = 0.20f;   // fallback: 20% annual vol

        // Frozen state from snapshot — captured by the closure below
        float   sigmaTE   = snap.Sigma_TE;
        int     initClock = snap.WashClock;
        float   costBasis = snap.B;
        float   shares    = snap.Shares;
        var     purchase  = DateOnly.FromDayNumber(snap.PurchaseDayNumber);
        var     t0Date    = _prices.GetDate(snap.Timestep);
        var frozenLedger = snap.Ledger;

        // Delegate path simulation and first-passage counting to GbmSimulator.
        // Per-snapshot Random, deterministically seeded from (Symbol, Timestep) with a
        // stable polynomial hash (string.GetHashCode is randomized per process) so
        // Y_Soft_GBM is reproducible across runs and safe under Parallel.For.
        int seed = snap.Timestep;
        foreach (char c in snap.Symbol) seed = unchecked(seed * 31 + c);
        var rng = new Random(seed);

        bool Fires(float price, int s) =>
            StepLabel(price, s, CalendarDaysAhead(snap.Timestep, s), costBasis, shares, purchase,
                      t0Date.AddDays(CalendarDaysAhead(snap.Timestep, s)), initClock,
                      sigmaTE, frozenLedger) == 1;

        if (_fhs is not null)
        {
            float f = _fhs.FractionFiring(snap.Symbol, snap.Timestep, currentClose, Fires, rng);
            if (!float.IsNaN(f)) return f;
            rng = new Random(seed);   // fallback path: same draws as the GBM arm
        }

        return _gbm.FractionFiring(
            startPrice:  currentClose,
            annualSigma: annualSigma,
            firesOnStep: Fires,
            rng: rng);
    }

    // ── Backtesting soft label ────────────────────────────────────────────────

    /// <summary>Long-horizon variant of the backtest label (#17): 90 trading days.</summary>
    public const int WindowLong = 90;

    /// <summary>
    /// The realized-path label family, one walk forward on the actual prices:
    ///   Y_Soft_BT     — occupation fraction of the next 30 days (NaN if t+30 ≥ T);
    ///   Y_Soft_BT_90  — the same over 90 days (NaN if t+90 ≥ T); 1[·&gt;0] is the 90-day hit;
    ///   Y_TaxWeighted — TaxValue at the FIRST firing step within 30 days, 0 if none (NaN if t+30 ≥ T):
    ///                   the dollar-weighted propensity, a warm start for the v0.4 value function.
    /// Y_Persist (occupation given a hit) is derived, not exported: Y_Soft_BT / 1[Y_Soft_BT &gt; 0].
    /// </summary>
    // [math:y_soft_bt] — DataMemo/spec/SymbolTable.md
    private (float Bt30, float Bt90, float TaxWeighted) ComputeBT(LotStateVector snap)
    {
        int t0   = snap.Timestep;
        int tMax = _prices.DayCount;

        // Not enough forward data — NaN (excluded from training)
        if (t0 + Window >= tMax) return (float.NaN, float.NaN, float.NaN);
        bool long90 = t0 + WindowLong < tMax;

        float   sigmaTE   = snap.Sigma_TE;
        int     initClock = snap.WashClock;
        float   costBasis = snap.B;
        float   shares    = snap.Shares;
        var     purchase  = DateOnly.FromDayNumber(snap.PurchaseDayNumber);
        var     t0Date    = _prices.GetDate(snap.Timestep);
        var frozenLedger = snap.Ledger;

        int days30 = 0, days90 = 0;
        float taxWeighted = 0f;
        bool fired = false;
        int horizon = long90 ? WindowLong : Window;

        for (int s = 1; s <= horizon; s++)
        {
            int t = t0 + s;
            if (!_prices.HasData(snap.Symbol, t)) continue;

            float price = _prices.GetClose(snap.Symbol, t);

            if (StepLabel(price, s, CalendarDaysAhead(t0, s), costBasis, shares, purchase,
                          t0Date.AddDays(CalendarDaysAhead(t0, s)), initClock,
                          sigmaTE, frozenLedger, out decimal tv) == 1)
            {
                days90++;
                if (s <= Window)
                {
                    days30++;
                    if (!fired) { fired = true; taxWeighted = (float)tv; }
                }
            }
        }

        return ((float)days30 / Window, long90 ? (float)days90 / WindowLong : float.NaN, taxWeighted);
    }

    /// <summary>
    /// Calendar days from trading day t to t+s — read off the real calendar where it
    /// exists, and extrapolated at 7 calendar days per 5 trading days past its end
    /// (GBM forward paths near the tail of the data).
    /// </summary>
    private int CalendarDaysAhead(int t, int s) =>
        t + s < _prices.DayCount
            ? _prices.GetDate(t + s).DayNumber - _prices.GetDate(t).DayNumber
            : (_prices.GetDate(_prices.DayCount - 1).DayNumber - _prices.GetDate(t).DayNumber)
              + (7 * (t + s - (_prices.DayCount - 1)) + 4) / 5;

    // ── Trailing volatility estimate ─────────────────────────────────────────

    // [math:sigma_hat_trailing] — DataMemo/spec/SymbolTable.md
    private float EstimateVol(string symbol, int t)
    {
        var returns = _prices.GetReturnArray(symbol);
        int start   = Math.Max(0, t - VolWindow);
        int count   = 0;
        double sum  = 0, sumSq = 0;

        for (int i = start; i < t; i++)
        {
            float r = returns[i];
            if (float.IsNaN(r)) continue;
            sum   += r;
            sumSq += (double)r * r;
            count++;
        }

        if (count < 5) return float.NaN;

        double mean     = sum / count;
        double variance = sumSq / count - mean * mean;
        double dailyStd = Math.Sqrt(Math.Max(variance, 0));

        return (float)(dailyStd * Math.Sqrt(252));   // annualise
    }
}
