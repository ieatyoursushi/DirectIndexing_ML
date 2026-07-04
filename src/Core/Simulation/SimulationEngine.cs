using DirectIndexing.Core.Oracle;
using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Simulation;

/// <summary>
/// The main backtesting simulation engine.
///
/// Day loop (t = WarmupDays … DayCount−1):
///   1. Look up all close prices at day t.
///   2. Compute portfolio value and update σ_TE proxy.
///   3. For each open lot: extract features → call OracleBoundary → if fires, harvest.
///   4. Process reopen queue (lots whose 30-day wash-sale window has expired).
///   5. Advance wash-sale clocks; reset G_YTD at year boundaries.
///
/// Output: List&lt;LotSnapshot&gt; with Y_Soft_GBM = 0 and Y_Soft_BT = 0 as placeholders.
/// SoftLabelBuilder fills those fields in a second pass.
/// </summary>
public sealed class SimulationEngine
{
    // ── Dependencies ─────────────────────────────────────────────────────────
    private readonly PriceLoader         _prices;
    private readonly PortfolioState      _state  = new();
    private readonly TrackingErrorProxy  _te;

    // ── Seeding amount — stored so year-end reset can re-seed same value ──────
    private decimal _seedAmount;

    // ── Output ────────────────────────────────────────────────────────────────
    private readonly List<LotStateVector> _snapshots = new(128_000);

    // ── Reopen queue: reopenDay → list of (symbol, sector, dollars) ──────────
    private readonly Dictionary<int, List<(string Symbol, string Sector, decimal Dollars)>>
        _reopenQueue = new();

    // ── Lot count cache: symbol → number of currently open lots ──────────────
    private readonly Dictionary<string, int> _lotCount = new();

    public SimulationEngine(PriceLoader prices)
    {
        _prices = prices;
        _te     = new TrackingErrorProxy(prices);
    }
    // ── Public entry point ───────────────────────────────────────────────────

    /// <param name="initialPortfolioValue">Total dollars invested at simulation start.</param>
    public List<LotStateVector> Run(decimal initialPortfolioValue = 10_000_000m)
    {
        InitializePortfolio(PriceLoader.WarmupDays, initialPortfolioValue);

        for (int t = PriceLoader.WarmupDays; t < _prices.DayCount; t++)
        {
            ProcessDay(t);

            if (t % 50 == 0)
                Console.WriteLine($"  [Engine] Day {t}/{_prices.DayCount - 1}  " +
                                  $"open={_state.OpenLots.Count}  " +
                                  $"snapshots={_snapshots.Count}  " +
                                  $"G_YTD={_state.G_YTD:F0}");
        }

        Console.WriteLine($"[SimulationEngine] Complete. Total snapshots: {_snapshots.Count}");
        return _snapshots;
    }

    // ── Private: day loop ─────────────────────────────────────────────────────

    private void ProcessDay(int t)
    {
        var closes = _prices.GetClosesDecimal(t);

        // Portfolio value (only lots with a valid close price today)
        decimal portValue = _state.OpenLots
            .Where(l => closes.ContainsKey(l.Symbol))
            .Sum(l => l.Shares * closes[l.Symbol]);

        if (portValue <= 0m) portValue = 1m;   // guard against empty portfolio

        // Equal-weighted return of open lots — avoids structural jumps from harvest/reopen events
        float sigmaTE = _te.Update(_state.OpenLots.Select(l => l.Symbol));

        // Extract snapshot + oracle for every open lot (iterate over copy; harvests mutate list)
        foreach (var lot in _state.OpenLots.ToList())
        {
            if (!closes.TryGetValue(lot.Symbol, out decimal close)) continue;

            var snap = ExtractSnapshot(lot, t, close, portValue, sigmaTE);
            _snapshots.Add(snap);

            if (snap.Y_Oracle == 1)
                Harvest(lot, close, t, portValue);
        }

        // Reopen lots whose wash-sale window cleared on exactly this day
        if (_reopenQueue.TryGetValue(t, out var toReopen))
        {
            foreach (var (sym, sector, dollars) in toReopen)
            {
                if (!closes.TryGetValue(sym, out decimal price) || price <= 0m) continue;
                int shares = (int)(dollars / price);
                if (shares == 0) continue;
                var lot = new Lot(sym, sector, price, shares, t);
                _state.OpenLot(lot);
                _lotCount[sym] = (_lotCount.GetValueOrDefault(sym) + 1);
            }
            _reopenQueue.Remove(t);
        }

        _state.AdvanceDay();

        // Year-end reset (G_YTD ← 0, wash clocks persist)
        var today    = _prices.GetDate(t);
        var tomorrow = t + 1 < _prices.DayCount ? _prices.GetDate(t + 1) : today.AddDays(1);
        if (tomorrow.Year != today.Year)
        {
            _state.ResetForNewYear();       // rolls net loss into LossCarryforward
            _state.SeedGYTD(_seedAmount);   // re-seed for the new tax year
            Console.WriteLine($"  [Engine] Year-end reset — G_YTD re-seeded to {_seedAmount:C0}, " +
                              $"carryforward = {_state.Ledger.LossCarryforward:C0}");
        }
    }

    private LotStateVector ExtractSnapshot(
        Lot lot, int t, decimal close, decimal portValue, float sigmaTE)
    {
        int holdingDays = lot.HoldingPeriod(t);
        int washClock   = _state.GetWashClock(lot.Symbol);

        decimal unrealized = lot.UnrealizedReturn(close);

        int yOracle = OracleBoundary.Label(
            unrealizedReturn: unrealized,
            sigmaTE:          sigmaTE,
            gYtd:             _state.G_YTD,
            washClock:        washClock);

        var date    = _prices.GetDate(t);
        int daysToYE = new DateOnly(date.Year, 12, 31).DayNumber - date.DayNumber;

        // taxValue_k = g(ledger_t, h_k, ℓ_k) — capacity-aware harvest value.
        // Loss in dollars is 0 for lots not at a loss (winners have no harvestable loss).
        decimal lossDollars = unrealized < 0m ? (lot.CostBasis - close) * lot.Shares : 0m;
        decimal taxValue    = _state.Ledger.ComputeTaxValue(lossDollars, holdingDays);

        return new LotStateVector
        {
            // Lot-level
            L          = (float)unrealized,
            H          = holdingDays,
            S          = lot.IsLongTerm(t) ? 1 : 0,
            B          = (float)lot.CostBasis,
            W          = portValue > 0m ? (float)(lot.Shares * close / portValue) : 0f,
            K          = _lotCount.GetValueOrDefault(lot.Symbol, 1),

            // Portfolio-level (shared TaxLedger + risk state)
            RealizedGainsYTD     = (float)_state.Ledger.RealizedGainsYTD,
            LossCarryforward     = (float)_state.Ledger.LossCarryforward,
            OrdinaryOffsetBudget = (float)_state.Ledger.OrdinaryOffsetBudget,
            Sigma_TE   = sigmaTE,
            WashClock  = washClock,

            // Asset-level
            R_t        = _prices.DailyReturn(lot.Symbol, t),
            SigmaRange = _prices.RangeVol(lot.Symbol, t),
            DeltaMA50  = _prices.DeviationFromMA(lot.Symbol, t, 50),
            DeltaMA200 = _prices.DeviationFromMA(lot.Symbol, t, 200),

            // Derived
            TaxValue   = (float)taxValue,
            DaysToYE   = daysToYE,

            // Labels (soft labels filled in second pass by SoftLabelBuilder)
            Y_Oracle   = yOracle,
            Y_Soft_GBM = 0f,
            Y_Soft_BT  = 0f,
            Y_TaxValue = (float)taxValue,

            // Metadata
            Symbol   = lot.Symbol,
            Sector   = lot.Sector,
            Timestep = t
        };
    }

    private void Harvest(Lot lot, decimal close, int t, decimal portValue)
    {
        decimal dollars = lot.Shares * close;
        int     lotsBefore = _lotCount.GetValueOrDefault(lot.Symbol, 1);

        _state.HarvestLot(lot, close);

        _lotCount[lot.Symbol] = Math.Max(0, lotsBefore - 1);

        // Schedule reopen after wash-sale window
        int reopenDay = t + OracleBoundary.WashSaleDays;
        if (reopenDay < _prices.DayCount)
        {
            if (!_reopenQueue.TryGetValue(reopenDay, out var list))
                _reopenQueue[reopenDay] = list = new();
            list.Add((lot.Symbol, lot.Sector, dollars));
        }
    }

    // ── Private: portfolio initialisation ────────────────────────────────────

    private void InitializePortfolio(int day0, decimal totalValue)
    {
        var closes   = _prices.GetClosesDecimal(day0);
        int n        = closes.Count;
        if (n == 0) throw new InvalidOperationException("No price data on warmup day.");
        decimal perLot = totalValue / n;

        foreach (var (symbol, price) in closes)
        {
            if (price <= 0m) continue;
            int shares = (int)(perLot / price);
            if (shares == 0) shares = 1;

            string sector = _prices.GetSector(symbol);
            var lot = new Lot(symbol, sector, price, shares, day0);
            _state.OpenLot(lot);
            _lotCount[symbol] = 1;
        }

        // Seed G_YTD with 10 % of portfolio value to simulate prior-year or external
        // realized gains — calibrated to the S&P 500's ~10% historical annual return
        // (the client realizes gains elsewhere at roughly the index's pace).
        // Without this the oracle's gYtd > 0 gate is permanently closed and no harvests fire.
        _seedAmount = totalValue * 0.10m;
        _state.SeedGYTD(_seedAmount);

        Console.WriteLine(
            $"[SimulationEngine] Portfolio initialised: {_state.OpenLots.Count} lots " +
            $"on day {day0} ({_prices.GetDate(day0)}), value ≈ {totalValue:C0}  " +
            $"G_YTD seeded to {_seedAmount:C0}");
    }

}
