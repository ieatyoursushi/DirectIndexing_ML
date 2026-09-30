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
///   5. Advance wash-sale clocks; roll the TaxLedger at year boundaries.
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
    private readonly OracleConfig        _oracle;
    private readonly ContributionPolicy  _contrib;

    // ── Initial book value — the base for the exogenous contribution schedule ──
    private decimal _initialValue;

    // ── Running totals for the contribution ablation's summary line ───────────
    private decimal _contributedTotal;
    private int     _contributedLots;

    // ── Output ────────────────────────────────────────────────────────────────
    private readonly List<LotStateVector> _snapshots = new(128_000);

    // ── Trade log: every lot opened or sold (audited by WashSaleAudit) ─────────
    private readonly List<TradeEvent> _trades = new();

    /// <summary>Every executed buy (initial book, reopen, contribution) and sale (harvest).</summary>
    public IReadOnlyList<TradeEvent> Trades => _trades;

    // ── Reopen queue: reopenDay → list of (symbol, sector, dollars) ──────────
    private readonly Dictionary<int, List<(string Symbol, string Sector, decimal Dollars)>>
        _reopenQueue = new();

    // ── Lot count cache: symbol → number of currently open lots ──────────────
    private readonly Dictionary<string, int> _lotCount = new();

    public SimulationEngine(
        PriceLoader prices,
        OracleConfig? oracleConfig = null,
        ContributionPolicy? contributionPolicy = null)
    {
        _prices  = prices;
        _te      = new TrackingErrorProxy(prices);
        _oracle  = oracleConfig ?? OracleConfig.Default;
        _contrib = contributionPolicy ?? ContributionPolicy.Off;
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
                                  $"G_net={_state.Ledger.RealizedGainsYTD:F0}");
        }

        Console.WriteLine($"[SimulationEngine] Complete. Total snapshots: {_snapshots.Count}");
        if (_contrib.Enabled)
            Console.WriteLine($"[SimulationEngine] Contributions: {_contributedTotal:C0} across " +
                              $"{_contributedLots:N0} new lots ({_contrib.Describe()})");
        return _snapshots;
    }

    // ── Private: day loop ─────────────────────────────────────────────────────

    private void ProcessDay(int t)
    {
        var closes = _prices.GetClosesDecimal(t);

        // Portfolio value (only lots with a valid close price today)
        // [math:portfolio_value] — DataMemo/spec/SymbolTable.md
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

        // [math:reopen] — DataMemo/spec/SymbolTable.md
        // Reopen lots whose wash-sale window cleared on exactly this day
        if (_reopenQueue.TryGetValue(t, out var toReopen))
        {
            foreach (var (sym, sector, dollars) in toReopen)
            {
                if (!closes.TryGetValue(sym, out decimal price) || price <= 0m) continue;

                // Wash-sale re-check before buying back (v0.3 fix). A reopen is scheduled
                // for harvest_day + 30, when the clock would normally read exactly 30. But
                // the ticker can be harvested AGAIN while this reopen is pending — including
                // earlier on this very day, since the harvest loop runs before this block —
                // which resets its clock. Buying now would disallow that newer loss under
                // IRS §1091, so defer the buy until the window genuinely clears.
                //
                // This could not occur before v0.3: with one lot per ticker there was never
                // a second lot to re-harvest while the reopen was pending. Contributions
                // make tickers multi-lot and thereby expose it (measured: 17.4% of
                // run-opened lots violated before this fix, 0% after).
                int clock = _state.GetWashClock(sym);
                if (clock < OracleBoundary.WashSaleDays)
                {
                    int retry = t + (OracleBoundary.WashSaleDays - clock);
                    if (retry < _prices.DayCount)
                    {
                        if (!_reopenQueue.TryGetValue(retry, out var deferred))
                            _reopenQueue[retry] = deferred = new();
                        deferred.Add((sym, sector, dollars));
                    }
                    continue;
                }

                int shares = (int)(dollars / price);
                if (shares == 0) continue;
                var lot = new Lot(sym, sector, price, shares, t);
                _state.OpenLot(lot);
                _trades.Add(new TradeEvent(_prices.GetDate(t), t, sym, TradeKind.Buy, lot, price, 0m));
                _lotCount[sym] = (_lotCount.GetValueOrDefault(sym) + 1);
            }
            _reopenQueue.Remove(t);
        }

        // Contributions run AFTER harvests and reopens so they see today's wash clocks —
        // a ticker harvested today has WashClock = 0 and is correctly ineligible — and
        // BEFORE AdvanceDay, which is what increments those clocks.
        ProcessContribution(t, closes, portValue);

        _state.AdvanceDay();

        // Year-end reset (G_net ← 0, net loss rolls into carryforward, wash clocks persist)
        var today    = _prices.GetDate(t);
        var tomorrow = t + 1 < _prices.DayCount ? _prices.GetDate(t + 1) : today.AddDays(1);
        if (tomorrow.Year != today.Year)
        {
            _state.ResetForNewYear();
            Console.WriteLine($"  [Engine] Year-end reset — carryforward = {_state.Ledger.LossCarryforward:C0}");
        }
    }

    // [math:y_oracle] [math:y_taxvalue] [math:y_utility] — DataMemo/spec/SymbolTable.md
    private LotStateVector ExtractSnapshot(
        Lot lot, int t, decimal close, decimal portValue, float sigmaTE)
    {
        int holdingDays = lot.HoldingPeriod(t);
        int washClock   = _state.GetWashClock(lot.Symbol);

        decimal unrealized = lot.UnrealizedReturn(close);

        var date    = _prices.GetDate(t);
        int daysToYE = new DateOnly(date.Year, 12, 31).DayNumber - date.DayNumber;

        // taxValue_k = g(ledger_t, h_k, ℓ_k) — capacity-aware harvest value.
        // Loss in dollars is 0 for lots not at a loss (winners have no harvestable loss).
        decimal lossDollars = unrealized < 0m ? (lot.CostBasis - close) * lot.Shares : 0m;
        decimal taxValue    = _state.Ledger.ComputeTaxValue(lossDollars, holdingDays);

        var snap = new LotStateVector
        {
            // Lot-level
            L          = (float)unrealized,
            H          = holdingDays,
            S          = lot.IsLongTerm(t) ? 1 : 0,
            B          = (float)lot.CostBasis,
            W          = portValue > 0m ? (float)(lot.Shares * close / portValue) : 0f,
            K          = _lotCount.GetValueOrDefault(lot.Symbol, 1),
            Shares     = lot.Shares,   // in-memory plumbing for soft-label re-dollarization

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
            Y_Oracle   = 0,
            Y_Soft_GBM = 0f,
            Y_Soft_BT  = 0f,
            Y_TaxValue = (float)taxValue,

            // Metadata
            Symbol   = lot.Symbol,
            Sector   = lot.Sector,
            Timestep = t
        };

        // Oracle labels ride the snapshot (canonical call-site coupling, §5.3):
        // the acting oracle and the raw utility score.
        return snap with
        {
            Y_Oracle  = OracleBoundary.Label(snap, _oracle),
            Y_Utility = (float)OracleBoundary.Utility(taxValue, sigmaTE, _oracle),
        };
    }

    private void Harvest(Lot lot, decimal close, int t, decimal portValue)
    {
        decimal dollars = lot.Shares * close;
        int     lotsBefore = _lotCount.GetValueOrDefault(lot.Symbol, 1);

        _trades.Add(new TradeEvent(_prices.GetDate(t), t, lot.Symbol, TradeKind.Sell, lot, close,
                                   (close - lot.CostBasis) * lot.Shares));
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

    // ── Private: contributions (v0.3 P0 — the cost-basis-aging fix) ───────────

    /// <summary>
    /// On a contribution day, deposit cash and mint fresh lots at today's prices in the most
    /// underweight <i>eligible</i> tickers. Eligibility excludes any ticker inside its
    /// wash-sale window — buying one back within 30 days of its loss sale would disallow that
    /// loss (IRS §1091), so the contribution path can never invalidate a booked harvest.
    ///
    /// Fresh lots carry <i>today's</i> cost basis, which is the entire point: they can dip
    /// below it in the next drawdown, whereas a 2007-basis lot cannot.
    /// </summary>
    // [math:contribution] — DataMemo/spec/SymbolTable.md
    private void ProcessContribution(int t, Dictionary<string, decimal> closes, decimal portValue)
    {
        if (!_contrib.Enabled) return;

        int elapsed = t - PriceLoader.WarmupDays;
        if (elapsed <= 0 || elapsed % _contrib.IntervalDays != 0) return;

        // Actual dollar value currently held per ticker (0 for fully-harvested names).
        var heldValue = new Dictionary<string, decimal>();
        foreach (var lot in _state.OpenLots)
            if (closes.TryGetValue(lot.Symbol, out decimal px))
                heldValue[lot.Symbol] = heldValue.GetValueOrDefault(lot.Symbol) + lot.Shares * px;

        // Eligible = priced today AND clear of the wash-sale window.
        var eligible = closes
            .Where(kv => kv.Value > 0m &&
                         _state.GetWashClock(kv.Key) >= OracleBoundary.WashSaleDays)
            .Select(kv => kv.Key)
            .ToList();
        if (eligible.Count == 0) return;

        // Rank by underweight vs an equal-weight target over the priced universe.
        decimal targetWeight = 1m / closes.Count;
        var picks = eligible
            .OrderByDescending(sym => targetWeight - heldValue.GetValueOrDefault(sym) / portValue)
            .Take(_contrib.NamesPerContribution)
            .ToList();
        if (picks.Count == 0) return;

        decimal cash    = _contrib.AmountPer(_initialValue);
        decimal perName = cash / picks.Count;

        foreach (var symbol in picks)
        {
            decimal price = closes[symbol];
            int shares = (int)(perName / price);
            if (shares == 0) continue;

            var lot = new Lot(symbol, _prices.GetSector(symbol), price, shares, t);
            _state.OpenLot(lot);
            _trades.Add(new TradeEvent(_prices.GetDate(t), t, symbol, TradeKind.Buy, lot, price, 0m));
            _lotCount[symbol] = _lotCount.GetValueOrDefault(symbol) + 1;
            _contributedTotal += shares * price;
            _contributedLots++;
        }
    }

    // ── Private: portfolio initialisation ────────────────────────────────────

    private void InitializePortfolio(int day0, decimal totalValue)
    {
        var closes   = _prices.GetClosesDecimal(day0);
        int n        = closes.Count;
        if (n == 0) throw new InvalidOperationException("No price data on warmup day.");
        decimal perLot = totalValue / n;
        _initialValue  = totalValue;   // base for the exogenous contribution schedule

        foreach (var (symbol, price) in closes)
        {
            if (price <= 0m) continue;
            int shares = (int)(perLot / price);
            if (shares == 0) shares = 1;

            string sector = _prices.GetSector(symbol);
            var lot = new Lot(symbol, sector, price, shares, day0);
            _state.OpenLot(lot);
            _trades.Add(new TradeEvent(_prices.GetDate(day0), day0, symbol, TradeKind.Buy, lot, price, 0m));
            _lotCount[symbol] = 1;
        }

        // No external-gains seed: the honest loss-only book (offset capacity = the
        // $3k/yr ordinary allowance until the book realizes gains of its own).
        Console.WriteLine(
            $"[SimulationEngine] Portfolio initialised: {_state.OpenLots.Count} lots " +
            $"on day {day0} ({_prices.GetDate(day0)}), value ≈ {totalValue:C0}");
    }

}
