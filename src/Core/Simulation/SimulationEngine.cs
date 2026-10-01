using DirectIndexing.Core.Policy;
using DirectIndexing.Core.Simulation.Volatility;
using DirectIndexing.Core.Simulation.Covariance;
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
    private readonly PortfolioState      _state;
    private readonly TrackingErrorProxy  _te;
    private readonly OracleConfig        _oracle;
    private readonly ContributionPolicy  _contrib;
    private readonly TrimPolicy          _trim;
    private readonly VolState            _vol;
    private readonly IHarvestPolicy      _policy;

    // ── RunMetrics accumulators (v0.3-11) ─────────────────────────────────────
    private decimal _benefitUsedNow, _benefitBanked, _gainTaxCost, _sumTradeW;
    private decimal _tradedDollars, _rollTrueUp;
    private decimal _cash;   // uninvested proceeds (share rounding, unpriced reopen days, unspent trim proceeds)
    private int     _lossSales, _gainSales, _days;
    private double  _sumTeExAnte;
    private readonly List<double> _activeReturns = new();
    private Dictionary<string, decimal>? _prevCloses;

    // ── Initial book value — the base for the exogenous contribution schedule ──
    private decimal _initialValue;

    // ── Running totals for the contribution ablation's summary line ───────────
    private decimal _contributedTotal;
    private int     _contributedLots;
    private decimal _trimmedGains;
    private int     _trimmedLots;

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
        ContributionPolicy? contributionPolicy = null,
        TrimPolicy? trimPolicy = null,
        RiskModel? riskModel = null,
        IHarvestPolicy? policy = null)
    {
        _prices  = prices;
        var risk = riskModel ?? RiskModel.Default;
        _te      = new TrackingErrorProxy(prices, CovarianceFactory.Create(prices, risk.Covariance), risk.Weighting);
        _oracle  = oracleConfig ?? OracleConfig.Default;
        _state   = new PortfolioState { ReharvestGuard = _oracle.ReharvestGuard };
        _contrib = contributionPolicy ?? ContributionPolicy.Off;
        _trim    = trimPolicy ?? TrimPolicy.Off;
        _vol     = new VolState(prices);   // EWMA σ̂ paths, 𝓕_t (v0.3-8)
        _policy  = policy ?? new OraclePolicy();
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
        if (_trim.Enabled)
            Console.WriteLine($"[SimulationEngine] Trim: {_trimmedLots:N0} gain lots sold, " +
                              $"{_trimmedGains:C0} realized gains ({_trim.Describe()})");
        return _snapshots;
    }

    // ── Private: day loop ─────────────────────────────────────────────────────

    private void ProcessDay(int t)
    {
        var today  = _prices.GetDate(t);
        _state.SetDate(today);                  // all §1091 clocks are calendar-date differences
        var closes = _prices.GetClosesDecimal(t);

        // Portfolio value (only lots with a valid close price today)
        // [math:portfolio_value] — DataMemo/spec/SymbolTable.md
        decimal portValue = _state.OpenLots
            .Where(l => closes.ContainsKey(l.Symbol))
            .Sum(l => l.Shares * closes[l.Symbol]);

        if (portValue <= 0m) portValue = 1m;   // guard against empty portfolio

        // σ_TE = √(252 δwᵀ Σ̂_t δw) — Σ̂_t point-in-time by default (v0.3-6, F1); the quadratic form
        // avoids the structural jumps a return-based estimate shows at harvest/reopen events
        float sigmaTE = _te.Update(t, _state.OpenLots, closes);
        AccumulateRisk(t, closes, sigmaTE);

        // Extract snapshot + oracle for every open lot (iterate over copy; harvests mutate list)
        foreach (var lot in _state.OpenLots.ToList())
        {
            if (!closes.TryGetValue(lot.Symbol, out decimal close)) continue;

            var snap = ExtractSnapshot(lot, t, close, portValue, sigmaTE);
            _snapshots.Add(snap);

            // labels describe the oracle f*; the POLICY decides the action (v0.3-11 seam)
            if (_policy.ShouldHarvest(snap))
                Harvest(lot, close, t, portValue);
        }

        // [math:reopen] — DataMemo/spec/SymbolTable.md
        // Reopen lots whose wash-sale window cleared on exactly this day
        if (_reopenQueue.TryGetValue(t, out var toReopen))
        {
            foreach (var (sym, sector, dollars) in toReopen)
            {
                if (!closes.TryGetValue(sym, out decimal price) || price <= 0m) { _cash += dollars; continue; }

                // §1091 after-side re-check (v0.3-1): the ticker may have been loss-sold
                // AGAIN while this reopen was pending — including earlier today, since the
                // harvest loop runs first. Buying inside that window would disallow the newer
                // loss, so defer to the first trading day more than 30 calendar days after it.
                if (!_state.CanBuy(sym))
                {
                    int retry = _prices.FirstIndexOnOrAfter(_state.EarliestBuyDate(sym));
                    if (!_reopenQueue.TryGetValue(retry, out var deferred))
                        _reopenQueue[retry] = deferred = new();
                    deferred.Add((sym, sector, dollars));
                    continue;
                }

                int shares = (int)(dollars / price);
                _cash += dollars - shares * price;       // whole shares only; the remainder stays cash
                if (shares == 0) continue;
                var lot = new Lot(sym, sector, price, shares, t, today);
                _state.OpenLot(lot);
                _trades.Add(new TradeEvent(_prices.GetDate(t), t, sym, TradeKind.Buy, lot, price, 0m));
                _lotCount[sym] = (_lotCount.GetValueOrDefault(sym) + 1);
            }
            _reopenQueue.Remove(t);
        }

        // Contributions run AFTER harvests and reopens so they see today's loss sales —
        // a ticker loss-sold today fails CanBuy and is correctly ineligible.
        ProcessContribution(t, closes, portValue);
        ProcessTrim(t, closes, portValue);

        // Year-end reset (G_net ← 0, net loss rolls into carryforward, wash clocks persist)
        var tomorrow = t + 1 < _prices.DayCount ? _prices.GetDate(t + 1) : today.AddDays(1);
        if (tomorrow.Year != today.Year)
        {
            decimal wBeforeRoll = _state.Ledger.TaxPosition;
            _state.ResetForNewYear();
            _rollTrueUp += _state.Ledger.TaxPosition - wBeforeRoll;   // banked carry → next year's $3k line
            Console.WriteLine($"  [Engine] Year-end reset — carryforward = {_state.Ledger.LossCarryforward:C0}");
        }
    }

    // [math:y_oracle] [math:y_taxvalue] [math:y_utility] — DataMemo/spec/SymbolTable.md
    private LotStateVector ExtractSnapshot(
        Lot lot, int t, decimal close, decimal portValue, float sigmaTE)
    {
        int holdingDays = lot.HoldingPeriod(t);
        int washClock   = _state.WashClock(lot);

        decimal unrealized = lot.UnrealizedReturn(close);

        var date    = _prices.GetDate(t);
        int daysToYE = new DateOnly(date.Year, 12, 31).DayNumber - date.DayNumber;

        // taxValue_k = g(ledger_t, h_k, ℓ_k) — capacity-aware harvest value.
        // Loss in dollars is 0 for lots not at a loss (winners have no harvestable loss).
        decimal lossDollars = unrealized < 0m ? (lot.CostBasis - close) * lot.Shares : 0m;
        bool    longTerm    = lot.IsLongTerm(date);
        decimal taxValue    = _state.Ledger.ComputeTaxValue(lossDollars, longTerm);

        // barrier coordinate (v0.3-8): log-distance to the loss trigger in forecast σ over T_fwd
        double z = LossBarrier.Z(LossBarrier.Distance((double)close, (double)lot.CostBasis, (double)_oracle.LossThreshold),
                             _vol.HorizonVariance(lot.Symbol, t, SoftLabelBuilder.Window));

        var snap = new LotStateVector
        {
            // Lot-level
            L          = (float)unrealized,
            H          = holdingDays,
            S          = longTerm ? 1 : 0,
            B          = (float)lot.CostBasis,
            W          = portValue > 0m ? (float)(lot.Shares * close / portValue) : 0f,
            K          = _lotCount.GetValueOrDefault(lot.Symbol, 1),
            Shares     = lot.Shares,   // in-memory plumbing for soft-label re-dollarization
            PurchaseDayNumber = lot.PurchaseDate.DayNumber,   // in-memory plumbing (§1222 forward)

            // Portfolio-level (shared TaxLedger + risk state)
            NetST                = (float)_state.Ledger.NetShortTerm,
            NetLT                = (float)_state.Ledger.NetLongTerm,
            CarryST              = (float)_state.Ledger.CarryShortTerm,
            CarryLT              = (float)_state.Ledger.CarryLongTerm,
            OrdinaryOffsetBudget = (float)_state.Ledger.OrdinaryOffsetBudget,
            Sigma_TE   = sigmaTE,
            WashClock  = washClock,

            // Asset-level
            R_t        = _prices.DailyReturn(lot.Symbol, t),
            SigmaRange = _prices.RangeVol(lot.Symbol, t),
            DeltaMA50  = _prices.DeviationFromMA(lot.Symbol, t, 50),
            DeltaMA200 = _prices.DeviationFromMA(lot.Symbol, t, 200),
            SigmaHat   = _vol.SigmaHat(lot.Symbol, t),
            SigmaMkt   = _vol.SigmaMkt(t),

            // Derived
            TaxValue   = (float)taxValue,
            DaysToYE   = daysToYE,
            ZBarrier   = (float)z,
            PBarrier   = (float)LossBarrier.TouchProbability(z),

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
        RealizeWithAccounting(lot, close);

        _lotCount[lot.Symbol] = Math.Max(0, lotsBefore - 1);

        // Schedule the same-ticker reopen on the first trading day MORE than 30 calendar
        // days after the sale (§1091's window is inclusive: day +30 is still inside it).
        int reopenDay = _prices.FirstIndexOnOrAfter(
            _prices.GetDate(t).AddDays(PortfolioState.WashWindowDays + 1));
        // queued even past the last day: the proceeds are still the client's cash
        // (RunMetrics counts them as PendingReopenCash) — never silently dropped
        if (!_reopenQueue.TryGetValue(reopenDay, out var list))
            _reopenQueue[reopenDay] = list = new();
        list.Add((lot.Symbol, lot.Sector, dollars));
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

        _contributedTotal += BuyUnderweight(t, closes, portValue, _contrib.AmountPer(_initialValue),
                                            _contrib.NamesPerContribution, _contrib.SkipHarvestableNames,
                                            exclude: null, ref _contributedLots);
    }

    /// <summary>
    /// The shared buy path (contributions and trim reinvestment): split <paramref name="cash"/>
    /// over the <paramref name="names"/> most underweight eligible tickers vs an equal-weight
    /// target. Eligible = priced, passes §1091's after-side (<c>CanBuy</c>), not excluded, and —
    /// when <paramref name="skipHarvestable"/> — holding no harvestable lot (a fresh lot would be
    /// a §1091 replacement and block that harvest for 30 days). Returns dollars invested.
    /// </summary>
    private decimal BuyUnderweight(int t, Dictionary<string, decimal> closes, decimal portValue,
        decimal cash, int names, bool skipHarvestable, ISet<string>? exclude, ref int lotsMinted)
    {
        // Actual dollar value currently held per ticker (0 for fully-harvested names).
        var heldValue = new Dictionary<string, decimal>();
        foreach (var lot in _state.OpenLots)
            if (closes.TryGetValue(lot.Symbol, out decimal px))
                heldValue[lot.Symbol] = heldValue.GetValueOrDefault(lot.Symbol) + lot.Shares * px;

        var eligible = closes
            .Where(kv => kv.Value > 0m && _state.CanBuy(kv.Key) &&
                         (exclude is null || !exclude.Contains(kv.Key)) &&
                         !(skipHarvestable && HasHarvestableLot(kv.Key, kv.Value)))
            .Select(kv => kv.Key)
            .ToList();
        if (eligible.Count == 0) return 0m;

        // Rank by underweight vs an equal-weight target over the priced universe.
        decimal targetWeight = 1m / closes.Count;
        var picks = eligible
            .OrderByDescending(sym => targetWeight - heldValue.GetValueOrDefault(sym) / portValue)
            .Take(names)
            .ToList();
        if (picks.Count == 0) return 0m;

        decimal perName  = cash / picks.Count;
        decimal invested = 0m;

        foreach (var symbol in picks)
        {
            decimal price = closes[symbol];
            int shares = (int)(perName / price);
            if (shares == 0) continue;

            var lot = new Lot(symbol, _prices.GetSector(symbol), price, shares, t, _prices.GetDate(t));
            _state.OpenLot(lot);
            _trades.Add(new TradeEvent(_prices.GetDate(t), t, symbol, TradeKind.Buy, lot, price, 0m));
            _lotCount[symbol] = _lotCount.GetValueOrDefault(symbol) + 1;
            invested += shares * price;
            lotsMinted++;
        }
        return invested;
    }

    /// <summary>
    /// Sell-winner trim (v0.3-4, <see cref="TrimPolicy"/>): sell whole GAIN lots of names above
    /// (1 + band) × equal weight, highest basis first, while each lot fits inside the name's
    /// excess; reinvest the proceeds through <see cref="BuyUnderweight"/>. Gain sales go to the
    /// ledger pool of their §1222 character and open no §1091 window.
    /// </summary>
    // [math:trim] — DataMemo/spec/SymbolTable.md
    private void ProcessTrim(int t, Dictionary<string, decimal> closes, decimal portValue)
    {
        if (!_trim.Enabled) return;

        int elapsed = t - PriceLoader.WarmupDays;
        if (elapsed <= 0 || elapsed % _trim.IntervalDays != 0) return;

        var heldValue = new Dictionary<string, decimal>();
        foreach (var lot in _state.OpenLots)
            if (closes.TryGetValue(lot.Symbol, out decimal px))
                heldValue[lot.Symbol] = heldValue.GetValueOrDefault(lot.Symbol) + lot.Shares * px;

        decimal target = portValue / closes.Count;
        var overweight = heldValue
            .Where(kv => kv.Value > (1m + _trim.Band) * target)
            .OrderByDescending(kv => kv.Value)
            .Take(_trim.NamesPerTrim)
            .ToList();
        if (overweight.Count == 0) return;

        var date = _prices.GetDate(t);
        decimal proceeds = 0m;
        var trimmed = new HashSet<string>();
        foreach (var (symbol, held) in overweight)
        {
            decimal close  = closes[symbol];
            decimal excess = held - target;
            var gainLots = _state.OpenLotsOf(symbol)
                .Where(l => close >= l.CostBasis)
                .OrderByDescending(l => l.CostBasis)
                .ToList();
            foreach (var lot in gainLots)
            {
                decimal value = lot.Shares * close;
                if (value > excess) continue;          // whole lots only; never overshoot the target
                decimal gain = (close - lot.CostBasis) * lot.Shares;
                _trades.Add(new TradeEvent(date, t, symbol, TradeKind.Sell, lot, close, gain));
                RealizeWithAccounting(lot, close);     // realizes into the ledger; a gain opens no window
                _lotCount[symbol] = Math.Max(0, _lotCount.GetValueOrDefault(symbol, 1) - 1);
                excess        -= value;
                proceeds      += value;
                _trimmedGains += gain;
                _trimmedLots++;
                trimmed.Add(symbol);
            }
        }
        if (proceeds <= 0m) return;

        int minted = 0;
        _cash += proceeds - BuyUnderweight(t, closes, portValue, proceeds, _trim.NamesPerTrim,
                                           skipHarvestable: true, exclude: trimmed, ref minted);
    }

    /// <summary>
    /// Every sale goes through here: realize into the ledger and book ΔW_tax, split for losses
    /// into this-year tax saved (−ΔT) and newly banked carryforward value (τ_f·δ·ΔΣC').
    /// </summary>
    private void RealizeWithAccounting(Lot lot, decimal close)
    {
        decimal gain = (close - lot.CostBasis) * lot.Shares;
        var before = _state.Ledger.State.Close();
        decimal w0 = _state.Ledger.TaxPosition;
        _state.HarvestLot(lot, close);
        var after = _state.Ledger.State.Close();
        decimal dW = _state.Ledger.TaxPosition - w0;
        _sumTradeW     += dW;
        _tradedDollars += lot.Shares * close;
        if (gain < 0m)
        {
            _lossSales++;
            _benefitUsedNow += before.Tax - after.Tax;
            _benefitBanked  += TaxLedger.TauFuture * TaxLedger.CarryforwardDiscount *
                               ((after.CarryShortTerm + after.CarryLongTerm) - (before.CarryShortTerm + before.CarryLongTerm));
        }
        else
        {
            _gainSales++;
            _gainTaxCost -= dW;
        }
    }

    /// <summary>
    /// Ex-ante σ_TE (sum) and the realized active return of the book held INTO today (lots open
    /// at both closes, before today's trades) vs the equal-weight priced universe.
    /// </summary>
    private void AccumulateRisk(int t, Dictionary<string, decimal> closes, float sigmaTE)
    {
        _days++;
        _sumTeExAnte += sigmaTE;
        if (_prevCloses is not null)
        {
            decimal v0 = 0m, v1 = 0m;
            foreach (var lot in _state.OpenLots)
                if (_prevCloses.TryGetValue(lot.Symbol, out var p0) && closes.TryGetValue(lot.Symbol, out var p1))
                { v0 += lot.Shares * p0; v1 += lot.Shares * p1; }
            double sum = 0; int n = 0;
            foreach (var sym in closes.Keys)
            {
                float r = _prices.DailyReturn(sym, t);
                if (!float.IsNaN(r)) { sum += r; n++; }
            }
            if (v0 > 0m && n > 0) _activeReturns.Add((double)(v1 / v0 - 1m) - sum / n);
        }
        _prevCloses = closes;
    }

    /// <summary>
    /// The scoreboard of one run (v0.3-11; DataMemo/decisions/PolicyLayer_v04.md §6). Call after
    /// <see cref="Run"/>. Wealth is holdings + cash awaiting its §1091 reopen; after-tax wealth adds
    /// the tax-position potential W_tax; liquidation value additionally closes every open lot
    /// through Schedule D (TLH defers tax — this is the honest terminal number).
    /// </summary>
    // [math:run_metrics] — DataMemo/spec/SymbolTable.md
    public RunMetrics Metrics()
    {
        int tEnd = _prices.DayCount - 1;
        var closes = _prices.GetClosesDecimal(tEnd);
        decimal holdings = 0m;
        var liquidation = _state.Ledger.State;
        var date = _prices.GetDate(tEnd);
        foreach (var lot in _state.OpenLots)
            if (closes.TryGetValue(lot.Symbol, out var px))
            {
                holdings += lot.Shares * px;
                liquidation = liquidation.With((px - lot.CostBasis) * lot.Shares, lot.IsLongTerm(date));
            }
        decimal pending = _reopenQueue.Where(kv => kv.Key > tEnd).Sum(kv => kv.Value.Sum(x => x.Dollars));
        decimal w = _state.Ledger.TaxPosition;
        var lc = liquidation.Close();
        decimal wLiq = -_state.Ledger.TaxPaid - lc.Tax
                     + TaxLedger.TauFuture * TaxLedger.CarryforwardDiscount * (lc.CarryShortTerm + lc.CarryLongTerm);
        double years = Math.Max(_days, 1) / 252.0;
        double mean  = _activeReturns.Count > 0 ? _activeReturns.Average() : 0;
        double te    = _activeReturns.Count > 1
            ? Math.Sqrt(_activeReturns.Sum(x => (x - mean) * (x - mean)) / (_activeReturns.Count - 1) * 252) : 0;
        int violations = WashSaleAudit.Violations(_trades).Count;
        return new RunMetrics(
            Policy: _policy.Name, Days: _days,
            InitialValue: _initialValue, Contributions: _contributedTotal,
            TerminalHoldings: holdings, PendingReopenCash: pending, Cash: _cash,
            TaxPosition: w, AfterTaxWealth: holdings + pending + _cash + w,
            LiquidationValue: holdings + pending + _cash + wLiq,
            LossSales: _lossSales, GainSales: _gainSales,
            BenefitUsedNow: _benefitUsedNow, BenefitBanked: _benefitBanked, GainTaxCost: _gainTaxCost,
            SumTradeDeltaW: _sumTradeW, RollTrueUp: _rollTrueUp,
            ExAnteTeMean: _days > 0 ? _sumTeExAnte / _days : 0, RealizedTe: te,
            AnnualTurnover: _initialValue > 0 ? (double)(_tradedDollars / _initialValue) / years : 0,
            TradingCosts: _oracle.CTrade * _lossSales,
            WashViolations: violations);
    }

    private bool HasHarvestableLot(string symbol, decimal close) =>
        _state.OpenLotsOf(symbol).Any(l => l.UnrealizedReturn(close) <= -_oracle.LossThreshold);

    // ── Private: portfolio initialisation ────────────────────────────────────

    private void InitializePortfolio(int day0, decimal totalValue)
    {
        var closes   = _prices.GetClosesDecimal(day0);
        int n        = closes.Count;
        if (n == 0) throw new InvalidOperationException("No price data on warmup day.");
        decimal perLot = totalValue / n;
        _initialValue  = totalValue;   // base for the exogenous contribution schedule
        _state.SetDate(_prices.GetDate(day0));

        foreach (var (symbol, price) in closes)
        {
            if (price <= 0m) continue;
            int shares = (int)(perLot / price);
            if (shares == 0) shares = 1;

            string sector = _prices.GetSector(symbol);
            var lot = new Lot(symbol, sector, price, shares, day0, _prices.GetDate(day0));
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
