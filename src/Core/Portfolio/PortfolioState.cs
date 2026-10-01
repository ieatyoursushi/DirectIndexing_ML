namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// The complete portfolio state triple  𝒮_t = (μ_t, ledger_t, 𝒲_t):
///
///   μ_t      = OpenLots              — full lot measure across all assets
///   ledger_t = Ledger                — TaxLedger: net realized P&amp;L, loss
///                                      carryforward, ordinary-offset budget
///   𝒲_t      = wash-sale state       — last loss-sale date per ticker + open-lot
///                                      acquisition dates (calendar), see WashClock
///
/// v0.25: the bare G_YTD scalar became the TaxLedger (issue #23); its read alias
/// and the external-gains seed were retired with the gated oracle (pre-v0.3
/// downsizing).
///
/// Sign convention for Ledger.RealizedGainsYTD (G^net_t):
///   Harvesting a LOSING lot contributes a NEGATIVE delta (currentPrice &lt; CostBasis).
///   It oscillates throughout the year as gains are realised and losses are
///   harvested against them; the oracle reads it only through taxValue's
///   offset capacity cap_t = max(G^net_t, 0) + O_t.
///
/// §1091 (v0.3-1, ROADMAP F7): the wash-sale window is ±30 CALENDAR days around a loss
/// sale, on BOTH sides. Time enters only through <see cref="SetDate"/>; clocks are date
/// differences, never trading-day counts.
///   • before-side — a lot may not be harvested within 30 days after a DIFFERENT lot of
///     the ticker was acquired (that lot would be the replacement);
///   • after-side  — the ticker may not be bought within 30 days after a loss sale
///     (<see cref="CanBuy"/>).
/// HarvestLot() implements the state transition: remove the atom from μ_t, record
/// realized P&amp;L in the ledger, and (for a loss) date-stamp the ticker's loss sale.
/// </summary>
// [math:state] — DataMemo/spec/SymbolTable.md
public class PortfolioState
{
    // ledger_t — deterministic Schedule D bookkeeping (see TaxLedger)
    public TaxLedger Ledger { get; } = new();

    /// <summary>§1091 window half-width in calendar days: clean iff distance &gt; 30.</summary>
    public const int WashWindowDays = 30;

    /// <summary>Sentinel clock value: no wash-relevant event on record.</summary>
    public const int NeverClock = 999;

    /// <summary>
    /// Re-harvest guard (v0.3-2b). When on (default, the v0.1–v0.3-1 behavior), 𝒲 also counts
    /// days since the ticker's own last loss sale, so a second lot of a ticker cannot be
    /// harvested within 30 days of the first. That is STRICTER than §1091: a loss sale is
    /// disallowed only by a replacement ACQUISITION within ±30 days, and selling another
    /// old lot acquires nothing. Off (--no-reharvest-guard) = the law's two sides only;
    /// the after-side (<see cref="CanBuy"/>) is unaffected either way.
    /// </summary>
    public bool ReharvestGuard { get; init; } = true;

    // Last LOSS sale per ticker (gain sales never start a wash window).
    private readonly Dictionary<string, DateOnly> _lastLossSale = new();

    // Acquisition dates of lots of a ticker that were bought AND sold within the last 30
    // calendar days. A closed lot is still a §1091 replacement for another lot's loss sale
    // if it was acquired inside that sale's window (Reg. 1.1091-1) — the open-lot scan alone
    // misses it. Under the re-harvest guard this set is redundant (the closed lot's own sale
    // came after its purchase, so DaysSinceLossSale is always smaller); without it, it is the law.
    private readonly Dictionary<string, List<DateOnly>> _recentClosedBuys = new();

    // μ_t = { atoms currently open }, plus a per-ticker index of the same atoms
    public List<Lot> OpenLots { get; } = new();
    private readonly Dictionary<string, List<Lot>> _openBySymbol = new();

    /// <summary>The simulation's current calendar date (set once per day by the engine).</summary>
    public DateOnly Today { get; private set; }

    public void SetDate(DateOnly today) => Today = today;

    // ─── Wash-sale state (§1091, calendar days, both sides) ────────────────────

    /// <summary>Calendar days since the ticker's last loss sale (<see cref="NeverClock"/> if none).</summary>
    public int DaysSinceLossSale(string symbol) =>
        _lastLossSale.TryGetValue(symbol, out var d)
            ? Math.Min(NeverClock, Today.DayNumber - d.DayNumber)
            : NeverClock;

    /// <summary>
    /// 𝒲 for one lot — the calendar distance to the nearest wash-relevant event:
    /// min(days since the ticker's last loss sale [only under <see cref="ReharvestGuard"/>],
    /// days since the most recent acquisition of a DIFFERENT lot of the ticker — open, or
    /// closed but acquired within the window), capped at 999.
    /// Harvesting the lot is wash-clean iff 𝒲 &gt; 30.
    /// </summary>
    // [math:wash_clock] — DataMemo/spec/SymbolTable.md
    public int WashClock(Lot lot)
    {
        int clock = ReharvestGuard ? DaysSinceLossSale(lot.Symbol) : NeverClock;
        if (_openBySymbol.TryGetValue(lot.Symbol, out var lots))
            foreach (var other in lots)
                if (!ReferenceEquals(other, lot))
                    clock = Math.Min(clock, Today.DayNumber - other.PurchaseDate.DayNumber);
        if (_recentClosedBuys.TryGetValue(lot.Symbol, out var closed))
            foreach (var bought in closed)
                clock = Math.Min(clock, Today.DayNumber - bought.DayNumber);
        return Math.Clamp(clock, 0, NeverClock);
    }

    /// <summary>After-side of §1091: may the ticker be BOUGHT today without disallowing a recent loss?</summary>
    // [math:can_buy] — DataMemo/spec/SymbolTable.md
    public bool CanBuy(string symbol) => DaysSinceLossSale(symbol) > WashWindowDays;

    /// <summary>First calendar date on which <see cref="CanBuy"/> turns true (today if it already is).</summary>
    public DateOnly EarliestBuyDate(string symbol) =>
        _lastLossSale.TryGetValue(symbol, out var d) && !CanBuy(symbol)
            ? d.AddDays(WashWindowDays + 1)
            : Today;

    /// <summary>Open lots of one ticker (empty if none).</summary>
    public IReadOnlyList<Lot> OpenLotsOf(string symbol) =>
        _openBySymbol.TryGetValue(symbol, out var lots) ? lots : Array.Empty<Lot>();

    // ─── State transitions ───────────────────────────────────────────────────

    public void OpenLot(Lot lot)
    {
        OpenLots.Add(lot);
        if (!_openBySymbol.TryGetValue(lot.Symbol, out var lots))
            _openBySymbol[lot.Symbol] = lots = new List<Lot>();
        lots.Add(lot);
    }

    /// <summary>
    /// Realise the P&amp;L of a lot and remove it from the measure.
    /// ΔG = q_k · (P_t − p_k)  — negative when harvesting a loss.
    /// secondary goal: two TE reductable routes of either repurchasing back after the 30d wash sale rule timer or replacing the harvested with a colinear asset that meets as many colinearity conditions as possible (like simlar sector, weight, covariance, etc.). The reducable can be chosen from the lots that got reduced away in the step 0 PCA/K-means dimensionality reduction.
    /// </summary>
    // [math:harvest] — DataMemo/spec/SymbolTable.md
    public void HarvestLot(Lot lot, decimal currentPrice)
    {
        var gain = (currentPrice - lot.CostBasis) * lot.Shares;
        Ledger.RecordRealized(gain);   // negative delta for a loss — sign is self-consistent
        lot.IsOpen    = false;
        OpenLots.Remove(lot);
        _openBySymbol[lot.Symbol].Remove(lot);
        if (gain < 0m)
            _lastLossSale[lot.Symbol] = Today;   // a LOSS sale opens the §1091 window

        // a lot bought within the window stays a potential replacement after it is sold
        if (Today.DayNumber - lot.PurchaseDate.DayNumber <= WashWindowDays)
        {
            if (!_recentClosedBuys.TryGetValue(lot.Symbol, out var closed))
                _recentClosedBuys[lot.Symbol] = closed = new List<DateOnly>();
            closed.RemoveAll(d => Today.DayNumber - d.DayNumber > WashWindowDays);
            closed.Add(lot.PurchaseDate);
        }
    }

    // ─── Derived quantities ──────────────────────────────────────────────────

    public decimal PortfolioValue(Dictionary<string, decimal> currentPrices) =>
        OpenLots.Sum(lot => lot.Shares * currentPrices[lot.Symbol]);

    /// <summary>
    /// Year boundary (Jan 1): roll the ledger — net loss beyond the $3k
    /// ordinary allowance banks into LossCarryforward (which survives),
    /// the annual accumulator resets to 0.
    /// Wash-sale clocks intentionally persist — the IRS window crosses year-end.
    /// </summary>
    public void ResetForNewYear() =>
        Ledger.RollYearEnd();
}
