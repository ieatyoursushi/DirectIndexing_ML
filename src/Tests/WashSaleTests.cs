using System.Diagnostics;
using DirectIndexing.Core.Portfolio;
using DirectIndexing.Core.Simulation;

/// <summary>
/// §1091 — the independent audit (a restatement of the law over a trade log) and,
/// through it, the engine's wash-sale gating. SymbolTable rows `wash_audit`, `wash_clock`.
/// </summary>
public class WashSaleTests
{
    private static readonly DateOnly D0 = new(2024, 3, 1);

    private static TradeEvent Buy(Lot lot, int dayOffset) =>
        new(D0.AddDays(dayOffset), dayOffset, lot.Symbol, TradeKind.Buy, lot, lot.CostBasis, 0m);

    private static TradeEvent Sell(Lot lot, int dayOffset, decimal gain) =>
        new(D0.AddDays(dayOffset), dayOffset, lot.Symbol, TradeKind.Sell, lot, lot.CostBasis, gain);

    // The audit's definition, pinned at its edges: ±30 calendar days inclusive, a
    // DIFFERENT lot of the same ticker, loss sales only.
    public void Test_Audit_WindowEdges_SameLot_AndGains()
    {
        var old = new Lot("AAA", "X", 100m, 10, 0);
        var fresh = new Lot("AAA", "X", 90m, 10, 0);

        // replacement bought 30 days BEFORE a loss sale → violation (inside the window)
        int v1 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(fresh, 0), Sell(old, 30, -50m) }).Count;
        // …31 days before → clean
        int v2 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(fresh, 0), Sell(old, 31, -50m) }).Count;
        // replacement bought 30 days AFTER → violation; 31 days after → clean
        int v3 = WashSaleAudit.Violations(new[] { Buy(old, -400), Sell(old, 0, -50m), Buy(fresh, 30) }).Count;
        int v4 = WashSaleAudit.Violations(new[] { Buy(old, -400), Sell(old, 0, -50m), Buy(fresh, 31) }).Count;
        // selling the SAME lot you just bought, at a loss, is not a wash sale
        int v5 = WashSaleAudit.Violations(new[] { Buy(fresh, 0), Sell(fresh, 10, -50m) }).Count;
        // a GAIN sale next to a buy is not a wash sale
        int v6 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(fresh, 0), Sell(old, 5, +50m) }).Count;
        // another ticker's buy is irrelevant
        var other = new Lot("BBB", "X", 50m, 10, 0);
        int v7 = WashSaleAudit.Violations(new[] { Buy(old, -400), Buy(other, 0), Sell(old, 5, -50m) }).Count;

        Debug.Assert(v1 == 1 && v2 == 0, $"before-side edge: expected 1/0, got {v1}/{v2}");
        Debug.Assert(v3 == 1 && v4 == 0, $"after-side edge: expected 1/0, got {v3}/{v4}");
        Debug.Assert(v5 == 0 && v6 == 0 && v7 == 0, $"exclusions: expected 0/0/0, got {v5}/{v6}/{v7}");
        Console.WriteLine("WashSale Test 1 passed: audit window is ±30 d_cal inclusive; same lot, gains, other tickers excluded");
    }

    // The before-side on PortfolioState: a DIFFERENT lot acquired within 30 calendar days
    // makes the older lot unharvestable; the fresh lot itself is not its own replacement.
    public void Test_State_BeforeSide_LotLevelClock()
    {
        var d0 = new DateOnly(2024, 3, 1);
        var state = new PortfolioState();
        state.SetDate(d0);
        var old   = new Lot("AAA", "X", 100m, 10, 0, d0.AddDays(-400));
        var fresh = new Lot("AAA", "X",  80m, 10, 0, d0);
        state.OpenLot(old);
        state.OpenLot(fresh);

        state.SetDate(d0.AddDays(30));
        int oldAt30 = state.WashClock(old), freshAt30 = state.WashClock(fresh);
        state.SetDate(d0.AddDays(31));
        int oldAt31 = state.WashClock(old);

        Debug.Assert(oldAt30 == 30, $"old lot's clock is the fresh buy 30 days ago, got {oldAt30}");
        Debug.Assert(oldAt31 == 31, $"…and 31 a day later, got {oldAt31}");
        Debug.Assert(freshAt30 == 430, $"the fresh lot sees only the OTHER lot's purchase (430 d), got {freshAt30}");

        // a gain sale never opens a window
        state.HarvestLot(fresh, 120m);
        Debug.Assert(state.CanBuy("AAA"), "a gain sale must not block buying");
        Console.WriteLine("WashSale Test 2 passed: lot-level clock (before-side) and gain sales exempt");
    }

    // Acceptance: after v0.3-1 the independent audit finds ZERO wash sales on the worlds
    // where it found 24.3% (weekday calendar, contributions) and 97.7% (daily calendar,
    // day-30 reopen) before the fix — with and without the contribution skip rule.
    public void Test_Engine_ZeroViolations_OnWorldsThatHadThem()
    {
        var weekday = PriceLoader.FromGbm(PriceLoader.UniformGbmUniverse(60, 0.35f), 1260, seed: 7);
        var contrib = ContributionPolicy.Off with { Enabled = true, IntervalDays = 21, NamesPerContribution = 5 };
        foreach (var skip in new[] { true, false })
        {
            var e = new SimulationEngine(weekday, contributionPolicy: contrib with { SkipHarvestableNames = skip });
            e.Run(10_000_000m);
            int losses = e.Trades.Count(x => x.Kind == TradeKind.Sell && x.RealizedGain < 0m);
            var v = WashSaleAudit.Violations(e.Trades);
            Debug.Assert(losses > 100, $"world must actually harvest (got {losses})");
            Debug.Assert(v.Count == 0, $"contrib (skip={skip}): {v.Count} §1091 violations of {losses}");
            Console.WriteLine($"WashSale Test 3 passed: weekday contrib world (skip={skip}): 0 violations in {losses} loss sales");
        }

        var daily = DailyCalendarWorld();
        var eng = new SimulationEngine(daily);
        eng.Run(10_000_000m);
        int n = eng.Trades.Count(x => x.Kind == TradeKind.Sell && x.RealizedGain < 0m);
        int bad = WashSaleAudit.Violations(eng.Trades).Count;
        Debug.Assert(n > 100 && bad == 0, $"daily-calendar world: {bad} violations of {n}");
        Console.WriteLine($"WashSale Test 4 passed: daily-calendar world: 0 violations in {n} loss sales (reopen at +31)");
    }

    // A world whose calendar has every day (so trading day = calendar day), with two
    // drawdowns — the configuration in which a +30-trading-day reopen violated §1091.
    private static PriceLoader DailyCalendarWorld()
    {
        const int N = 30, T = 700;
        var rng = new Random(20260930);
        double G() => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
        var mkt = new double[T];
        for (int t = 1; t < T; t++)
            mkt[t] = ((t > 300 && t < 360) ? -0.004 : (t > 520 && t < 560 ? -0.006 : 0.0004)) + 0.012 * G();
        var d = new Dictionary<string, float[]>();
        for (int i = 0; i < N; i++)
        {
            var r = new float[T]; r[0] = float.NaN;
            for (int t = 1; t < T; t++) r[t] = (float)((0.6 + 0.02 * i) * mkt[t] + 0.012 * G());
            d[$"D{i:D2}"] = r;
        }
        return PriceLoader.CreateForTesting(d);
    }
}
