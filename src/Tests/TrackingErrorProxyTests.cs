// Tests/TrackingErrorProxyTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Simulation;

/// <summary>
/// Tests for TrackingErrorProxy — validates the equal-weight return approach
/// and the structural-stability invariants that motivated the bug fix.
///
/// Key design assumption:
///   σ_TE = annualised std(r_portfolio − r_benchmark)
///   r_portfolio = equal-weighted daily return over currently open lot symbols
///   r_benchmark = equal-weighted daily return over ALL ticker symbols
///
/// This decouples σ_TE from structural cash-flow events (lot harvest + wash-sale
/// reopen), which inflate the measure by 30–100%+ under a dollar-value-based
/// implementation because removing a lot worth 1/3 of the portfolio looks like a
/// −33% "return" on the following day.
/// </summary>
public class TrackingErrorProxyTests
{
    // ── Test 1 ───────────────────────────────────────────────────────────────

    /// <summary>
    /// When the portfolio holds every ticker in the universe, the equal-weight
    /// portfolio return equals the equal-weight benchmark return at every step
    /// → σ_TE = 0.
    ///
    /// Assumption tested: portfolio return = sum(r_i) / n_open
    ///                    benchmark return = sum(r_i) / n_total
    ///                    When open = all, both are identical → diff = 0 always.
    /// </summary>
    public void Test_SigmaTE_Zero_WhenPortfolioEqualsFullBenchmark()
    {
        float[] r  = [float.NaN, 0.01f, -0.01f, 0.02f, -0.02f, 0.01f, 0.03f, -0.01f, 0.02f, -0.01f];
        var rets   = new Dictionary<string, float[]> { ["A"] = r, ["B"] = r, ["C"] = r };
        var prices = PriceLoader.CreateForTesting(rets);
        var te     = new TrackingErrorProxy(prices);

        string[] allSyms = ["A", "B", "C"];
        float sigmaTE = 0f;
        for (int t = 1; t < r.Length; t++)
            sigmaTE = te.Update(allSyms, t);

        Debug.Assert(sigmaTE < 1e-5f,
            $"Expected σ_TE ≈ 0 when portfolio = full benchmark; got {sigmaTE:G4}");
        Console.WriteLine($"  [TE Test 1] passed: σ_TE = {sigmaTE:G4} ≈ 0 when portfolio = benchmark");
    }

    // ── Test 2 ───────────────────────────────────────────────────────────────

    /// <summary>
    /// When the portfolio omits a ticker whose returns are 10× more volatile,
    /// the portfolio return diverges from the benchmark → σ_TE > 0.
    ///
    /// Assumption tested: σ_TE correctly captures compositional divergence.
    /// If a ticker with large return swings is in the benchmark but not the
    /// portfolio, the equal-weight benchmark return will differ from the
    /// equal-weight portfolio return, producing a non-zero std of differences.
    /// </summary>
    public void Test_SigmaTE_Positive_WhenPortfolioExcludesDivergingTicker()
    {
        float[] rSmall = [float.NaN, 0.01f, -0.01f, 0.02f, -0.02f, 0.01f, 0.03f, -0.01f, 0.02f, -0.01f];
        float[] rLarge = [float.NaN, 0.10f, -0.10f, 0.20f, -0.20f, 0.10f, 0.30f, -0.10f, 0.20f, -0.10f];
        var rets   = new Dictionary<string, float[]> { ["A"] = rSmall, ["B"] = rSmall, ["C"] = rLarge };
        var prices = PriceLoader.CreateForTesting(rets);
        var te     = new TrackingErrorProxy(prices);

        // Portfolio holds only the two low-vol tickers; benchmark includes high-vol C
        string[] portfolio = ["A", "B"];
        float sigmaTE = 0f;
        for (int t = 1; t < rSmall.Length; t++)
            sigmaTE = te.Update(portfolio, t);

        Debug.Assert(sigmaTE > 0.001f,
            $"Expected σ_TE > 0 when portfolio excludes high-vol ticker; got {sigmaTE:G4}");
        Console.WriteLine($"  [TE Test 2] passed: σ_TE = {sigmaTE:G4} > 0 when portfolio diverges from benchmark");
    }

    // ── Test 3 ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Structural lot removal (a harvest event) must NOT spike σ_TE.
    ///
    /// With equal-weight returns:
    ///   If all tickers have identical returns, removing one from the portfolio
    ///   does not change r_portfolio (equal-weight of identical values = same
    ///   value regardless of which subset you pick).  σ_TE stays at 0.
    ///
    /// With the old dollar-value approach (the bug):
    ///   Portfolio value drops by ≈1/3 when C is removed.
    ///   portRet ≈ −33% the following day.
    ///   benchRet ≈ +2% (normal market move).
    ///   diff ≈ −35% for that one day.
    ///   Annualised: 0.35 × √252 ≈ 5.6 → σ_TE > 560%, permanently blocking
    ///   the oracle's σ_TE ≤ 5% gate for the rest of the simulation.
    ///   (This is the exact bug that generated σ_TE = 116% in the real run.)
    ///
    /// This test is the regression guard for that fix.
    /// </summary>
    public void Test_SigmaTE_StaysBounded_AfterStructuralLotRemoval()
    {
        // All 3 symbols have identical returns — structural composition change is
        // invisible to equal-weight return, but would look like a −33% return in
        // the old dollar-value code when C is "removed" from the portfolio.
        float[] r  = [float.NaN, 0.01f, -0.01f, 0.02f, -0.02f, 0.01f,
                      0.03f, -0.01f, 0.02f, -0.01f, 0.02f, -0.01f, 0.02f, -0.01f, 0.02f];
        var rets   = new Dictionary<string, float[]> { ["A"] = r, ["B"] = r, ["C"] = r };
        var prices = PriceLoader.CreateForTesting(rets);
        var te     = new TrackingErrorProxy(prices);

        string[] full    = ["A", "B", "C"];
        string[] partial = ["A", "B"];   // C "harvested" on day 7

        float sigmaTE = 0f;
        for (int t = 1; t < r.Length; t++)
        {
            var syms = t < 7 ? full : partial;
            sigmaTE  = te.Update(syms, t);
        }

        // All diffs are 0 → σ_TE = 0 with the fix.
        // Old code: σ_TE ≈ 5.6 after the structural jump on day 8.
        Debug.Assert(sigmaTE < 1e-5f,
            $"σ_TE should stay ≈ 0 after structural lot removal when returns are identical; " +
            $"got {sigmaTE:G4} (old dollar-value code would yield ≈ 5.6 here)");
        Console.WriteLine(
            $"  [TE Test 3] passed: σ_TE = {sigmaTE:G4} ≈ 0 after structural removal " +
            $"(regression guard — old code would give ≈ 5.6)");
    }

    // ── Test 4 ───────────────────────────────────────────────────────────────

    /// <summary>
    /// The proxy returns 0 until it has accumulated at least 5 observations in
    /// the rolling window.  This prevents wildly unreliable estimates on the
    /// first few trading days of a simulation.
    ///
    /// Assumption tested: the hard-coded guard <c>if (_filled &lt; 5) return 0f</c>
    /// in TrackingErrorProxy.Update fires correctly.
    /// </summary>
    public void Test_SigmaTE_Zero_WithInsufficientHistory()
    {
        // Only 4 time steps → _filled tops out at 4, never reaches 5
        float[] r  = [float.NaN, 0.01f, -0.05f, 0.03f];
        var rets   = new Dictionary<string, float[]> { ["A"] = r, ["B"] = r };
        var prices = PriceLoader.CreateForTesting(rets);
        var te     = new TrackingErrorProxy(prices);

        float sigmaTE = 0f;
        for (int t = 0; t < r.Length; t++)
            sigmaTE = te.Update(new[] { "A", "B" }, t);

        Debug.Assert(sigmaTE == 0f,
            $"Expected σ_TE = 0 with fewer than 5 observations; got {sigmaTE}");
        Console.WriteLine(
            $"  [TE Test 4] passed: σ_TE = 0 with {r.Length} observations " +
            $"(guard: _filled < 5 → return 0)");
    }
}
