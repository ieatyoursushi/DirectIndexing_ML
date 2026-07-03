// Tests/OracleScalarizedTests.cs
using System.Diagnostics;
using DirectIndexing.Core.Oracle;

/// <summary>
/// Unit tests for the v0.25 scalarized oracle (issue #23):
/// f*(x) = 𝟙[ℓ ≤ −θ₁] · 𝟙[𝒲 ≥ 30] · 𝟙[σ_TE ≤ θ_max] · 𝟙[U(x) &gt; 0],
/// U(x) = taxValue − λσ_TE² − c_trade.
/// </summary>
public class OracleScalarizedTests
{
    private static readonly OracleConfig Cfg = OracleConfig.Scalarized;

    // Test 1: the U > 0 level set replaces the gains gate — fires with zero
    // net realized gains (the exact case the old gate wrongly vetoed).
    public void Test_Fires_WithoutRealizedGains()
    {
        // Deep loss, calm TE, wash clear, taxValue well above the λσ² penalty.
        // netRealizedYtd = 0 — the old gate 3 would block this row.
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.05m,
            sigmaTE:           0.025f,          // penalty = 90000·0.000625 ≈ $56
            netRealizedYtd:    0m,
            washClock:         999,
            taxValue:          400m,            // U ≈ 400 − 56 > 0
            config:            Cfg);
        Debug.Assert(label == 1,
            "Scalarized oracle must fire on a valuable loss with zero realized gains");
        Console.WriteLine("Scalarized Test 1 passed: fires without realized gains (gains gate gone)");
    }

    // Test 2: U ≤ 0 blocks — marginal taxValue loses to the TE penalty even
    // though every hard gate is open. The boundary is the level set {U = 0}.
    public void Test_Blocked_WhenUtilityNegative()
    {
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.05m,
            sigmaTE:           0.05f,           // penalty = 90000·0.0025 = $225
            netRealizedYtd:    1_000_000m,      // irrelevant in scalarized mode
            washClock:         999,
            taxValue:          150m,            // U = 150 − 225 < 0
            config:            Cfg);
        Debug.Assert(label == 0,
            "Scalarized oracle must block when taxValue < λσ_TE² (U ≤ 0)");
        Console.WriteLine("Scalarized Test 2 passed: blocked when U(x) ≤ 0");
    }

    // Test 3: the economic tradeoff is continuous — the same taxValue that
    // fails at elevated TE clears at calm TE (no box corner, a curved boundary).
    public void Test_TradeOff_TaxValueVsTrackingError()
    {
        int atCalmTE = OracleBoundary.Label(-0.05m, 0.025f, 0m, 999, 150m, Cfg);
        int atHighTE = OracleBoundary.Label(-0.05m, 0.050f, 0m, 999, 150m, Cfg);
        Debug.Assert(atCalmTE == 1 && atHighTE == 0,
            $"Same lot must clear at calm TE and fail at elevated TE, got {atCalmTE}/{atHighTE}");
        Console.WriteLine("Scalarized Test 3 passed: same taxValue clears at calm TE, fails at high TE");
    }

    // Test 4: θ_max is a tail-only circuit breaker — even an enormous taxValue
    // cannot buy unlimited benchmark deviation.
    public void Test_HardCeiling_BindsInPathologicalRegimes()
    {
        // σ_TE = 20% — beyond the 15% ceiling (never observed in 20y of data;
        // this is the v0.0 33%-TE-artifact class of pathology).
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.50m,
            sigmaTE:           0.20f,
            netRealizedYtd:    0m,
            washClock:         999,
            taxValue:          1_000_000m,      // U ≫ 0 — but the ceiling vetoes
            config:            Cfg);
        Debug.Assert(label == 0,
            "θ_max ceiling must veto regardless of how large U(x) is");
        Console.WriteLine("Scalarized Test 4 passed: θ_max circuit breaker binds in pathological regimes");
    }

    // Test 5: the two legal/threshold hard gates survive unchanged.
    public void Test_LossAndWashGates_StillBind()
    {
        // Loss too shallow (−1% > −θ₁)
        int shallow = OracleBoundary.Label(-0.01m, 0.02f, 0m, 999, 500m, Cfg);
        Debug.Assert(shallow == 0, "Loss-depth gate must still bind");

        // Wash-sale window active (clock 15 < 30)
        int washed = OracleBoundary.Label(-0.05m, 0.02f, 0m, 15, 500m, Cfg);
        Debug.Assert(washed == 0, "Wash-sale gate must still bind (IRS §1091)");

        Console.WriteLine("Scalarized Test 5 passed: loss-depth and wash-sale hard gates unchanged");
    }

    // Test 6: gated mode through the SAME config-driven entry point is
    // bit-identical to the legacy overload (the ablation baseline contract).
    public void Test_GatedMode_MatchesLegacyOverload()
    {
        var cases = new (decimal ell, float te, decimal gYtd, int wash)[]
        {
            (-0.05m, 0.01f,  500m, 999),   // all gates open → 1
            (-0.05m, 0.01f,    0m, 999),   // gains gate closed → 0
            (-0.05m, 0.06f,  500m, 999),   // legacy TE cap binds → 0
            (-0.01m, 0.01f,  500m, 999),   // loss too shallow → 0
            (-0.05m, 0.01f,  500m,  10),   // wash active → 0
        };
        foreach (var (ell, te, gYtd, wash) in cases)
        {
            int legacy = OracleBoundary.Label(ell, te, gYtd, wash);
            int routed = OracleBoundary.Label(ell, te, gYtd, wash,
                                              taxValue: 12345m, OracleConfig.Gated);
            Debug.Assert(legacy == routed,
                $"Gated mode must ignore taxValue and match legacy: case ({ell},{te},{gYtd},{wash})");
        }
        Console.WriteLine("Scalarized Test 6 passed: gated mode bit-identical to legacy overload");
    }

    // Test 7: Utility arithmetic — default-agnostic on CTrade so the PR-3
    // calibration ($10 round-trip default) doesn't silently break the identity.
    public void Test_Utility_Arithmetic_And_CTrade()
    {
        decimal u = OracleBoundary.Utility(400m, 0.05f, Cfg);
        Debug.Assert(u == 400m - Cfg.Lambda * 0.0025m - Cfg.CTrade,
            $"U must equal taxValue − λσ² − c_trade, got {u}");

        var withCost = Cfg with { CTrade = Cfg.CTrade + 100m };
        decimal uc = OracleBoundary.Utility(400m, 0.05f, withCost);
        Debug.Assert(uc == u - 100m, "c_trade must subtract additively from U");

        var freeTrade = Cfg with { CTrade = 0m };
        decimal uf = OracleBoundary.Utility(400m, 0.05f, freeTrade);
        Debug.Assert(uf == 400m - Cfg.Lambda * 0.0025m,
            "the --ctrade=0 ablation arm must recover the frictionless U");

        Console.WriteLine("Scalarized Test 7 passed: U = taxValue − λσ² − c_trade");
    }
}
