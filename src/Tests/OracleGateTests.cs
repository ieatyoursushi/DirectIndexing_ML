// Tests/OracleGateTests.cs
// Each test isolates a single oracle gate — one condition is the binding constraint,
// all others are deliberately satisfied.  This way a failure immediately identifies
// which gate is broken without cross-condition ambiguity.
using System.Diagnostics;
using DirectIndexing.Core.Oracle;
//model generated tests - good to have to interpret the model's underlying assumptions when validating the implementation. To be reviewd.
public class OracleBoundaryTests
{
    // ── Helper: known-good baseline (all four gates open) ───────────────────
    //
    //   ℓ = −0.05  (5% loss    — past the 2% threshold)
    //   σ_TE = 0.01 (1%        — within the 5% budget)
    //   G_YTD = 500m           — positive (gains to offset)
    //   washClock = 35         — past the 30-day window
    //   Expected: 1
    //
    private static int Baseline() =>
        OracleBoundary.Label(-0.05m, 0.01f, 500m, 35);

    // Test 1: all four conditions satisfied → oracle fires
    public void Test_Oracle_FiresWhenAllConditionsMet()
    {
        int label = Baseline();
        Debug.Assert(label == 1,
            $"Oracle should fire (1) when all gates open — got {label}");
        Console.WriteLine("OracleGate Test 1 passed: fires when all conditions met");
    }

    // Test 2: loss insufficient (ℓ > −θ₁) — first gate closed
    // Binding constraint: unrealizedReturn = −0.01 (only 1% loss, below 2% threshold)
    public void Test_Oracle_Blocked_WhenLossInsufficient()
    {
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.01m,   // < LossThreshold of 2% → gate closed
            sigmaTE:           0.01f,
            gYtd:              500m,
            washClock:         35
        );
        Debug.Assert(label == 0,
            $"Oracle should NOT fire when loss < θ₁ — got {label}");
        Console.WriteLine("OracleBoundary Test 2 passed: blocked when loss insufficient");
    }

    // Test 3: tracking error over budget — second gate closed
    // Binding constraint: sigmaTE = 0.08f (8% > 5% cap)
    public void Test_Oracle_Blocked_WhenTEOverBudget()
    {
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.05m,
            sigmaTE:           0.08f,   // > TrackingErrorCap of 5% → gate closed
            gYtd:              500m,
            washClock:         35
        );
        Debug.Assert(label == 0,
            $"Oracle should NOT fire when σ_TE > θ₂ — got {label}");
        Console.WriteLine("OracleBoundary Test 3 passed: blocked when tracking error over budget");
    }

    // Test 4: G_YTD non-positive — third gate closed
    // Binding constraint: gYtd = 0 (exactly zero — boundary case, oracle requires > 0)
    public void Test_Oracle_Blocked_WhenGYTD_Zero()
    {
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.05m,
            sigmaTE:           0.01f,
            gYtd:              0m,      // not > 0 → gate closed
            washClock:         35
        );
        Debug.Assert(label == 0,
            $"Oracle should NOT fire when G_YTD = 0 — got {label}");
        Console.WriteLine("OracleBoundary Test 4 passed: blocked when G_YTD = 0");
    }

    // Test 5: wash-sale window active — fourth gate closed
    // Binding constraint: washClock = 29 (one day short of the 30-day window)
    public void Test_Oracle_Blocked_WhenWashSaleActive()
    {
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.05m,
            sigmaTE:           0.01f,
            gYtd:              500m,
            washClock:         29       // < WashSaleDays of 30 → gate closed
        );
        Debug.Assert(label == 0,
            $"Oracle should NOT fire when washClock < 30 — got {label}");
        Console.WriteLine("OracleBoundary Test 5 passed: blocked when wash-sale window active");
    }

    // Test 6: washClock at exactly 30 — boundary condition, should fire
    public void Test_Oracle_Fires_AtWashSaleBoundary()
    {
        int label = OracleBoundary.Label(
            unrealizedReturn: -0.05m,
            sigmaTE:           0.01f,
            gYtd:              500m,
            washClock:         OracleBoundary.WashSaleDays   // = 30, exactly at boundary
        );
        Debug.Assert(label == 1,
            $"Oracle should fire at exactly washClock = 30 — got {label}");
        Console.WriteLine("OracleBoundary Test 6 passed: fires at wash-sale boundary (clock = 30)");
    }
}
