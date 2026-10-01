namespace DirectIndexing.Core.Simulation.Volatility;

/// <summary>
/// QLIKE(σ̂², σ̃²) = σ̃²/σ̂² − ln(σ̃²/σ̂²) − 1 ≥ 0, minimized at σ̂² = σ̃² — robust to an unbiased
/// noisy proxy σ̃² (Patton 2011): ranking forecasts by E[QLIKE] against r² gives the same ranking
/// as against the true variance. Proxy floored at <see cref="ProxyFloor"/> (a zero return would
/// send ln to −∞). (DataMemo/decisions/VolatilityModel_v03.md §4.)
/// </summary>
// [math:qlike] — DataMemo/spec/SymbolTable.md
public static class QLike
{
    public const double ProxyFloor = 1e-10;

    public static double Loss(double forecast, double proxy)
    {
        double ratio = Math.Max(proxy, ProxyFloor) / forecast;
        return ratio - Math.Log(ratio) - 1.0;
    }
}
