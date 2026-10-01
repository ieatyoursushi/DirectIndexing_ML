namespace DirectIndexing.Core.Simulation.Volatility;

/// <summary>
/// The barrier coordinate of DataMemo/decisions/VolatilityModel_v03.md §6:
///   d = (ln(P_t / ((1−θ₁)·p_k)))⁺  — log-distance to the harvest trigger (0 once past it)
///   z = d / √V_{t,h}                — that distance in forecast standard deviations over h days
///   P = 2Φ(−z) = erfc(z/√2)          — reflection-principle probability that a driftless
///                                       log-price touches the trigger within h days.
/// A COORDINATE (the loss gate only — wash/TE/U gates ignored), never a label.
/// </summary>
// [math:z_barrier] — DataMemo/spec/SymbolTable.md
public static class LossBarrier
{
    public static double Distance(double price, double costBasis, double lossThreshold) =>
        costBasis > 0 && price > 0 ? Math.Max(0.0, Math.Log(price / ((1 - lossThreshold) * costBasis))) : double.NaN;

    public static double Z(double distance, double horizonVariance) =>
        horizonVariance > 0 ? distance / Math.Sqrt(horizonVariance) : double.NaN;

    public static double TouchProbability(double z) => double.IsNaN(z) ? double.NaN : Erfc(z / Math.Sqrt(2));

    /// <summary>Complementary error function (Numerical Recipes erfcc; |relative error| &lt; 1.2e-7).</summary>
    public static double Erfc(double x)
    {
        double z = Math.Abs(x), t = 1.0 / (1.0 + 0.5 * z);
        double ans = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
                     t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
                     t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? ans : 2.0 - ans;
    }
}

/// <summary>Precomputed EWMA σ̂ paths for every name and the equal-weight market — the engine's σ̂ state.</summary>
public sealed class VolState
{
    private readonly Dictionary<string, VolPath> _byName;
    private readonly VolPath _market;

    public VolState(PriceLoader prices, IVolEstimator? estimator = null)
    {
        var est = estimator ?? new EwmaVol();
        _byName = prices.Symbols.ToDictionary(s => s, s => est.Path(prices.GetReturnArray(s)));
        _market = est.Path(VolEval.MarketReturns(prices));
    }

    /// <summary>Annualized one-step σ̂ of a name at t (NaN during warm-up).</summary>
    public float SigmaHat(string symbol, int t) => Annualize(_byName[symbol].Var1[t]);
    public float SigmaMkt(int t) => Annualize(_market.Var1[t]);
    public double HorizonVariance(string symbol, int t, int h) => _byName[symbol].Horizon(t, h);

    private static float Annualize(double v) => double.IsNaN(v) ? float.NaN : (float)Math.Sqrt(252 * v);
}
