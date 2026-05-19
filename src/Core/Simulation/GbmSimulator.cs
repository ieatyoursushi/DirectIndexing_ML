namespace DirectIndexing.Core.Simulation;

/// <summary>
/// A self-contained, testable Geometric Brownian Motion (GBM) simulation engine.
///
/// The model follows the log-normal SDE:
///   dS = μ S dt + σ S dW
///
/// In discrete form with step size Δ = 1/252 (one trading day):
///   S_{t+Δ} = S_t · exp((μ − σ²/2)·Δ + σ·√Δ·Z),   Z ~ N(0,1)
///
/// The (μ − σ²/2) term is the Itô drift correction — it ensures the median
/// (not the mean) of the log-price process tracks the risk-neutral drift μ.
/// Without it the simulated paths would have an upward bias equal to σ²/2·T.
///
/// Typical usage:
///   var sim = new GbmSimulator(paths: 200, horizon: 30);
///   float prob = sim.FractionFiring(startPrice, annualSigma, (p, s) => p / basis &lt; 0.98f);
///
/// For the full path matrix use <see cref="SimulatePaths"/>.
/// </summary>
public sealed class GbmSimulator
{
    // Number of trading days per year — used for time-step scaling and annualisation.
    private const float TradingDays = 252f;

    /// <summary>Number of Monte Carlo paths per call.</summary>
    public int Paths { get; }

    /// <summary>Forward simulation horizon in trading days.</summary>
    public int Horizon { get; }

    /// <summary>
    /// Annualised risk-neutral drift μ.  Use 0 (default) for risk-neutral pricing.
    /// A positive value tilts paths upward; negative tilts downward.
    /// </summary>
    public float AnnualDrift { get; }

    /// <summary>
    /// Creates a new <see cref="GbmSimulator"/>.
    /// </summary>
    /// <param name="paths">Number of Monte Carlo paths to simulate per call (default 200).</param>
    /// <param name="horizon">Forward simulation horizon in trading days (default 30).</param>
    /// <param name="annualDrift">
    /// Annualised continuous drift μ.  Under risk-neutral pricing this is 0;
    /// supply a non-zero value to embed a drift view (e.g. expected excess return).
    /// </param>
    public GbmSimulator(int paths = 200, int horizon = 30, float annualDrift = 0f)
    {
        if (paths   <= 0) throw new ArgumentOutOfRangeException(nameof(paths),   "Must be positive.");
        if (horizon <= 0) throw new ArgumentOutOfRangeException(nameof(horizon), "Must be positive.");

        Paths       = paths;
        Horizon     = horizon;
        AnnualDrift = annualDrift;
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// <summary>
    /// Simulates <see cref="Paths"/> independent GBM paths of length
    /// <see cref="Horizon"/>+1 and returns the full price matrix.
    ///
    /// Index convention:
    ///   result[p][0]   = <paramref name="startPrice"/> (the known current price)
    ///   result[p][s]   = simulated price at step s ∈ [1, Horizon]
    ///
    /// The returned jagged array is freshly allocated on every call — callers may
    /// read it freely without taking a defensive copy.
    /// </summary>
    /// <param name="startPrice">Current price S_0 (must be &gt; 0).</param>
    /// <param name="annualSigma">
    /// Annualised volatility σ (must be &gt; 0).  A reasonable fallback when
    /// calibration data is unavailable is 0.20 (20%).
    /// </param>
    /// <param name="rng">
    /// Optional <see cref="Random"/> instance.  When null a new instance is
    /// created internally.  Provide a seeded instance for reproducibility in tests.
    /// </param>
    /// <returns>Jagged array [Paths][Horizon+1] of simulated prices.</returns>
    public float[][] SimulatePaths(float startPrice, float annualSigma, Random? rng = null)
    {
        if (startPrice  <= 0f) throw new ArgumentOutOfRangeException(nameof(startPrice),  "Must be positive.");
        if (annualSigma <= 0f) throw new ArgumentOutOfRangeException(nameof(annualSigma), "Must be positive.");

        rng ??= new Random();

        float dt         = 1f / TradingDays;                                 // Δ = 1/252
        float dailySigma = annualSigma * MathF.Sqrt(dt);                     // σ√Δ
        float drift      = (AnnualDrift - 0.5f * annualSigma * annualSigma) * dt; // (μ − σ²/2)Δ

        var paths = new float[Paths][];
        for (int p = 0; p < Paths; p++)
        {
            paths[p] = new float[Horizon + 1];
            paths[p][0] = startPrice;

            float price = startPrice;
            for (int s = 1; s <= Horizon; s++)
            {
                float z = NextGaussian(rng);
                price    = price * MathF.Exp(drift + dailySigma * z);
                paths[p][s] = price;
            }
        }
        return paths;
    }

    /// <summary>
    /// Estimates the probability that a first-passage event defined by
    /// <paramref name="firesOnStep"/> occurs at any step in [1, <see cref="Horizon"/>].
    ///
    /// First-passage semantics: once a path fires, it is counted and the
    /// remaining steps on that path are skipped.  This matches the tax-loss
    /// harvesting decision — once the oracle fires on a given path, harvesting
    /// would happen and subsequent steps on that path are counterfactual.
    ///
    /// Returns the fraction of paths that fired, i.e. an unbiased Monte Carlo
    /// estimate of P(∃ s ∈ [1,Horizon] : firesOnStep(S_s, s) = true).
    /// </summary>
    /// <param name="startPrice">Current price S_0 (must be &gt; 0).</param>
    /// <param name="annualSigma">
    /// Annualised volatility σ (must be &gt; 0).  Falls back to 0.20 if the caller
    /// cannot calibrate from data.
    /// </param>
    /// <param name="firesOnStep">
    /// Predicate evaluated at each simulated step.  Arguments:
    ///   float price     — simulated price at this step
    ///   int   stepIndex — step number in [1, Horizon]
    /// Return true to count the path as fired and stop simulating it.
    /// </param>
    /// <param name="rng">
    /// Optional <see cref="Random"/> instance for reproducibility.
    /// </param>
    /// <returns>
    /// A value in [0, 1] representing the estimated first-passage probability.
    /// Returns <see cref="float.NaN"/> if <paramref name="startPrice"/> or
    /// <paramref name="annualSigma"/> are invalid.
    /// </returns>
    public float FractionFiring(
        float startPrice,
        float annualSigma,
        Func<float, int, bool> firesOnStep,
        Random? rng = null)
    {
        if (startPrice  <= 0f || float.IsNaN(startPrice))  return float.NaN;
        if (annualSigma <= 0f || float.IsNaN(annualSigma)) return float.NaN;

        rng ??= new Random();

        float dt         = 1f / TradingDays;
        float dailySigma = annualSigma * MathF.Sqrt(dt);
        float drift      = (AnnualDrift - 0.5f * annualSigma * annualSigma) * dt;

        int fired = 0;

        for (int p = 0; p < Paths; p++)
        {
            float price = startPrice;
            for (int s = 1; s <= Horizon; s++)
            {
                float z = NextGaussian(rng);
                price    = price * MathF.Exp(drift + dailySigma * z);

                if (firesOnStep(price, s))
                {
                    fired++;
                    break;   // first-passage: stop advancing this path
                }
            }
        }

        return (float)fired / Paths;
    }

    // ── Public static utility ─────────────────────────────────────────────────

    /// <summary>
    /// Generates a single standard-normal variate using the Box-Muller transform.
    ///
    /// Box-Muller: given U₁ ~ Uniform(0,1] and U₂ ~ Uniform[0,1),
    ///   Z = √(−2 ln U₁) · cos(2π U₂)  →  Z ~ N(0,1)
    ///
    /// U₁ is drawn from (0, 1] (i.e. 1 − NextDouble()) to avoid ln(0).
    /// Only one of the two variates produced per pair is returned; the second
    /// is discarded to keep the method stateless and thread-safe.
    /// </summary>
    /// <param name="rng">Source of uniform random variates (must not be null).</param>
    /// <returns>A single N(0,1) draw.</returns>
    public static float NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();   // (0, 1] — guards against log(0)
        double u2 = rng.NextDouble();          // [0, 1)
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
