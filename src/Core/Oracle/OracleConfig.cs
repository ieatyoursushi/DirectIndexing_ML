namespace DirectIndexing.Core.Oracle;

/// <summary>Which oracle generates labels (and drives harvests) in a run.</summary>
public enum OracleMode
{
    /// <summary>v0.2-legacy 4-gate AND, including the G_YTD &gt; 0 gains gate
    /// (kept for ablation baselines; requires the external-gains seed).</summary>
    Gated,

    /// <summary>v0.25 composite (issue #23): three hard gates · 𝟙[U(x) &gt; 0],
    /// U(x) = taxValue − λσ_TE² − c_trade. No gains gate, no seed.</summary>
    Scalarized,
}

/// <summary>
/// Immutable configuration for <see cref="OracleBoundary"/> — replaces the
/// compile-time constants now that the oracle has tunable economic terms
/// (λ, θ_max, c_trade) and two modes to support `--oracle=gated|scalarized`
/// ablation runs.
///
/// Calibration provenance (20y gated run, 1.85M rows — see GYTD_Redesign_Plan.md v2):
///   • realized σ_TE: median 0.0253, p99 0.0338, p99.9 0.0482, max 0.0519.
///   • TrackingErrorCeiling (θ_max) = 0.15 ≈ 3× the old θ₂ cap: never binds in
///     20 years of history incl. the GFC — a tail-only circuit breaker for
///     pathological regimes (cf. the v0.0 33%-TE artifact war story).
///   • Lambda = 90,000 $/unit-σ²: sets the penalty λσ² to ≈ $180 — the median
///     TaxValue of a marginal harvest (−3% &lt; ℓ ≤ −2%) — at σ_TE = 0.045 ≈ p99.9.
///     At the median σ_TE the penalty is ≈ $58, so typical harvests
///     (median TaxValue ≈ $352) clear while marginal ones trade off against TE
///     inside the observed operating band — the quadratic level set
///     {U = 0} curves in-sample rather than being vacuous.
/// </summary>
public sealed record OracleConfig
{
    // ── Hard gates (both modes) ──────────────────────────────────────────────

    /// <summary>θ₁ — minimum unrealized loss to justify harvesting (ℓ ≤ −θ₁).</summary>
    public decimal LossThreshold { get; init; } = 0.02m;

    /// <summary>IRS §1091 wash-sale window in days.</summary>
    public int WashSaleDays { get; init; } = 30;

    // ── Mode selection ───────────────────────────────────────────────────────

    public OracleMode Mode { get; init; } = OracleMode.Scalarized;

    // ── Gated mode only ──────────────────────────────────────────────────────

    /// <summary>θ₂ — the v0.2 fine-grained TE cap (gated mode only).</summary>
    public decimal LegacyTrackingErrorCap { get; init; } = 0.05m;

    // ── Scalarized mode ──────────────────────────────────────────────────────

    /// <summary>θ_max — LOOSE hard TE ceiling; tail-risk circuit breaker only.</summary>
    public decimal TrackingErrorCeiling { get; init; } = 0.15m;

    /// <summary>λ — dollars of penalty per unit σ_TE² inside U(x).</summary>
    public decimal Lambda { get; init; } = 90_000m;

    /// <summary>
    /// c_trade — flat friction of one harvest subtracted from U(x)
    /// (Betterment's benefit-net-of-cost test). Config parameter, NOT a
    /// feature: a constant cannot discriminate rows.
    ///
    /// Calibration (v0.25 PR 3): $10 per harvest = the ROUND TRIP (sell the
    /// loss lot + reopen/substitute buy after the wash window), assuming
    /// zero-commission retail execution and ~2.5 bps effective half-spread
    /// per leg on the ~$20k lots this simulation trades. Deliberately flat —
    /// lot-size-proportional cost is a recorded non-goal until c_trade is
    /// modeled as varying (at which point it becomes a lot-level feature).
    /// Ablation: rerun with --ctrade=0 to quantify the term's contribution.
    /// </summary>
    public decimal CTrade { get; init; } = 10m;

    // ── Derived ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Whether the engine seeds exogenous gains. Only the gated oracle needs a
    /// seed (its gains gate is permanently closed without one); the scalarized
    /// oracle runs the honest loss-only book (offsetCapacity = $3k/yr floor —
    /// the low-outside-activity client persona).
    /// </summary>
    public bool SeedExternalGains => Mode == OracleMode.Gated;

    /// <summary>File suffix used by Program.cs to keep ablation datasets apart.</summary>
    public string DatasetTag => Mode == OracleMode.Gated ? "_gated" : "";

    public static OracleConfig Gated      { get; } = new() { Mode = OracleMode.Gated };
    public static OracleConfig Scalarized { get; } = new() { Mode = OracleMode.Scalarized };
}
