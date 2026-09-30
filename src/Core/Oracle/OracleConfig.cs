namespace DirectIndexing.Core.Oracle;

/// <summary>
/// Immutable configuration for <see cref="OracleBoundary"/> — the scalarized oracle's
/// tunable economic terms (λ, θ_max, c_trade) and hard-gate thresholds.
///
/// Single mode since the pre-v0.3 downsizing: the v0.2 gated arm (4-gate AND with a
/// G_YTD &gt; 0 gains gate and an external-gains seed) was retired — its findings are in
/// DataMemo/archive/RetiredComponents.md §6, its code at tag archive/v0.3-pre-downsize.
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
    // ── Hard gates ───────────────────────────────────────────────────────────

    /// <summary>θ₁ — minimum unrealized loss to justify harvesting (ℓ ≤ −θ₁).</summary>
    public decimal LossThreshold { get; init; } = 0.02m;

    /// <summary>IRS §1091 wash-sale window in days.</summary>
    public int WashSaleDays { get; init; } = 30;

    // ── Economic terms of U(x) = taxValue − λσ_TE² − c_trade ──────────────────

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

    /// <summary>The canonical configuration (all defaults above).</summary>
    public static OracleConfig Default { get; } = new();
}
