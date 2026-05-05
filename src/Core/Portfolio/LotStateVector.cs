namespace DirectIndexing.Core.Portfolio;

/// <summary>
/// The image of the feature extraction map   g : 𝒮_t × P_t → 𝒳^{|𝒦_t|}
/// applied to a SINGLE lot at a single timestep.
///
/// Each instance is ONE row in lots.csv and ONE observation (x, y) ∈ 𝒳 × 𝒴
/// fed to the ML model.  It is an immutable frozen snapshot — a "photograph"
/// of the joint (lot state, portfolio state, asset state) at time t.
///
/// Field types are float (not decimal) for ML.NET IDataView compatibility.
/// Labels are kept as separate fields so the same CSV feeds both hard-label
/// classifiers (Y_Oracle) and soft-label regressors / probabilistic models (Y_Soft).
///
/// Sign conventions (see PortfolioMath.md §3 for derivations):
///   L   — negative for a harvestable lot  (ℓ = (P_t − p_k)/p_k &lt; 0)
///   G_YTD — positive means net gains exist to offset; oracle fires only when &gt; 0
/// TLDR this is like the graph of the multivariate X x Y represented by an R^n vector feature space (so feature space + soft label image which is subsetted in R from [0, 1]). Subject to change
/// </summary>
public record LotSnapshot
{
    // ── Lot-level features ───────────────────────────────────────────────────

    /// <summary>ℓ = (P_t − p_k)/p_k ∈ (−1, ∞)  — normalised unrealised return</summary>
    public float L           { get; init; }

    /// <summary>h = t − s_k ∈ ℤ_{≥0}  — holding period in days</summary>
    public int   H           { get; init; }

    /// <summary>s = 𝟙[h ≥ 365] ∈ {0,1}  — short/long-term flag</summary>
    public int   S           { get; init; }

    /// <summary>p_k — cost basis per share (dollars)</summary>
    public float B           { get; init; }

    /// <summary>w_k = q_k P_t / V_t ∈ (0,1)  — lot weight in portfolio</summary>
    public float W           { get; init; }

    /// <summary>k — number of open lots in the same ticker</summary>
    public int   K           { get; init; }

    // ── Portfolio-level features (shared state 𝒮_t) ─────────────────────────

    /// <summary>G_t^YTD ∈ ℝ  — net realised gain/loss this calendar year</summary>
    public float G_YTD       { get; init; }

    /// <summary>σ_TE  — annualised tracking error vs. benchmark at time t</summary>
    public float Sigma_TE    { get; init; }

    /// <summary>𝒲_t^{A_i} — days since last harvest of this ticker (999 = never)</summary>
    public int   WashClock   { get; init; }

    // ── Asset-level features (from price series) ─────────────────────────────

    /// <summary>r_t = (P_t − P_{t−1})/P_{t−1}  — daily return</summary>
    public float R_t         { get; init; }

    /// <summary>(H_t − L_t)/P_{t−1}  — range-based realised volatility proxy</summary>
    public float SigmaRange  { get; init; }

    /// <summary>(P_t − MA_50) / MA_50  — deviation from 50-day moving average</summary>
    public float DeltaMA50   { get; init; }

    /// <summary>(P_t − MA_200) / MA_200  — deviation from 200-day moving average</summary>
    public float DeltaMA200  { get; init; }

    // ── Derived / composite features ─────────────────────────────────────────

    /// <summary>
    /// α_tax = τ(h) · |G_lot| · 𝟙[G_YTD &gt; 0]
    /// — estimated tax alpha from harvesting this lot right now
    /// </summary>
    public float TaxAlpha    { get; init; }

    /// <summary>Calendar days remaining in the tax year (resets Jan 1)</summary>
    public int   DaysToYE    { get; init; }

    // ── Labels ───────────────────────────────────────────────────────────────

    /// <summary>f*(x) ∈ {0,1}  — oracle hard label</summary>
    public int   Y_Oracle    { get; init; }

    /// <summary>ỹ(x) ∈ [0,1]  — soft label from forward simulation window</summary>
    public float Y_Soft      { get; init; }

    // ── Metadata (for EDA — drop before modelling) ───────────────────────────

    public string Symbol     { get; init; } = "";
    public string Sector     { get; init; } = "";

    /// <summary>Simulation day index t</summary>
    public int   Timestep    { get; init; }
}
