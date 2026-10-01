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
///   NetST / NetLT — signed net realized P&amp;L YTD by §1222 character (G^ST, G^LT);
///   positive means net gains exist to offset, negative after net-loss harvests
///
/// Schema v5 (v0.3-3): d = 19 numeric features, 27 exported columns — v4's blended
/// RealizedGainsYTD / LossCarryforward split by character (ROADMAP F8).
/// Schema v4 (pre-v0.3 downsizing): v3 minus the retired Y_Oracle_GatedSpec label.
/// TLDR this is like the graph of the multivariate X x Y represented by an R^n vector feature space (so feature space + soft label image which is subsetted in R from [0, 1]). Subject to change
/// </summary>
public record LotStateVector
{
    // ── Lot-level features ───────────────────────────────────────────────────

    /// <summary>ℓ = (P_t − p_k)/p_k ∈ (−1, ∞)  — normalised unrealised return</summary>
    public float L           { get; init; }

    /// <summary>h = t − s_k ∈ ℤ_{≥0}  — lot age in TRADING days (feature coordinate; §1222 uses S)</summary>
    public int   H           { get; init; }

    /// <summary>s = 𝟙[date(t) &gt; date(s_k) + 1 calendar year] ∈ {0,1}  — §1222 long-term flag</summary>
    public int   S           { get; init; }

    /// <summary>p_k — cost basis per share (dollars)</summary>
    public float B           { get; init; }

    /// <summary>w_k = q_k P_t / V_t ∈ (0,1)  — lot weight in portfolio</summary>
    public float W           { get; init; }

    /// <summary>k — number of open lots in the same ticker</summary>
    public int   K           { get; init; }

    /// <summary>
    /// q_k — share count of the lot. IN-MEMORY PLUMBING ONLY, never exported to
    /// CSV (the feature schema is unchanged): the soft-label builders need it to
    /// re-dollarize the loss along forward price paths for the scalarized
    /// oracle's taxValue term. Redundant with (W, B) given portfolio value, but
    /// V_t is not carried on the snapshot.
    /// </summary>
    public float Shares      { get; init; }

    /// <summary>
    /// Calendar day number of the lot's purchase date. IN-MEMORY PLUMBING ONLY (never
    /// exported): the soft-label builders re-evaluate the §1222 character along forward
    /// steps, which needs the purchase date, not the trading-day count H.
    /// </summary>
    public int   PurchaseDayNumber { get; init; }

    // ── Portfolio-level features (shared state 𝒮_t) — TaxLedger + risk state ─

    /// <summary>ledger_t.G^ST ∈ ℝ — signed net SHORT-term realized P&amp;L this year. Resets at year-end.</summary>
    public float NetST                { get; init; }

    /// <summary>ledger_t.G^LT ∈ ℝ — signed net LONG-term realized P&amp;L this year. Resets at year-end.</summary>
    public float NetLT                { get; init; }

    /// <summary>ledger_t.C^ST ∈ ℝ≥0 — short-term loss carryforward from prior years (§1212(b)).</summary>
    public float CarryST              { get; init; }

    /// <summary>ledger_t.C^LT ∈ ℝ≥0 — long-term loss carryforward from prior years (§1212(b)).</summary>
    public float CarryLT              { get; init; }

    /// <summary>The frozen ledger (G^ST, G^LT, C^ST, C^LT) as a value, for counterfactual valuation.</summary>
    public LedgerState Ledger => new((decimal)NetST, (decimal)NetLT, (decimal)CarryST, (decimal)CarryLT);

    /// <summary>
    /// ledger_t.OrdinaryOffsetBudget ∈ [0, 3000] — remaining ordinary-income
    /// offset allowance this year (26 USC §1211(b)).
    /// </summary>
    public float OrdinaryOffsetBudget { get; init; }

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
    /// taxValue_k = τ(h)·min(loss, offsetCapacity) + τ_future·max(loss − offsetCapacity, 0)·δ
    /// — capacity-aware dollar value of harvesting this lot right now, joining
    /// the shared ledger state with (h_k, ℓ_k). Supersedes the v0.2 TaxAlpha.
    /// 0 for lots not at a loss.
    /// </summary>
    public float TaxValue    { get; init; }

    /// <summary>Calendar days remaining in the tax year (resets Jan 1)</summary>
    public int   DaysToYE    { get; init; }

    // ── Labels ───────────────────────────────────────────────────────────────

    /// <summary>f*(x) ∈ {0,1}  — oracle hard label (backtesting)</summary>
    public int   Y_Oracle    { get; init; }

    /// <summary>
    /// ỹ_GBM(x) ∈ [0,1]  — fraction of 200 GBM forward paths where oracle fires
    /// within the next 30 trading days (frozen portfolio state, per-stock σ from
    /// trailing 21-day realised vol).
    /// </summary>
    public float Y_Soft_GBM  { get; init; }

    /// <summary>
    /// ỹ_BT(x) ∈ [0,1]  — fraction of the next 30 actual trading days where oracle
    /// fires (frozen portfolio state, real historical prices).
    /// NaN when fewer than 30 days remain in the data window.
    /// </summary>
    public float Y_Soft_BT   { get; init; }

    /// <summary>
    /// Y_TaxValue ∈ ℝ≥0 — continuous regression target: taxValue_k of this lot
    /// at this timestep (cross-sectional, σ(𝓕_t)-measurable). Numerically equal
    /// to the TaxValue feature by construction in v0.25 — regression runs on
    /// this target MUST exclude TaxValue from the feature set.
    /// </summary>
    public float Y_TaxValue  { get; init; }

    /// <summary>
    /// U(x) = TaxValue − λσ_TE² − c_trade ∈ ℝ — the scalarized objective's raw
    /// score before thresholding (issue #17 family; the v0.4 RL per-decision
    /// reward). Label/diagnostic, never a feature: 𝟙[U &gt; 0] is the oracle's own
    /// boundary. Computed under the run's OracleConfig.
    /// </summary>
    public float Y_Utility   { get; init; }

    // ── Metadata (for EDA — drop before modelling) ───────────────────────────

    public string Symbol     { get; init; } = "";
    public string Sector     { get; init; } = "";

    /// <summary>Simulation day index t</summary>
    public int   Timestep    { get; init; }
}
