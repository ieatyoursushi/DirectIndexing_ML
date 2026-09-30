namespace DirectIndexing.Core.Simulation;

/// <summary>
/// Periodic cash contributions that mint fresh lots at current prices (v0.3 P0).
///
/// <para><b>The defect this fixes.</b> The v0.1–v0.26 simulator opens every lot once, on the
/// warmup day, and never mints another except when re-buying a harvested position. Over a long
/// window the surviving lots accumulate years of appreciation, so a position carrying a 2007
/// basis is so deep in the money by 2020 that a one-third market crash cannot push it 2% below
/// cost. The book <i>ages out of harvestability</i>: the oracle rate falls 1.6% → 0.20% and the
/// harvest signal is nearly extinct after the first decade even through real crashes.</para>
///
/// <para>That is a <i>simulation-design</i> artifact, not a market fact — a real direct-indexing
/// account receives ongoing contributions that continuously mint lots at current prices, which
/// <i>can</i> dip. v0.26 quantified the cost: under purged temporal splits the test-period
/// positive rate collapses to 0.22% and PR-AUC falls with it, while ROC-AUC holds (the ranking
/// was always fine — there was simply almost nothing left to rank). Restoring harvestable supply
/// is therefore the v0.3 P0.</para>
///
/// <para><b>Two constraints shape this design.</b></para>
/// <list type="number">
///   <item><b>Wash-sale safety (§1091, both sides, calendar days — v0.3-1).</b> A contribution
///   may not buy a ticker within 30 days after its last loss sale (<c>PortfolioState.CanBuy</c>),
///   or it would disallow that loss. And because a fresh lot is a §1091 <i>replacement</i> for
///   30 days, by default a contribution also skips any ticker that currently holds a
///   harvestable lot (<see cref="SkipHarvestableNames"/>): buying the fallen names you are about
///   to harvest is exactly what made ~24% of the P0 arm's harvests wash sales before v0.3-1.</item>
///   <item><b>Bounded lot growth.</b> Minting into every ticker on every contribution would grow
///   the open-lot set without bound and make the row count explode (rows ≈ Σ_t |open lots|).
///   Each contribution instead buys the <c>NamesPerContribution</c> most <i>underweight</i>
///   eligible tickers — which is also what a real manager does, since directing new cash at
///   drift is the cheapest way to correct it without selling.</item>
/// </list>
///
/// <para><b>Default is disabled</b>, so an unflagged run is the baseline arm. Turning
/// contributions on is a boundary-shaping change and gets its own ablation arm (standing rules 1
/// and 2 in <c>ROADMAP.md</c>).</para>
/// </summary>
public sealed record ContributionPolicy
{
    /// <summary>Off by default — an unflagged run is the no-contribution baseline arm.</summary>
    public bool Enabled { get; init; } = false;

    /// <summary>
    /// Trading days between contributions. Default 63 ≈ quarterly. Shorter intervals restore
    /// prevalence faster but grow the open-lot set (and therefore run time and dataset size)
    /// proportionally.
    /// </summary>
    public int IntervalDays { get; init; } = 63;

    /// <summary>
    /// Annual contribution as a fraction of the <i>initial</i> portfolio value. Default 0.10 —
    /// a client adding roughly 10% of the opening book per year. Held relative to the initial
    /// value (not the current value) so the schedule is exogenous and does not compound with
    /// market performance, which keeps the ablation interpretable.
    /// </summary>
    public decimal AnnualRate { get; init; } = 0.10m;

    /// <summary>
    /// How many distinct tickers each contribution buys, chosen as the most underweight
    /// eligible names. Bounds lot growth; 20 of ~400 names per quarter keeps the open-lot set
    /// growing slowly enough that the 20-year run stays tractable.
    /// </summary>
    public int NamesPerContribution { get; init; } = 20;

    /// <summary>
    /// Skip tickers holding a lot at or below the oracle's loss threshold (default on). The
    /// engine's §1091 before-side gate already keeps harvests legal either way — this decides
    /// whether contributions <i>block</i> harvests (off) or steer around them (on, what a real
    /// manager does). <c>--contrib-allow-harvestable</c> turns it off for the ablation.
    /// </summary>
    public bool SkipHarvestableNames { get; init; } = true;

    /// <summary>
    /// Dollars per contribution event: the annual rate pro-rated over the interval,
    /// assuming 252 trading days per year.
    /// </summary>
    public decimal AmountPer(decimal initialPortfolioValue) =>
        initialPortfolioValue * AnnualRate * IntervalDays / 252m;

    /// <summary>Suffix so contribution-arm datasets never overwrite the baseline.</summary>
    public string DatasetTag => Enabled ? "_contrib" : "";

    public static ContributionPolicy Off { get; } = new();

    public string Describe() =>
        Enabled
            ? $"every {IntervalDays}d, {AnnualRate:P0}/yr of initial value, " +
              $"{NamesPerContribution} most-underweight names" +
              (SkipHarvestableNames ? ", skipping names with a harvestable lot" : "")
            : "disabled";
}
