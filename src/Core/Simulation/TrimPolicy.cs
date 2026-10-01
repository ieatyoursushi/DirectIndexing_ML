namespace DirectIndexing.Core.Simulation;

/// <summary>
/// Sell-winner trim (v0.3-4): periodically sell GAIN lots of overweight names back toward
/// equal weight and reinvest the proceeds in the most underweight eligible names.
///
/// <para><b>Why it exists.</b> Until v0.3-4 the simulated book only ever realized losses, so the
/// Schedule D ledger (v0.3-3) had nothing to net: capacity sat at the $3k ordinary line, the
/// carryforward only grew, and τ only ever applied to that line. A real direct-indexing account
/// realizes gains — drift correction, cash needs, index changes — and those gains are what a
/// harvested loss is worth most against. The trim makes gains <i>endogenous</i>, which is what
/// exercises carryforward consumption and cross-character netting.</para>
///
/// <para><b>Mechanics.</b> Every <see cref="IntervalDays"/>, the names whose weight exceeds
/// (1 + <see cref="Band"/>) × the equal-weight target are trimmed, the
/// <see cref="NamesPerTrim"/> most overweight first. Within a name, whole gain lots are sold
/// highest basis first (the least gain per dollar sold — the standard "HIFO" relief) while the
/// lot fits inside the remaining excess. Loss lots are never trimmed (that is the oracle's
/// decision), and a gain sale opens no §1091 window. Proceeds are reinvested exactly like a
/// contribution: the most underweight names that pass <c>CanBuy</c> and hold no harvestable lot.</para>
///
/// <para><b>Default is disabled</b> (<c>--trim</c> turns it on): a boundary-shaping change with
/// its own ablation arm. The v0.3-11 policy seam absorbs it as a second action type.</para>
/// </summary>
public sealed record TrimPolicy
{
    /// <summary>Off by default — an unflagged run never sells a winner.</summary>
    public bool Enabled { get; init; } = false;

    /// <summary>Trading days between trims. Default 63 ≈ quarterly.</summary>
    public int IntervalDays { get; init; } = 63;

    /// <summary>Tolerance band: trim a name when its weight exceeds (1 + Band) × target.</summary>
    public decimal Band { get; init; } = 0.5m;

    /// <summary>At most this many names trimmed (and bought) per event.</summary>
    public int NamesPerTrim { get; init; } = 10;

    /// <summary>Suffix so trim-arm datasets never overwrite the baseline.</summary>
    public string DatasetTag => Enabled ? "_trim" : "";

    public static TrimPolicy Off { get; } = new();

    public string Describe() =>
        Enabled
            ? $"every {IntervalDays}d, names above {1m + Band:0.##}× equal weight, " +
              $"≤{NamesPerTrim} names, highest-basis gain lots first"
            : "disabled";
}
