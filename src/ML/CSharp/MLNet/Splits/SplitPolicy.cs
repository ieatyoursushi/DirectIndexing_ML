namespace DirectIndexing.ML.MLNet.Splits;

/// <summary>How train/test and CV partitions are drawn (v0.26, validation hardening).</summary>
public enum SplitMode
{
    /// <summary>v0.1–v0.25 behaviour: per-class shuffle, random assignment.</summary>
    StratifiedRandom,

    /// <summary>
    /// Chronological split on Timestep with a purge/embargo gap. Required for
    /// honest evaluation of the forward-looking soft labels: Y_Soft_BT at time t
    /// is a function of prices over (t, t+30], so adjacent rows share future
    /// context and a random split leaks it across the train/test boundary.
    /// </summary>
    TemporalPurged,
}

/// <summary>
/// Process-wide split policy, set once by Program.cs from the CLI
/// (<c>--split=temporal</c>, <c>--embargo=N</c>, <c>--testfrac=F</c>) before any
/// trainer runs. Trainers call the <see cref="DataSplit"/> facade, which
/// dispatches on this policy — 12 call sites stay signature-identical while the
/// partition semantics change in exactly one place.
///
/// Default is the legacy stratified-random mode so existing commands reproduce
/// v0.25 numbers bit-for-bit unless the flag is passed.
/// </summary>
public static class SplitPolicy
{
    public static SplitMode Mode { get; set; } = SplitMode.StratifiedRandom;

    /// <summary>
    /// Purge gap in trading days (Timestep units) excised around every
    /// train/boundary edge. Must be ≥ the forward-label horizon (30 for
    /// Y_Soft_BT / Y_Soft_GBM) so no training label window overlaps the
    /// evaluation period.
    /// </summary>
    public static int EmbargoDays { get; set; } = 30;

    /// <summary>
    /// When set, overrides the testFraction trainers pass at their call sites
    /// (they hardcode 0.20). Used for the decade walk-forward experiment
    /// (e.g. 0.5 → train 2006–16, test 2016–26).
    /// </summary>
    public static double? TestFractionOverride { get; set; } = null;

    /// <summary>Suffix for artifact directories so ablation arms never overwrite each other.</summary>
    public static string ArtifactTag =>
        Mode == SplitMode.TemporalPurged ? "-temporal" : "";

    public static string Describe() =>
        Mode == SplitMode.StratifiedRandom
            ? "stratified-random (legacy)"
            : $"temporal-purged (embargo={EmbargoDays}d" +
              (TestFractionOverride is { } f ? $", testFrac={f:0.##}" : "") + ")";
}
