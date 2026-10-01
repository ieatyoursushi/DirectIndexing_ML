namespace DirectIndexing.ML.MLNet.Schema;

/// <summary>
/// Single source of truth for column names referenced by every stage of the
/// ML.NET pipeline. Mirrors the names declared on <c>LotStateVector</c>
/// exactly; if a field is added there, add it here.
/// </summary>
public static class FeatureLists
{
    // Schema v6 (v0.3-8): d = 23 — + SigmaHat, SigmaMkt (asset block) and ZBarrier, PBarrier
    // (derived block), the σ̂ feature role of DataMemo/decisions/VolatilityModel_v03.md §6.
    // Schema v5 (v0.3-3): the ledger by character (NetST, NetLT, CarryST, CarryLT,
    // OrdinaryOffsetBudget). v4 dropped Y_Oracle_GatedSpec; v3 (v0.25) G_YTD → TaxLedger.
    public static readonly string[] AllNumericFeatures =
    {
        "L", "H", "S", "B", "W", "K",
        "NetST", "NetLT", "CarryST", "CarryLT", "OrdinaryOffsetBudget",
        "Sigma_TE", "WashClock",
        "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200", "SigmaHat", "SigmaMkt",
        "TaxValue", "DaysToYE", "ZBarrier", "PBarrier",
    };

    /// <summary>The σ̂ feature role (v0.3-8) — dropped as a block by <c>--features=no-vol</c>.</summary>
    public static readonly string[] VolFeatures = { "SigmaHat", "SigmaMkt", "ZBarrier", "PBarrier" };

    /// <summary>Process-wide ablation switch, set once by Program.cs (like SplitPolicy).</summary>
    public static bool ExcludeVolFeatures { get; set; } = false;

    /// <summary>Artifact-directory suffix for the ablation arm.</summary>
    public static string ArtifactTag => ExcludeVolFeatures ? "-novol" : "";

    /// <summary>The numeric features the trainers use: the schema, minus any ablated block.</summary>
    public static string[] NumericFeatures =>
        ExcludeVolFeatures ? AllNumericFeatures.Except(VolFeatures).ToArray() : AllNumericFeatures;

    public static readonly string[] CategoricalFeatures = { "Sector" };

    public const string SectorRaw     = "Sector";
    public const string SectorClean   = "SectorClean";
    public const string SectorOneHot  = "SectorOneHot";
    public const string FeaturesCol   = "Features";
    public const string WeightCol     = "Weight";
    public const string LabelCol      = "Label";

    public const string TargetOracle   = "Y_Oracle";
    public const string TargetSoftBT   = "Y_Soft_BT";

    /// <summary>
    /// Continuous regression target: taxValue_k. Numerically equal to the
    /// TaxValue feature by construction (v0.25), so regression runs on this
    /// target must exclude "TaxValue" from the feature set.
    /// </summary>
    public const string TargetTaxValue = "Y_TaxValue";

    /// <summary>Raw scalarized objective U(x) — diagnostic/RL-reward export, never a feature.</summary>
    public const string TargetUtility = "Y_Utility";

    /// <summary>Feature set for the Y_TaxValue regression: everything except TaxValue.</summary>
    public static string[] NumericFeaturesTaxValueRegression =>
        NumericFeatures.Where(f => f != "TaxValue").ToArray();
}
