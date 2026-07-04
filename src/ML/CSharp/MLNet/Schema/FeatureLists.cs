namespace DirectIndexing.ML.MLNet.Schema;

/// <summary>
/// Single source of truth for column names referenced by every stage of the
/// ML.NET pipeline. Mirrors the names declared on <c>LotStateVector</c>
/// exactly; if a field is added there, add it here.
/// </summary>
public static class FeatureLists
{
    // Schema v3 (v0.25): d = 17 numeric features. G_YTD → three TaxLedger
    // columns; TaxAlpha → capacity-aware TaxValue.
    public static readonly string[] NumericFeatures =
    {
        "L", "H", "S", "B", "W", "K",
        "RealizedGainsYTD", "LossCarryforward", "OrdinaryOffsetBudget",
        "Sigma_TE", "WashClock",
        "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200",
        "TaxValue", "DaysToYE",
    };

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
}
