namespace DirectIndexing.ML.MLNet.Schema;

/// <summary>
/// Single source of truth for column names referenced by every stage of the
/// ML.NET pipeline. Mirrors the names declared on <c>LotStateVector</c>
/// exactly; if a field is added there, add it here.
/// </summary>
public static class FeatureLists
{
    public static readonly string[] NumericFeatures =
    {
        "L", "H", "S", "B", "W", "K",
        "G_YTD", "Sigma_TE", "WashClock",
        "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200",
        "TaxAlpha", "DaysToYE",
    };

    public static readonly string[] CategoricalFeatures = { "Sector" };

    public const string SectorRaw     = "Sector";
    public const string SectorClean   = "SectorClean";
    public const string SectorOneHot  = "SectorOneHot";
    public const string FeaturesCol   = "Features";
    public const string WeightCol     = "Weight";
    public const string LabelCol      = "Label";

    public const string TargetOracle = "Y_Oracle";
    public const string TargetSoftBT = "Y_Soft_BT";
}
