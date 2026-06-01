namespace DirectIndexing.ML.MLNet.Preprocessing;

/// <summary>
/// The typed row that flows into <c>MLContext.Data.LoadFromEnumerable</c>.
/// Differs from <c>LotStateVector</c> in three ways, each of which is a
/// deliberate preprocessing step:
///   1. Int features (H, S, K, WashClock, DaysToYE) are widened to float so the
///      EstimatorChain can normalise them alongside the natural-float features.
///   2. Float NaNs have already been replaced by training-fold medians.
///   3. A typed <c>Label</c> (bool) and <c>Weight</c> (float) sit alongside, ready
///      for the trainer with no further column gymnastics.
/// Raw <c>Sector</c> survives untouched — the empty→"Unknown" rewrite happens
/// inside the EstimatorChain via a CustomMapping so the cleaning is visible in
/// the typed pipeline rather than buried in pre-loop C#.
/// </summary>
public record MLReadyRow
{
    public float L { get; init; }
    public float H { get; init; }
    public float S { get; init; }
    public float B { get; init; }
    public float W { get; init; }
    public float K { get; init; }
    public float G_YTD { get; init; }
    public float Sigma_TE { get; init; }
    public float WashClock { get; init; }
    public float R_t { get; init; }
    public float SigmaRange { get; init; }
    public float DeltaMA50 { get; init; }
    public float DeltaMA200 { get; init; }
    public float TaxAlpha { get; init; }
    public float DaysToYE { get; init; }

    public string Sector { get; init; } = "";

    public bool  Label { get; init; }
    // FloatLabel mirrors Label as 1f/0f; consumed by the regression trainer
    // which requires a float label column (regression does not use bool Label).
    public float FloatLabel { get; init; }
    public float Weight { get; init; } = 1f;
    public string Symbol { get; init; } = "";
}
