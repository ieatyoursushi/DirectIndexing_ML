namespace DirectIndexing.ML.MLNet.Tuning;

/// <summary>
/// Lightweight CV summary returned by each trainer's <c>RunCV()</c> method.
/// <c>RunAllSupervised</c> collects one per model, ranks by <see cref="MeanCvScore"/>,
/// and calls the full <c>Run()</c> on the winner(s) only — keeping the test set
/// untouched until champion selection is complete.
/// </summary>
public record CvResult(
    string ModelName,
    IReadOnlyDictionary<string, object> BestParams,
    double MeanCvScore,
    double[] PerFoldScores,
    IReadOnlyList<(IReadOnlyDictionary<string, object> Params, double MeanScore)> AllConfigs);
