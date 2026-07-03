using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.ML.MLNet.Preprocessing;

/// <summary>
/// Wrapper row that pairs a <see cref="LotStateVector"/> with a per-row training
/// weight. Created by <see cref="ClassWeights.AttachBalancedWeights"/> and consumed
/// by the trainer via <c>exampleWeightColumnName: "Weight"</c>.
/// </summary>
public record WeightedRow : LotStateVector
{
    public float Weight { get; init; }

    public static WeightedRow From(LotStateVector v, float weight) => new()
    {
        L = v.L, H = v.H, S = v.S, B = v.B, W = v.W, K = v.K,
        RealizedGainsYTD = v.RealizedGainsYTD, LossCarryforward = v.LossCarryforward,
        OrdinaryOffsetBudget = v.OrdinaryOffsetBudget,
        Sigma_TE = v.Sigma_TE, WashClock = v.WashClock,
        R_t = v.R_t, SigmaRange = v.SigmaRange,
        DeltaMA50 = v.DeltaMA50, DeltaMA200 = v.DeltaMA200,
        TaxValue = v.TaxValue, DaysToYE = v.DaysToYE, Shares = v.Shares,
        Y_Oracle = v.Y_Oracle, Y_Soft_GBM = v.Y_Soft_GBM, Y_Soft_BT = v.Y_Soft_BT,
        Y_TaxValue = v.Y_TaxValue, Y_Utility = v.Y_Utility,
        Y_Oracle_GatedSpec = v.Y_Oracle_GatedSpec,
        Symbol = v.Symbol, Sector = v.Sector, Timestep = v.Timestep,
        Weight = weight,
    };
}

/// <summary>
/// sklearn-equivalent balanced class weights: <c>w_k = N / (n_classes · n_k)</c>.
///
/// <para><b>Ordering invariant.</b> The only entry point is
/// <see cref="AttachBalancedWeights"/>, which takes a single list — the
/// <i>training fold</i>. It computes <c>n_k</c> from that list and ONLY that list,
/// then attaches per-row weights. There is no overload that accepts the full
/// dataset, so leaking the test-set class distribution into the weights is
/// structurally unrepresentable. The Python pipeline gets this right by accident
/// (sklearn does it inside <c>.fit()</c>); the ML.NET version makes it explicit.</para>
/// </summary>
public static class ClassWeights
{
    public static List<WeightedRow> AttachBalancedWeights(
        IReadOnlyList<LotStateVector> trainingFold,
        Func<LotStateVector, int> labelSelector)
    {
        // Step (b) per the leak-free ordering invariant: count on the training fold only.
        var counts = new Dictionary<int, int>();
        foreach (var r in trainingFold)
        {
            int y = labelSelector(r);
            counts[y] = counts.TryGetValue(y, out var c) ? c + 1 : 1;
        }

        int n = trainingFold.Count;
        int nClasses = counts.Count;

        var weightByClass = new Dictionary<int, float>(counts.Count);
        foreach (var (cls, nk) in counts)
            weightByClass[cls] = (float)n / (nClasses * nk);

        // Step (c): attach.
        var result = new List<WeightedRow>(trainingFold.Count);
        foreach (var r in trainingFold)
            result.Add(WeightedRow.From(r, weightByClass[labelSelector(r)]));

        return result;
    }
}
