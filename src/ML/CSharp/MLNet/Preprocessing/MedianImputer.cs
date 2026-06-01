using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.ML.MLNet.Preprocessing;

/// <summary>
/// Median imputation for float-valued numeric features.
///
/// <para>ML.NET's <c>ReplaceMissingValues</c> exposes Mean natively but not
/// Median, and the Python pipeline uses Median — so we compute it manually here
/// before handing rows to <c>LoadFromEnumerable</c>. Imputing means on
/// heavy-tailed features like <c>L</c> would diverge from the Python branch in
/// a visible, traceable way; median keeps the two comparable.</para>
///
/// <para><b>Ordering invariant.</b> <see cref="Fit"/> takes a <i>training fold</i>
/// and is the only path to a median dictionary. <see cref="Apply"/> takes a
/// dictionary computed by <see cref="Fit"/> plus any list of rows (train, val,
/// or test). The medians never see test data.</para>
/// </summary>
public static class MedianImputer
{
    // Only float-valued numeric features need imputation. The int features
    // (H, S, K, WashClock, DaysToYE) are never NaN in the CSV — they're written
    // as 0 when the source value is missing — so they pass through as-is.
    private static readonly string[] FloatNumericFeatures =
    {
        "L", "B", "W",
        "G_YTD", "Sigma_TE",
        "R_t", "SigmaRange", "DeltaMA50", "DeltaMA200",
        "TaxAlpha",
    };

    public static Dictionary<string, float> Fit(IReadOnlyList<LotStateVector> trainingFold)
    {
        var medians = new Dictionary<string, float>(FloatNumericFeatures.Length);
        foreach (var col in FloatNumericFeatures)
        {
            var values = trainingFold
                .Select(r => Get(r, col))
                .Where(v => !float.IsNaN(v))
                .OrderBy(v => v)
                .ToArray();

            medians[col] = values.Length == 0 ? 0f : Median(values);
        }
        return medians;
    }

    public static List<MLReadyRow> Apply(
        IReadOnlyList<LotStateVector> rows,
        Dictionary<string, float> medians,
        Func<LotStateVector, bool> labelSelector)
    {
        var result = new List<MLReadyRow>(rows.Count);
        foreach (var r in rows)
        {
            result.Add(new MLReadyRow
            {
                L          = ImputeFloat(r.L,          medians["L"]),
                B          = ImputeFloat(r.B,          medians["B"]),
                W          = ImputeFloat(r.W,          medians["W"]),
                G_YTD      = ImputeFloat(r.G_YTD,      medians["G_YTD"]),
                Sigma_TE   = ImputeFloat(r.Sigma_TE,   medians["Sigma_TE"]),
                R_t        = ImputeFloat(r.R_t,        medians["R_t"]),
                SigmaRange = ImputeFloat(r.SigmaRange, medians["SigmaRange"]),
                DeltaMA50  = ImputeFloat(r.DeltaMA50,  medians["DeltaMA50"]),
                DeltaMA200 = ImputeFloat(r.DeltaMA200, medians["DeltaMA200"]),
                TaxAlpha   = ImputeFloat(r.TaxAlpha,   medians["TaxAlpha"]),

                H          = r.H,
                S          = r.S,
                K          = r.K,
                WashClock  = r.WashClock,
                DaysToYE   = r.DaysToYE,

                Sector     = r.Sector ?? "",
                Symbol     = r.Symbol ?? "",
                Label      = labelSelector(r),
                FloatLabel = labelSelector(r) ? 1f : 0f,
            });
        }
        return result;
    }

    private static float ImputeFloat(float v, float median) =>
        float.IsNaN(v) ? median : v;

    private static float Get(LotStateVector r, string col) => col switch
    {
        "L"          => r.L,
        "B"          => r.B,
        "W"          => r.W,
        "G_YTD"      => r.G_YTD,
        "Sigma_TE"   => r.Sigma_TE,
        "R_t"        => r.R_t,
        "SigmaRange" => r.SigmaRange,
        "DeltaMA50"  => r.DeltaMA50,
        "DeltaMA200" => r.DeltaMA200,
        "TaxAlpha"   => r.TaxAlpha,
        _ => throw new ArgumentException($"unknown float feature '{col}'"),
    };

    private static float Median(float[] sorted)
    {
        int n = sorted.Length;
        return n % 2 == 1
            ? sorted[n / 2]
            : 0.5f * (sorted[n / 2 - 1] + sorted[n / 2]);
    }
}
