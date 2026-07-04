using System.Globalization;
using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.ML.MLNet.Data;

/// <summary>
/// Reads <c>data/lots.csv</c> (produced by <c>SimulationExporter.WriteCsv</c>) back
/// into a typed <c>List&lt;LotStateVector&gt;</c>. Empty cells are parsed as
/// <c>float.NaN</c>, matching the writer's NaN→empty-string convention.
///
/// This is the ONLY boundary at which CSV exists in this pipeline. Once read,
/// the list flows through <c>MLContext.Data.LoadFromEnumerable&lt;LotStateVector&gt;</c>
/// with no further serialization — the named, typed schema is preserved end-to-end.
/// </summary>
public static class LotStateVectorCsvReader
{
    public static List<LotStateVector> Read(string csvPath)
    {
        using var sr = new StreamReader(csvPath);
        var header = sr.ReadLine()
            ?? throw new InvalidDataException($"empty file: {csvPath}");

        var cols = header.Split(',');
        var idx = new Dictionary<string, int>(cols.Length);
        for (int i = 0; i < cols.Length; i++) idx[cols[i]] = i;

        // Required columns are everything SimulationExporter writes; fail fast
        // if the schema drifts so downstream stages don't get silent NaN soup.
        string[] required =
        {
            "L","H","S","B","W","K",
            "RealizedGainsYTD","LossCarryforward","OrdinaryOffsetBudget","Sigma_TE","WashClock",
            "R_t","SigmaRange","DeltaMA50","DeltaMA200",
            "TaxValue","DaysToYE",
            "Y_Oracle","Y_Soft_GBM","Y_Soft_BT","Y_TaxValue","Y_Utility","Y_Oracle_GatedSpec",
            "Symbol","Sector","Timestep",
        };
        foreach (var c in required)
            if (!idx.ContainsKey(c))
                throw new InvalidDataException($"missing column '{c}' in {csvPath}");

        var rows = new List<LotStateVector>(capacity: 128_000);
        string? line;
        while ((line = sr.ReadLine()) is not null)
        {
            if (line.Length == 0) continue;
            var f = SplitRespectingQuotes(line);

            rows.Add(new LotStateVector
            {
                L          = ParseFloat(f[idx["L"]]),
                H          = ParseInt(f[idx["H"]]),
                S          = ParseInt(f[idx["S"]]),
                B          = ParseFloat(f[idx["B"]]),
                W          = ParseFloat(f[idx["W"]]),
                K          = ParseInt(f[idx["K"]]),
                RealizedGainsYTD     = ParseFloat(f[idx["RealizedGainsYTD"]]),
                LossCarryforward     = ParseFloat(f[idx["LossCarryforward"]]),
                OrdinaryOffsetBudget = ParseFloat(f[idx["OrdinaryOffsetBudget"]]),
                Sigma_TE   = ParseFloat(f[idx["Sigma_TE"]]),
                WashClock  = ParseInt(f[idx["WashClock"]]),
                R_t        = ParseFloat(f[idx["R_t"]]),
                SigmaRange = ParseFloat(f[idx["SigmaRange"]]),
                DeltaMA50  = ParseFloat(f[idx["DeltaMA50"]]),
                DeltaMA200 = ParseFloat(f[idx["DeltaMA200"]]),
                TaxValue   = ParseFloat(f[idx["TaxValue"]]),
                DaysToYE   = ParseInt(f[idx["DaysToYE"]]),
                Y_Oracle   = ParseInt(f[idx["Y_Oracle"]]),
                Y_Soft_GBM = ParseFloat(f[idx["Y_Soft_GBM"]]),
                Y_Soft_BT  = ParseFloat(f[idx["Y_Soft_BT"]]),
                Y_TaxValue = ParseFloat(f[idx["Y_TaxValue"]]),
                Y_Utility  = ParseFloat(f[idx["Y_Utility"]]),
                Y_Oracle_GatedSpec = ParseInt(f[idx["Y_Oracle_GatedSpec"]]),
                Symbol     = Unquote(f[idx["Symbol"]]),
                Sector     = Unquote(f[idx["Sector"]]),
                Timestep   = ParseInt(f[idx["Timestep"]]),
            });
        }
        return rows;
    }

    private static float ParseFloat(string s) =>
        string.IsNullOrEmpty(s) ? float.NaN
            : float.Parse(s, CultureInfo.InvariantCulture);

    private static int ParseInt(string s) =>
        string.IsNullOrEmpty(s) ? 0
            : int.Parse(s, CultureInfo.InvariantCulture);

    private static string Unquote(string s)
    {
        if (s.Length >= 2 && s[0] == '"' && s[^1] == '"')
            return s[1..^1].Replace("\"\"", "\"");
        return s;
    }

    // Minimal RFC-4180-ish splitter: handles quoted fields containing commas.
    private static string[] SplitRespectingQuotes(string line)
    {
        var result = new List<string>(24);
        var sb = new System.Text.StringBuilder(line.Length);
        bool inQuotes = false;
        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];
            if (c == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"')
                { sb.Append('"'); i++; }
                else inQuotes = !inQuotes;
            }
            else if (c == ',' && !inQuotes)
            {
                result.Add(sb.ToString()); sb.Clear();
            }
            else sb.Append(c);
        }
        result.Add(sb.ToString());
        return result.ToArray();
    }
}
