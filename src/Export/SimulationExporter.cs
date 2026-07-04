using System.Globalization;
using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Export;

/// <summary>
/// Writes a List&lt;LotStateVector&gt; to a CSV file.
///
/// Column order matches the LotStateVector field declaration order.
/// NaN values are written as empty strings (Pandas/R read these as NaN natively).
/// No third-party dependencies — plain StreamWriter with manual formatting.
/// </summary>
public static class SimulationExporter
{
    // Schema v3 (v0.25, issue #23): G_YTD → three TaxLedger columns,
    // TaxAlpha → TaxValue, new regression label Y_TaxValue. d = 17 features.
    private const string Header =
        "L,H,S,B,W,K," +
        "RealizedGainsYTD,LossCarryforward,OrdinaryOffsetBudget,Sigma_TE,WashClock," +
        "R_t,SigmaRange,DeltaMA50,DeltaMA200," +
        "TaxValue,DaysToYE," +
        "Y_Oracle,Y_Soft_GBM,Y_Soft_BT,Y_TaxValue," +
        "Symbol,Sector,Timestep";

    public static void WriteCsv(IReadOnlyList<LotStateVector> snapshots, string outputPath)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

        using var w = new StreamWriter(outputPath, append: false,
            encoding: System.Text.Encoding.UTF8,
            bufferSize: 1 << 20);   // 1 MB write buffer

        w.WriteLine(Header);

        foreach (var s in snapshots)
        {
            w.Write(Fmt(s.L));         w.Write(',');
            w.Write(s.H);              w.Write(',');
            w.Write(s.S);              w.Write(',');
            w.Write(Fmt(s.B));         w.Write(',');
            w.Write(Fmt(s.W));         w.Write(',');
            w.Write(s.K);              w.Write(',');
            w.Write(Fmt(s.RealizedGainsYTD));     w.Write(',');
            w.Write(Fmt(s.LossCarryforward));     w.Write(',');
            w.Write(Fmt(s.OrdinaryOffsetBudget)); w.Write(',');
            w.Write(Fmt(s.Sigma_TE));  w.Write(',');
            w.Write(s.WashClock);      w.Write(',');
            w.Write(Fmt(s.R_t));       w.Write(',');
            w.Write(Fmt(s.SigmaRange));w.Write(',');
            w.Write(Fmt(s.DeltaMA50)); w.Write(',');
            w.Write(Fmt(s.DeltaMA200));w.Write(',');
            w.Write(Fmt(s.TaxValue));  w.Write(',');
            w.Write(s.DaysToYE);       w.Write(',');
            w.Write(s.Y_Oracle);       w.Write(',');
            w.Write(Fmt(s.Y_Soft_GBM));w.Write(',');
            w.Write(Fmt(s.Y_Soft_BT)); w.Write(',');
            w.Write(Fmt(s.Y_TaxValue));w.Write(',');
            w.Write(Escape(s.Symbol)); w.Write(',');
            w.Write(Escape(s.Sector)); w.Write(',');
            w.WriteLine(s.Timestep);
        }

        Console.WriteLine($"[SimulationExporter] Wrote {snapshots.Count} rows → {outputPath}");
    }

    // float.NaN → empty string (Pandas/R treat as NaN); otherwise 6 significant figures
    private static string Fmt(float v) =>
        float.IsNaN(v) ? "" : v.ToString("G6", CultureInfo.InvariantCulture);

    // Quote strings containing commas or quotes
    private static string Escape(string s) =>
        s.Contains(',') || s.Contains('"') ? $"\"{s.Replace("\"", "\"\"")}\"" : s;
}
