using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace DirectIndexing.ML.MLNet.Io;

/// <summary>
/// Thin JSON / CSV writers used by the ML.NET pipeline. The schema is
/// deliberately flat and JSON-camel-cased to make it easy for the Python
/// renderer to consume without sklearn-shaped expectations.
/// </summary>
public static class Artifacts
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    public static void WriteJson<T>(T payload, string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var json = JsonSerializer.Serialize(payload, JsonOpts);
        File.WriteAllText(path, json);
        Console.WriteLine($"[Artifacts] wrote {path}");
    }

    public static void WriteCsv(string path, IEnumerable<string> header, IEnumerable<IEnumerable<object>> rows)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var sb = new StringBuilder();
        sb.AppendLine(string.Join(",", header));
        foreach (var row in rows)
        {
            var cells = row.Select(c => c switch
            {
                double d => d.ToString("G6", CultureInfo.InvariantCulture),
                float f  => f.ToString("G6", CultureInfo.InvariantCulture),
                _ => c?.ToString() ?? "",
            });
            sb.AppendLine(string.Join(",", cells));
        }
        File.WriteAllText(path, sb.ToString());
        Console.WriteLine($"[Artifacts] wrote {path}");
    }
}
