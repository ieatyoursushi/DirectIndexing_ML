using System.Text.Json;

namespace DirectIndexing.DataCollection;

public sealed class MarketDataDownloader
{
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() }
    };

    private readonly string _apiKey;
    private readonly HttpClient _http;

    public MarketDataDownloader(string apiKey, HttpClient? http = null)
    {
        _apiKey = apiKey;
        _http = http ?? new HttpClient();
    }

    public async Task<List<SP500Constituent>> GetSP500Symbols()
    {
        var url = $"https://financialmodelingprep.com/v3/sp500-constituent?apikey={_apiKey}";
        var json = await _http.GetStringAsync(url);
        return JsonSerializer.Deserialize<List<SP500Constituent>>(json, JsonOpts)
               ?? throw new InvalidOperationException("FMP returned null for S&P 500 constituents.");
    }

    public async Task<string> FetchHistoricalPrices(string symbol, DateOnly from, DateOnly to)
    {
        var url = $"https://financialmodelingprep.com/stable/historical-price-eod/full" +
                  $"?symbol={symbol}" +
                  $"&from={from:yyyy-MM-dd}" +
                  $"&to={to:yyyy-MM-dd}" +
                  $"&apikey={_apiKey}";

        return await _http.GetStringAsync(url);
    }

    public async Task DownloadAllHistoricalData(string outputDir, int years = 2)
    {
        Directory.CreateDirectory(outputDir);

        var constituents = await GetSP500Symbols();
        Console.WriteLine($"Fetched {constituents.Count} S&P 500 constituents.");

        // Save constituent metadata (symbol, sector, weight) alongside the raw price files
        // so the simulation engine can load sector labels without hitting the API again.
        var constituentsPath = Path.Combine(outputDir, "..", "constituents.json");
        await File.WriteAllTextAsync(constituentsPath,
            JsonSerializer.Serialize(constituents, new JsonSerializerOptions { WriteIndented = false }));

        var to         = DateOnly.FromDateTime(DateTime.Today);
        var windowFrom = to.AddYears(-years);   // rolling window floor — data older than this is dropped

        int completed = 0;
        int skipped   = 0;
        int updated   = 0;
        int failed    = 0;

        foreach (var stock in constituents)
        {
            var filePath = Path.Combine(outputDir, $"{stock.Symbol}.json");

            try
            {
                List<DailyPrice>? existing = null;
                DateOnly fetchFrom = windowFrom;

                if (File.Exists(filePath))
                {
                    var existingJson = await File.ReadAllTextAsync(filePath);
                    existing = JsonSerializer.Deserialize<List<DailyPrice>>(existingJson, JsonOpts);

                    if (existing is { Count: > 0 })
                    {
                        var maxDate = existing.Max(p => p.Date);

                        if (maxDate >= to)
                        {
                            // Already up to date — no fetch needed
                            skipped++;
                            completed++;
                            if (completed % 50 == 0)
                                LogProgress(completed, constituents.Count, skipped, updated, failed);
                            continue;
                        }

                        fetchFrom = maxDate.AddDays(1);
                    }
                }

                var rawJson   = await FetchHistoricalPrices(stock.Symbol, fetchFrom, to);
                var newPrices = JsonSerializer.Deserialize<List<DailyPrice>>(rawJson, JsonOpts) ?? [];

                List<DailyPrice> merged;
                if (existing is { Count: > 0 } && newPrices.Count > 0)
                {
                    // Merge: combine, deduplicate by date, apply rolling window, sort newest-first
                    merged = existing
                        .Concat(newPrices)
                        .GroupBy(p => p.Date)
                        .Select(g => g.First())
                        .Where(p => p.Date >= windowFrom)
                        .OrderByDescending(p => p.Date)
                        .ToList();
                    updated++;
                }
                else
                {
                    // First download or no new data — write raw as-is (already newest-first from FMP)
                    merged = newPrices.Count > 0 ? newPrices : existing ?? [];
                }

                await File.WriteAllTextAsync(filePath,
                    JsonSerializer.Serialize(merged, new JsonSerializerOptions { WriteIndented = false }));
            }
            catch (Exception ex)
            {
                failed++;
                Console.WriteLine($"[WARN] Failed {stock.Symbol}: {ex.Message}");
            }

            completed++;
            if (completed % 50 == 0)
                LogProgress(completed, constituents.Count, skipped, updated, failed);

            await Task.Delay(250);
        }

        LogProgress(completed, constituents.Count, skipped, updated, failed);
        Console.WriteLine("Download complete.");
    }

    private static void LogProgress(int completed, int total, int skipped, int updated, int failed) =>
        Console.WriteLine($"[{completed}/{total}] skipped={skipped} updated={updated} failed={failed}");
}
