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
        var url = $"https://financialmodelingprep.com/stable/sp500-constituent?apikey={_apiKey}";
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

        var to   = DateOnly.FromDateTime(DateTime.Today);
        var from = to.AddYears(-years);

        int completed = 0;
        int skipped   = 0;
        int failed    = 0;

        foreach (var stock in constituents)
        {
            var filePath = Path.Combine(outputDir, $"{stock.Symbol}.json");

            if (File.Exists(filePath))
            {
                skipped++;
                completed++;
                if (completed % 50 == 0)
                    LogProgress(completed, constituents.Count, skipped, failed);
                continue;
            }

            try
            {
                var rawJson = await FetchHistoricalPrices(stock.Symbol, from, to);
                await File.WriteAllTextAsync(filePath, rawJson);
            }
            catch (Exception ex)
            {
                failed++;
                Console.WriteLine($"[WARN] Failed {stock.Symbol}: {ex.Message}");
            }

            completed++;
            if (completed % 50 == 0)
                LogProgress(completed, constituents.Count, skipped, failed);

            await Task.Delay(250);
        }

        LogProgress(completed, constituents.Count, skipped, failed);
        Console.WriteLine("Download complete.");
    }

    private static void LogProgress(int completed, int total, int skipped, int failed) =>
        Console.WriteLine($"[{completed}/{total}] skipped={skipped} failed={failed}");
}
