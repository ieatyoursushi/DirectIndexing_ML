using System.Text.Json;
using ClosedXML.Excel;
using System.Globalization;

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
        const string url =
            "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx";

        await using var stream = await _http.GetStreamAsync(url);
        using var workbook = new XLWorkbook(stream);
        var worksheet = workbook.Worksheets.First();

        var rows = worksheet.RowsUsed().ToList();

        var headerRow = rows.FirstOrDefault(row =>
            row.CellsUsed().Any(cell =>
                string.Equals(cell.GetString().Trim(), "Ticker", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(cell.GetString().Trim(), "Symbol", StringComparison.OrdinalIgnoreCase)));

        if (headerRow is null)
            throw new InvalidOperationException("Could not find holdings header row in SPY XLSX.");

        var headers = headerRow.CellsUsed()
            .ToDictionary(
                cell => cell.GetString().Trim().ToLowerInvariant(),
                cell => cell.Address.ColumnNumber);

        int tickerCol = FindCol(headers, "ticker", "symbol");
        int nameCol = FindCol(headers, "name", "security name", "holding name");
        int weightCol = FindCol(headers, "weight", "weight (%)", "% of fund");
        int sectorCol = FindOptionalCol(headers, "sector");

        var constituents = new List<SP500Constituent>();

        foreach (var row in rows.Where(row => row.RowNumber() > headerRow.RowNumber()))
        {
            var rawTicker = row.Cell(tickerCol).GetString().Trim();

            if (string.IsNullOrWhiteSpace(rawTicker))
                continue;

            if (rawTicker.Equals("Ticker", StringComparison.OrdinalIgnoreCase) ||
                rawTicker.Equals("CASH_USD", StringComparison.OrdinalIgnoreCase) ||
                rawTicker.Contains("USD", StringComparison.OrdinalIgnoreCase))
                continue;

            var symbol = NormalizeTickerForFmp(rawTicker);

            var name = row.Cell(nameCol).GetString().Trim();
            if (string.IsNullOrWhiteSpace(name))
                name = symbol;

            var weight = ParseWeight(row.Cell(weightCol).GetString());

            var sector = sectorCol > 0
                ? row.Cell(sectorCol).GetString().Trim()
                : string.Empty;

            constituents.Add(new SP500Constituent(
                symbol,
                name,
                sector,
                string.Empty,
                weight));
        }

        constituents = constituents
            .Where(constituent => !string.IsNullOrWhiteSpace(constituent.Symbol))
            .GroupBy(constituent => constituent.Symbol)
            .Select(group => group.First())
            .OrderBy(constituent => constituent.Symbol)
            .ToList();

        if (constituents.Count < 450)
            throw new InvalidOperationException(
                $"SPY holdings scrape only returned {constituents.Count} tickers. XLSX format likely changed.");

        return constituents;
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

    public async Task DownloadAllHistoricalData(
        string outputDir,
        int years = 2,
        int warmupTradingDays = 200)
    {
        Directory.CreateDirectory(outputDir);

        var constituents = await GetSP500Symbols();
        Console.WriteLine($"Fetched {constituents.Count} S&P 500 constituents.");

        // Save constituent metadata (symbol, sector, weight) alongside the raw price files
        // so the simulation engine can load sector labels without hitting the API again.
        var constituentsPath = Path.Combine(outputDir, "..", "constituents.json");
        await File.WriteAllTextAsync(constituentsPath,
            JsonSerializer.Serialize(constituents, new JsonSerializerOptions { WriteIndented = true }));

        var to = DateOnly.FromDateTime(DateTime.Today);

        // Extend the fetch window backward to cover the warmup period (e.g. 200 trading days
        // for MA-200) so the warmup sits *before* the n-year simulation window rather than
        // consuming part of it.  200 trading days ≈ 290 calendar days (200 × 365.25 / 252).
        int extraCalendarDays = (int)Math.Ceiling(warmupTradingDays * 365.25 / 252.0);
        var windowFrom = to.AddYears(-years).AddDays(-extraCalendarDays);  // rolling window floor

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
                    try
                    {
                        var existingJson = await File.ReadAllTextAsync(filePath);
                        existing = JsonSerializer.Deserialize<List<DailyPrice>>(existingJson, JsonOpts);
                    }
                    catch (JsonException ex)
                    {
                        // Corrupted or truncated file — discard it and do a full re-fetch
                        Console.WriteLine($"[WARN] Corrupted cache for {stock.Symbol}, re-fetching: {ex.Message}");
                        existing = null;
                    }

                    if (existing is { Count: > 0 })
                    {
                        var maxDate = existing.Max(p => p.Date);
                        var minDate = existing.Min(p => p.Date);

                        bool needsNewData = maxDate < to;
                        bool needsOldData = minDate > windowFrom;  // file doesn't reach back far enough

                        if (!needsNewData && !needsOldData)
                        {
                            // Both ends of the window are covered — no fetch needed
                            skipped++;
                            completed++;
                            if (completed % 50 == 0)
                                LogProgress(completed, constituents.Count, skipped, updated, failed);
                            continue;
                        }

                        // If we need older history, start the fetch from the window floor so FMP
                        // returns the full missing range; the merge+dedup step below reconciles it.
                        // If we only need newer data, start the day after what we already have.
                        fetchFrom = needsOldData ? windowFrom : maxDate.AddDays(1);
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
                    JsonSerializer.Serialize(merged, new JsonSerializerOptions { WriteIndented = true }));
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

    private static int FindCol(Dictionary<string, int> headers, params string[] names)
    {
        foreach (var name in names)
        {
            if (headers.TryGetValue(name.ToLowerInvariant(), out var col))
                return col;
        }

        throw new InvalidOperationException($"Missing required column. Tried: {string.Join(", ", names)}");
    }

    private static int FindOptionalCol(Dictionary<string, int> headers, params string[] names)
    {
        foreach (var name in names)
        {
            if (headers.TryGetValue(name.ToLowerInvariant(), out var col))
                return col;
        }

        return -1;
    }

    private static string NormalizeTickerForFmp(string ticker)
    {
        ticker = ticker.Trim().ToUpperInvariant();
        return ticker.Replace('.', '-');
    }

    private static decimal ParseWeight(string raw)
    {
        raw = raw.Trim()
            .Replace("%", "")
            .Replace(",", "");

        if (decimal.TryParse(raw, NumberStyles.Any, CultureInfo.InvariantCulture, out var value))
            return value;

        return 0m;
    }
}
