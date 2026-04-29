using DirectIndexing.DataCollection;

var apiKey = Environment.GetEnvironmentVariable("FMP_API_KEY")
             ?? throw new InvalidOperationException(
                 "FMP_API_KEY environment variable is not set. " +
                 "Export it before running: export FMP_API_KEY=your_key_here");

var downloader = new MarketDataDownloader(apiKey);
await downloader.DownloadAllHistoricalData("data/raw", years: 2);
