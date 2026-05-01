using DirectIndexing.DataCollection;

//simulation singletons (acting as sub-runtimes within the this .NET runtime) that utilize the portfoliostate and lot models/state spaces

var mode = args.FirstOrDefault() ?? "simulate";
switch (mode)
{
    case "download":
    {
        var apiKey = Environment.GetEnvironmentVariable("FMP_API_KEY")
                     ?? throw new InvalidOperationException(
                         "FMP_API_KEY environment variable is not set. " +
                         "Export it before running: export FMP_API_KEY=your_key_here");

        await new MarketDataDownloader(apiKey).DownloadAllHistoricalData("data/raw", years: 2); 
    }
    break;
    case "simulate": throw new NotImplementedException();break;
    case "train": throw new NotImplementedException(); break;
    case "test":
    //production unit tests move to a tests project for .NET grade testing features.
    PortfolioStateTests tests = new PortfolioStateTests();
    tests.Test_HarvestLoss_DecreasesGYTD();
    tests.Test_WashSaleClock_StartsAtZeroAfterHarvest();
    tests.Test_OracleBlocked_WhenGYTD_IsNegative();
    Console.WriteLine("All tests passed.");
    break;
}
