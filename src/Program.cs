using DirectIndexing.Core.Simulation;
using DirectIndexing.DataCollection;
using DirectIndexing.Export;

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
    case "simulate":
    {
        var loader = new PriceLoader();
        loader.Load("data/raw", "data/constituents.json");

        var engine    = new SimulationEngine(loader);
        var snapshots = engine.Run(initialPortfolioValue: 1_000_000m);

        var softLabeller = new SoftLabelBuilder(loader);
        softLabeller.Label(snapshots);

        SimulationExporter.WriteCsv(snapshots, "data/lots.csv");
    }
    break;
    // Alternate: Monte Carlo simulation with synthetic GBM prices.
    // Calibrates per-stock σ from the same real data then generates N days of GBM paths.
    // Produces data/lots-mc.csv alongside the backtesting data/lots.csv.
    case "simulate-mc":
    {
        var loader = new PriceLoader();
        loader.Load("data/raw", "data/constituents.json");

        var mcEngine  = new MonteCarloEngine(loader, annualDrift: 0f);
        var snapshots = mcEngine.Run(
            initialPortfolioValue: 1_000_000m,
            simDays:    504,
            warmupDays: 200,
            seed:       42);

        SimulationExporter.WriteCsv(snapshots, "data/lots-mc.csv");
    }
    break;
    case "train": throw new NotImplementedException("Training not yet built.");
    //case "results" will likely be in python to generate the ipynb performance report
    case "test":
    {
        // v0.1 smoke tests — simple Debug.Assert runners.
        // Move to a proper xUnit/NUnit project when the simulation layer is added.

        var portfolioTests = new PortfolioStateTests();
        portfolioTests.Test_HarvestLoss_DecreasesGYTD();
        portfolioTests.Test_WashSaleClock_StartsAtZeroAfterHarvest();
        portfolioTests.Test_OracleBlocked_WhenGYTD_IsNegative();
        portfolioTests.Test_SeedGYTD_EnablesOracleGate();
        portfolioTests.Test_SeedGYTD_ReSeeds_AfterYearEndReset();

        var oracleTests = new OracleBoundaryTests();
        oracleTests.Test_Oracle_FiresWhenAllConditionsMet();
        oracleTests.Test_Oracle_Blocked_WhenLossInsufficient();
        oracleTests.Test_Oracle_Blocked_WhenTEOverBudget();
        oracleTests.Test_Oracle_Blocked_WhenGYTD_Zero();
        oracleTests.Test_Oracle_Blocked_WhenWashSaleActive();
        oracleTests.Test_Oracle_Fires_AtWashSaleBoundary();

        var teTests = new TrackingErrorProxyTests();
        teTests.Test_SigmaTE_Zero_WhenPortfolioEqualsFullBenchmark();
        teTests.Test_SigmaTE_Positive_WhenPortfolioExcludesDivergingTicker();
        teTests.Test_SigmaTE_StaysBounded_AfterStructuralLotRemoval();
        teTests.Test_SigmaTE_Zero_WithInsufficientHistory();

        var gbmTests = new GbmSimulatorTests();
        gbmTests.Test_SimulatePaths_AllPricesPositive();
        gbmTests.Test_SimulatePaths_SeedPriceAtStepZero();
        gbmTests.Test_FractionFiring_Zero_WhenConditionNeverMet();
        gbmTests.Test_FractionFiring_One_WhenConditionAlwaysFires();
        gbmTests.Test_FractionFiring_InRange_ForRealisticPredicate();
        gbmTests.Test_NextGaussian_NearStandardNormal();

        Console.WriteLine("All tests passed.");
    }
    break;
}
