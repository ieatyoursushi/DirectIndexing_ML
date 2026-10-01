using System.Text.Json;

namespace DirectIndexing.Core.Policy;

/// <summary>
/// One run's economic scoreboard (v0.3-11) — the rung record of the ladder
/// (DataMemo/decisions/PolicyLayer_v04.md §6). Dollars unless stated.
/// Identities (tested): TaxPosition = SumTradeDeltaW + RollTrueUp, and
/// SumTradeDeltaW = BenefitUsedNow + BenefitBanked − GainTaxCost.
/// </summary>
public sealed record RunMetrics(
    string  Policy,
    int     Days,
    decimal InitialValue,
    decimal Contributions,
    decimal TerminalHoldings,
    decimal PendingReopenCash,
    decimal Cash,
    decimal TaxPosition,
    decimal AfterTaxWealth,
    decimal LiquidationValue,
    int     LossSales,
    int     GainSales,
    decimal BenefitUsedNow,
    decimal BenefitBanked,
    decimal GainTaxCost,
    decimal SumTradeDeltaW,
    decimal RollTrueUp,
    double  ExAnteTeMean,
    double  RealizedTe,
    double  AnnualTurnover,
    decimal TradingCosts,
    int     WashViolations)
{
    public void Write(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true }));
    }
}
