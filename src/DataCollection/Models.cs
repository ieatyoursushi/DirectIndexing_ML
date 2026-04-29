using System.Text.Json.Serialization;

namespace DirectIndexing.DataCollection;

public record SP500Constituent(
    [property: JsonPropertyName("symbol")]   string Symbol,
    [property: JsonPropertyName("name")]     string Name,
    [property: JsonPropertyName("sector")]   string Sector,
    [property: JsonPropertyName("subSector")] string SubSector,
    [property: JsonPropertyName("weight")]   decimal Weight
);

public record DailyPrice(
    [property: JsonPropertyName("symbol")]        string   Symbol,
    [property: JsonPropertyName("date")]          DateOnly Date,
    [property: JsonPropertyName("open")]          decimal  Open,
    [property: JsonPropertyName("high")]          decimal  High,
    [property: JsonPropertyName("low")]           decimal  Low,
    [property: JsonPropertyName("close")]         decimal  Close,
    [property: JsonPropertyName("volume")]        long     Volume,
    [property: JsonPropertyName("change")]        decimal  Change,
    [property: JsonPropertyName("changePercent")] decimal  ChangePercent,
    [property: JsonPropertyName("vwap")]          decimal  Vwap
);
