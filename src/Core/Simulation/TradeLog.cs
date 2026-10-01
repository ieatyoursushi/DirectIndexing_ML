using DirectIndexing.Core.Portfolio;

namespace DirectIndexing.Core.Simulation;

public enum TradeKind { Buy, Sell }

/// <summary>
/// One executed trade. The engine appends one per lot opened (initial book, reopen,
/// contribution) and one per lot sold (harvest). <see cref="Lot"/> is the lot object
/// itself, so a sale and the acquisition of the lot being sold can be told apart by
/// reference — which is exactly the distinction §1091 turns on.
/// </summary>
public sealed record TradeEvent(
    DateOnly  Date,
    int       Day,
    string    Symbol,
    TradeKind Kind,
    Lot       Lot,
    decimal   Price,
    decimal   RealizedGain);

/// <summary>
/// An independent restatement of 26 USC §1091 over a trade log — deliberately NOT the
/// engine's own gating logic, so it can audit that logic:
///
///   a loss sale S of ticker A on date d is a wash sale iff some OTHER lot of A was
///   acquired on a date d' with |d − d'| ≤ 30 calendar days (the 61-day window).
///
/// The lot being sold is excluded: buying a lot and selling that same lot at a loss is
/// not a wash sale; acquiring *replacement* shares around the sale is.
/// [math:wash_audit] — DataMemo/spec/SymbolTable.md
/// </summary>
public static class WashSaleAudit
{
    public const int WindowCalendarDays = 30;

    public static List<(TradeEvent Sale, TradeEvent Buy)> Violations(IReadOnlyList<TradeEvent> trades)
    {
        var found = new List<(TradeEvent, TradeEvent)>();
        foreach (var bySymbol in trades.GroupBy(e => e.Symbol))
        {
            var buys = bySymbol.Where(e => e.Kind == TradeKind.Buy).OrderBy(e => e.Date).ToList();
            foreach (var sale in bySymbol.Where(e => e.Kind == TradeKind.Sell && e.RealizedGain < 0m))
            {
                int d = sale.Date.DayNumber;
                var replacement = buys.FirstOrDefault(b =>
                    !ReferenceEquals(b.Lot, sale.Lot) &&
                    Math.Abs(b.Date.DayNumber - d) <= WindowCalendarDays);
                if (replacement is not null)
                    found.Add((sale, replacement));
            }
        }
        return found;
    }
}
