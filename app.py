//@version=5
strategy("Custom Reversal Engine (Column C + Coppock Matrix)", overlay=true, initial_capital=100000, default_qty_type=strategy.percent_of_equity, default_qty_value=100)

// ==========================================
// TIME WINDOW FILTER (Strict 2-Year Freeze: Jan 2025 - Dec 2026)
// ==========================================
startYear  = input.int(2025, title="Start Year")
startMonth = input.int(1, title="Start Month")
startDay   = input.int(1, title="Start Day")

endYear    = input.int(2026, title="End Year (Freeze)")
endMonth   = input.int(12, title="End Month (Freeze)")
endDay     = input.int(31, title="End Day (Freeze)")

// Check if current bar falls within the active 2-year freeze window
inDateRange = (time >= timestamp(startYear, startMonth, startDay, 00, 00)) and (time <= timestamp(endYear, endMonth, endDay, 23, 59))

// Ensure script strictly processes when on a 1-Hour Chart (60 minutes)
isHourly = timeframe.isminutes and timeframe.multiplier == 60

// ==========================================
// ENGINE 1: Column C (Delta Pulse Simulation)
// ==========================================
deltaLookback = input.int(14, title="Delta Pulse Lookback")
columnC_Delta = close - close[deltaLookback]

// ==========================================
// ENGINE 2: Coppock Hybrid Matrix
// ==========================================
longROCPeriod  = input.int(14, title="Coppock Long ROC")
shortROCPeriod = input.int(11, title="Coppock Short ROC")
wmaSmoothing   = input.int(10, title="Coppock WMA Smoothing")

longROC      = ta.roc(close, longROCPeriod)
shortROC     = ta.roc(close, shortROCPeriod)
rawMatrixSum = longROC + shortROC
coppockCurve = ta.wma(rawMatrixSum, wmaSmoothing)

// ==========================================
// SIGNAL CONDITIONS & CONVERGENCE LOOP
// ==========================================
// Bullish Reversal Hint: Column C turns positive & Coppock structural curve rises above 0
bullishHint = inDateRange and isHourly and (columnC_Delta > 0) and ta.crossover(coppockCurve, 0)

// Bearish Exhaustion Hint: Column C turns negative & Coppock curve drops below 0
bearishHint = inDateRange and isHourly and (columnC_Delta < 0) and ta.crossunder(coppockCurve, 0)

// ==========================================
// STRATEGY EXECUTION LOOPS (Position Management)
// ==========================================
if (bullishHint)
    strategy.entry("Long Hint", strategy.long)

if (bearishHint)
    strategy.entry("Short Hint", strategy.short)

// If time range expires, force close all positions to strictly freeze data testing
if (not inDateRange)
    strategy.close_all("Freeze Window End")

// ==========================================
// VISUAL PLOTS AND BACKGROUNDS
// ==========================================
plotshape(bullishHint and inDateRange, title="BUY HINT", style=shape.triangleup, location=location.belowbar, color=color.green, size=size.normal, text="LONG")
plotshape(bearishHint and inDateRange, title="SELL HINT", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.normal, text="SHORT")

bgcolor(bullishHint and inDateRange ? color.new(color.green, 85) : bearishHint and inDateRange ? color.new(color.red, 85) : na)
