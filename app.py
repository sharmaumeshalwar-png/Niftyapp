//@version=5
indicator("Custom Reversal Engine (Column C + Coppock Matrix)", overlay=true, initial_capital=100000)

// ==========================================
// TIME WINDOW FILTER (January 2026 Onwards)
// ==========================================
startYear  = input.int(2026, title="Start Year")
startMonth = input.int(1, title="Start Month")
startDay   = input.int(1, title="Start Day")

// Check if current bar falls within the active backtest window
inDateRange = time >= timestamp(startYear, startMonth, startDay, 00, 00)
// Ensure script strictly processes when on a 1-Hour Chart (60 minutes)
isHourly = timeframe.isminutes and timeframe.multiplier == 60

// ==========================================
// ENGINE 1: Column C (Delta Pulse Simulation)
// ==========================================
// Column C measures raw high-velocity momentum shifts
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
// VISUAL PLOTS AND ALERTS (On Chart)
// ==========================================
// Plot shapes on the price chart exactly when hints are locked
plotshape(bullishHint, title="BUY HINT (Reversal)", style=shape.triangleup, location=location.belowbar, color=color.green, size=size.normal, text="LONG HINT")
plotshape(bearishHint, title="SELL HINT (Exhaustion)", style=shape.triangledown, location=location.abovebar, color=color.red, size=size.normal, text="SHORT HINT")

// Background tinting for clear visual zone identification
bgcolor(bullishHint ? color.new(color.green, 85) : bearishHint ? color.new(color.red, 85) : na)

// ==========================================
// SYSTEM ALERT TRIGGERS
// ==========================================
alertcondition(bullishHint, title="Bullish Convergence Alert", message="System detected a Strong Reversal Long Hint - Column C & Coppock Matrix Aligned!")
alertcondition(bearishHint, title="Bearish Exhaustion Alert", message="System detected a Bearish Exhaustion Hint - Exiting/Short Alignment Locked!")
