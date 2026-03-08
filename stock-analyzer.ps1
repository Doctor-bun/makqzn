[CmdletBinding(DefaultParameterSetName = "Api")]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string]$Symbol,

    [Parameter(ParameterSetName = "Api")]
    [string]$ApiKey = $env:ALPHAVANTAGE_API_KEY,

    [Parameter(ParameterSetName = "Csv", Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$CsvPath,

    [Parameter(ParameterSetName = "Api")]
    [ValidateSet("compact", "full")]
    [string]$OutputSize = "compact",

    [ValidateSet("text", "json")]
    [string]$Format = "text",

    [double]$AccountSize,
    [double]$RiskPerTradePercent = 1,
    [double]$StopLossPercent = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-AlphaVantageDailySeries {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Ticker,

        [Parameter(Mandatory = $true)]
        [string]$Token,

        [Parameter(Mandatory = $true)]
        [string]$Size
    )

    if ([string]::IsNullOrWhiteSpace($Token)) {
        throw "缺少 Alpha Vantage API Key。请传入 -ApiKey 或设置环境变量 ALPHAVANTAGE_API_KEY。"
    }

    $uri = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=$Ticker&outputsize=$Size&apikey=$Token"
    $response = Invoke-RestMethod -Uri $uri -Method Get

    if ($response.Note) {
        throw "Alpha Vantage 限流: $($response.Note)"
    }

    if ($response.Information) {
        throw "Alpha Vantage 返回信息: $($response.Information)"
    }

    if ($response.'Error Message') {
        throw "无效代码或请求失败: $($response.'Error Message')"
    }

    $series = $response.'Time Series (Daily)'
    if (-not $series) {
        throw "没有拿到日线数据。"
    }

    $items = foreach ($dateKey in ($series.PSObject.Properties.Name | Sort-Object)) {
        $day = $series.$dateKey
        [PSCustomObject]@{
            Date   = [datetime]$dateKey
            Open   = [double]$day.'1. open'
            High   = [double]$day.'2. high'
            Low    = [double]$day.'3. low'
            Close  = [double]$day.'4. close'
            Volume = [double]$day.'5. volume'
        }
    }

    return $items
}

function Resolve-ColumnName {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Row,

        [Parameter(Mandatory = $true)]
        [string[]]$Candidates
    )

    foreach ($candidate in $Candidates) {
        if ($Row.PSObject.Properties.Name -contains $candidate) {
            return $candidate
        }
    }

    return $null
}

function Import-DailySeriesCsv {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "CSV 文件不存在: $Path"
    }

    $raw = Import-Csv -LiteralPath $Path
    if (-not $raw -or $raw.Count -eq 0) {
        throw "CSV 文件为空: $Path"
    }

    $probe = $raw[0]
    $dateColumn = Resolve-ColumnName -Row $probe -Candidates @("Date", "date", "timestamp")
    $openColumn = Resolve-ColumnName -Row $probe -Candidates @("Open", "open")
    $highColumn = Resolve-ColumnName -Row $probe -Candidates @("High", "high")
    $lowColumn = Resolve-ColumnName -Row $probe -Candidates @("Low", "low")
    $closeColumn = Resolve-ColumnName -Row $probe -Candidates @("Close", "close", "Adj Close", "adj_close")
    $volumeColumn = Resolve-ColumnName -Row $probe -Candidates @("Volume", "volume")

    if (-not ($dateColumn -and $openColumn -and $highColumn -and $lowColumn -and $closeColumn -and $volumeColumn)) {
        throw "CSV 需要包含 date/open/high/low/close/volume 这些列。"
    }

    $items = foreach ($row in $raw) {
        [PSCustomObject]@{
            Date   = [datetime]$row.$dateColumn
            Open   = [double]$row.$openColumn
            High   = [double]$row.$highColumn
            Low    = [double]$row.$lowColumn
            Close  = [double]$row.$closeColumn
            Volume = [double]$row.$volumeColumn
        }
    }

    return $items | Sort-Object Date
}

function Get-SmaSeries {
    param(
        [Parameter(Mandatory = $true)]
        [double[]]$Values,

        [Parameter(Mandatory = $true)]
        [int]$Period
    )

    $result = New-Object 'double[]' $Values.Length
    for ($i = 0; $i -lt $Values.Length; $i++) {
        if ($i + 1 -lt $Period) {
            $result[$i] = [double]::NaN
            continue
        }

        $sum = 0.0
        for ($j = $i - $Period + 1; $j -le $i; $j++) {
            $sum += $Values[$j]
        }

        $result[$i] = $sum / $Period
    }

    return $result
}

function Get-EmaSeries {
    param(
        [Parameter(Mandatory = $true)]
        [double[]]$Values,

        [Parameter(Mandatory = $true)]
        [int]$Period
    )

    $result = New-Object 'double[]' $Values.Length
    $multiplier = 2.0 / ($Period + 1)

    for ($i = 0; $i -lt $Values.Length; $i++) {
        if ($i -eq 0) {
            $result[$i] = $Values[$i]
            continue
        }

        $result[$i] = (($Values[$i] - $result[$i - 1]) * $multiplier) + $result[$i - 1]
    }

    return $result
}

function Get-RsiSeries {
    param(
        [Parameter(Mandatory = $true)]
        [double[]]$Values,

        [Parameter(Mandatory = $true)]
        [int]$Period
    )

    $result = New-Object 'double[]' $Values.Length
    for ($i = 0; $i -lt $Values.Length; $i++) {
        $result[$i] = [double]::NaN
    }

    if ($Values.Length -le $Period) {
        return $result
    }

    $gain = 0.0
    $loss = 0.0

    for ($i = 1; $i -le $Period; $i++) {
        $change = $Values[$i] - $Values[$i - 1]
        if ($change -ge 0) {
            $gain += $change
        }
        else {
            $loss += [math]::Abs($change)
        }
    }

    $avgGain = $gain / $Period
    $avgLoss = $loss / $Period
    $result[$Period] = if ($avgLoss -eq 0) { 100.0 } else { 100.0 - (100.0 / (1.0 + ($avgGain / $avgLoss))) }

    for ($i = $Period + 1; $i -lt $Values.Length; $i++) {
        $change = $Values[$i] - $Values[$i - 1]
        $currentGain = if ($change -gt 0) { $change } else { 0.0 }
        $currentLoss = if ($change -lt 0) { [math]::Abs($change) } else { 0.0 }

        $avgGain = (($avgGain * ($Period - 1)) + $currentGain) / $Period
        $avgLoss = (($avgLoss * ($Period - 1)) + $currentLoss) / $Period

        $result[$i] = if ($avgLoss -eq 0) { 100.0 } else { 100.0 - (100.0 / (1.0 + ($avgGain / $avgLoss))) }
    }

    return $result
}

function Get-AtrSeries {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Bars,

        [Parameter(Mandatory = $true)]
        [int]$Period
    )

    $trueRanges = New-Object 'double[]' $Bars.Length
    $atr = New-Object 'double[]' $Bars.Length

    for ($i = 0; $i -lt $Bars.Length; $i++) {
        $atr[$i] = [double]::NaN
        if ($i -eq 0) {
            $trueRanges[$i] = [double]($Bars[$i].High - $Bars[$i].Low)
            continue
        }

        $highLow = [double]($Bars[$i].High - $Bars[$i].Low)
        $highPrevClose = [math]::Abs([double]($Bars[$i].High - $Bars[$i - 1].Close))
        $lowPrevClose = [math]::Abs([double]($Bars[$i].Low - $Bars[$i - 1].Close))
        $trueRanges[$i] = [math]::Max($highLow, [math]::Max($highPrevClose, $lowPrevClose))
    }

    if ($Bars.Length -le $Period) {
        return $atr
    }

    $sum = 0.0
    for ($i = 0; $i -lt $Period; $i++) {
        $sum += $trueRanges[$i]
    }

    $atr[$Period - 1] = $sum / $Period

    for ($i = $Period; $i -lt $Bars.Length; $i++) {
        $atr[$i] = (($atr[$i - 1] * ($Period - 1)) + $trueRanges[$i]) / $Period
    }

    return $atr
}

function Get-AnnualizedVolatility {
    param(
        [Parameter(Mandatory = $true)]
        [double[]]$Values,

        [Parameter(Mandatory = $true)]
        [int]$Lookback
    )

    if ($Values.Length -lt ($Lookback + 1)) {
        return [double]::NaN
    }

    $returns = New-Object 'double[]' $Lookback
    $startIndex = $Values.Length - $Lookback

    for ($i = 0; $i -lt $Lookback; $i++) {
        $current = $Values[$startIndex + $i]
        $previous = $Values[$startIndex + $i - 1]
        $returns[$i] = [math]::Log($current / $previous)
    }

    $mean = ($returns | Measure-Object -Average).Average
    $sumSquares = 0.0
    foreach ($value in $returns) {
        $sumSquares += [math]::Pow(($value - $mean), 2)
    }

    $variance = $sumSquares / ($returns.Length - 1)
    return [math]::Sqrt($variance) * [math]::Sqrt(252)
}

function Get-MaxDrawdown {
    param(
        [Parameter(Mandatory = $true)]
        [double[]]$Values
    )

    if ($Values.Length -eq 0) {
        return [double]::NaN
    }

    $peak = $Values[0]
    $maxDrawdown = 0.0

    foreach ($value in $Values) {
        if ($value -gt $peak) {
            $peak = $value
        }

        $drawdown = ($value - $peak) / $peak
        if ($drawdown -lt $maxDrawdown) {
            $maxDrawdown = $drawdown
        }
    }

    return $maxDrawdown
}

function Get-MacdSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [double[]]$Values
    )

    $ema12 = Get-EmaSeries -Values $Values -Period 12
    $ema26 = Get-EmaSeries -Values $Values -Period 26
    $macdLine = New-Object 'double[]' $Values.Length

    for ($i = 0; $i -lt $Values.Length; $i++) {
        $macdLine[$i] = $ema12[$i] - $ema26[$i]
    }

    $signal = Get-EmaSeries -Values $macdLine -Period 9
    $lastIndex = $Values.Length - 1

    return [PSCustomObject]@{
        Line      = [math]::Round($macdLine[$lastIndex], 4)
        Signal    = [math]::Round($signal[$lastIndex], 4)
        Histogram = [math]::Round(($macdLine[$lastIndex] - $signal[$lastIndex]), 4)
    }
}

function Get-LatestNumber {
    param(
        [Parameter(Mandatory = $true)]
        [double[]]$Series
    )

    for ($i = $Series.Length - 1; $i -ge 0; $i--) {
        if (-not [double]::IsNaN($Series[$i])) {
            return $Series[$i]
        }
    }

    return [double]::NaN
}

function Get-DecisionLabel {
    param(
        [int]$TrendScore,
        [int]$MomentumScore,
        [int]$RiskScore
    )

    if ($RiskScore -lt 40 -or $TrendScore -le -2) {
        return "回避或等待企稳"
    }

    if ($TrendScore -ge 2 -and $MomentumScore -ge 1 -and $RiskScore -ge 65) {
        return "偏多, 只考虑分批参与"
    }

    if ($TrendScore -ge 1 -and $MomentumScore -ge 0 -and $RiskScore -ge 50) {
        return "中性偏多, 适合放入观察名单"
    }

    return "中性观望"
}

function Get-PositionSizing {
    param(
        [double]$Price,
        [double]$Account,
        [double]$RiskPercent,
        [double]$StopPercent
    )

    if ($Account -le 0 -or $RiskPercent -le 0 -or $StopPercent -le 0) {
        return $null
    }

    $riskBudget = $Account * ($RiskPercent / 100.0)
    $stopPrice = $Price * (1.0 - ($StopPercent / 100.0))
    $riskPerShare = $Price - $stopPrice

    if ($riskPerShare -le 0) {
        return $null
    }

    $shares = [math]::Floor($riskBudget / $riskPerShare)
    $positionValue = $shares * $Price

    return [PSCustomObject]@{
        RiskBudget    = [math]::Round($riskBudget, 2)
        StopPrice     = [math]::Round($stopPrice, 2)
        RiskPerShare  = [math]::Round($riskPerShare, 2)
        Shares        = [int]$shares
        PositionValue = [math]::Round($positionValue, 2)
    }
}

function Format-Percent {
    param([double]$Value)

    if ([double]::IsNaN($Value)) {
        return "N/A"
    }

    return ("{0:P2}" -f $Value)
}

function Format-Number {
    param([double]$Value)

    if ([double]::IsNaN($Value)) {
        return "N/A"
    }

    return ("{0:N2}" -f $Value)
}

function Write-TextReport {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Report
    )

    Write-Output "股票分析: $($Report.Symbol)"
    Write-Output "最新交易日: $($Report.AsOfDate)"
    Write-Output "收盘价: $($Report.Price)"
    Write-Output ""
    Write-Output "技术面"
    Write-Output "  SMA20 / SMA50 / SMA200: $($Report.Indicators.SMA20) / $($Report.Indicators.SMA50) / $($Report.Indicators.SMA200)"
    Write-Output "  RSI14: $($Report.Indicators.RSI14)"
    Write-Output "  MACD: $($Report.Indicators.MACD.Line)  Signal: $($Report.Indicators.MACD.Signal)  Hist: $($Report.Indicators.MACD.Histogram)"
    Write-Output "  ATR14: $($Report.Indicators.ATR14)"
    Write-Output ""
    Write-Output "风险面"
    Write-Output "  年化波动率(20日): $($Report.Risk.AnnualizedVolatility20)"
    Write-Output "  最大回撤: $($Report.Risk.MaxDrawdown)"
    Write-Output "  风险分: $($Report.Risk.RiskScore)/100"
    Write-Output ""
    Write-Output "信号"
    Write-Output "  趋势分: $($Report.Signal.TrendScore)"
    Write-Output "  动量分: $($Report.Signal.MomentumScore)"
    Write-Output "  决策建议: $($Report.Signal.Decision)"
    Write-Output "  备注: $($Report.Signal.Rationale)"

    if ($Report.PositionSizing) {
        Write-Output ""
        Write-Output "仓位控制"
        Write-Output "  单笔风险预算: $($Report.PositionSizing.RiskBudget)"
        Write-Output "  参考止损价: $($Report.PositionSizing.StopPrice)"
        Write-Output "  每股风险: $($Report.PositionSizing.RiskPerShare)"
        Write-Output "  理论股数: $($Report.PositionSizing.Shares)"
        Write-Output "  理论仓位金额: $($Report.PositionSizing.PositionValue)"
    }

    Write-Output ""
    Write-Output "免责声明: 这个工具只做规则化分析，不保证盈利，也不能替代你自己的研究和风险承受能力判断。"
}

if ($PSCmdlet.ParameterSetName -eq "Api") {
    $bars = Get-AlphaVantageDailySeries -Ticker $Symbol -Token $ApiKey -Size $OutputSize
}
else {
    $bars = Import-DailySeriesCsv -Path $CsvPath
}

if (-not $bars -or $bars.Count -lt 35) {
    throw "历史数据不足，至少需要 35 根日线。"
}

$bars = $bars | Sort-Object Date
$closes = [double[]]($bars | ForEach-Object { [double]$_.Close })
$lastBar = $bars[-1]

$sma20Series = Get-SmaSeries -Values $closes -Period 20
$sma50Series = Get-SmaSeries -Values $closes -Period 50
$sma200Series = Get-SmaSeries -Values $closes -Period 200
$rsiSeries = Get-RsiSeries -Values $closes -Period 14
$atrSeries = Get-AtrSeries -Bars $bars -Period 14
$macd = Get-MacdSnapshot -Values $closes

$sma20 = Get-LatestNumber -Series $sma20Series
$sma50 = Get-LatestNumber -Series $sma50Series
$sma200 = Get-LatestNumber -Series $sma200Series
$rsi14 = Get-LatestNumber -Series $rsiSeries
$atr14 = Get-LatestNumber -Series $atrSeries
$volatility20 = Get-AnnualizedVolatility -Values $closes -Lookback 20
$maxDrawdown = Get-MaxDrawdown -Values $closes
$atrPercent = if ([double]::IsNaN($atr14)) { [double]::NaN } else { $atr14 / $lastBar.Close }

$trendScore = 0
if (-not [double]::IsNaN($sma20)) {
    if ($lastBar.Close -gt $sma20) { $trendScore += 1 } else { $trendScore -= 1 }
}
if (-not [double]::IsNaN($sma50) -and -not [double]::IsNaN($sma20)) {
    if ($sma20 -gt $sma50) { $trendScore += 1 } else { $trendScore -= 1 }
}
if (-not [double]::IsNaN($sma200) -and -not [double]::IsNaN($sma50)) {
    if ($sma50 -gt $sma200) { $trendScore += 1 } else { $trendScore -= 1 }
}

$momentumScore = 0
if (-not [double]::IsNaN($rsi14)) {
    if ($rsi14 -ge 50 -and $rsi14 -le 70) { $momentumScore += 1 }
    elseif ($rsi14 -gt 75) { $momentumScore -= 1 }
    elseif ($rsi14 -lt 40) { $momentumScore -= 1 }
}
if ($macd.Line -gt $macd.Signal) { $momentumScore += 1 } else { $momentumScore -= 1 }

$riskScore = 100
if (-not [double]::IsNaN($volatility20)) {
    if ($volatility20 -gt 0.60) { $riskScore -= 30 }
    elseif ($volatility20 -gt 0.40) { $riskScore -= 20 }
    elseif ($volatility20 -gt 0.25) { $riskScore -= 10 }
}
if (-not [double]::IsNaN($maxDrawdown)) {
    if ($maxDrawdown -lt -0.50) { $riskScore -= 30 }
    elseif ($maxDrawdown -lt -0.30) { $riskScore -= 20 }
    elseif ($maxDrawdown -lt -0.15) { $riskScore -= 10 }
}
if (-not [double]::IsNaN($atrPercent)) {
    if ($atrPercent -gt 0.05) { $riskScore -= 20 }
    elseif ($atrPercent -gt 0.03) { $riskScore -= 10 }
}
$riskScore = [math]::Max(0, [math]::Min(100, $riskScore))

$rationaleParts = New-Object System.Collections.Generic.List[string]
if ($trendScore -ge 2) {
    $rationaleParts.Add("均线结构偏强")
}
elseif ($trendScore -le -2) {
    $rationaleParts.Add("均线结构偏弱")
}
else {
    $rationaleParts.Add("趋势没有明显倾向")
}

if (-not [double]::IsNaN($rsi14)) {
    if ($rsi14 -gt 75) {
        $rationaleParts.Add("RSI 过热")
    }
    elseif ($rsi14 -lt 40) {
        $rationaleParts.Add("RSI 偏弱")
    }
    else {
        $rationaleParts.Add("RSI 处于可接受区间")
    }
}

if ($riskScore -lt 50) {
    $rationaleParts.Add("波动或回撤偏大")
}
elseif ($riskScore -ge 70) {
    $rationaleParts.Add("风险指标相对可控")
}

$positionSizing = Get-PositionSizing -Price $lastBar.Close -Account $AccountSize -RiskPercent $RiskPerTradePercent -StopPercent $StopLossPercent

$report = [PSCustomObject]@{
    Symbol         = $Symbol.ToUpperInvariant()
    AsOfDate       = $lastBar.Date.ToString("yyyy-MM-dd")
    Price          = Format-Number -Value $lastBar.Close
    DataPoints     = $bars.Count
    Indicators     = [PSCustomObject]@{
        SMA20  = Format-Number -Value $sma20
        SMA50  = Format-Number -Value $sma50
        SMA200 = Format-Number -Value $sma200
        RSI14  = Format-Number -Value $rsi14
        ATR14  = Format-Number -Value $atr14
        MACD   = $macd
    }
    Risk           = [PSCustomObject]@{
        AnnualizedVolatility20 = Format-Percent -Value $volatility20
        MaxDrawdown            = Format-Percent -Value $maxDrawdown
        RiskScore              = $riskScore
    }
    Signal         = [PSCustomObject]@{
        TrendScore    = $trendScore
        MomentumScore = $momentumScore
        Decision      = Get-DecisionLabel -TrendScore $trendScore -MomentumScore $momentumScore -RiskScore $riskScore
        Rationale     = ($rationaleParts -join "; ")
    }
    PositionSizing = $positionSizing
    Disclaimer     = "规则化分析，不保证盈利，不构成个性化投资建议。"
}

if ($Format -eq "json") {
    $report | ConvertTo-Json -Depth 5
}
else {
    Write-TextReport -Report $report
}
