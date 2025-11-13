# Task 015: Competitive Baseline Tracking

**Status**: Pending
**Estimated Duration**: 3-4 days
**Priority**: High - Market positioning validation

## Objective

Automate weekly competitive benchmarks (Neo4j, Qdrant) to track relative performance over time. Generate automated reports showing competitive positioning with historical trend analysis.

## Architecture

```bash
#!/bin/bash
# scripts/weekly_competitive_benchmark.sh

echo "=== Weekly Competitive Benchmark ==="
date

# Run all competitive scenarios
for scenario in scenarios/competitive/*.toml; do
    name=$(basename "$scenario" .toml)

    echo "Running $name..."
    ./scripts/m17_performance_check.sh "weekly_$name" before --competitive

    # Extract results
    latest=$(ls -t tmp/m17_performance/competitive_weekly_${name}_before_*.json | head -1)
    jq -r '{p99: .p99_latency_ms, throughput: .overall_throughput}' "$latest" > "tmp/weekly/$name.json"
done

# Generate trend report
python3 scripts/analyze_competitive_trends.py \
    --input tmp/weekly/ \
    --output tmp/weekly/trend_report.html

# Upload to dashboard
curl -X POST https://metrics.engram.io/competitive \
    -H "Authorization: Bearer $METRICS_API_KEY" \
    --data @tmp/weekly/trend_report.json
```

## Trend Analysis

```python
# scripts/analyze_competitive_trends.py

def analyze_trend(historical_data: List[dict]) -> dict:
    """Detect performance trends over time."""

    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['timestamp'])
    df.set_index('date', inplace=True)

    # Calculate rolling average (4-week window)
    df['p99_ma'] = df['p99_latency_ms'].rolling(window=4).mean()

    # Detect trend (linear regression)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['p99_latency_ms'].values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]

    # Classify trend
    if slope < -0.5:
        trend = "improving"
    elif slope > 0.5:
        trend = "regressing"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "slope_ms_per_week": slope,
        "current_p99": df['p99_latency_ms'].iloc[-1],
        "4week_avg": df['p99_ma'].iloc[-1],
    }
```

## Success Criteria

- **Automation**: Weekly benchmarks run without manual intervention
- **Trend Detection**: Detect >5% monthly regression automatically
- **Alerting**: Slack notification if competitive gap shrinks
- **Historical Tracking**: 6+ months of data for trend analysis

## Files

- `scripts/weekly_competitive_benchmark.sh` (200 lines)
- `scripts/analyze_competitive_trends.py` (350 lines)
- `scripts/upload_to_dashboard.sh` (80 lines)
- `docs/operations/competitive_benchmarking.md` (180 lines)
