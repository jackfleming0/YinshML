# Memory Management Dashboard Guide

This guide covers the YINSH ML memory management dashboard system, providing real-time monitoring, alerting, and visualization capabilities for memory pool performance.

## Overview

The memory management dashboard provides comprehensive monitoring of:
- **Memory Pool Performance**: Real-time utilization, hit rates, allocation patterns
- **System Health**: Memory pressure, fragmentation, performance bottlenecks  
- **Alerting**: Configurable rules with multiple notification channels
- **Performance Metrics**: Latency percentiles, throughput, trends
- **Export Capabilities**: Prometheus, Grafana, JSON, CSV formats

## Dashboard Modes

### 1. Streamlit Web Dashboard

**Interactive web-based monitoring with real-time charts and visualizations.**

```bash
# Launch web dashboard
python scripts/run_memory_dashboard.py --mode streamlit

# Custom port and settings
python scripts/run_memory_dashboard.py --mode streamlit \
    --port 8502 \
    --update-interval 2.0 \
    --enable-alerts \
    --enable-health-checks
```

**Features:**
- Real-time metric charts and graphs
- Interactive pool utilization monitoring
- Alert status and health check displays
- Auto-refresh with configurable intervals
- Export controls and data visualization

**Access:** Open browser to `http://localhost:8501` (or specified port)

### 2. Console Dashboard

**Terminal-based monitoring for servers and headless environments.**

```bash
# Console monitoring
python scripts/run_memory_dashboard.py --mode console

# With alerts and duration limit
python scripts/run_memory_dashboard.py --mode console \
    --enable-alerts \
    --enable-health-checks \
    --duration 3600  # Run for 1 hour
```

**Features:**
- Real-time terminal updates
- Memory overview and performance metrics
- Active alert notifications
- Pool utilization status
- Clean text-based interface

### 3. Export Mode

**Batch data collection and export for analysis and integration.**

```bash
# Export to JSON
python scripts/run_memory_dashboard.py --mode export \
    --format json \
    --output memory_report.json \
    --duration 300  # Collect for 5 minutes

# Export to Prometheus format
python scripts/run_memory_dashboard.py --mode export \
    --format prometheus \
    --output metrics.prom

# Export Grafana dashboard config
python scripts/run_memory_dashboard.py --mode export \
    --format grafana \
    --output yinsh_dashboard.json
```

## Configuration Options

### Core Settings

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--update-interval` | Metrics collection interval (seconds) | 5.0 | `--update-interval 2.0` |
| `--max-data-points` | Maximum metrics history | 1000 | `--max-data-points 5000` |
| `--log-level` | Logging verbosity | INFO | `--log-level DEBUG` |

### Monitoring Features

| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `--enable-alerts` | Activate alerting system | False | `--enable-alerts` |
| `--enable-health-checks` | Enable health monitoring | False | `--enable-health-checks` |
| `--export-directory` | Auto-export data location | None | `--export-directory ./exports` |
| `--export-interval` | Auto-export frequency (seconds) | 300 | `--export-interval 600` |

### Export Settings

| Option | Description | Values | Example |
|--------|-------------|--------|---------|
| `--format` | Output format | json, csv, prometheus, grafana | `--format prometheus` |
| `--output` | Output file path | Auto-generated | `--output metrics.prom` |
| `--duration` | Collection duration (seconds) | 60 | `--duration 1800` |

## Monitoring Components

### Metrics Collection

The dashboard collects comprehensive metrics including:

**Memory Pool Metrics:**
- Pool utilization percentages
- Hit/miss rates
- Current and maximum pool sizes
- Allocation/deallocation rates

**Performance Metrics:**
- Allocation latency percentiles (P50, P95, P99)
- Memory pressure and fragmentation
- System resource utilization
- Garbage collection statistics

**System Health:**
- Overall health scores
- Individual check results
- Trend analysis
- Performance recommendations

### Alert System

**Default Alert Rules:**
- High memory pressure (>85%)
- Low pool hit rates (<90%)
- High allocation latency (>1000μs P95)
- Memory fragmentation issues
- System resource constraints

**Custom Alert Configuration:**
```python
# Example custom alert rule
from yinsh_ml.monitoring import AlertRule, AlertCondition, AlertSeverity

custom_rule = AlertRule(
    name="custom_latency_alert",
    condition=AlertCondition.THRESHOLD,
    metric_name="allocation_latency_p95",
    threshold=500.0,  # 500μs
    severity=AlertSeverity.WARNING,
    message="Allocation latency exceeding 500μs"
)
```

### Health Monitoring

**Health Check Categories:**
- **Pool Health**: Utilization, hit rates, sizing
- **Allocation Performance**: Latency, throughput
- **Memory Fragmentation**: Index calculations, trends  
- **System Resources**: CPU, memory, I/O pressure
- **Allocation Patterns**: Request distribution, hotspots

**Health Status Levels:**
- `HEALTHY`: All systems operating normally
- `WARNING`: Performance degradation detected
- `CRITICAL`: Immediate attention required
- `UNKNOWN`: Unable to determine status

## Integration with External Systems

### Prometheus Integration

**1. Export Prometheus Configuration:**
```bash
python scripts/run_memory_dashboard.py --mode export \
    --format prometheus \
    --output /etc/prometheus/yinsh_metrics.prom
```

**2. Configure Prometheus Scraping:**
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'yinsh-memory'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    metrics_path: '/metrics'
```

**3. Start Metrics Server:**
```python
from yinsh_ml.monitoring import MemoryMetricsCollector, MetricsExporter

# In your application
collector = MemoryMetricsCollector(memory_manager)
exporter = MetricsExporter(collector)
exporter.start_prometheus_server(port=8000)
```

### Grafana Dashboard

**1. Generate Dashboard Config:**
```bash
python scripts/run_memory_dashboard.py --mode export \
    --format grafana \
    --output yinsh_grafana_dashboard.json
```

**2. Import to Grafana:**
- Open Grafana web interface
- Navigate to **Dashboards > Import**
- Upload the generated JSON file
- Configure data source (Prometheus)

**3. Dashboard Features:**
- Memory pool utilization charts
- Allocation performance graphs
- Alert status panels
- Health check summaries
- System resource monitoring

### Custom Integrations

**Programmatic API Usage:**
```python
from yinsh_ml.monitoring import (
    MemoryMetricsCollector, 
    MemoryDashboard, 
    DashboardConfig
)

# Create and configure dashboard
config = DashboardConfig(
    update_interval=1.0,
    enable_alerts=True,
    export_directory="./monitoring_data"
)

dashboard = MemoryDashboard(metrics_collector, config)
dashboard.start_monitoring()

# Access real-time data
current_metrics = dashboard.metrics_collector.get_current_metrics()
active_alerts = dashboard.alert_manager.get_active_alerts()
health_status = dashboard.health_checker.run_comprehensive_health_check()
```

## Production Deployment

### Service Configuration

**Systemd Service Example:**
```ini
# /etc/systemd/system/yinsh-dashboard.service
[Unit]
Description=YINSH ML Memory Dashboard
After=network.target

[Service]
Type=simple
User=yinsh
WorkingDirectory=/opt/yinsh-ml
ExecStart=/opt/yinsh-ml/venv/bin/python scripts/run_memory_dashboard.py --mode console --enable-alerts --enable-health-checks
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Docker Deployment:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Web dashboard
EXPOSE 8501
# Metrics endpoint  
EXPOSE 8000

CMD ["python", "scripts/run_memory_dashboard.py", "--mode", "streamlit", "--enable-alerts"]
```

### Security Considerations

**Network Security:**
- Run dashboards on internal networks only
- Use reverse proxy for external access
- Implement authentication for Streamlit dashboard
- Restrict Prometheus metrics access

**Data Privacy:**
- Configure export directory permissions
- Rotate exported metric files
- Sanitize sensitive information from alerts
- Use secure channels for alert notifications

## Troubleshooting

### Common Issues

**Dashboard Won't Start:**
```bash
# Check dependencies
pip install streamlit plotly pandas

# Verify Python path
export PYTHONPATH=/path/to/yinsh-ml:$PYTHONPATH

# Check port availability
netstat -tlnp | grep 8501
```

**No Metrics Data:**
```bash
# Check memory manager initialization
python -c "from yinsh_ml.memory import MemoryManager; print(MemoryManager())"

# Verify metrics collection
python -c "
from yinsh_ml.monitoring import MemoryMetricsCollector
collector = MemoryMetricsCollector(None)
print(collector.get_current_metrics())
"
```

**Alert System Issues:**
```bash
# Test alert configuration
python -c "
from yinsh_ml.monitoring import AlertManager, create_default_alert_rules
manager = AlertManager(None)
rules = create_default_alert_rules()
print(f'Loaded {len(rules)} alert rules')
"
```

**Performance Issues:**
- Reduce `--update-interval` if CPU usage is high
- Decrease `--max-data-points` to reduce memory usage
- Disable health checks if they're too resource-intensive
- Use console mode instead of Streamlit for lower overhead

### Diagnostic Commands

**System Health Check:**
```bash
# Quick health assessment
python scripts/run_memory_dashboard.py --mode export \
    --format json \
    --duration 30 \
    --output health_check.json

# View health details
python -c "
import json
with open('health_check.json') as f:
    data = json.load(f)
print(json.dumps(data.get('health_status', {}), indent=2))
"
```

**Performance Analysis:**
```bash
# Collect detailed metrics
python scripts/run_memory_dashboard.py --mode export \
    --format csv \
    --duration 300 \
    --output performance_analysis.csv

# Analyze with external tools
python -c "
import pandas as pd
df = pd.read_csv('performance_analysis.csv')
print(df.describe())
"
```

## Best Practices

### Monitoring Strategy

1. **Start with Console Mode** for initial deployment verification
2. **Enable Alerts Early** to catch issues before they impact performance
3. **Use Health Checks** to maintain proactive monitoring
4. **Export Data Regularly** for historical analysis and capacity planning
5. **Monitor Trends** rather than just current values

### Performance Optimization

1. **Tune Update Intervals** based on system load and monitoring requirements
2. **Limit Data History** to prevent excessive memory usage
3. **Use Targeted Monitoring** - disable unnecessary features in production
4. **Implement Log Rotation** for dashboard logs and exported data
5. **Monitor the Monitor** - ensure dashboard doesn't impact application performance

### Alert Management

1. **Start with Default Rules** and customize based on observed behavior
2. **Avoid Alert Fatigue** by tuning thresholds appropriately
3. **Test Alert Channels** regularly to ensure notifications work
4. **Document Alert Responses** for operational procedures
5. **Review and Update** alert rules based on system evolution

## Support and Maintenance

### Updating the Dashboard

```bash
# Update monitoring components
git pull origin main
pip install -r requirements.txt

# Restart services
sudo systemctl restart yinsh-dashboard
```

### Data Retention

```bash
# Clean old export files
find ./exports -name "memory_metrics_*.json" -mtime +7 -delete

# Rotate dashboard logs
logrotate /etc/logrotate.d/yinsh-dashboard
```

### Monitoring Dashboard Health

```bash
# Monitor dashboard process
ps aux | grep run_memory_dashboard

# Check resource usage
top -p $(pgrep -f run_memory_dashboard)

# Verify metrics endpoint
curl -s http://localhost:8000/metrics | head -20
```

For additional support, refer to the main [Memory Management Guide](memory_management_guide.md) and project documentation. 