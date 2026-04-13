#!/bin/bash
# Start large-scale self-play data collection with monitoring

set -e

# Configuration
CONFIG_FILE="configs/large_scale_selfplay.yaml"
OUTPUT_DIR="large_scale_selfplay_data"
MONITOR_INTERVAL=300  # 5 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to check if a process is running
is_process_running() {
    local pid=$1
    kill -0 "$pid" 2>/dev/null
}

# Function to cleanup on exit
cleanup() {
    print_status "Shutting down..."
    
    if [ ! -z "$SELFPLAY_PID" ] && is_process_running "$SELFPLAY_PID"; then
        print_status "Stopping self-play process (PID: $SELFPLAY_PID)"
        kill -TERM "$SELFPLAY_PID" 2>/dev/null || true
        wait "$SELFPLAY_PID" 2>/dev/null || true
    fi
    
    if [ ! -z "$MONITOR_PID" ] && is_process_running "$MONITOR_PID"; then
        print_status "Stopping monitor process (PID: $MONITOR_PID)"
        kill -TERM "$MONITOR_PID" 2>/dev/null || true
        wait "$MONITOR_PID" 2>/dev/null || true
    fi
    
    print_success "Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check if configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check disk space
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
print_status "Available disk space: $AVAILABLE_SPACE"

# Check if we have enough space (need at least 2GB)
AVAILABLE_GB=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
if [ "$AVAILABLE_GB" -lt 2 ]; then
    print_warning "Low disk space: ${AVAILABLE_GB}GB available (recommended: 2GB+)"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start the self-play process
print_status "Starting large-scale self-play data collection..."
python run_large_scale_selfplay.py --config "$CONFIG_FILE" &
SELFPLAY_PID=$!

# Wait a moment for the process to start
sleep 5

# Check if self-play process is running
if ! is_process_running "$SELFPLAY_PID"; then
    print_error "Self-play process failed to start"
    exit 1
fi

print_success "Self-play process started (PID: $SELFPLAY_PID)"

# Start the monitoring process
print_status "Starting monitoring process..."
python scripts/monitor_large_scale_selfplay.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --check-interval "$MONITOR_INTERVAL" &
MONITOR_PID=$!

# Wait a moment for the monitor to start
sleep 2

# Check if monitor process is running
if ! is_process_running "$MONITOR_PID"; then
    print_warning "Monitor process failed to start"
    MONITOR_PID=""
fi

if [ ! -z "$MONITOR_PID" ]; then
    print_success "Monitor process started (PID: $MONITOR_PID)"
fi

# Print status information
print_status "=== Large-Scale Self-Play Started ==="
print_status "Self-play PID: $SELFPLAY_PID"
if [ ! -z "$MONITOR_PID" ]; then
    print_status "Monitor PID: $MONITOR_PID"
fi
print_status "Output directory: $OUTPUT_DIR"
print_status "Configuration: $CONFIG_FILE"
print_status "Monitor interval: ${MONITOR_INTERVAL}s"
print_status ""
print_status "Press Ctrl+C to stop both processes"

# Monitor the processes
while true; do
    # Check if self-play process is still running
    if ! is_process_running "$SELFPLAY_PID"; then
        print_warning "Self-play process stopped"
        break
    fi
    
    # Check if monitor process is still running (if it was started)
    if [ ! -z "$MONITOR_PID" ] && ! is_process_running "$MONITOR_PID"; then
        print_warning "Monitor process stopped"
        MONITOR_PID=""
    fi
    
    # Sleep for a bit before checking again
    sleep 30
done

# Wait for processes to finish
print_status "Waiting for processes to finish..."
wait "$SELFPLAY_PID" 2>/dev/null || true
if [ ! -z "$MONITOR_PID" ]; then
    wait "$MONITOR_PID" 2>/dev/null || true
fi

print_success "Large-scale self-play completed"
