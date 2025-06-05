"""
Enhanced Memory Logging System

This module provides structured logging capabilities for memory monitoring,
including JSON logging, CSV export, and integration with monitoring systems.
"""

import json
import logging
import logging.handlers
import os
import csv
import gzip
import threading
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TextIO

from .events import MemoryEvent, EventSeverity, EventType


@dataclass
class LogEntry:
    """Structured log entry for memory monitoring."""
    timestamp: float
    level: str
    message: str
    component: str
    memory_usage_bytes: Optional[int] = None
    memory_percentage: Optional[float] = None
    event_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'level': self.level,
            'message': self.message,
            'component': self.component,
            'memory_usage_bytes': self.memory_usage_bytes,
            'memory_percentage': self.memory_percentage,
            'event_type': self.event_type,
            'details': self.details or {}
        }


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': record.created,
            'datetime': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add custom fields if present
        for attr in ['memory_usage_bytes', 'memory_percentage', 'event_type', 'component']:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)
                
        return json.dumps(log_data)


class MemoryLogRotator:
    """Handles log rotation based on size and time."""
    
    def __init__(self, 
                 base_path: str,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 compress: bool = True):
        """Initialize log rotator."""
        self.base_path = Path(base_path)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.compress = compress
        self._lock = threading.Lock()
        
    def should_rotate(self, filepath: str) -> bool:
        """Check if log file should be rotated."""
        if not os.path.exists(filepath):
            return False
        return os.path.getsize(filepath) >= self.max_bytes
        
    def rotate_file(self, filepath: str) -> None:
        """Rotate log file."""
        with self._lock:
            if not self.should_rotate(filepath):
                return
                
            base = Path(filepath)
            
            # Remove oldest backup if exists
            oldest = base.with_suffix(f'{base.suffix}.{self.backup_count}')
            if self.compress:
                oldest = oldest.with_suffix(oldest.suffix + '.gz')
            if oldest.exists():
                oldest.unlink()
                
            # Rotate existing backups
            for i in range(self.backup_count - 1, 0, -1):
                current = base.with_suffix(f'{base.suffix}.{i}')
                next_num = i + 1
                next_file = base.with_suffix(f'{base.suffix}.{next_num}')
                
                if self.compress:
                    current = current.with_suffix(current.suffix + '.gz')
                    next_file = next_file.with_suffix(next_file.suffix + '.gz')
                    
                if current.exists():
                    current.rename(next_file)
                    
            # Rotate current file to .1
            if base.exists():
                rotated = base.with_suffix(f'{base.suffix}.1')
                if self.compress:
                    # Compress the rotated file
                    with open(base, 'rb') as f_in:
                        with gzip.open(rotated.with_suffix(rotated.suffix + '.gz'), 'wb') as f_out:
                            f_out.writelines(f_in)
                    base.unlink()
                else:
                    base.rename(rotated)


class StructuredMemoryLogger:
    """
    Enhanced memory logger with structured logging, multiple formats,
    and automatic rotation.
    """
    
    def __init__(self,
                 log_dir: str = "./logs",
                 log_level: int = logging.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True,
                 enable_csv: bool = False,
                 max_log_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 compress_backups: bool = True):
        """
        Initialize the structured memory logger.
        
        Args:
            log_dir: Directory for log files
            log_level: Minimum log level
            enable_console: Enable console output
            enable_file: Enable text file logging
            enable_json: Enable JSON file logging
            enable_csv: Enable CSV logging
            max_log_size: Maximum size before rotation
            backup_count: Number of backup files to keep
            compress_backups: Whether to compress rotated logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.log_level = log_level
        self.enable_csv = enable_csv
        
        # Initialize logger
        self.logger = logging.getLogger("yinsh_ml.memory")
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()  # Remove existing handlers
        
        # Timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up console handler
        if enable_console:
            self._setup_console_handler()
            
        # Set up file handler
        if enable_file:
            log_file = self.log_dir / f"memory_{timestamp}.log"
            self._setup_file_handler(str(log_file), max_log_size, backup_count)
            
        # Set up JSON handler
        if enable_json:
            json_file = self.log_dir / f"memory_{timestamp}.json"
            self._setup_json_handler(str(json_file), max_log_size, backup_count)
            
        # CSV logging setup
        if enable_csv:
            self.csv_file = self.log_dir / f"memory_{timestamp}.csv"
            self.csv_entries: deque[LogEntry] = deque(maxlen=10000)
            self.csv_lock = threading.Lock()
            self._init_csv_file()
            
        # Log rotator
        self.rotator = MemoryLogRotator(
            str(self.log_dir),
            max_log_size,
            backup_count,
            compress_backups
        )
        
        self.logger.info("Structured memory logger initialized")
        
    def _setup_console_handler(self) -> None:
        """Set up console logging handler."""
        handler = logging.StreamHandler()
        handler.setLevel(self.log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def _setup_file_handler(self, filepath: str, max_size: int, backup_count: int) -> None:
        """Set up rotating file handler."""
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=max_size,
            backupCount=backup_count
        )
        handler.setLevel(self.log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def _setup_json_handler(self, filepath: str, max_size: int, backup_count: int) -> None:
        """Set up JSON logging handler."""
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=max_size,
            backupCount=backup_count
        )
        handler.setLevel(self.log_level)
        handler.setFormatter(JSONFormatter())
        self.logger.addHandler(handler)
        
    def _init_csv_file(self) -> None:
        """Initialize CSV file with headers."""
        if not hasattr(self, 'csv_file'):
            return
            
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'datetime', 'level', 'message', 'component',
                'memory_usage_bytes', 'memory_percentage', 'event_type', 'details'
            ])
            
    def log_event(self, event: MemoryEvent) -> None:
        """Log a memory event with appropriate level and context."""
        # Determine log level based on event severity
        level_map = {
            EventSeverity.INFO: logging.INFO,
            EventSeverity.WARNING: logging.WARNING,
            EventSeverity.CRITICAL: logging.CRITICAL,
            EventSeverity.EMERGENCY: logging.CRITICAL
        }
        
        log_level = level_map.get(event.severity, logging.INFO)
        
        # Create message
        message = self._format_event_message(event)
        
        # Create log record with custom attributes
        extra = {
            'memory_usage_bytes': event.memory_usage_bytes,
            'memory_percentage': event.memory_percentage,
            'event_type': event.event_type.value,
            'component': event.source_component
        }
        
        # Log the event
        self.logger.log(log_level, message, extra=extra)
        
        # Add to CSV if enabled
        if self.enable_csv:
            self._log_to_csv(event, message, log_level)
            
    def _format_event_message(self, event: MemoryEvent) -> str:
        """Format a memory event into a human-readable message."""
        memory_mb = event.memory_usage_bytes / (1024 * 1024)
        
        base_msg = (f"{event.event_type.value.replace('_', ' ').title()}: "
                   f"{event.memory_type} memory at {memory_mb:.1f}MB "
                   f"({event.memory_percentage:.1f}%)")
        
        # Add threshold information if available
        if event.threshold_bytes:
            threshold_mb = event.threshold_bytes / (1024 * 1024)
            base_msg += f" [Threshold: {threshold_mb:.1f}MB]"
            
        # Add device information for GPU events
        if event.device_id is not None:
            base_msg += f" [GPU {event.device_id}]"
            
        # Add previous value if it's a change event
        if event.previous_value:
            prev_mb = event.previous_value / (1024 * 1024)
            change_mb = memory_mb - prev_mb
            base_msg += f" [Change: {change_mb:+.1f}MB]"
            
        return base_msg
        
    def _log_to_csv(self, event: MemoryEvent, message: str, log_level: int) -> None:
        """Add log entry to CSV queue."""
        entry = LogEntry(
            timestamp=event.timestamp,
            level=logging.getLevelName(log_level),
            message=message,
            component=event.source_component,
            memory_usage_bytes=event.memory_usage_bytes,
            memory_percentage=event.memory_percentage,
            event_type=event.event_type.value,
            details=event.details
        )
        
        with self.csv_lock:
            self.csv_entries.append(entry)
            
            # Flush to file periodically
            if len(self.csv_entries) % 100 == 0:
                self._flush_csv()
                
    def _flush_csv(self) -> None:
        """Flush CSV entries to file."""
        if not hasattr(self, 'csv_file') or not self.csv_entries:
            return
            
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for entry in self.csv_entries:
                writer.writerow([
                    entry.timestamp,
                    datetime.fromtimestamp(entry.timestamp).isoformat(),
                    entry.level,
                    entry.message,
                    entry.component,
                    entry.memory_usage_bytes,
                    entry.memory_percentage,
                    entry.event_type,
                    json.dumps(entry.details) if entry.details else ""
                ])
        
        self.csv_entries.clear()
        
    def log_memory_summary(self,
                          system_usage: int,
                          system_total: int,
                          process_usage: int,
                          gpu_usage: Optional[Dict[int, int]] = None) -> None:
        """Log a periodic memory usage summary."""
        system_pct = (system_usage / system_total) * 100 if system_total > 0 else 0
        
        message = (f"Memory Summary - System: {system_pct:.1f}% "
                  f"({system_usage // (1024*1024):.0f}MB/"
                  f"{system_total // (1024*1024):.0f}MB), "
                  f"Process: {process_usage // (1024*1024):.0f}MB")
        
        if gpu_usage:
            gpu_parts = []
            for device_id, usage in gpu_usage.items():
                gpu_parts.append(f"GPU{device_id}: {usage // (1024*1024):.0f}MB")
            if gpu_parts:
                message += f", {', '.join(gpu_parts)}"
                
        extra = {
            'memory_usage_bytes': system_usage,
            'memory_percentage': system_pct,
            'component': 'summary'
        }
        
        self.logger.info(message, extra=extra)
        
    def log_pressure_change(self,
                           memory_type: str,
                           old_level: str,
                           new_level: str,
                           current_usage: float) -> None:
        """Log a memory pressure level change."""
        message = (f"Memory pressure transition: {memory_type} "
                  f"{old_level} -> {new_level} ({current_usage:.1f}%)")
        
        # Determine log level based on new pressure level
        if new_level in ['CRITICAL', 'EMERGENCY']:
            log_level = logging.CRITICAL
        elif new_level == 'WARNING':
            log_level = logging.WARNING
        else:
            log_level = logging.INFO
            
        extra = {
            'memory_percentage': current_usage,
            'event_type': 'pressure_change',
            'component': 'pressure_monitor'
        }
        
        self.logger.log(log_level, message, extra=extra)
        
    def log_collection_error(self, error: Exception, component: str = "monitor") -> None:
        """Log a memory collection error."""
        message = f"Memory collection error in {component}: {str(error)}"
        
        extra = {
            'event_type': 'collection_error',
            'component': component
        }
        
        self.logger.error(message, extra=extra)
        
    def export_logs_json(self,
                        output_file: str,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> int:
        """
        Export log entries to JSON file.
        
        Args:
            output_file: Path to output file
            start_time: Start time filter (optional)
            end_time: End time filter (optional)
            
        Returns:
            Number of entries exported
        """
        if not hasattr(self, 'csv_entries'):
            return 0
            
        entries = []
        with self.csv_lock:
            for entry in self.csv_entries:
                entry_time = datetime.fromtimestamp(entry.timestamp)
                
                # Apply time filters
                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue
                    
                entries.append(entry.to_dict())
                
        with open(output_file, 'w') as f:
            json.dump(entries, f, indent=2)
            
        return len(entries)
        
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'log_handlers': len(self.logger.handlers),
            'log_level': logging.getLevelName(self.log_level),
            'csv_enabled': self.enable_csv,
            'log_directory': str(self.log_dir)
        }
        
        if hasattr(self, 'csv_entries'):
            stats['csv_entries_pending'] = len(self.csv_entries)
            
        # Get log file sizes
        log_files = list(self.log_dir.glob("memory_*.log*"))
        json_files = list(self.log_dir.glob("memory_*.json*"))
        csv_files = list(self.log_dir.glob("memory_*.csv*"))
        
        stats['file_counts'] = {
            'log_files': len(log_files),
            'json_files': len(json_files),
            'csv_files': len(csv_files)
        }
        
        # Calculate total disk usage
        total_size = 0
        for file_path in log_files + json_files + csv_files:
            if file_path.exists():
                total_size += file_path.stat().st_size
                
        stats['total_disk_usage_bytes'] = total_size
        stats['total_disk_usage_mb'] = total_size / (1024 * 1024)
        
        return stats
        
    def flush_all(self) -> None:
        """Flush all pending log entries."""
        # Flush CSV entries
        if self.enable_csv:
            with self.csv_lock:
                self._flush_csv()
                
        # Flush all handlers
        for handler in self.logger.handlers:
            handler.flush()
            
    def close(self) -> None:
        """Close the logger and all handlers."""
        self.flush_all()
        
        # Close all handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler) 