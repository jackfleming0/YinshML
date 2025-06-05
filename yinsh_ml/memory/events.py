"""
Memory Event Notification System

This module provides a comprehensive event system for memory monitoring,
including event types, severities, and an observer pattern for subscriptions.
"""

import threading
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Set


class EventSeverity(Enum):
    """Severity levels for memory events."""
    INFO = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


class EventType(Enum):
    """Types of memory events that can occur."""
    # Pressure state changes
    PRESSURE_NORMAL = "pressure_normal"
    PRESSURE_WARNING = "pressure_warning"
    PRESSURE_CRITICAL = "pressure_critical"
    PRESSURE_EMERGENCY = "pressure_emergency"
    PRESSURE_RESOLVED = "pressure_resolved"
    
    # Threshold crossings
    THRESHOLD_CROSSED = "threshold_crossed"
    THRESHOLD_RECOVERED = "threshold_recovered"
    
    # System events
    MONITOR_STARTED = "monitor_started"
    MONITOR_STOPPED = "monitor_stopped"
    COLLECTION_ERROR = "collection_error"
    
    # Memory events
    MEMORY_LOW = "memory_low"
    MEMORY_EXHAUSTED = "memory_exhausted"
    MEMORY_RECOVERED = "memory_recovered"
    
    # GPU events
    GPU_MEMORY_HIGH = "gpu_memory_high"
    GPU_MEMORY_CRITICAL = "gpu_memory_critical"
    GPU_OOM = "gpu_out_of_memory"
    
    # Process events
    PROCESS_MEMORY_HIGH = "process_memory_high"
    PROCESS_MEMORY_SPIKE = "process_memory_spike"
    
    # Custom events
    CUSTOM = "custom"


@dataclass
class MemoryEvent:
    """Represents a memory-related event with all relevant context."""
    
    # Core event information
    timestamp: float
    event_type: EventType
    severity: EventSeverity
    
    # Memory information
    memory_usage_bytes: int
    memory_total_bytes: int
    memory_percentage: float
    
    # Context information
    memory_type: str  # "system", "process", "gpu"
    source_component: str = "memory_monitor"
    
    # Optional details
    threshold_bytes: Optional[int] = None
    previous_value: Optional[int] = None
    device_id: Optional[int] = None  # For GPU events
    
    # Additional context
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and compute derived fields."""
        if self.memory_total_bytes > 0:
            self.memory_percentage = (self.memory_usage_bytes / self.memory_total_bytes) * 100
        else:
            self.memory_percentage = 0.0
            
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp).isoformat(),
            'event_type': self.event_type.value,
            'severity': self.severity.name,
            'memory_usage_bytes': self.memory_usage_bytes,
            'memory_total_bytes': self.memory_total_bytes,
            'memory_percentage': round(self.memory_percentage, 2),
            'memory_type': self.memory_type,
            'source_component': self.source_component,
            'threshold_bytes': self.threshold_bytes,
            'previous_value': self.previous_value,
            'device_id': self.device_id,
            'details': self.details
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEvent':
        """Create event from dictionary."""
        # Convert string event_type back to enum
        event_type = EventType(data['event_type'])
        severity = EventSeverity[data['severity']]
        
        return cls(
            timestamp=data['timestamp'],
            event_type=event_type,
            severity=severity,
            memory_usage_bytes=data['memory_usage_bytes'],
            memory_total_bytes=data['memory_total_bytes'],
            memory_percentage=data['memory_percentage'],
            memory_type=data['memory_type'],
            source_component=data.get('source_component', 'memory_monitor'),
            threshold_bytes=data.get('threshold_bytes'),
            previous_value=data.get('previous_value'),
            device_id=data.get('device_id'),
            details=data.get('details', {})
        )


@dataclass
class EventFilter:
    """Filter configuration for event subscriptions."""
    event_types: Optional[Set[EventType]] = None
    severities: Optional[Set[EventSeverity]] = None
    memory_types: Optional[Set[str]] = None
    min_memory_percentage: Optional[float] = None
    max_memory_percentage: Optional[float] = None
    source_components: Optional[Set[str]] = None
    
    def matches(self, event: MemoryEvent) -> bool:
        """Check if an event matches this filter."""
        if self.event_types and event.event_type not in self.event_types:
            return False
            
        if self.severities and event.severity not in self.severities:
            return False
            
        if self.memory_types and event.memory_type not in self.memory_types:
            return False
            
        if self.min_memory_percentage and event.memory_percentage < self.min_memory_percentage:
            return False
            
        if self.max_memory_percentage and event.memory_percentage > self.max_memory_percentage:
            return False
            
        if self.source_components and event.source_component not in self.source_components:
            return False
            
        return True


class EventSubscription:
    """Represents a subscription to memory events."""
    
    def __init__(self, 
                 callback: Callable[[MemoryEvent], None],
                 event_filter: Optional[EventFilter] = None,
                 subscription_id: Optional[str] = None):
        """Initialize subscription."""
        self.callback = callback
        self.filter = event_filter or EventFilter()
        self.subscription_id = subscription_id or f"sub_{id(self)}"
        self.created_at = time.time()
        self.event_count = 0
        self.last_event_time = 0.0
        self.active = True
        
    def notify(self, event: MemoryEvent) -> bool:
        """Notify subscriber if event matches filter."""
        if not self.active:
            return False
            
        if self.filter.matches(event):
            try:
                self.callback(event)
                self.event_count += 1
                self.last_event_time = event.timestamp
                return True
            except Exception as e:
                # Log error but don't fail other subscribers
                print(f"Error in event callback {self.subscription_id}: {e}")
                return False
        return False
        
    def deactivate(self):
        """Deactivate this subscription."""
        self.active = False


class MemoryEventManager:
    """
    Manages memory event subscriptions and notifications using the Observer pattern.
    
    Provides thread-safe event notification, subscription management, and event history.
    """
    
    def __init__(self, max_history_size: int = 1000):
        """Initialize the event manager."""
        self.max_history_size = max_history_size
        
        # Thread synchronization
        self._lock = threading.RLock()
        
        # Subscriptions management
        self._subscriptions: Dict[str, EventSubscription] = {}
        self._subscription_counter = 0
        
        # Event history
        self._event_history: deque[MemoryEvent] = deque(maxlen=max_history_size)
        
        # Statistics
        self._total_events_published = 0
        self._events_by_type: Dict[EventType, int] = defaultdict(int)
        self._events_by_severity: Dict[EventSeverity, int] = defaultdict(int)
        
        # Rate limiting (prevent spam)
        self._last_event_times: Dict[str, float] = {}
        self._min_interval_seconds = 1.0  # Minimum interval between similar events
        
    def subscribe(self, 
                  callback: Callable[[MemoryEvent], None],
                  event_filter: Optional[EventFilter] = None,
                  subscription_id: Optional[str] = None) -> str:
        """
        Subscribe to memory events.
        
        Args:
            callback: Function to call when matching events occur
            event_filter: Filter to specify which events to receive
            subscription_id: Optional custom ID for the subscription
            
        Returns:
            Subscription ID that can be used to unsubscribe
        """
        with self._lock:
            if subscription_id is None:
                self._subscription_counter += 1
                subscription_id = f"subscription_{self._subscription_counter}"
            
            subscription = EventSubscription(callback, event_filter, subscription_id)
            self._subscriptions[subscription_id] = subscription
            
            return subscription_id
            
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from memory events.
        
        Args:
            subscription_id: ID returned from subscribe()
            
        Returns:
            True if subscription was found and removed, False otherwise
        """
        with self._lock:
            if subscription_id in self._subscriptions:
                self._subscriptions[subscription_id].deactivate()
                del self._subscriptions[subscription_id]
                return True
            return False
            
    def publish(self, event: MemoryEvent) -> int:
        """
        Publish an event to all matching subscribers.
        
        Args:
            event: The memory event to publish
            
        Returns:
            Number of subscribers that were notified
        """
        # Rate limiting check
        event_key = f"{event.event_type.value}_{event.memory_type}"
        current_time = event.timestamp
        
        with self._lock:
            # Check rate limiting
            if event_key in self._last_event_times:
                time_since_last = current_time - self._last_event_times[event_key]
                if time_since_last < self._min_interval_seconds:
                    return 0  # Skip this event due to rate limiting
                    
            self._last_event_times[event_key] = current_time
            
            # Add to history
            self._event_history.append(event)
            
            # Update statistics
            self._total_events_published += 1
            self._events_by_type[event.event_type] += 1
            self._events_by_severity[event.severity] += 1
            
            # Notify subscribers
            notified_count = 0
            for subscription in list(self._subscriptions.values()):
                if subscription.notify(event):
                    notified_count += 1
                    
            return notified_count
            
    def get_event_history(self, 
                         max_events: Optional[int] = None,
                         event_filter: Optional[EventFilter] = None) -> List[MemoryEvent]:
        """
        Get historical events, optionally filtered.
        
        Args:
            max_events: Maximum number of events to return
            event_filter: Filter to apply to events
            
        Returns:
            List of events matching criteria
        """
        with self._lock:
            events = list(self._event_history)
            
        # Apply filter if provided
        if event_filter:
            events = [event for event in events if event_filter.matches(event)]
            
        # Apply limit if provided
        if max_events:
            events = events[-max_events:]
            
        return events
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get event manager statistics."""
        with self._lock:
            return {
                'total_events_published': self._total_events_published,
                'active_subscriptions': len(self._subscriptions),
                'history_size': len(self._event_history),
                'max_history_size': self.max_history_size,
                'events_by_type': dict(self._events_by_type),
                'events_by_severity': dict(self._events_by_severity),
                'subscription_details': {
                    sub_id: {
                        'event_count': sub.event_count,
                        'last_event_time': sub.last_event_time,
                        'created_at': sub.created_at,
                        'active': sub.active
                    }
                    for sub_id, sub in self._subscriptions.items()
                }
            }
            
    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._event_history.clear()
            
    def set_rate_limit(self, min_interval_seconds: float) -> None:
        """Set the minimum interval between similar events."""
        self._min_interval_seconds = max(0.0, min_interval_seconds)
        
    def export_events_json(self, filepath: str, 
                          event_filter: Optional[EventFilter] = None) -> int:
        """
        Export events to JSON file.
        
        Args:
            filepath: Path to output file
            event_filter: Optional filter for events to export
            
        Returns:
            Number of events exported
        """
        import json
        
        events = self.get_event_history(event_filter=event_filter)
        event_dicts = [event.to_dict() for event in events]
        
        with open(filepath, 'w') as f:
            json.dump(event_dicts, f, indent=2)
            
        return len(event_dicts)
        
    def export_events_csv(self, filepath: str,
                         event_filter: Optional[EventFilter] = None) -> int:
        """
        Export events to CSV file.
        
        Args:
            filepath: Path to output file
            event_filter: Optional filter for events to export
            
        Returns:
            Number of events exported
        """
        import csv
        
        events = self.get_event_history(event_filter=event_filter)
        if not events:
            return 0
            
        # Get all field names from events
        all_fields = set()
        event_dicts = []
        for event in events:
            event_dict = event.to_dict()
            all_fields.update(event_dict.keys())
            event_dicts.append(event_dict)
            
        # Write CSV
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
            writer.writeheader()
            writer.writerows(event_dicts)
            
        return len(event_dicts) 