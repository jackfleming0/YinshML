"""
Graceful Degradation and Recovery Strategies

This module implements intelligent degradation mechanisms for memory pressure situations,
including priority-based task scheduling, resource reduction strategies, and automatic
recovery when memory conditions improve.
"""

import time
import threading
import logging
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from enum import Enum, IntEnum
from datetime import datetime, timedelta
import numpy as np

# Import memory management components
from .adaptive_monitor import (
    AdaptiveMemoryMonitor, MemoryPrediction, ResourceType, PredictionAccuracy
)
from .resource_manager import DynamicResourceManager, ResourceAllocation
from .monitor import MemoryPressureLevel, MemoryMetrics
from .events import MemoryEvent, EventType, EventSeverity

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Priority levels for tasks during degradation."""
    CRITICAL = 1     # Must continue running (safety, core game logic)
    HIGH = 2         # Important but can be reduced (training, main MCTS)
    MEDIUM = 3       # Useful but deferrable (exploration, optimization)
    LOW = 4          # Nice to have (debugging, detailed logging)
    BACKGROUND = 5   # Can be completely paused (metrics collection, cleanup)


class DegradationLevel(Enum):
    """Levels of system degradation."""
    NONE = "none"           # Full functionality
    MINIMAL = "minimal"     # Small reductions (10-20%)
    MODERATE = "moderate"   # Significant reductions (30-50%)
    SEVERE = "severe"       # Major limitations (50-70% reduction)
    EMERGENCY = "emergency" # Bare minimum functionality (>70% reduction)


class DegradationStrategy(Enum):
    """Types of degradation strategies."""
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    REDUCE_MCTS_SIMULATIONS = "reduce_mcts_simulations"
    LOWER_PRECISION = "lower_precision"
    USE_SMALLER_MODEL = "use_smaller_model"
    DEFER_BACKGROUND_TASKS = "defer_background_tasks"
    REDUCE_BUFFER_SIZES = "reduce_buffer_sizes"
    SIMPLIFY_CALCULATIONS = "simplify_calculations"
    PAUSE_NON_CRITICAL = "pause_non_critical"


@dataclass
class DegradationRule:
    """Rule for applying degradation based on conditions."""
    strategy: DegradationStrategy
    pressure_threshold: MemoryPressureLevel
    degradation_level: DegradationLevel
    reduction_factor: float  # 0.0 to 1.0, amount to reduce resource usage
    target_priorities: Set[TaskPriority]
    description: str
    enabled: bool = True
    
    def applies_to_pressure(self, pressure: MemoryPressureLevel) -> bool:
        """Check if this rule applies to the given pressure level."""
        return self.enabled and pressure.value >= self.pressure_threshold.value
    
    def applies_to_task(self, priority: TaskPriority) -> bool:
        """Check if this rule applies to a task with given priority."""
        return priority in self.target_priorities


@dataclass
class TaskConfiguration:
    """Configuration for a specific task or component."""
    task_id: str
    priority: TaskPriority
    resource_requirements: Dict[str, Any]  # e.g., {'memory_mb': 100, 'cpu_cores': 2}
    degradation_options: Dict[DegradationStrategy, Any]  # Strategy-specific parameters
    can_be_paused: bool = True
    can_be_deferred: bool = True
    recovery_time_seconds: float = 5.0  # Time needed to restore full functionality
    
    def get_degraded_requirements(self, level: DegradationLevel, 
                                strategy: DegradationStrategy) -> Dict[str, Any]:
        """Get resource requirements after applying degradation."""
        if strategy not in self.degradation_options:
            return self.resource_requirements.copy()
        
        degraded = self.resource_requirements.copy()
        strategy_params = self.degradation_options[strategy]
        
        # Apply strategy-specific degradation
        if strategy == DegradationStrategy.REDUCE_BATCH_SIZE:
            if 'batch_size' in degraded:
                factor = strategy_params.get('reduction_factor', 0.5)
                degraded['batch_size'] = max(1, int(degraded['batch_size'] * factor))
        
        elif strategy == DegradationStrategy.REDUCE_MCTS_SIMULATIONS:
            if 'simulation_count' in degraded:
                factor = strategy_params.get('reduction_factor', 0.3)
                degraded['simulation_count'] = max(1, int(degraded['simulation_count'] * factor))
        
        elif strategy == DegradationStrategy.REDUCE_BUFFER_SIZES:
            for key in ['buffer_size', 'memory_mb', 'cache_size']:
                if key in degraded:
                    factor = strategy_params.get('reduction_factor', 0.6)
                    degraded[key] = max(1, int(degraded[key] * factor))
        
        return degraded


@dataclass
class DegradationEvent:
    """Records a degradation or recovery event."""
    timestamp: float
    event_type: str  # "degradation" or "recovery"
    strategy: DegradationStrategy
    level: DegradationLevel
    affected_tasks: List[str]
    memory_pressure: MemoryPressureLevel
    estimated_memory_saved: int
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'strategy': self.strategy.value,
            'level': self.level.value,
            'affected_tasks': self.affected_tasks,
            'memory_pressure': self.memory_pressure.name,
            'estimated_memory_saved': self.estimated_memory_saved,
            'success': self.success,
            'details': self.details
        }


class TaskScheduler:
    """Priority-based task scheduler for memory pressure situations."""
    
    def __init__(self):
        """Initialize task scheduler."""
        self.registered_tasks: Dict[str, TaskConfiguration] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}  # Current task states
        self.paused_tasks: Dict[str, Dict[str, Any]] = {}  # Paused task states
        self.deferred_tasks: deque = deque()  # Tasks waiting to resume
        self._lock = threading.RLock()
        
    def register_task(self, config: TaskConfiguration) -> None:
        """Register a task with the scheduler."""
        with self._lock:
            self.registered_tasks[config.task_id] = config
            self.active_tasks[config.task_id] = {
                'status': 'active',
                'current_requirements': config.resource_requirements.copy(),
                'last_updated': time.time()
            }
            logger.debug(f"Registered task '{config.task_id}' with priority {config.priority.name}")
    
    def get_tasks_by_priority(self, max_priority: TaskPriority) -> List[TaskConfiguration]:
        """Get all tasks at or below the specified priority level."""
        with self._lock:
            return [
                config for config in self.registered_tasks.values()
                if config.priority.value <= max_priority.value
            ]
    
    def pause_task(self, task_id: str, reason: str = "memory_pressure") -> bool:
        """Pause a task and save its state."""
        with self._lock:
            if task_id not in self.registered_tasks:
                return False
            
            config = self.registered_tasks[task_id]
            if not config.can_be_paused:
                return False
            
            if task_id in self.active_tasks:
                # Move from active to paused
                self.paused_tasks[task_id] = self.active_tasks.pop(task_id)
                self.paused_tasks[task_id]['pause_reason'] = reason
                self.paused_tasks[task_id]['pause_time'] = time.time()
                logger.info(f"Paused task '{task_id}' due to {reason}")
                return True
            
            return False
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused task."""
        with self._lock:
            if task_id not in self.paused_tasks:
                return False
            
            config = self.registered_tasks[task_id]
            
            # Move from paused to active
            task_state = self.paused_tasks.pop(task_id)
            task_state['status'] = 'active'
            task_state['resume_time'] = time.time()
            task_state.pop('pause_reason', None)
            task_state.pop('pause_time', None)
            
            self.active_tasks[task_id] = task_state
            logger.info(f"Resumed task '{task_id}'")
            return True
    
    def defer_task(self, task_id: str, defer_until: float) -> bool:
        """Defer a task until a specific time."""
        with self._lock:
            if task_id not in self.registered_tasks:
                return False
            
            config = self.registered_tasks[task_id]
            if not config.can_be_deferred:
                return False
            
            self.deferred_tasks.append({
                'task_id': task_id,
                'defer_until': defer_until,
                'deferred_at': time.time()
            })
            
            # Pause the task
            self.pause_task(task_id, "deferred")
            logger.info(f"Deferred task '{task_id}' until {defer_until}")
            return True
    
    def process_deferred_tasks(self) -> List[str]:
        """Process deferred tasks that are ready to resume."""
        with self._lock:
            current_time = time.time()
            resumed_tasks = []
            
            # Check deferred tasks
            while self.deferred_tasks:
                task_info = self.deferred_tasks[0]
                if task_info['defer_until'] <= current_time:
                    # Time to resume this task
                    self.deferred_tasks.popleft()
                    task_id = task_info['task_id']
                    
                    if self.resume_task(task_id):
                        resumed_tasks.append(task_id)
                else:
                    # Still too early, stop checking
                    break
            
            return resumed_tasks
    
    def get_scheduler_statistics(self) -> Dict[str, Any]:
        """Get statistics about task scheduling."""
        with self._lock:
            return {
                'registered_tasks': len(self.registered_tasks),
                'active_tasks': len(self.active_tasks),
                'paused_tasks': len(self.paused_tasks),
                'deferred_tasks': len(self.deferred_tasks),
                'task_priorities': {
                    priority.name: len([
                        t for t in self.registered_tasks.values() 
                        if t.priority == priority
                    ])
                    for priority in TaskPriority
                },
                'active_task_ids': list(self.active_tasks.keys()),
                'paused_task_ids': list(self.paused_tasks.keys())
            }


class DegradationPolicyManager:
    """Manages degradation policies and rules."""
    
    def __init__(self):
        """Initialize degradation policy manager."""
        self.rules: List[DegradationRule] = []
        self.custom_policies: Dict[str, List[DegradationRule]] = {}
        self._lock = threading.RLock()
        
        # Load default rules
        self._create_default_rules()
    
    def _create_default_rules(self) -> None:
        """Create default degradation rules."""
        default_rules = [
            # Minimal degradation for WARNING pressure
            DegradationRule(
                strategy=DegradationStrategy.REDUCE_BATCH_SIZE,
                pressure_threshold=MemoryPressureLevel.WARNING,
                degradation_level=DegradationLevel.MINIMAL,
                reduction_factor=0.8,
                target_priorities={TaskPriority.MEDIUM, TaskPriority.LOW, TaskPriority.BACKGROUND},
                description="Reduce batch sizes by 20% for non-critical tasks"
            ),
            
            # Moderate degradation for CRITICAL pressure
            DegradationRule(
                strategy=DegradationStrategy.REDUCE_MCTS_SIMULATIONS,
                pressure_threshold=MemoryPressureLevel.CRITICAL,
                degradation_level=DegradationLevel.MODERATE,
                reduction_factor=0.5,
                target_priorities={TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW},
                description="Reduce MCTS simulations by 50% for non-critical tasks"
            ),
            
            DegradationRule(
                strategy=DegradationStrategy.DEFER_BACKGROUND_TASKS,
                pressure_threshold=MemoryPressureLevel.CRITICAL,
                degradation_level=DegradationLevel.MODERATE,
                reduction_factor=1.0,
                target_priorities={TaskPriority.BACKGROUND},
                description="Defer all background tasks during critical pressure"
            ),
            
            # Severe degradation for EMERGENCY pressure
            DegradationRule(
                strategy=DegradationStrategy.PAUSE_NON_CRITICAL,
                pressure_threshold=MemoryPressureLevel.EMERGENCY,
                degradation_level=DegradationLevel.EMERGENCY,
                reduction_factor=1.0,
                target_priorities={TaskPriority.LOW, TaskPriority.BACKGROUND},
                description="Pause all non-critical tasks during emergency"
            ),
            
            DegradationRule(
                strategy=DegradationStrategy.REDUCE_BUFFER_SIZES,
                pressure_threshold=MemoryPressureLevel.EMERGENCY,
                degradation_level=DegradationLevel.SEVERE,
                reduction_factor=0.3,
                target_priorities={TaskPriority.HIGH, TaskPriority.MEDIUM},
                description="Aggressively reduce buffer sizes during emergency"
            )
        ]
        
        with self._lock:
            self.rules.extend(default_rules)
    
    def add_rule(self, rule: DegradationRule) -> None:
        """Add a custom degradation rule."""
        with self._lock:
            self.rules.append(rule)
            logger.debug(f"Added degradation rule: {rule.description}")
    
    def get_applicable_rules(self, pressure: MemoryPressureLevel, 
                           task_priority: TaskPriority) -> List[DegradationRule]:
        """Get rules that apply to the given conditions."""
        with self._lock:
            return [
                rule for rule in self.rules
                if rule.applies_to_pressure(pressure) and rule.applies_to_task(task_priority)
            ]
    
    def create_policy(self, name: str, rules: List[DegradationRule]) -> None:
        """Create a named policy with specific rules."""
        with self._lock:
            self.custom_policies[name] = rules.copy()
            logger.info(f"Created degradation policy '{name}' with {len(rules)} rules")
    
    def apply_policy(self, policy_name: str) -> bool:
        """Apply a named policy, replacing current rules."""
        with self._lock:
            if policy_name not in self.custom_policies:
                return False
            
            self.rules = self.custom_policies[policy_name].copy()
            logger.info(f"Applied degradation policy '{policy_name}'")
            return True


class GracefulDegradationManager:
    """Main manager for graceful degradation and recovery strategies."""
    
    def __init__(self, 
                 memory_monitor: AdaptiveMemoryMonitor,
                 resource_manager: DynamicResourceManager):
        """Initialize graceful degradation manager.
        
        Args:
            memory_monitor: Adaptive memory monitor for pressure events
            resource_manager: Dynamic resource manager for coordination
        """
        self.memory_monitor = memory_monitor
        self.resource_manager = resource_manager
        
        # Core components
        self.task_scheduler = TaskScheduler()
        self.policy_manager = DegradationPolicyManager()
        
        # State tracking
        self.current_degradation_level = DegradationLevel.NONE
        self.active_degradations: Dict[DegradationStrategy, Dict[str, Any]] = {}
        self.degradation_history: deque = deque(maxlen=100)
        self.recovery_queue: deque = deque()
        
        # Threading
        self._lock = threading.RLock()
        self._recovery_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Register with memory monitor
        self._register_with_monitor()
        
        logger.info("Initialized GracefulDegradationManager")
    
    def _register_with_monitor(self) -> None:
        """Register callbacks with the memory monitor."""
        # Register for memory pressure predictions
        self.memory_monitor.register_resource_allocator(
            ResourceType.SYSTEM_OVERHEAD,
            self,
            self._handle_memory_prediction
        )
        
        # Subscribe to memory pressure events
        self.memory_monitor.subscribe_to_events(
            self._handle_memory_event,
            event_filter=None
        )
    
    def register_task(self, config: TaskConfiguration) -> None:
        """Register a task for degradation management."""
        self.task_scheduler.register_task(config)
        logger.debug(f"Registered task '{config.task_id}' for degradation management")
    
    def _handle_memory_prediction(self, prediction: MemoryPrediction) -> None:
        """Handle memory pressure predictions."""
        try:
            if prediction.confidence in [PredictionAccuracy.HIGH, PredictionAccuracy.MEDIUM]:
                # Proactive degradation based on prediction
                time_to_pressure = prediction.time_until_prediction()
                
                if time_to_pressure < 60:  # Less than 1 minute
                    logger.info(f"Proactive degradation for predicted {prediction.predicted_pressure.name}")
                    self._apply_degradation(prediction.predicted_pressure, proactive=True)
                
        except Exception as e:
            logger.error(f"Error handling memory prediction for degradation: {e}")
    
    def _handle_memory_event(self, event: MemoryEvent) -> None:
        """Handle memory pressure events."""
        try:
            if event.event_type in [EventType.PRESSURE_WARNING, EventType.PRESSURE_CRITICAL, 
                                  EventType.PRESSURE_EMERGENCY]:
                # Reactive degradation
                pressure_mapping = {
                    EventType.PRESSURE_WARNING: MemoryPressureLevel.WARNING,
                    EventType.PRESSURE_CRITICAL: MemoryPressureLevel.CRITICAL,
                    EventType.PRESSURE_EMERGENCY: MemoryPressureLevel.EMERGENCY
                }
                pressure_level = pressure_mapping.get(event.event_type, MemoryPressureLevel.NORMAL)
                
                logger.warning(f"Reactive degradation for {pressure_level.name} pressure")
                self._apply_degradation(pressure_level, proactive=False)
            
            elif event.event_type == EventType.PRESSURE_RESOLVED:
                # Start recovery process
                logger.info("Memory pressure resolved, starting recovery")
                self._start_recovery()
                
        except Exception as e:
            logger.error(f"Error handling memory event for degradation: {e}")
    
    def _apply_degradation(self, pressure_level: MemoryPressureLevel, proactive: bool = False) -> None:
        """Apply degradation strategies based on pressure level."""
        with self._lock:
            # Determine target degradation level
            if pressure_level == MemoryPressureLevel.WARNING:
                target_level = DegradationLevel.MINIMAL
            elif pressure_level == MemoryPressureLevel.CRITICAL:
                target_level = DegradationLevel.MODERATE
            elif pressure_level == MemoryPressureLevel.EMERGENCY:
                target_level = DegradationLevel.SEVERE
            else:
                return
            
            # Don't degrade further if already at higher level
            current_severity = self._get_degradation_severity(self.current_degradation_level)
            target_severity = self._get_degradation_severity(target_level)
            
            if current_severity >= target_severity:
                return
            
            # Apply degradation strategies
            applied_strategies = []
            total_memory_saved = 0
            
            # Get all registered tasks
            for task_config in self.task_scheduler.registered_tasks.values():
                applicable_rules = self.policy_manager.get_applicable_rules(
                    pressure_level, task_config.priority
                )
                
                for rule in applicable_rules:
                    if rule.strategy not in self.active_degradations:
                        success, memory_saved = self._apply_strategy(rule, task_config)
                        
                        if success:
                            applied_strategies.append(rule.strategy)
                            total_memory_saved += memory_saved
                            
                            # Track active degradation
                            self.active_degradations[rule.strategy] = {
                                'rule': rule,
                                'applied_at': time.time(),
                                'affected_tasks': [task_config.task_id],
                                'memory_saved': memory_saved
                            }
            
            if applied_strategies:
                self.current_degradation_level = target_level
                
                # Record degradation event
                event = DegradationEvent(
                    timestamp=time.time(),
                    event_type="degradation",
                    strategy=applied_strategies[0],  # Primary strategy
                    level=target_level,
                    affected_tasks=[t.task_id for t in self.task_scheduler.registered_tasks.values()],
                    memory_pressure=pressure_level,
                    estimated_memory_saved=total_memory_saved,
                    success=True,
                    details={
                        'applied_strategies': [s.value for s in applied_strategies],
                        'proactive': proactive
                    }
                )
                self.degradation_history.append(event)
                
                logger.warning(f"Applied {len(applied_strategies)} degradation strategies, "
                             f"estimated {total_memory_saved:,} bytes saved")
    
    def _apply_strategy(self, rule: DegradationRule, task_config: TaskConfiguration) -> Tuple[bool, int]:
        """Apply a specific degradation strategy to a task."""
        try:
            memory_saved = 0
            
            if rule.strategy == DegradationStrategy.PAUSE_NON_CRITICAL:
                if task_config.priority in rule.target_priorities:
                    success = self.task_scheduler.pause_task(task_config.task_id, "degradation")
                    if success:
                        # Estimate memory saved (rough approximation)
                        memory_saved = task_config.resource_requirements.get('memory_mb', 50) * 1_000_000
                    return success, memory_saved
            
            elif rule.strategy == DegradationStrategy.DEFER_BACKGROUND_TASKS:
                if task_config.priority in rule.target_priorities:
                    defer_until = time.time() + 300  # Defer for 5 minutes
                    success = self.task_scheduler.defer_task(task_config.task_id, defer_until)
                    if success:
                        memory_saved = task_config.resource_requirements.get('memory_mb', 30) * 1_000_000
                    return success, memory_saved
            
            elif rule.strategy in [DegradationStrategy.REDUCE_BATCH_SIZE, 
                                 DegradationStrategy.REDUCE_MCTS_SIMULATIONS,
                                 DegradationStrategy.REDUCE_BUFFER_SIZES]:
                # Get degraded requirements
                degraded_reqs = task_config.get_degraded_requirements(rule.degradation_level, rule.strategy)
                
                # Calculate memory saved
                original_memory = task_config.resource_requirements.get('memory_mb', 0)
                degraded_memory = degraded_reqs.get('memory_mb', 0)
                memory_saved = (original_memory - degraded_memory) * 1_000_000
                
                # Update task requirements (this would need integration with actual task execution)
                task_state = self.task_scheduler.active_tasks.get(task_config.task_id)
                if task_state:
                    task_state['current_requirements'] = degraded_reqs
                    task_state['degradation_applied'] = rule.strategy.value
                    task_state['degradation_time'] = time.time()
                
                return True, max(0, memory_saved)
            
            return False, 0
            
        except Exception as e:
            logger.error(f"Error applying degradation strategy {rule.strategy.value}: {e}")
            return False, 0
    
    def _start_recovery(self) -> None:
        """Start the recovery process."""
        with self._lock:
            if self.current_degradation_level == DegradationLevel.NONE:
                return
            
            # Queue recovery tasks
            recovery_tasks = []
            
            # Resume paused tasks
            for task_id in list(self.task_scheduler.paused_tasks.keys()):
                recovery_tasks.append(('resume', task_id, time.time() + 5))
            
            # Restore degraded settings
            for strategy, degradation_info in self.active_degradations.items():
                recovery_tasks.append(('restore', strategy, time.time() + 10))
            
            # Add to recovery queue
            for task_type, task_info, scheduled_time in recovery_tasks:
                self.recovery_queue.append({
                    'type': task_type,
                    'info': task_info,
                    'scheduled_time': scheduled_time
                })
            
            logger.info(f"Queued {len(recovery_tasks)} recovery tasks")
    
    def _process_recovery_queue(self) -> None:
        """Process queued recovery tasks."""
        with self._lock:
            current_time = time.time()
            completed_recoveries = []
            
            # Process due recovery tasks
            while self.recovery_queue:
                recovery_task = self.recovery_queue[0]
                
                if recovery_task['scheduled_time'] <= current_time:
                    self.recovery_queue.popleft()
                    
                    task_type = recovery_task['type']
                    task_info = recovery_task['info']
                    
                    try:
                        if task_type == 'resume':
                            # Resume a paused task
                            if self.task_scheduler.resume_task(task_info):
                                completed_recoveries.append(f"Resumed task {task_info}")
                        
                        elif task_type == 'restore':
                            # Restore original settings
                            strategy = task_info
                            if strategy in self.active_degradations:
                                self._restore_strategy(strategy)
                                completed_recoveries.append(f"Restored {strategy.value}")
                        
                    except Exception as e:
                        logger.error(f"Error in recovery task {task_type}: {e}")
                else:
                    break
            
            # Check if recovery is complete
            if not self.recovery_queue and self.current_degradation_level != DegradationLevel.NONE:
                self.current_degradation_level = DegradationLevel.NONE
                self.active_degradations.clear()
                
                # Record recovery event
                event = DegradationEvent(
                    timestamp=time.time(),
                    event_type="recovery",
                    strategy=list(self.active_degradations.keys())[0] if self.active_degradations else DegradationStrategy.PAUSE_NON_CRITICAL,
                    level=DegradationLevel.NONE,
                    affected_tasks=[],
                    memory_pressure=self.memory_monitor.get_pressure_state().get_overall_level(),
                    estimated_memory_saved=0,
                    success=True,
                    details={'completed_recoveries': completed_recoveries}
                )
                self.degradation_history.append(event)
                
                logger.info(f"Recovery complete: {len(completed_recoveries)} operations restored")
    
    def _restore_strategy(self, strategy: DegradationStrategy) -> bool:
        """Restore original settings for a degradation strategy."""
        try:
            if strategy not in self.active_degradations:
                return False
            
            degradation_info = self.active_degradations[strategy]
            affected_tasks = degradation_info['affected_tasks']
            
            # Restore original task configurations
            for task_id in affected_tasks:
                if task_id in self.task_scheduler.registered_tasks:
                    config = self.task_scheduler.registered_tasks[task_id]
                    task_state = self.task_scheduler.active_tasks.get(task_id)
                    
                    if task_state:
                        # Restore original requirements
                        task_state['current_requirements'] = config.resource_requirements.copy()
                        task_state.pop('degradation_applied', None)
                        task_state.pop('degradation_time', None)
                        task_state['restoration_time'] = time.time()
            
            # Remove from active degradations
            del self.active_degradations[strategy]
            
            logger.debug(f"Restored strategy {strategy.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring strategy {strategy.value}: {e}")
            return False
    
    def _get_degradation_severity(self, level: DegradationLevel) -> int:
        """Get numeric severity for degradation level comparison."""
        severity_map = {
            DegradationLevel.NONE: 0,
            DegradationLevel.MINIMAL: 1,
            DegradationLevel.MODERATE: 2,
            DegradationLevel.SEVERE: 3,
            DegradationLevel.EMERGENCY: 4
        }
        return severity_map.get(level, 0)
    
    def force_degradation(self, level: DegradationLevel, reason: str = "manual") -> bool:
        """Manually force degradation to a specific level."""
        with self._lock:
            logger.warning(f"Forcing degradation to {level.value} - reason: {reason}")
            
            # Map level to pressure for applying strategies
            pressure_map = {
                DegradationLevel.MINIMAL: MemoryPressureLevel.WARNING,
                DegradationLevel.MODERATE: MemoryPressureLevel.CRITICAL,
                DegradationLevel.SEVERE: MemoryPressureLevel.EMERGENCY,
                DegradationLevel.EMERGENCY: MemoryPressureLevel.EMERGENCY
            }
            
            if level in pressure_map:
                self._apply_degradation(pressure_map[level], proactive=True)
                return True
            
            return False
    
    def force_recovery(self) -> bool:
        """Manually force recovery from current degradation."""
        with self._lock:
            logger.info("Forcing recovery from degradation")
            self._start_recovery()
            return True
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        with self._lock:
            return {
                'degradation_level': self.current_degradation_level.value,
                'active_degradations': {
                    strategy.value: {
                        'applied_at': info['applied_at'],
                        'affected_tasks': info['affected_tasks'],
                        'memory_saved': info['memory_saved']
                    }
                    for strategy, info in self.active_degradations.items()
                },
                'recovery_queue_size': len(self.recovery_queue),
                'scheduler_stats': self.task_scheduler.get_scheduler_statistics()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive degradation statistics."""
        with self._lock:
            # Process recent events
            recent_events = list(self.degradation_history)[-10:]
            
            return {
                'current_status': self.get_current_status(),
                'degradation_history': {
                    'total_events': len(self.degradation_history),
                    'degradation_events': sum(1 for e in self.degradation_history if e.event_type == 'degradation'),
                    'recovery_events': sum(1 for e in self.degradation_history if e.event_type == 'recovery'),
                    'total_memory_saved': sum(e.estimated_memory_saved for e in self.degradation_history),
                    'recent_events': [e.to_dict() for e in recent_events]
                },
                'policy_manager': {
                    'total_rules': len(self.policy_manager.rules),
                    'custom_policies': len(self.policy_manager.custom_policies)
                },
                'task_scheduler': self.task_scheduler.get_scheduler_statistics()
            }
    
    def start_monitoring(self) -> None:
        """Start the degradation monitoring thread."""
        if self._recovery_thread is None or not self._recovery_thread.is_alive():
            self._running = True
            self._recovery_thread = threading.Thread(
                target=self._monitoring_loop,
                name="DegradationRecoveryMonitoring",
                daemon=True
            )
            self._recovery_thread.start()
            logger.info("Started degradation recovery monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the degradation monitoring thread."""
        self._running = False
        logger.info("Stopped degradation recovery monitoring")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop for recovery processing."""
        while self._running:
            try:
                # Process deferred tasks
                resumed_tasks = self.task_scheduler.process_deferred_tasks()
                if resumed_tasks:
                    logger.debug(f"Resumed {len(resumed_tasks)} deferred tasks")
                
                # Process recovery queue
                self._process_recovery_queue()
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in degradation monitoring loop: {e}")
                time.sleep(30) 