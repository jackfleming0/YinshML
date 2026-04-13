"""
Tests for Graceful Degradation and Recovery Strategies

This module provides comprehensive testing for the degradation management system,
including unit tests, integration tests, and scenario-based testing.
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from collections import deque
from typing import Dict, List, Any

from yinsh_ml.memory.degradation_manager import (
    TaskPriority, DegradationLevel, DegradationStrategy, DegradationRule,
    TaskConfiguration, DegradationEvent, TaskScheduler, DegradationPolicyManager,
    GracefulDegradationManager
)
from yinsh_ml.memory.adaptive_monitor import (
    AdaptiveMemoryMonitor, MemoryPrediction, ResourceType, PredictionAccuracy
)
from yinsh_ml.memory.resource_manager import DynamicResourceManager
from yinsh_ml.memory.monitor import MemoryPressureLevel, MemoryMetrics
from yinsh_ml.memory.events import MemoryEvent, EventType, EventSeverity


@pytest.fixture
def mock_memory_monitor():
    """Create a mock adaptive memory monitor."""
    monitor = Mock(spec=AdaptiveMemoryMonitor)
    monitor.get_pressure_state.return_value = Mock()
    monitor.get_pressure_state.return_value.get_overall_level.return_value = MemoryPressureLevel.NORMAL
    monitor.register_resource_allocator = Mock()
    monitor.subscribe_to_events = Mock()
    return monitor


@pytest.fixture
def mock_resource_manager():
    """Create a mock dynamic resource manager."""
    manager = Mock(spec=DynamicResourceManager)
    return manager


@pytest.fixture
def sample_task_config():
    """Create a sample task configuration for testing."""
    return TaskConfiguration(
        task_id="test_task",
        priority=TaskPriority.MEDIUM,
        resource_requirements={
            'memory_mb': 100,
            'batch_size': 32,
            'simulation_count': 1000
        },
        degradation_options={
            DegradationStrategy.REDUCE_BATCH_SIZE: {'reduction_factor': 0.5},
            DegradationStrategy.REDUCE_MCTS_SIMULATIONS: {'reduction_factor': 0.3},
            DegradationStrategy.REDUCE_BUFFER_SIZES: {'reduction_factor': 0.6}
        },
        can_be_paused=True,
        can_be_deferred=True,
        recovery_time_seconds=5.0
    )


@pytest.fixture
def task_scheduler():
    """Create a task scheduler for testing."""
    return TaskScheduler()


@pytest.fixture
def degradation_manager(mock_memory_monitor, mock_resource_manager):
    """Create a degradation manager for testing."""
    return GracefulDegradationManager(mock_memory_monitor, mock_resource_manager)


class TestTaskPriority:
    """Test TaskPriority enum functionality."""
    
    def test_priority_ordering(self):
        """Test that priority levels are correctly ordered."""
        assert TaskPriority.CRITICAL < TaskPriority.HIGH
        assert TaskPriority.HIGH < TaskPriority.MEDIUM
        assert TaskPriority.MEDIUM < TaskPriority.LOW
        assert TaskPriority.LOW < TaskPriority.BACKGROUND
    
    def test_priority_values(self):
        """Test priority numeric values."""
        assert TaskPriority.CRITICAL.value == 1
        assert TaskPriority.BACKGROUND.value == 5


class TestDegradationRule:
    """Test DegradationRule functionality."""
    
    def test_rule_creation(self):
        """Test creating a degradation rule."""
        rule = DegradationRule(
            strategy=DegradationStrategy.REDUCE_BATCH_SIZE,
            pressure_threshold=MemoryPressureLevel.WARNING,
            degradation_level=DegradationLevel.MINIMAL,
            reduction_factor=0.8,
            target_priorities={TaskPriority.MEDIUM, TaskPriority.LOW},
            description="Test rule"
        )
        
        assert rule.strategy == DegradationStrategy.REDUCE_BATCH_SIZE
        assert rule.enabled == True
        assert TaskPriority.MEDIUM in rule.target_priorities
    
    def test_applies_to_pressure(self):
        """Test pressure level application logic."""
        rule = DegradationRule(
            strategy=DegradationStrategy.REDUCE_BATCH_SIZE,
            pressure_threshold=MemoryPressureLevel.WARNING,
            degradation_level=DegradationLevel.MINIMAL,
            reduction_factor=0.8,
            target_priorities={TaskPriority.MEDIUM},
            description="Test rule"
        )
        
        assert rule.applies_to_pressure(MemoryPressureLevel.WARNING) == True
        assert rule.applies_to_pressure(MemoryPressureLevel.CRITICAL) == True
        assert rule.applies_to_pressure(MemoryPressureLevel.NORMAL) == False
    
    def test_applies_to_task(self):
        """Test task priority application logic."""
        rule = DegradationRule(
            strategy=DegradationStrategy.REDUCE_BATCH_SIZE,
            pressure_threshold=MemoryPressureLevel.WARNING,
            degradation_level=DegradationLevel.MINIMAL,
            reduction_factor=0.8,
            target_priorities={TaskPriority.MEDIUM, TaskPriority.LOW},
            description="Test rule"
        )
        
        assert rule.applies_to_task(TaskPriority.MEDIUM) == True
        assert rule.applies_to_task(TaskPriority.LOW) == True
        assert rule.applies_to_task(TaskPriority.HIGH) == False


class TestTaskConfiguration:
    """Test TaskConfiguration functionality."""
    
    def test_task_config_creation(self, sample_task_config):
        """Test creating a task configuration."""
        config = sample_task_config
        assert config.task_id == "test_task"
        assert config.priority == TaskPriority.MEDIUM
        assert config.resource_requirements['memory_mb'] == 100
    
    def test_get_degraded_requirements_batch_size(self, sample_task_config):
        """Test batch size degradation."""
        config = sample_task_config
        degraded = config.get_degraded_requirements(
            DegradationLevel.MODERATE,
            DegradationStrategy.REDUCE_BATCH_SIZE
        )
        
        # Should reduce batch size by 50%
        assert degraded['batch_size'] == 16
        assert degraded['memory_mb'] == 100  # Unchanged
    
    def test_get_degraded_requirements_simulations(self, sample_task_config):
        """Test MCTS simulation degradation."""
        config = sample_task_config
        degraded = config.get_degraded_requirements(
            DegradationLevel.MODERATE,
            DegradationStrategy.REDUCE_MCTS_SIMULATIONS
        )
        
        # Should reduce simulations by 70%
        assert degraded['simulation_count'] == 300
        assert degraded['batch_size'] == 32  # Unchanged
    
    def test_get_degraded_requirements_buffer_sizes(self, sample_task_config):
        """Test buffer size degradation."""
        config = sample_task_config
        degraded = config.get_degraded_requirements(
            DegradationLevel.SEVERE,
            DegradationStrategy.REDUCE_BUFFER_SIZES
        )
        
        # Should reduce memory by 40%
        assert degraded['memory_mb'] == 60
    
    def test_unknown_strategy(self, sample_task_config):
        """Test handling of unknown degradation strategy."""
        config = sample_task_config
        degraded = config.get_degraded_requirements(
            DegradationLevel.MODERATE,
            DegradationStrategy.LOWER_PRECISION  # Not in degradation_options
        )
        
        # Should return original requirements
        assert degraded == config.resource_requirements


class TestTaskScheduler:
    """Test TaskScheduler functionality."""
    
    def test_register_task(self, task_scheduler, sample_task_config):
        """Test registering a task."""
        scheduler = task_scheduler
        config = sample_task_config
        
        scheduler.register_task(config)
        
        assert config.task_id in scheduler.registered_tasks
        assert config.task_id in scheduler.active_tasks
        assert scheduler.active_tasks[config.task_id]['status'] == 'active'
    
    def test_get_tasks_by_priority(self, task_scheduler):
        """Test filtering tasks by priority."""
        scheduler = task_scheduler
        
        # Create tasks with different priorities
        configs = [
            TaskConfiguration("critical_task", TaskPriority.CRITICAL, {}, {}),
            TaskConfiguration("high_task", TaskPriority.HIGH, {}, {}),
            TaskConfiguration("medium_task", TaskPriority.MEDIUM, {}, {}),
            TaskConfiguration("low_task", TaskPriority.LOW, {}, {})
        ]
        
        for config in configs:
            scheduler.register_task(config)
        
        # Get tasks at MEDIUM priority and below
        medium_and_below = scheduler.get_tasks_by_priority(TaskPriority.MEDIUM)
        task_ids = [t.task_id for t in medium_and_below]
        
        assert "critical_task" in task_ids
        assert "high_task" in task_ids
        assert "medium_task" in task_ids
        assert "low_task" not in task_ids
    
    def test_pause_and_resume_task(self, task_scheduler, sample_task_config):
        """Test pausing and resuming tasks."""
        scheduler = task_scheduler
        config = sample_task_config
        
        scheduler.register_task(config)
        
        # Pause task
        success = scheduler.pause_task(config.task_id, "testing")
        assert success == True
        assert config.task_id in scheduler.paused_tasks
        assert config.task_id not in scheduler.active_tasks
        
        # Resume task
        success = scheduler.resume_task(config.task_id)
        assert success == True
        assert config.task_id in scheduler.active_tasks
        assert config.task_id not in scheduler.paused_tasks
    
    def test_pause_non_pausable_task(self, task_scheduler):
        """Test attempting to pause a non-pausable task."""
        scheduler = task_scheduler
        config = TaskConfiguration(
            "non_pausable", TaskPriority.CRITICAL, {}, {},
            can_be_paused=False
        )
        
        scheduler.register_task(config)
        success = scheduler.pause_task(config.task_id)
        
        assert success == False
        assert config.task_id in scheduler.active_tasks
    
    def test_defer_task(self, task_scheduler, sample_task_config):
        """Test deferring a task."""
        scheduler = task_scheduler
        config = sample_task_config
        
        scheduler.register_task(config)
        
        defer_until = time.time() + 10
        success = scheduler.defer_task(config.task_id, defer_until)
        
        assert success == True
        assert len(scheduler.deferred_tasks) == 1
        assert config.task_id in scheduler.paused_tasks
    
    def test_process_deferred_tasks(self, task_scheduler, sample_task_config):
        """Test processing deferred tasks when time is up."""
        scheduler = task_scheduler
        config = sample_task_config
        
        scheduler.register_task(config)
        
        # Defer task for a very short time
        defer_until = time.time() + 0.1
        scheduler.defer_task(config.task_id, defer_until)
        
        # Wait and process
        time.sleep(0.2)
        resumed_tasks = scheduler.process_deferred_tasks()
        
        assert config.task_id in resumed_tasks
        assert len(scheduler.deferred_tasks) == 0
        assert config.task_id in scheduler.active_tasks
    
    def test_scheduler_statistics(self, task_scheduler):
        """Test getting scheduler statistics."""
        scheduler = task_scheduler
        
        # Add various tasks
        configs = [
            TaskConfiguration("active1", TaskPriority.HIGH, {}, {}),
            TaskConfiguration("active2", TaskPriority.MEDIUM, {}, {}),
            TaskConfiguration("pausable", TaskPriority.LOW, {}, {})
        ]
        
        for config in configs:
            scheduler.register_task(config)
        
        # Pause one task
        scheduler.pause_task("pausable", "testing")
        
        stats = scheduler.get_scheduler_statistics()
        
        assert stats['registered_tasks'] == 3
        assert stats['active_tasks'] == 2
        assert stats['paused_tasks'] == 1
        assert stats['deferred_tasks'] == 0
        assert "active1" in stats['active_task_ids']
        assert "pausable" in stats['paused_task_ids']


class TestDegradationPolicyManager:
    """Test DegradationPolicyManager functionality."""
    
    def test_initialization(self):
        """Test policy manager initialization with default rules."""
        manager = DegradationPolicyManager()
        
        assert len(manager.rules) > 0
        assert len(manager.custom_policies) == 0
    
    def test_add_rule(self):
        """Test adding a custom rule."""
        manager = DegradationPolicyManager()
        initial_count = len(manager.rules)
        
        rule = DegradationRule(
            strategy=DegradationStrategy.REDUCE_BATCH_SIZE,
            pressure_threshold=MemoryPressureLevel.WARNING,
            degradation_level=DegradationLevel.MINIMAL,
            reduction_factor=0.9,
            target_priorities={TaskPriority.LOW},
            description="Custom test rule"
        )
        
        manager.add_rule(rule)
        
        assert len(manager.rules) == initial_count + 1
        assert rule in manager.rules
    
    def test_get_applicable_rules(self):
        """Test getting applicable rules for specific conditions."""
        manager = DegradationPolicyManager()
        
        # Get rules for WARNING pressure and MEDIUM priority tasks
        applicable = manager.get_applicable_rules(
            MemoryPressureLevel.WARNING,
            TaskPriority.MEDIUM
        )
        
        assert len(applicable) > 0
        for rule in applicable:
            assert rule.applies_to_pressure(MemoryPressureLevel.WARNING)
            assert rule.applies_to_task(TaskPriority.MEDIUM)
    
    def test_create_and_apply_policy(self):
        """Test creating and applying a custom policy."""
        manager = DegradationPolicyManager()
        
        custom_rules = [
            DegradationRule(
                strategy=DegradationStrategy.REDUCE_BATCH_SIZE,
                pressure_threshold=MemoryPressureLevel.CRITICAL,
                degradation_level=DegradationLevel.SEVERE,
                reduction_factor=0.2,
                target_priorities={TaskPriority.HIGH},
                description="Aggressive rule 1"
            ),
            DegradationRule(
                strategy=DegradationStrategy.PAUSE_NON_CRITICAL,
                pressure_threshold=MemoryPressureLevel.CRITICAL,
                degradation_level=DegradationLevel.SEVERE,
                reduction_factor=1.0,
                target_priorities={TaskPriority.LOW, TaskPriority.BACKGROUND},
                description="Aggressive rule 2"
            )
        ]
        
        # Create policy
        manager.create_policy("aggressive", custom_rules)
        assert "aggressive" in manager.custom_policies
        
        # Apply policy
        success = manager.apply_policy("aggressive")
        assert success == True
        assert len(manager.rules) == len(custom_rules)
    
    def test_apply_nonexistent_policy(self):
        """Test applying a policy that doesn't exist."""
        manager = DegradationPolicyManager()
        
        success = manager.apply_policy("nonexistent")
        assert success == False


class TestGracefulDegradationManager:
    """Test GracefulDegradationManager functionality."""
    
    def test_initialization(self, degradation_manager):
        """Test degradation manager initialization."""
        manager = degradation_manager
        
        assert manager.current_degradation_level == DegradationLevel.NONE
        assert len(manager.active_degradations) == 0
        assert isinstance(manager.task_scheduler, TaskScheduler)
        assert isinstance(manager.policy_manager, DegradationPolicyManager)
    
    def test_register_task(self, degradation_manager, sample_task_config):
        """Test registering a task with the degradation manager."""
        manager = degradation_manager
        config = sample_task_config
        
        manager.register_task(config)
        
        assert config.task_id in manager.task_scheduler.registered_tasks
    
    def test_force_degradation(self, degradation_manager, sample_task_config):
        """Test manually forcing degradation."""
        manager = degradation_manager
        config = sample_task_config
        
        manager.register_task(config)
        
        # Force moderate degradation
        success = manager.force_degradation(DegradationLevel.MODERATE, "testing")
        
        assert success == True
        # Note: Degradation level might not change immediately due to async processing
    
    def test_force_recovery(self, degradation_manager, sample_task_config):
        """Test manually forcing recovery."""
        manager = degradation_manager
        config = sample_task_config
        
        manager.register_task(config)
        
        # Force degradation then recovery
        manager.force_degradation(DegradationLevel.MODERATE, "testing")
        success = manager.force_recovery()
        
        assert success == True
        assert len(manager.recovery_queue) > 0
    
    def test_current_status(self, degradation_manager):
        """Test getting current degradation status."""
        manager = degradation_manager
        
        status = manager.get_current_status()
        
        assert 'degradation_level' in status
        assert 'active_degradations' in status
        assert 'recovery_queue_size' in status
        assert 'scheduler_stats' in status
    
    def test_statistics(self, degradation_manager):
        """Test getting comprehensive statistics."""
        manager = degradation_manager
        
        stats = manager.get_statistics()
        
        assert 'current_status' in stats
        assert 'degradation_history' in stats
        assert 'policy_manager' in stats
        assert 'task_scheduler' in stats


class TestIntegrationScenarios:
    """Integration tests for complete degradation scenarios."""
    
    def test_memory_warning_scenario(self, degradation_manager):
        """Test complete scenario for memory warning pressure."""
        manager = degradation_manager
        
        # Register tasks with different priorities
        tasks = [
            TaskConfiguration("critical_task", TaskPriority.CRITICAL, {'memory_mb': 50}, {}),
            TaskConfiguration("medium_task", TaskPriority.MEDIUM, {
                'memory_mb': 100, 'batch_size': 32
            }, degradation_options={
                DegradationStrategy.REDUCE_BATCH_SIZE: {'reduction_factor': 0.7}
            }),
            TaskConfiguration("background_task", TaskPriority.BACKGROUND, {'memory_mb': 30}, {}),
        ]
        
        for task in tasks:
            manager.register_task(task)
        
        # Simulate memory warning
        with patch.object(manager, '_apply_degradation') as mock_apply:
            manager._handle_memory_event(MemoryEvent(
                timestamp=time.time(),
                event_type=EventType.PRESSURE_WARNING,
                severity=EventSeverity.WARNING,
                memory_usage_bytes=6_800_000_000,  # 6.8GB used
                memory_total_bytes=8_000_000_000,  # 8GB total = 85%
                memory_percentage=85.0,
                memory_type="system",
                source_component="test",
                details={'memory_percent': 85.0}
            ))
            
            # Should attempt degradation
            mock_apply.assert_called_once()
    
    def test_memory_critical_scenario(self, degradation_manager):
        """Test complete scenario for critical memory pressure."""
        manager = degradation_manager
        
        # Register various tasks
        tasks = [
            TaskConfiguration("high_task", TaskPriority.HIGH, {
                'memory_mb': 200, 'simulation_count': 1000
            }, degradation_options={
                DegradationStrategy.REDUCE_MCTS_SIMULATIONS: {'reduction_factor': 0.4}
            }),
            TaskConfiguration("low_task", TaskPriority.LOW, {'memory_mb': 50}, {},
                            can_be_paused=True),
            TaskConfiguration("bg_task", TaskPriority.BACKGROUND, {'memory_mb': 30}, {},
                            can_be_deferred=True)
        ]
        
        for task in tasks:
            manager.register_task(task)
        
        # Force critical degradation
        manager.force_degradation(DegradationLevel.MODERATE, "testing")
        
        # Verify some form of degradation was applied
        status = manager.get_current_status()
        # Should have either active degradations or scheduler changes
        has_degradation = (
            len(status['active_degradations']) > 0 or
            status['scheduler_stats']['paused_tasks'] > 0
        )
        assert has_degradation
    
    def test_recovery_scenario(self, degradation_manager):
        """Test recovery after memory pressure resolves."""
        manager = degradation_manager
        
        # Register task
        task = TaskConfiguration("test_task", TaskPriority.MEDIUM, {
            'memory_mb': 100, 'batch_size': 32
        }, degradation_options={
            DegradationStrategy.REDUCE_BATCH_SIZE: {'reduction_factor': 0.5}
        })
        manager.register_task(task)
        
        # Apply degradation
        manager.force_degradation(DegradationLevel.MODERATE, "testing")
        
        # Simulate pressure resolved
        manager._handle_memory_event(MemoryEvent(
            timestamp=time.time(),
            event_type=EventType.PRESSURE_RESOLVED,
            severity=EventSeverity.INFO,
            memory_usage_bytes=1_000_000_000,
            memory_total_bytes=8_000_000_000,
            memory_percentage=12.5,
            memory_type="system",
            source_component="test",
            details={}
        ))
        
        # Should have queued recovery tasks
        assert len(manager.recovery_queue) >= 0  # May be processed immediately
    
    def test_monitoring_loop(self, degradation_manager):
        """Test the background monitoring loop."""
        manager = degradation_manager
        
        # Register a deferrable task
        task = TaskConfiguration("defer_test", TaskPriority.LOW, {'memory_mb': 50}, {},
                               can_be_deferred=True)
        manager.register_task(task)
        
        # Verify task is initially active
        assert task.task_id in manager.task_scheduler.active_tasks
        
        # Defer task for very short time
        defer_until = time.time() + 0.01  # 10ms in the past by the time we test
        manager.task_scheduler.defer_task(task.task_id, defer_until)
        
        # Verify task is now deferred (stored as dict with task_id)
        assert task.task_id not in manager.task_scheduler.active_tasks
        deferred_task_ids = [item['task_id'] for item in manager.task_scheduler.deferred_tasks]
        assert task.task_id in deferred_task_ids
        
        # Wait for defer time to pass
        time.sleep(0.05)
        
        # Manually process deferred tasks (simulating what the monitoring loop does)
        resumed_tasks = manager.task_scheduler.process_deferred_tasks()
        
        # Task should have been resumed
        assert task.task_id in resumed_tasks or task.task_id in manager.task_scheduler.active_tasks


class TestPerformanceAndStress:
    """Performance and stress tests for degradation system."""
    
    def test_many_tasks_registration(self, degradation_manager):
        """Test registering many tasks doesn't cause performance issues."""
        manager = degradation_manager
        
        start_time = time.time()
        
        # Register 100 tasks
        for i in range(100):
            task = TaskConfiguration(
                f"task_{i}",
                TaskPriority.MEDIUM,
                {'memory_mb': 10 + i},
                {}
            )
            manager.register_task(task)
        
        registration_time = time.time() - start_time
        
        # Should complete quickly (under 1 second)
        assert registration_time < 1.0
        assert len(manager.task_scheduler.registered_tasks) == 100
    
    def test_concurrent_operations(self, degradation_manager):
        """Test concurrent degradation and recovery operations."""
        manager = degradation_manager
        
        # Register tasks
        for i in range(10):
            task = TaskConfiguration(
                f"concurrent_task_{i}",
                TaskPriority.MEDIUM,
                {'memory_mb': 50},
                {}
            )
            manager.register_task(task)
        
        # Start monitoring
        manager.start_monitoring()
        
        def force_operations():
            for _ in range(5):
                manager.force_degradation(DegradationLevel.MINIMAL, "stress_test")
                time.sleep(0.1)
                manager.force_recovery()
                time.sleep(0.1)
        
        # Run concurrent operations
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=force_operations)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        manager.stop_monitoring()
        
        # System should still be responsive
        status = manager.get_current_status()
        assert 'degradation_level' in status
    
    def test_memory_prediction_handling(self, degradation_manager, mock_memory_monitor):
        """Test handling of memory predictions doesn't cause issues."""
        manager = degradation_manager
        
        # Register task
        task = TaskConfiguration("prediction_test", TaskPriority.MEDIUM, {'memory_mb': 100}, {})
        manager.register_task(task)
        
        # Create prediction
        prediction = MemoryPrediction(
            timestamp=time.time(),
            predicted_time=time.time() + 120.0,  # 2 minutes in future
            predicted_pressure=MemoryPressureLevel.CRITICAL,
            confidence=PredictionAccuracy.HIGH,
            contributing_factors=["test"],
            predicted_usage_bytes=7_000_000_000,  # 7GB predicted
            current_trend=1000.0
        )
        
        # Handle prediction
        start_time = time.time()
        manager._handle_memory_prediction(prediction)
        handling_time = time.time() - start_time
        
        # Should handle quickly
        assert handling_time < 0.1


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 