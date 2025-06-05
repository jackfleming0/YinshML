#!/usr/bin/env python3
"""Integration adapter for seamless switching between traditional and tiered storage."""

import time
import logging
import threading
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..experience_buffer import (
    MemoryMappedExperienceBuffer,
    ExperienceRecord,
    BufferConfig,
    BufferStatistics
)
from .tiered_storage import (
    TieredExperienceBuffer,
    TieredStorageConfig,
    ExperienceTier
)
from .migration_tools import (
    migrate_experience_buffer,
    MigrationConfig,
    MigrationProgress
)


class StorageBackend(Enum):
    """Available storage backends."""
    TRADITIONAL = "traditional"
    TIERED = "tiered"
    HYBRID = "hybrid"  # Gradually migrate from traditional to tiered


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    decompression_time_ms: float = 0.0
    batch_formation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    samples_per_second: float = 0.0
    memory_usage_bytes: int = 0
    compression_ratio: float = 1.0
    tier_access_counts: Dict[ExperienceTier, int] = None
    
    def __post_init__(self):
        if self.tier_access_counts is None:
            self.tier_access_counts = {tier: 0 for tier in ExperienceTier}


@dataclass
class AdapterConfig:
    """Configuration for the storage adapter."""
    backend: StorageBackend = StorageBackend.TRADITIONAL
    performance_monitoring_enabled: bool = True
    auto_migration_enabled: bool = False
    migration_trigger_threshold: float = 0.8  # Memory usage threshold to trigger migration
    prefetch_enabled: bool = True
    prefetch_buffer_size: int = 1000
    cache_enabled: bool = True
    cache_size: int = 10000
    fallback_enabled: bool = True  # Fall back to traditional buffer on tiered errors


class ExperienceBufferAdapter:
    """
    Adapter that provides a unified interface for both traditional and tiered storage.
    
    This adapter allows seamless switching between storage backends and provides
    performance monitoring, automatic migration, and fallback capabilities.
    """
    
    def __init__(
        self,
        config: AdapterConfig,
        traditional_buffer_path: Optional[str] = None,
        tiered_config: Optional[TieredStorageConfig] = None,
        migration_config: Optional[MigrationConfig] = None
    ):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ExperienceBufferAdapter")
        
        # Performance monitoring
        self.metrics = PerformanceMetrics()
        self._metrics_lock = threading.RLock()
        
        # Storage backends
        self.traditional_buffer: Optional[MemoryMappedExperienceBuffer] = None
        self.tiered_buffer: Optional[TieredExperienceBuffer] = None
        self.active_backend = config.backend
        
        # Migration settings
        self.migration_config = migration_config or MigrationConfig()
        self.migration_in_progress = False
        self._migration_lock = threading.RLock()
        
        # Initialize backends based on configuration
        self._initialize_backends(traditional_buffer_path, tiered_config)
        
        # Performance cache
        self._experience_cache: Dict[str, ExperienceRecord] = {}
        self._cache_access_times: Dict[str, float] = {}
        self._cache_lock = threading.RLock()
        
        self.logger.info(f"ExperienceBufferAdapter initialized with {self.active_backend.value} backend")
    
    def _initialize_backends(
        self, 
        traditional_buffer_path: Optional[str],
        tiered_config: Optional[TieredStorageConfig]
    ) -> None:
        """Initialize the storage backends based on configuration."""
        
        if self.config.backend in [StorageBackend.TRADITIONAL, StorageBackend.HYBRID]:
            if traditional_buffer_path:
                self.traditional_buffer = MemoryMappedExperienceBuffer(
                    file_path=traditional_buffer_path,
                    create_if_missing=True
                )
                self.logger.info(f"Traditional buffer initialized: {traditional_buffer_path}")
        
        if self.config.backend in [StorageBackend.TIERED, StorageBackend.HYBRID]:
            if tiered_config:
                self.tiered_buffer = TieredExperienceBuffer(tiered_config)
                self.logger.info(f"Tiered buffer initialized: {tiered_config.directory}")
    
    def add_experience(self, experience: ExperienceRecord, experience_id: Optional[str] = None) -> str:
        """Add experience to the active storage backend."""
        start_time = time.time()
        
        try:
            # Determine target backend
            backend = self._get_write_backend()
            
            if backend == StorageBackend.TRADITIONAL and self.traditional_buffer:
                result_id = self.traditional_buffer.add_experience(experience, experience_id)
            elif backend == StorageBackend.TIERED and self.tiered_buffer:
                result_id = self.tiered_buffer.add_experience(experience, experience_id)
            else:
                raise RuntimeError(f"No available backend for writing (backend: {backend})")
            
            # Update cache if enabled
            if self.config.cache_enabled and result_id:
                self._update_cache(result_id, experience)
            
            # Check for auto-migration trigger
            if self.config.auto_migration_enabled:
                self._check_migration_trigger()
            
            return result_id
            
        except Exception as e:
            # Fallback handling
            if self.config.fallback_enabled and self.traditional_buffer:
                self.logger.warning(f"Falling back to traditional buffer: {e}")
                return self.traditional_buffer.add_experience(experience, experience_id)
            raise
    
    def get_experience(self, index: int) -> Optional[ExperienceRecord]:
        """Get experience by index from the active backend."""
        start_time = time.time()
        
        try:
            backend = self._get_read_backend()
            
            if backend == StorageBackend.TRADITIONAL and self.traditional_buffer:
                result = self.traditional_buffer.get_experience(index)
            elif backend == StorageBackend.TIERED and self.tiered_buffer:
                result = self.tiered_buffer.get_experience(index)
            else:
                result = None
            
            # Update performance metrics
            if self.config.performance_monitoring_enabled:
                elapsed_ms = (time.time() - start_time) * 1000
                self._update_access_metrics(elapsed_ms)
            
            return result
            
        except Exception as e:
            # Fallback handling
            if self.config.fallback_enabled and self.traditional_buffer:
                self.logger.warning(f"Falling back to traditional buffer: {e}")
                return self.traditional_buffer.get_experience(index)
            raise
    
    def get_experience_by_id(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Get experience by ID from the active backend."""
        start_time = time.time()
        
        # Check cache first
        if self.config.cache_enabled:
            cached_experience = self._get_from_cache(experience_id)
            if cached_experience:
                self._update_cache_metrics(True)
                return cached_experience
            self._update_cache_metrics(False)
        
        try:
            backend = self._get_read_backend()
            
            if backend == StorageBackend.TRADITIONAL and self.traditional_buffer:
                result = self.traditional_buffer.get_experience_by_id(experience_id)
            elif backend == StorageBackend.TIERED and self.tiered_buffer:
                result = self.tiered_buffer.get_experience_by_id(experience_id)
            else:
                result = None
            
            # Update cache
            if result and self.config.cache_enabled:
                self._update_cache(experience_id, result)
            
            # Update performance metrics
            if self.config.performance_monitoring_enabled:
                elapsed_ms = (time.time() - start_time) * 1000
                self._update_access_metrics(elapsed_ms)
            
            return result
            
        except Exception as e:
            # Fallback handling
            if self.config.fallback_enabled and self.traditional_buffer:
                self.logger.warning(f"Falling back to traditional buffer: {e}")
                return self.traditional_buffer.get_experience_by_id(experience_id)
            raise
    
    def sample_batch(
        self, 
        batch_size: int, 
        phase_weights: Optional[Dict[str, float]] = None
    ) -> List[ExperienceRecord]:
        """Sample a batch of experiences from the active backend."""
        start_time = time.time()
        
        try:
            backend = self._get_read_backend()
            
            if backend == StorageBackend.TRADITIONAL and self.traditional_buffer:
                result = self.traditional_buffer.sample_batch(batch_size, phase_weights)
            elif backend == StorageBackend.TIERED and self.tiered_buffer:
                result = self.tiered_buffer.sample_batch(batch_size, phase_weights)
                # Track tier access for tiered storage
                self._update_tier_access_metrics(result)
            else:
                result = []
            
            # Update performance metrics
            if self.config.performance_monitoring_enabled:
                elapsed_ms = (time.time() - start_time) * 1000
                samples_per_second = len(result) / max(elapsed_ms / 1000, 1e-6)
                self._update_batch_metrics(elapsed_ms, samples_per_second)
            
            return result
            
        except Exception as e:
            # Fallback handling
            if self.config.fallback_enabled and self.traditional_buffer:
                self.logger.warning(f"Falling back to traditional buffer: {e}")
                return self.traditional_buffer.sample_batch(batch_size, phase_weights)
            raise
    
    def get_buffer_statistics(self) -> BufferStatistics:
        """Get comprehensive buffer statistics from the active backend."""
        backend = self._get_read_backend()
        
        if backend == StorageBackend.TRADITIONAL and self.traditional_buffer:
            return self.traditional_buffer.get_buffer_statistics()
        elif backend == StorageBackend.TIERED and self.tiered_buffer:
            return self.tiered_buffer.get_buffer_statistics()
        else:
            # Return empty statistics
            return BufferStatistics(
                capacity=0, count=0, utilization=0.0, write_position=0,
                read_position=0, is_full=False, memory_usage_bytes=0,
                memory_budget_bytes=0, memory_budget_utilization=0.0,
                experience_size_bytes=0, phase_distribution={},
                oldest_timestamp=0.0, newest_timestamp=0.0,
                hit_rate=0.0, miss_rate=0.0, eviction_count=0,
                lru_evictions=0, total_additions=0, total_accesses=0
            )
    
    def migrate_to_tiered_storage(
        self, 
        tiered_config: TieredStorageConfig,
        manifest_path: Optional[str] = None
    ) -> MigrationProgress:
        """
        Migrate from traditional to tiered storage.
        
        Args:
            tiered_config: Configuration for the target tiered storage
            manifest_path: Optional path to save migration manifest
            
        Returns:
            MigrationProgress with detailed migration statistics
        """
        with self._migration_lock:
            if self.migration_in_progress:
                raise RuntimeError("Migration already in progress")
            
            if not self.traditional_buffer:
                raise RuntimeError("No traditional buffer to migrate from")
            
            self.migration_in_progress = True
            self.logger.info("Starting migration to tiered storage")
            
            try:
                # Perform migration
                progress = migrate_experience_buffer(
                    source_buffer_path=self.traditional_buffer.file_path,
                    target_directory=tiered_config.directory,
                    migration_config=self.migration_config,
                    tiered_config=tiered_config,
                    manifest_path=manifest_path
                )
                
                # Initialize tiered buffer
                self.tiered_buffer = TieredExperienceBuffer(tiered_config)
                
                # Switch to tiered backend
                self.active_backend = StorageBackend.TIERED
                
                # Clear cache to prevent stale data
                self._clear_cache()
                
                self.logger.info("Migration to tiered storage completed successfully")
                return progress
                
            finally:
                self.migration_in_progress = False
    
    def switch_backend(self, backend: StorageBackend) -> None:
        """Switch the active storage backend."""
        if backend == self.active_backend:
            return
        
        if backend == StorageBackend.TRADITIONAL and not self.traditional_buffer:
            raise RuntimeError("Traditional buffer not available")
        
        if backend == StorageBackend.TIERED and not self.tiered_buffer:
            raise RuntimeError("Tiered buffer not available")
        
        self.active_backend = backend
        self._clear_cache()  # Clear cache when switching backends
        self.logger.info(f"Switched to {backend.value} backend")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._metrics_lock:
            return self.metrics
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics to zero."""
        with self._metrics_lock:
            self.metrics = PerformanceMetrics()
    
    def _get_read_backend(self) -> StorageBackend:
        """Determine which backend to use for reading."""
        if self.active_backend == StorageBackend.HYBRID:
            # For hybrid mode, prefer tiered if available, otherwise traditional
            if self.tiered_buffer:
                return StorageBackend.TIERED
            elif self.traditional_buffer:
                return StorageBackend.TRADITIONAL
        
        return self.active_backend
    
    def _get_write_backend(self) -> StorageBackend:
        """Determine which backend to use for writing."""
        if self.active_backend == StorageBackend.HYBRID:
            # For hybrid mode, prefer tiered for new writes
            if self.tiered_buffer:
                return StorageBackend.TIERED
            elif self.traditional_buffer:
                return StorageBackend.TRADITIONAL
        
        return self.active_backend
    
    def _check_migration_trigger(self) -> None:
        """Check if automatic migration should be triggered."""
        if (self.traditional_buffer and 
            not self.tiered_buffer and 
            not self.migration_in_progress):
            
            stats = self.traditional_buffer.get_buffer_statistics()
            if stats.memory_budget_utilization > self.config.migration_trigger_threshold:
                self.logger.info(f"Auto-migration triggered (memory usage: {stats.memory_budget_utilization:.1%})")
                # This would need to be handled asynchronously in a real implementation
                # For now, just log the trigger
    
    def _update_cache(self, experience_id: str, experience: ExperienceRecord) -> None:
        """Update the experience cache."""
        if not self.config.cache_enabled:
            return
        
        with self._cache_lock:
            # Implement LRU eviction if cache is full
            if len(self._experience_cache) >= self.config.cache_size:
                oldest_id = min(self._cache_access_times, key=self._cache_access_times.get)
                del self._experience_cache[oldest_id]
                del self._cache_access_times[oldest_id]
            
            self._experience_cache[experience_id] = experience
            self._cache_access_times[experience_id] = time.time()
    
    def _get_from_cache(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Get experience from cache."""
        if not self.config.cache_enabled:
            return None
        
        with self._cache_lock:
            if experience_id in self._experience_cache:
                self._cache_access_times[experience_id] = time.time()
                return self._experience_cache[experience_id]
        
        return None
    
    def _clear_cache(self) -> None:
        """Clear the experience cache."""
        with self._cache_lock:
            self._experience_cache.clear()
            self._cache_access_times.clear()
    
    def _update_access_metrics(self, elapsed_ms: float) -> None:
        """Update access performance metrics."""
        with self._metrics_lock:
            # Use exponential moving average for smooth metrics
            alpha = 0.1
            self.metrics.decompression_time_ms = (
                alpha * elapsed_ms + (1 - alpha) * self.metrics.decompression_time_ms
            )
    
    def _update_batch_metrics(self, elapsed_ms: float, samples_per_second: float) -> None:
        """Update batch formation metrics."""
        with self._metrics_lock:
            alpha = 0.1
            self.metrics.batch_formation_time_ms = (
                alpha * elapsed_ms + (1 - alpha) * self.metrics.batch_formation_time_ms
            )
            self.metrics.samples_per_second = (
                alpha * samples_per_second + (1 - alpha) * self.metrics.samples_per_second
            )
    
    def _update_cache_metrics(self, hit: bool) -> None:
        """Update cache hit/miss metrics."""
        with self._metrics_lock:
            alpha = 0.01  # Slower update for hit rate
            hit_value = 1.0 if hit else 0.0
            self.metrics.cache_hit_rate = (
                alpha * hit_value + (1 - alpha) * self.metrics.cache_hit_rate
            )
    
    def _update_tier_access_metrics(self, experiences: List[ExperienceRecord]) -> None:
        """Update tier access count metrics for tiered storage."""
        if not self.tiered_buffer:
            return
        
        with self._metrics_lock:
            # This is a simplified version - in practice you'd want to track
            # which tier each experience came from
            self.metrics.tier_access_counts[ExperienceTier.HOT] += len(experiences)
    
    def close(self) -> None:
        """Close all storage backends."""
        if self.traditional_buffer:
            self.traditional_buffer.close()
        
        if self.tiered_buffer:
            self.tiered_buffer.close()
        
        self.logger.info("ExperienceBufferAdapter closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 