"""Tiered storage system for experience compression."""

import os
import time
import threading
import logging
import pickle
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, OrderedDict

from ..experience_buffer import (
    ExperienceRecord, 
    MemoryMappedExperienceBuffer, 
    BufferConfig,
    BufferStatistics
)
from .compression_strategies import CompressionStrategy, CompressionFactory, CompressionResult

logger = logging.getLogger(__name__)


class ExperienceTier(Enum):
    """Experience storage tiers."""
    HOT = "hot"      # Recent experiences, uncompressed, fast access
    WARM = "warm"    # Older experiences, light compression, moderate access
    COLD = "cold"    # Oldest experiences, heavy compression, slow access


@dataclass
class TierThresholds:
    """Thresholds for tier transitions."""
    warm_age_seconds: float = 3600.0    # 1 hour
    cold_age_seconds: float = 86400.0   # 24 hours
    warm_access_threshold: int = 5       # Access count for staying warm
    cold_access_threshold: int = 1       # Access count for staying cold
    max_hot_experiences: int = 10000     # Maximum experiences in hot tier
    max_warm_experiences: int = 50000    # Maximum experiences in warm tier


@dataclass 
class TieredStorageConfig(BufferConfig):
    """Configuration for tiered storage system."""
    
    # Tier thresholds
    tier_thresholds: TierThresholds = field(default_factory=TierThresholds)
    
    # Compression settings
    warm_compression_strategy: str = "lz4"
    warm_compression_kwargs: Dict[str, Any] = field(default_factory=lambda: {'compression_level': 1})
    cold_compression_strategy: str = "lzma" 
    cold_compression_kwargs: Dict[str, Any] = field(default_factory=lambda: {'preset': 6})
    
    # Storage paths
    warm_storage_dir: str = "./storage/warm"
    cold_storage_dir: str = "./storage/cold"
    metadata_storage_dir: str = "./storage/metadata"
    
    # Background processing
    enable_background_compression: bool = True
    background_check_interval_seconds: float = 300.0  # 5 minutes
    compression_batch_size: int = 100
    max_compression_threads: int = 2
    
    # Caching for compressed tiers
    enable_decompression_cache: bool = True
    decompression_cache_size: int = 1000  # Number of experiences to cache
    cache_ttl_seconds: float = 1800.0     # 30 minutes
    
    # Migration and recovery
    enable_atomic_transitions: bool = True
    backup_before_compression: bool = True
    recovery_on_startup: bool = True
    
    def validate(self):
        """Validate tiered storage configuration."""
        super().validate()
        
        # Validate tier thresholds
        thresholds = self.tier_thresholds
        if thresholds.warm_age_seconds >= thresholds.cold_age_seconds:
            raise ValueError("Warm age threshold must be less than cold age threshold")
        
        if thresholds.max_hot_experiences <= 0:
            raise ValueError("Max hot experiences must be positive")
        
        # Validate compression strategies
        available_strategies = CompressionFactory.get_available_strategies()
        if self.warm_compression_strategy not in available_strategies:
            raise ValueError(f"Invalid warm compression strategy: {self.warm_compression_strategy}")
        
        if self.cold_compression_strategy not in available_strategies:
            raise ValueError(f"Invalid cold compression strategy: {self.cold_compression_strategy}")
        
        # Validate directories
        for dir_name in [self.warm_storage_dir, self.cold_storage_dir, self.metadata_storage_dir]:
            if not dir_name:
                raise ValueError("Storage directories cannot be empty")


@dataclass
class TierMetadata:
    """Metadata for a tiered experience."""
    experience_id: str
    tier: ExperienceTier
    original_timestamp: float
    tier_transition_timestamp: float
    access_count: int
    last_access_timestamp: float
    compression_algorithm: Optional[str] = None
    compression_metadata: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None
    file_offset: Optional[int] = None
    compressed_size: Optional[int] = None
    original_size: Optional[int] = None


@dataclass 
class TierStatistics:
    """Statistics for a storage tier."""
    tier: ExperienceTier
    experience_count: int
    total_original_size: int
    total_compressed_size: int
    compression_ratio: float
    avg_access_count: float
    oldest_timestamp: float
    newest_timestamp: float
    last_compression_time: float


class DecompressionCache:
    """LRU cache for decompressed experiences."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 1800.0):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[ExperienceRecord, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Get experience from cache if available and not expired."""
        with self._lock:
            if experience_id not in self._cache:
                self._misses += 1
                return None
            
            experience, timestamp = self._cache[experience_id]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[experience_id]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(experience_id)
            self._hits += 1
            return experience
    
    def put(self, experience_id: str, experience: ExperienceRecord) -> None:
        """Add experience to cache."""
        with self._lock:
            current_time = time.time()
            
            # Remove if already exists
            if experience_id in self._cache:
                del self._cache[experience_id]
            
            # Add to end
            self._cache[experience_id] = (experience, current_time)
            
            # Evict oldest if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }


class TieredStorageManager:
    """Manages tiered storage of experiences with compression."""
    
    def __init__(self, config: TieredStorageConfig, hot_buffer: Optional[MemoryMappedExperienceBuffer] = None):
        self.config = config
        self.hot_buffer = hot_buffer  # Reference to the hot tier buffer
        self.logger = logging.getLogger(f"{__name__}.TieredStorageManager")
        
        # Create compression strategies
        self.warm_compressor = CompressionFactory.create(
            config.warm_compression_strategy, 
            **config.warm_compression_kwargs
        )
        self.cold_compressor = CompressionFactory.create(
            config.cold_compression_strategy,
            **config.cold_compression_kwargs
        )
        
        # Initialize storage directories
        self._ensure_directories()
        
        # Metadata tracking
        self._tier_metadata: Dict[str, TierMetadata] = {}
        self._metadata_lock = threading.RLock()
        
        # Decompression cache
        self._decompression_cache = None
        if config.enable_decompression_cache:
            self._decompression_cache = DecompressionCache(
                max_size=config.decompression_cache_size,
                ttl_seconds=config.cache_ttl_seconds
            )
        
        # Thread pool for compression operations
        self._compression_executor = ThreadPoolExecutor(
            max_workers=config.max_compression_threads,
            thread_name_prefix="TieredStorage"
        )
        
        # Background processing
        self._background_running = False
        self._background_thread = None
        self._shutdown_event = threading.Event()
        
        # Statistics
        self._stats = {
            ExperienceTier.HOT: TierStatistics(ExperienceTier.HOT, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0),
            ExperienceTier.WARM: TierStatistics(ExperienceTier.WARM, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0),
            ExperienceTier.COLD: TierStatistics(ExperienceTier.COLD, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0)
        }
        self._stats_lock = threading.RLock()
        
        # Load existing metadata
        self._load_metadata()
        
        # Start background processing if enabled
        if config.enable_background_compression:
            self.start_background_processing()
    
    def _ensure_directories(self) -> None:
        """Ensure all storage directories exist."""
        directories = [
            self.config.warm_storage_dir,
            self.config.cold_storage_dir, 
            self.config.metadata_storage_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")
    
    def _load_metadata(self) -> None:
        """Load tier metadata from disk."""
        metadata_file = Path(self.config.metadata_storage_dir) / "tier_metadata.json"
        
        if not metadata_file.exists():
            self.logger.info("No existing tier metadata found")
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            with self._metadata_lock:
                for exp_id, metadata_data in metadata_dict.items():
                    # Convert tier string back to enum
                    metadata_data['tier'] = ExperienceTier(metadata_data['tier'])
                    
                    # Create TierMetadata object
                    metadata = TierMetadata(**metadata_data)
                    self._tier_metadata[exp_id] = metadata
            
            self.logger.info(f"Loaded metadata for {len(self._tier_metadata)} experiences")
            
        except Exception as e:
            self.logger.error(f"Failed to load tier metadata: {e}")
    
    def _save_metadata(self) -> None:
        """Save tier metadata to disk."""
        metadata_file = Path(self.config.metadata_storage_dir) / "tier_metadata.json"
        
        try:
            with self._metadata_lock:
                # Convert metadata to serializable format
                metadata_dict = {}
                for exp_id, metadata in self._tier_metadata.items():
                    metadata_data = {
                        'experience_id': metadata.experience_id,
                        'tier': metadata.tier.value,  # Convert enum to string
                        'original_timestamp': metadata.original_timestamp,
                        'tier_transition_timestamp': metadata.tier_transition_timestamp,
                        'access_count': metadata.access_count,
                        'last_access_timestamp': metadata.last_access_timestamp,
                        'compression_algorithm': metadata.compression_algorithm,
                        'compression_metadata': metadata.compression_metadata,
                        'file_path': metadata.file_path,
                        'file_offset': metadata.file_offset,
                        'compressed_size': metadata.compressed_size,
                        'original_size': metadata.original_size
                    }
                    metadata_dict[exp_id] = metadata_data
            
            # Write atomically using temporary file
            temp_file = metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
            
            temp_file.replace(metadata_file)
            self.logger.debug(f"Saved metadata for {len(metadata_dict)} experiences")
            
        except Exception as e:
            self.logger.error(f"Failed to save tier metadata: {e}")
    
    def start_background_processing(self) -> None:
        """Start background compression processing."""
        if self._background_running:
            return
        
        self._background_running = True
        self._shutdown_event.clear()
        
        self._background_thread = threading.Thread(
            target=self._background_processing_loop,
            name="TieredStorageBackground",
            daemon=True
        )
        self._background_thread.start()
        
        self.logger.info("Started background compression processing")
    
    def stop_background_processing(self) -> None:
        """Stop background compression processing."""
        if not self._background_running:
            return
        
        self._background_running = False
        self._shutdown_event.set()
        
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)
            if self._background_thread.is_alive():
                self.logger.warning("Background thread did not stop gracefully")
        
        self.logger.info("Stopped background compression processing")
    
    def _background_processing_loop(self) -> None:
        """Main loop for background compression processing."""
        self.logger.info("Background compression processing started")
        
        cycle_count = 0
        last_metadata_save = time.time()
        metadata_save_interval = 300.0  # Save metadata every 5 minutes
        
        while self._background_running and not self._shutdown_event.is_set():
            try:
                cycle_count += 1
                cycle_start = time.time()
                
                # Check if we're in a low activity period before doing intensive work
                is_low_activity = self.is_low_activity_period()
                
                if is_low_activity or cycle_count % 3 == 0:  # Force processing every 3rd cycle
                    # Check for experiences that need tier transitions
                    self._process_tier_transitions()
                else:
                    self.logger.debug("Skipping tier transitions due to high system activity")
                
                # Save metadata periodically
                current_time = time.time()
                if current_time - last_metadata_save >= metadata_save_interval:
                    self._save_metadata()
                    last_metadata_save = current_time
                
                # Log periodic status
                if cycle_count % 10 == 0:  # Every 10 cycles
                    metrics = self.get_compression_metrics()
                    self.logger.info(
                        f"Background processing cycle {cycle_count}: "
                        f"Hot tier: {metrics['tier_counts'].get('hot', 0)}, "
                        f"Warm tier: {metrics['tier_counts'].get('warm', 0)}, "
                        f"Cold tier: {metrics['tier_counts'].get('cold', 0)}, "
                        f"Cache hit rate: {metrics['decompression_cache'].get('hit_rate', 0):.2%}"
                    )
                
                # Adaptive wait time based on activity and cycle performance
                cycle_duration = time.time() - cycle_start
                base_wait = self.config.background_check_interval_seconds
                
                if is_low_activity:
                    wait_time = max(base_wait - cycle_duration, 10.0)  # More frequent during low activity
                else:
                    wait_time = max(base_wait * 2 - cycle_duration, 30.0)  # Less frequent during high activity
                
                # Wait for next cycle
                if self._shutdown_event.wait(wait_time):
                    break  # Shutdown requested
                    
            except Exception as e:
                self.logger.error(f"Error in background processing cycle {cycle_count}: {e}")
                # Wait longer after errors to avoid rapid failure loops
                if self._shutdown_event.wait(30.0):
                    break
        
        # Final metadata save on shutdown
        try:
            self._save_metadata()
        except Exception as e:
            self.logger.error(f"Failed to save metadata on shutdown: {e}")
        
        self.logger.info(f"Background compression processing stopped after {cycle_count} cycles")
    
    def _process_tier_transitions(self) -> None:
        """Process experiences that need to transition between tiers."""
        current_time = time.time()
        thresholds = self.config.tier_thresholds
        
        # Get candidates for tier transitions
        hot_to_warm_candidates = self._get_hot_to_warm_candidates(current_time)
        warm_to_cold_candidates = self._get_warm_to_cold_candidates(current_time)
        
        # Process transitions in batches
        batch_size = self.config.compression_batch_size
        
        # Submit compression jobs to thread pool
        futures = []
        
        # Process hot to warm transitions
        for i in range(0, len(hot_to_warm_candidates), batch_size):
            batch = hot_to_warm_candidates[i:i + batch_size]
            if batch:
                future = self._compression_executor.submit(
                    self._process_hot_to_warm_batch, batch
                )
                futures.append(future)
        
        # Process warm to cold transitions  
        for i in range(0, len(warm_to_cold_candidates), batch_size):
            batch = warm_to_cold_candidates[i:i + batch_size]
            if batch:
                future = self._compression_executor.submit(
                    self._process_warm_to_cold_batch, batch
                )
                futures.append(future)
        
        # Wait for all compression jobs to complete (with timeout)
        completed_jobs = 0
        failed_jobs = 0
        
        for future in futures:
            try:
                result = future.result(timeout=30.0)  # 30 second timeout per batch
                if result:
                    completed_jobs += 1
                else:
                    failed_jobs += 1
            except Exception as e:
                self.logger.error(f"Compression batch failed: {e}")
                failed_jobs += 1
        
        if completed_jobs > 0 or failed_jobs > 0:
            self.logger.info(
                f"Tier transitions completed: {completed_jobs} successful, {failed_jobs} failed"
            )
    
    def _get_hot_to_warm_candidates(self, current_time: float) -> List[str]:
        """Get experience IDs that should transition from hot to warm tier."""
        candidates = []
        thresholds = self.config.tier_thresholds
        
        # Check hot buffer for experiences that are old enough
        try:
            # Get experiences from hot buffer that exceed warm age threshold
            hot_stats = self.hot_buffer.get_buffer_statistics()
            
            # Check if hot buffer is over capacity
            if hot_stats.count > thresholds.max_hot_experiences:
                # Get oldest experiences to transition
                oldest_experiences = self.hot_buffer.get_oldest_experiences(
                    hot_stats.count - thresholds.max_hot_experiences + self.config.compression_batch_size
                )
                
                for exp in oldest_experiences:
                    if exp.experience_id and (current_time - exp.timestamp) >= thresholds.warm_age_seconds:
                        candidates.append(exp.experience_id)
                
                self.logger.debug(f"Found {len(candidates)} hot->warm candidates (capacity)")
            
            # Also check for aged-out experiences
            oldest_experiences = self.hot_buffer.get_oldest_experiences(self.config.compression_batch_size * 2)
            for exp in oldest_experiences:
                if exp.experience_id and exp.experience_id not in candidates:
                    if (current_time - exp.timestamp) >= thresholds.warm_age_seconds:
                        candidates.append(exp.experience_id)
            
            self.logger.debug(f"Found {len(candidates)} total hot->warm candidates")
            
        except Exception as e:
            self.logger.error(f"Error finding hot->warm candidates: {e}")
        
        return candidates
    
    def _get_warm_to_cold_candidates(self, current_time: float) -> List[str]:
        """Get experience IDs that should transition from warm to cold tier."""
        candidates = []
        thresholds = self.config.tier_thresholds
        
        with self._metadata_lock:
            warm_count = 0
            for exp_id, metadata in self._tier_metadata.items():
                if metadata.tier == ExperienceTier.WARM:
                    warm_count += 1
                    
                    # Check if experience should move to cold tier
                    age = current_time - metadata.original_timestamp
                    time_in_warm = current_time - metadata.tier_transition_timestamp
                    
                    should_transition = (
                        age >= thresholds.cold_age_seconds or
                        (warm_count > thresholds.max_warm_experiences and
                         time_in_warm >= thresholds.warm_age_seconds)
                    )
                    
                    if should_transition:
                        candidates.append(exp_id)
        
        self.logger.debug(f"Found {len(candidates)} warm->cold candidates")
        return candidates
    
    def _process_hot_to_warm_batch(self, experience_ids: List[str]) -> bool:
        """Process a batch of experiences transitioning from hot to warm tier."""
        successful_transitions = 0
        
        for exp_id in experience_ids:
            try:
                # Get experience from hot buffer
                experience = self.hot_buffer.get_experience_by_id(exp_id)
                if experience is None:
                    self.logger.warning(f"Experience {exp_id} not found in hot buffer")
                    continue
                
                # Compress and store in warm tier
                success = self._compress_to_warm_tier(exp_id, experience)
                if success:
                    successful_transitions += 1
                    
                    # Update statistics
                    with self._stats_lock:
                        self._stats[ExperienceTier.WARM].experience_count += 1
                        
            except Exception as e:
                self.logger.error(f"Failed to transition {exp_id} to warm tier: {e}")
        
        self.logger.debug(f"Hot->Warm batch: {successful_transitions}/{len(experience_ids)} successful")
        return successful_transitions > 0
    
    def _process_warm_to_cold_batch(self, experience_ids: List[str]) -> bool:
        """Process a batch of experiences transitioning from warm to cold tier."""
        successful_transitions = 0
        
        for exp_id in experience_ids:
            try:
                # Get experience from warm tier storage
                experience = self._decompress_from_warm_tier(exp_id)
                if experience is None:
                    self.logger.warning(f"Experience {exp_id} not found in warm tier")
                    continue
                
                # Compress and store in cold tier
                success = self._compress_to_cold_tier(exp_id, experience)
                if success:
                    # Remove from warm tier
                    self._remove_from_warm_tier(exp_id)
                    successful_transitions += 1
                    
                    # Update statistics
                    with self._stats_lock:
                        self._stats[ExperienceTier.WARM].experience_count -= 1
                        self._stats[ExperienceTier.COLD].experience_count += 1
                        
            except Exception as e:
                self.logger.error(f"Failed to transition {exp_id} to cold tier: {e}")
        
        self.logger.debug(f"Warm->Cold batch: {successful_transitions}/{len(experience_ids)} successful")
        return successful_transitions > 0
    
    def _compress_to_warm_tier(self, experience_id: str, experience: ExperienceRecord) -> bool:
        """Compress and store experience in warm tier."""
        try:
            # Compress the experience
            compression_result = self.warm_compressor.compress(experience)
            
            # Create storage file path
            warm_dir = Path(self.config.warm_storage_dir)
            warm_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a sharded directory structure for better performance
            shard = hash(experience_id) % 100
            shard_dir = warm_dir / f"shard_{shard:02d}"
            shard_dir.mkdir(exist_ok=True)
            
            file_path = shard_dir / f"{experience_id}.warm"
            
            # Write compressed data atomically
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(compression_result.compressed_data)
            
            temp_path.replace(file_path)
            
            # Update metadata
            current_time = time.time()
            with self._metadata_lock:
                self._tier_metadata[experience_id] = TierMetadata(
                    experience_id=experience_id,
                    tier=ExperienceTier.WARM,
                    original_timestamp=experience.timestamp,
                    tier_transition_timestamp=current_time,
                    access_count=0,
                    last_access_timestamp=current_time,
                    compression_algorithm=compression_result.algorithm,
                    compression_metadata=compression_result.metadata,
                    file_path=str(file_path),
                    file_offset=0,
                    compressed_size=compression_result.compressed_size,
                    original_size=compression_result.original_size
                )
            
            self.logger.debug(
                f"Compressed {experience_id} to warm tier: "
                f"{compression_result.original_size} -> {compression_result.compressed_size} bytes "
                f"(ratio: {compression_result.compression_ratio:.2f}x)"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compress {experience_id} to warm tier: {e}")
            return False
    
    def _compress_to_cold_tier(self, experience_id: str, experience: ExperienceRecord) -> bool:
        """Compress and store experience in cold tier."""
        try:
            # Compress the experience
            compression_result = self.cold_compressor.compress(experience)
            
            # Create storage file path
            cold_dir = Path(self.config.cold_storage_dir)
            cold_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a sharded directory structure for better performance
            shard = hash(experience_id) % 100
            shard_dir = cold_dir / f"shard_{shard:02d}"
            shard_dir.mkdir(exist_ok=True)
            
            file_path = shard_dir / f"{experience_id}.cold"
            
            # Write compressed data atomically
            temp_path = file_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                f.write(compression_result.compressed_data)
            
            temp_path.replace(file_path)
            
            # Update metadata
            current_time = time.time()
            with self._metadata_lock:
                self._tier_metadata[experience_id] = TierMetadata(
                    experience_id=experience_id,
                    tier=ExperienceTier.COLD,
                    original_timestamp=experience.timestamp,
                    tier_transition_timestamp=current_time,
                    access_count=0,
                    last_access_timestamp=current_time,
                    compression_algorithm=compression_result.algorithm,
                    compression_metadata=compression_result.metadata,
                    file_path=str(file_path),
                    file_offset=0,
                    compressed_size=compression_result.compressed_size,
                    original_size=compression_result.original_size
                )
            
            self.logger.debug(
                f"Compressed {experience_id} to cold tier: "
                f"{compression_result.original_size} -> {compression_result.compressed_size} bytes "
                f"(ratio: {compression_result.compression_ratio:.2f}x)"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compress {experience_id} to cold tier: {e}")
            return False
    
    def _decompress_from_warm_tier(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Decompress experience from warm tier storage."""
        return self._decompress_from_tier(experience_id, ExperienceTier.WARM)
    
    def _decompress_from_cold_tier(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Decompress experience from cold tier storage."""
        return self._decompress_from_tier(experience_id, ExperienceTier.COLD)
    
    def _decompress_from_tier(self, experience_id: str, tier: ExperienceTier) -> Optional[ExperienceRecord]:
        """Decompress experience from specified tier."""
        try:
            # Check cache first
            if self._decompression_cache:
                cached = self._decompression_cache.get(experience_id)
                if cached is not None:
                    self.logger.debug(f"Cache hit for {experience_id}")
                    return cached
            
            # Get metadata
            with self._metadata_lock:
                metadata = self._tier_metadata.get(experience_id)
                if metadata is None or metadata.tier != tier:
                    return None
            
            # Read compressed data from file
            if not metadata.file_path or not Path(metadata.file_path).exists():
                self.logger.error(f"File not found for {experience_id}: {metadata.file_path}")
                return None
            
            with open(metadata.file_path, 'rb') as f:
                if metadata.file_offset:
                    f.seek(metadata.file_offset)
                compressed_data = f.read()
            
            # Select appropriate decompressor
            if tier == ExperienceTier.WARM:
                decompressor = self.warm_compressor
            else:
                decompressor = self.cold_compressor
            
            # Decompress
            experience = decompressor.decompress(compressed_data, metadata.compression_metadata or {})
            
            # Update access statistics
            current_time = time.time()
            with self._metadata_lock:
                metadata.access_count += 1
                metadata.last_access_timestamp = current_time
            
            # Add to cache
            if self._decompression_cache:
                self._decompression_cache.put(experience_id, experience)
            
            self.logger.debug(f"Decompressed {experience_id} from {tier.value} tier")
            return experience
            
        except Exception as e:
            self.logger.error(f"Failed to decompress {experience_id} from {tier.value} tier: {e}")
            return None
    
    def _remove_from_warm_tier(self, experience_id: str) -> bool:
        """Remove experience from warm tier storage."""
        try:
            with self._metadata_lock:
                metadata = self._tier_metadata.get(experience_id)
                if metadata and metadata.tier == ExperienceTier.WARM:
                    # Remove file
                    if metadata.file_path and Path(metadata.file_path).exists():
                        Path(metadata.file_path).unlink()
                        self.logger.debug(f"Removed warm tier file: {metadata.file_path}")
                    
                    return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove {experience_id} from warm tier: {e}")
            return False
    
    def get_experience_from_tier(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Get experience from any compressed tier."""
        with self._metadata_lock:
            metadata = self._tier_metadata.get(experience_id)
            if metadata is None:
                return None
            
            if metadata.tier == ExperienceTier.WARM:
                return self._decompress_from_warm_tier(experience_id)
            elif metadata.tier == ExperienceTier.COLD:
                return self._decompress_from_cold_tier(experience_id)
            
        return None
    
    def shutdown(self) -> None:
        """Shutdown the tiered storage manager."""
        self.stop_background_processing()
        self._compression_executor.shutdown(wait=True)
        self._save_metadata()
        
        if self._decompression_cache:
            self._decompression_cache.clear()
        
        self.logger.info("Tiered storage manager shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def get_tier_statistics(self) -> Dict[ExperienceTier, TierStatistics]:
        """Get statistics for all tiers."""
        with self._stats_lock:
            return self._stats.copy()
    
    def get_compression_metrics(self) -> Dict[str, Any]:
        """Get compression performance metrics."""
        metrics = {
            'decompression_cache': {},
            'tier_counts': {},
            'compression_ratios': {},
            'background_processing': {
                'running': self._background_running,
                'thread_alive': self._background_thread.is_alive() if self._background_thread else False
            }
        }
        
        # Decompression cache metrics
        if self._decompression_cache:
            metrics['decompression_cache'] = self._decompression_cache.get_stats()
        
        # Tier statistics
        with self._stats_lock:
            for tier, stats in self._stats.items():
                metrics['tier_counts'][tier.value] = stats.experience_count
                metrics['compression_ratios'][tier.value] = stats.compression_ratio
        
        # Metadata counts
        with self._metadata_lock:
            tier_counts = defaultdict(int)
            for metadata in self._tier_metadata.values():
                tier_counts[metadata.tier.value] += 1
            metrics['metadata_tier_counts'] = dict(tier_counts)
        
        return metrics
    
    def pause_background_processing(self) -> None:
        """Pause background compression processing."""
        if self._background_running:
            self._shutdown_event.set()
            self.logger.info("Background processing paused")
    
    def resume_background_processing(self) -> None:
        """Resume background compression processing."""
        if self._background_running and self._shutdown_event.is_set():
            self._shutdown_event.clear()
            self.logger.info("Background processing resumed")
        elif not self._background_running:
            self.start_background_processing()
    
    def is_low_activity_period(self) -> bool:
        """Check if system is in low activity period for compression."""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Consider low activity if CPU < 30% and memory usage < 80%
            return cpu_usage < 30.0 and memory.percent < 80.0
        except ImportError:
            # Fallback: assume it's always low activity if psutil not available
            self.logger.warning("psutil not available, assuming low activity period")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to check system activity: {e}")
            return True


class TieredExperienceBuffer:
    """Experience buffer with tiered storage and compression."""
    
    def __init__(
        self,
        hot_buffer_path: str,
        hot_buffer_capacity: int = 10000,
        state_shape: Tuple[int, ...] = (10, 11, 11),
        policy_size: int = 213,
        config: Optional[TieredStorageConfig] = None
    ):
        """
        Initialize tiered experience buffer.
        
        Args:
            hot_buffer_path: Path for the hot tier memory-mapped buffer
            hot_buffer_capacity: Capacity of the hot tier buffer
            state_shape: Shape of state tensors
            policy_size: Size of policy vectors
            config: Tiered storage configuration
        """
        if config is None:
            config = TieredStorageConfig()
        
        config.validate()
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.TieredExperienceBuffer")
        
        # Initialize hot tier buffer (existing memory-mapped buffer)
        self.hot_buffer = MemoryMappedExperienceBuffer(
            file_path=hot_buffer_path,
            capacity=hot_buffer_capacity,
            state_shape=state_shape,
            policy_size=policy_size,
            create_if_missing=True,
            config=config  # Use base config for hot buffer
        )
        
        # Initialize tiered storage manager
        self.storage_manager = TieredStorageManager(config, self.hot_buffer)
        
        self.logger.info(f"Initialized tiered experience buffer with hot capacity {hot_buffer_capacity}")
    
    def add_experience(self, experience: ExperienceRecord, experience_id: Optional[str] = None) -> str:
        """Add experience to the buffer (always starts in hot tier)."""
        return self.hot_buffer.add_experience(experience, experience_id)
    
    def get_experience(self, index: int) -> Optional[ExperienceRecord]:
        """Get experience by index from hot buffer."""
        return self.hot_buffer.get_experience(index)
    
    def get_experience_by_id(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Get experience by ID from any tier."""
        # First check hot buffer
        experience = self.hot_buffer.get_experience_by_id(experience_id)
        if experience is not None:
            return experience
        
        # Check compressed tiers through storage manager
        return self._get_compressed_experience(experience_id)
    
    def _get_compressed_experience(self, experience_id: str) -> Optional[ExperienceRecord]:
        """Get experience from compressed tiers."""
        return self.storage_manager.get_experience_from_tier(experience_id)
    
    def sample_batch(self, batch_size: int, phase_weights: Optional[Dict[str, float]] = None) -> List[ExperienceRecord]:
        """Sample batch from all tiers."""
        # For now, sample primarily from hot tier
        # TODO: Implement intelligent sampling across tiers based on recency and importance
        return self.hot_buffer.sample_batch(batch_size, phase_weights)
    
    def get_buffer_statistics(self) -> BufferStatistics:
        """Get comprehensive buffer statistics across all tiers."""
        hot_stats = self.hot_buffer.get_buffer_statistics()
        
        # Add compressed tier statistics
        tier_stats = self.storage_manager.get_tier_statistics()
        
        # For now, return hot buffer stats with basic tier info
        # TODO: Create comprehensive multi-tier statistics
        return hot_stats
    
    def shutdown(self) -> None:
        """Shutdown the tiered buffer."""
        self.storage_manager.shutdown()
        # Hot buffer cleanup handled by its context manager
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    
    def force_tier_transitions(self) -> None:
        """Force immediate tier transitions (useful for testing)."""
        if self.storage_manager._background_running:
            self.storage_manager._process_tier_transitions()
        else:
            self.logger.warning("Background processing not running, cannot force transitions")
    
    def get_tier_statistics(self) -> Dict[ExperienceTier, TierStatistics]:
        """Get statistics for all tiers."""
        return self.storage_manager.get_tier_statistics()
    
    def get_compression_metrics(self) -> Dict[str, Any]:
        """Get compression performance metrics."""
        return self.storage_manager.get_compression_metrics() 