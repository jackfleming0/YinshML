#!/usr/bin/env python3
"""Migration tools for converting existing experience buffers to tiered storage."""

import os
import sys
import json
import time
import shutil
import hashlib
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
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
    ExperienceTier,
    TierThresholds
)


@dataclass
class MigrationConfig:
    """Configuration for experience buffer migration."""
    batch_size: int = 1000  # Number of experiences to process at once
    validation_sample_size: int = 100  # Number of experiences to validate
    backup_original: bool = True  # Whether to backup original files
    parallel_workers: int = 2  # Number of parallel migration workers
    target_tier_distribution: Dict[ExperienceTier, float] = None  # Target distribution across tiers
    
    def __post_init__(self):
        if self.target_tier_distribution is None:
            self.target_tier_distribution = {
                ExperienceTier.HOT: 0.2,   # 20% most recent in hot tier
                ExperienceTier.WARM: 0.5,  # 50% in warm tier
                ExperienceTier.COLD: 0.3   # 30% oldest in cold tier
            }


@dataclass
class MigrationProgress:
    """Tracks migration progress and statistics."""
    total_experiences: int = 0
    migrated_experiences: int = 0
    failed_experiences: int = 0
    validated_experiences: int = 0
    start_time: float = 0.0
    end_time: Optional[float] = None
    compression_ratios: Dict[ExperienceTier, float] = None
    data_integrity_checks_passed: int = 0
    data_integrity_checks_failed: int = 0
    bytes_processed: int = 0
    bytes_after_compression: int = 0
    
    def __post_init__(self):
        if self.compression_ratios is None:
            self.compression_ratios = {}
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    @property
    def progress_percentage(self) -> float:
        """Calculate migration progress percentage."""
        if self.total_experiences == 0:
            return 0.0
        return (self.migrated_experiences / self.total_experiences) * 100.0
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time."""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def migration_rate(self) -> float:
        """Calculate experiences per second."""
        elapsed = self.elapsed_time
        if elapsed == 0:
            return 0.0
        return self.migrated_experiences / elapsed
    
    @property
    def overall_compression_ratio(self) -> float:
        """Calculate overall compression ratio."""
        if self.bytes_after_compression == 0:
            return 0.0
        return self.bytes_processed / self.bytes_after_compression


@dataclass
class ValidationResult:
    """Result of data integrity validation."""
    experience_id: str
    passed: bool
    original_checksum: str
    migrated_checksum: str
    error_message: Optional[str] = None
    statistical_properties_match: bool = True
    compression_ratio: Optional[float] = None


class MigrationValidator:
    """Validates data integrity during migration."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.MigrationValidator")
    
    def calculate_experience_checksum(self, experience: ExperienceRecord) -> str:
        """Calculate SHA-256 checksum of experience data."""
        # Serialize experience data in a consistent way
        data_to_hash = (
            experience.state.tobytes() +
            experience.policy.tobytes() +
            str(experience.value).encode() +
            str(int(experience.phase)).encode() +
            str(experience.timestamp).encode()
        )
        return hashlib.sha256(data_to_hash).hexdigest()
    
    def validate_experience_pair(
        self, 
        original: ExperienceRecord, 
        migrated: ExperienceRecord,
        experience_id: str
    ) -> ValidationResult:
        """Validate that original and migrated experiences match."""
        original_checksum = self.calculate_experience_checksum(original)
        migrated_checksum = self.calculate_experience_checksum(migrated)
        
        passed = original_checksum == migrated_checksum
        error_message = None if passed else "Checksum mismatch after migration"
        
        # Additional statistical validation
        statistical_match = self._validate_statistical_properties(original, migrated)
        
        return ValidationResult(
            experience_id=experience_id,
            passed=passed and statistical_match,
            original_checksum=original_checksum,
            migrated_checksum=migrated_checksum,
            error_message=error_message,
            statistical_properties_match=statistical_match
        )
    
    def _validate_statistical_properties(
        self, 
        original: ExperienceRecord, 
        migrated: ExperienceRecord
    ) -> bool:
        """Validate that statistical properties remain consistent."""
        try:
            # Check state tensor properties
            if not np.allclose(original.state, migrated.state, rtol=1e-6):
                return False
            
            # Check policy distribution properties
            if not np.allclose(original.policy, migrated.policy, rtol=1e-6):
                return False
            
            # Check exact matches for non-float fields
            if (original.value != migrated.value or 
                original.phase != migrated.phase or
                abs(original.timestamp - migrated.timestamp) > 1e-6):
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Statistical validation failed: {e}")
            return False
    
    def batch_validate(
        self, 
        original_experiences: List[ExperienceRecord],
        migrated_experiences: List[ExperienceRecord],
        experience_ids: List[str]
    ) -> List[ValidationResult]:
        """Validate a batch of experiences."""
        results = []
        
        for orig, migr, exp_id in zip(original_experiences, migrated_experiences, experience_ids):
            result = self.validate_experience_pair(orig, migr, exp_id)
            results.append(result)
        
        return results


class ExperienceBufferMigrator:
    """Migrates experiences from legacy buffer to tiered storage."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.validator = MigrationValidator(config)
        self.logger = logging.getLogger(f"{__name__}.ExperienceBufferMigrator")
        self._migration_lock = threading.RLock()
        
    def migrate_buffer(
        self,
        source_buffer_path: str,
        target_config: TieredStorageConfig,
        manifest_path: Optional[str] = None
    ) -> MigrationProgress:
        """
        Migrate an entire experience buffer to tiered storage.
        
        Args:
            source_buffer_path: Path to the source memory-mapped buffer
            target_config: Configuration for the target tiered storage
            manifest_path: Optional path to save migration manifest
        
        Returns:
            MigrationProgress object with detailed migration statistics
        """
        self.logger.info(f"Starting migration from {source_buffer_path}")
        
        # Initialize progress tracking
        progress = MigrationProgress()
        
        try:
            # Load source buffer
            source_buffer = MemoryMappedExperienceBuffer(
                file_path=source_buffer_path,
                create_if_missing=False
            )
            
            # Get buffer statistics
            source_stats = source_buffer.get_buffer_statistics()
            progress.total_experiences = source_stats.count
            
            self.logger.info(f"Source buffer contains {progress.total_experiences} experiences")
            
            # Create target tiered buffer
            target_buffer = TieredExperienceBuffer(target_config)
            
            # Backup original if requested
            if self.config.backup_original:
                self._backup_original_buffer(source_buffer_path)
            
            # Perform migration in batches
            batch_count = 0
            experiences_migrated = 0
            
            for batch in self._get_experience_batches(source_buffer, self.config.batch_size):
                batch_count += 1
                self.logger.info(f"Processing batch {batch_count}, experiences {experiences_migrated}-{experiences_migrated + len(batch)}")
                
                # Assign tiers based on age and target distribution
                batch_with_tiers = self._assign_tiers_to_batch(batch, progress.total_experiences)
                
                # Migrate batch
                batch_results = self._migrate_batch(batch_with_tiers, target_buffer)
                
                # Update progress
                successful_migrations = sum(1 for success, _ in batch_results if success)
                progress.migrated_experiences += successful_migrations
                progress.failed_experiences += len(batch) - successful_migrations
                
                experiences_migrated += len(batch)
                
                # Validate sample if requested
                if self.config.validation_sample_size > 0:
                    validation_results = self._validate_migration_sample(
                        batch, batch_with_tiers, target_buffer
                    )
                    progress.validated_experiences += len(validation_results)
                    progress.data_integrity_checks_passed += sum(1 for r in validation_results if r.passed)
                    progress.data_integrity_checks_failed += sum(1 for r in validation_results if not r.passed)
            
            # Finalize migration
            progress.end_time = time.time()
            
            # Calculate compression statistics
            target_stats = target_buffer.get_buffer_statistics()
            tier_stats = target_buffer.storage_manager.get_tier_statistics()
            
            for tier, stats in tier_stats.items():
                if stats.total_size_bytes > 0 and stats.compressed_size_bytes > 0:
                    progress.compression_ratios[tier] = stats.total_size_bytes / stats.compressed_size_bytes
            
            # Save migration manifest
            if manifest_path:
                self._save_migration_manifest(progress, source_buffer_path, target_config, manifest_path)
            
            self.logger.info(f"Migration completed: {progress.migrated_experiences}/{progress.total_experiences} experiences migrated")
            
            return progress
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            progress.end_time = time.time()
            raise
        finally:
            # Cleanup resources
            try:
                source_buffer.close()
                target_buffer.close()
            except:
                pass
    
    def _backup_original_buffer(self, source_path: str) -> str:
        """Create backup of original buffer."""
        backup_path = f"{source_path}.backup_{int(time.time())}"
        self.logger.info(f"Creating backup: {backup_path}")
        shutil.copy2(source_path, backup_path)
        return backup_path
    
    def _get_experience_batches(
        self, 
        buffer: MemoryMappedExperienceBuffer, 
        batch_size: int
    ) -> Generator[List[Tuple[ExperienceRecord, str]], None, None]:
        """Generate batches of experiences from the buffer."""
        buffer_stats = buffer.get_buffer_statistics()
        total_count = buffer_stats.count
        
        for start_idx in range(0, total_count, batch_size):
            end_idx = min(start_idx + batch_size, total_count)
            batch = []
            
            for idx in range(start_idx, end_idx):
                experience = buffer.get_experience(idx)
                if experience is not None:
                    # Get experience ID (or generate one if missing)
                    experience_id = experience.experience_id or f"migrated_{idx}_{int(time.time())}"
                    batch.append((experience, experience_id))
            
            if batch:
                yield batch
    
    def _assign_tiers_to_batch(
        self, 
        batch: List[Tuple[ExperienceRecord, str]], 
        total_experiences: int
    ) -> List[Tuple[ExperienceRecord, str, ExperienceTier]]:
        """Assign target tiers to experiences based on age and distribution."""
        # Sort by timestamp (newest first)
        sorted_batch = sorted(batch, key=lambda x: x[0].timestamp, reverse=True)
        
        batch_with_tiers = []
        target_dist = self.config.target_tier_distribution
        
        for i, (experience, exp_id) in enumerate(sorted_batch):
            # Calculate position in overall dataset (approximate)
            relative_position = i / len(sorted_batch)
            
            # Assign tier based on target distribution
            if relative_position < target_dist[ExperienceTier.HOT]:
                tier = ExperienceTier.HOT
            elif relative_position < target_dist[ExperienceTier.HOT] + target_dist[ExperienceTier.WARM]:
                tier = ExperienceTier.WARM
            else:
                tier = ExperienceTier.COLD
            
            batch_with_tiers.append((experience, exp_id, tier))
        
        return batch_with_tiers
    
    def _migrate_batch(
        self, 
        batch_with_tiers: List[Tuple[ExperienceRecord, str, ExperienceTier]],
        target_buffer: TieredExperienceBuffer
    ) -> List[Tuple[bool, Optional[str]]]:
        """Migrate a batch of experiences to the target buffer."""
        results = []
        
        for experience, exp_id, target_tier in batch_with_tiers:
            try:
                # Add experience to target buffer
                migrated_id = target_buffer.add_experience(experience, exp_id)
                
                # TODO: Move experience to appropriate tier based on target_tier
                # This would require additional API on TieredExperienceBuffer
                
                results.append((True, migrated_id))
                
            except Exception as e:
                self.logger.error(f"Failed to migrate experience {exp_id}: {e}")
                results.append((False, str(e)))
        
        return results
    
    def _validate_migration_sample(
        self, 
        original_batch: List[Tuple[ExperienceRecord, str]],
        migrated_batch: List[Tuple[ExperienceRecord, str, ExperienceTier]],
        target_buffer: TieredExperienceBuffer
    ) -> List[ValidationResult]:
        """Validate a sample of migrated experiences."""
        sample_size = min(self.config.validation_sample_size, len(original_batch))
        sample_indices = np.random.choice(len(original_batch), size=sample_size, replace=False)
        
        validation_results = []
        
        for idx in sample_indices:
            original_exp, original_id = original_batch[idx]
            
            # Retrieve migrated experience
            migrated_exp = target_buffer.get_experience_by_id(original_id)
            
            if migrated_exp is not None:
                result = self.validator.validate_experience_pair(
                    original_exp, migrated_exp, original_id
                )
                validation_results.append(result)
            else:
                # Experience not found in target buffer
                validation_results.append(ValidationResult(
                    experience_id=original_id,
                    passed=False,
                    original_checksum="",
                    migrated_checksum="",
                    error_message="Experience not found in target buffer"
                ))
        
        return validation_results
    
    def _save_migration_manifest(
        self, 
        progress: MigrationProgress,
        source_path: str,
        target_config: TieredStorageConfig, 
        manifest_path: str
    ) -> None:
        """Save detailed migration manifest."""
        manifest = {
            "migration_timestamp": datetime.now().isoformat(),
            "source_buffer_path": source_path,
            "target_config": {
                "directory": target_config.directory,
                "tier_thresholds": asdict(target_config.tier_thresholds),
                "compression_batch_size": target_config.compression_batch_size
            },
            "migration_config": asdict(self.config),
            "progress": asdict(progress),
            "validation_summary": {
                "total_checks": progress.data_integrity_checks_passed + progress.data_integrity_checks_failed,
                "passed": progress.data_integrity_checks_passed,
                "failed": progress.data_integrity_checks_failed,
                "success_rate": progress.data_integrity_checks_passed / (progress.data_integrity_checks_passed + progress.data_integrity_checks_failed) if (progress.data_integrity_checks_passed + progress.data_integrity_checks_failed) > 0 else 0.0
            }
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        self.logger.info(f"Migration manifest saved to {manifest_path}")


def migrate_experience_buffer(
    source_buffer_path: str,
    target_directory: str,
    migration_config: Optional[MigrationConfig] = None,
    tiered_config: Optional[TieredStorageConfig] = None,
    manifest_path: Optional[str] = None
) -> MigrationProgress:
    """
    Convenience function to migrate an experience buffer to tiered storage.
    
    Args:
        source_buffer_path: Path to source memory-mapped buffer
        target_directory: Directory for tiered storage
        migration_config: Migration configuration (uses defaults if None)
        tiered_config: Tiered storage configuration (uses defaults if None)
        manifest_path: Path to save migration manifest
    
    Returns:
        MigrationProgress with detailed statistics
    """
    if migration_config is None:
        migration_config = MigrationConfig()
    
    if tiered_config is None:
        tiered_config = TieredStorageConfig(directory=target_directory)
    
    migrator = ExperienceBufferMigrator(migration_config)
    return migrator.migrate_buffer(source_buffer_path, tiered_config, manifest_path) 