"""Compression strategies for different storage tiers."""

import pickle
import logging
import lz4.frame
import lzma
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of compression operation."""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: str
    metadata: Dict[str, Any]


class CompressionStrategy(ABC):
    """Abstract base class for compression strategies."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def compress(self, data: Any) -> CompressionResult:
        """Compress the given data."""
        pass
    
    @abstractmethod
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Any:
        """Decompress the given data using metadata."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this compression strategy."""
        return {
            'name': self.name,
            'description': self.__doc__ or "No description available"
        }


class LZ4CompressionStrategy(CompressionStrategy):
    """LZ4 compression strategy for warm tier - fast compression/decompression."""
    
    def __init__(self, compression_level: int = 1):
        """
        Initialize LZ4 compression strategy.
        
        Args:
            compression_level: LZ4 compression level (1-12, default 1 for speed)
        """
        super().__init__("LZ4")
        self.compression_level = compression_level
        
        # Validate compression level
        if not 1 <= compression_level <= 12:
            raise ValueError(f"LZ4 compression level must be 1-12, got {compression_level}")
    
    def compress(self, data: Any) -> CompressionResult:
        """Compress data using LZ4."""
        try:
            # Serialize the data first using pickle
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            original_size = len(serialized_data)
            
            # Compress using LZ4 frame compression with correct parameters
            compressed_data = lz4.frame.compress(
                serialized_data,
                compression_level=self.compression_level,
                content_checksum=True
            )
            
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0.0
            
            metadata = {
                'pickle_protocol': pickle.HIGHEST_PROTOCOL,
                'compression_level': self.compression_level,
                'original_size': original_size
            }
            
            self.logger.debug(
                f"LZ4 compressed {original_size} bytes to {compressed_size} bytes "
                f"(ratio: {compression_ratio:.2f}x)"
            )
            
            return CompressionResult(
                compressed_data=compressed_data,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                algorithm="LZ4",
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"LZ4 compression failed: {e}")
            raise
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Any:
        """Decompress LZ4 compressed data."""
        try:
            # Decompress using LZ4
            decompressed_data = lz4.frame.decompress(compressed_data)
            
            # Deserialize using pickle
            original_data = pickle.loads(decompressed_data)
            
            # Verify size if available in metadata
            if 'original_size' in metadata:
                expected_size = metadata['original_size']
                actual_size = len(decompressed_data)
                if expected_size != actual_size:
                    self.logger.warning(
                        f"Size mismatch: expected {expected_size}, got {actual_size}"
                    )
            
            self.logger.debug(f"LZ4 decompressed {len(compressed_data)} bytes to {len(decompressed_data)} bytes")
            
            return original_data
            
        except Exception as e:
            self.logger.error(f"LZ4 decompression failed: {e}")
            raise


class LZMACompressionStrategy(CompressionStrategy):
    """LZMA compression strategy for cold tier - high compression ratio."""
    
    def __init__(self, preset: int = 6, check: int = lzma.CHECK_CRC64):
        """
        Initialize LZMA compression strategy.
        
        Args:
            preset: LZMA preset level (0-9, default 6 for balance of speed/ratio)
            check: Integrity check to use (default CRC64)
        """
        super().__init__("LZMA")
        self.preset = preset
        self.check = check
        
        # Validate preset
        if not 0 <= preset <= 9:
            raise ValueError(f"LZMA preset must be 0-9, got {preset}")
    
    def compress(self, data: Any) -> CompressionResult:
        """Compress data using LZMA."""
        try:
            # Serialize the data first using pickle
            serialized_data = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            original_size = len(serialized_data)
            
            # Compress using LZMA
            compressed_data = lzma.compress(
                serialized_data,
                format=lzma.FORMAT_XZ,
                check=self.check,
                preset=self.preset
            )
            
            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 0.0
            
            metadata = {
                'pickle_protocol': pickle.HIGHEST_PROTOCOL,
                'preset': self.preset,
                'check': self.check,
                'format': lzma.FORMAT_XZ,
                'original_size': original_size
            }
            
            self.logger.debug(
                f"LZMA compressed {original_size} bytes to {compressed_size} bytes "
                f"(ratio: {compression_ratio:.2f}x)"
            )
            
            return CompressionResult(
                compressed_data=compressed_data,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                algorithm="LZMA",
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"LZMA compression failed: {e}")
            raise
    
    def decompress(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Any:
        """Decompress LZMA compressed data."""
        try:
            # Decompress using LZMA
            decompressed_data = lzma.decompress(
                compressed_data,
                format=metadata.get('format', lzma.FORMAT_XZ),
                memlimit=None  # No memory limit for decompression
            )
            
            # Deserialize using pickle
            original_data = pickle.loads(decompressed_data)
            
            # Verify size if available in metadata
            if 'original_size' in metadata:
                expected_size = metadata['original_size']
                actual_size = len(decompressed_data)
                if expected_size != actual_size:
                    self.logger.warning(
                        f"Size mismatch: expected {expected_size}, got {actual_size}"
                    )
            
            self.logger.debug(f"LZMA decompressed {len(compressed_data)} bytes to {len(decompressed_data)} bytes")
            
            return original_data
            
        except Exception as e:
            self.logger.error(f"LZMA decompression failed: {e}")
            raise


class CompressionFactory:
    """Factory for creating compression strategies."""
    
    _strategies = {
        'lz4': LZ4CompressionStrategy,
        'lzma': LZMACompressionStrategy,
    }
    
    @classmethod
    def create(cls, strategy_name: str, **kwargs) -> CompressionStrategy:
        """Create a compression strategy by name."""
        strategy_name = strategy_name.lower()
        
        if strategy_name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"Unknown compression strategy '{strategy_name}'. Available: {available}")
        
        strategy_class = cls._strategies[strategy_name]
        return strategy_class(**kwargs)
    
    @classmethod
    def get_available_strategies(cls) -> list[str]:
        """Get list of available compression strategies."""
        return list(cls._strategies.keys())
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type[CompressionStrategy]) -> None:
        """Register a new compression strategy."""
        if not issubclass(strategy_class, CompressionStrategy):
            raise ValueError("Strategy class must inherit from CompressionStrategy")
        
        cls._strategies[name.lower()] = strategy_class
        logger.info(f"Registered compression strategy: {name}") 