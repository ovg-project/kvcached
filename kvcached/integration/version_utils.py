# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""
Version detection and management utilities for kvcached integration.
"""

import importlib
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    pass

from packaging import version

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


class VersionRange:
    """Represents a version range like '>=0.9.0,<0.10.1' or '>=0.10.1'"""

    def __init__(self, range_spec: str):
        self.range_spec = range_spec
        self.constraints = self._parse_range(range_spec)

    def contains(self, ver: str) -> bool:
        """Check if version falls within this range"""
        try:
            v = version.parse(ver)
            for constraint in self.constraints:
                if not constraint(v):
                    return False
            return True
        except Exception as e:
            logger.warning(f"Error checking version {ver} against range {self.range_spec}: {e}")
            return False

    def _parse_range(self, range_spec: str) -> List[Callable]:
        """Parse range specifications like '>=0.9.0,<0.10.1' into constraint functions"""
        constraints = []

        # Split by comma and process each constraint
        for constraint_str in range_spec.split(","):
            constraint_str = constraint_str.strip()

            if constraint_str.startswith(">="):
                min_version = version.parse(constraint_str[2:])
                constraints.append(lambda v, mv=min_version: v >= mv)
            elif constraint_str.startswith(">"):
                min_version = version.parse(constraint_str[1:])
                constraints.append(lambda v, mv=min_version: v > mv)
            elif constraint_str.startswith("<="):
                max_version = version.parse(constraint_str[2:])
                constraints.append(lambda v, mv=max_version: v <= mv)
            elif constraint_str.startswith("<"):
                max_version = version.parse(constraint_str[1:])
                constraints.append(lambda v, mv=max_version: v < mv)
            elif constraint_str.startswith("=="):
                exact_version = version.parse(constraint_str[2:])
                constraints.append(lambda v, ev=exact_version: v == ev)
            else:
                # Assume exact match
                exact_version = version.parse(constraint_str)
                constraints.append(lambda v, ev=exact_version: v == ev)

        return constraints

    def __str__(self):
        return self.range_spec


class VersionedCallable(Protocol):
    """Protocol for callables with version attributes"""
    _version_ranges: List[VersionRange]
    _version_spec: str


def version_range(range_spec: str):
    """Decorator to mark version compatibility for patch methods"""

    def decorator(func: Callable[..., Any]) -> VersionedCallable:
        if not hasattr(func, "_version_ranges"):
            func._version_ranges = []  # type: ignore
        func._version_ranges.append(VersionRange(range_spec))  # type: ignore
        func._version_spec = range_spec  # type: ignore # For backward compatibility
        return func  # type: ignore

    return decorator


class LibrarySpecificCallable(Protocol):
    """Protocol for callables with library attributes"""
    _library: str


def library_specific(library: str):
    """Decorator to mark library-specific implementations"""

    def decorator(func: Callable[..., Any]) -> LibrarySpecificCallable:
        func._library = library  # type: ignore
        return func  # type: ignore

    return decorator


class VersionManager:
    """Manages version detection and patch selection"""

    _instance: Optional['VersionManager'] = None
    _initialized: bool = False

    def __new__(cls) -> 'VersionManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once to avoid resetting cache
        if not self._initialized:
            self.logger = logger
            self._version_cache: Dict[str, Optional[str]] = {}
            VersionManager._initialized = True

    @classmethod
    def get_instance(cls) -> 'VersionManager':
        """Get the singleton instance of VersionManager"""
        return cls()

    def detect_version(self, library_name: str, force_refresh: bool = False) -> Optional[str]:
        """Detect installed version of library"""
        if not force_refresh and library_name in self._version_cache:
            return self._version_cache[library_name]

        detected_version = None
        try:
            lib = importlib.import_module(library_name)
            detected_version = getattr(lib, "__version__", None)

            if detected_version is None:
                # Try alternative version attributes
                for attr_name in ["version", "VERSION", "_version"]:
                    detected_version = getattr(lib, attr_name, None)
                    if detected_version is not None:
                        break

            # Convert to string if it's not already
            if detected_version is not None:
                detected_version = str(detected_version)

        except ImportError:
            self.logger.debug(f"Could not import {library_name}")
        except Exception as e:
            self.logger.warning(f"Error detecting version for {library_name}: {e}")

        self._version_cache[library_name] = detected_version
        return detected_version

    def is_method_applicable(self, method: Callable[..., Any], library_name: str, version_str: str) -> bool:
        """Check if a method is applicable to the given library version"""
        # Check library specificity
        if hasattr(method, "_library") and method._library != library_name:
            return False

        # Check version ranges
        if hasattr(method, "_version_ranges"):
            for version_range_obj in method._version_ranges:
                if version_range_obj.contains(version_str):
                    return True
            return False

        # If no version constraints, assume it's applicable
        return True

    def get_applicable_methods(
        self, methods: List[Callable[..., Any]], library_name: str, version_str: str
    ) -> List[Callable[..., Any]]:
        """Get methods applicable to the given library version"""
        applicable = []
        for method in methods:
            if self.is_method_applicable(method, library_name, version_str):
                applicable.append(method)
        return applicable

    def log_version_info(self, library_name: str, detected_version: Optional[str]):
        """Log version detection results"""
        if detected_version:
            self.logger.info(f"Detected {library_name} version: {detected_version}")
        else:
            self.logger.warning(f"Could not detect {library_name} version")


class VersionAwarePatch:
    """Mixin class to add version awareness to patches"""

    # These attributes should be defined by subclasses
    library: str
    logger: Any  # Logger instance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version_manager = VersionManager.get_instance()
        self.detected_version: Optional[str] = None
        self.applicable_methods: List[Callable[..., Any]] = []

    def initialize_version_info(self) -> bool:
        """Initialize version information for this patch"""
        self.detected_version = self.version_manager.detect_version(self.library)

        if self.detected_version is None:
            self.logger.warning(f"Could not detect version for {self.library}")
            return False

        self.version_manager.log_version_info(self.library, self.detected_version)

        # Get all methods that have version decorators
        versioned_methods = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_version_ranges"):
                versioned_methods.append(attr)

        # Filter to applicable methods
        self.applicable_methods = self.version_manager.get_applicable_methods(
            versioned_methods, self.library, self.detected_version
        )

        self.logger.debug(
            f"Found {len(self.applicable_methods)} applicable methods for "
            f"{self.library} {self.detected_version}"
        )

        return True

    def is_version_supported(self) -> bool:
        """Check if the detected version is supported by this patch"""
        if not self.initialize_version_info():
            return False
        return len(self.applicable_methods) > 0 if self.applicable_methods else False
