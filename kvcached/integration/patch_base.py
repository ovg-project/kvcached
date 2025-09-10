"""
Shared patch infrastructure for unified patch application across libraries.
"""

import importlib
import os
import types
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


def enable_kvcached() -> bool:
    return os.getenv("ENABLE_KVCACHED", "false").lower() in ("true", "1")


class BasePatch(ABC):
    """Base class for all patches"""

    library: str = None  # Override in subclass ("vllm" or "sglang")
    target_module: str = None  # Module to patch
    target_class: Optional[str] = None  # Class to patch (optional)
    patch_name: str = None  # Human-readable name for logging

    def __init__(self):
        if self.patch_name is None:
            self.patch_name = self.__class__.__name__
        self.logger = logger

    @abstractmethod
    def apply(self, target_module: types.ModuleType) -> bool:
        """Apply the patch to the target module. Returns True on success."""
        pass

    def can_apply(self, target_module: types.ModuleType) -> bool:
        """Check if this patch can be applied to the target module"""
        if self.target_class:
            return hasattr(target_module, self.target_class)
        return True

    def _get_target_class(self, target_module: types.ModuleType):
        """Helper to get target class from module"""
        if self.target_class is None:
            raise ValueError(f"target_class not specified for {self.patch_name}")
        return getattr(target_module, self.target_class, None)

    def _is_already_patched(self, obj: Any, patch_marker: str = None) -> bool:
        """Check if object is already patched"""
        if patch_marker is None:
            patch_marker = f"__kvcached_patched_{self.patch_name}__"
        return getattr(obj, patch_marker, False) is True

    def _mark_as_patched(self, obj: Any, patch_marker: str = None) -> None:
        """Mark object as patched"""
        if patch_marker is None:
            patch_marker = f"__kvcached_patched_{self.patch_name}__"
        setattr(obj, patch_marker, True)


class PatchManager:
    """Manages application of patches for a library"""

    def __init__(self, library_name: str):
        self.library_name = library_name
        self.patches: List[BasePatch] = []
        self.logger = logger

    def register_patch(self, patch: BasePatch) -> None:
        """Register a patch to be applied"""
        if patch.library != self.library_name:
            raise ValueError(
                f"Patch {patch.patch_name} is for {patch.library}, not {self.library_name}"
            )
        self.patches.append(patch)

    def register_patches(self, patches: List[BasePatch]) -> None:
        """Register multiple patches"""
        for patch in patches:
            self.register_patch(patch)

    def apply_all_patches(self) -> Dict[str, bool]:
        """Apply all registered patches and return results"""
        results = {}

        self.logger.info(f"Applying {len(self.patches)} patches for {self.library_name}")

        for patch in self.patches:
            try:
                success = self._apply_single_patch(patch)
                results[patch.patch_name] = success

                if success:
                    self.logger.debug(f"Successfully applied {patch.patch_name}")
                else:
                    self.logger.warning(f"Failed to apply {patch.patch_name}")

            except Exception as e:
                self.logger.error(f"Error applying {patch.patch_name}: {e}")
                results[patch.patch_name] = False

        return results

    def _apply_single_patch(self, patch: BasePatch) -> bool:
        """Apply a single patch"""
        try:
            # Import target module
            target_module = importlib.import_module(patch.target_module)

            # Check if patch can be applied
            if not patch.can_apply(target_module):
                self.logger.debug(f"Skipping {patch.patch_name} - prerequisites not met")
                return False

            # Apply the patch
            return patch.apply(target_module)

        except ImportError as e:
            self.logger.error(
                f"Could not import target module {patch.target_module} for {patch.patch_name}: {e}"
            )
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error applying {patch.patch_name}: {e}")
            return False


def log_patch_results(library_name: str, results: Dict[str, bool]) -> None:
    """Helper function to log patch application results"""
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    if successful:
        logger.info(f"Successfully patched {library_name}: %s", ", ".join(successful))
    if failed:
        logger.warning(f"Failed to patch {library_name}: %s", ", ".join(failed))
