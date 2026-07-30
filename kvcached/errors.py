# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Transaction failures shared by native allocation and worker IPC."""


class MapQuarantinedError(RuntimeError):
    """Unpublished mappings remain; exclude their page IDs before retrying."""


class StateConsistencyError(RuntimeError):
    """Mapping safety cannot be established; stop the affected engine."""


class QuarantinedResizeError(ValueError):
    """Capacity cannot be changed while the pool owns quarantined pages."""
