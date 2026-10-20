# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

"""Transaction failures shared by native allocation and worker IPC."""


class MapRetainedError(RuntimeError):
    """Confirmed mappings remain owned and must retry as the same page batch."""


class MapQuarantinedError(RuntimeError):
    """Unpublished mappings remain; exclude their page IDs before retrying."""


class StateConsistencyError(RuntimeError):
    """Mapping safety cannot be established; stop the affected engine."""


class QuarantinedResizeError(ValueError):
    """Capacity cannot be changed while the pool owns quarantined pages."""


class RetainedResizeError(QuarantinedResizeError):
    """Retry the quota once a confirmed retained map batch has recovered."""
