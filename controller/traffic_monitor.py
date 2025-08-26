import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional

from kvcached.utils import get_kvcached_logger

logger = get_kvcached_logger()


@dataclass
class RequestStats:
    """Statistics for a single request"""
    timestamp: float
    model_name: str
    endpoint_path: str
    response_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ModelActivityStats:
    """Activity statistics for a model"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_activity_time: float = 0.0
    avg_response_time: float = 0.0
    request_timestamps: deque = field(
        default_factory=lambda: deque(maxlen=1000))

    def add_request(self, request_stats: RequestStats):
        """Add a new request to the statistics"""
        self.total_requests += 1
        self.last_activity_time = request_stats.timestamp
        self.request_timestamps.append(request_stats.timestamp)

        if request_stats.success:
            self.successful_requests += 1
            if request_stats.response_time:
                # Update running average of response time
                total_time = self.avg_response_time * (
                    self.successful_requests - 1)
                self.avg_response_time = (
                    total_time +
                    request_stats.response_time) / self.successful_requests
        else:
            self.failed_requests += 1

    def get_request_rate(self, window_seconds: int = 60) -> float:
        """Calculate request rate over the specified time window"""
        if not self.request_timestamps:
            return 0.0

        current_time = time.time()
        cutoff_time = current_time - window_seconds

        # Count requests within the time window
        recent_requests = sum(1 for ts in self.request_timestamps
                              if ts >= cutoff_time)
        return recent_requests / window_seconds

    def get_idle_time(self) -> float:
        """Get the idle time in seconds since last activity"""
        if self.last_activity_time == 0:
            return float('inf')  # Never had any activity
        return time.time() - self.last_activity_time

    def is_idle(self, idle_threshold_seconds: int = 300) -> bool:
        """Check if the model has been idle longer than the threshold (default 5 minutes)"""
        return self.get_idle_time() > idle_threshold_seconds


class TrafficMonitor:
    """Monitor traffic statistics for all models"""

    def __init__(self, idle_threshold_seconds: int = 300):
        self.model_stats: Dict[str, ModelActivityStats] = defaultdict(
            ModelActivityStats)
        self.idle_threshold_seconds = idle_threshold_seconds
        self._lock = Lock()
        self._request_history: List[RequestStats] = []

        # Start background cleanup task
        self._cleanup_task = None
        self._running = False

    async def start(self):
        """Start the traffic monitor background tasks"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Traffic monitor started")

    async def stop(self):
        """Stop the traffic monitor"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Traffic monitor stopped")

    def record_request_start(self, model_name: str,
                             endpoint_path: str) -> RequestStats:
        """Record the start of a request"""
        request_stats = RequestStats(timestamp=time.time(),
                                     model_name=model_name,
                                     endpoint_path=endpoint_path)

        with self._lock:
            self._request_history.append(request_stats)
            # Keep only recent history (last 10000 requests)
            if len(self._request_history) > 10000:
                self._request_history = self._request_history[-5000:]

        return request_stats

    def record_request_end(self,
                           request_stats: RequestStats,
                           success: bool = True,
                           response_time: Optional[float] = None,
                           error_message: Optional[str] = None):
        """Record the end of a request"""
        if response_time is None:
            response_time = time.time() - request_stats.timestamp

        request_stats.response_time = response_time
        request_stats.success = success
        request_stats.error_message = error_message

        with self._lock:
            self.model_stats[request_stats.model_name].add_request(
                request_stats)

        logger.debug(f"Request completed for {request_stats.model_name}: "
                     f"success={success}, response_time={response_time:.3f}s")

    def get_model_stats(self, model_name: str) -> Optional[ModelActivityStats]:
        """Get statistics for a specific model"""
        with self._lock:
            return self.model_stats.get(model_name)

    def get_all_model_stats(self) -> Dict[str, ModelActivityStats]:
        """Get statistics for all models"""
        with self._lock:
            return dict(self.model_stats)

    def get_idle_models(self,
                        idle_threshold_seconds: Optional[int] = None
                        ) -> List[str]:
        """Get list of models that have been idle longer than threshold"""
        if idle_threshold_seconds is None:
            idle_threshold_seconds = self.idle_threshold_seconds

        idle_models = []
        with self._lock:
            for model_name, stats in self.model_stats.items():
                if stats.is_idle(idle_threshold_seconds):
                    idle_models.append(model_name)

        return idle_models

    def get_active_models(self,
                          idle_threshold_seconds: Optional[int] = None
                          ) -> List[str]:
        """Get list of models that are currently active (not idle)"""
        if idle_threshold_seconds is None:
            idle_threshold_seconds = self.idle_threshold_seconds

        active_models = []
        with self._lock:
            for model_name, stats in self.model_stats.items():
                if not stats.is_idle(idle_threshold_seconds):
                    active_models.append(model_name)

        return active_models

    def get_traffic_summary(self, window_seconds: int = 60) -> Dict[str, Dict]:
        """Get a summary of traffic for all models"""
        summary = {}
        with self._lock:
            for model_name, stats in self.model_stats.items():
                summary[model_name] = {
                    'total_requests': stats.total_requests,
                    'successful_requests': stats.successful_requests,
                    'failed_requests': stats.failed_requests,
                    'request_rate': stats.get_request_rate(window_seconds),
                    'avg_response_time': stats.avg_response_time,
                    'last_activity_time': stats.last_activity_time,
                    'idle_time_seconds': stats.get_idle_time(),
                    'is_idle': stats.is_idle(self.idle_threshold_seconds)
                }

        return summary

    async def _periodic_cleanup(self):
        """Periodically clean up old data"""
        while self._running:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Clean up old request history
                cutoff_time = time.time() - 3600  # Keep 1 hour of history
                with self._lock:
                    self._request_history = [
                        req for req in self._request_history
                        if req.timestamp >= cutoff_time
                    ]

                logger.debug("Cleaned up old traffic data")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in traffic monitor cleanup: {e}")
                await asyncio.sleep(60)  # Wait a bit before retrying


# Global traffic monitor instance
traffic_monitor = TrafficMonitor()
