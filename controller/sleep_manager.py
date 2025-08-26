import asyncio
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from kvcached.utils import get_kvcached_logger
from traffic_monitor import traffic_monitor

logger = get_kvcached_logger()


@dataclass
class SleepConfig:
    """Configuration for sleep mode management"""
    idle_threshold_seconds: int = 300  # 5 minutes
    check_interval_seconds: int = 60   # Check every minute
    auto_sleep_enabled: bool = False   # Whether to automatically put models to sleep
    wake_on_request: bool = True       # Whether to automatically wake models on request
    min_sleep_duration: int = 60       # Minimum time to keep model asleep (seconds)


class SleepManager:
    """Manages sleep mode for idle models to save resources"""
    
    def __init__(self, config: Optional[SleepConfig] = None):
        self.config = config or SleepConfig()
        self.sleeping_models: Dict[str, float] = {}  # model_name -> sleep_start_time
        self.manual_sleep_models: Set[str] = set()   # Models manually put to sleep
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the sleep manager"""
        self._running = True
        if self.config.auto_sleep_enabled:
            self._monitor_task = asyncio.create_task(self._monitor_idle_models())
        logger.info(f"Sleep manager started (auto_sleep: {self.config.auto_sleep_enabled})")
    
    async def stop(self):
        """Stop the sleep manager"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Sleep manager stopped")
    
    async def put_model_to_sleep(self, model_name: str, manual: bool = False) -> bool:
        """
        Put a specific model to sleep
        
        Args:
            model_name: Name of the model to put to sleep
            manual: Whether this is a manual sleep request
        
        Returns:
            True if model was put to sleep, False if already sleeping or error
        """
        if model_name in self.sleeping_models:
            logger.info(f"Model {model_name} is already sleeping")
            return False
        
        try:
            # In a real implementation, you would:
            # 1. Send a signal to the model process to reduce resources
            # 2. Move model weights to storage if needed
            # 3. Pause request processing for this model
            
            self.sleeping_models[model_name] = time.time()
            if manual:
                self.manual_sleep_models.add(model_name)
            
            logger.info(f"Put model {model_name} to sleep (manual: {manual})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to put model {model_name} to sleep: {e}")
            return False
    
    async def wake_model(self, model_name: str) -> bool:
        """
        Wake up a sleeping model
        
        Args:
            model_name: Name of the model to wake up
        
        Returns:
            True if model was woken up, False if not sleeping or error
        """
        if model_name not in self.sleeping_models:
            logger.info(f"Model {model_name} is not sleeping")
            return False
        
        try:
            # Check minimum sleep duration
            sleep_start_time = self.sleeping_models[model_name]
            sleep_duration = time.time() - sleep_start_time
            logger.info(f"Model {model_name} has only been sleeping for {sleep_duration:.1f}s, "
                          f"minimum is {self.config.min_sleep_duration}s")
            if sleep_duration < self.config.min_sleep_duration:
                logger.info(f"Model {model_name} has only been sleeping for {sleep_duration:.1f}s, "
                          f"minimum is {self.config.min_sleep_duration}s")
                return False
            
            # Right now, we just wake up the model by sending a signal to the model process
            # In a real implementation, you would:
            # 1. Load model weights back to GPU if needed
            # 2. Resume request processing for this model
            # 3. Send wake signal to model process
            
            del self.sleeping_models[model_name]
            self.manual_sleep_models.discard(model_name)
            
            logger.info(f"Woke up model {model_name} after {sleep_duration:.1f}s of sleep")
            return True
            
        except Exception as e:
            logger.error(f"Failed to wake up model {model_name}: {e}")
            return False
    
    def is_model_sleeping(self, model_name: str) -> bool:
        """Check if a model is currently sleeping"""
        return model_name in self.sleeping_models
    
    def get_sleeping_models(self) -> Dict[str, Dict]:
        """Get information about all sleeping models"""
        current_time = time.time()
        return {
            model_name: {
                'sleep_start_time': sleep_start_time,
                'sleep_duration': current_time - sleep_start_time,
                'manual_sleep': model_name in self.manual_sleep_models
            }
            for model_name, sleep_start_time in self.sleeping_models.items()
        }
    
    def get_sleep_candidates(self) -> List[str]:
        """Get models that are candidates for sleep mode based on activity"""
        idle_models = traffic_monitor.get_idle_models(self.config.idle_threshold_seconds)
        # Filter out already sleeping models
        return [model for model in idle_models if model not in self.sleeping_models]
    
    async def _monitor_idle_models(self):
        """Background task to automatically put idle models to sleep"""
        while self._running:
            try:
                await asyncio.sleep(self.config.check_interval_seconds)
                
                if not self.config.auto_sleep_enabled:
                    continue
                
                # Get models that should be put to sleep
                candidates = self.get_sleep_candidates()
                
                for model_name in candidates:
                    # Don't auto-sleep manually controlled models
                    if model_name not in self.manual_sleep_models:
                        await self.put_model_to_sleep(model_name, manual=False)
                
                if candidates:
                    logger.info(f"Auto-sleep check: put {len(candidates)} models to sleep: {candidates}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in sleep monitor: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    async def handle_request_wake(self, model_name: str) -> bool:
        """
        Handle wake-up request when a request comes for a sleeping model
        
        Args:
            model_name: Name of the model that needs to be woken up
        
        Returns:
            True if model was woken up or already awake, False if wake failed
        """
        if not self.config.wake_on_request:
            return False
            
        if model_name not in self.sleeping_models:
            return True  # Already awake
        
        logger.info(f"Incoming request for sleeping model {model_name}, attempting to wake up")
        return await self.wake_model(model_name)
    
    def update_config(self, **kwargs):
        """Update sleep manager configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated sleep config: {key} = {value}")
            else:
                logger.warning(f"Unknown config key: {key}")


# Global sleep manager instance
sleep_manager = SleepManager()