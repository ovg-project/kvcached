import threading
from typing import Union


class NoOpLock:
    """A no-op lock that implements the same interface as threading.RLock"""

    def acquire(self, blocking=True, timeout=-1):
        return True

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def locked(self):
        return False


class NoOpCondition:
    """A no-op condition that implements the same interface as threading.Condition"""

    def __init__(self, lock: Union[threading.RLock, NoOpLock]):
        self.lock = lock

    def wait(self, timeout=None):
        return True

    def wait_for(self, predicate, timeout=None):
        return predicate()

    def notify(self, n=1):
        pass

    def notify_all(self):
        pass

    def acquire(self, *args):
        return self.lock.acquire(*args)

    def release(self):
        return self.lock.release()

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, *args):
        return self.lock.__exit__(*args)
