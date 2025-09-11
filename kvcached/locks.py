# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Optional, Protocol


class LockLike(Protocol):

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        ...

    def release(self) -> None:
        ...

    def __enter__(self) -> bool:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...


class ConditionLike(Protocol):

    def wait(self, timeout: Optional[float] = None) -> bool:
        ...

    def wait_for(self,
                 predicate: Callable[[], bool],
                 timeout: Optional[float] = None) -> bool:
        ...

    def notify(self, n: int = 1) -> None:
        ...

    def notify_all(self) -> None:
        ...

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        ...

    def release(self) -> None:
        ...

    def __enter__(self) -> bool:
        ...

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...


class NoOpLock(LockLike):
    """A no-op lock that implements the same interface as threading.RLock"""

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        pass

    def __enter__(self) -> bool:
        return True

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class NoOpCondition(ConditionLike):
    """A no-op condition that implements the same interface as threading.Condition"""

    def __init__(self, lock: LockLike):
        self.lock = lock

    def wait(self, timeout: Optional[float] = None) -> bool:
        return True

    def wait_for(self,
                 predicate: Callable[[], bool],
                 timeout: Optional[float] = None) -> bool:
        return predicate()

    def notify(self, n: int = 1) -> None:
        pass

    def notify_all(self) -> None:
        pass

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return self.lock.acquire(blocking, timeout)

    def release(self) -> None:
        return self.lock.release()

    def __enter__(self) -> bool:
        return self.lock.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return self.lock.__exit__(exc_type, exc_val, exc_tb)
