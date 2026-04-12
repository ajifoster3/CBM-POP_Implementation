import heapq
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    OPERATOR_COMPLETE = auto()
    ROBOT_ARRIVAL     = auto()


@dataclass(order=True)
class Event:
    time: float
    seq:  int
    type: EventType = field(compare=False)
    data: Any       = field(compare=False)


class EventQueue:
    """Min-heap priority queue ordered by (time, seq)."""

    def __init__(self):
        self._heap: list[Event] = []
        self._counter = 0

    def push(self, time: float, etype: EventType, data: Any) -> None:
        self._counter += 1
        heapq.heappush(self._heap, Event(time, self._counter, etype, data))

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek_time(self) -> float:
        return self._heap[0].time if self._heap else float('inf')

    def is_empty(self) -> bool:
        return not self._heap

    def __len__(self) -> int:
        return len(self._heap)
