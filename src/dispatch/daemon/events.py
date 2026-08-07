"""The daemon's internal event bus.

One bus serves three consumers: connected clients (pushed over the socket), notification
sinks, and the demand-driven sampler. That last one is why the bus tracks subscriber
counts per topic -- with nobody watching ``system``, the daemon does not measure the
machine at all, which is most of how idle CPU stays at zero.

Every subscriber queue is **bounded**. A client that stops reading -- a suspended TUI, a
frozen SSH pipe -- must not be able to grow the daemon's memory or delay a job transition.
When a queue fills, its backlog is dropped and replaced with a single ``resync`` event
telling that client to refetch. Over months of uptime, unbounded queues are how daemons
die.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterable
from typing import Any

from dispatch.ipc.protocol import EVENT_TOPICS, Event, Notification, Topic

__all__ = ["EventBus", "Subscription"]

log = logging.getLogger(__name__)

DEFAULT_QUEUE_SIZE = 256


class Subscription:
    """One consumer's view of the bus.

    Args:
        maxsize: Queue depth before the backlog is dropped in favour of a resync.
    """

    def __init__(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> None:
        self.queue: asyncio.Queue[Notification] = asyncio.Queue(maxsize=maxsize)
        self.topics: set[str] = set()
        self.dropped = 0

    def wants(self, topic: Topic) -> bool:
        """Whether this subscriber has asked for ``topic``."""
        return str(topic) in self.topics

    def set_topics(self, topics: Iterable[str]) -> None:
        """Replace this subscriber's topic set."""
        self.topics = {str(t) for t in topics}

    def offer(self, notification: Notification) -> None:
        """Enqueue a notification, degrading to a resync rather than blocking.

        Never awaits and never raises: this is called from the middle of a state
        transition, and a slow client must not be able to reach back into the scheduler.
        """
        try:
            self.queue.put_nowait(notification)
        except asyncio.QueueFull:
            self.dropped += self.queue.qsize()
            _drain(self.queue)
            with contextlib.suppress(asyncio.QueueFull):  # the queue was just drained
                self.queue.put_nowait(
                    Notification(event=str(Event.RESYNC), data={"dropped": self.dropped})
                )
            log.warning(
                "Client fell behind; dropped %d events and asked it to resync", self.dropped
            )


def _drain(queue: asyncio.Queue[Notification]) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


class EventBus:
    """Fan-out of daemon events to subscribers.

    Args:
        on_demand_change: Called with the set of topics that currently have at least one
            subscriber, whenever that set changes. The sampler uses this to start and stop
            measuring rather than running a timer nobody reads.
    """

    def __init__(self, *, on_demand_change: Callable[[set[str]], None] | None = None) -> None:
        self._subscribers: set[Subscription] = set()
        self._on_demand_change = on_demand_change

    def subscribe(self, maxsize: int = DEFAULT_QUEUE_SIZE) -> Subscription:
        """Register a new subscriber."""
        subscription = Subscription(maxsize)
        self._subscribers.add(subscription)
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscriber and recompute demand."""
        self._subscribers.discard(subscription)
        self._notify_demand()

    def set_topics(self, subscription: Subscription, topics: Iterable[str]) -> None:
        """Change one subscriber's topics and recompute demand."""
        subscription.set_topics(topics)
        self._notify_demand()

    @property
    def active_topics(self) -> set[str]:
        """Topics with at least one subscriber."""
        active: set[str] = set()
        for subscription in self._subscribers:
            active |= subscription.topics
        return active

    def has_subscribers(self, topic: Topic) -> bool:
        """Whether anybody is listening to ``topic``."""
        return any(sub.wants(topic) for sub in self._subscribers)

    @property
    def subscriber_count(self) -> int:
        """Total connected subscribers."""
        return len(self._subscribers)

    def publish(self, event: Event, data: dict[str, Any] | None = None) -> None:
        """Deliver an event to every subscriber that asked for its topic.

        Synchronous and non-blocking by design, so publishing from inside a state
        transition costs a dictionary lookup and a queue append.
        """
        topic = EVENT_TOPICS.get(event, Topic.DAEMON)
        notification = Notification(event=str(event), data=data or {})
        for subscription in self._subscribers:
            if subscription.wants(topic):
                subscription.offer(notification)

    def _notify_demand(self) -> None:
        if self._on_demand_change:
            self._on_demand_change(self.active_topics)
