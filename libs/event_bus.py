"""features.txt component 2 (Ingestion Pipeline): "a fast, ordered, durable
belt to whichever consumer needs them... every consumer — both lanes, and
the Decoy — can read independently, without any of them waiting on each
other." That independence (FR-1, FR-12) is a consumer-group property: the
Bouncer and the Graph Builder must each see every event, not compete for
one shared copy. Both backends below honor that.

Two implementations of one Protocol:
  - InMemoryEventBus: asyncio queues, one per (topic, consumer group).
    Used for the demo and the test suite — no broker to stand up.
  - RedpandaEventBus: a real aiokafka client against a real
    Redpanda/Kafka-API broker. This is the production path (see
    infra/helm/kronus/templates/redpanda.yaml); it is not exercised by
    the test suite here, which has no broker to talk to, but it is real,
    complete code, not a stub.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class EventBus(ABC):
    """The Ingestion Pipeline's Event Stream, as every other component sees
    it. `key` is used for the same partition-affinity purpose a real Kafka
    key serves (e.g. keying by source_ip keeps one host's events ordered
    relative to each other) even in the in-memory backend, which is
    otherwise single-partition — see the `ponytail:` note on that class.
    """

    @abstractmethod
    async def publish(self, topic: str, key: str, value: bytes) -> None: ...

    @abstractmethod
    def subscribe(self, topic: str, group: str) -> AsyncIterator[bytes]:
        """An async generator: `async for value in bus.subscribe(...)`.
        Each distinct `group` receives its own full copy of the topic —
        independent consumer groups, not competing consumers.
        """

    @abstractmethod
    async def close(self) -> None: ...


class InMemoryEventBus(EventBus):
    # ponytail: single logical partition per topic (one asyncio.Queue per
    # (topic, group) pair) — correct ordering, no parallelism. Fine at demo/
    # test volume (spec.md NFR-5's 3-5k events/sec target is a *cluster*
    # number); Redpanda's real partitioning is what NFR-5 actually leans on
    # in production. Upgrade path: shard by key hash into N queues per
    # group if a local load test ever needs it.
    def __init__(self) -> None:
        self._queues: dict[tuple[str, str], asyncio.Queue[bytes]] = {}
        self._known_groups: dict[str, set[str]] = {}

    def _queue_for(self, topic: str, group: str) -> asyncio.Queue[bytes]:
        key = (topic, group)
        if key not in self._queues:
            self._queues[key] = asyncio.Queue()
            self._known_groups.setdefault(topic, set()).add(group)
        return self._queues[key]

    async def publish(self, topic: str, key: str, value: bytes) -> None:
        # Fan out to every group already subscribed to this topic — this is
        # what makes the Bouncer and the Graph Builder genuinely independent
        # readers of the same stream (features.txt connectivity map, edges
        # 6 & 7) instead of one stealing the other's messages.
        for group in self._known_groups.get(topic, ()):
            await self._queues[(topic, group)].put(value)

    async def subscribe(self, topic: str, group: str) -> AsyncIterator[bytes]:
        queue = self._queue_for(topic, group)
        while True:
            yield await queue.get()

    async def close(self) -> None:
        self._queues.clear()
        self._known_groups.clear()


class RedpandaEventBus(EventBus):
    """Production adapter: real aiokafka client against Redpanda's
    Kafka-API-compatible broker. Constructing this does not connect;
    `publish`/`subscribe` lazily start the underlying producer/consumers,
    matching aiokafka's own lifecycle.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap_servers = bootstrap_servers
        self._producer = None  # type: ignore[var-annotated]
        self._consumers: list = []

    async def _get_producer(self):
        from aiokafka import AIOKafkaProducer

        if self._producer is None:
            self._producer = AIOKafkaProducer(bootstrap_servers=self._bootstrap_servers)
            await self._producer.start()
        return self._producer

    async def publish(self, topic: str, key: str, value: bytes) -> None:
        producer = await self._get_producer()
        await producer.send_and_wait(topic, value=value, key=key.encode())

    async def subscribe(self, topic: str, group: str) -> AsyncIterator[bytes]:
        from aiokafka import AIOKafkaConsumer

        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=group,
            auto_offset_reset="latest",
            enable_auto_commit=True,
        )
        await consumer.start()
        self._consumers.append(consumer)
        try:
            async for record in consumer:
                yield record.value
        finally:
            await consumer.stop()

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
        for consumer in self._consumers:
            await consumer.stop()
