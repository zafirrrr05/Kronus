import asyncio

import pytest

from libs.event_bus import InMemoryEventBus


@pytest.mark.asyncio
async def test_single_group_receives_published_messages_in_order():
    bus = InMemoryEventBus()
    received = []

    async def reader():
        async for msg in bus.subscribe("telemetry-events", "reader-1"):
            received.append(msg)
            if len(received) == 3:
                return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.01)  # let the subscriber register before publishing
    for i in range(3):
        await bus.publish("telemetry-events", key="h1", value=str(i).encode())
    await asyncio.wait_for(task, timeout=1)

    assert received == [b"0", b"1", b"2"]


@pytest.mark.asyncio
async def test_two_consumer_groups_each_see_every_message_independently():
    # This is the property FR-1 and FR-12 depend on: the Bouncer and the
    # Graph Builder are independent readers of the same Event Stream, not
    # competing for one shared copy of each event.
    bus = InMemoryEventBus()
    bouncer_seen, graph_builder_seen = [], []

    async def bouncer_reader():
        async for msg in bus.subscribe("telemetry-events", "bouncer-consumers"):
            bouncer_seen.append(msg)
            if len(bouncer_seen) == 2:
                return

    async def graph_builder_reader():
        async for msg in bus.subscribe("telemetry-events", "graph-builder-consumers"):
            graph_builder_seen.append(msg)
            if len(graph_builder_seen) == 2:
                return

    t1 = asyncio.create_task(bouncer_reader())
    t2 = asyncio.create_task(graph_builder_reader())
    await asyncio.sleep(0.01)
    await bus.publish("telemetry-events", key="h1", value=b"flood-event-1")
    await bus.publish("telemetry-events", key="h1", value=b"flood-event-2")
    await asyncio.wait_for(asyncio.gather(t1, t2), timeout=1)

    assert bouncer_seen == [b"flood-event-1", b"flood-event-2"]
    assert graph_builder_seen == [b"flood-event-1", b"flood-event-2"]


@pytest.mark.asyncio
async def test_publish_before_any_subscriber_is_dropped_not_buffered():
    # ponytail ceiling from the ordering note in event_bus.py: a group that
    # doesn't exist yet gets nothing retroactively — matches "latest" offset
    # semantics, not "earliest". Documented behavior, not a silent bug.
    bus = InMemoryEventBus()
    await bus.publish("telemetry-events", key="h1", value=b"missed-me")

    result_box: list[bytes] = []

    async def reader():
        async for msg in bus.subscribe("telemetry-events", "late-reader"):
            result_box.append(msg)
            return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.01)  # let the subscriber register (create its queue)
    await bus.publish("telemetry-events", key="h1", value=b"seen")
    await asyncio.wait_for(task, timeout=1)

    assert result_box == [b"seen"]  # "missed-me" never reached this group
