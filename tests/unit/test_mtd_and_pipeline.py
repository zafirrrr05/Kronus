import asyncio

import asyncssh
import pytest

from libs.event_bus import InMemoryEventBus
from services.decoy.mtd import MTDConfig, MTDController
from services.decoy.pipeline import DecoyPipeline, decoy_verdict
from services.decoy.ssh_honeypot import DecoySession

MTD_DECOY_PORT = 22030
MTD_REAL_PORT = 22031


@pytest.mark.asyncio
async def test_mtd_well_known_port_serves_the_decoy_not_the_real_service():
    sessions: list[DecoySession] = []
    config = MTDConfig(service_name="ssh", well_known_port=MTD_DECOY_PORT, relocated_port=MTD_REAL_PORT)
    controller = MTDController(config, on_session_complete=sessions.append)
    await controller.activate()
    try:
        # connecting to the well-known port must reach the decoy: it
        # accepts arbitrary credentials, which a real sshd would not.
        async with asyncssh.connect(
            "127.0.0.1", port=MTD_DECOY_PORT, username="root", password="anything",
            known_hosts=None,
        ):
            pass
        await asyncio.sleep(0.05)
    finally:
        await controller.deactivate()

    assert len(sessions) == 1
    assert sessions[0].credentials_tried == ["root:anything"]


@pytest.mark.asyncio
async def test_mtd_relocated_port_serves_a_different_listener():
    config = MTDConfig(service_name="ssh", well_known_port=MTD_DECOY_PORT + 1, relocated_port=MTD_REAL_PORT + 1)
    controller = MTDController(config, on_session_complete=lambda s: None)
    await controller.activate()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", MTD_REAL_PORT + 1), timeout=2
        )
        banner = await asyncio.wait_for(reader.readline(), timeout=2)
        writer.close()
        # the real service stub, not asyncssh's protocol negotiation —
        # confirms the two ports genuinely serve two different things.
        assert banner.startswith(b"SSH-2.0-OpenSSH")
    finally:
        await controller.deactivate()


def test_decoy_verdict_has_fixed_confidence_and_correct_tier():
    from libs.constants import Label, Tier

    session = DecoySession(source_ip="203.0.113.5", dest_port=22, credentials_tried=["root:toor"])
    verdict = decoy_verdict(session, window_id="w-1")
    assert verdict.tier == Tier.DECOY
    assert verdict.label == Label.DECOY_INTERACTION
    assert verdict.confidence == 1.0


@pytest.mark.asyncio
async def test_pipeline_publishes_event_and_invokes_verdict_callback():
    bus = InMemoryEventBus()
    received_verdicts = []
    pipeline = DecoyPipeline(bus, on_verdict=received_verdicts.append)

    subscriber_seen = []

    async def reader():
        async for raw in bus.subscribe("telemetry-events", "logbook-consumers"):
            subscriber_seen.append(raw)
            return

    task = asyncio.create_task(reader())
    await asyncio.sleep(0.01)

    session = DecoySession(
        source_ip="198.51.100.4", dest_port=22, credentials_tried=["admin:admin"],
        commands_typed=["id"],
    )
    session.ended_at = session.started_at
    verdict = await pipeline.handle_session(session)

    await asyncio.wait_for(task, timeout=1)
    assert len(subscriber_seen) == 1
    assert received_verdicts == [verdict]
    assert verdict.confidence == 1.0
