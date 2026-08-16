"""features.txt component 11: "the real daemon moves to a non-default
port, and Cowrie (or an equivalent) answers on 22 instead... The pattern
is config-driven and generalizes to any commonly-fingerprinted service...
without new engineering per service."

MTDController owns exactly that relocation decision — which port the
Decoy binds to (the well-known one) and which port the real service is
told to use instead — as plain configuration, not a new mechanism per
service (SSH is the flagship instance here; the same controller and the
same DecoySSHHoneypot generalize to another service by changing
`well_known_port` and the real listener's bind port, nothing else).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from libs.observability import ACTIVE_GAUGE
from services.decoy.ssh_honeypot import DecoySSHHoneypot


@dataclass(frozen=True)
class MTDConfig:
    service_name: str
    well_known_port: int  # where the Decoy listens (e.g. 22 for SSH)
    relocated_port: int  # where the real service now actually listens


class RealServiceStub:
    """Stands in for "the real internal service" at its relocated port.
    In a genuine deployment this is the actual daemon (reconfigured to a
    non-default port); here it's a minimal legitimate TCP listener so the
    relocation itself is concretely testable end-to-end — connecting to
    the relocated port reaches this, connecting to the well-known port
    reaches the Decoy, and those are two different things.
    """

    def __init__(self, port: int) -> None:
        self._port = port
        self._server: asyncio.Server | None = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writer.write(b"SSH-2.0-OpenSSH_9.6\r\n")
        await writer.drain()
        writer.close()

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, host="", port=self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()


class MTDController:
    """Runs the Decoy on the well-known port and the real service stub on
    the relocated port, per `config` — the concrete implementation of
    "reconnaissance assumes a well-known port means what it says" being
    false (features.txt component 11's WHY IT EXISTS).
    """

    def __init__(self, config: MTDConfig, on_session_complete) -> None:
        self._config = config
        self._decoy = DecoySSHHoneypot(port=config.well_known_port, on_session_complete=on_session_complete)
        self._real_service = RealServiceStub(port=config.relocated_port)

    async def activate(self) -> None:
        await self._real_service.start()
        await self._decoy.start()
        ACTIVE_GAUGE.labels(component="decoy", kind="mtd_active").set(1)

    async def deactivate(self) -> None:
        await self._decoy.stop()
        await self._real_service.stop()
        ACTIVE_GAUGE.labels(component="decoy", kind="mtd_active").set(0)

    @property
    def config(self) -> MTDConfig:
        return self._config
