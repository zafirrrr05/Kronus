"""features.txt component 11 (The Decoy): "a decoy that mimics a real
service on the port attackers expect to find it... speaking real protocol,
accepting any credentials, faking a shell."

Cowrie is the spec's named reference ("Cowrie or equivalent honeypot" —
spec.md §5, features.txt) but is a full standalone Twisted-based service
not meant to be imported as a library. `asyncssh` gives a real SSH
protocol implementation (genuine key exchange, genuine auth negotiation,
genuine channel/session semantics — this is not a fake banner, it is an
actual SSH server) that this module wires up to do exactly what a
honeypot needs: accept the connection, accept any credentials, hand back
a shell that looks real and logs everything typed. See
tests/unit/test_decoy.py, which connects a real asyncssh *client* to this
server to verify the interaction end-to-end, not just unit-level mocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import asyncssh

from libs.observability import observe

FAKE_HOSTNAME = "prod-app-server-03"
FAKE_PROMPT = f"root@{FAKE_HOSTNAME}:~# "
FAKE_BANNER = "Welcome to Ubuntu 22.04.4 LTS\nLast login: {ts} from 10.0.0.14\n"


@dataclass
class DecoySession:
    """Everything captured from one honeypot interaction — this becomes
    both a TelemetryEvent (via services.telemetry_exporter.converters) and
    a confidence=1.0 DetectionVerdict (via services/decoy/pipeline.py).
    """

    source_ip: str
    dest_port: int
    decoy_service: str = "ssh"
    credentials_tried: list[str] = field(default_factory=list)
    commands_typed: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None

    @property
    def duration_ms(self) -> int:
        end = self.ended_at or datetime.now(timezone.utc)
        return int((end - self.started_at).total_seconds() * 1000)


class _DecoySSHServer(asyncssh.SSHServer):
    """spec.md §3.1 honeypot_detail fields, populated live as an attacker
    interacts: every credential attempt and every command typed. The
    session object is attached to the connection via set_extra_info so
    _handle_shell (registered separately as process_factory) can retrieve
    it — asyncssh's standard pattern for per-connection state.
    """

    def __init__(self, on_session_complete: Callable[[DecoySession], None]) -> None:
        self._on_session_complete = on_session_complete
        self._session: DecoySession | None = None

    def connection_made(self, conn: asyncssh.SSHServerConnection) -> None:
        peer = conn.get_extra_info("peername")
        source_ip = peer[0] if peer else "unknown"
        sockname = conn.get_extra_info("sockname")
        with observe("decoy", "connection_made", source_ip=source_ip):
            self._session = DecoySession(source_ip=source_ip, dest_port=sockname[1] if sockname else 0)
            conn.set_extra_info(decoy_session=self._session)

    def connection_lost(self, exc: Exception | None) -> None:
        if self._session is not None:
            self._session.ended_at = datetime.now(timezone.utc)
            self._on_session_complete(self._session)

    def begin_auth(self, username: str) -> bool:
        return True  # require an auth step so credentials actually get tried

    def password_auth_supported(self) -> bool:
        return True

    async def validate_password(self, username: str, password: str) -> bool:
        # Accept anything — spec.md component 11: "accepting any
        # credentials" is the mechanism, not a bug in the auth check.
        if self._session is not None:
            self._session.credentials_tried.append(f"{username}:{password}")
        return True


async def _handle_shell(process: asyncssh.SSHServerProcess) -> None:
    session: DecoySession | None = process.get_extra_info("decoy_session")
    process.stdout.write(FAKE_BANNER.format(ts=datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S %Y")))
    process.stdout.write(FAKE_PROMPT)
    async for line in process.stdin:
        command = line.rstrip("\n")
        if not command:
            process.stdout.write(FAKE_PROMPT)
            continue
        if session is not None:
            session.commands_typed.append(command)
        if command in ("exit", "logout"):
            break
        binary = command.split()[0]
        process.stdout.write(f"bash: {binary}: command not found\n{FAKE_PROMPT}")
    process.exit(0)


class DecoySSHHoneypot:
    """Owns the listening socket. `on_session_complete` is called with a
    finished DecoySession every time a connection ends — the caller
    (services/decoy/pipeline.py) turns that into a TelemetryEvent and a
    DetectionVerdict.
    """

    def __init__(self, port: int, on_session_complete: Callable[[DecoySession], None]) -> None:
        self._port = port
        self._on_session_complete = on_session_complete
        self._server: asyncssh.SSHAcceptor | None = None
        self._host_key = asyncssh.generate_private_key("ssh-rsa")

    async def start(self) -> None:
        self._server = await asyncssh.create_server(
            lambda: _DecoySSHServer(self._on_session_complete),
            host="",
            port=self._port,
            server_host_keys=[self._host_key],
            process_factory=_handle_shell,
        )

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
