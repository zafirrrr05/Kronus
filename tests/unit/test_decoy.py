import asyncio

import asyncssh
import pytest

from services.decoy.ssh_honeypot import FAKE_PROMPT, DecoySSHHoneypot, DecoySession

TEST_PORT = 22022  # unprivileged, unlikely to collide


@pytest.mark.asyncio
async def test_honeypot_accepts_any_credentials_and_logs_a_session():
    completed: list[DecoySession] = []
    honeypot = DecoySSHHoneypot(port=TEST_PORT, on_session_complete=completed.append)
    await honeypot.start()
    try:
        async with asyncssh.connect(
            "127.0.0.1", port=TEST_PORT, username="root", password="toor",
            known_hosts=None,
        ) as conn:
            # An interactive shell session, not conn.run() — conn.run() uses
            # SSH "exec" semantics (the command travels as a request
            # parameter), which never reaches _handle_shell's
            # `async for line in process.stdin` at all. A real attacker
            # typing into a shell is what this needs to simulate; that is
            # `create_process()` with stdin writes, matching how a PTY
            # session actually behaves. Caught for real: conn.run() hung
            # this test indefinitely waiting on stdin that was never coming.
            async with conn.create_process(term_type="ansi") as process:
                await asyncio.wait_for(process.stdout.readuntil(FAKE_PROMPT), timeout=2)
                process.stdin.write("whoami\n")
                await asyncio.wait_for(process.stdout.readuntil(FAKE_PROMPT), timeout=2)
                process.stdin.write("ls -la\n")
                await asyncio.wait_for(process.stdout.readuntil(FAKE_PROMPT), timeout=2)
                process.stdin.write("exit\n")
                await asyncio.wait_for(process.wait_closed(), timeout=2)

        await asyncio.sleep(0.05)  # let connection_lost fire and finalize the session
    finally:
        await honeypot.stop()

    assert len(completed) == 1
    session = completed[0]
    assert session.source_ip == "127.0.0.1"
    assert session.credentials_tried == ["root:toor"]
    assert "whoami" in session.commands_typed
    assert "ls -la" in session.commands_typed
    assert session.duration_ms >= 0


@pytest.mark.asyncio
async def test_honeypot_accepts_completely_different_credentials_too():
    # The whole point: there is no "correct" password to find.
    completed: list[DecoySession] = []
    honeypot = DecoySSHHoneypot(port=TEST_PORT + 1, on_session_complete=completed.append)
    await honeypot.start()
    try:
        async with asyncssh.connect(
            "127.0.0.1", port=TEST_PORT + 1, username="admin", password="hunter2",
            known_hosts=None,
        ) as conn:
            async with conn.create_process(term_type="ansi") as process:
                await asyncio.wait_for(process.stdout.readuntil(FAKE_PROMPT), timeout=2)
                process.stdin.write("exit\n")
                await asyncio.wait_for(process.wait_closed(), timeout=2)
        await asyncio.sleep(0.05)
    finally:
        await honeypot.stop()

    assert completed[0].credentials_tried == ["admin:hunter2"]
