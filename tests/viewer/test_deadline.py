"""The browser tests' hard deadline must actually interrupt a stuck call."""

from __future__ import annotations

import socket
import time

import pytest

from tests.viewer.conftest import DeadlineExceeded, deadline


def test_a_slow_body_is_cut_off():
    started = time.monotonic()
    with pytest.raises(DeadlineExceeded, match="sleepy"):
        with deadline(0.2, "sleepy"):
            time.sleep(30)
    assert time.monotonic() - started < 5, "the deadline did not fire promptly"


def test_a_blocking_socket_read_is_cut_off():
    """Playwright blocks on a socket; the alarm has to break that, not just sleeps."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname())
    server, _ = listener.accept()
    try:
        with pytest.raises(DeadlineExceeded):
            with deadline(0.2, "blocked read"):
                client.recv(1)  # nothing is ever sent
    finally:
        client.close()
        server.close()
        listener.close()


def test_a_fast_body_is_untouched():
    with deadline(5, "quick"):
        result = sum(range(100))
    assert result == 4950
    # the timer must be disarmed again, or the next test inherits it
    time.sleep(0.4)
