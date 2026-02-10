from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UdpJsonMessage:
    addr: tuple[str, int]
    payload: dict[str, Any]


class UdpJsonSender:
    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._sock.sendto(data, self._addr)


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, queue: asyncio.Queue[UdpJsonMessage]) -> None:
        self._queue = queue

    def datagram_received(self, data: bytes, addr):  # noqa: ANN001
        try:
            payload = json.loads(data.decode("utf-8", errors="replace"))
            if isinstance(payload, dict):
                self._queue.put_nowait(UdpJsonMessage(addr=addr, payload=payload))
        except Exception:
            return


class UdpJsonReceiver:
    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = int(port)
        self._queue: asyncio.Queue[UdpJsonMessage] = asyncio.Queue(maxsize=1000)
        self._transport = None

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(self._queue),
            local_addr=(self._host, self._port),
        )

    async def stop(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    async def recv(self) -> UdpJsonMessage:
        return await self._queue.get()

