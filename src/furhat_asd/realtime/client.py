from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator


@dataclass(frozen=True)
class FurhatMessage:
    type: str
    data: dict[str, Any]


class FurhatRealtimeClient:
    """
    Minimal Furhat Realtime API client.

    Protocol is JSON messages with a required top-level `type` field.
    """

    def __init__(self, ip: str, port: int, api_key: str) -> None:
        self._ip = ip
        self._port = port
        self._api_key = api_key
        self._log = logging.getLogger("furhat_asd.furhat")
        self._ws = None
        self._recv_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[FurhatMessage] = asyncio.Queue(maxsize=500)

    @property
    def ws_url(self) -> str:
        return f"ws://{self._ip}:{self._port}/v1/events"

    async def connect(self) -> None:
        try:
            import websockets
        except Exception as e:  # pragma: no cover
            raise RuntimeError("websockets is required") from e

        try:
            self._ws = await websockets.connect(self.ws_url, max_queue=32)
        except OSError as e:
            raise RuntimeError(
                "Failed to connect to Furhat Realtime API.\n"
                f"- URL: {self.ws_url}\n"
                "Common causes:\n"
                "- Wrong `furhat.ip` / `furhat.ws_port` in config\n"
                "- Realtime API not enabled on the robot\n"
                "- Robot not on same LAN / port blocked by firewall\n"
                "Quick checks (Windows PowerShell):\n"
                f"- ping {self._ip}\n"
                f"- Test-NetConnection -ComputerName {self._ip} -Port {self._port}\n"
            ) from e
        self._recv_task = asyncio.create_task(self._recv_loop())

        if self._api_key:
            await self.send({"type": "request.auth", "key": self._api_key})
            msg = await self.receive_one(timeout_s=5.0)
            if msg.type != "response.auth" or not bool(msg.data.get("access", False)):
                raise RuntimeError(f"Auth failed: {msg.data}")

    async def close(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            self._recv_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected")
        await self._ws.send(json.dumps(payload))

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    obj = json.loads(raw)
                    msg_type = str(obj.get("type", ""))
                    data = {k: v for k, v in obj.items() if k != "type"}
                    await self._queue.put(FurhatMessage(type=msg_type, data=data))
                except Exception:
                    self._log.exception("Failed to parse message")
        except asyncio.CancelledError:
            return
        except Exception:
            self._log.exception("WebSocket receive loop failed")

    async def messages(self) -> AsyncIterator[FurhatMessage]:
        while True:
            yield await self._queue.get()

    async def receive_one(self, timeout_s: float) -> FurhatMessage:
        return await asyncio.wait_for(self._queue.get(), timeout=timeout_s)

    async def start_users(self) -> None:
        await self.send({"type": "request.users.start"})

    async def users_once(self) -> None:
        await self.send({"type": "request.users.once"})

    async def stop_users(self) -> None:
        await self.send({"type": "request.users.stop"})

    async def start_camera(self) -> None:
        await self.send({"type": "request.camera.start"})

    async def camera_once(self) -> None:
        await self.send({"type": "request.camera.once"})

    async def stop_camera(self) -> None:
        await self.send({"type": "request.camera.stop"})

    async def attend_user(self, user_id: str) -> None:
        await self.send({"type": "request.attend.user", "user_id": user_id})

    async def attend_nobody(self) -> None:
        await self.send({"type": "request.attend.nobody"})
