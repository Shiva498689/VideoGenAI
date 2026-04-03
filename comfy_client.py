"""
ComfyUI client — single instance, single GPU, one ngrok tunnel.

Both the txt2img (image-gen) and img2vid (video-gen) workflows are sent
to the SAME ComfyUI endpoint sequentially.  A global asyncio.Lock (GPU_LOCK)
ensures only one job ever runs on the GPU at a time, even if concurrent
HTTP requests come in to the FastAPI server.

Flow for every job:
  acquire GPU_LOCK
    → (optional) upload seed image, inject filename into workflow JSON
    → POST /prompt          (queue the workflow)
    → ws://…/ws             (listen until execution_complete)
    → GET  /history         (resolve output filename)
    → GET  /view            (stream-download the file)
  release GPU_LOCK
"""

import json
import uuid
import asyncio
import httpx
import websockets
from pathlib import Path

from config import settings

# One lock for the entire process — only one GPU job at a time
GPU_LOCK = asyncio.Lock()


class ComfyUIClient:
    def __init__(self, base_url: str):
        self.base_url  = base_url.rstrip("/")
        self.ws_url    = (
            self.base_url
            .replace("https://", "wss://")
            .replace("http://",  "ws://")
        ) + "/ws"
        self.client_id = str(uuid.uuid4())

    # ── Upload image to ComfyUI input folder ──────────────────────────────────
    async def upload_image(self, img_path: Path) -> str:
        """Uploads a local PNG/JPG to ComfyUI and returns its server-side filename."""
        url = f"{self.base_url}/upload/image"
        async with httpx.AsyncClient(timeout=1000) as client:
            with open(img_path, "rb") as f:
                resp = await client.post(
                    url,
                    files={"image": (img_path.name, f, "image/png")},
                    data={"overwrite": "true"},
                    headers={"ngrok-skip-browser-warning": "true"},
                )
        resp.raise_for_status()
        return resp.json()["name"]

    # ── Queue a workflow ──────────────────────────────────────────────────────
    async def queue_prompt(self, workflow: dict) -> str:
        url     = f"{self.base_url}/prompt"
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with httpx.AsyncClient(timeout=1000) as client:
            resp = await client.post(
                url, json=payload,
                headers={"ngrok-skip-browser-warning": "true"},
            )
        # ← CHANGED: capture body before raising so you see ComfyUI's real error
        if resp.status_code != 200:
            raise RuntimeError(
                f"ComfyUI /prompt returned {resp.status_code}.\n"
                f"Body: {resp.text}"
            )
        return resp.json()["prompt_id"]

    # ── Wait for completion via WebSocket ─────────────────────────────────────
    async def wait_for_output(self, prompt_id: str, timeout: int = 10000) -> str:
        """
        Listens on the ComfyUI WebSocket until our prompt_id finishes.
        Returns the output filename from history.
        Timeout is generous (300s) because LTX-2.3 generation can be slow.
        """
        ws_uri = f"{self.ws_url}?clientId={self.client_id}"

        async with websockets.connect(ws_uri) as ws:
            deadline = asyncio.get_event_loop().time() + timeout
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    raise TimeoutError(f"ComfyUI did not finish within {timeout}s")
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"ComfyUI did not finish within {timeout}s")

                if isinstance(raw, bytes):
                    continue  # skip binary preview frames

                msg   = json.loads(raw)
                mtype = msg.get("type")

                if mtype == "executing":
                    d = msg.get("data", {})
                    # node=None signals the prompt finished
                    if d.get("prompt_id") == prompt_id and d.get("node") is None:
                        break

                elif mtype == "execution_error":
                    raise RuntimeError(
                        f"ComfyUI execution error for prompt {prompt_id}: {msg}"
                    )

        return await self._resolve_output_filename(prompt_id)

    # ── Resolve output filename from /history ─────────────────────────────────
    async def _resolve_output_filename(self, prompt_id: str) -> str:
        url = f"{self.base_url}/history/{prompt_id}"
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"ngrok-skip-browser-warning": "true"})
        resp.raise_for_status()
        history = resp.json()

        try:
            outputs = history[prompt_id]["outputs"]
        except KeyError:
            raise ValueError(f"prompt_id {prompt_id} not found in history")

        # Search every output node for a file
        for node_out in outputs.values():
            for key in ("gifs", "videos", "images", "files"):
                items = node_out.get(key, [])
                if items:
                    return items[0]["filename"]

        raise ValueError(
            f"No output file found in history for prompt {prompt_id}. "
            f"Nodes with output: {list(outputs.keys())}"
        )

    # ── Stream-download output file ───────────────────────────────────────────
    async def download_output(self, filename: str, dest_path: Path) -> Path:
        """Downloads a file from ComfyUI /view to dest_path."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url}/view?filename={filename}&type=output"

        async with httpx.AsyncClient(timeout=1000) as client:
            async with client.stream(
                "GET", url,
                headers={"ngrok-skip-browser-warning": "true"},
            ) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(8192):
                        f.write(chunk)
        return dest_path

    # ── Master entry-point: acquire lock → run job → release lock ────────────
    async def run_job(
        self,
        workflow:       dict,
        dest_path:      Path,
        timeout:        int        = 100000,
        img_to_upload:  Path | None = None,
    ) -> Path:
        """
        Acquires GPU_LOCK, runs ONE full ComfyUI generation, releases lock.

        For image-gen:  call with img_to_upload=None
        For video-gen:  call with img_to_upload=<seed image path>
                        → uploads image + injects filename into VID_IMAGE_NODE_ID

        Because every call awaits this lock, image-gen and video-gen jobs
        will never overlap on the single GPU, regardless of concurrency.
        """
        async with GPU_LOCK:
            if img_to_upload is not None:
                comfy_img_name = await self.upload_image(img_to_upload)
                workflow[settings.VID_IMAGE_NODE_ID]["inputs"]["image"] = comfy_img_name

            prompt_id = await self.queue_prompt(workflow)
            out_fname = await self.wait_for_output(prompt_id, timeout=timeout)
            return await self.download_output(out_fname, dest_path)


# ── Singleton — imported everywhere ──────────────────────────────────────────
comfy = ComfyUIClient(settings.COMFY_URL)
