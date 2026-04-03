"""
Workflow JSON builders.

Both workflows run on the SAME ComfyUI instance (single GPU).
GPU_LOCK in comfy_client.py ensures they never run concurrently.

image_workflow_template.json  → txt2img  (sdxl / KSampler)
video_workflow_template.json  → img2vid  (your existing video workflow)

Templates are loaded fresh on each call (no caching) so you can hot-swap
the JSON files without restarting the server.
"""

import copy
import json
from config import settings


def _load(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_image_workflow(prompt: str) -> dict:
    """
    Loads image_workflow_template.json and injects the enhanced image prompt
    into node IMG_PROMPT_NODE_ID (CLIPTextEncode / Positive prompt).

    # The template already has the correct LTX-2.3 wiring:
      Load Checkpoint → KSampler ← EmptyLatentImage
                      ← Positive (injected here)
                      ← Negative (fixed "blurry, distorted…")
      KSampler → VAEDecode → SaveImage
    """
    wf = _load("image_workflow_template.json")
    wf = copy.deepcopy(wf)   # never mutate the cached dict
    wf[settings.IMG_PROMPT_NODE_ID]["inputs"]["text"] = prompt
    return wf


def build_video_workflow(prompt: str) -> dict:
    """
    Loads video_workflow_template.json and injects the clip prompt.

    NOTE: The seed image is NOT injected here.
    comfy_client.run_job() uploads the image and injects the filename into
    VID_IMAGE_NODE_ID atomically while holding GPU_LOCK, ensuring no race
    between upload and workflow dispatch.
    """
    wf = _load("video_workflow_template.json")
    wf = copy.deepcopy(wf)
    wf[settings.VID_PROMPT_NODE_ID]["inputs"]["text"] = prompt
    return wf
