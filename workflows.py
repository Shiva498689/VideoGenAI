import copy
import json
from pathlib import Path

from config import settings

_BASE = Path(__file__).parent

def _load(filename: str) -> dict:
    path = _BASE / filename
    with open(path, "r") as f:
        return json.load(f)

def build_image_workflow(prompt: str) -> dict:
    wf = _load("image_workflow_template.json")
    wf = copy.deepcopy(wf)
    wf[settings.IMG_PROMPT_NODE_ID]["inputs"]["text"] = prompt
    return wf

def build_video_workflow(prompt: str) -> dict:
    wf = _load("video_workflow_template.json")
    wf = copy.deepcopy(wf)
    wf[settings.VID_PROMPT_NODE_ID]["inputs"]["text"] = prompt
    return wf
