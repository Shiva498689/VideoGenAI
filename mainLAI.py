"""
CineForge FastAPI Backend
=========================

Full pipeline per scene
────────────────────────
 1. POST /projects/{pid}/scenes
      → Groq enhances raw prompt → 5 clip prompts + 1 image prompt
      → Queues txt2img job (GPU_LOCK serialises with video jobs)
      → Returns immediately; poll for status

 2. GET  /projects/{pid}/scenes/{idx}
      → Poll until status == "confirming"
      → Response includes source_image_url for the UI to display

 3. POST /projects/{pid}/scenes/{idx}/confirm  { "confirmed": true }
      → Approved: starts 5-clip generation chain in background
    POST /projects/{pid}/scenes/{idx}/confirm  { "confirmed": false }
      → Rejected: regenerates source image (same prompt)
    POST /projects/{pid}/scenes/{idx}/regenerate  { "prompt": "..." }
      → Rejected with new description: re-enhances + regenerates

 4. Background clip loop (for each of 5 clips):
      upload seed image → img2vid → download clip
      extract last frame → save as clip_{i}_last_frame.png (per-clip seed)
      use that last frame as seed for next clip
    → On completion: assemble 5 clips → 20s scene.mp4

 5. POST /projects/{pid}/finalize
      → Stitches all done scenes with lightning-flash transition → final.mp4

 6. POST /projects/{pid}/scenes/{idx}/clips/{clip_idx}/regenerate
      { "prompt": "optional new prompt override" }   ← prompt is optional
      → Regenerates ONE clip after scene is fully done
      → Re-extracts last-frame from the new clip
      → Re-generates all downstream clips (clip_{clip_idx+1} … clip_4)
        using their existing (or overridden) prompts + the new seed chain
      → Re-assembles scene.mp4
      → scene status temporarily becomes "regenerating_clip" during this,
        then returns to "done" when finished

 7. GET /projects/{pid}/scenes/{idx}/clips/{clip_idx}
      → Returns clip status, URL, and current prompt

CLIP REGENERATION DETAILS:
  - Only allowed when scene status is "done" OR "error" (post-generation).
  - The per-clip last frame is stored as clips/clip_{i}_last_frame.png so
    re-chaining downstream clips always has the correct seed image.
  - If a new prompt is supplied it overrides that clip's enhanced_prompts[i]
    for this regeneration AND is persisted in enhanced_prompts so future
    downstream regenerations pick it up.
  - Clip regeneration is serialised via GPU_LOCK just like normal generation.

FIXES APPLIED:
  - extract_last_frame, assemble_scene, concatenate_scenes all use
    subprocess.run() (blocking). Wrapped in asyncio.to_thread() so they
    never freeze the FastAPI event loop during long video operations.
  - Per-clip last frames stored at clip_{i}_last_frame.png instead of a
    single last_frame.png, enabling correct re-seeding for downstream clips.
"""

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from comfy_client import comfy
from config import settings
from frame_extractor import extract_last_frame
from prompt_enhancer import enhance_image_prompt, enhance_prompt
from storage import StorageManager
from video_processing import assemble_scene, concatenate_scenes
from workflows import build_image_workflow, build_video_workflow

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CineForge — AI Video Pipeline", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = StorageManager(settings.OUTPUT_DIR)

# Static mounts
app.mount("/files", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="files")
app.mount("/ui",    StaticFiles(directory="frontend", html=True),    name="ui")


# ── Pydantic models ───────────────────────────────────────────────────────────
class CreateProjectReq(BaseModel):
    title: str = "Untitled Project"

class SceneReq(BaseModel):
    prompt: str

class ConfirmReq(BaseModel):
    confirmed: bool   # True → proceed with clip gen, False → regen image

class ClipRegenerateReq(BaseModel):
    prompt: Optional[str] = None  # If omitted, reuses the existing enhanced prompt


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "ok", "ui": "/ui", "docs": "/docs"}


# ── Projects ──────────────────────────────────────────────────────────────────
@app.post("/projects", status_code=201)
async def create_project(req: CreateProjectReq):
    return storage.create_project(title=req.title)


@app.get("/projects")
async def list_projects():
    return storage.list_projects()


@app.get("/projects/{pid}")
async def get_project(pid: str):
    p = storage.get_project(pid)
    if not p:
        raise HTTPException(404, "Project not found")
    return _hydrate_project(p)


def _hydrate_project(p: dict) -> dict:
    """Attach URL fields to all scenes inside a project dict."""
    p = dict(p)
    p["scenes"] = [_hydrate_scene(p["project_id"], s) for s in p.get("scenes", [])]
    return p


# ── Scene: create ─────────────────────────────────────────────────────────────
@app.post("/projects/{pid}/scenes", status_code=201)
async def create_scene(pid: str, req: SceneReq, bg: BackgroundTasks):
    """
    Step 1 of the pipeline.
    Enhances the raw prompt via Groq, then kicks off txt2img in the background.
    Returns the scene stub immediately — poll GET /…/scenes/{idx} for updates.
    """
    if not storage.get_project(pid):
        raise HTTPException(404, "Project not found")

    try:
        clip_prompts = await enhance_prompt(req.prompt)
        img_prompt   = await enhance_image_prompt(req.prompt)
    except Exception as e:
        raise HTTPException(502, f"Prompt enhancement failed: {e}")

    scene     = storage.add_scene(pid, req.prompt, clip_prompts)
    scene_idx = scene["scene_idx"]
    storage.update_scene(pid, scene_idx,
                         enhanced_image_prompt=img_prompt,
                         status="generating_image")

    bg.add_task(_gen_source_image, pid, scene_idx, img_prompt)
    return _hydrate_scene(pid, storage.get_scene(pid, scene_idx))


# ── Scene: read ───────────────────────────────────────────────────────────────
@app.get("/projects/{pid}/scenes/{scene_idx}")
async def get_scene(pid: str, scene_idx: int):
    s = storage.get_scene(pid, scene_idx)
    if not s:
        raise HTTPException(404, "Scene not found")
    return _hydrate_scene(pid, s)


def _hydrate_scene(pid: str, s: dict) -> dict:
    """Attach *_url fields so the frontend can render images/videos directly."""
    s = dict(s)
    if s.get("source_image"):
        s["source_image_url"] = storage.rel_url(s["source_image"])
    s["clip_urls"] = [storage.rel_url(c) for c in s.get("clips", []) if c]
    if s.get("scene_video"):
        s["scene_video_url"] = storage.rel_url(s["scene_video"])
    # Expose per-clip prompts for frontend display
    s["clip_prompts"] = s.get("enhanced_prompts", [])
    return s


# ── Scene: confirm / reject source image ─────────────────────────────────────
@app.post("/projects/{pid}/scenes/{scene_idx}/confirm")
async def confirm_image(pid: str, scene_idx: int, req: ConfirmReq, bg: BackgroundTasks):
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    if scene["status"] != "confirming":
        raise HTTPException(400, f"Expected status 'confirming', got '{scene['status']}'")

    if not req.confirmed:
        img_prompt = scene.get("enhanced_image_prompt") or scene["raw_prompt"]
        storage.update_scene(pid, scene_idx, status="generating_image", error=None)
        bg.add_task(_gen_source_image, pid, scene_idx, img_prompt)
        return {"status": "generating_image", "message": "Regenerating source image…"}

    storage.update_scene(pid, scene_idx, status="generating_clips", error=None)
    bg.add_task(_gen_clips, pid, scene_idx)
    return {"status": "generating_clips", "message": "Generating 5 clips…"}


# ── Scene: regenerate with new prompt ─────────────────────────────────────────
@app.post("/projects/{pid}/scenes/{scene_idx}/regenerate")
async def regenerate_image(pid: str, scene_idx: int, req: SceneReq, bg: BackgroundTasks):
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")

    try:
        img_prompt = await enhance_image_prompt(req.prompt)
    except Exception as e:
        raise HTTPException(502, f"Prompt enhancement failed: {e}")

    storage.update_scene(pid, scene_idx,
                         raw_prompt=req.prompt,
                         enhanced_image_prompt=img_prompt,
                         status="generating_image",
                         error=None)
    bg.add_task(_gen_source_image, pid, scene_idx, img_prompt)
    return {"status": "generating_image"}


# ── Clip: read ────────────────────────────────────────────────────────────────
@app.get("/projects/{pid}/scenes/{scene_idx}/clips/{clip_idx}")
async def get_clip(pid: str, scene_idx: int, clip_idx: int):
    """
    Returns the current status and URL of a specific clip.
    Also exposes the prompt that was used (or will be used) for this clip.
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")
    if clip_idx < 0 or clip_idx >= settings.CLIPS_PER_SCENE:
        raise HTTPException(400, f"clip_idx must be 0–{settings.CLIPS_PER_SCENE - 1}")

    clip_path = storage.clip_path(pid, scene_idx, clip_idx)
    clip_meta = storage.get_clip_meta(pid, scene_idx, clip_idx)
    prompts   = scene.get("enhanced_prompts", [])

    return {
        "clip_idx":  clip_idx,
        "scene_idx": scene_idx,
        "project_id": pid,
        "status":    clip_meta.get("status", "unknown"),
        "error":     clip_meta.get("error"),
        "prompt":    prompts[clip_idx] if clip_idx < len(prompts) else None,
        "clip_url":  storage.rel_url(clip_path) if clip_path.exists() else None,
    }


# ── Clip: regenerate (post-scene-completion) ──────────────────────────────────
@app.post("/projects/{pid}/scenes/{scene_idx}/clips/{clip_idx}/regenerate")
async def regenerate_clip(
    pid: str,
    scene_idx: int,
    clip_idx: int,
    req: ClipRegenerateReq,
    bg: BackgroundTasks,
):
    """
    Regenerates a single clip AFTER the scene has been fully generated.

    Allowed when scene status is "done" or "error".
    If req.prompt is supplied it overrides enhanced_prompts[clip_idx] and
    is persisted so downstream clips re-chain correctly.

    What happens in the background:
      1. Regenerate clip_{clip_idx} using:
           - seed = clip_{clip_idx - 1}_last_frame.png  (or source_image for clip 0)
           - prompt = req.prompt (if given) else enhanced_prompts[clip_idx]
      2. Extract last frame of new clip_{clip_idx} → clip_{clip_idx}_last_frame.png
      3. Re-generate clip_{clip_idx+1} … clip_4 in sequence (re-chain), each
         seeded by the previous clip's new last frame, using their current prompts.
      4. Re-assemble scene.mp4 from all 5 clips.
      5. scene status returns to "done".
    """
    scene = storage.get_scene(pid, scene_idx)
    if not scene:
        raise HTTPException(404, "Scene not found")

    if scene["status"] not in ("done", "error"):
        raise HTTPException(
            400,
            f"Clip regeneration is only allowed when scene status is 'done' or 'error'. "
            f"Current status: '{scene['status']}'",
        )

    if clip_idx < 0 or clip_idx >= settings.CLIPS_PER_SCENE:
        raise HTTPException(400, f"clip_idx must be 0–{settings.CLIPS_PER_SCENE - 1}")

    # If a new prompt was supplied, persist it immediately
    if req.prompt:
        try:
            # Optionally enhance the raw override prompt through Groq for quality
            # (We send it through the video enhancer for a single clip prompt)
            enhanced_override = await _enhance_single_clip_prompt(req.prompt)
        except Exception:
            # Fallback: use the raw prompt as-is if enhancement fails
            enhanced_override = req.prompt

        prompts = list(scene.get("enhanced_prompts", []))
        if clip_idx < len(prompts):
            prompts[clip_idx] = enhanced_override
        else:
            while len(prompts) <= clip_idx:
                prompts.append(enhanced_override)
        storage.update_scene(pid, scene_idx, enhanced_prompts=prompts, error=None)

    # Mark scene as regenerating so frontend knows something is happening
    storage.update_scene(pid, scene_idx, status="regenerating_clip", error=None)
    storage.update_clip_meta(pid, scene_idx, clip_idx, status="generating", error=None)

    bg.add_task(_regen_clip_chain, pid, scene_idx, clip_idx)
    return {
        "status": "regenerating_clip",
        "clip_idx": clip_idx,
        "message": (
            f"Regenerating clip {clip_idx} and re-chaining "
            f"clips {clip_idx}–{settings.CLIPS_PER_SCENE - 1}, "
            f"then reassembling scene."
        ),
    }


# ── Finalize ──────────────────────────────────────────────────────────────────
@app.post("/projects/{pid}/finalize")
async def finalize_project(pid: str, bg: BackgroundTasks):
    project = storage.get_project(pid)
    if not project:
        raise HTTPException(404, "Project not found")

    done = [s for s in project["scenes"]
            if s["status"] == "done" and s.get("scene_video")]
    if not done:
        raise HTTPException(400, "No completed scenes to finalize")

    storage.update_project(pid, final_status="processing")
    bg.add_task(_finalize, pid, done)
    return {"status": "processing", "scene_count": len(done)}


@app.get("/projects/{pid}/final-video")
async def download_final(pid: str):
    p = storage.get_project(pid)
    if not p:
        raise HTTPException(404, "Project not found")
    if not p.get("final_video"):
        raise HTTPException(404, "Final video not ready yet")
    return FileResponse(
        p["final_video"],
        media_type="video/mp4",
        filename=f"{p['title'].replace(' ', '_')}_final.mp4",
    )


# ── Background tasks ──────────────────────────────────────────────────────────

async def _gen_source_image(pid: str, scene_idx: int, image_prompt: str):
    """
    Runs the txt2img workflow on ComfyUI.
    On success → status becomes 'confirming' so the UI can show the image.
    """
    try:
        workflow  = build_image_workflow(image_prompt)
        dest_path = storage.source_image_path(pid, scene_idx)
        await comfy.run_job(workflow, dest_path, timeout=1000)
        storage.update_scene(pid, scene_idx,
                             source_image=str(dest_path),
                             status="confirming")
    except Exception as e:
        storage.update_scene(pid, scene_idx, status="error", error=str(e))


async def _gen_clips(pid: str, scene_idx: int):
    """
    Generates 5 sequential 4-second clips for one scene.

    Clip chain:
      clip_0 ← source_image                    + enhanced_prompts[0]
      clip_1 ← clip_0_last_frame.png            + enhanced_prompts[1]
      clip_2 ← clip_1_last_frame.png            + enhanced_prompts[2]
      clip_3 ← clip_2_last_frame.png            + enhanced_prompts[3]
      clip_4 ← clip_3_last_frame.png            + enhanced_prompts[4]

    Each clip's last frame is stored separately as clip_{i}_last_frame.png
    so that a later regeneration of clip_N can correctly re-seed clip_N+1.

    FIX: extract_last_frame and assemble_scene use subprocess.run (blocking).
    Both are wrapped in asyncio.to_thread() to avoid freezing the event loop.
    """
    try:
        scene      = storage.get_scene(pid, scene_idx)
        prompts    = scene["enhanced_prompts"]
        seed_image = Path(scene["source_image"])

        # Initialise clip list to the right length
        storage.update_scene(pid, scene_idx, clips=[None] * settings.CLIPS_PER_SCENE)

        for i in range(settings.CLIPS_PER_SCENE):
            storage.update_clip_meta(pid, scene_idx, i, status="generating", error=None)
            clip_path = storage.clip_path(pid, scene_idx, i)
            workflow  = build_video_workflow(prompts[i])

            await comfy.run_job(
                workflow,
                dest_path=clip_path,
                timeout=10000,
                img_to_upload=seed_image,
            )

            storage.set_clip(pid, scene_idx, i, str(clip_path))
            storage.update_scene(pid, scene_idx, last_completed_clip=i)
            storage.update_clip_meta(pid, scene_idx, i, status="done", error=None)

            # Extract and SAVE last frame of this clip for downstream seeding
            clip_last_frame = storage.clip_last_frame_path(pid, scene_idx, i)
            await asyncio.to_thread(extract_last_frame, clip_path, clip_last_frame)

            # Use this clip's last frame as the seed for the next clip
            if i < settings.CLIPS_PER_SCENE - 1:
                seed_image = clip_last_frame

        # Assemble 5 clips → 20s scene.mp4
        clip_paths  = [storage.clip_path(pid, scene_idx, i)
                       for i in range(settings.CLIPS_PER_SCENE)]
        scene_video = storage.scene_video_path(pid, scene_idx)
        await asyncio.to_thread(assemble_scene, clip_paths, scene_video)

        storage.update_scene(pid, scene_idx,
                             scene_video=str(scene_video),
                             status="done")
    except Exception as e:
        storage.update_scene(pid, scene_idx, status="error", error=str(e))


async def _regen_clip_chain(pid: str, scene_idx: int, start_clip_idx: int):
    """
    Regenerates clip_{start_clip_idx} and all downstream clips, then reassembles.

    Seed resolution for each clip:
      - clip_0   → source_image.png
      - clip_N   → clip_{N-1}_last_frame.png   (the persisted per-clip last frame)

    This means regenerating clip_2 will:
      - use clip_1_last_frame.png as seed for the new clip_2
      - extract new clip_2_last_frame.png
      - regenerate clip_3 seeded by new clip_2_last_frame.png
      - extract new clip_3_last_frame.png
      - regenerate clip_4 seeded by new clip_3_last_frame.png
      - reassemble scene.mp4
    """
    try:
        scene   = storage.get_scene(pid, scene_idx)
        prompts = scene["enhanced_prompts"]

        # Determine the seed for start_clip_idx
        if start_clip_idx == 0:
            seed_image = Path(scene["source_image"])
        else:
            seed_image = storage.clip_last_frame_path(pid, scene_idx, start_clip_idx - 1)
            if not seed_image.exists():
                raise FileNotFoundError(
                    f"Seed image not found for clip {start_clip_idx}: {seed_image}. "
                    f"Ensure the previous clip's last frame was extracted during initial generation."
                )

        # Re-generate from start_clip_idx to the last clip
        for i in range(start_clip_idx, settings.CLIPS_PER_SCENE):
            storage.update_clip_meta(pid, scene_idx, i, status="generating", error=None)
            clip_path = storage.clip_path(pid, scene_idx, i)
            workflow  = build_video_workflow(prompts[i])

            await comfy.run_job(
                workflow,
                dest_path=clip_path,
                timeout=10000,
                img_to_upload=seed_image,
            )

            storage.set_clip(pid, scene_idx, i, str(clip_path))
            storage.update_scene(pid, scene_idx, last_completed_clip=i)
            storage.update_clip_meta(pid, scene_idx, i, status="done", error=None)

            # Extract and save the new last frame for downstream chaining
            clip_last_frame = storage.clip_last_frame_path(pid, scene_idx, i)
            await asyncio.to_thread(extract_last_frame, clip_path, clip_last_frame)

            # Feed into next clip
            if i < settings.CLIPS_PER_SCENE - 1:
                seed_image = clip_last_frame

        # Re-assemble scene.mp4 from all 5 (possibly mixed old+new) clips
        clip_paths  = [storage.clip_path(pid, scene_idx, i)
                       for i in range(settings.CLIPS_PER_SCENE)]
        scene_video = storage.scene_video_path(pid, scene_idx)
        await asyncio.to_thread(assemble_scene, clip_paths, scene_video)

        storage.update_scene(pid, scene_idx,
                             scene_video=str(scene_video),
                             status="done",
                             error=None)

    except Exception as e:
        # Mark scene error but also tag which clip failed
        storage.update_clip_meta(pid, scene_idx, start_clip_idx, status="error", error=str(e))
        storage.update_scene(pid, scene_idx, status="error", error=str(e))


async def _finalize(pid: str, done_scenes: list):
    """
    Concatenates all done scene videos with lightning-flash transitions.

    FIX: concatenate_scenes uses subprocess.run (blocking).
    Wrapped in asyncio.to_thread() to avoid freezing the event loop.
    """
    try:
        scene_paths = [Path(s["scene_video"]) for s in done_scenes]
        out_path    = storage.final_video_path(pid)
        await asyncio.to_thread(concatenate_scenes, scene_paths, out_path)

        storage.update_project(pid,
                               final_video=str(out_path),
                               final_video_url=storage.rel_url(out_path),
                               final_status="done")
    except Exception as e:
        storage.update_project(pid, final_status="error", final_error=str(e))


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _enhance_single_clip_prompt(raw_prompt: str) -> str:
    """
    Enhances a single clip description through Groq for one clip override.
    Uses the same video system prompt logic but asks for just ONE prompt string.
    We call enhance_prompt (returns 5) and take the first result,
    or fall back to a simplified inline call.
    """
    import httpx
    from config import settings as cfg

    system = """\
You are a cinematic AI video director and prompt engineer.
Expand the user's short scene description into ONE detailed video prompt
for an img2vid model (LTX-Video 2.3) describing a 4-second clip.
Include: subject action, camera movement, lighting, atmosphere.
Be specific and vivid. Respond ONLY with the prompt string — no preamble,
no quotes, no markdown.
"""
    payload = {
        "model": cfg.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": raw_prompt},
        ],
        "temperature": 0.8,
        "max_tokens":  400,
    }
    headers = {
        "Authorization": f"Bearer {cfg.GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'")
