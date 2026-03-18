from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import os
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Optional
import logging
from datetime import datetime

from diffusers import StableAudioPipeline
import edge_tts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI(
    title="AI Audio Generation API",
    description="Generate dialogue audio with ambient music using Stable Audio",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TARGET_SECONDS = 4.0
PAUSE_SECONDS = 0.4
SR_TARGET = 24000

VOICES = {
    "person1": "en-GB-RyanNeural",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    TORCH_DTYPE = torch.float16
    logger.info(f"✓ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    logger.info(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info(f"✓ Using float16 (memory efficient)")
else:
    TORCH_DTYPE = torch.float32
    logger.info(f"⚠ CPU DETECTED (will be slow!)")
    logger.info(f"✓ Using float32")

logger.info(f"Device: {DEVICE.upper()}")


class DialogueLine(BaseModel):
    speaker: str = Field(..., description="Speaker identifier")
    text: str = Field(..., description="Dialogue text")

class AudioGenerationRequest(BaseModel):
    ambient_prompt: str = Field(...)
    dialogue: list[DialogueLine] = Field(default=[])
    negative_prompt: str = Field(default="music, noise, distortion, low quality")
    num_inference_steps: int = Field(default=50, ge=20, le=100)

class AudioGenerationResponse(BaseModel):
    success: bool
    message: str
    duration: Optional[float] = None
    file_url: Optional[str] = None
    error: Optional[str] = None


logger.info("=" * 70)
logger.info("Loading Stable Audio Model...")
logger.info("RTX 3050: First time = 10-20 minutes (model swapping to disk)")
logger.info("Next time: 1-2 minutes (cached)")
logger.info("=" * 70)

audio_pipeline = None

try:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set! Check .env file!")
    
    logger.info("Downloading/Loading from Hugging Face...")
    
    audio_pipeline = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=TORCH_DTYPE,
        token=hf_token
    )
    
    # Move to device
    audio_pipeline = audio_pipeline.to(DEVICE)
    
    # Memory optimizations
    if DEVICE == "cuda":
        try:
            audio_pipeline.enable_attention_slicing()
            audio_pipeline.enable_model_cpu_offload()
            logger.info("✓ GPU memory optimizations enabled")
        except Exception as e:
            logger.warning(f"Could not enable all optimizations: {e}")
        
        torch.cuda.empty_cache()
    
    logger.info("=" * 70)
    logger.info("✓✓✓ STABLE AUDIO LOADED SUCCESSFULLY! ✓✓✓")
    logger.info("=" * 70)
    
except Exception as e:
    logger.error(f"FAILED TO LOAD: {str(e)}")
    logger.error("Make sure HF_TOKEN is in .env file!")
    audio_pipeline = None

def generate_ambient_audio(prompt: str, negative_prompt: str, num_steps: int = 50) -> np.ndarray:
    if audio_pipeline is None:
        raise RuntimeError("Audio pipeline not loaded!")
    
    logger.info(f"Generating ambient: '{prompt}'")
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        result = audio_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            audio_end_in_s=4,
            generator=torch.Generator(device=DEVICE).manual_seed(42)
        )
    
    audio_tensor = result.audios[0]
    audio_np = audio_tensor.T.float().cpu().numpy()
    
    # Convert to mono if stereo
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=1)
    
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    logger.info(f"✓ Ambient audio generated: {len(audio_np)} samples (mono)")
    return audio_np

async def generate_tts_line(text: str, voice: str) -> np.ndarray:
    logger.info(f"Generating TTS: '{text[:40]}...'")
    
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        tts = edge_tts.Communicate(text=text, voice=voice)
        await tts.save(temp_path)
        
        # Use soundfile instead of librosa
        audio, sr = sf.read(temp_path)
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:  # If stereo/multi-channel
            audio = np.mean(audio, axis=1)  # Average channels to mono
        
        # Resample if needed (simple numpy interpolation)
        if sr != SR_TARGET:
            num_samples = int(len(audio) * SR_TARGET / sr)
            old_indices = np.linspace(0, len(audio) - 1, len(audio))
            new_indices = np.linspace(0, len(audio) - 1, num_samples)
            audio = np.interp(new_indices, old_indices, audio)
        
        logger.info(f"✓ TTS generated: {len(audio)} samples (mono)")
        return audio
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "AI Audio Generation API",
        "device": DEVICE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate-audio")
async def generate_audio(request: AudioGenerationRequest) -> AudioGenerationResponse:
    try:
        logger.info(f"=== NEW REQUEST: {len(request.dialogue)} dialogue lines ===")
        
        # Generate ambient
        logger.info("STEP 1: Generating ambient audio...")
        ambient = generate_ambient_audio(
            request.ambient_prompt,
            request.negative_prompt,
            request.num_inference_steps
        )
        ambient_sr = SR_TARGET
        
        if len(request.dialogue) == 0:
            final_audio = ambient
            final_sr = ambient_sr
            duration = len(final_audio) / final_sr
        
        else:
            logger.info(f"STEP 2: Generating dialogue ({len(request.dialogue)} lines)...")
            
            all_speech = []
            try:
                for i, line in enumerate(request.dialogue):
                    logger.info(f"Line {i+1}: speaker='{line.speaker}', text='{line.text[:30]}...'")
                    
                    if line.speaker not in VOICES:
                        raise ValueError(f"Unknown speaker: {line.speaker}. Must be: {list(VOICES.keys())}")
                    
                    voice = VOICES[line.speaker]
                    logger.info(f"Generating TTS with voice: {voice}")
                    
                    speech_audio = await generate_tts_line(line.text, voice)
                    logger.info(f"✓ TTS audio: {len(speech_audio)} samples = {len(speech_audio)/SR_TARGET:.2f}s")
                    
                    all_speech.append(speech_audio)
            except Exception as e:
                logger.error(f"ERROR in dialogue generation: {str(e)}", exc_info=True)
                raise
            
            logger.info("STEP 3: Concatenating all speeches (NO silence between)...")
            full_dialogue = np.concatenate(all_speech, axis=0)
            raw_duration = len(full_dialogue) / SR_TARGET
            logger.info(f"Total: {raw_duration:.2f}s, shape: {full_dialogue.shape}")
            
            logger.info("STEP 4: Time-stretching to 30 seconds (keeps quality, slows speech)...")
            stretch_factor = raw_duration / TARGET_SECONDS
            logger.info(f"Stretch factor: {stretch_factor:.3f}x (slower)")
            
            try:
                # Use librosa time-stretch (proper phase vocoder)
                dialogue_stretched = librosa.effects.time_stretch(full_dialogue, rate=stretch_factor)
                logger.info(f"✓ Stretched: {len(dialogue_stretched)} samples = {len(dialogue_stretched)/SR_TARGET:.2f}s")
            except Exception as e:
                logger.warning(f"Librosa stretch failed ({e}), using interpolation...")
                target_len = int(TARGET_SECONDS * SR_TARGET)
                old_indices = np.linspace(0, len(full_dialogue) - 1, len(full_dialogue))
                new_indices = np.linspace(0, len(full_dialogue) - 1, target_len)
                dialogue_stretched = np.interp(new_indices, old_indices, full_dialogue)
            
            logger.info(f"Dialogue after stretch: level max={np.max(np.abs(dialogue_stretched)):.4f}")
            
            logger.info("STEP 5: Matching lengths...")
            if len(ambient) > len(dialogue_stretched):
                ambient = ambient[:len(dialogue_stretched)]
            else:
                ambient = np.pad(ambient, (0, len(dialogue_stretched) - len(ambient)), mode='constant')
            
            logger.info("STEP 6: Mixing dialogue + ambient (no gaps)...")
            # DIALOGUE LOUD, AMBIENT SOFT
            final_audio = dialogue_stretched * 1.0 + ambient * 0.3
            
            logger.info(f"Before normalization - max: {np.abs(final_audio).max()}")
            
            max_val = np.abs(final_audio).max()
            if max_val > 0:
                final_audio = final_audio / max_val
            
            logger.info(f"After normalization - max: {np.abs(final_audio).max()}")
            
            final_sr = SR_TARGET
            duration = len(final_audio) / final_sr
            logger.info(f"Final audio: {duration:.2f}s, shape={final_audio.shape}")
        
        logger.info("STEP 7: Saving...")
        
        # Create audio_outputs folder in PROJECT directory
        output_folder = os.path.join(os.path.dirname(__file__), "audio_outputs")
        os.makedirs(output_folder, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_folder, f"audio_{timestamp}.wav")
        
        sf.write(output_path, final_audio, final_sr)
        logger.info(f"✓ Audio saved: {output_path}")
        
        return AudioGenerationResponse(
            success=True,
            message=f"Audio generated successfully ({duration:.2f}s)",
            duration=duration,
            file_url=output_path,
            error=None
        )
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return AudioGenerationResponse(
            success=False,
            message="Failed to generate audio",
            duration=None,
            file_url=None,
            error=error_msg
        )

@app.get("/download-audio")
async def download_audio(filepath: str):
    try:
        # Allow files from audio_outputs folder
        if "audio_outputs" not in filepath:
            raise ValueError("Invalid path")
        if not os.path.exists(filepath):
            raise FileNotFoundError("File not found")
        return FileResponse(filepath, media_type="audio/wav", filename=os.path.basename(filepath))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": DEVICE, "pipeline_loaded": audio_pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
