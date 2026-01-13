import os
import warnings
import huggingface_hub as _hfh
import tempfile
import threading
import time
from collections import deque
import numpy as np

# ------------------------------------------------------------------
# Shim for older Gradio versions...
# (Keep your existing HfFolder shim and Gradio patches unchanged)
# ------------------------------------------------------------------

import gradio as gr
from transformers import pipeline
import torch

# New Gemini SDK
from google import genai
from google.genai import types

warnings.filterwarnings("ignore")

# ==================== VAD SETUP ====================
# Try multiple VAD options with fallbacks
VAD_AVAILABLE = False
vad_model = None
get_speech_timestamps = None
read_audio = None

# Option 1: Try Silero VAD (PyTorch version, no ONNX)
try:
    import torchaudio
    # Try loading Silero VAD without ONNX to avoid onnxruntime dependency
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False  # Use PyTorch, not ONNX
    )
    (get_speech_timestamps, _, read_audio, _, _) = vad_utils
    VAD_AVAILABLE = True
    print("✅ Silero VAD (PyTorch) loaded successfully!")
except Exception as e1:
    print(f"⚠️  Silero VAD (PyTorch) failed: {e1}")
    
    # Option 2: Try simple energy-based VAD as fallback
    try:
        import librosa
        VAD_AVAILABLE = True
        print("✅ Using energy-based VAD fallback (librosa)")
        
        def simple_vad(audio_path, frame_length=2048, hop_length=512, energy_threshold=0.01):
            """Simple energy-based VAD using librosa"""
            y, sr = librosa.load(audio_path, sr=16000)
            # Calculate frame energy
            frame_energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            # Detect speech frames (energy above threshold)
            speech_frames = frame_energy > energy_threshold
            # Convert to timestamps
            timestamps = []
            in_speech = False
            start = None
            
            for i, is_speech in enumerate(speech_frames):
                time_ms = (i * hop_length / sr) * 1000
                if is_speech and not in_speech:
                    start = time_ms
                    in_speech = True
                elif not is_speech and in_speech:
                    timestamps.append({'start': start, 'end': time_ms})
                    in_speech = False
            
            if in_speech:
                timestamps.append({'start': start, 'end': len(y) / sr * 1000})
            
            return timestamps, y, sr
        
        get_speech_timestamps = lambda wav, model, **kwargs: simple_vad(None)[0] if isinstance(wav, str) else []
        read_audio = lambda path, sr: (librosa.load(path, sr=sr)[0], sr)
        
    except Exception as e2:
        print(f"⚠️  Energy-based VAD fallback failed: {e2}")
        VAD_AVAILABLE = False
        print("⚠️  VAD disabled - will transcribe full audio files")

# ==================== STREAMING STATE ====================
# Global state for streaming transcription
streaming_state = {
    "active": False,
    "processed_chunks": [],  # List of processed chunks with their transcriptions
    "accumulated_text": "",  # All cleaned chunks accumulated
    "processed_files": set(),  # Track processed audio files
    "lock": threading.Lock(),
    "current_audio_file": None,  # Current audio file being processed
    "chunk_counter": 0,
    "audio_buffer": [],  # Buffer for streaming audio chunks
    "last_process_time": 0
}

# ==================== LOAD MEDASR MODEL ====================
print("Loading Google MedASR model (google/medasr)...")

# Detect device (GPU if available)
device =-1

medasr = pipeline(
    "automatic-speech-recognition",
    model="google/medasr",
    device=device,  # GPU or CPU
)

print("MedASR model loaded successfully!")

# ==================== GEMINI SETUP (NEW SDK) ====================
print("Initializing Gemini model...")

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "AIzaSyDP-QvZ8zqYz2t_9OqDcjm1W_RQCPFvUzk"

if api_key:
    # New client-based initialization
    gemini_client = genai.Client(api_key=api_key)
    gemini_model_name = "gemini-2.0-flash-lite"  # or "gemini-1.5-flash" etc.
    print("Gemini model initialized successfully!")
else:
    print("Warning: GEMINI_API_KEY or GOOGLE_API_KEY not found. Gemini formatting disabled.")
    gemini_client = None
    gemini_model_name = None

def cleanup_chunk_with_gemini(text: str) -> str:
    """
    Cleanup a single chunk with Gemini - ONLY spelling, remove pauses, remove duplicates
    NO sentence generation, NO word addition
    """
    if not gemini_client:
        return text

    try:
        prompt = f"""Clean up this transcription chunk. You can ONLY:
1. Fix spelling errors
2. Remove pause sounds (um, uh, er, ah, etc.)
3. Remove duplicate words/phrases
4. Remove filler words (like, you know, etc.)

CRITICAL RULES:
- DO NOT add any words not in the text
- DO NOT create sentences
- DO NOT add punctuation unless it's clearly missing
- DO NOT change medical terms
- ONLY clean up what's there

Input text:
{text}

Return ONLY the cleaned text:"""

        response = gemini_client.models.generate_content(
            model=gemini_model_name,
            contents=prompt
        )
        return response.text.strip()

    except Exception as e:
        return text  # Return original if cleanup fails

def merge_final_text_with_gemini(accumulated_text: str) -> str:
    """
    Final merge: Take all accumulated cleaned chunks and make one coherent document
    """
    if not gemini_client:
        return accumulated_text

    try:
        prompt = f"""Merge these transcription chunks into one coherent medical report.

The text below is from multiple 6-second audio chunks that have been cleaned.
Your job is to:
1. Merge them into one flowing document
2. Fix any remaining formatting issues
3. Ensure proper sentence structure
4. Apply medical abbreviations (R, L, R > L, w/, w/o, ↑, ↓, Δ)
5. Convert "fill stop" → period, "next line" → line break
6. Fix capitalization and punctuation

CRITICAL: Use ONLY the words from the input. Do NOT add new clinical information.

Input chunks (merge these):
{accumulated_text}

Return the final merged and formatted medical report:"""

        response = gemini_client.models.generate_content(
            model=gemini_model_name,
            contents=prompt
        )
        return response.text.strip()

    except Exception as e:
        return accumulated_text  # Return original if merge fails

def transcribe_audio_file(audio_path, use_gemini_formatting=True):
    """
    Transcribe a complete audio file with MedASR + optional Gemini formatting
    """
    if not audio_path or not os.path.exists(audio_path):
        return "Invalid audio file path."

    try:
        # Transcribe with MedASR
        result = medasr(audio_path)
        raw_transcription = result["text"].strip()
        
        # Clean up MedASR output - remove special tokens
        raw_transcription = raw_transcription.replace("</s>", "").replace("<s>", "").strip()
        # Remove any other common ASR artifacts
        raw_transcription = raw_transcription.replace("  ", " ")  # Multiple spaces

        if use_gemini_formatting and gemini_client:
            # Use final merge function for formatting
            formatted_text = merge_final_text_with_gemini(raw_transcription)
            output = f"""**📝 Raw Transcription (Google MedASR):**
{raw_transcription}

**✨ Formatted Radiology Report (Gemini):**
{formatted_text}"""
        else:
            output = f"""**📝 Transcription (Google MedASR):**
{raw_transcription}"""

        return output

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"

def transcribe_audio(audio, use_gemini_formatting=True):
    """
    Transcribe with MedASR + optional Gemini formatting
    Handles both file uploads and streaming chunks
    """
    if audio is None:
        return "No audio provided. Please record or upload an audio file."

    try:
        # Gradio Audio(type="filepath") gives a file path string
        if isinstance(audio, tuple):
            _, data = audio
            audio_path = data if isinstance(data, str) else None
        else:
            audio_path = audio if isinstance(audio, str) else None

        return transcribe_audio_file(audio_path, use_gemini_formatting)

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"


def process_chunks_with_updates(audio_path, use_gemini_formatting=True):
    """
    Process audio in 6-second chunks and yield updates for real-time display
    """
    import librosa
    import soundfile as sf
    
    print(f"🔊 Loading audio from: {audio_path}")
    full_audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(full_audio) / sr
    chunk_duration = 6.0  # 6 seconds
    num_chunks = int(np.ceil(duration / chunk_duration))
    
    print(f"📊 Audio loaded: {duration:.1f}s, {num_chunks} chunks")
    
    accumulated_cleaned_chunks = []
    all_chunks_display = []
    
    # Process each 6-second chunk
    for i in range(num_chunks):
        start_time = i * chunk_duration
        end_time = min((i + 1) * chunk_duration, duration)
        
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        chunk_audio = full_audio[start_sample:end_sample]
        
        # Skip empty chunks
        if len(chunk_audio) < sr * 0.5:
            continue
        
        # Save chunk to temp file for MedASR
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            sf.write(tmp.name, chunk_audio, sr)
            chunk_path = tmp.name
        
        try:
            # Transcribe chunk with MedASR
            result = medasr(chunk_path)
            chunk_text = result["text"].strip()
            
            # Clean up MedASR output
            chunk_text = chunk_text.replace("</s>", "").replace("<s>", "").strip()
            chunk_text = chunk_text.replace("  ", " ").strip()
            
            if chunk_text:
                # Clean chunk with Gemini (spelling, pauses, duplicates)
                if use_gemini_formatting and gemini_client:
                    cleaned_chunk = cleanup_chunk_with_gemini(chunk_text)
                else:
                    cleaned_chunk = chunk_text
                
                accumulated_cleaned_chunks.append(cleaned_chunk)
                
                # Create display for this chunk
                chunk_display = f"**Chunk {i+1}** [{start_time:.1f}s - {end_time:.1f}s]:\n{cleaned_chunk}"
                all_chunks_display.append(chunk_display)
                
                # Yield current accumulated text for real-time display
                current_accumulated = " ".join(accumulated_cleaned_chunks)
                
                yield f"""🔴 **Processing Chunk {i+1}/{num_chunks}** ({start_time:.1f}s - {end_time:.1f}s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 **LATEST CHUNK:**
{cleaned_chunk}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **ALL CHUNKS SO FAR:**

{chr(10).join(all_chunks_display)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ **ACCUMULATED TEXT (Combined):**
{current_accumulated}"""
                
                # Small delay to make streaming visible
                time.sleep(0.1)
        
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
        
        finally:
            try:
                os.unlink(chunk_path)
            except:
                pass
    
    # Final merge with Gemini
    accumulated_text = " ".join(accumulated_cleaned_chunks)
    
    with streaming_state["lock"]:
        streaming_state["accumulated_text"] = accumulated_text
        streaming_state["processed_chunks"] = all_chunks_display
    
    if use_gemini_formatting and gemini_client and accumulated_text:
        final_merged = merge_final_text_with_gemini(accumulated_text)
        yield f"""✅ **ALL {num_chunks} CHUNKS PROCESSED!**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 **EACH 6-SECOND CHUNK (As Processed):**

{chr(10).join(all_chunks_display)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧹 **ACCUMULATED CLEANED TEXT:**
{accumulated_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ **FINAL MERGED REPORT (One Coherent Document):**

{final_merged}"""
    else:
        yield f"""✅ **ALL {num_chunks} CHUNKS PROCESSED!**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 **EACH 6-SECOND CHUNK:**

{chr(10).join(all_chunks_display)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**FINAL TRANSCRIPTION:**
{accumulated_text}"""

def transcribe_recorded_audio_streaming(audio, use_gemini_formatting=True):
    """
    Generator function that yields updates as each 6-second chunk is processed
    This enables real-time streaming updates in Gradio
    """
    print(f"🎤 transcribe_recorded_audio_streaming called with audio: {audio}")
    
    if audio is None:
        yield "⚠️ No audio provided. Please record or upload an audio file."
        return

    try:
        # Get audio file path
        if isinstance(audio, tuple):
            _, data = audio
            audio_path = data if isinstance(data, str) else None
        else:
            audio_path = audio if isinstance(audio, str) else None

        print(f"📁 Audio path: {audio_path}")
        
        if not audio_path or not os.path.exists(audio_path):
            yield f"❌ Invalid audio file path: {audio_path}"
            return
        
        yield "🔄 Starting transcription... Loading audio..."
        
        # Reset state for new file
        with streaming_state["lock"]:
            if audio_path != streaming_state.get("current_audio_file"):
                streaming_state["current_audio_file"] = audio_path
                streaming_state["processed_chunks"] = []
                streaming_state["accumulated_text"] = ""
                streaming_state["chunk_counter"] = 0
        
        # Process chunks and yield updates in real-time
        for update in process_chunks_with_updates(audio_path, use_gemini_formatting):
            print(f"📤 Yielding update...")
            yield update
            # Store for final access
            with streaming_state["lock"]:
                streaming_state["accumulated_text"] = update

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        yield error_msg

def transcribe_recorded_audio(audio, use_gemini_formatting=True):
    """
    Wrapper that processes audio and returns final result
    For streaming, use the generator version
    """
    if audio is None:
        return "No audio provided. Please record or upload an audio file."

    try:
        # Get audio file path
        if isinstance(audio, tuple):
            _, data = audio
            audio_path = data if isinstance(data, str) else None
        else:
            audio_path = audio if isinstance(audio, str) else None

        if not audio_path or not os.path.exists(audio_path):
            return "Invalid audio file path."
        
        # Get final result from generator
        final_result = None
        for update in transcribe_recorded_audio_streaming(audio, use_gemini_formatting):
            final_result = update
        
        return final_result if final_result else "Processing..."

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"

def detect_pause_with_vad(audio_data, sample_rate, pause_threshold_ms=800):
    """
    Detect if there's a pause/silence at the end of audio using VAD
    Returns True if pause detected (sentence likely complete)
    """
    import librosa
    
    try:
        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_float = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio_float = audio_data.astype(np.float32)
        
        # Resample to 16kHz if needed (Silero VAD requires 16kHz)
        if sample_rate != 16000:
            audio_float = librosa.resample(audio_float, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Use Silero VAD if available
        if VAD_AVAILABLE and vad_model is not None:
            try:
                # Get speech timestamps
                audio_tensor = torch.from_numpy(audio_float)
                speech_timestamps = get_speech_timestamps(
                    audio_tensor, 
                    vad_model,
                    sampling_rate=sample_rate,
                    threshold=0.5
                )
                
                # Check if there's silence at the end
                if len(speech_timestamps) > 0:
                    last_speech_end = speech_timestamps[-1]['end']
                    audio_end = len(audio_float)
                    
                    # Calculate silence duration at the end
                    silence_samples = audio_end - last_speech_end
                    silence_ms = (silence_samples / sample_rate) * 1000
                    
                    print(f"🔍 VAD: Last speech ended at {last_speech_end}/{audio_end} samples, silence: {silence_ms:.0f}ms")
                    
                    # If silence > threshold, it's a pause
                    return silence_ms >= pause_threshold_ms
                else:
                    # No speech detected in this chunk
                    return False
                    
            except Exception as e:
                print(f"⚠️ Silero VAD failed: {e}, falling back to energy-based")
        
        # Fallback: Energy-based silence detection
        # Check last 1 second of audio
        check_duration = min(1.0, len(audio_float) / sample_rate)
        check_samples = int(check_duration * sample_rate)
        last_segment = audio_float[-check_samples:]
        
        # Calculate RMS energy
        energy = np.sqrt(np.mean(last_segment ** 2))
        
        # Low energy = silence/pause
        silence_threshold = 0.01  # Adjust based on testing
        is_silent = energy < silence_threshold
        
        print(f"🔍 Energy-based: Last {check_duration:.1f}s energy={energy:.4f}, silent={is_silent}")
        
        return is_silent
        
    except Exception as e:
        print(f"⚠️ VAD detection error: {e}")
        return False

def process_streaming_audio(audio, use_gemini_formatting=True):
    """
    Process audio chunks as they come in during recording (every ~1 second from Gradio)
    Accumulate chunks and transcribe when pause detected (complete sentence)
    """
    import librosa
    import soundfile as sf
    
    if audio is None:
        return "🎤 Start recording to see real-time transcription..."
    
    try:
        # Get the sample rate and audio data
        sample_rate, audio_data = audio
        
        print(f"📥 Received audio chunk: {len(audio_data)} samples at {sample_rate}Hz")
        
        # Add to buffer
        with streaming_state["lock"]:
            streaming_state["audio_buffer"].append(audio_data)
            
            # Calculate total buffered duration
            total_samples = sum(len(chunk) for chunk in streaming_state["audio_buffer"])
            total_duration = total_samples / sample_rate
            
            print(f"📊 Buffer: {total_duration:.1f}s accumulated")
            
            # Concatenate current buffer for VAD check
            full_audio = np.concatenate(streaming_state["audio_buffer"])
            
            # Check for pause using VAD (sentence boundary detection)
            pause_detected = detect_pause_with_vad(full_audio, sample_rate, pause_threshold_ms=800)
            
            # Also check max duration to prevent too-long buffers
            max_duration_exceeded = total_duration >= 15.0
            
            # Require minimum duration before transcribing
            min_duration_met = total_duration >= 2.0
            
            should_transcribe = min_duration_met and (pause_detected or max_duration_exceeded)
            
            if should_transcribe:
                reason = "pause detected (sentence complete)" if pause_detected else "max duration (15s)"
                print(f"✅ Transcribing buffer: {reason}")
                
                # Convert to float32 and normalize if needed
                if full_audio.dtype == np.int16:
                    full_audio = full_audio.astype(np.float32) / 32768.0
                elif full_audio.dtype == np.int32:
                    full_audio = full_audio.astype(np.float32) / 2147483648.0
                
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    full_audio = librosa.resample(full_audio, orig_sr=sample_rate, target_sr=16000)
                
                # Save to temp file for MedASR
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    sf.write(tmp.name, full_audio, 16000)
                    chunk_path = tmp.name
                
                try:
                    # Transcribe with MedASR
                    result = medasr(chunk_path)
                    chunk_text = result["text"].strip()
                    
                    # Clean up MedASR output
                    chunk_text = chunk_text.replace("</s>", "").replace("<s>", "").strip()
                    chunk_text = chunk_text.replace("  ", " ").strip()
                    
                    if chunk_text:
                        # Clean chunk with Gemini
                        if use_gemini_formatting and gemini_client:
                            cleaned_chunk = cleanup_chunk_with_gemini(chunk_text)
                        else:
                            cleaned_chunk = chunk_text
                        
                        # Add to processed chunks
                        streaming_state["chunk_counter"] += 1
                        chunk_num = streaming_state["chunk_counter"]
                        chunk_display = f"**Sentence {chunk_num}** ({total_duration:.1f}s):\n{cleaned_chunk}"
                        streaming_state["processed_chunks"].append(chunk_display)
                        streaming_state["accumulated_text"] += " " + cleaned_chunk
                        
                        print(f"✅ Sentence {chunk_num} processed: {cleaned_chunk[:50]}...")
                        
                        # Clear buffer for next sentence
                        streaming_state["audio_buffer"] = []
                        
                        # Return current state
                        return f"""🔴 **RECORDING IN PROGRESS** - Sentence {chunk_num} transcribed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 **LATEST SENTENCE ({chunk_num}):**
{cleaned_chunk}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **ALL SENTENCES SO FAR:**

{chr(10).join(streaming_state["processed_chunks"])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ **ACCUMULATED TEXT:**
{streaming_state["accumulated_text"].strip()}"""
                    
                finally:
                    try:
                        os.unlink(chunk_path)
                    except:
                        pass
            
            # Still accumulating audio, waiting for pause
            chunk_count = streaming_state["chunk_counter"]
            if chunk_count == 0:
                return f"🎤 **RECORDING...** ({total_duration:.1f}s buffered) - Waiting for pause to detect sentence..."
            else:
                return f"""🔴 **RECORDING IN PROGRESS** - {chunk_count} sentences transcribed

📊 Buffer: {total_duration:.1f}s accumulated (waiting for pause...)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **ALL SENTENCES SO FAR:**

{chr(10).join(streaming_state["processed_chunks"])}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ **ACCUMULATED TEXT:**
{streaming_state["accumulated_text"].strip()}"""
    
    except Exception as e:
        import traceback
        error = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error)
        return error

def reset_streaming_state():
    """Reset streaming state when recording stops"""
    with streaming_state["lock"]:
        streaming_state["processed_chunks"] = []
        streaming_state["accumulated_text"] = ""
        streaming_state["active"] = False
        streaming_state["processed_files"] = set()
        streaming_state["audio_buffer"] = []
        streaming_state["chunk_counter"] = 0
        streaming_state["last_process_time"] = 0

# ==================== GRADIO INTERFACE ====================
with gr.Blocks(title="MedASR + Gemini Radiology Dictation") as demo:
    gr.Markdown(
        """
        # 🎤 Radiology Dictation → Professional Report (Sentence-by-Sentence)
        
        **Medical ASR** using **Google MedASR** with **VAD-based sentence detection**.
        
        **How it works:**
        1. Click **Record** and start speaking
        2. **VAD (Voice Activity Detection)** detects pauses between sentences
        3. Each complete sentence: **MedASR** transcribes → **Gemini** cleans
        4. Sentences appear in real-time as you speak!
        5. **Final merge**: Click button to combine all sentences into one coherent report
        
        **Processing Pipeline:**
        - **Pause Detection**: VAD detects sentence boundaries (800ms silence)
        - **MedASR**: Medical domain transcription (per sentence)
        - **Gemini Cleanup** (per sentence): Fix spelling, remove pauses, remove duplicates
        - **Gemini Final Merge**: Combine all sentences into one polished document
        
        **Important**: 
        - Accept MedASR license: https://huggingface.co/google/medasr
        - Set `GEMINI_API_KEY` for cleanup and formatting
        - **Pause briefly between sentences** for best results!
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",  # Changed to numpy for streaming
                label="Record or Upload Dictation",
                streaming=True  # Enable streaming for real-time chunks
            )
            use_gemini = gr.Checkbox(
                label="Use Gemini Formatting (recommended)",
                value=True
            )
            transcribe_btn = gr.Button("Final Merge →", variant="primary")
            reset_btn = gr.Button("Reset", variant="secondary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Result",
                lines=20,
                placeholder="🎤 Click record and start speaking! Each sentence will appear as you pause..."
            )

    # Real-time streaming: process audio chunks as they come in during recording
    audio_input.stream(
        fn=process_streaming_audio,
        inputs=[audio_input, use_gemini],
        outputs=output_text
    )
    
    # Final merge button: combine all sentences into one coherent report
    def finalize_transcription(use_gemini_formatting):
        with streaming_state["lock"]:
            accumulated = streaming_state["accumulated_text"].strip()
            sentences = streaming_state["processed_chunks"]
        
        if not accumulated:
            return "No transcription to merge. Record some audio first!"
        
        if use_gemini_formatting and gemini_client:
            final_merged = merge_final_text_with_gemini(accumulated)
            return f"""✅ **FINAL REPORT**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **ALL SENTENCES:**

{chr(10).join(sentences)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧹 **ACCUMULATED TEXT:**
{accumulated}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✨ **FINAL MERGED REPORT:**

{final_merged}"""
        else:
            return f"""✅ **FINAL TRANSCRIPTION**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 **ALL SENTENCES:**

{chr(10).join(sentences)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**FINAL TEXT:**
{accumulated}"""
    
    transcribe_btn.click(
        fn=finalize_transcription,
        inputs=[use_gemini],
        outputs=output_text
    )
    
    # Reset streaming state
    def reset_and_clear():
        reset_streaming_state()
        return ""
    
    reset_btn.click(
        fn=reset_and_clear,
        inputs=[],
        outputs=[output_text]
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
       footer_links=["gradio", "settings"]
    )