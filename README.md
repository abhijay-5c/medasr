# 🎤 Medical ASR - Real-Time Radiology Dictation System

A real-time medical dictation system that transcribes radiology reports using **Google MedASR** with intelligent **sentence-by-sentence processing** powered by **Voice Activity Detection (VAD)**.

## 🌟 Features

- **Real-Time Transcription**: See your dictation transcribed sentence-by-sentence as you speak
- **Intelligent Sentence Detection**: VAD-based pause detection for natural sentence boundaries
- **Medical Domain Accuracy**: Uses Google's MedASR model trained on medical terminology
- **AI-Powered Cleanup**: Gemini AI fixes spelling, removes filler words, and corrects medical terminology
- **Professional Formatting**: Automatic conversion to polished radiology reports with proper medical abbreviations
- **Streaming Architecture**: Process audio as you speak, not after you finish

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  1. User speaks into microphone                             │
│     ↓                                                        │
│  2. Gradio streams audio chunks (~1 second)                 │
│     ↓                                                        │
│  3. Buffer accumulates audio + VAD checks for pauses        │
│     ↓                                                        │
│  4. Pause detected (800ms silence) → Complete sentence!     │
│     ↓                                                        │
│  5. MedASR transcribes full sentence (with context)         │
│     ↓                                                        │
│  6. Gemini cleans up: spelling, pauses, duplicates          │
│     ↓                                                        │
│  7. Display cleaned sentence immediately                    │
│     ↓                                                        │
│  8. Repeat for each sentence...                             │
│     ↓                                                        │
│  9. Final merge: Gemini creates coherent medical report     │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

- **Python**: 3.9+
- **Hardware**: 
  - Microphone for real-time dictation
  - Recommended: 8GB+ RAM (for model loading)
  - Optional: GPU for faster inference (CPU works fine)

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install gradio transformers torch google-generativeai librosa soundfile numpy torchaudio
```

Or from requirements file:

```bash
pip install -r requirements.txt
```

### 2. Accept MedASR License

Visit [https://huggingface.co/google/medasr](https://huggingface.co/google/medasr) and accept the model license.

Login to Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login
```

### 3. Set Up API Keys

Create a `.env` file or set environment variables:

```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
# OR
export GOOGLE_API_KEY="your-google-api-key-here"
```

Get your Gemini API key from: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

## 🎯 Usage

### Start the Application

```bash
python medsr.py
```

This will:
1. Load the Silero VAD model for pause detection
2. Load Google MedASR model for medical transcription
3. Initialize Gemini client for text cleanup
4. Start Gradio web interface

### Using the Interface

1. **Open Browser**: Navigate to the URL shown (e.g., `http://localhost:7860`)

2. **Start Recording**: Click the microphone icon to begin recording

3. **Dictate Normally**: 
   - Speak your radiology findings clearly
   - **Pause briefly between sentences** (800ms+)
   - Each completed sentence will appear on screen in real-time

4. **Watch Real-Time Transcription**:
   - See sentences appear as you speak
   - View accumulated text building up
   - Monitor which sentence is currently being processed

5. **Stop Recording**: Click stop when finished

6. **Generate Final Report**: Click "Final Merge →" button to create a polished, coherent medical report

### Example Workflow

**You speak:**
> "Bronchovascular markings are noted in the bilateral lung fields." *[pause 1s]*

**Screen shows:**
```
🔴 RECORDING IN PROGRESS - Sentence 1 transcribed

📝 LATEST SENTENCE (1):
Bronchovascular markings are noted in the bilateral lung fields.
```

**You continue:**
> "Paucity of vessels in the right lower zone, suggestive of consolidation." *[pause 1s]*

**Screen updates:**
```
🔴 RECORDING IN PROGRESS - Sentence 2 transcribed

📝 LATEST SENTENCE (2):
Paucity of vessels in the R lower zone, suggestive of consolidation.

📋 ALL SENTENCES SO FAR:
Sentence 1: Bronchovascular markings are noted in the bilateral lung fields.
Sentence 2: Paucity of vessels in the R lower zone, suggestive of consolidation.
```

## 🔧 Configuration

### Adjust Pause Detection Sensitivity

In `medsr.py`, modify the `pause_threshold_ms` parameter:

```python
# Shorter pause = more sensitive (may split mid-sentence)
pause_detected = detect_pause_with_vad(full_audio, sample_rate, pause_threshold_ms=500)

# Longer pause = less sensitive (waits longer between sentences)
pause_detected = detect_pause_with_vad(full_audio, sample_rate, pause_threshold_ms=1200)
```

**Default: 800ms** - works well for natural medical dictation

### Change Maximum Buffer Duration

Prevents extremely long sentences from delaying transcription:

```python
# Current default: 15 seconds
max_duration_exceeded = total_duration >= 15.0

# Increase for longer complex sentences
max_duration_exceeded = total_duration >= 20.0
```

### Customize Gemini Cleanup Behavior

Modify the `cleanup_chunk_with_gemini()` prompt to adjust how text is cleaned.

## 🧪 Technical Details

### Models Used

1. **Silero VAD**: Voice Activity Detection for pause detection
   - Detects speech vs silence in real-time
   - Identifies sentence boundaries

2. **Google MedASR**: Medical Automatic Speech Recognition
   - Trained on medical terminology
   - Domain-specific accuracy for radiology reports

3. **Gemini 2.0 Flash Lite**: AI text cleanup and formatting
   - Per-sentence cleanup: spelling, filler removal
   - Final merge: coherent document generation
   - Medical abbreviation standardization

### Processing Pipeline

#### Per-Sentence Cleanup (Gemini)
- Fix spelling errors
- Remove pause sounds (um, uh, er)
- Remove duplicate words/phrases
- Remove filler words
- **Does NOT** add new words or create sentences

#### Final Merge (Gemini)
- Combines all sentences into flowing document
- Applies medical abbreviations (R, L, w/, w/o, ↑, ↓)
- Converts dictation markers ("fill stop" → `.`, "next line" → line break)
- Fixes capitalization and punctuation
- Ensures proper medical report structure

### VAD Logic

```python
if total_duration >= 2.0:  # Minimum 2 seconds
    if pause_detected (800ms silence):
        → Transcribe as complete sentence
    elif total_duration >= 15.0:  # Maximum 15 seconds
        → Force transcribe (failsafe)
    else:
        → Keep accumulating
```

## 🐛 Troubleshooting

### "No module named 'google.genai'"

```bash
pip install --upgrade google-generativeai
```

### "Model google/medasr not found"

1. Accept the license: https://huggingface.co/google/medasr
2. Login: `huggingface-cli login`

### Transcription Quality Issues

**Problem**: Sentences are cut off or incomplete

**Solution**: 
- Pause longer between sentences (1+ second)
- Increase `pause_threshold_ms` to 1000-1200ms
- Speak at a moderate pace

**Problem**: Transcription is delayed

**Solution**:
- Reduce `pause_threshold_ms` to 600-700ms
- Reduce `max_duration_exceeded` threshold

**Problem**: Medical terms are incorrect

**Solution**:
- Ensure MedASR license is accepted
- Speak medical terms clearly
- Use the Final Merge button for Gemini corrections

### "No microphone found"

- Check browser permissions for microphone access
- Ensure microphone is connected and working
- Try a different browser (Chrome/Firefox recommended)

### VAD Not Working

The system automatically falls back to energy-based detection if Silero VAD fails. Check terminal output for:
- `✅ Silero VAD (PyTorch) loaded successfully!`
- `✅ Using energy-based VAD fallback (librosa)`

## 📊 Performance

- **Latency**: ~2-4 seconds per sentence (depends on sentence length)
- **Memory**: ~2-3GB RAM (models loaded)
- **Accuracy**: High for medical terminology (MedASR is domain-specific)

## 🔐 Privacy & Security

- All processing runs **locally** except Gemini API calls
- Audio is not stored permanently (temporary files deleted immediately)
- Set `use_gemini_formatting=False` to avoid any external API calls

## 📄 License

This project uses:
- **Google MedASR**: Subject to Google's terms and MedASR license
- **Gemini API**: Subject to Google AI Studio terms
- **Silero VAD**: MIT License

## 🤝 Contributing

Suggestions and improvements welcome! Key areas:
- Better pause detection algorithms
- Support for more languages
- Alternative ASR models
- Enhanced medical terminology post-processing

## 📞 Support

For issues:
1. Check terminal output for error messages
2. Verify all dependencies are installed
3. Ensure API keys are set correctly
4. Check model licenses are accepted

## 🎓 Citation

If you use this in research or production:

```bibtex
@software{medical_asr_realtime,
  title={Real-Time Medical ASR with VAD-Based Sentence Detection},
  year={2026},
  note={Uses Google MedASR, Silero VAD, and Gemini AI}
}
```

---

**Built with ❤️ for medical professionals who need accurate, real-time dictation**

