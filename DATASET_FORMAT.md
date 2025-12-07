# VibeVoice Bangla Dataset Format

To fine-tune VibeVoice, arrange your data in the following structure.

## Directory Structure

```plaintext
my_bangla_dataset/
├── wavs/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── metadata.csv
```

## Audio Files (`wavs/`)
- **Format**: WAV
- **Sampling Rate**: 24kHz recommended (code will resample if needed, but pre-resampling is faster).
- **Channels**: Mono (1 channel).
- **Duration**: 2-15 seconds per file is ideal for training.

## Metadata File (`metadata.csv`)
A pipe-delimited text file (typical LJSpeech format) is recommended.

**Format**: `ID|Transcription`

**Example**:
```text
sample1|এই বইটি কেমন?
sample2|আজ আবহাওয়া বেশ ভালো।
```
*Note: The `ID` must match the filename in `wavs/` (excluding the `.wav` extension).*

## Alternative: JSON Format
You can also use a JSON file:
```json
[
  {
    "id": "sample1",
    "text": "এই বইটি কেমন?"
  },
  {
    "id": "sample2",
    "text": "আজ আবহাওয়া বেশ ভালো।"
  }
]
```

## Check Your Data
Before training, ensure:
1. All IDs in metadata exist as `.wav` files.
2. Transcriptions are in valid Bangla text.
3. Audio files are not corrupted.
