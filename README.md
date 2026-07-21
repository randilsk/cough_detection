# Cough/Sneeze/Snore/Speech Classifier — ML Training Pipeline

Trains and quantizes a TinyML model that classifies short audio clips as
**cough / sneeze / snore / speech / background**, producing a TFLite Micro
model ready to deploy on an ESP32.

This is the ML half of a another project: an ESP32-based wake-on-sound
respiratory event classifier — a battery-powered device that listens
continuously, wakes on detected sound, and classifies it on-device.
The hardware/firmware half lives in a separate repo:
**[esp32-cough-classifier-firmware](#)** _(link once created)_.

Built to learn TinyML deployment, FreeRTOS, bare-metal I2S/DMA programming,
and I2C/SPI/UART protocols in one integrated build, for a final-year EE
portfolio project — rather than as isolated tutorial exercises.

## Why two repos

The ML pipeline (Python, TensorFlow, dataset wrangling) and the firmware
(C, ESP-IDF, FreeRTOS) have completely different toolchains, dependencies,
and audiences. Keeping them separate means each repo stays focused and
buildable on its own — this repo produces one artifact (`cough_model.tflite`)
that the firmware repo consumes as an input.

## Project status

**Phase 1 — Signal processing fundamentals (complete)**

- Learned FFT/DFT, Nyquist limit, conjugate symmetry, frequency resolution
- `scripts/fft_explore.py` — loads a wav file, computes rfft, plots magnitude vs frequency
- `scripts/mel_explore.py` — computes and visualizes a log-mel spectrogram from a real recording
- Full pipeline understood: waveform → overlapping Hann-windowed chunks → FFT →
  mel filterbank → log → stacked spectrogram (32 mel bins × 32 time windows per clip)

**Phase 2 — Dataset construction (complete)**

Built a balanced, labeled dataset from four public sources:

| Class      | Source(s)                                          | Count |
| ---------- | -------------------------------------------------- | ----- |
| cough      | ESC-50 + VocalSound (MIT)                          | 2000  |
| sneeze     | ESC-50 + VocalSound (MIT)                          | 2000  |
| snore      | ESC-50 + Kaggle snoring dataset                    | 540   |
| background | ESC-50 + Kaggle snoring dataset (non-snore folder) | 820   |
| speech     | Google Speech Commands (8 words)                   | 2000  |

**Total: 7360 samples.** `build_manifest.py` gathers all four sources, applies
label mapping, randomly samples down oversized classes (seeded for
reproducibility), and writes `manifest.csv`.

**Phase 3 — Feature extraction, training, and quantization (complete)**

- `extract_features.py` — loads every file in `manifest.csv`, extracts the
  loudest 1-second window (via RMS energy) for clips longer than 1 second,
  computes a log-mel spectrogram per clip, caches all 7360 as
  `features_cache.npz`
- `train_model.py` — small CNN (Conv2D 8→16→16, GlobalAveragePooling, Dense 32,
  Dropout 0.3, Dense 5 softmax; 4,277 parameters, 16.7KB float32), trained with
  class weights (to handle the snore/background imbalance) and early stopping
- `convert_tflite.py` — converts to int8 TFLite Micro format using a
  representative dataset for calibration
- `verify_tflite.py` — sanity-checks the quantized model's input/output
  quantization parameters and spot-checks accuracy against the float model

**Results:**

|               | Float model (Keras) | Quantized (int8 TFLite) |
| ------------- | ------------------- | ----------------------- |
| Test accuracy | 87.0%               | 85.0%\*                 |
| Model size    | 16.7 KB             | 11.0 KB                 |

\*measured on a 200-sample spot check, not the full held-out test set — close
enough to the float model's accuracy to confirm quantization introduced only
the expected, minor accuracy cost.

Per-class performance (float model, full test set of 1104 samples):

| Class      | Precision | Recall | F1   |
| ---------- | --------- | ------ | ---- |
| background | 0.80      | 0.89   | 0.84 |
| cough      | 0.87      | 0.88   | 0.88 |
| sneeze     | 0.85      | 0.78   | 0.81 |
| snore      | 0.91      | 0.89   | 0.90 |
| speech     | 0.90      | 0.94   | 0.92 |

Main confusion is between cough and sneeze (acoustically similar, both short
broadband bursts) — expected and documented rather than a bug. See
`experiments/confusion_matrix_v2_earlystop.png` and
`experiments/training_curves_v2_earlystop.png`.

**Phase 4 — Hardware integration (moved to separate repo)**

ESP-IDF bare-metal I2S/DMA audio capture, wake-on-sound VAD, FreeRTOS task
pipeline, OLED display, SD logging, and on-device inference using this repo's
`cough_model.tflite` output now live in the firmware repo.

## Setup

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install tensorflow-cpu librosa numpy scipy soundfile scikit-learn matplotlib pandas tqdm
```

Requires Python 3.11 specifically — TensorFlow doesn't yet support 3.14.

## Datasets

Not tracked in git (too large). Reproduce locally:

```bash
git clone https://github.com/karolpiczak/ESC-50.git

curl.exe -L -o vocalsound_16k.zip "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1"
tar -xf vocalsound_16k.zip -C vocalsound

kaggle datasets download -d tareqkhanemu/snoring
tar -xf snoring.zip -C snoring

curl.exe -o speech_commands.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir speech_commands
tar -xzf speech_commands.tar.gz -C speech_commands
```

## Running the full pipeline

```bash
python build_manifest.py
python extract_features.py
python train_model.py
python convert_tflite.py
python verify_tflite.py
```

Output: `tflite_model/cough_model.tflite` — the file to hand off to the
firmware repo.

## Design notes / decisions made along the way

- **ESP32-C6 was ruled out** in favor of ESP32-WROOM-32 — single-core RISC-V
  has weaker TFLite Micro / ESP-DSP support than the dual-core Xtensa classic ESP32.
- **Chose per-file-source data gathering over manual recording** to save time,
  accepting that this requires careful class-imbalance handling rather than
  controlling data collection directly.
- **Rejected combining aggressive upsampling AND class weights simultaneously**
  — decided this risks overcorrection; used class weights as the primary
  balancing mechanism instead.
- **Mel spectrogram parameters**: 16kHz sample rate, 1024-sample FFT window
  (~64ms), 512-sample hop (50% overlap), 32 mel bins — chosen to keep the
  resulting per-clip "image" small (32×32) for a genuinely tiny on-device
  model, while covering the frequency range relevant to human respiratory/vocal
  sounds (up to 8kHz).
- **Added early stopping (v2)** after v1 showed accuracy still climbing at
  epoch 30 with no fixed epoch count clearly "correct" in advance — improved
  test accuracy from 85.6% to 87.0%.
- **Full int8 quantization** (not hybrid) required specifically for TFLite
  Micro compatibility on the ESP32 — confirmed via `verify_tflite.py` that
  quantization only cost ~2 percentage points of accuracy.
