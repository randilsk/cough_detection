# ESP32 Wake-on-Sound Respiratory Event Classifier

A battery-powered ESP32 device that listens continuously in a low-power state,
wakes on detected sound, and classifies the audio as **cough / sneeze / snore /
speech / background** using an on-device TinyML model (TFLite Micro).

TinyML deployment, FreeRTOS,bare-metal I2S/DMA programming, and I2C/SPI/UART protocols in one integrated build

## Hardware

- ESP32-WROOM-32 DevKit (dual-core Xtensa)
- INMP441 I2S MEMS microphone
- 0.96" SSD1306 OLED, I2C
- MicroSD module (SPI)
- 18650 Li-ion battery

## Project status

**Phase 1 — Signal processing fundamentals (complete)**

- Learned FFT/DFT, Nyquist limit, conjugate symmetry, frequency resolution
- `scripts/fft_explore.py` — loads a wav file, computes rfft, plots magnitude vs frequency
- `scripts/mel_explore.py` — computes and visualizes a log-mel spectrogram from a real recording
- Understand the full pipeline: waveform → overlapping Hann-windowed chunks → FFT →
  mel filterbank → log → stacked spectrogram

**Phase 2 — Dataset construction (complete)**
Built a balanced, labeled dataset from four public sources:

| Class      | Source(s)                                                           | Count |
| ---------- | ------------------------------------------------------------------- | ----- |
| cough      | ESC-50 + VocalSound (MIT)                                           | 2000  |
| sneeze     | ESC-50 + VocalSound (MIT)                                           | 2000  |
| snore      | ESC-50 + Kaggle snoring dataset                                     | 540   |
| background | ESC-50 + Kaggle snoring dataset (non-snore folder)                  | 820   |
| speech     | Google Speech Commands (8 words: yes/no/stop/go/up/down/left/right) | 2000  |

**Total: 7360 samples.** Snore and background are kept below the 2000 target
deliberately (limited real data available) — will be handled via light
augmentation (~1.5-2x) for snore, and class weighting during training, rather
than aggressive upsampling that risks overfitting to a small set of source recordings.

`scripts/build_manifest.py` gathers all four sources, applies label mapping,
randomly samples down oversized classes (seeded for reproducibility), and
writes `manifest.csv` — the single source of truth for what's in the training set.

**Phase 3 — Feature extraction + training (next)**

- Loop over `manifest.csv`, compute log-mel spectrogram per file
- Light augmentation for the snore class
- Train small CNN, convert to TFLite Micro (int8 quantized)

**Phase 4 — Hardware integration (planned)**

- ESP-IDF bare-metal I2S/DMA audio capture with wake-on-sound VAD
- FreeRTOS task pipeline: capture → VAD → feature extraction → inference → UI/logging
- OLED status display, SD card event logging, battery power management

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

Then:

```bash
python build_manifest.py
```

## Design notes / decisions made along the way

- **ESP32-C6 was ruled out** in favor of ESP32-WROOM-32 — single-core RISC-V
  has weaker TFLite Micro / ESP-DSP support than the dual-core Xtensa classic ESP32.
- **Chose per-file-source data gathering over manual recording** to save time,
  accepting that this requires careful class-imbalance handling rather than
  controlling data collection directly.
- **Rejected combining aggressive upsampling AND class weights simultaneously**
  for the imbalanced classes — decided this risks overcorrection; using primarily
  class weights, with light augmentation only where real data is very limited (snore).
- **Mel spectrogram parameters**: 16kHz sample rate, 1024-sample FFT window (~64ms),
  512-sample hop (50% overlap), 32 mel bins — chosen to keep the resulting per-clip
  "image" small (~32x32) for a genuinely tiny
