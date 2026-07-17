import pandas as pd
import os
import random

random.seed(42)

all_data = []

# --- ESC-50 dataset ---
esc50_df = pd.read_csv("ESC-50/meta/esc50.csv")
label_map = {
    'coughing': 'cough',
    'sneezing': 'sneeze',
    'snoring': 'snore',
    'breathing': 'background',
    'vacuum_cleaner': 'background',
    'rain': 'background',
    'wind': 'background',
    'clock_tick': 'background',
    'laughing': 'background',
    'pouring_water': 'background',
    'clapping': 'background',
}
esc50_filtered = esc50_df[esc50_df['category'].isin(label_map.keys())].copy()
esc50_filtered['label'] = esc50_filtered['category'].map(label_map)
esc50_filtered['filepath'] = "ESC-50/audio/" + esc50_filtered['filename']

for _, row in esc50_filtered.iterrows():
    all_data.append((row['filepath'], row['label']))

print("after ESC-50:", len(all_data))

# --- VocalSound (cough + sneeze only) ---
vs_folder = "vocalsound/audio_16k"
vs_files = os.listdir(vs_folder)

for f in vs_files:
    label = f.split("_")[2].replace(".wav", "")
    if label == "cough" or label == "sneeze":
        full_path = vs_folder + "/" + f
        all_data.append((full_path, label))

print("after VocalSound:", len(all_data))

# --- Kaggle Snoring ---
snore_folder = "snoring/Snoring Dataset/1"
nonsnore_folder = "snoring/Snoring Dataset/0"

for f in os.listdir(snore_folder):
    full_path = snore_folder + "/" + f
    all_data.append((full_path, "snore"))

for f in os.listdir(nonsnore_folder):
    full_path = nonsnore_folder + "/" + f
    all_data.append((full_path, "background"))

print("after snoring:", len(all_data))

# --- Speech Commands ---
words = ["yes", "no", "stop", "go", "up", "down", "left", "right"]
base_folder = "speech_commands"

for word in words:
    word_folder = os.path.join(base_folder, word)
    for f in os.listdir(word_folder):
        full_path = os.path.join(word_folder, f)
        all_data.append((full_path, "speech"))

print("after speech:", len(all_data))


#grouping by label before sampling to ensure we have a balanced dataset

by_label = {}

for filepath, label in all_data:
    if label not in by_label:
        by_label[label] = []
    by_label[label].append((filepath, label))

for label in by_label:
    print(label, len(by_label[label]))


#sampling

target_counts = {
    "cough": 2000,
    "sneeze": 2000,
    "snore": 540,
    "background": 820,
    "speech": 2000,
}

final_data = []

for label in by_label:
    available = by_label[label]
    n = target_counts[label]
    if len(available) > n:
        sampled = random.sample(available, n)  #pick exactly n samples randomly form target counts
    else:
        sampled = available
    final_data.extend(sampled)
    print(label, "->", len(sampled))

print("TOTAL:", len(final_data))


#saving the final as a csv file

manifest_df = pd.DataFrame(final_data, columns=["filepath", "label"])
manifest_df.to_csv("manifest.csv", index=False)

print(manifest_df.shape)
print(manifest_df['label'].value_counts())
