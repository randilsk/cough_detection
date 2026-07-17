import os

words = ["yes", "no", "stop", "go", "up", "down", "left", "right"]
base_folder = "speech_commands"

results = []

for word in words:
    word_folder = os.path.join(base_folder, word)
    files = os.listdir(word_folder)
    for f in files:
        full_path = os.path.join(word_folder, f)
        results.append((full_path, "speech"))

print(len(results))
print(results[0])
print(results[-1])