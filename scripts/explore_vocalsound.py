import os
#what importing os does is that it allows you to interact with the operating system in a way that is portable across different platforms. You can use it to perform tasks such as reading or writing files, navigating the file system, and executing system commands. In the context of your script, it might be used to handle file paths, check for the existence of files or directories, or create new directories as needed.

folder = "vocalsound/audio_16k"
files = os.listdir(folder)

print(len(files))
print(files[0])

parts = files[0].split("_")
print(parts)

label =parts[2].replace(".wav", "")
print(label)


counts = {}

for f in files:
    label = f.split("_")[2].replace(".wav", "")
    if label in counts:
        counts[label] += 1
    else:
        counts[label] = 1

print(counts)

results =[]

for f in files:
    label = f.split("_")[2].replace(".wav","")
    if label == "cough" or label == "sneeze":
       full_path = "vocalsound/audio_16k/" + f
       results.append((full_path, label))

print(len(results))
print(results[0])
print(results[-1])