import os

snore_folder = "snoring/1"
nonsnore_folder = "snoring/0"

snore_files = os.listdir(snore_folder)
nonsnore_files = os.listdir(nonsnore_folder)

print(len(snore_files))
print(len(nonsnore_files))
print(snore_files[0])

results = []

for f in snore_files:
    full_path = snore_folder + "/" + f
    results.append((full_path, "snore"))

for f in nonsnore_files:
    full_path = nonsnore_folder + "/" + f
    results.append((full_path, "background")) 

print(len(results))
print(results[0])
print(results[-1])