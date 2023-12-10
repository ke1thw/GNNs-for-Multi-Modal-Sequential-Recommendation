import os

# Define the directory containing the .tar files
directory = "/scratch/yw5326/MMSRec/dataset/webvid/preprocess/CLIP-WebVid/data/train"

# List all the .tar files in the directory
all_files = sorted([f for f in os.listdir(directory) if f.endswith('.tar')])

# Calculate the number of files to retain which here we keep 1/10 of original files
# num_to_retain = len(all_files) // 10

# Determine the files to retain and the files to delete
files_to_retain = all_files[:30]
files_to_delete = all_files[30:]

# Delete the files
for file in files_to_delete:
    os.remove(os.path.join(directory, file))

print(f"Retained {len(files_to_retain)} files and deleted {len(files_to_delete)} files.")
