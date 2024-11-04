from pathlib import Path
import os
import collections

def count_files_in_directory(directory):
    # Create a Path object
    path = Path(directory)
    # Count only files (ignoring subdirectories)
    return sum(1 for f in path.iterdir() if f.is_file())

# Replace "directory_path" with the path to your specific folder


def CountDatasetEntry(directory_path):
    identities=  sum(1 for f in Path(directory_path).iterdir() if f.is_dir())
    print(f"identities : " +str(identities))
    count = 0
    for file in os.listdir(directory_path):
        path = os.path.join(directory_path, file)
        if os.path.isdir(path):
            count += count_files_in_directory(path)
        
    return count



dataset = "GoogleDataset/GgDataset"
print(CountDatasetEntry(dataset))

