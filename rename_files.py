import os
import sys

def rename_files(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return

    # Iterate through all files in the folder
    count = 0
    for filename in os.listdir(folder_path):
        # Check if file starts with 'H_' and ends with '.pt'
        if filename.startswith('H_') and filename.endswith('.pt'):
            # Create new filename (remove 'H_' prefix)
            new_filename = filename[2:]  # Skip first two characters 'H_'
            
            # Construct full file paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            try:
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {str(e)}")
    
    print(f"\nDone! Renamed {count} files.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_files.py <folder_path>")
    else:
        folder_path = sys.argv[1]
        rename_files(folder_path)