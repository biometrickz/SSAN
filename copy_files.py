import os
import shutil
import pandas as pd
from tqdm import tqdm

def copy_files_to_folders(csv_file, destination_folder, output_csv):
    """
    Reads a CSV file containing filenames and labels, and copies files to the specified
    destination folder into subfolders based on their labels.

    Args:
        csv_file (str): Path to the CSV file with filenames and labels.
        source_folder (str): Path to the source folder where files are currently located.
        destination_folder (str): Path to the destination folder where files will be copied.

    CSV Format:
        The CSV file should have two columns:
        - 'filename': The name of the file to copy.
        - 'label': The label indicating the subfolder ('1' for 'live', otherwise 'spoof').

    Folder Structure:
        - destination_folder
          - live (for files with label 1)
          - spoof (for files with any other label)
    """
    
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Define paths for 'live' and 'spoof' subfolders
    live_folder = os.path.join(destination_folder, 'live')
    spoof_folder = os.path.join(destination_folder, 'spoof')
    
    # Create subfolders if they don't exist
    os.makedirs(live_folder, exist_ok=True)
    os.makedirs(spoof_folder, exist_ok=True)
    new_file_data = []

    # Iterate over the rows in the CSV
    for _, row in tqdm(data.iterrows()):
        filename = row['fname']
        label = row['label']
        
        # Determine the destination folder based on the label
        if label == 1:
            dest_folder = live_folder
        else:
            dest_folder = spoof_folder
        
        # Determine the destination file path
        dest_path = os.path.join(dest_folder, os.path.basename(filename))
        
        # Copy the file
        try:
            shutil.copy(filename, dest_path)
            new_file_data.append({'fname': dest_path, 'label': label})

        except FileNotFoundError as e:
            print(f"Error: {e}")
        # Convert the list to a DataFrame
    new_file_df = pd.DataFrame(new_file_data)

    # Save the new file paths and labels to the specified CSV file
    new_file_df.to_csv(output_csv, index=False)
    print(f"New CSV file saved to {output_csv}")

# Example usage:
copy_files_to_folders('/mnt/8TB/ml_projects_yeldar/patchnet/val/val_list10.csv', '/mnt/8TB/ml_projects_yeldar/patchnet/M1_3_4_val', '/mnt/8TB/ml_projects_yeldar/patchnet/val/val_list11.csv')
