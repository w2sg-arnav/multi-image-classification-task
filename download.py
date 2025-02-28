import gdown
import os
import zipfile

def download_and_unzip(file_id, output_path, extract_path):
    """Downloads a file from Google Drive and unzips it.
    
    Args:
        file_id: The Google Drive file ID.
        output_path: The path where the downloaded file should be saved.
        extract_path: The path where the contents of the zip should be extracted
    """
    # Remove any existing corrupted file
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed existing file: {output_path}")
    
    print(f"Downloading file ID {file_id} to {output_path}...")
    gdown.download(id=file_id, output=output_path, quiet=False)
    
    try:
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            # Create extract directory if it doesn't exist
            os.makedirs(extract_path, exist_ok=True)
            print(f"Extracting {output_path} to {extract_path}...")
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"Error: {output_path} is not a valid zip file, or it is corrupted.")
        os.remove(output_path)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("You may need to check directory permissions or choose a different extraction path.")

if __name__ == "__main__":
    file_id = "1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ"
    output_file = "dataset.zip"
    extract_path = "./data"  # Changed to current directory/data
    
    download_and_unzip(file_id, output_file, extract_path)