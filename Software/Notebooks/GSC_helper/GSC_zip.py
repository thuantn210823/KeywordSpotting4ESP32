import os
from zipfile import ZipFile
from tqdm import tqdm

# Create object of ZipFile
def zipzip(input_directory, zip_file_name):
    with ZipFile(zip_file_name, 'w') as zip_object:
    # Traverse all files in directory
        for folder_name, sub_folders, file_names in os.walk(input_directory):
            for filename in tqdm(file_names, desc = 'zipping...' ):
                # Create filepath of files in directory
                file_path = os.path.join(folder_name, filename)
                # Add files to zip file
                zip_object.write(file_path, os.path.basename(file_path))

    if os.path.exists(zip_file_name):
        print(f"{zip_file_name} created")
    else:
        print(f"{zip_file_name} created")

def unzipzip(zip_file_name, output_directory):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    with ZipFile(zip_file_name, 'r') as zip_obj:
        zip_obj.extractall(output_directory)

    print(f"Extracted {zip_file_name}")