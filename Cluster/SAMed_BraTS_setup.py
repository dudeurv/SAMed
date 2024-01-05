import os
import zipfile

def setup_environment_and_data():
    # Repository and data setup
    CODE_DIR = 'samed_codes'
    os.makedirs(f'./{CODE_DIR}', exist_ok=True)

    # Clone the repository if it hasn't been cloned yet
    if not os.path.isdir(f"./{CODE_DIR}/.git"):
        os.system(f'git clone https://github.com/hitachinsk/SAMed.git {CODE_DIR}')
    else:
        print("Repository already cloned.")

    os.chdir(f'./{CODE_DIR}')

    # Install the SAM library from Facebook Research
    os.system('pip install git+https://github.com/facebookresearch/segment-anything.git')

    # Define the links to your dataset and weights
    slices_zip_link = 'https://drive.google.com/uc?id=1nHZWlCBpudbT4zzPyqyu2Vi5uILcxSrv'
    epoch_weights_link = 'https://drive.google.com/uc?id=1P0Bm-05l-rfeghbrT1B62v5eN-3A-uOr'
    sam_weights_link = 'https://drive.google.com/uc?id=1_oCdoEEu3mNhRfFxeWyRerOKt8OEUvcg'

    # Download and extract dataset if it hasn't been done yet
    if not os.path.exists("Slices.zip"):
        os.system(f'wget "{slices_zip_link}" -O Slices.zip')
        with zipfile.ZipFile('Slices.zip', 'r') as zip_ref:
            zip_ref.extractall()
    else:
        print("Dataset already downloaded and extracted.")

    # Download epoch weights if they haven't been downloaded yet
    if not os.path.exists("epoch_159.pth"):
        os.system(f'wget "{epoch_weights_link}" -O epoch_159.pth')
    else:
        print("Epoch weights already downloaded.")

    # Download the pre-trained SAM model if it hasn't been downloaded yet
    if not os.path.exists("sam_vit_b_01ec64.pth"):
        os.system(f'wget "{sam_weights_link}" -O sam_vit_b_01ec64.pth')
    else:
        print("Pre-trained SAM model already downloaded.")

if __name__ == "__main__":
    setup_environment_and_data()
