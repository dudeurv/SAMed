#!/bin/bash

# Activate your virtual environment
source /cluster/project7/SAMed/LapVideoUS-SUSI/bin/activate

# Declare an array of package names with specific versions
declare -a packages=(
    "einops==0.6.1"
    "icecream==2.1.3"
    "MedPy==0.4.0"
    "monai==1.1.0"
    "opencv_python==4.5.4.58"
    "SimpleITK==2.2.1"
    "tensorboardX==2.6"
    "ml-collections==0.1.1"
    "onnx==1.13.1"
    "onnxruntime==1.14.1"
    "torchmetrics"
)

# Loop through the array and install each package individually
for package in "${packages[@]}"
do
    echo "Installing $package..."
    pip install $package
    # Check if the package was installed successfully
    if [ $? -eq 0 ]; then
        echo "$package installed successfully."
    else
        echo "Failed to install $package. Exiting..."
        exit 1
    fi
done

echo "All packages installed successfully."
