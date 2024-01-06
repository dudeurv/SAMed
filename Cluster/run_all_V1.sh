#!/bin/bash -l
#$ -l gpu=1
#$ -l h_rt=5:00:0
#$ -l tmem=16G
#$ -N myJobName
#$ -wd /home/ududeja

# Activate the virtual environment
source /cluster/project7/SAMed/LapVideoUS-SUSI/bin/activate

# Navigate to the directory containing the scripts
cd /cluster/project7/SAMed

# Run the setup script
echo "Running SAMed_BraTS_setup.py..."
python SAMed_BraTS_setup.py

# Check if setup was successful, then run the data loader script
if [ $? -eq 0 ]; then
    echo "Setup completed successfully. Proceeding to SAMed_BraTS_dataloader.py..."
    python SAMed_BraTS_dataloader.py
else
    echo "Setup failed. Exiting..."
    exit 1
fi

# Check if data loading was successful, then run the training script
if [ $? -eq 0 ]; then
    echo "Data loading completed successfully. Proceeding to SAMed_BraTS_V2_training.py..."
    python SAMed_BraTS_V1_training.py
else
    echo "Data loading failed. Exiting..."
    exit 1
fi

# Check if training was successful, then run the inference script
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Proceeding to SAMed_BraTS_inference.py..."
    python SAMed_BraTS_inference.py
else
    echo "Training failed. Exiting..."
    exit 1
fi

echo "All scripts completed successfully!"
