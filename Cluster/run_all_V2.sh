# Run the setup script
echo "Running setup..."
python SAMed_BraTS_setup.py

# Check if setup was successful, then run the data loader script
if [ $? -eq 0 ]; then
    echo "Setup completed successfully. Proceeding to data loading..."
    python SAMed_BraTS_dataloader.py
else
    echo "Setup failed. Exiting..."
    exit 1
fi

# Check if data loading was successful, then run the training script
if [ $? -eq 0 ]; then
    echo "Data loading completed successfully. Proceeding to training..."
    python SAMed_BraTS_V2_training.py
else
    echo "Data loading failed. Exiting..."
    exit 1
fi

# Check if training was successful, then run the inference script
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Proceeding to inference..."
    python SAMed_BraTS_inference.py
else
    echo "Training failed. Exiting..."
    exit 1
fi

echo "All scripts completed successfully!"
