#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=test
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1



python3 /home/RUS_CIP/st178893/dl-lab-25w-team07/diabetic_retinopathy/main.py dataset=eyepacs dataset.path=/home/data/EyePACS/graham dataset.batch_size=32 model=own_cnn model.dropout_rate=0.0 model.num_base_filters=8 epochs=20 dataset.image_size=256 lr=1e-3 log_interval=1000
