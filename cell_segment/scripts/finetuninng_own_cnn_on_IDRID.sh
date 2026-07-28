#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=test
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 

python3 /home/RUS_CIP/st178893/dl-lab-25w-team07/diabetic_retinopathy/main.py dataset=idrid dataset.path=/home/data/IDRID_dataset dataset.batch_size=32 model=own_cnn model.dropout_rate=0.0 checkpoint_path=checkpoints/basic_cnn_eyepacs.pt checkpoint_name=eyepacs_own_cnn.pt epochs=160 dataset.image_size=256 weight_decay=1e-4 lr=1e-4
