#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=test
#SBATCH --output=job_name-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 

python3 /home/RUS_CIP/st178893/dl-lab-25w-team07/diabetic_retinopathy/main.py dataset=idrid model=densenet121 dataset.batch_size=32 log_interval=1000 train_mode=finetune checkpoint_path=checkpoints/ckpt_OwnCNN_42_final_epoch3.pt finetune_epochs=10 epochs=5 model.dropout_rate=0.3 model.num_classes=2
