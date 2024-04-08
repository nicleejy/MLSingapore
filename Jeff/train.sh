#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jeffl@comp.nus.edu.sg
#SBATCH --gpus=a100:1
#SBATCH --partition=long
#SBATCH -t 4-24:00:00

TMPDIR=`mktemp -d`
cp ~/cs3264/* $TMPDIR

srun pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location
srun pip3 install gdown --no-warn-script-location
srun pip3 install ultralytics --no-warn-script-location
# srun gdown 1na7VT3ywXAOqw9_VL19dyQIyS_8EObTF -O $TMPDIR/dataset.zip
# srun unzip "dataset.zip" -d "datasets"

srun python3 main.py
# srun yolo task=classify mode=train model=yolov8m-cls.pt data='datasets' epochs=10 imgsz=640
# cp $TMPDIR/runs/classify ~/results/
# rm -rf $TMPDIR




