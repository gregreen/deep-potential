module load anaconda3-py39 gpu/cuda/11.7

conda create --name=dpot python==3.9.13
conda activate dpot

pip install -r dpot_requirements.txt
