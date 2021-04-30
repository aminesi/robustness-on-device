echo "start creating venv"
if [ ! -d venv ]; then
  module load python/3.7.4
  virtualenv --no-download venv
  source venv/bin/activate
else
  source venv/bin/activate
fi

pip install --no-index tensorflow_gpu
pip install scipy --no-index
pip install opencv-python --no-index
pip install numpy --no-index
pip install tqdm --no-index
pip install --no-index torch torchvision
echo "venv created"
export TF_CPP_MIN_LOG_LEVEL=3
export DATA_ROOT=/scratch/aminesi/image_net/

