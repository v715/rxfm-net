# Make sure the right version of CUDA is being used
export CUDA_HOME="/usr/local/cuda-11.3"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
export PATH="/usr/local/cuda-11.3/bin:$PATH"

# Install pytorch dependencies
conda create -n rxfm python=3.8 conda==4.10.3
conda activate rxfm
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -c conda-forge -f

# Install the se3CNN submodule
git submodule init
git submodule update
pip install -e se3cnn/.

# Install pytorch3d
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt110/download.html

# Install other rxfm-net dependencies
pip install nilearn matplotlib seaborn seaborn_image

# Misc installs
conda install black flake8 nb_conda_kernel -f
