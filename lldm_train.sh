CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
conda activate simpler_env


GPU_ID=$1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"

python libero/lifelong/main.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO