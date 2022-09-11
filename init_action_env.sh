# conda create -n mmaction2 python=3.8 -y
# conda activate mmaction2
pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
# cd mmaction2
pip3 install -e .
pip3 install future tensorboard
pip3 install setuptools==59.5.0
pip3 install rich
pip3 install yacs

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"