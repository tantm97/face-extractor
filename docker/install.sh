apt-get -y update
apt-get -y upgrade
apt-get -y install htop python3.8.13
apt-get -y install python3-pip
apt-get -y install build-essential cmake unzip pkg-config
apt-get -y install libjpeg-dev libpng-dev libtiff-dev
apt-get -y install libgtk-3-dev
apt-get -y install vim tmux
yes | pip3 install opencv-python==4.2.0.34 opencv-contrib-python==4.2.0.34
yes | pip3 install pyyaml scikit-image tqdm numpy==1.18.3
yes | pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
yes | pip3 install tensorboard
python3 -m pip install setuptools==59.5.0
yes | pip3 install black==23.1.0 commitizen==2.40.0 flake8==6.0.0 isort==5.12.0
yes | pip3 install pre-commit==3.0.2 pre-commit-hooks==4.4.0 pytest==7.2.1
yes | pip3 install toml==0.10.2 tomli==2.0.1
