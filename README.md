# To setup dev environment

1. Install Homebrew and create a conda environment

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# Install conda to create virtual environments for Python
brew install --cask miniconda
conda init zsh

# Close and re-open a Terminal window

# Setup a conda environment
conda create -y --name EyeTracking python=3.9
conda activate EyeTracking
```

2. Install dependencies

```
pip3 install coremltools protobuf==3.20.1
conda install -y -c apple tensorflow-deps
pip3 install tensorflow-macos
pip3 install tensorflow-metal
pip3 install mediapipe-silicon
conda install -y pytorch torchvision torchaudio -c pytorch-nightly
pip3 install depthai scipy
```

# To Run

```
python3 head_and_iris_and_gaze_oak.py
```
