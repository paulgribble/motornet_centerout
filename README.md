# motornet_centerout

simple demo of training a motornet system

- two joint planar arm (shoulder, elbow)
- 6 Hill-type muscles based on Kistemaker, Wong & Gribble (2010) J. Neurophysiol. 104(6):2985-94
- training on point-to-point reaches to random targets in the workspace
- null-field training by default though curl FFs are an option
- force-channel probe trials are an option
- center-out reaches to 'test' the network
- saving a network and loading it up again to test it later
- various plots

## Installing motornet

Assumption: you have python3.12 installed.

I use `pip` to organize Python environments. First I create a venv for motornet that includes python3.12, and install some needed packages:

```{shell}
python3.12 -m venv ~/venvs/motornet
source ~/venvs/motornet/bin/activate
python3 -m pip install -U pip
pip install tqdm setuptools ipykernel
```

Then I install MotorNet:

```{shell}
pip install motornet
```

To save all dependencies:

```{shell}
pip freeze > requirements.txt
```

## Other

To re-install from the requirements.txt file:
```{shell}
pip install -r requirements.txt
```

## Other

Or for the speedup version of motornet:

I install the `speeduptorch` branch of MotorNet:

```{shell}
pip install git+https://github.com/OlivierCodol/MotorNet.git@speeduptorch
```

Then I install PyTorch:
```{shell}
pip install torch torchvision torchaudio
```

To save all dependencies:

```{shell}
pip freeze > requirements_speedup.txt
```

