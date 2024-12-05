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

Assumption: you have python3.12 installed. On MacOS:

```{shell}
brew install python@3.12
```

I use `pip` to organize Python environments.

```{shell}
python3.12 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
pip install git+https://github.com/OlivierCodol/MotorNet.git@speeduptorch
pip install tqdm setuptools ipykernel nbconvert joblib dPCA scipy scikit-learn numexpr numba pandas
pip freeze > requirements.txt
```

## Starting point

After you install motornet and the libraries above, activate the venv:

```{shell}
source .venv/bin/activate
```

Then the `go.py` script is the starting point.

After you can use `golook.ipynb` to load up some results and make some plots.


