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

Assumption: you have python3.13 installed. On MacOS:

```{shell}
brew install python@3.13
```

I use `pip` to organize Python environments.

```{shell}
python3.13 -m venv .venv
source .venv/bin/activate
python3.13 -m pip install -U pip
pip install numpy matplotlib torch gymnasium tqdm joblib 
```

## Starting point

After you install motornet and the libraries above, activate the venv:

```{shell}
source .venv/bin/activate
```

Then the `go.py` script is the starting point.

```{shell}
python3.13 go.py
```

After you can use the jupyter notebook `golook.ipynb` to load up some results and make some plots. You may need to install the following packages to run the notebook:

```{shell}
pip install setuptools ipykernel nbconvert dPCA scipy scikit-learn numexpr numba pandas
```


