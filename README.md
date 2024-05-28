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

I use `pip` to organize Python environments. After creating a venv for motornet that includes python3 I install the `speeduptorch` branch of motornet and the nightly version of PyTorch:

```{shell}
pip3 install git+https://github.com/OlivierCodol/MotorNet.git@speeduptorch --force
```

then
```{shell}
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir --force
```

