# bmitoolbox
Toolbox for Brain Machine Interface.

## Installation or Update
```
pip install -U git+https://github.com/arailly/bmitoolbox
```

## Examples
```
import bmitoolbox as bt
filtered = bt.bandpass_filter(array, lowcut=1, highcut=40, fs=2000, numtaps=255)
```
