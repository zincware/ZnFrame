[![zincware](https://img.shields.io/badge/Powered%20by-zincware-darkcyan)](https://github.com/zincware)
[![codecov](https://codecov.io/gh/zincware/ZnFrame/graph/badge.svg?token=ZURLRO9WTI)](https://codecov.io/gh/zincware/ZnFrame)
[![PyPI version](https://badge.fury.io/py/znframe.svg)](https://badge.fury.io/py/znframe)

# ZnFrame - ASE-like Interface based on dataclasses

This package is designed for light-weight applications that require a structure
for managing atomic structures. It's primary focus lies on the conversion to /
from JSON, to send data around easily.

```python
from znframe import Frame
from ase.build import molecule

frame = Frame.from_atoms(molecule("NH3"))

print(frame.to_json())
```

# Installation

`pip install znframe`
