# ZnFrame - ASE-like Interface based on dataclasses

This package is designed for light-weight applications that require a structure for managing atomic structures.
It's primary focus lies on the conversion to / from JSON, to send data around easily.

```python
from znframe import Frame
from ase.build import molecule

frame = Frame.from_atoms(molecule("NH3"))

print(frame.to_json())
```
