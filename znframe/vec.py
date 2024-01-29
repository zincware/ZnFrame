import numpy as np
import dataclasses
import typing as t


@dataclasses.dataclass
class LatticeVecField:
    vectors: t.Union[np.ndarray, list] 
    box: t.Union[np.ndarray, list]
    density: t.Union[np.ndarray, list]
    origin: t.Union[np.ndarray, list] 
    color: str = "#e6194B" # standard-color is red

    def __post_init__(self):
        for item in ["vectors", "box", "density", "origin"]:
            attr = getattr(self, item)
            if isinstance(attr, np.ndarray):
                setattr(self, item, attr.tolist())
        
        self.check()

    def check(self):
        length = len(self.vectors)
        if np.prod(self.density) != length:
            raise ValueError("The number of vectors does not match the given density")

    def to_dict(self) -> dict:
        return {"vectors": self.vectors,
                "box": self.box,
                "density": self.density,
                "origin": self.origin,
                "color": self.color}

@dataclasses.dataclass
class OriginVecField:
    vectors: t.Union[np.ndarray, list]
    origins: t.Union[np.ndarray, list]
    color: str = "#e6194B" # standard-color is red

    def __post_init__(self):
        for item in ["vectors", "origins"]:
            attr = getattr(self, item)
            if isinstance(attr, np.ndarray):
                setattr(self, item, attr.tolist())

        self.check()

    def check(self):
        length = len(self.vectors)
        if length != len(self.origins):
            raise ValueError("The number of vectors does not match the number of origins")

    def to_dict(self) -> dict:
        return {"vectors": self.vectors,
                "origins": self.origins,
                "color": self.color}