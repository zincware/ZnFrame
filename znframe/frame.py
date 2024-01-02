from attrs import define, field, cmp_using, Factory
import attrs
import numpy as np
import ase.cell
from ase.data.colors import jmol_colors
from copy import deepcopy
import json
import networkx as nx

from znframe.bonds import ASEComputeBonds

def _cell_to_array(cell: np.ndarray | ase.cell.Cell) -> np.ndarray:
    if isinstance(cell, np.ndarray):
        return cell
    if isinstance(cell, list):
        return np.array(cell)
    return cell.array


def _list_to_array(array: dict | list) -> dict | np.ndarray:
    if isinstance(array, list):
        return np.array(array)
    if isinstance(array, dict):
        for key, value in array.items():
            array[key] = _list_to_array(value)
        return array
    return array


def _ndarray_to_list(array: dict | np.ndarray) -> dict | list:
    if isinstance(array, np.ndarray):
        return array.tolist()
    if isinstance(array, dict):
        for key, value in array.items():
            array[key] = _ndarray_to_list(value)
        return array
    return array


@define
class Frame:
    numbers: np.ndarray = field(converter=_list_to_array, eq=cmp_using(np.array_equal))
    positions: np.ndarray = field(
        converter=_list_to_array, eq=cmp_using(np.array_equal)
    )
    arrays: dict[str, np.ndarray] = field(converter=_list_to_array, eq=False)
    info: dict[str, float | int | np.ndarray] = field(
        converter=_list_to_array, eq=False
    )
    pbc: np.ndarray = field(converter=_list_to_array, eq=cmp_using(np.array_equal))
    cell: np.ndarray = field(converter=_cell_to_array, eq=cmp_using(np.array_equal))

    connectivity: nx.Graph = nx.empty_graph() 
    # this should be replaced with field.
    # Furthermore if you use from_json, you get a list instead of a graph. 
    # this should also be possible as it is not very effficient to convert the list to a graph
    # and then back to a list.

    def __attrs_post_init__(self):
        
        if not isinstance(self.connectivity, list):
            ase_bond_calculator = ASEComputeBonds()
            if self.connectivity.number_of_nodes() == 0:
                self.connectivity = ase_bond_calculator.build_graph(self.to_atoms())
            self.connectivity = ase_bond_calculator.get_bonds(self.connectivity)

        if "colors" not in self.arrays:
            self.arrays["colors"] = [
                self.rgb2hex(jmol_colors[number]) for number in self.numbers
                ]
        if "radii" not in self.arrays:
            self.arrays["radii"] = [
                self.get_radius(number) for number in self.numbers
            ]

    @classmethod
    def from_atoms(cls, atoms: ase.Atoms):
        arrays = deepcopy(atoms.arrays)
        info = deepcopy(atoms.info)

        return cls(
            numbers=arrays.pop("numbers"),
            positions=arrays.pop("positions"),
            arrays=arrays,
            info=info,
            pbc=atoms.pbc,
            cell=atoms.cell,
        )

    def to_atoms(self) -> ase.Atoms:
        atoms = ase.Atoms(
            numbers=self.numbers,
            positions=self.positions,
            pbc=self.pbc,
            cell=self.cell,
        )

        atoms.arrays.update(self.arrays)
        atoms.info.update(self.info)

        return atoms

    def to_dict(self) -> dict:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    def to_json(self) -> str:
        data = self.to_dict()
        data = _ndarray_to_list(data)
        return json.dumps(data)

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))

    def rgb2hex(self, value):
        r, g, b = np.array(value * 255, dtype=int)
        return "#%02x%02x%02x" % (r, g, b)
    
    def get_radius(self, value):
        return (0.25 * (2 - np.exp(-0.2 * value)),)
    
