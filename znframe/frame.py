from attrs import define, field, cmp_using
import attrs
import numpy as np
import ase.cell
from ase.data.colors import jmol_colors
from ase.calculators.singlepoint import SinglePointCalculator
from copy import deepcopy
import json
import typing as t
import enum

from znframe.bonds import ASEComputeBonds
from znframe.vec import LatticeVecField, OriginVecField


class ComputeProperties(enum.Enum):
    bonds = "bonds"
    radii = "radii"
    colors = "colors"


def _cell_to_array(cell: t.Union[np.ndarray, ase.cell.Cell]) -> np.ndarray:
    if isinstance(cell, np.ndarray):
        return cell
    if isinstance(cell, list):
        return np.array(cell)
    return cell.array


def _list_to_array(array: t.Union[dict, list]) -> t.Union[dict, np.ndarray]:
    if isinstance(array, list):
        return np.array(array)
    if isinstance(array, dict):
        for key, value in array.items():
            array[key] = _list_to_array(value)
        return array
    return array


def _ndarray_to_list(array: t.Union[dict, np.ndarray]) -> t.Union[dict, list]:
    if isinstance(array, np.ndarray):
        return array.tolist()
    if isinstance(array, dict):
        for key, value in array.items():
            array[key] = _ndarray_to_list(value)
        return array
    return array


def _npnumber_to_number(number):
    if isinstance(number, np.floating):
        return float(number)
    if isinstance(number, np.integer):
        return int(number)
    return number


def _vec_to_dict(vec) -> dict:
    if isinstance(vec, LatticeVecField) or isinstance(vec, OriginVecField):
        vec = vec.to_dict()
    vec = _ndarray_to_list(vec)
    return vec


@define
class Frame:
    numbers: np.ndarray = field(converter=_list_to_array, eq=cmp_using(np.array_equal))
    positions: np.ndarray = field(
        converter=_list_to_array, eq=cmp_using(np.array_equal)
    )

    connectivity: np.ndarray = field(
        converter=_list_to_array, eq=cmp_using(np.array_equal), default=None
    )

    arrays: dict[str, np.ndarray] = field(
        converter=_list_to_array, eq=False, factory=dict
    )
    info: dict[str, t.Union[float, int, np.ndarray]] = field(
        converter=_list_to_array, eq=False, factory=dict
    )

    calc: dict[str, t.Union[float, int, np.ndarray]] = field(
        converter=_list_to_array, eq=False, factory=dict
    )

    vector_field: t.Union[OriginVecField, LatticeVecField] = field(
        converter=_vec_to_dict, eq=False, factory=dict
    )

    pbc: np.ndarray = field(
        converter=_list_to_array,
        eq=cmp_using(np.array_equal),
        default=np.array([True, True, True]),
    )
    cell: np.ndarray = field(
        converter=_cell_to_array, eq=cmp_using(np.array_equal), default=np.zeros(3)
    )

    recompute: t.List[ComputeProperties] = field(
        factory=lambda: [
            ComputeProperties.bonds,
            ComputeProperties.radii,
            ComputeProperties.colors,
        ]
    )

    def __attrs_post_init__(self):
        if ComputeProperties.bonds in self.recompute:
            self.connectivity = None
        if ComputeProperties.radii in self.recompute:
            self.arrays.pop("radii", None)
        if ComputeProperties.colors in self.recompute:
            self.arrays.pop("colors", None)
        if self.connectivity is None:
            ase_bond_calculator = ASEComputeBonds()
            self.connectivity = ase_bond_calculator.build_graph(self.to_atoms())
            self.connectivity = ase_bond_calculator.get_bonds(self.connectivity)

        if "colors" not in self.arrays:
            self.arrays["colors"] = np.array(
                [rgb2hex(jmol_colors[number]) for number in self.numbers]
            )
        if "radii" not in self.arrays:
            self.arrays["radii"] = np.array(
                [get_radius(number) for number in self.numbers]
            )

    @classmethod
    def from_atoms(
        cls,
        atoms: ase.Atoms,
    ):
        arrays = deepcopy(atoms.arrays)
        info = deepcopy(atoms.info)

        if hasattr(atoms.calc, "results"):
            duplicates_arrays = list(
                set(atoms.calc.results.keys()) & set(atoms.arrays.keys())
            )
            for key in duplicates_arrays:
                arrays.pop(key)

            duplicates_info = list(
                set(atoms.calc.results.keys()) & set(atoms.info.keys())
            )
            for key in duplicates_info:
                info.pop(key)

        frame = cls(
            numbers=arrays.pop("numbers"),
            positions=arrays.pop("positions"),
            arrays=arrays,
            info=info,
            pbc=atoms.pbc,
            cell=atoms.cell,
        )

        try:
            calc_data = {}
            for key, value in atoms.calc.results.items():
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                calc_data[key] = value
            frame.calc = calc_data
        except AttributeError:
            pass

        try:
            frame.connectivity = atoms.connectivity
        except AttributeError:
            pass

        return frame

    def to_atoms(self) -> ase.Atoms:
        atoms = ase.Atoms(
            numbers=self.numbers,
            positions=self.positions,
            pbc=self.pbc,
            cell=self.cell,
        )

        if self.calc:
            atoms.calc = SinglePointCalculator(atoms)
            atoms.calc.results = {
                key: np.array(val) if isinstance(val, list) else val
                for key, val in self.calc.items()
            }

        atoms.arrays.update(self.arrays)
        atoms.info.update(self.info)
        atoms.connectivity = self.connectivity

        return atoms

    def to_dict(self, built_in_types: bool = True) -> dict:
        data = attrs.asdict(self, filter=lambda attr, _: attr.name != "recompute")
        if built_in_types:
            return data
        else:
            data = _ndarray_to_list(data)
            for key, value in data["info"].items():
                if isinstance(value, np.generic):
                    data["info"][key] = _npnumber_to_number(value)
                else:
                    data["info"][key] = value
            return data

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    def to_json(self) -> str:
        data = self.to_dict(built_in_types=False)
        return json.dumps(data)

    @classmethod
    def from_json(cls, s: str):
        return cls.from_dict(json.loads(s))


def rgb2hex(value):
    r, g, b = np.array(value * 255, dtype=int)
    return "#%02x%02x%02x" % (r, g, b)


def get_radius(value):
    return (0.25 * (2 - np.exp(-0.2 * value)),)
