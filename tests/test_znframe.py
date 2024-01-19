from ase.build import molecule
import ase.io
from ase.calculators.singlepoint import SinglePointCalculator
from znframe import Frame
import pytest
import numpy as np


@pytest.fixture
def water() -> Frame:
    return Frame.from_atoms(molecule("H2O"))


@pytest.fixture
def ammonia() -> Frame:
    ammonia = molecule("NH3")
    ammonia.arrays["momenta"] = np.random.random((4, 3))
    ammonia.arrays["forces"] = np.random.random((4, 3))
    ammonia.info["energy"] = -1234
    return Frame.from_atoms(ammonia)


@pytest.fixture
def waterWithCalc() -> ase.Atoms:
    atoms = molecule("H2O")
    atoms.cell = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    atoms.pbc = [True, True, True]
    atoms.calc = SinglePointCalculator(
        atoms, energy=-1234, forces=np.random.random((3, 3)), stress=np.random.random(6)
    )
    return atoms


def test_frame_from_ase_molecule(ammonia):
    assert ammonia.numbers.shape == (4,)
    assert ammonia.positions.shape == (4, 3)
    assert set(ammonia.arrays) == {"momenta", "radii", "colors", "forces"}
    assert set(ammonia.info) == {"energy"}
    assert ammonia.pbc.shape == (3,)
    assert ammonia.cell.shape == (3, 3)


def test_frame_to_ase():
    atoms = molecule("H2O")
    frame = Frame.from_atoms(atoms)
    assert frame.to_atoms() == atoms


def test_frame_eq(water, ammonia):
    assert water != ammonia


def test_frame_to_dict(water):
    assert water.to_dict() == {
        "numbers": water.numbers,
        "positions": water.positions,
        "connectivity": water.connectivity,
        "arrays": water.arrays,
        "info": water.info,
        "pbc": water.pbc,
        "cell": water.cell,
    }


def test_frame_from_dict(ammonia):
    assert Frame.from_dict(ammonia.to_dict()) == ammonia


def test_to_json(ammonia):
    assert Frame.from_json(ammonia.to_json()) == ammonia


def test_water_with_calc(waterWithCalc):
    frame = Frame.from_atoms(waterWithCalc)
    assert frame.to_atoms().calc == waterWithCalc.calc
