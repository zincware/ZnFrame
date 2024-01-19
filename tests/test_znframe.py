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
    assert "forces" in waterWithCalc.calc.results
    assert "stress" in waterWithCalc.calc.results
    assert "energy" in waterWithCalc.calc.results

    assert "forces" not in waterWithCalc.info
    assert "stress" not in waterWithCalc.arrays
    assert "energy" not in waterWithCalc.arrays

    assert "forces" not in waterWithCalc.arrays
    assert "stress" not in waterWithCalc.info
    assert "energy" not in waterWithCalc.info

    frame = Frame.from_atoms(waterWithCalc)
    intersection = set(frame.info) & set(frame.arrays)
    if intersection:
        raise ValueError(f"Duplicate keys: {intersection}")

    assert "forces" not in frame.arrays
    assert "stress" not in frame.arrays
    assert "energy" not in frame.arrays

    assert "stress" not in frame.info
    assert "energy" not in frame.info
    assert "forces" not in frame.info

    assert "forces" in frame.calc
    assert "stress" in frame.calc
    assert "energy" in frame.calc


    atoms = frame.to_atoms()
    for key in atoms.calc.results.keys():
        if isinstance(atoms.calc.results[key], np.ndarray):
            np.testing.assert_array_equal(
                atoms.calc.results[key], waterWithCalc.calc.results[key]
            )
        else:
            assert atoms.calc.results[key] == waterWithCalc.calc.results[key]
