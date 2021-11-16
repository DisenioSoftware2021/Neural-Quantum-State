import NQS

import numpy as np

import pytest

omega = [
    1.0,
    2.0,
    3.0,
]

expected_harmonic = [
    1.991606497212622,
    3.9664259888504874,
    5.924458474913596,
]

expected_coulomb = [
    3.277551763254751,
    5.2523712548926165,
    7.210403740955725,
]

expected_calogero = [
    4.877963087448465,
    6.157331023061322,
    7.622944249082751,
]

input_values_harmonic = [
    (
        omega[0],
        "harmonic_oscillator",
        1 / np.sqrt(2 * omega[0]),
        expected_harmonic[0],
    ),
    (
        omega[1],
        "harmonic_oscillator",
        1 / np.sqrt(2 * omega[1]),
        expected_harmonic[1],
    ),
    (
        omega[2],
        "harmonic_oscillator",
        1 / np.sqrt(2 * omega[2]),
        expected_harmonic[2],
    ),
]

input_values_coulomb = [
    (omega[0], "coulomb", 1 / np.sqrt(2 * omega[0]), expected_coulomb[0]),
    (omega[1], "coulomb", 1 / np.sqrt(2 * omega[1]), expected_coulomb[1]),
    (omega[2], "coulomb", 1 / np.sqrt(2 * omega[2]), expected_coulomb[2]),
]

input_values_calogero = [
    (omega[0], "calogero", 1 / np.sqrt(2 * omega[0]), expected_calogero[0]),
    (omega[1], "calogero", 1 / np.sqrt(2 * omega[1]), expected_calogero[1]),
    (omega[2], "calogero", 1 / np.sqrt(2 * omega[2]), expected_calogero[2]),
]


@pytest.mark.parametrize(
    "omega,interaction,sigma,expected_harmonic", input_values_harmonic
)
def test_harmonic_oscillator(omega, interaction, sigma, expected_harmonic):
    qs = NQS.GaussianNQS(4, 2, 2, sigma, 1.0, seed=100)

    ham = NQS.Hamiltonian(omega, interaction)

    assert np.allclose(ham.local_energy(qs), expected_harmonic, atol=1e-7)


@pytest.mark.parametrize(
    "omega,interaction,sigma,expected_coulomb", input_values_coulomb
)
def test_coulomb(omega, interaction, sigma, expected_coulomb):
    qs = NQS.GaussianNQS(4, 2, 2, sigma, 1.0, seed=100)

    ham = NQS.Hamiltonian(omega, interaction)

    assert np.allclose(ham.local_energy(qs), expected_coulomb, atol=1e-7)


@pytest.mark.parametrize(
    "omega,interaction,sigma,expected_calogero", input_values_calogero
)
def test_calogero(omega, interaction, sigma, expected_calogero):
    qs = NQS.GaussianNQS(2, 1, 2, sigma, 1.0, seed=100)

    print(qs.visible_values_, qs.calogero(qs.visible_values_))

    ham = NQS.Hamiltonian(omega, interaction)

    assert np.allclose(ham.local_energy(qs), expected_calogero, atol=1e-7)
