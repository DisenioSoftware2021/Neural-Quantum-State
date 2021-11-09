import NQS

import numpy as np

import pytest

vectorr_list = [
    [2.0, 2.0],
    [2.0, 2.0, 2.0],
    [2.0, 2.0, 2.0, 2.0],
]

expected_list_exponential = [
    [28.83422699, 36.29351474],
    [29.61280337, 66.23228459, -39.31138972],
    [-2.31107436, -5.37416041, 12.61154441, -54.07361377],
]

input_values_exponential = [
    (2, 2, 1, vectorr_list[0], expected_list_exponential[0]),
    (3, 3, 1, vectorr_list[1], expected_list_exponential[1]),
    (4, 2, 2, vectorr_list[2], expected_list_exponential[2]),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,visible,expect_exp", input_values_exponential
)
def test_gaussian_nqs_exponential_argument(
    hidden, dim, partic, visible, expect_exp
):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)

    assert np.allclose(qs.exponential_argument(visible), expect_exp, atol=1e-7)


expected_list_psi = [8.68436451877e-68, 1.2755019064985346e-107, 0.0]

input_values_psi = [
    (
        2,
        2,
        1,
        vectorr_list[0],
        expected_list_exponential[0],
        expected_list_psi[0],
    ),
    (
        3,
        3,
        1,
        vectorr_list[1],
        expected_list_exponential[1],
        expected_list_psi[1],
    ),
    (
        4,
        2,
        2,
        vectorr_list[2],
        expected_list_exponential[2],
        expected_list_psi[2],
    ),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,visible,q,expect_psi", input_values_psi
)
def test_gaussian_nqs_psi(hidden, dim, partic, visible, q, expect_psi):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)

    assert np.allclose(qs.psi(visible, q), expect_psi, atol=1e-7)


expected_list_sigmoid = [
    [1.0, 1.0],
    [1.0000000e00, 1.0000000e00, 8.4582471e-18],
    [9.02099309e-02, 4.61342609e-03, 9.99996667e-01, 3.28191949e-24],
]
input_values_sigmoid = [
    (
        2,
        2,
        1,
        vectorr_list[0],
        expected_list_exponential[0],
        expected_list_sigmoid[0],
    ),
    (
        3,
        3,
        1,
        vectorr_list[1],
        expected_list_exponential[1],
        expected_list_sigmoid[1],
    ),
    (
        4,
        2,
        2,
        vectorr_list[2],
        expected_list_exponential[2],
        expected_list_sigmoid[2],
    ),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,visible,q,expect_sig", input_values_sigmoid
)
def test_gaussian_nqs_sigmoid(hidden, dim, partic, visible, q, expect_sig):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)

    assert np.allclose(qs.sigmoid(q), expect_sig, atol=1e-7)


expected_list_der_sigmoid = [
    [3.00230167e-13, 1.72952490e-16],
    [1.37823382e-13, 1.72061727e-29, 8.45824710e-18],
    [8.20720993e-02, 4.59214239e-03, 3.33328892e-06, 3.28191949e-24],
]

input_values_der_sigmoid = [
    (2, 2, 1, expected_list_exponential[0], expected_list_der_sigmoid[0]),
    (3, 3, 1, expected_list_exponential[1], expected_list_der_sigmoid[1]),
    (4, 2, 2, expected_list_exponential[2], expected_list_der_sigmoid[2]),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,q,expect_der_sig", input_values_der_sigmoid
)
def test_gaussian_nqs_der_sigmoid(hidden, dim, partic, q, expect_der_sig):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)

    assert np.allclose(qs.derivative_sigmoid_q(q), expect_der_sig, atol=1e-7)


expected_list_laplacian = [
    -15582.145635791683,
    -24819.88750141783,
    -38045.44472907044,
]
input_values_laplacian = [
    (
        2,
        2,
        1,
        vectorr_list[0],
        expected_list_exponential[0],
        expected_list_laplacian[0],
    ),
    (
        3,
        3,
        1,
        vectorr_list[1],
        expected_list_exponential[1],
        expected_list_laplacian[1],
    ),
    (
        4,
        2,
        2,
        vectorr_list[2],
        expected_list_exponential[2],
        expected_list_laplacian[2],
    ),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,visible,q,expect_lap", input_values_laplacian
)
def test_gausssian_nqs_laplacian(hidden, dim, partic, visible, q, expect_lap):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)
    qs.sigmoid(q)
    qs.derivative_sigmoid_q(q)

    assert np.allclose(qs.laplacian(visible), expect_lap, atol=1e-7)


expected_list_laplacian_alfa = [
    [
        96.09572965,
        97.28013178,
        0.5,
        0.5,
        41.74908153,
        41.74908153,
        29.82770135,
        29.82770135,
    ],
    [
        9.72801318e01,
        1.04806913e02,
        9.46449567e01,
        5.00000000e-01,
        5.00000000e-01,
        4.22912355e-18,
        4.17490815e01,
        4.17490815e01,
        3.53124048e-16,
        2.98277013e01,
        2.98277013e01,
        2.52290069e-16,
        1.44431621e01,
        1.44431621e01,
        1.22163834e-16,
    ],
    [
        1.04806913e02,
        9.46449567e01,
        9.64927217e01,
        9.64751327e01,
        4.51049654e-02,
        2.30671305e-03,
        4.99998333e-01,
        1.64095975e-24,
        3.76618176e00,
        1.92606302e-01,
        4.17489424e01,
        1.37017124e-22,
        2.69075488e00,
        1.37607896e-01,
        2.98276019e01,
        9.78921144e-23,
        1.30291665e00,
        6.66324608e-02,
        1.44431139e01,
        4.74012952e-23,
        1.93732911e-01,
        9.90769485e-03,
        2.14757138e00,
        7.04817985e-24,
    ],
]
input_values_laplacian_alfa = [
    (
        2,
        2,
        1,
        vectorr_list[0],
        expected_list_exponential[0],
        expected_list_laplacian_alfa[0],
    ),
    (
        3,
        3,
        1,
        vectorr_list[1],
        expected_list_exponential[1],
        expected_list_laplacian_alfa[1],
    ),
    (
        4,
        2,
        2,
        vectorr_list[2],
        expected_list_exponential[2],
        expected_list_laplacian_alfa[2],
    ),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,visible,q,expect_lap_alfa", input_values_laplacian_alfa
)
def test_gaussian_nqs_laplacian_alfa(
    hidden, dim, partic, visible, q, expect_lap_alfa
):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)
    qs.sigmoid(q)

    assert np.allclose(qs.laplacian_alfa(visible), expect_lap_alfa, atol=1e-7)


vectorr_list_dist = [
    [2.0, 1.0],
    [2.0, 2.0, 3.0],
    [1.0, 2.0, 3.0, 4.0],
]


expected_list_inverse_distance = [0.0, 0.0, [[0, 0.35355339], [0, 0]]]
input_values_inverse_distance = [
    (2, 2, 1, vectorr_list_dist[0], expected_list_inverse_distance[0]),
    (3, 3, 1, vectorr_list_dist[1], expected_list_inverse_distance[1]),
    (4, 2, 2, vectorr_list_dist[2], expected_list_inverse_distance[2]),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,visible,expect_inverse", input_values_inverse_distance
)
def test_gaussian_nqs_inverse_distance(
    hidden, dim, partic, visible, expect_inverse
):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)

    assert np.allclose(qs.inverse_distance(visible), expect_inverse, atol=1e-7)


expected_list_calogero = [4.0, 4.0, 4.0]
input_values_calogero = [
    (2, 2, 1, vectorr_list[0], expected_list_calogero[0]),
    (3, 3, 1, vectorr_list[1], expected_list_calogero[1]),
    (4, 2, 2, vectorr_list[2], expected_list_calogero[2]),
]


@pytest.mark.parametrize(
    "hidden,dim,partic,visible,expect_calogero", input_values_calogero
)
def test_gaussian_nqs_calogero(hidden, dim, partic, visible, expect_calogero):
    qs = NQS.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)

    assert np.allclose(qs.calogero(visible), expect_calogero, atol=1e-7)
