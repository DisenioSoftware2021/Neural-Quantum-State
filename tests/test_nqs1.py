import numpy as np
import pytest

import NQS_1

vectorr_list = [
    [2.0, 2.0],
    [2.0, 2.0, 2.0],
    [2.0, 2.0, 2.0, 2.0],
]

expected_list = [
    [28.83422699, 36.29351474],
    [29.61280337, 66.23228459, -39.31138972],
    [-2.31107436, -5.37416041, 12.61154441, -54.07361377],
]

input_values = [
    (2, 2, 1, vectorr_list[0], expected_list[0]),
    (3, 3, 1, vectorr_list[1], expected_list[1]),
    (4, 2, 2, vectorr_list[2], expected_list[2]),
]


@pytest.mark.parametrize("hidden,dim,partic,visible,expect", input_values)
def test_gaussian_nqs_exponential_argument(
    hidden, dim, partic, visible, expect
):
    qs = nqs.GaussianNQS(hidden, dim, partic, 0.1, 1.0, seed=100)

    assert np.allclose(qs.exponential_argument(visible), expect, atol=1e-7)
