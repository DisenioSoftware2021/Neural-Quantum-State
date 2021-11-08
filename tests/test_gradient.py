import NQS

import numpy as np

import pytest

# vectorr_list = [
#     [2.0, 2.0],
#     [2.0, 2.0, 2.0],
#     [2.0, 2.0, 2.0, 2.0],
# ]

# expected_list = [
#     [28.83422699, 36.29351474],
#     [29.61280337, 66.23228459, -39.31138972],
#     [-2.31107436, -5.37416041, 12.61154441, -54.07361377],
# ]

# input_values = [
#     (2, 2, 1, vectorr_list[0], expected_list[0]),
#     (3, 3, 1, vectorr_list[1], expected_list[1]),
#     (4, 2, 2, vectorr_list[2], expected_list[2]),
# ]

# n_hidden = 5
# n_visible = 5
# learning_rate = 0.1
# gamma = 0.5
# n_parameter = n_hidden + n_visible + (n_visible * n_hidden)
# grad = np.ones(n_hidden + n_visible + (n_visible * n_hidden))
# expected_shift = np.zeros(n_parameter)

gradient_list = [
    np.ones(5 + 5 + (5 * 5)),
    np.ones(4 + 4 + (4 * 4)) * 2,
    np.ones(3 + 3 + (3 * 3)) * 3,
]

input_values_shift = [
    (0.1, 0.5, 5, 5, gradient_list[0], np.ones(5 + 5 + (5 * 5)) * -0.1),
    (1.0, 0.5, 4, 4, gradient_list[1], np.ones(4 + 4 + (4 * 4)) * -2),
    (5.0, 0.5, 3, 3, gradient_list[2], np.ones(3 + 3 + (3 * 3)) * -15),
]  # [(learning_rate, gamma, n_hidden, n_visible, grad, expected)]


@pytest.mark.parametrize(
    "learning_rate, gamma, n_hidden, n_visible, grad, expected_shift",
    input_values_shift,
)  # [(learning_rate, gamma, n_hidden, n_visible, grad, expected)])
def test_gradient_parameter_shift(
    learning_rate, gamma, n_hidden, n_visible, grad, expected_shift
):

    gd = NQS.Gradient(learning_rate, gamma, n_hidden, n_visible)

    assert np.allclose(gd.parameter_shift(grad), expected_shift, atol=1e-7)


input_values_adam = [
    (
        0.1,
        0.5,
        5,
        5,
        gradient_list[0],
        2,
        np.ones(5 + 5 + (5 * 5)) * -0.07424598,
    ),
    (
        1.0,
        0.5,
        4,
        4,
        gradient_list[1],
        4,
        np.ones(4 + 4 + (4 * 4)) * -0.57721541,
    ),
    (
        5.0,
        0.5,
        3,
        3,
        gradient_list[2],
        6,
        np.ones(3 + 3 + (3 * 3)) * -2.58141198,
    ),
]  # [(learning_rate, gamma, n_hidden, n_visible, grad, iteration, expected)]


@pytest.mark.parametrize(
    (
        "learning_rate, gamma, n_hidden, n_visible, grad, \
        iteration, expected_adam"
    ),
    input_values_adam,
)
def test_adam(
    learning_rate, gamma, n_hidden, n_visible, grad, iteration, expected_adam
):

    gd = NQS.Gradient(learning_rate, gamma, n_hidden, n_visible)

    assert np.allclose(gd.adam(grad, iteration), expected_adam, atol=1e-7)
