import numpy as np
import ejemplo as nqs  # solo tienen que ser import nqs

def test_GaussianNQS_exponential_argument():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)
    expected =[
    -2.31107436,
    -5.37416041,
    12.61154441,
    -54.07361377
    ]
    assert np.allclose(qs.exponential_argument([2.0, 2.0, 2.0, 2.0]), expected,atol=1e-7)

def test_GaussianNQS_sigmoid():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)

    expected = [
        9.02099307e-02,
        4.61342609e-03,
        9.99996667e-01,
        3.28191948e-24
    ]
    assert np.allclose(qs.sigmoid([-2.31107436,-5.37416041,12.61154441,-54.07361377]),expected,atol=1e-7)

def test_GaussianNQS_psi():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)
    
    expected = [0.0]
    assert np.allclose(qs.psi([2.0, 2.0, 2.0, 2.0],
    [ -2.31107436,  -5.37416041,  12.61154441, -54.07361377])
    , expected,atol=1e-7)

def test_GaussianNQS_derivative_sigmoid():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)
    
    expected = [
        8.20720991e-02,
        4.59214239e-03,
        3.33328891e-06,
        3.28191948e-24
    ]
    assert np.allclose(qs.derivative_sigmoid_Q([-2.31107436,-5.37416041,12.61154441,-54.07361377])
    , expected,atol=1e-7)

def test_GaussianNQS_laplacian():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)

    expected = [-38045.44472907044]
    x=[2.0, 2.0, 2.0, 2.0]
    q=qs.exponential_argument([2.0, 2.0, 2.0, 2.0])
    sigmoid=qs.sigmoid(q)
    der_sigmoid=qs.derivative_sigmoid_Q(q)

    assert np.allclose(qs.laplacian(x,sigmoid,der_sigmoid)
    , expected,atol=1e-7)
def test_GaussianNQS_laplacian_alfa():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)
    expected=[
       1.04806913e+02, 9.46449567e+01, 9.64927217e+01, 9.64751327e+01,
       4.51049654e-02, 2.30671305e-03, 4.99998333e-01, 1.64095975e-24,
       3.76618176e+00, 1.92606302e-01, 4.17489424e+01, 1.37017124e-22,
       2.69075488e+00, 1.37607896e-01, 2.98276019e+01, 9.78921144e-23,
       1.30291665e+00, 6.66324608e-02, 1.44431139e+01, 4.74012952e-23,
       1.93732911e-01, 9.90769485e-03, 2.14757138e+00, 7.04817985e-24
    ]
    x=[2.0, 2.0, 2.0, 2.0]
    sigmoid=(qs.sigmoid([-2.31107436,-5.37416041,12.61154441,-54.07361377]))
    assert np.allclose(qs.laplacian_alfa(x,sigmoid), expected,atol=1e-7)

def test_GaussianNQS_inverse_distance():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)
    expected=[[0., 0.],
       [0., 0.]]
    x=[2.0, 2.0, 2.0, 2.0]
    assert np.allclose(qs.inverse_distance(x), expected,atol=1e-7)

def test_GaussianNQS_calogero():
    qs = nqs.GaussianNQS(4, 2, 2, 0.1, 1.0, seed=100)
    expected=4.
    x=[2.0, 2.0, 2.0, 2.0]
    assert np.allclose(qs.calogero(x), expected,atol=1e-7)
