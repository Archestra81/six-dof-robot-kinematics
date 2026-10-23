import numpy as np
from dh import dh_transform

def test_dh_transform_identity():
    # Test that zero parameters give a 4x4 identity matrix
    T = dh_transform(a=0.0, alpha=0.0, d=0.0, theta=0.0)
    assert T.shape == (4, 4)
    expected = np.eye(4)
    np.testing.assert_allclose(T, expected, atol=1e-7)

def test_dh_transform_translation():
    # Test translation along d
    T = dh_transform(a=0.0, alpha=0.0, d=5.0, theta=0.0)
    assert T[2, 3] == 5.0
