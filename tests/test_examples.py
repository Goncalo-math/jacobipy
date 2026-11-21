import pytest
import sympy as sp

from jacobipy import examples as ex


def test_symplectic_dimension_and_matrix():
    P = ex.symplectic(2)
    assert P.dim == 4
    assert P.Pi[0, 2] == 1 and P.Pi[2, 0] == -1


def test_lotka_volterra_rejects_bad_input():
    A = sp.Matrix([[0, 1], [1, 0]])  # symmetric, not skew
    with pytest.raises(ValueError):
        ex.Lotka_Volterra(2, A)
    with pytest.raises(ValueError):
        ex.Lotka_Volterra(3, sp.zeros(2))
    with pytest.raises(ValueError):
        ex.Lotka_Volterra(2, sp.zeros(2), coords=sp.symbols("x1:4"))


def test_rigid_body_coords_argument():
    P = ex.Rigid_Body(coords="u")
    assert [str(c) for c in P.coords] == ["u1", "u2", "u3"]
    with pytest.raises(ValueError):
        ex.Rigid_Body(coords=sp.symbols("a b"))


@pytest.mark.parametrize("fn,n", [
    (ex.jacobi_two_dimensional, 3),
    (ex.jacobi_three_dimensional, 2),
    (ex.jacobi_four_dimensional, 3),
])
def test_wrong_number_of_coordinates(fn, n):
    with pytest.raises(ValueError):
        fn(coords=sp.symbols(f"a1:{n + 1}"))


def test_contact_dimension():
    assert ex.contact(3).dim == 7


def test_jacobi_rigid_body_requires_a_sum_zero():
    with pytest.raises(ValueError):
        ex.jacobi_Rigid_Body([1, 1, 1], [0, 0, 0])
