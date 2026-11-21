import pytest
import sympy as sp

from jacobipy import jacobi, poisson
from jacobipy import examples as ex

x, y, z = sp.symbols("x y z")


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------
def test_wrong_number_of_coords():
    with pytest.raises(ValueError):
        jacobi(2, sp.zeros(2), sp.zeros(2, 1), [x])


def test_wrong_lambda_shape():
    with pytest.raises(ValueError):
        jacobi(2, sp.zeros(3), sp.zeros(2, 1), [x, y])


def test_wrong_E_shape():
    with pytest.raises(ValueError):
        jacobi(2, sp.zeros(2), sp.zeros(3, 1), [x, y])


def test_coords_are_stored_as_list():
    J = ex.contact(1)  # sympy.symbols returns a tuple
    assert isinstance(J.coords, list)


# ---------------------------------------------------------------------------
# is_jacobi / is_poisson
# ---------------------------------------------------------------------------
def test_not_skew_symmetric_does_not_crash():
    J = jacobi(2, sp.Matrix([[1, 0], [0, 0]]), sp.zeros(2, 1), [x, y])
    assert J.is_jacobi() is False
    ok, reason = J.is_jacobi(return_reason=True)
    assert ok is False
    assert "skew-symmetric" in reason


def test_non_poisson_bivector_is_detected():
    # Skew-symmetric but not Poisson: v = (Pi^{23}, Pi^{31}, Pi^{12}) with
    # v . curl v != 0
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    Pi = sp.Matrix([[0, 1, -x1], [-1, 0, -x2], [x1, x2, 0]])
    P = poisson(3, Pi, [x1, x2, x3])
    assert P.is_poisson() is False
    ok, reason = P.is_poisson(return_reason=True)
    assert ok is False and "Jacobi condition" in reason


def test_second_condition_is_detected():
    # Lambda = d/dx ^ d/dy with E = x d/dy : [E, Lambda] != 0 and
    # the first condition fails too, but is_jacobi must say False.
    Lam = sp.Matrix([[0, 1], [-1, 0]])
    E = sp.Matrix([x * y, 0])
    assert jacobi(2, Lam, E, [x, y]).is_jacobi() is False


def test_returns_plain_bool():
    assert ex.Rigid_Body().is_poisson() is True
    assert ex.contact(1).is_jacobi() is True


POISSON_EXAMPLES = [
    ("symplectic(1)", lambda: ex.symplectic(1)),
    ("symplectic(3)", lambda: ex.symplectic(3)),
    ("Rigid_Body", lambda: ex.Rigid_Body()),
    ("Lotka_Volterra(3)",
     lambda: ex.Lotka_Volterra(3, sp.Matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]))),
]

JACOBI_EXAMPLES = [
    ("contact(1)", lambda: ex.contact(1)),
    ("contact(2)", lambda: ex.contact(2)),
    ("jacobi_two_dimensional", lambda: ex.jacobi_two_dimensional()),
    ("jacobi_three_dimensional", lambda: ex.jacobi_three_dimensional()),
    ("jacobi_four_dimensional", lambda: ex.jacobi_four_dimensional()),
    ("jacobi_Lotka_Volterra", lambda: ex.jacobi_Lotka_Volterra(1, 2, 3)),
    ("jacobi_Rigid_Body", lambda: ex.jacobi_Rigid_Body([1, 1, -2], [1, 2, 3])),
]


@pytest.mark.parametrize("name,build", POISSON_EXAMPLES, ids=[n for n, _ in POISSON_EXAMPLES])
def test_poisson_examples(name, build):
    P = build()
    assert isinstance(P, poisson)
    assert P.is_poisson() is True


@pytest.mark.parametrize("name,build", JACOBI_EXAMPLES, ids=[n for n, _ in JACOBI_EXAMPLES])
def test_jacobi_examples(name, build):
    assert build().is_jacobi() is True


# ---------------------------------------------------------------------------
# Bracket properties
# ---------------------------------------------------------------------------
def test_poisson_bracket_values_symplectic():
    P = ex.symplectic(1)
    q1, p1 = P.coords
    assert P.bracket(q1, p1) == 1
    assert P.bracket(p1, q1) == -1
    assert P.bracket(q1, q1) == 0


@pytest.mark.parametrize("build", [lambda: ex.contact(1), lambda: ex.jacobi_three_dimensional()])
def test_bracket_is_a_lie_bracket_and_leibniz_like(build):
    J = build()
    c = J.coords
    f = c[0] * c[1] + c[-1] ** 2
    g = c[0] ** 2 + c[1]
    h = c[1] * c[-1] + 3 * c[0]

    # skew-symmetry
    assert sp.simplify(J.bracket(f, g) + J.bracket(g, f)) == 0

    # Jacobi identity
    jac = (J.bracket(f, J.bracket(g, h))
           + J.bracket(g, J.bracket(h, f))
           + J.bracket(h, J.bracket(f, g)))
    assert sp.simplify(jac) == 0

    # {f, gh} = {f, g} h + g {f, h} + g h E(f)
    lhs = J.bracket(f, g * h)
    rhs = J.bracket(f, g) * h + g * J.bracket(f, h) + g * h * J.sharp_E(f)
    assert sp.simplify(lhs - rhs) == 0


def test_poisson_bracket_satisfies_leibniz():
    P = ex.Rigid_Body()
    a, b, c = P.coords
    f, g, h = a * b, b + c ** 2, a * c
    lhs = P.bracket(f, g * h)
    rhs = P.bracket(f, g) * h + g * P.bracket(f, h)
    assert sp.simplify(lhs - rhs) == 0


# ---------------------------------------------------------------------------
# Hamiltonian vector fields
# ---------------------------------------------------------------------------
def test_hamiltonian_vf_symplectic():
    P = ex.symplectic(1)
    q1, p1 = P.coords
    X = P.hamiltonian_vf(p1 ** 2 / 2 + q1 ** 2 / 2)
    # Convention of this module: X_H = Pi^#(dH) , i.e. X_H(f) = {f, H}
    assert X == sp.Matrix([p1, -q1])


def test_hamiltonian_vf_matches_bracket_convention():
    J = ex.contact(1)
    q1, p1, zz = J.coords
    H = q1 * p1 + zz ** 2
    f = q1 + p1 * zz
    X = J.hamiltonian_vf(H)
    Xf = sum(sp.diff(f, c) * X[i] for i, c in enumerate(J.coords))
    # X_H(f) = {f, H} - f E(H)
    assert sp.simplify(Xf - (J.bracket(f, H) - f * J.sharp_E(H))) == 0


# ---------------------------------------------------------------------------
# Casimirs
# ---------------------------------------------------------------------------
def test_casimir_rigid_body():
    P = ex.Rigid_Body()
    x1, x2, x3 = P.coords
    assert P.is_Casimir(x1 ** 2 + x2 ** 2 + x3 ** 2) is True
    assert P.is_Casimir(x1) is False


def test_casimir_symplectic_has_only_constants():
    P = ex.symplectic(1)
    q1, p1 = P.coords
    assert P.is_Casimir(5) is True
    assert P.is_Casimir(q1) is False


def test_casimir_jacobi_consistent_with_bracket():
    J = ex.contact(1)
    q1, p1, zz = J.coords
    assert J.is_Casimir(0) is True
    assert J.is_Casimir(zz) is False
    # constants are NOT Casimirs of a Jacobi structure with E != 0:
    # {1, g} = E(g)
    assert J.is_Casimir(1) is False


# ---------------------------------------------------------------------------
# Poissonization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,build", JACOBI_EXAMPLES, ids=[n for n, _ in JACOBI_EXAMPLES])
def test_poissonization_is_poisson(name, build):
    J = build()
    P = J.poissonization()
    assert isinstance(P, poisson)
    assert P.dim == J.dim + 1
    assert P.coords[:-1] == list(J.coords)
    assert P.is_poisson() is True


def test_poissonization_custom_symbol_and_clash():
    J = ex.jacobi_two_dimensional()
    s = sp.Symbol("s")
    assert J.poissonization(t=s).coords[-1] == s
    with pytest.raises(ValueError):
        ex.jacobi_two_dimensional(coords=[sp.Symbol("t"), y]).poissonization()


def test_poissonization_of_poisson_is_block_form():
    P = ex.symplectic(1)
    Pp = P.poissonization()
    t = Pp.coords[-1]
    assert Pp.Lambda[0, 1] == 1 / t
    assert Pp.Lambda[0, 2] == 0 and Pp.Lambda[2, 0] == 0


# ---------------------------------------------------------------------------
# is_jacobi must agree with the Jacobi identity of the bracket
# ---------------------------------------------------------------------------
def _bracket_satisfies_jacobi_identity(J):
    c = J.coords
    samples = [
        (c[0] * c[1] + c[-1] ** 2, c[0] ** 2 + c[1], c[1] * c[-1] + 3 * c[0]),
        (c[0] ** 2 * c[-1], c[1] ** 2 + c[0], c[0] + c[1] * c[-1] ** 2),
    ]
    for f, g, h in samples:
        jac = (J.bracket(f, J.bracket(g, h))
               + J.bracket(g, J.bracket(h, f))
               + J.bracket(h, J.bracket(f, g)))
        if sp.simplify(jac) != 0:
            return False
    return True


def _contact_with_sign(sign):
    J = ex.contact(1)
    return jacobi(J.dim, J.Lambda, sign * J.E, J.coords)


CONSISTENCY_CASES = [
    ("contact E=-dz", lambda: _contact_with_sign(+1), True),
    ("contact E=+dz", lambda: _contact_with_sign(-1), False),
    ("2D example", lambda: ex.jacobi_two_dimensional(), True),
    ("2D bad E", lambda: jacobi(2, sp.Matrix([[0, 1], [-1, 0]]),
                                sp.Matrix([x * y, 0]), [x, y]), False),
    ("3D example", lambda: ex.jacobi_three_dimensional(), True),
    ("3D Rigid Body", lambda: ex.jacobi_Rigid_Body([1, 1, -2], [1, 2, 3]), True),
]


@pytest.mark.parametrize("name,build,expected", CONSISTENCY_CASES,
                         ids=[c[0] for c in CONSISTENCY_CASES])
def test_is_jacobi_agrees_with_bracket(name, build, expected):
    J = build()
    assert _bracket_satisfies_jacobi_identity(J) is expected
    assert J.is_jacobi() is expected
