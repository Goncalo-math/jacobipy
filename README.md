# JacobiPy

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Python module for (local) symbolic calculus on **Jacobi** and **Poisson**
manifolds, built on top of [SymPy](https://www.sympy.org).

A Jacobi structure on a manifold is a pair `(Λ, E)` formed by a bivector field
`Λ` and a vector field `E` whose bracket

```
{f, g} = Λ(df, dg) + f E(g) − g E(f)
```

satisfies the Jacobi identity. When `E = 0` this is a Poisson structure.
`JacobiPy` works in local coordinates, representing `Λ` as a skew-symmetric
`SymPy` matrix and `E` as a column vector.

| `jacobi` class | `poisson` class (subclass, `E = 0`) |
|---|---|
| `is_jacobi` | `is_poisson` |
| `sharp_Lambda` | `sharp_Pi` |
| `sharp_E` | `Pi` (property) |
| `bracket` | `bracket` |
| `hamiltonian_vf` | `hamiltonian_vf` |
| `is_Casimir` | `is_Casimir` |
| `poissonization` | `poissonization` |

> **Remark.** `is_jacobi`, `is_poisson` and `is_Casimir` need to decide whether
> a symbolic expression is identically zero. They rely on `sympy.simplify`, so
> they are limited by the usual theoretical and practical constraints of
> symbolic zero-testing.

## Conventions

For a Jacobi structure `(Λ, E)` in coordinates `x = (x¹, …, xⁿ)`:

* **Bracket:** `{f, g} = Σ Λⁱʲ ∂ᵢf ∂ⱼg + f E(g) − g E(f)`.
* **Jacobi conditions:** the bracket satisfies the Jacobi identity if and only if

  ```
  Σ_cyc Λⁱᵃ ∂ₐΛʲᵏ + (E ∧ Λ)ⁱʲᵏ = 0      and      L_E Λ = [E, Λ] = 0.
  ```

* **Hamiltonian vector field:** `X_H = Λ^#(dH) − H E`, where `Λ^#(α) = Λ · α`
  (matrix times column vector). In the Poisson case, `X_H(f) = {f, H}`.
* **Casimir function:** `{C, g} = 0` for all `g`, equivalently
  `Λ^#(dC) = C E` (i.e. `X_C = 0`) and `E(C) = 0`.
* **Poissonization:** the Poisson structure on `M × ℝ₊` (new coordinate `t`)
  given by `Π = (1/t) Λ + ∂ₜ ∧ E`.

## Installation

Clone the repository and install it in a virtual environment:

```bash
git clone https://github.com/Goncalo-math/jacobipy.git
cd jacobipy
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .
```

To also install the test dependencies use `pip install -e ".[dev]"`.

## Quick start

```python
>>> import sympy as sp
>>> from jacobipy import jacobi, poisson
>>> from jacobipy.examples import symplectic, Rigid_Body, contact
```

**Poisson structures**

```python
>>> P = symplectic(1)                  # canonical structure on R^2
>>> q1, p1 = P.coords
>>> P.is_poisson()
True
>>> P.bracket(q1, p1)
1
>>> P.hamiltonian_vf(p1**2 / 2 + q1**2 / 2)
Matrix([
[ p1],
[-q1]])

>>> R = Rigid_Body()                   # rigid body (so(3)*) structure
>>> x1, x2, x3 = R.coords
>>> R.is_Casimir(x1**2 + x2**2 + x3**2)
True
```

**Jacobi structures**

```python
>>> J = contact(1)                     # contact structure on R^3
>>> q1, p1, z = J.coords
>>> J.is_jacobi()
True
>>> J.bracket(q1, z)
-q1
>>> J.poissonization().is_poisson()    # Poisson structure on R^3 x R_+
True
```

**Building your own structure**

```python
>>> x, y = sp.symbols("x y")
>>> Lam = sp.Matrix([[0, x**2 + y**2], [-(x**2 + y**2), 0]])
>>> E = sp.Matrix([-2 * y, 2 * x])
>>> J = jacobi(dim=2, Lambda=Lam, E=E, coords=[x, y])
>>> J.is_jacobi()
True
```

If a check fails you can ask why:

```python
>>> bad = jacobi(2, sp.Matrix([[1, 0], [0, 0]]), sp.zeros(2, 1), [x, y])
>>> bad.is_jacobi(return_reason=True)
(False, 'Lambda is not skew-symmetric: Lambda[0, 0] + Lambda[0, 0] != 0')
```

## Examples module

`jacobipy.examples` contains ready-made structures:

| Function | Returns |
|---|---|
| `symplectic(n)` | canonical symplectic structure on ℝ²ⁿ (`poisson`) |
| `Lotka_Volterra(n, A)` | Lotka–Volterra Poisson structure `Πᵢⱼ = Aᵢⱼ xᵢ xⱼ` (`poisson`) |
| `Rigid_Body()` | rigid body Poisson structure on ℝ³ (`poisson`) |
| `contact(n)` | contact Jacobi structure on ℝ²ⁿ⁺¹, `E = −∂_z` (`jacobi`) |
| `jacobi_two_dimensional()` | a Jacobi structure on ℝ² |
| `jacobi_three_dimensional()` | a Jacobi structure on ℝ³ |
| `jacobi_four_dimensional()` | a Jacobi structure on ℝ⁴ |
| `jacobi_Lotka_Volterra(a, d, f, Gamma=0)` | Jacobi structure of Lotka–Volterra type on ℝ² |
| `jacobi_Rigid_Body(a, d)` | Jacobi structure of rigid body type on ℝ³ (`a₁+a₂+a₃ = 0`) |

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

## Bugs & contributions

Our issue tracker is at <https://github.com/Goncalo-math/jacobipy/issues>.
Please report any bugs that you find. Or, even better, if you are interested in
the project you can fork the repository on GitHub and open a pull request.

## Licence

[MIT licence](LICENSE).

## Authors

This work is developed and maintained by:

* Gonçalo Oliveira — [@Goncalo-math](https://github.com/Goncalo-math)

## Citation

Thanks for citing this work if you use it. See [`CITATION.cff`](CITATION.cff),
or use GitHub's “Cite this repository” button.
