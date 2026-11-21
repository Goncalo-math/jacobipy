"""JacobiPy: symbolic calculus on Jacobi and Poisson manifolds."""

from .jacobi import jacobi, poisson
from . import examples

__all__ = ["jacobi", "poisson", "examples"]
__version__ = "0.1.0"
