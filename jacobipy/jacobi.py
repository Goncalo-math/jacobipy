import sympy as sp
from sympy import Matrix, shape


class jacobi:
    def __init__(self, dim, Lambda, E, coords):
        """
        Initializes a Jacobi structure.

        Parameters
        ----------
        dim : int
            The dimension of the manifold.
        Lambda : sympy.Matrix
            The skew-symmetric bivector matrix.
        E : sympy.Matrix
            The vector field.
        coords : list or tuple
            The coordinate variables.
        """
        self.dim = dim

        if len(coords) != self.dim:
            raise ValueError(
                f"Expected {self.dim} coordinates, "
                f"but received {len(coords)}."
            )
        self.coords = list(coords)

        if shape(Lambda) != (dim, dim):
            raise ValueError(
                "Lambda must be a square matrix of size %d." % dim
            )
        self.Lambda = Lambda

        if shape(E) != (dim, 1):
            raise ValueError(
                "E must be a column vector of size %d." % dim
            )
        self.E = E

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _skew_violation(self):
        """
        Returns the first pair (i, j) with Lambda[i, j] + Lambda[j, i] != 0,
        or None if Lambda is skew-symmetric.
        """
        n = self.dim
        for i in range(n):
            for j in range(n):
                if sp.simplify(self.Lambda[i, j] + self.Lambda[j, i]) != 0:
                    return i, j
        return None

    def _schouten_term(self, i, j, k, simplify=True):
        """
        Component (i, j, k) of

            sum_cyc Lambda^{ia} d_a Lambda^{jk}  +  (E ^ Lambda)^{ijk}.

        This is the coordinate form of the first Jacobi condition, with the
        sign fixed by the bracket {f, g} = Lambda(df, dg) + f E(g) - g E(f):
        it vanishes for all (i, j, k) iff that bracket satisfies the Jacobi
        identity (given the second condition). For E = 0 this is the usual
        Poisson condition [Lambda, Lambda] = 0.
        """
        term = 0
        for a in range(self.dim):
            term += (
                self.Lambda[i, a] * sp.diff(self.Lambda[j, k], self.coords[a])
                + self.Lambda[j, a] * sp.diff(self.Lambda[k, i], self.coords[a])
                + self.Lambda[k, a] * sp.diff(self.Lambda[i, j], self.coords[a])
            )

        term += (
            self.E[i] * self.Lambda[j, k]
            + self.E[j] * self.Lambda[k, i]
            + self.E[k] * self.Lambda[i, j]
        )

        return sp.simplify(term) if simplify else term

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_jacobi(self, simplify=True, return_reason=False):
        """
        Symbolically checks whether (Lambda, E) defines a Jacobi structure,
        i.e. whether the bracket

            {f, g} = Lambda(df, dg) + f E(g) - g E(f)

        satisfies the Jacobi identity. In coordinates this is equivalent to

            sum_cyc Lambda^{ia} d_a Lambda^{jk} + (E ^ Lambda)^{ijk} = 0
            (first condition, i.e. [Lambda, Lambda] = -2 E ^ Lambda in the
            Schouten convention  [Lambda, Lambda]^{ijk} = 2 sum_cyc ...)

        and

            L_E Lambda = [E, Lambda] = 0   (second condition).

        Parameters
        ----------
        simplify : bool
            Whether to simplify the resulting expressions.
        return_reason : bool
            If True, returns a tuple ``(bool, str)`` where the string
            explains why the check failed (empty if it succeeded).

        Returns
        -------
        bool or (bool, str)
            True if both Jacobi conditions hold identically.
        """
        n = self.dim

        def result(ok, reason=""):
            return (ok, reason) if return_reason else ok

        # Check skew-symmetry
        bad = self._skew_violation()
        if bad is not None:
            i, j = bad
            return result(
                False,
                f"Lambda is not skew-symmetric: "
                f"Lambda[{i}, {j}] + Lambda[{j}, {i}] != 0",
            )

        # First Jacobi condition (see _schouten_term)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    term = self._schouten_term(i, j, k, simplify)
                    if term != 0:
                        return result(
                            False,
                            f"First Jacobi condition failed at "
                            f"({i}, {j}, {k}): {term} != 0",
                        )

        # Second Jacobi condition:
        # [E, Lambda] = 0
        for i in range(n):
            for j in range(n):
                term = sum(
                    self.E[a] * sp.diff(self.Lambda[i, j], self.coords[a])
                    - self.Lambda[a, j] * sp.diff(self.E[i], self.coords[a])
                    - self.Lambda[i, a] * sp.diff(self.E[j], self.coords[a])
                    for a in range(n)
                )

                if simplify:
                    term = sp.simplify(term)

                if term != 0:
                    return result(
                        False,
                        f"Second Jacobi condition failed at "
                        f"({i}, {j}): {term} != 0",
                    )

        return result(True)

    def sharp_Lambda(self, alpha):
        """
        Computes the sharp map of Lambda applied to a 1-form alpha.

        Parameters
        ----------
        alpha : sympy.Matrix
            A column vector representing a 1-form.

        Returns
        -------
        sympy.Matrix
            The result of applying the sharp map of Lambda to alpha.
        """
        if shape(alpha) != (self.dim, 1):
            raise ValueError(
                "alpha must be a column vector of size %d." % self.dim
            )

        return self.Lambda * alpha

    def sharp_E(self, f):
        """
        Computes the derivative of a function f along the vector field E,
        i.e. E(f).

        Parameters
        ----------
        f : sympy expression
            A scalar function.

        Returns
        -------
        sympy expression
            The function E(f).
        """
        ff = sp.sympify(f)
        result = 0
        for i, coord in enumerate(self.coords):
            result += sp.diff(ff, coord) * self.E[i]
        return sp.simplify(result)

    def bracket(self, f, g):
        """
        Computes the Jacobi bracket of two functions f and g:

            {f, g} = Lambda(df, dg) + f E(g) - g E(f).

        Parameters
        ----------
        f : sympy expression
            A scalar function.
        g : sympy expression
            A scalar function.

        Returns
        -------
        sympy expression
            The Jacobi bracket {f, g}.
        """
        ff = sp.sympify(f)
        gg = sp.sympify(g)
        df = Matrix([sp.diff(ff, coord) for coord in self.coords])
        dg = Matrix([sp.diff(gg, coord) for coord in self.coords])

        return (df.T * self.sharp_Lambda(dg))[0] + ff * self.sharp_E(gg) - gg * self.sharp_E(ff)

    def hamiltonian_vf(self, H):
        """
        Computes the Hamiltonian vector field associated with a function H:

            X_H = Lambda^#(dH) - H E.

        Parameters
        ----------
        H : sympy expression
            A scalar function.

        Returns
        -------
        sympy.Matrix
            The Hamiltonian vector field X_H.
        """
        HH = sp.sympify(H)
        dH = Matrix([sp.diff(HH, coord) for coord in self.coords])
        return self.sharp_Lambda(dH) - HH * self.E

    def is_Casimir(self, C):
        """
        Checks if a function C is a Casimir function for the Jacobi structure,
        i.e. {C, g} = 0 for every function g. With the bracket used in this
        module this is equivalent to

            Lambda^#(dC) = C E   (that is, X_C = 0)   and   E(C) = 0.

        For a Poisson structure (E = 0) it reduces to Lambda^#(dC) = 0.

        Parameters
        ----------
        C : sympy expression
            A scalar function.

        Returns
        -------
        bool
            True if C is a Casimir function, False otherwise.
        """
        CC = sp.sympify(C)
        X_C = self.hamiltonian_vf(CC).applyfunc(sp.simplify)
        return bool(X_C.is_zero_matrix) and sp.simplify(self.sharp_E(CC)) == 0

    def poissonization(self, t=None):
        """
        Computes the Poissonization of the Jacobi structure, i.e. the Poisson
        structure on M x R_{>0} given by

            Pi = (1 / t) Lambda + d/dt ^ E.

        Parameters
        ----------
        t : sympy.Symbol, optional
            The extra coordinate. Defaults to ``Symbol('t')``.

        Returns
        -------
        poisson
            The Poissonization, with coordinates ``coords + [t]``.
        """
        n = self.dim
        if t is None:
            t = sp.Symbol('t')
        if t in self.coords:
            raise ValueError(
                f"The symbol {t} is already one of the coordinates; "
                "choose another symbol for the extra variable."
            )

        Lambda_p = Matrix.zeros(n + 1, n + 1)

        for i in range(n):
            for j in range(n):
                Lambda_p[i, j] = self.Lambda[i, j] / t

        for i in range(n):
            Lambda_p[i, n] = -self.E[i]
            Lambda_p[n, i] = self.E[i]

        return poisson(dim=n + 1, Pi=Lambda_p, coords=self.coords + [t])


class poisson(jacobi):
    def __init__(self, dim, Pi, coords):
        """
        Initializes a Poisson structure (a Jacobi structure with E = 0).

        Parameters
        ----------
        dim : int
            The dimension of the manifold.
        Pi : sympy.Matrix
            The skew-symmetric bivector matrix.
        coords : list or tuple
            The coordinate variables.
        """
        super().__init__(dim, Pi, Matrix.zeros(dim, 1), coords)

    @property
    def Pi(self):
        """
        Returns the Poisson bivector matrix.

        Returns
        -------
        sympy.Matrix
            The Poisson bivector matrix.
        """
        return self.Lambda

    def is_poisson(self, simplify=True, return_reason=False):
        """
        Symbolically checks whether Pi defines a Poisson structure,
        i.e. whether [Pi, Pi] = 0.

        Parameters
        ----------
        simplify : bool
            Whether to simplify the resulting expressions.
        return_reason : bool
            If True, returns a tuple ``(bool, str)`` where the string
            explains why the check failed (empty if it succeeded).

        Returns
        -------
        bool or (bool, str)
            True if the Jacobi condition holds identically.
        """
        # With E = 0 the second Jacobi condition is trivial and the first
        # one is exactly [Pi, Pi] = 0.
        return self.is_jacobi(simplify=simplify, return_reason=return_reason)

    def sharp_Pi(self, alpha):
        """
        Computes the sharp map of Pi applied to a 1-form alpha.

        Parameters
        ----------
        alpha : sympy.Matrix
            A column vector representing a 1-form.

        Returns
        -------
        sympy.Matrix
            The result of applying the sharp map of Pi to alpha.
        """
        return super().sharp_Lambda(alpha)

    def bracket(self, f, g):
        """
        Computes the Poisson bracket of two functions f and g.

        Parameters
        ----------
        f : sympy.Expr
            The first function.
        g : sympy.Expr
            The second function.

        Returns
        -------
        sympy.Expr
            The result of the Poisson bracket {f, g}.
        """
        return super().bracket(f, g)

    def hamiltonian_vf(self, H):
        """
        Computes the Hamiltonian vector field associated with a function H.

        Parameters
        ----------
        H : sympy.Expr
            The Hamiltonian function.

        Returns
        -------
        sympy.Matrix
            The Hamiltonian vector field X_H.
        """
        return super().hamiltonian_vf(H)

    def is_Casimir(self, C):
        """
        Checks if a function C is a Casimir function for the Poisson structure.

        Parameters
        ----------
        C : sympy.Expr
            The function to check.

        Returns
        -------
        bool
            True if C is a Casimir function, False otherwise.
        """
        return super().is_Casimir(C)
