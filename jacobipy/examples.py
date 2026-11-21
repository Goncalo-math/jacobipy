from .jacobi import jacobi, poisson
import sympy as sp


def symplectic(n):
    """
    Returns a symplectic structure of dimension 2n.
    """
    I = sp.eye(n)
    Z = sp.zeros(n)
    Pi = sp.Matrix([[Z, I], [-I, Z]])
    q = sp.symbols(f"q1:{n + 1}")
    p = sp.symbols(f"p1:{n + 1}")

    return poisson(dim=2 * n, Pi=Pi, coords=list(q) + list(p))

def Lotka_Volterra(n, A, coords = None):
    """
    Returns the Lotka-Volterra Poisson structure.
    """
    if coords is None:
        x = sp.symbols(f'x1:{n+1}')
        coords = x
    if len(coords) != n:
        raise ValueError("The length of coords must be equal to %d." % n)
    if A.shape != (n, n):
        raise ValueError("The shape of A must be (%d, %d)." % (n, n))

    if any(
            (A[i, j] + A[j, i]) != 0
            for i in range(n)
            for j in range(n)
        ):
        raise ValueError("The matrix A must be skew-symmetric.")

    Pi = sp.zeros(n)
    for i in range(n):
        for j in range(n):
            Pi[i, j] = A[i, j] * coords[i] * coords[j]

    return poisson(dim=n, Pi=Pi, coords=coords)

def Rigid_Body(coords = None):
    """
    Returns the rigid body Poisson structure.
    """
    if coords is None:
        x = sp.symbols(f'x1:{3+1}')
        coords = x
    if len(coords) == 1:
         coords = sp.symbols(f'{coords}1:{3 + 1}')
    elif len(coords) != 3:
                raise ValueError(
                    f"Expected {3} coordinates, "
                    f"but received {len(coords)}."
                )
    else:
        coords = coords

    Pi = sp.Matrix([[0, -coords[2], coords[1]], [coords[2], 0, -coords[0]], [-coords[1], coords[0], 0]])

    return poisson(dim=3, Pi=Pi, coords=coords)

def contact(n, coords = None):
    """
    Returns the contact Jacobi structure on R^(2n+1), with E = -d/dz.
    """

    if coords is None:
        q = sp.symbols(f'q1:{n+1}')
        p = sp.symbols(f'p1:{n+1}')
        z = sp.symbols('z')

        coords = q + p + (z,)

    if len(coords) != 2 * n + 1:
        raise ValueError("The length of coords must be equal to %d." % (2 * n + 1))

    Lambda = sp.zeros(2 * n + 1)
    for i in range(n):
        Lambda[i, n + i] = -1
        Lambda[n + i, i] = 1
        Lambda[n+i, 2*n] = coords[n+i]
        Lambda[2*n, n+i] = -coords[n+i]

    E = sp.zeros(2 * n + 1, 1)
    E[2 * n] = -1

    return jacobi(dim=2 * n + 1, Lambda=Lambda, E=E, coords=coords)

def jacobi_two_dimensional(coords = None):
    """
    Returns a two-dimensional Jacobi structure.
    """
    if coords is None:
            x, y = sp.symbols('x y')
            coords = [x, y]

    if len(coords) != 2:
        raise ValueError("The length of coords must be equal to 2." )

    Lambda = sp.Matrix([[0, coords[0]**2+coords[1]**2], [-coords[0]**2-coords[1]**2, 0]])
    E = sp.Matrix([[-2*coords[1]], [2*coords[0]]])

    return jacobi(dim=2, Lambda=Lambda, E=E, coords=coords)

def jacobi_three_dimensional(coords = None):
    """
    Returns a three-dimensional Jacobi structure.
    """
    if coords is None:
            x, y, z = sp.symbols('x y z')
            coords = [x, y, z]

    if len(coords) != 3:
            raise ValueError("The length of coords must be equal to 3." )

    Lambda = sp.Matrix([[0, 2*coords[1], 2*coords[2]], [-2*coords[1], 0, 0], [-2*coords[2], 0, 0]])
    E = sp.Matrix([[1], [0], [0]])

    return jacobi(dim=3, Lambda=Lambda, E=E, coords=coords)

def jacobi_four_dimensional(coords = None):
    """
    Returns a four-dimensional Jacobi structure.
    """
    if coords is None:
             x = sp.symbols(f'x1:{2+1}')
             y = sp.symbols(f'y1:{2+1}')

             coords = x + y

    if len(coords) != 4:
        raise ValueError("The length of coords must be equal to 4.")

    Lambda = sp.Matrix([[0, sp.cos(coords[2]), 0, 0], [-sp.cos(coords[2]), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    E = sp.Matrix([[sp.exp(coords[3])*coords[1]], [sp.exp(coords[3])*coords[0]], [0], [0]])
    return jacobi(dim=4, Lambda=Lambda, E=E, coords=coords)

def jacobi_Lotka_Volterra(a, d, f, Gamma = 0,coords = None):
    """
    Returns a two-dimensional Jacobi Lotka-Volterra type structure.
    """
    if coords is None:
            x, y = sp.symbols('x y')
            coords = [x, y]

    if len(coords) != 2:
        raise ValueError("The length of coords must be equal to 2." )

    GG = sp.simplify(Gamma)
    Lambda = sp.Matrix([[0, -a*coords[0]*coords[1]], [a*coords[0]*coords[1], 0]])
    Gamma_x = sp.diff(GG, coords[0])
    Gamma_y = sp.diff(GG, coords[1])
    E = sp.Matrix([[d*coords[0] + coords[0]*coords[1]*Gamma_y], [f*coords[1] - coords[0]*coords[1]*Gamma_x]])
    return jacobi(dim=2, Lambda=Lambda, E=E, coords=coords)

def jacobi_Rigid_Body(a, d, coords = None):
    """
    Returns a three-dimensional Jacobi rigid body type structure.
    """
    if coords is None:
            x = sp.symbols(f'x1:{3+1}')
            coords = x

    if len(coords) != 3:
        raise ValueError("The length of coords must be equal to 3." )

    if len(a) !=3 or len(d) !=3:
        raise ValueError("The length of a or d must be equal to 3.")
    if a[0]+a[1]+a[2]!=0:
        raise ValueError("a_1+a_2+a_3 must be equal to 0.")

    Lambda = sp.Matrix([[0, -coords[2], coords[1]], [coords[2], 0, -coords[0]], [-coords[1], coords[0], 0]])

    f1 = a[0]*coords[1]*coords[2] + d[2]*coords[1] + d[0]*coords[2]
    f2 = a[1]*coords[0]*coords[2] - d[2]*coords[0] + d[1]*coords[2]
    f3 = a[2]*coords[0]*coords[1] - d[0]*coords[0] - d[1]*coords[1]

    E = sp.Matrix([[f1],[f2],[f3]])

    return jacobi(dim=3, Lambda=Lambda, E=E, coords=coords)

