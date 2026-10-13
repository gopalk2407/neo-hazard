"""Minimum Orbit Intersection Distance (MOID) between an orbit and Earth's."""
import numpy as np
from scipy.optimize import minimize

# Earth heliocentric elements, J2000 ecliptic (i ~ 0, so node is arbitrary).
EARTH = dict(a=1.00000261, e=0.01671123, i=0.0, om=0.0, w=102.93768193)


def _frame(i: float, om: float, w: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors P (to perihelion) and Q in the ecliptic frame."""
    i, om, w = np.radians([i, om, w])
    co, so, ci, si, cw, sw = np.cos(om), np.sin(om), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    P = np.array([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si])
    Q = np.array([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si])
    return P, Q


def orbit_points(a, e, i, om, w, f) -> np.ndarray:
    """Heliocentric positions (AU), shape (len(f), 3), at true anomalies f (rad)."""
    P, Q = _frame(i, om, w)
    r = a * (1 - e ** 2) / (1 + e * np.cos(f))
    return (r * np.cos(f))[:, None] * P + (r * np.sin(f))[:, None] * Q


def moid(a, e, i, om, w, n_obj: int = 720, n_earth: int = 360) -> float:
    """MOID vs Earth in AU, via a coarse grid search then local refinement.

    Only closed orbits (e < 1) are supported.
    """
    if not (0 <= e < 1 and a > 0):
        return float("nan")
    fo = np.linspace(0, 2 * np.pi, n_obj, endpoint=False)
    fe = np.linspace(0, 2 * np.pi, n_earth, endpoint=False)
    po = orbit_points(a, e, i, om, w, fo)
    pe = orbit_points(**EARTH, f=fe)
    d2 = ((po[:, None, :] - pe[None, :, :]) ** 2).sum(-1)

    def dist(x):
        return np.linalg.norm(orbit_points(a, e, i, om, w, x[:1]) - orbit_points(**EARTH, f=x[1:]))

    # Refine from the best grid cell of each object-orbit quarter (MOID can have
    # up to 4 local minima, typically near the two nodes).
    best = np.inf
    for chunk in np.array_split(np.arange(n_obj), 4):
        k = chunk[d2[chunk].min(1).argmin()]
        j = d2[k].argmin()
        res = minimize(dist, [fo[k], fe[j]], method="Nelder-Mead",
                       options=dict(xatol=1e-10, fatol=1e-12, maxiter=2000))
        best = min(best, res.fun, np.sqrt(d2[k, j]))
    return float(best)
