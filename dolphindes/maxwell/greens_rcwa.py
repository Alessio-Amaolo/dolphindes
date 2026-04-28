"""
Analytical RCWA Green's tensor operator for QCQP bounds.

Evaluates the exact pixel-basis Galerkin projection of the vacuum dyadic
Green's function onto the Fourier (xy) lattice.
"""

from typing import Tuple

import numpy as np
import scipy.sparse as sp

from dolphindes.geometry import PeriodicLayerGeometry
from dolphindes.types import ComplexArray


def _compute_galerkin_integrals(
    kz: complex, dz: float, Nz: int
) -> Tuple[ComplexArray, ComplexArray]:
    r"""
    Compute exact 1D double integrals of the exponential propagating kernel.

    Analytically evaluates I_ij = (1/dz) \int \int e^{i k_z |z-z'|} dz dz'
    and J_ij = (1/dz) \int \int sgn(z-z') e^{i k_z |z-z'|} dz dz' over the
    pixel basis functions for layers i and j.
    """
    I_mat = np.zeros((Nz, Nz), dtype=complex)
    J_mat = np.zeros((Nz, Nz), dtype=complex)

    # If kz is very small, use Taylor expansion that cancels the apparent singularity.
    if np.abs(kz) < 1e-10:
        I_ii = dz + 1j * kz * dz**2 / 3.0
        I_off = dz  # the next term is O(kz^3), way below numerical precision
    else:
        I_ii = 2j / kz - 2 * (np.exp(1j * kz * dz) - 1) / (dz * kz**2)
        I_off = (4 * np.sin(kz * dz / 2) ** 2) / (dz * kz**2)

    for i in range(Nz):
        for j in range(Nz):
            if i == j:
                I_mat[i, j] = I_ii
                J_mat[i, j] = 0.0
            else:
                dist = np.abs(i - j) * dz
                phase = np.exp(1j * kz * dist)
                I_mat[i, j] = I_off * phase

                sign = 1.0 if i > j else -1.0
                J_mat[i, j] = I_off * phase * sign

    return I_mat, J_mat


def _build_harmonic_block(
    omega: complex, kx: float, ky: float, dz: float, Nz: int
) -> ComplexArray:
    """
    Assemble the 3Nz x 3Nz dense dyadic block for a single spatial harmonic.

    Stacking order: [P_x(z_1...z_Nz), P_y(z_1...z_Nz), P_z(z_1...z_Nz)].
    """
    k_parallel_sq = kx**2 + ky**2
    kz_sq = omega**2 - k_parallel_sq
    kz = np.sqrt(kz_sq + 0j)

    if np.abs(kz) <= 1e-12:
        raise ValueError(
            f"kz is zero (kz={kz}), leading to a singular Green's function. "
            "This corresponds to a grazing incidence condition. "
            "If you are integrating over the BZ, this should be an integrable pole. "
            "If you have an objective where these modes do not contribute "
            "(such as NFRHT), you can likely just pick a better quadrature."
            "You can also consider adding a small imaginary part to omega to regularize"
            " the pole. If you actually need to evaluate at grazing incidence, "
            "what you should do is likely problem dependent."
        )

    # Enforce strictly outward/decaying radiation condition
    if np.imag(kz) < 0:
        kz = -kz

    I_mat, J_mat = _compute_galerkin_integrals(kz, dz, Nz)

    # Serious issues here with kz=0. Go back and check.
    prefactor = 1j / (2 * kz)
    w2 = omega**2

    # Diagonal and in-plane tensor components
    G_xx = prefactor * (I_mat - (kx * kx / w2) * I_mat)
    G_yy = prefactor * (I_mat - (ky * ky / w2) * I_mat)

    # The analytical integral of \delta(z-z') over pulse basis gives strictly -1/(w2)
    G_zz = prefactor * (I_mat - (kz * kz / w2) * I_mat) - np.eye(Nz) / (w2)

    # Off-diagonal transverse components
    G_xy = prefactor * (-(kx * ky / w2) * I_mat)
    G_yx = G_xy

    # Mixed longitudinal-transverse components
    G_xz = (-1j * kx / (2 * w2)) * J_mat
    G_zx = G_xz

    G_yz = (-1j * ky / (2 * w2)) * J_mat
    G_zy = G_yz

    row1 = np.hstack([G_xx, G_xy, G_xz])
    row2 = np.hstack([G_yx, G_yy, G_yz])
    row3 = np.hstack([G_zx, G_zy, G_zz])
    return np.vstack([row1, row2, row3])


def get_RCWA_G_DD(
    omega: complex,
    k_in_x: float,
    k_in_y: float,
    geometry: "PeriodicLayerGeometry",
) -> sp.csc_array:
    """
    Compute the block-diagonal analytical Green's function for the design region.

    Because the background is translationally invariant, transverse momenta are
    perfectly decoupled. The resulting matrix is strictly block-diagonal, mapping
    directly to `SparseSharedProjQCQP` for highly efficient factorizations.

    Parameters
    ----------
    omega : complex
        Operating frequency.
    k_in_x : float
        Incident transverse momentum in x.
    k_in_y : float
        Incident transverse momentum in y.
    geometry : PeriodicLayerGeometry
        Geometry definition containing pitch, thickness, and harmonic limits.

    Returns
    -------
    scipy.sparse.csc_array
        The block-diagonal Green's function matrix.
    """
    Gx, Gy = geometry.get_reciprocal_lattice()
    dz = geometry.thickness / geometry.Nz
    Nz = geometry.Nz

    blocks = []

    for gx, gy in zip(Gx, Gy):
        kx = k_in_x + gx
        ky = k_in_y + gy

        harmonic_block = _build_harmonic_block(omega, kx, ky, dz, Nz)
        blocks.append(harmonic_block)

    # Assembly as a sparse block-diagonal matrix
    G_DD_sparse = sp.block_diag(blocks, format="csc")

    return G_DD_sparse
