"""Tests for the RCWA Green's tensor implementation in dolphindes.maxwell.greens_rcwa.py."""

import numpy as np
import pytest

from dolphindes.geometry import CartesianFDFDGeometry, PeriodicLayerGeometry
from dolphindes.maxwell import TM_FDFD, get_RCWA_G_DD

# from dolphindes.geometry import PeriodicLayerGeometry
# from dolphindes.maxwell import get_RCWA_G_DD


@pytest.fixture
def basic_geom():
    """Fixture providing a standard 1D periodic geometry."""
    return PeriodicLayerGeometry(
        pitch_x=1.0,
        pitch_y=1.0,
        thickness=2.0,
        Nz=40,
        num_harmonics_x=0,
        num_harmonics_y=0,
    )


def test_g_dd_symmetry(basic_geom):
    """
    Test 1: Verify the analytical symmetries of the dyadic tensor blocks.
    Diagonal components should be symmetric. Mixed longitudinal components
    should be antisymmetric due to the sgn(z-z') resulting from cross derivatives.
    """
    omega = 2.0 + 0.1j
    kx = 0.3
    ky = 0.4

    G_DD = get_RCWA_G_DD(omega, kx, ky, basic_geom)
    Nz = basic_geom.Nz

    # Extract the 3x3 block of Nz x Nz submatrices for the 0th harmonic
    G_dense = G_DD.toarray()[: 3 * Nz, : 3 * Nz]

    G_xx = G_dense[0:Nz, 0:Nz]
    G_yy = G_dense[Nz : 2 * Nz, Nz : 2 * Nz]
    G_zz = G_dense[2 * Nz : 3 * Nz, 2 * Nz : 3 * Nz]

    G_xy = G_dense[0:Nz, Nz : 2 * Nz]
    G_yx = G_dense[Nz : 2 * Nz, 0:Nz]

    G_xz = G_dense[0:Nz, 2 * Nz : 3 * Nz]
    G_zx = G_dense[2 * Nz : 3 * Nz, 0:Nz]

    # Transverse and Diagonal blocks are Symmetric
    np.testing.assert_allclose(G_xx, G_xx.T, err_msg="G_xx must be symmetric")
    np.testing.assert_allclose(G_yy, G_yy.T, err_msg="G_yy must be symmetric")
    np.testing.assert_allclose(G_zz, G_zz.T, err_msg="G_zz must be symmetric")

    # Cross-transverse is symmetric
    np.testing.assert_allclose(G_xy, G_yx, err_msg="G_xy must equal G_yx")
    np.testing.assert_allclose(G_xy, G_xy.T, err_msg="G_xy must be symmetric")

    # Mixed longitudinal blocks are antisymmetric
    np.testing.assert_allclose(
        G_xz, -G_xz.T, err_msg="G_xz must be exactly antisymmetric"
    )
    np.testing.assert_allclose(G_zx, G_xz, err_msg="G_zx must equal G_xz structurally")


def test_g_dd_analytical_limit(basic_geom):
    """
    Test 2: Compare the numerical extraction of the auto-coupling blocks
    to the exact analytical formulas at normal incidence (kx=0, ky=0).
    """
    omega = 1.5 + 0.2j
    dz = basic_geom.thickness / basic_geom.Nz

    G_DD = get_RCWA_G_DD(omega, 0.0, 0.0, basic_geom)
    Nz = basic_geom.Nz
    G_dense = G_DD.toarray()[: 3 * Nz, : 3 * Nz]

    G_xx = G_dense[0:Nz, 0:Nz]
    G_zz = G_dense[2 * Nz : 3 * Nz, 2 * Nz : 3 * Nz]

    # At normal incidence k_parallel = 0, so kz = omega
    kz = omega

    # Analytical evaluation of the exact pulse integral I_ii
    I_ii = (2j / kz) - 2 * (np.exp(1j * kz * dz) - 1) / (dz * kz**2)

    # Analytical G_xx[0,0]
    expected_Gxx_auto = (1j / (2 * kz)) * I_ii
    np.testing.assert_allclose(G_xx[0, 0], expected_Gxx_auto, rtol=1e-10)

    # Analytical G_zz[0,0] is strictly the depolarizing delta function
    expected_Gzz_auto = -1.0 / (omega**2)
    np.testing.assert_allclose(G_zz[0, 0], expected_Gzz_auto, rtol=1e-10)


def test_g_dd_fdfd_equivalence():
    """
    Test 3: Cross-validation against the 2D FDFD Maxwell Solver.
    We simulate a completely uniform 1D current sheet.
    RCWA applies P_x and computes E_x.
    FDFD applies J_z and computes E_z.
    By setting P_x = (i / (omega * dz)), they must output the exact same field profile.
    """
    omega = 2.0 + 0.1j
    dz = 0.05
    Ny_nonpml = 60
    Npml = 20

    # FDFD solves: -Laplacian(E_z) - omega^2 E_z = i * omega * J_z
    geom_fdfd = CartesianFDFDGeometry(
        Nx=3,
        Ny=Ny_nonpml + 2 * Npml,
        Npmlx=0,
        Npmly=Npml,
        dx=dz,
        dy=dz,
        bloch_x=0.0,
        bloch_y=0.0,
    )
    solver = TM_FDFD(omega=omega, geometry=geom_fdfd)

    # Create a uniform current sheet of J_z amplitude = 1 at the center
    sourcegrid = np.zeros((geom_fdfd.Nx, geom_fdfd.Ny), dtype=complex)
    source_y_idx = Npml + Ny_nonpml // 2
    sourcegrid[:, source_y_idx] = 1.0 / dz

    E_fdfd_full = solver.get_TM_field(sourcegrid)
    E_1d_fdfd = E_fdfd_full[0, Npml : Npml + Ny_nonpml]

    # RCWA equivalent wave eq: Curl(Curl(E_x)) - omega^2 E_x = omega^2 P_x
    # Matching the RHS yields: P_x = (i / omega) * J_z
    geom_rcwa = PeriodicLayerGeometry(
        pitch_x=1.0,
        pitch_y=1.0,
        thickness=Ny_nonpml * dz,
        Nz=Ny_nonpml,
        num_harmonics_x=0,
        num_harmonics_y=0,
    )
    G_DD = get_RCWA_G_DD(omega, 0.0, 0.0, geom_rcwa)

    G_xx = G_DD.toarray()[:Ny_nonpml, :Ny_nonpml]

    P_x_vec = np.zeros(Ny_nonpml, dtype=complex)
    P_x_vec[Ny_nonpml // 2] = 1j / (omega * dz)

    # In standard convention, E = omega^2 * G * P
    E_1d_rcwa = (omega**2) * (G_xx @ P_x_vec)

    # They may not agree exactly
    np.testing.assert_allclose(E_1d_rcwa, E_1d_fdfd, rtol=0.05, atol=1e-5)


def test_g_dd_light_line_exception(basic_geom):
    """
    Test 4: Ensure the solver aggressively aborts if passed a mathematically
    singular momentum corresponding to a perfectly grazing Rayleigh anomaly.
    """
    omega = 2.0 + 0.0j

    # Set kx exactly equal to omega to force kz = 0
    kx = 2.0
    ky = 0.0

    with pytest.raises(ValueError, match="leading to a singular Green's function"):
        get_RCWA_G_DD(omega, kx, ky, basic_geom)


def test_g_dd_multi_harmonic_assembly():
    """
    Test 5: Verify the dimensional expansion and block-diagonal sparsity
    structure when retaining multiple spatial harmonics.
    """
    N_x = 3
    N_y = 3
    Nz = 20

    geom_multi = PeriodicLayerGeometry(
        pitch_x=0.8,
        pitch_y=1.2,
        thickness=2.0,
        Nz=Nz,
        num_harmonics_x=N_x,
        num_harmonics_y=N_y,
    )

    omega = 3.0 + 0.1j
    kx = 0.1
    ky = 0.2

    G_DD = get_RCWA_G_DD(omega, kx, ky, geom_multi)

    total_harmonics = (2 * N_x + 1) * (2 * N_y + 1)
    expected_dim = 3 * Nz * total_harmonics

    assert G_DD.shape == (expected_dim, expected_dim), (
        f"Expected shape ({expected_dim}, {expected_dim}), got {G_DD.shape}"
    )

    expected_nnz = total_harmonics * (3 * Nz) ** 2
    actual_nnz = G_DD.nnz

    assert actual_nnz == expected_nnz, (
        f"Sparsity profile failed. Expected {expected_nnz} non-zeros for strict "
        f"block-diagonal structure, got {actual_nnz}. Off-diagonal leakage detected."
    )

    # 3. Verify reciprocal lattice generation maps correctly
    Gx, Gy = geom_multi.get_reciprocal_lattice()
    assert len(Gx) == total_harmonics, "Reciprocal lattice X array length mismatch."
    assert len(Gy) == total_harmonics, "Reciprocal lattice Y array length mismatch."

    print(f"Sparsity of G_DD is {G_DD.nnz / (G_DD.shape[0] * G_DD.shape[1])}")
