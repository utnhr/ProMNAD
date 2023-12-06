#!-*- coding: utf-8 -*-

import constants
from tbfmodule import Tbf
from propagator import propagate

def propagate_tbfs(world):

    # construct TBF Hamiltonian

    n_tbf = world.get_total_tbf_count()

    S = np.zeros( (n_tbf, n_tbf) )
    H = np.zeros( (n_tbf, n_tbf) )

    for i_tbf in range(n_tbf):
        
        guy_i = world.tbfs[i_tbf]

        if not guy_i.is_alive:
            continue

        for j_tbf in range(n_tbf):

            guy_j = world.tbfs[j_tbf]

            if not guy_j.is_alive:
                continue

            g_ij = Tbf.get_gaussian_overlap(guy_i, guy_j)

            S_ij = Tbf.get_wf_overlap(guy_i, guy_j, gaussian_overlap = g_ij)
            H_ij = Tbf.get_tbf_hamiltonian_element_BAT(guy_i, guy_j, gaussian_overlap = g_ij)

            S[i,j] = S_ij
            #S[j,i] = S_ij
            H[i,j] = H_ij
            #H[j,i] = H_ij
    
    # update electronic part

    for guy in world.tbfs(): # 'guy' is an individual tbf
        
        if not guy.is_alive:
            continue

        t = guy.get_t()

        guy.e_part.update_position_velocity_time(
            guy.get_position(), guy.get_velocity(), t
        )

        guy.e_part.update_matrices()

        estate_energies = guy.get_estate_energies()

        tdnac = guy.e_part.get_tdnac()

        n_estate = guy.e_part.get_n_estate()

        H_el = -1.0j * H_DIRAC * tdnac
        for i_estate in range(n_estate):
            H_el[i_estate,i_estate] += estate_energies[i_estate]

        e_coeffs_tderiv = (-1.0j / H_DIRAC) * np.dot(H_el, e_coeffs)

        force = guy.e_part.get_force()
    
    # construct remaining terms in TBF equation of motion   

    # < \psi_m | d/dt | \psi_n >

    for i_tbf in range(n_tbf):
        
        guy_i = world.tbfs[i_tbf]

        if not guy_i.is_alive:
            continue

        for j_tbf in range(n_tbf):

            guy_j = world.tbfs[j_tbf]

            if not guy_j.is_alive:
                continue

            g_ij = Tbf.get_gaussian_overlap(guy_i, guy_j)

            val = Tbf.get_tbf_derivative_coupling(cls, guy_i, guy_j, g_ij)

            H[i,j] -= val

    # symmetrize S & hermitize H

    for i in range(n_tbf):
        for j in range(i, n_tbf):

            val = 0.5 * (S[i,j] + S[j,i])
            S[i,j] = val
            S[j,i] = val

            val = 0.5 * ( H[i,j] + np.conjugate(H[j,i]) )
            H[i,j] = val
            H[j,i] = np.conjugate(val)
    
    # time derivative of TBF coeffs
    
    tbf_coeffs = world.get_tbf_coeffs()

    tbf_coeffs_tderiv = (-1.0j / H_DIRAC) * np.dot(
        np.linalg.inv(S), np.dot(H, tbf_coeffs)
    )

    world.set_new_tbf_coeffs_tderiv(tbf_coeffs_tderiv)

    for i_tbf in range(n_tbf):

        guy = world.tbfs[i_tbf]

        # update position and momentum

        old_position = guy,get_old_position()
        old_velocity = guy,get_old_velocity()
        position     = guy,get_position()
        velocity     = guy.get_velocity()

        new_position = propagate(world.settings.propagator_type, world.settings.dt,
            r0=old_position, r1=position, v0=old_velocity, v1=velocity,
        )

        new_velocity = 

        # update TBF coefficients
        
        old_tbf_coeffs        = guy.get_old_tbf_coeffs()
        old_tbf_coeffs_tderiv = guy.get_old_tbf_coeffs_tderiv()
        tbf_coeffs            = guy.get_tbf_coeffs()
        tbf_coeffs_tderiv     = guy.get_tbf_coeffs_tderiv()

        new_tbf_coeffs = propagate(world.settings.propagator_type, world.settings.dt,
            r0=old_tbf_coeffs, r1=tbf_coeffs, v0=old_tbf_coffs_tderiv, v1=tbf_coeffs_tderiv,
        )

        # update electronic-part coefficients

        old_e_coeffs        = guy.e_part.get_old_e_coeffs()
        old_e_coeffs_tderiv = guy.e_part.get_old_e_coeffs_tderiv()
        e_coeffs            = guy.e_part.get_e_coeffs()
        e_coeffs_tderiv     = guy.e_part.get_e_coeffs_tderiv()

