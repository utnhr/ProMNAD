#!-*- coding: utf-8 -*-

import constants
from tbfmodule import tbf

def propagate_tbfs():

    # construct TBF Hamiltonian

    n_tbf = tbf.total_tbf_count

    S = np.zeros( (n_tbf, n_tbf) )
    H = np.zeros( (n_tbf, n_tbf) )

    for i_tbf in range(n_tbf):
        
        guy_i = tbf.tbfs[i_tbf]

        if not guy_i.is_alive:
            continue

        for j_tbf in range(i_tbf, n_tbf):

            guy_j = tbf.tbfs[j_tbf]

            if not guy_j.is_alive:
                continue

            g_ij = get_gaussian_overlap(guy_i, guy_j)

            S_ij = get_wf_overlap(guy_i, guy_j, gaussian_overlap = g_ij)
            H_ij = get_tbf_hamiltonian_element_BAT(guy_i, guy_j, gaussian_overlap = g_ij)
            T_ij = get_gaussian_NAcoupling_term(guy_i, guy_j, gaussian_overlap = g_ij)

            S[i,j] = S_ij
            S[j,i] = S_ij
            H[i,j] = H_ij
            H[j,i] = H_ij
    
    # update electronic part

    for guy in tbf.tbfs(): # 'guy' is an individual tbf
        
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

        H_el = -1.0j * tdnac
        for i_estate in range(n_estate):
            H_el[i_estate,i_estate] += estate_energies[i_estate]

        e_coeffs_tderiv = -1.0j * np.dot(H_el, e_coeffs)

        force = guy.e_part.get_force()
    
    # construct remaining terms in TBF equation of motion   

    
