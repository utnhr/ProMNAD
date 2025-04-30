#!-*- coding: utf-8 -*-

import utils
import numpy as np
from time import perf_counter_ns
from settingsmodule import load_setting
from copy import deepcopy
from math import sqrt
from pyscf import gto, scf, dft, grad

class pyscf_manager:
    
    def __init__(self, settings, atoms):

        self.gto = gto
        self.scf = scf
        self.dft = dft
        self.grad = grad

        self.mol = gto.Mole()
        #self.mol.unit = 'Angstrom'
        self.mol.unit = 'Bohr'
        self.mol.atom = atoms
        self.mol.basis = load_setting(settings, 'ao_basis')
        self.mol.symmetry = False
        self.mol.charge = load_setting(settings, 'charge')
        self.mol.verbose = 4
        self.mol.build()

        self.xc = load_setting(settings, 'xc')
        self.max_cycle = load_setting(settings, 'max_scf_cycle')

        self.ks = self.dft.KS(self.mol) # currently only RKS is supprted (mol.spin==0 assumed)
        self.ks.xc = self.xc
        self.ks.max_cycle = self.max_cycle
        
        self.level_shift = load_setting(settings, 'level_shift')

        return


    def update_geometry(self, coords):
        
        atoms = deepcopy(self.mol._atom)

        elems = [ atom[0] for atom in atoms ]

        new_atoms = [ [ elem, coord ] for elem, coord in zip(elems, coords) ]

        self.mol.atom = new_atoms

        self.mol.build()

        self.ks = self.dft.KS(self.mol) # currently only RKS is supprted (mol.spin==0 assumed)
        self.ks.xc = self.xc
        self.ks.max_cycle = self.max_cycle

        return


    def converge_scf(self, guess_dm = None, guess_mo = None, occ_mo = None,
                     check_stability = False, return_dm = False, return_mo = False):
        
        self.ks.level_shift = self.level_shift
        
        # if guess MO is given, use quasi-Newton
        if guess_mo is not None:
            self.ks = self.dft.KS(self.mol).newton()
            self.ks.kernel(guess_mo, occ_mo)
        else:
            self.ks.kernel(dm0 = guess_dm)
        
        if not self.ks.converged:
            utils.Printer.write_out('Trying second-order SCF.')
            self.ks.newton().run(self.ks.mo_coeff, self.ks.mo_occ)

        if check_stability:

            count = 0

            while True:

                mo_i, dummy1, stable_i, dummy2 = self.ks.stability(
                    internal=True, external=False, return_status=True
                )

                if stable_i:

                    break

                else:
                    
                    if count == 0:

                        utils.Printer.write_out('Trying second-order SCF.')
                        self.ks.newton().run(mo_i, self.ks.mo_occ)

                    elif count == 1:

                        guess_dm = self.scf.rhf.make_rdm1(mo_i, self.ks.mo_occ)

                        self.ks.level_shift = self.level_shift
                        self.ks.kernel(dm = guess_dm)

                    elif count > 1:
                        
                        break

                    count += 1

        if not self.ks.converged:
            #utils.stop_with_error('SCF failed to converge.')
            utils.Printer.write_out('WARNING! SCF NOT CONVERGED!!!')

        dm  = None
        mo  = None
        occ = None

        if self.ks.converged:

            if return_dm:
                dm = self.ks.make_rdm1()
            
            if return_mo:
                mo  = deepcopy(self.ks.mo_coeff)
                occ = deepcopy(self.ks.mo_occ)

        return dm, mo, occ


    def return_hamiltonian(self, rho, n_spin):
        
        hcore = self.scf.hf.get_hcore(self.mol)
        ts = perf_counter_ns()
        veff  = self.ks.get_veff(dm = rho)
        te = perf_counter_ns()
        print("Time for veff: %12.3f sec.\n" % ((te-ts)/1.0e+9)) ## Debug code

        H = hcore + veff

        if n_spin == 1:

            return np.array( [ H ] )

        else:

            utils.stop_with_error('Currently not compatible with open-shell systems.')

    
    def get_overlap_matrix(self):
        
        return deepcopy( self.mol.intor('int1e_ovlp') )


    def get_derivative_coupling(self, velocity_2d):
        
        ipovlp = deepcopy( self.mol.intor('int1e_ipovlp') )
        # int1e_ipovlp is derivative w.r.t. electronic coordinate -> minus of nuclear derivative. But...
        # <dp/dr|q> = -<dp/dR|q> = <p|dq/dR> ?

        atom_indices = [ int(line.split()[0]) for line in self.mol.ao_labels() ]

        n_ao = len(atom_indices)

        deriv_coupling = np.zeros( (n_ao, n_ao), dtype='float64' )

        for i_ao in range(n_ao):

            i_atom = atom_indices[i_ao]
            
            vel = velocity_2d[i_atom]

            deriv_coupling[i_ao, :] = np.dot(vel, ipovlp[0:3, i_ao, :])
        
        #deriv_coupling = -deriv_coupling.transpose() # ?
        #deriv_coupling = deriv_coupling.transpose() # ?
        #deriv_coupling = -deriv_coupling # ?
        #deriv_coupling[:,:] = 0.0 ## Debug code

        return deepcopy(deriv_coupling)


    def get_force_from_DM(self, n_spin, H_AO, rho_AO, Vinv, ortho_type):

        # Force from time-dependent electron density
        # X. Li et al., JCP, 123, 084106 (2005)
        # Assuming Lowdin orthogonalization (eq. 14)
        # V = S^(1/2), Vinv = S^(-1/2)

        utils.Timer.set_checkpoint_time('start_force')

        # currently, limited to restricted KS
        if n_spin > 1:
            utils.stop_with_error('Force calculation only available for RKS.')

        S = self.get_overlap_matrix().astype('complex128')

        F = H_AO[0,:,:]
        P = rho_AO[0,:,:]
        ## get converged Fock and 1RDM for debug
        #F = self.ks.get_hcore() + self.ks.get_veff()
        #P = self.ks.make_rdm1(self.ks.mo_coeff, self.ks.mo_occ)

        mf_grad = self.ks.nuc_grad_method()

        # Nuclear repulsion
        d_vn = mf_grad.grad_nuc(self.mol).astype('complex128')

        # Derivative generator of 1e term
        d_h1s = mf_grad.hcore_generator(self.mol)

        # Derivative of 2e term
        d_G2 = mf_grad.get_veff(self.mol, P).astype('complex128')

        # Derivative of AO overlap
        d_S = mf_grad.get_ovlp(self.mol).astype('complex128')

        # Eigenvalues & eigenvectors of S
        s_vals, s_vecs = np.linalg.eig(S)
        n_eig = len(s_vals)

        n_atom = self.mol.natm

        aoslices = self.mol.aoslice_by_atom()

        # buffer for total derivative
        d_E = np.zeros_like(d_vn)

        # Term 2 in r.h.s. of eq. 13
        term2 = np.zeros_like(d_vn)

        # Term 3 in r.h.s. of eq. 13
        term3 = np.zeros_like(d_vn)

        # prepare 1/(si^(1/2)+sj^(1/2)) matrix in eq. 14

        inv_sqrt = np.zeros( (n_eig, n_eig), dtype='complex128' )
        
        for i in range(n_eig):
            for j in range(n_eig):

                inv_sqrt[i,j] = 1.0 / ( sqrt(s_vals[i]) + sqrt(s_vals[j]) )

        # atom-wise part

        for i_atom in range(n_atom):

            atom_ID = i_atom + 1

            ps, pe = aoslices[i_atom, 2:]

            d_h1 = d_h1s(i_atom).astype('complex128') # generator for d_h1

            term2[i_atom,:] += np.einsum('xij,ji->x', d_h1, P, optimize='optimal')

            term2[i_atom,:] += 2.0 * np.einsum('xij,ji->x', d_G2[:,ps:pe,:], P[:,ps:pe], optimize='optimal') # x2 from bra,ket contributions, and x1/2
            #term2[i_atom,:] += 1.0 * np.einsum('xij,ij->x', d_G2[:,ps:pe,:], P[ps:pe,:], optimize='optimal') # x2 from bra,ket contributions, and x1/2

            if ortho_type == 'lowdin':
                
                d_V1 = np.einsum(
                    'pi,ij,ib,xbc,cj,jq->xpq',
                    s_vecs, inv_sqrt, s_vecs.transpose()[:,ps:pe], d_S[:,ps:pe,:], s_vecs, s_vecs.transpose(),
                    optimize='optimal',
                )
                #d_V2 = np.einsum(
                #    'pi,ij,ib,xcb,cj,jq->xpq',
                #    s_vecs, inv_sqrt, s_vecs.transpose(), d_S[:,ps:pe,:], s_vecs[ps:pe,:], s_vecs.transpose(),
                #    optimize='optimal',
                #)
                #d_V = d_V1 + d_V2
                d_V = d_V1

                # NOTE: Ptilde in JCP, 123, 084106 (2005) -> different from P?

                term3[i_atom,:] -= 2.0 * np.einsum('ij,jk,xkl,li->x', F, Vinv, d_V, P, optimize='optimal')
                term3[i_atom,:] -= 2.0 * np.einsum('ij,xjk,kl,li->x', P, d_V, Vinv, F, optimize='optimal')

                #term3[i_atom,:] -= 1.0 * np.einsum('ij,jk,xkl,li->x', F, P[:,ps:pe], d_S[:,ps:pe,:], P, optimize='optimal') # for converged P

            else:

                utils.stop_with_error("Unknown orthogonalization method %s .")

        d_E = d_vn + term2 + term3

        #print(d_vn.reshape(n_atom*3))
        #print(term2.reshape(n_atom*3))
        #print(term3.reshape(n_atom*3))
        #print(d_E.reshape(n_atom*3))

        elap = utils.Timer.set_checkpoint_time('end_force') - utils.Timer.get_checkpoint_time('start_force')

        utils.Printer.write_out(f"Force calculation: {elap:.6f} sec.\n")

        return -d_E.reshape(n_atom*3).real # force
