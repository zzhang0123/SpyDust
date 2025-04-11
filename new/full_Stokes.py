import numpy as np

class mode1():
    def omega_mapping(Omega):
        return Omega

    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 
        '''
        return 1. - u

    def int_alignm_part(cos_theta_b):
        return 1. - cos_theta_b**2

    def ext_alignm_part_StokesI(cos_theta_L):
        return 2. * cos_theta_L**2 + 2.

    def ext_alignm_part_StokesQ(cos_theta_L, phi_L):
        return - 2. * (1 - cos_theta_L**2) * np.cos(2*phi_L)

    def ext_alignm_part_StokesU(cos_theta_L, phi_L):
        return 2. * (1 - cos_theta_L**2) * np.sin(2*phi_L)

    def ext_alignm_part_StokesV(cos_theta_L):
        return - 4. * cos_theta_L

class mode2(mode1):
    def omega_mapping(Omega, beta, cos_theta_b):
        return Omega * np.abs(1. + beta * cos_theta_b)

    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 
        '''
        return u

    def int_alignm_part(cos_theta_b):
        return (1. + cos_theta_b)**2 / 4.

class mode3(mode1):
    def omega_mapping(Omega, beta, cos_theta_b):
        return Omega * np.abs(1. - beta * cos_theta_b)

    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 
        '''
        return u

    def int_alignm_part(cos_theta_b):
        return (1. - cos_theta_b)**2 / 4.

class mode4(mode1):
    def omega_mapping(Omega, beta, cos_theta_b):
        return Omega * np.abs(beta * cos_theta_b)

    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 
        '''
        return u

    def ext_alignm_part_StokesI(cos_theta_L):
        return 1. - cos_theta_L**2

    def ext_alignm_part_StokesQ(cos_theta_L, phi_L):
        return (1. - cos_theta_L**2) * np.cos(2*phi_L)

    def ext_alignm_part_StokesU(cos_theta_L, phi_L):
        return - (1. - cos_theta_L**2) * np.sin(2*phi_L)

    def ext_alignm_part_StokesV(cos_theta_L):
        return 0.



class single_grain_Stokes():
    def __init__(self, ):
        self.mode1 = mode1()
        self.mode2 = mode2()
        self.mode3 = mode3()
        self.mode4 = mode4()

    def 
