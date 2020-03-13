"""
This module stores the cosmological model in use; any functions that depend on cosmology
are defined here, referring to an atropy cosmology object that is global, and can be set
to something other than the default (Outer Rim) by passing through inps.py
"""
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM


# ------------------ constants ------------------
# c in km/s
# G in Mpc/h (Msun/h)^-1 (km/s)^2
# apr is arsec per radian
vc = const.c.to(u.km/u.s)
G = const.G.to(u.Mpc/u.solMass * u.km**2 / u.s**2).value
apr = 3600 * 180/(np.pi) 


# ------------------ cosmology ------------------

OuterRim_params = FlatLambdaCDM(H0=71, Om0=0.220, Ob0=0.02258*(0.71**2), name='OuterRim')
cosmo = OuterRim_params
def update_cosmology(new_cosmo):
    global cosmo
    cosmo = new_cosmo

def calc_mpp(L, cosmo, Np):
    # calc mpp in solMass/h
    pc = cosmo.critical_density(0).to(u.solMass/u.Mpc**3)
    pm = pc*cosmo.Om0
    V = ((L/cosmo.h)*u.Mpc)**3
    return (pm * V) / Np

OuterRim_setup = {'z_init':200, 'sim_steps':500, 'L':3000, 'mpp':calc_mpp(3000, cosmo, 10240**3), 'name':'OuterRim'}
sim = OuterRim_setup
def update_sim(new_sim):
    global sim
    for key in new_sim:
        sim[key] = new_sim[key]


# ----------- functions of cosmology ------------

# Nan previously had a +1e-8 in Dc2() and Da2(), so check that if a zero error occurs in the cfuncs
def Dc(z):
    return cosmo.comoving_distance(z).value
def Dc2(z1,z2):
    return Dc(z2) - Dc(z2)

def Da(z):
    return cosmo.angular_diameter_distance(z).value
def Da2(z1,z2):
    return (Da(z2) - Da(z1))

def projected_rho_mean(z1, z2):
    # return the mean density of the unvierse integrated across redshifts 
    # z1 and z2, in comoving (M_sun/h)(Mpc/h)^(-3)
    pc0 = cosmo.critical_density(0).to(u.solMass/u.Mpc**3).value
    Om0 = cosmo.Om0
    rho_mean_0 = Om0 * pc0
    
    d1 = cosmo.comoving_distance(z1).value
    d2 = cosmo.comoving_distance(z2).value
    return rho_mean_0 * (d2-d1)


def sigma_crit(zl, zs):
    # return the critical surface density for the lensing geometry of a lens-source pair
    # at redshifts z1 and z2, respectively, in proper (M_sun) / (Mpc)**2 
    sigma_crit = vc*vc/(4.0*np.pi*G)*Da(zs)/(Da(zl)*Da2(zl,zs))
    return sigma_crit * (1+zl)**2


def Nz_Chang2014(z_bin_edges, case='fiducial', sys='blending'):
    """
    Computes the predicted LSST lensing source density, n_eff, in arcmin^-2, per Chang+2014
    """

    fiducial = {'a':1.24, 'z0':0.51, 'B':1.01, 'neff_raw':37, 'neff_blending':31, 'neff_masking':26}
    optimistic = {'a':1.23, 'z0':0.59, 'B':1.05, 'neff_raw':48, 'neff_blending':36, 'neff_masking':31}
    conservative = {'a':1.28, 'z0':0.41, 'B':0.97, 'neff_raw':24, 'neff_blending':22, 'neff_masking':18}
    p = {'fiducial':fiducial, 'optimistic':optimistic, 'conservative':conservative}

    a = p[case]['a']
    z0 = p[case]['z0']
    B = p[case]['B']
    neff = p[case]['neff_{}'.format(sys)]

    # integrate over P(z) to z=4
    z_samp_all = np.linspace(0, 4, 10000)
    Pz_all = z_samp_all **a * np.exp(-(z_samp_all/z0)**B)
    tot_all = np.trapz(Pz_all, z_samp_all, np.diff(z_samp_all)[0])
    
    # integrate over the specified bounds
    neff_bins = np.zeros(len(z_bin_edges)-1)
    for i in range(len(z_bin_edges)-1):
        z_samp = np.linspace(z_bin_edges[i], z_bin_edges[i+1], 10000)
        Pz = z_samp**a * np.exp(-(z_samp/z0)**B)
        tot = np.trapz(Pz, z_samp, np.diff(z_samp)[0])

        # normalize P(z) and multiply by total neff
        neff_bins[i] = tot/tot_all * neff

    return neff_bins
