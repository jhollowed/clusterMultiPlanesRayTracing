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
apr = 360*3600/(2*np.pi) 

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
    return (pm * V) / Np * cosmo.h

OuterRim_setup = {'z_init':200, 'sim_steps':500, 'L':3000, 'mpp':calc_mpp(3000, cosmo, 10240**3), 'name':'OuterRim'}
sim = OuterRim_setup
def update_sim(new_sim):
    global sim
    for key in new_sim:
        sim[key] = new_sim[key]


# ----------- functions of cosmology ------------

# Nan previously had a +1e-8 in Dc2() and Da2(), so check that if a zero error occurs in the cfuncs
def Dc(z):
    return cosmo.comoving_distance(z).value*cosmo.h
def Dc2(z1,z2):
    return Dc(z2) - Dc(z2)
def Da(z):
    return Dc(z)/(1+z)
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
    return rho_mean_0 * (d2-d1) / cosmo.h