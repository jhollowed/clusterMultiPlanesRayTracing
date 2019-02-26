#---------------------------------------------------------------------------------
# inputs
#

nnn = 1024
bsz = 2000/3600. # degree
dsx = bsz/nnn
bsz_arc = bsz*3600. #" # in the units of Einstein Radius
dsx_arc = dsx*3600.
zs0 = 10.0
mpp = 1148276137.0888093*1.6 # solMass/h
npad = 5

import numpy as np
import glob

def make_r_coor(bs, nc):
    ds = bs/nc
    x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2,x1 = np.meshgrid(x1,x2)
    return x1,x2

def make_c_coor(bs, nc):
    ds = bs/nc
    x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
    x1,x2 = np.meshgrid(x1,x2)
    return x1,x2

# xi1, xi2 = make_c_coor(bsz_arc, nnn)
xi1, xi2 = make_r_coor(bsz_arc, nnn)

#--------------------------------------------------------------------
# input paths
#

halo_info = "halo_4771392210198134784_0/"

input_prtcls_dir = "./data/lenses/prtcls/" + halo_info
halo_prop_file = '{}/properties.csv'.format(input_prtcls_dir)
halo_props = np.genfromtxt(halo_prop_file, delimiter=',')
halo_shell = int(halo_props[1])

sdens_path = "./data/lenses/sdens/"
gals_path = "./data/sources/points/"
snapid_list = [s.split('Cutout')[-1] for s in glob.glob('{}/*Cutout*'.format(input_prtcls_dir))]

#--------------------------------------------------------------------
# C lib directories
#

lib_path = "./lib/"

#---------------------------------------------------------------------------------
# grids_maps outputs
#

outputs_path = "/projects/DarkUniverse_esp/jphollowed/test4/"

gmaps_path = outputs_path + "grids_maps/"
alpha_path = gmaps_path + "alpha_maps/"
kappa_path = gmaps_path + "kappa_maps/"
shear_path = gmaps_path + "shear_maps/"
mu_path = gmaps_path + "mu_maps/"

#---------------------------------------------------------------------------------
# ray tracing outputs
#

xj_path = outputs_path + "xj/"
rmaps_path = outputs_path + "ray_traced_maps/"

#---------------------------------------------------------------------------------
# mocks outputs
#

mocks_path = outputs_path + "mocks/"
