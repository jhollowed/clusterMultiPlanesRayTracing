#---------------------------------------------------------------------------------
# inputs
#
# nnn = 2048
# bsz = 0.404733 # degree
# dsx = bsz/nnn

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
#---------------------------------------------------------------------------------
# snapid_list = ['487', '475', '464', '453', '442', '432', '421', '411',
               # '401', '392', '382', '373', '365', '355', '347', '338',
               # '331', '323', '315', '307', '300', '293', '286', '279',
               # '272', '266', '259', '253', '247']

# zs_list = [0.012883, 0.043016, 0.063007, 0.092599, 0.109616, 0.137126,
           # 0.167583, 0.197314, 0.220915, 0.255037, 0.283525, 0.317080,
           # 0.349731, 0.383357, 0.413571, 0.449458, 0.484233, 0.516544,
           # 0.551844, 0.594717, 0.633318, 0.667911, 0.711212, 0.752245,
           # 0.797511, 0.836996, 0.884260, 0.929488, 0.976569]

# snapid_list = ['475', '464', '453', '442', '432', '421', '411', '401', '392',
               # '382', '373', '365', '355', '338', '331', '323', '315', '307',
               # '300', '293', '286', '279', '272', '266', '259', '253', '247']
snapid_list = ['247', '272', '300', '331', '373', '411', '453',
               '253', '279', '307', '338', '382', '421', '464',
               '259', '286', '315', '355', '392', '432', '475',
               '266', '293', '323', '365', '401', '442', '487']


snapid_dict = {'487':0.012883, '475':0.043016, '464':0.063007, '453':0.092599,
               '442':0.109616, '432':0.137126, '421':0.167583, '411':0.197314,
               '401':0.220915, '392':0.255037, '382':0.283525, '373':0.317080,
               '365':0.349731, '355':0.383357, '347':0.413571, '338':0.449458,
               '331':0.484233, '323':0.516544, '315':0.551844, '307':0.594717,
               '300':0.633318, '293':0.667911, '286':0.711212, '279':0.752245,
               '272':0.797511, '266':0.836996, '259':0.884260, '253':0.929488,
               '247':0.976569}

nlpl = len(snapid_list)
#--------------------------------------------------------------------
# input paths
#

# halo_info = "halo_244849942116_z0.302586317062_MFOF8.00235555127e+14/"
#halo_info = "halo_207057992014_z0.842707037926_MFOF4.2993538669e+14/"
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
