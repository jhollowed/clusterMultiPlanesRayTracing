import numpy as np
import glob
import os

#--------------------------------------------------------------------
# C lib directories
#

lib_path = "./lib/"

class inputs():
 
    def __init__(self, halo_cutout_parent_dir):
    
        #--------------------------------------------------------------------
        # input paths
        #
        
        self.halo_info = halo_cutout_parent_dir
        self.input_prtcls_dir = "./data/lenses/prtcls/" + self.halo_info
        self.halo_prop_file = '{}/properties.csv'.format(self.input_prtcls_dir)
        self.halo_props = np.genfromtxt(self.halo_prop_file, delimiter=',', names=True)
        self.halo_shell = int(self.halo_props['halo_lc_shell'])
        self.bsz = float(self.halo_props['boxRadius_arcsec']*2)/3600. # degree

        self.sdens_path = "./data/lenses/sdens/"
        self.gals_path = "./data/sources/points/"
        self.snapid_list = [s.split('Cutout')[-1] for s in glob.glob('{}/*Cutout*'.format(self.input_prtcls_dir))]

        #---------------------------------------------------------------------------------
        # params
        #

        self.nnn = 1024
        self.dsx = self.bsz/self.nnn
        self.bsz_arc = self.bsz*3600. #" # in the units of Einstein Radius
        self.dsx_arc = self.dsx*3600.
        self.zs0 = 10.0
        self.mpp = 1148276137.0888093*1.6 # solMass/h
        self.npad = 5

        # xi1, xi2 = make_c_coor(bsz_arc, nnn)
        self.xi1, self.xi2 = self.make_r_coor(self.bsz_arc, self.nnn)

        #---------------------------------------------------------------------------------
        # grids_maps outputs
        #

        self.outputs_path = "/projects/DarkUniverse_esp/jphollowed/outerRim/raytraced_halos/{}".format(self.halo_info)
        self.gmaps_path = self.outputs_path + "grids_maps/"
        self.alpha_path = self.gmaps_path + "alpha_maps/"
        self.kappa_path = self.gmaps_path + "kappa_maps/"
        self.shear_path = self.gmaps_path + "shear_maps/"
        self.mu_path = self.gmaps_path + "mu_maps/"

        #---------------------------------------------------------------------------------
        # ray tracing outputs
        #

        self.xj_path = self.outputs_path + "xj/"
        self.rmaps_path = self.outputs_path + "ray_traced_maps/"

        #---------------------------------------------------------------------------------
        # mocks outputs
        #
        self.mocks_path = self.outputs_path + "mocks/"


        for path in [self.outputs_path, self.gmaps_path, self.alpha_path, self.kappa_path, 
                     self.shear_path, self.mu_path, self.xj_path, self.rmaps_path, self.mocks_path]:
            if not os.path.exists(path):
                    os.makedirs(path)
    
    def make_r_coor(self, bs, nc):
        ds = bs/nc
        x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
        x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
        x2,x1 = np.meshgrid(x1,x2)
        return x1,x2

    def make_c_coor(self, bs, nc):
        ds = bs/nc
        x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
        x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
        x1,x2 = np.meshgrid(x1,x2)
        return x1,x2
