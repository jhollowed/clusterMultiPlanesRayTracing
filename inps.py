import os
import pdb
import glob
import numpy as np
from astropy.cosmology import WMAP7
lib_path = './lib/'

class inputs():
 
    def __init__(self, halo_cutout_parent_dir, output_dir, 
                 mean_lens_width=70, z_init=200, sim_steps=500, cosmo=WMAP7):
    
        #--------------------------------------------------------------------
        # input paths
        #
       
        self.input_prtcls_dir = halo_cutout_parent_dir
        self.halo_id = halo_cutout_parent_dir.split('halo_')[-1]
        self.halo_prop_file = '{}/properties.csv'.format(self.input_prtcls_dir)
        self.halo_props = np.genfromtxt(self.halo_prop_file, delimiter=',', names=True)
        
        #--------------------------------------------------------------------
        # cutout quantities
        #
        
        self.mean_lens_width = mean_lens_width
        self.halo_shell = int(self.halo_props['halo_lc_shell'])
        self.halo_redshift = float(self.halo_props['halo_redshift'])
        self.halo_mass = int(self.halo_props['sod_halo_mass'])
        self.bsz = float(self.halo_props['boxRadius_arcsec']*2)/3600. # degree
        self.bsz_mpc = float(self.halo_props['boxRadius_Mpc']*2) # Mpc
        self.snapid_list = [int(s.split('Cutout')[-1]) for s in 
                            glob.glob('{}/*Cutout*'.format(self.input_prtcls_dir))]

        #--------------------------------------------------------------------
        # define lens planes
        #
        
        comv = cosmo.comoving_distance
        self.max_redshift = 1 / np.linspace(1/(z_init+1), 1, sim_steps)[min(self.snapid_list)] - 1
        self.depth_mpc = comv(self.max_redshift).value
        self.num_lens_planes = int(self.depth_mpc / self.mean_lens_width)
        self.lens_plane_edges = np.linspace(0, self.max_redshift, self.num_lens_planes+1)
        
        # remove any lens plane edge that is within 10Mpc of the halo redshift
        bad_edges = []
        for i in range(len(self.lens_plane_edges)):
            if( abs(comv(self.halo_redshift).value - comv(self.lens_plane_edges[i]).value) <= 20.0):
                bad_edges.append(i)
        for i in bad_edges:
            self.lens_plane_edges = np.delete(self.lens_plane_edges, i)
            self.num_lens_planes -= 1
        
        #---------------------------------------------------------------------------------
        # lensing params
        #

        self.nnn = 1024
        self.dsx = self.bsz/self.nnn
        self.bsz_arc = self.bsz*3600.
        self.dsx_arc = self.dsx*3600.
        self.zs0 = 10.0
        self.mpp = 1148276137.0888093*1.6 # solMass/h
        self.npad = 5
    
        # gen grid points
        self.xi1, self.xi2 = self.make_r_coor(self.bsz_arc, self.nnn)

        #---------------------------------------------------------------------------------
        # outputs
        #

        self.outputs_path = output_dir
        self.dtfe_path = self.outputs_path + "/dtfe_dens/"
        self.xj_path = self.outputs_path + "/xj/"

        # create dirs
        for path in [self.outputs_path, self.dtfe_path, self.xj_path]:
            if not os.path.exists(path):
                    os.makedirs(path)

    
    def make_r_coor(self, bs, nc):
        ds = bs/nc
        x1 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
        x2 = np.linspace(0,bs-ds,nc)-bs/2.0+ds/2.0
        x2,x1 = np.meshgrid(x1,x2)
        return x1,x2
