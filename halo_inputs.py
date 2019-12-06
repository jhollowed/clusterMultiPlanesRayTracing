import os
import pdb
import glob
import shutil
import numpy as np
from astropy.cosmology import WMAP7

import cosmology as cm


# ======================================= base inputs class ============================================

class halo_inputs():
    def __init__(self, halo_cutout_parent_dir, output_dir, halo_id = None, cosmo=None, sim=None): 
    
        #----------------------------------- inputs paths --------------------------------------

        self.input_prtcls_dir = halo_cutout_parent_dir
        self.halo_prop_file = '{}/properties.csv'.format(self.input_prtcls_dir)
        self.halo_props = np.genfromtxt(self.halo_prop_file, delimiter=',', names=True)
        if(halo_id is None):
            self.halo_id = halo_cutout_parent_dir.split('halo_')[-1]
        else: self.halo_id = halo_id 

        #----------------------------------- get/set simulation parameters --------------------

        if(cosmo is not None): 
            cm.update_cosmology(cosmo)
        self.cosmo = cm.cosmo
        if(sim is not None):
            cm.update_sim(sim)
        self.sim = cm.sim
        
        #--------------------------------- cutout quantities -----------------------------------
        
        self.halo_redshift = float(self.halo_props['halo_redshift'])
        self.halo_mass = int(self.halo_props['sod_halo_mass'])
        self.bsz = float(self.halo_props['boxRadius_arcsec']*2)/3600. # degree
        self.bsz_mpc = float(self.halo_props['boxRadius_Mpc']*2) # Mpc
        self.mpp = self.sim['mpp'] 
        
        #--------------------------------- lensing params -------------------------------------

        self.nnn =1024
        self.dsx = self.bsz/self.nnn
        self.bsz_arc = self.bsz*3600.
        self.dsx_arc = self.dsx*3600.
        self.zs0 = 10.0

        #self.mpp = self.mpp * 100 # uncomment for downsampled inputs
        self.npad = 5
    
        # gen grid points
        ds = self.bsz_arc/self.nnn
        x1 = np.linspace(0,self.bsz_arc-ds,self.nnn)-self.bsz_arc/2.0+ds/2.0
        x2 = np.linspace(0,self.bsz_arc-ds,self.nnn)-self.bsz_arc/2.0+ds/2.0
        self.xi1, self.xi2 = np.meshgrid(x1,x2)

        #--------------------------------- outputs --------------------------------------------

        self.outputs_path = output_dir
        self.dtfe_path = self.outputs_path + "/dtfe_dens/"
        self.xj_path = self.outputs_path + "/xj/"

        # create dirs, copy properties file if necessary
        for path in [self.outputs_path, self.dtfe_path, self.xj_path]:
            if not os.path.exists(path):
                    os.makedirs(path)
        if( len(glob.glob('{}/properties.csv'.format(self.outputs_path))) == 0):
            shutil.copyfile(self.halo_prop_file, '{}/properties.csv'.format(self.outputs_path))


# ======================================= single lens place inputs =====================================

class single_plane_inputs(halo_inputs):
    def __init__(self, halo_cutout_parent_dir, output_dir, halo_id = None, cosmo=None, sim=None):
        halo_inputs.__init__(self, halo_cutout_parent_dir, output_dir, halo_id, cosmo, sim)
        self.num_lens_planes = 1


# ======================================= multi lens place inputs =====================================

class multi_plane_inputs(halo_inputs): 
    def __init__(self, halo_cutout_parent_dir, output_dir, min_depth=0, max_depth = None, 
                 safe_zone=20.0, mean_lens_width=70, halo_id = None, cosmo=None, sim=None):
        
        halo_inputs.__init__(self, halo_cutout_parent_dir, output_dir, halo_id, cosmo, sim)

        #-------------------------------- cutout quantities ------------------------------------
        
        self.mean_lens_width = mean_lens_width
        self.halo_shell = int(self.halo_props['halo_lc_shell'])
        self.snapid_list = np.array([int(s.split('Cutout')[-1]) for s in 
                                     glob.glob('{}/*Cutout*'.format(self.input_prtcls_dir))])
        self.snapid_redshift = 1 / np.linspace(1/(self.sim['z_init']+1), 1, self.sim['sim_steps'])[self.snapid_list] - 1

        # trim to depth given by max_depth
        comv = self.cosmo.comoving_distance
        if(max_depth is not None):
            depth_mask = self.snapid_redshift <= max_depth
            self.snapid_list = self.snapid_list[depth_mask]
            self.snapid_redshift = self.snapid_redshift[depth_mask]
        self.max_redshift = max(self.snapid_redshift)
        self.depth_mpc = comv(self.max_redshift).value

        #-------------------------------- define lens planes -----------------------------------
        
        # define lens plane edges
        self.num_lens_planes = int(self.depth_mpc / self.mean_lens_width)
        self.lens_plane_edges = np.linspace(0, self.max_redshift, self.num_lens_planes+1)
        
        # remove any lens plane edge that is within {safe_zone}Mpc of the halo redshift
        bad_edges = []
        for i in range(len(self.lens_plane_edges)):
            if( abs(comv(self.halo_redshift).value-comv(self.lens_plane_edges[i]).value) <= safe_zone):
                bad_edges.append(i)
        for i in bad_edges:
            self.lens_plane_edges = np.delete(self.lens_plane_edges, i)
            self.num_lens_planes -= 1
        
        # trim off lens planes below min_depth (for the use case that there is an empty LOS 
        # except for at some specific location, as in the simple NFW test provided)
        self.lens_plane_edges = self.lens_plane_edges[self.lens_plane_edges >= min_depth]
        self.num_lens_planes = len(self.lens_plane_edges)-1


