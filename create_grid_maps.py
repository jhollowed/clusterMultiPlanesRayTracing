import numpy as np
from astropy.table import Table
import h5py

import cfuncs as cf
import pdb
import matplotlib.pyplot as plt
import subprocess
import os
import pdb
import glob

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


class grid_map_generator():

    def __init__(self, inp, stdfe_exe = '/home/hollowed/repos/SDTFE/bin/dtfe'):
        '''
        This class implements functions for constructing lensing grid maps on a particle lightcone cutout.
        After initializing with a `halo_inputs` object, the LOS particle data should be read with 
        `read_cutout_particles()`, and the grid generated with `create_grid_maps_for_zs0`.

        Parameters
        ----------
        inp : halo_inputs instance
            A class instance of halo_inputs giving run parameters and read/write directories
        sdtfe_exe : string
            Location of STDFE executable to call for desnity estiamtion. Defaults to Cooley build
        '''

        self.dtfe_exe = stdfe_exe
        self.inp = inp
        self.out_file = h5py.File('{}/{}_{}_gmaps.hdf5'.format(
                                  self.inp.outputs_path, self.inp.halo_id, self.inp.zs0), 'w')
        self.nslices = self.inp.num_lens_planes
        self.pdir = self.inp.input_prtcls_dir
        
        self.zp_los = np.array([])
        for snapid in self.inp.snapid_list:
            self.zp_los = np.hstack([self.zp_los, 
                          np.fromfile('{0}/STEPCutout{1}/redshift.{1}.bin'.format(self.pdir,snapid), dtype = "f")])
        self.xxp_los = None
        self.yyp_los = None
        self.zzp_los = None


    def read_cutout_particles(self):
        '''
        Read in the particle positions and redshifts given in the cutout directory pointed to by the 
        halo input object, `self.inp`.
        '''
        
        self.xxp_los = np.array([])
        for snapid in self.inp.snapid_list:
            self.xxp_los = np.hstack([self.xxp_los, 
                          np.fromfile('{0}/STEPCutout{1}/x.{1}.bin'.format(self.pdir,snapid), dtype = "f")])
        self.yyp_los = np.array([])
        for snapid in self.inp.snapid_list:
            self.yyp_los = np.hstack([self.yyp_los, 
                          np.fromfile('{0}/STEPCutout{1}/y.{1}.bin'.format(self.pdir,snapid), dtype = "f")])
        self.zzp_los = np.array([])
        for snapid in self.inp.snapid_list:
            self.zzp_los = np.hstack([self.zzp_los, 
                          np.fromfile('{0}/STEPCutout{1}/z.{1}.bin'.format(self.pdir,snapid), dtype = "f")])
        self.tp_los = np.array([])
        for snapid in self.inp.snapid_list:
            self.tp_los = np.hstack([self.tp_los, 
                          np.fromfile('{0}/STEPCutout{1}/theta.{1}.bin'.format(self.pdir,snapid), dtype = "f")])
        self.pp_los = np.array([])
        for snapid in self.inp.snapid_list:
            self.pp_los = np.hstack([self.pp_los, 
                          np.fromfile('{0}/STEPCutout{1}/phi.{1}.bin'.format(self.pdir,snapid), dtype = "f")])

    
    def create_grid_maps_for_zs0(self, skip_sdens=True, output_dens_tiffs = False):
        '''
        Perform density estimation via DTFE and compute lensing quantities on the grid 
        for sources at ~infinity. Note: the output files will be closed after this function 
        executes, so if needed to be called again, this object will need to be re-initialized.

        Parameters
        ----------
        skip_sdens : boolean, optional
            Whether or not to attempt to skip the DTFE calculation by reading density outputs
            from destination pointed to by `inp` as `self.dtfe_path`. If `False`, will enforce that 
            `read_cutout_particles()` has been called by the user, and call the DTFE and write 
            result to file. If `True`, but no density file is found for read in, will implicitly 
            call `read_cutout_particles()` if needed and call the DTFE and write result to file.
            Defaults to `True`.
        output_sdens_tiffs : boolean or float
            Whether or not to output tiff images of the DTFE result. Can also be a `float`, in which
            case tiff images will be generated for all halos above that value in log(M), e.g. if
            output_sdens_tiffs = 15, then output images for all lens planes of cutouts for which the
            halo's mass is M >= 10^14 M_sun/h. Defaults to False.
        '''
        
        if(skip_sdens == False): 
            assert self.zp_los is not None, "read_cutout_particles() must be called before DTFE"

        for i in range(self.nslices):
            
            # get redshift bounds of lens plane
            lens_plane_bounds = [self.inp.lens_plane_edges[i], self.inp.lens_plane_edges[i+1]]

            # compute lensing quantities on the grid from ~infinity (zs0) 
            output_ar = self.grids_at_lens_plane(lens_plane_bounds, i, skip_sdens = skip_sdens)

            # write output to HDF5 
            if output_ar == 1:
                continue
            else:
                zl_ar = output_ar[0]
                zs_ar = output_ar[1]
                kappa0_ar = output_ar[2]
                alpha1_ar = output_ar[3]
                alpha2_ar = output_ar[4]
                shear1_ar = output_ar[5]
                shear2_ar = output_ar[6]
     
            zl_group = 'plane{}'.format(i)
            self.out_file.create_group(zl_group)
            self.out_file[zl_group]['zl'] = np.atleast_1d(zl_ar).astype('float32')
            self.out_file[zl_group]['zs'] = np.atleast_1d(zs_ar).astype('float32')
            self.out_file[zl_group]['kappa0'] = kappa0_ar.astype('float32')
            self.out_file[zl_group]['alpha1'] = alpha1_ar.astype('float32')
            self.out_file[zl_group]['alpha2'] = alpha2_ar.astype('float32')
            self.out_file[zl_group]['shear1'] = shear1_ar.astype('float32')
            self.out_file[zl_group]['shear2'] = shear2_ar.astype('float32')
        self.out_file.close()


    def grids_at_lens_plane(self, lens_plane_bounds, idx, skip_sdens=False):
        '''
        Perform density estimation via DTFE and compute lensing quantities on the grid 
        for sources at ~infinity for the lens plane defined as the projected volume between
        the redshift limits `lens_plane_bounds`
       
        Parameters
        ----------
        lens_plane_bounds : 2-element list
            The redshift bounds of this lens plane; used to select particles in the plane
            for density estimation.

        idx : int
            integer index of this lens plane (lowest redhisft plane assumed to be index 0)
            
        skip_sdens : boolean, optional
            Whether or not to attempt to skip the DTFE calculation by reading density outputs
            from destination pointed to by `inp` as `self.dtfe_path`. If `False`, call the DTFE 
            and write result to file. If `True`, but no density file is found for read in, will 
            implicitly call `read_cutout_particles()` if needed and call the DTFE and write result 
            to file. Defaults to `True`.

        Returns
        -------
        list of numpy arrays
            The resulting lensing quantities in the following form (all but the first) two
            elements of the list are arrays of length `ngp^2`:
            [lens redshift, source redshift, convergence, deflection component 1, deflection component 2, 
             shear component 1, shear component 2]

        '''
        
        print('\n---------- working on plane {}/{} ----------'.format(idx+1, self.inp.num_lens_planes))
        
        # write out/read in denisty file
        dtfe_file = '{}/plane{}_dtfe_input.bin'.format(self.inp.dtfe_path, idx)
        
        # read in particle data
        # Note: zp, zl, zs for particles, lens, sources
        plane_mask = np.logical_and(self.zp_los > lens_plane_bounds[0], self.zp_los <= lens_plane_bounds[1])
        zp = self.zp_los[plane_mask]
        zl_median = np.median(zp)
        mpin = np.ones(len(zp))*self.inp.mpp
            
        zs = self.inp.zs0
        ncc = self.inp.nnn
        bsz_mpc = self.inp.bsz_mpc
        bsz_arc = self.inp.bsz_arc
       
        # manually toggle the noskip boolean to force density calculation and ignore skip_sdens
        noskip = False
        if(skip_sdens == True and noskip == False): 
            try:
                # read in density result
                print('reading density')
                sdens_cmpch = np.fromfile('{}.rho.bin'.format(dtfe_file))
            except FileNotFoundError: 
                print('Cant skip density calculation for plane {}; doesnt exist'.format(idx))
                noskip = True
        
        if(skip_sdens == False or noskip == True):

            print('extrcting lens plane from LOS')
            xp = self.xxp_los[plane_mask]
            yp = self.yyp_los[plane_mask]
            zp = self.zzp_los[plane_mask]
            tp = self.tp_los[plane_mask]
            pp = self.pp_los[plane_mask]
            print('{} particles'.format(len(xp)))
           
            # check has at least minimum particles
            if len(xp) < 10:
                return 1
            
            # compute fov-centric coordinates:
            # x3 in Mpc/h along the LOS
            # x1 azimuthal angular coord in arcsec
            # x2 coaltitude angular coord in arcsec
            #
            # note that the coordinates transerve to the line of sight are angular, 
            # while the coordinate along the los is cartesian; this will result in density
            # estimations per lens plane that are uniform in angular resolution, but 
            # increase in spatial resolution with decreasing redshift. Is that fine? I dunno.
            #
            # uncommenting the second xin1 and xin2 declarations below will put all 
            # coordinates in Mpc/h, but that seems like it will screw up ray-tracing
            # given the cutout geometry

            xo3 = np.linalg.norm(np.vstack([xp, yp, zp]), axis=0)
            xc3 = (np.max(xo3)+np.min(xo3))*0.5
            x3in = xo3 - xc3
            
            xo1 = tp
            xc1 = (np.max(xo1)+np.min(xo1))*0.5
            x1in = xo1 - xc1
            #x1in = np.sin((xo1-xc1)/cf.apr) * xo3
            
            xo2 = pp
            xc2 = (np.max(xo2)+np.min(xo2))*0.5
            x2in = xo2 - xc2
            #x2in = np.sin((xo2-xc2)/cf.apr) * xo3
            
            #---------------------------------------
            # Calculate convergence maps
            #
            
            # ------ do density estiamtion via system call to SDTFE exe ------
     
            # x, y, z in column major
            dtfe_input_array = np.ravel(np.vstack([x1in, x2in, x3in]))
            dtfe_input_array.astype('f').tofile(dtfe_file)
            
            # Usage: dtfe [ path_to_file n_particles grid_dim center_x center_y center_z 
            #               field_width(arcsec) field_depth(Mpc/h) particle_mass mc_box_width 
            #               n_mc_samples sample_factor ]
            dtfe_args = ["%s"%s for s in 
                         [self.dtfe_exe,
                          dtfe_file, len(mpin), ncc, 0, 0, 0, 
                          bsz_arc*0.95, np.max(x3in)-np.min(x3in), self.inp.mpp, 
                          bsz_arc/ncc/4, 4, 1.0
                         ]
                        ]
            print(dtfe_args)
            
            # call process
            with cd(self.inp.dtfe_path):
                subprocess.run(dtfe_args)
            
            # read in result
            sdens_cmpch = np.fromfile('{}.rho.bin'.format(dtfe_file))
        
        
        print('computing convergence')
        sdens_cmpch = sdens_cmpch.reshape(ncc, ncc)
        kappa = sdens_cmpch*(1.0+zl_median)**2.0/cf.sigma_crit(zl_median,zs)
         
        #---------------------------------------
        # Calculate deflection maps
        #

        if lens_plane_bounds[0] < self.inp.halo_redshift and \
           lens_plane_bounds[1] > self.inp.halo_redshift:
            # The Snapshot contains the main halo.
            BoundaryCondition = "Isolated"
        else:
            BoundaryCondition = "Periodic"

        if BoundaryCondition == "Periodic":
            # if mean(mass) ~ mean(Universe)
            alpha1, alpha2 = cf.call_kappa0_to_alphas_p(kappa, self.inp.bsz_arc, ncc)
        elif BoundaryCondition == "Isolated":
            # if mean(mass) >> mean(Universe)
            alpha1, alpha2 = cf.call_kappa0_to_alphas(kappa, self.inp.bsz_arc, ncc)
        else:
            print("You should define the Boundary Condition first!!!")
        
        #---------------------------------------
        # Calculate higher order lensing maps
        #

        print('computing defelctions and shears')
        al11, al12 = np.gradient(alpha1, self.inp.dsx_arc)
        al21, al22 = np.gradient(alpha2, self.inp.dsx_arc)
        # mua = 1.0/(1.0 - (al11 + al22) + al11*al22 - al12*al21)
        shear1 = 0.5*(al11 - al22)
        shear2 = 0.5*(al12 + al21)

        return zl_median, zs, kappa, alpha1, alpha2, shear1, shear2
