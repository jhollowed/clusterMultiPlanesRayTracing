import os
import pdb
import sys
import glob
import h5py
import subprocess
import halo_inputs
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

import cfuncs as cf
import cosmology as cm


# ------------------------- utilities ----------------------------------

def printflush(s):
    print(s)
    sys.stdout.flush()

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

# ======================================================================================================

class grid_map_generator():

    def __init__(self, inp, overwrite=False, stdout=True):
        '''
        This class implements functions for constructing lensing grid maps on a particle lightcone cutout.
        After initializing with a `halo_inputs` object, the LOS particle data should be read with 
        `read_cutout_particles()`, and the grid generated with `create_grid_maps_for_zs0`.

        Parameters
        ----------
        inp : halo_inputs instance
            An object instance of a class from halo_inputs (either single_plane_inputs or multi_plane_inputs),
            giving run parameters and read/write directories
        overwrite : bool
            Whether or not to overwrite old outputs. Defaults to False (will crash if HDF5 file exists)
        stdout : bool
            Whether or not to supress print statements (useful for parallel runs). `False` means all
            print statements will be suppressed. Defaults to `True`.
        '''
        
        # for parallel runs where output is restricted to one rank
        if(stdout == False): 
            self.print = lambda s: None
            self.c_out = open(os.devnull, 'w')
        else: 
            self.print = printflush
            self.c_out = None
        
        # define dtfe exec and get inputs
        self.density_estimator = None # is set by create_grid_maps_for_z0
        self.dtfe_exe = cf.sdtfe_exe
        self.inp = inp
        self.pdir = self.inp.input_prtcls_dir
        self.multiplane = isinstance(inp, halo_inputs.multi_plane_inputs)
        
        # create outputs
        mode = 'w' if overwrite else 'a'
        self.out_file = h5py.File('{}/{}_{}_gmaps.hdf5'.format(
                                  self.inp.outputs_path, self.inp.halo_id, self.inp.zs0), mode)
        self.print('created out file {}'.format(self.out_file.filename))
        
        # set number of lens planes 
        if(self.multiplane):
            #pfx is lens plane directory prefix
            self.pfx = glob.glob('{}/*Cutout*'.format(self.pdir))[0].split('Cutout')[0].split('/')[-1]
            self.nslices = self.inp.num_lens_planes
        else:
            # single lens plane
            self.nslices = 1
        
        self.xxp_los = np.array([])
        self.yyp_los = np.array([])
        self.zzp_los = np.array([])
        self.zp_los = np.array([])
        self.tp_los = np.array([])
        self.pp_los = np.array([])


    # ------------------------------------------------------------------------------------------------------


    def read_cutout_particles(self, inv_h = True):
        '''
        Read in the particle positions and redshifts given in the cutout directory pointed to by the 
        halo input object, `self.inp`.

        Parameters
        ----------
        inv_h : boolean, optional
            Whether or not to remove a factor of H_0/100 from the particle positons; if input is HACC simulation 
            data, then this should be True. Otherwise, it can be False, as long as the user has ensured that the 
            input data is in Mpc (no little-h dependency).
        '''
        
        columns = ['redshift', 'x', 'y', 'z', 'theta', 'phi']
        arrs = [self.zp_los, self.xxp_los, self.yyp_los, self.zzp_los, self.tp_los, self.pp_los]
        
        # particle data from all lens planes will be read into single flattened arrays for multi-plane case
        for i in range(len(arrs)):
            if(self.multiplane):
                for snapid in self.inp.snapid_list:
                    arrs[i] = np.hstack([arrs[i], np.fromfile('{0}/{2}Cutout{1}/{2}.{1}.bin'.format(self.pdir, 
                                                               snapid, self.pfx, columns[i]), dtype = "f")])
            else: 
                arrs[i] = np.fromfile('{0}/{1}.bin'.format(self.pdir, columns[i]), dtype = "f")
        self.zp_los, self.xxp_los, self.yyp_los, self.zzp_los, self.tp_los, self.pp_los = arrs

        if(inv_h == True):
            self.print('inv_h is ON')
            #self.zp_los = self.zp_los / self.inp.cosmo.h
            self.xxp_los = self.xxp_los / self.inp.cosmo.h
            self.yyp_los = self.yyp_los / self.inp.cosmo.h
            self.zzp_los = self.zzp_los / self.inp.cosmo.h
            self.tp_los = self.tp_los / self.inp.cosmo.h
            self.pp_los = self.pp_los / self.inp.cosmo.h


    # ------------------------------------------------------------------------------------------------------


    def create_grid_maps_for_zs0(self, subtract_mean=True, skip_sdens=True, output_dens_tiffs=False, 
                                 output_density=False, output_positions=False, density_estimator='dtfe'):
        '''
        Perform density estimation via DTFE and compute lensing quantities on the grid 
        for sources at ~infinity. Note: the output files will be closed after this function 
        executes, so if needed to be called again, this object will need to be re-initialized.

        Parameters
        ----------
        subtract_mean : boolean, optional
            Whether or not to subtract the mean density from the DTFE result to obtain the overdensity. 
            Defaults to True.
        skip_sdens : boolean, optional
            Whether or not to attempt to skip the DTFE calculation by reading density outputs
            from destination pointed to by `inp` as `self.dtfe_path`. If `False`, will enforce that 
            `read_cutout_particles()` has been called by the user, and call the DTFE and write 
            result to file. If `True`, but no density file is found for read in, will implicitly 
            call `read_cutout_particles()` if needed and call the DTFE and write result to file.
            Defaults to `True`.
        output_density : boolean or float
            Whether or not to output the DTFE result on the grid (in units of the input particle positions 
            and masses). Can also be a `float`, in which case the density will be output for all halos above 
            that value in log(M), e.g. if output_density = 14.0, then output images for all lens planes of 
            cutouts for which the halo's mass is M >= 10^14 M_sun/h. Defaults to False.
        output_sdens_tiffs : boolean or float
            Whether or not to output tiff images of the DTFE result. Can also be a `float`, in which
            case tiff images will be generated for all halos above that value in log(M), e.g. if
            output_sdens_tiffs = 14.0, then output images for all lens planes of cutouts for which the
            halo's mass is M >= 10^14 M_sun/h. Defaults to False.
        output_positions : boolean or float
            Whether or not to output angular positions of each gridpoint, in arcsec. Can also be a `float`, 
            in which case positions will be output for all halos above that value in log(M), e.g. if
            output_sdens_tiffs = 14.0, then output images for all lens planes of cutouts for which the
            halo's mass is M >= 10^14 M_sun/h. Defaults to False, for storage reasons; this only needs to be
            turned on if the user intends to do profile fitting directly on the individual lens plane grids, 
            otherwise the subsequest raytracing and interpolation will provide these quantities.
        density_estimator : string, optional
            String specifying which density estimator to call. Options are `'dtfe'`, in wich case an external
            system call will be made to the SDTFE C package (with the location of the executable specified
            in cfuncs.sdtfe_exe), and 'sph', in wich case the density is estimated via an internal C function
            to apply SPH smoothing kernels to the particle popilation. 
        '''
        
        if(skip_sdens == False): 
            assert len(self.zp_los != 0), "read_cutout_particles() must be called before DTFE"

        for i in range(self.nslices):
    
            # get redshift bounds of lens plane for multi-plane case
            if(self.multiplane):
                lens_plane_bounds = [self.inp.lens_plane_edges[i], self.inp.lens_plane_edges[i+1]]
                if lens_plane_bounds[0] < self.inp.halo_redshift and \
                   lens_plane_bounds[1] > self.inp.halo_redshift:
                       group_prefix = 'halo_plane'
                else: group_prefix = 'plane'
            else:
                lens_plane_bounds = None
                group_prefix = 'halo_plane'
    
            # set density output flag
            if(isinstance(output_density, float)):
                if( np.log10(self.inp.halo_mass) >= output_density): output_density = True
                else: output_density = False
            elif(not isinstance(output_density, bool)):
                raise Exception('`output_density must either be a float or a bool`')
            
            # set image output flag
            if(isinstance(output_dens_tiffs, float)):
                if( np.log10(self.inp.halo_mass) >= output_dens_tiffs): output_dens_tiffs = True
                else: output_dens_tiffs = False
            elif(not isinstance(output_dens_tiffs, bool)):
                raise Exception('`output_dens_tiffs must either be a float or a bool`')
            
            # set angular position output flag
            if(isinstance(output_positions, float)):
                if( np.log10(self.inp.halo_mass) >= output_positions): output_positions = True
                else: output_positions = False
            elif(not isinstance(output_positions, bool)):
                raise Exception('`output_positions must either be a float or a bool`')

            # check density estimator
            assert density_estimator in ['dtfe', 'sph'], 'Unknown density estimator {}. Options are'\
                                                         '\'dtfe\' or \'sph\''.format(density_estimator)
            self.density_estimator = density_estimator
            
            # compute lensing quantities on the grid from ~infinity (zs0)
            if(self.multiplane):
                output_ar = self._grids_at_lens_plane(i, lens_plane_bounds, subtract_mean,
                                                      skip_sdens, output_dens_tiffs)
            else:
                output_ar = self._grids_at_lens_plane(None, None, subtract_mean, skip_sdens, output_dens_tiffs)
            
            # write output to HDF5, with one group per lens plane
            zl_ar = output_ar[0]
            zs_ar = output_ar[1]
            kappa0_ar = output_ar[2]
            alpha1_ar = output_ar[3]
            alpha2_ar = output_ar[4]
            shear1_ar = output_ar[5]
            shear2_ar = output_ar[6]
            density_ar = output_ar[7]
     
            if(self.multiplane): zl_group = '{}{}'.format(group_prefix, i)
            else: zl_group = group_prefix
            self.out_file.create_group(zl_group)
            self.out_file[zl_group]['zl'] = np.atleast_1d(zl_ar).astype('float32')
            self.out_file[zl_group]['zs'] = np.atleast_1d(zs_ar).astype('float32')
            self.out_file[zl_group]['kappa0'] = kappa0_ar.astype('float32')
            self.out_file[zl_group]['alpha1'] = alpha1_ar.astype('float32')
            self.out_file[zl_group]['alpha2'] = alpha2_ar.astype('float32')
            self.out_file[zl_group]['shear1'] = shear1_ar.astype('float32')
            self.out_file[zl_group]['shear2'] = shear2_ar.astype('float32')
            if(output_density):
                self.out_file[zl_group]['density'] = density_ar.astype('float32')
            if(output_positions):
                theta1_ar = self.inp.xi1
                theta2_ar = self.inp.xi2
                self.out_file[zl_group]['x1'] = theta1_ar.astype('float32')
                self.out_file[zl_group]['x2'] = theta2_ar.astype('float32') 
                
        self.out_file.close()
    
    
    # ------------------------------------------------------------------------------------------------------


    def _grids_at_lens_plane(self, idx=None, lens_plane_bounds=None, subtract_mean=True,
                             skip_sdens=False, image_out=False):
        '''
        Perform density estimation via DTFE and compute lensing quantities on the grid 
        for sources at ~infinity for the lens plane defined as the projected volume between
        the redshift limits `lens_plane_bounds`
       
        Parameters
        ----------
        idx : int, optional
            integer index of this lens plane (lowest redhisft plane assumed to be index 0). This 
            is only optional in the single-plane usage case, in which case the value defaults to None.
        lens_plane_bounds : 2-element list, optional
            The redshift bounds of this lens plane; used to select particles in the plane
            for density estimation. This is only optional in the single-plane usage case, in
            which case the value defaults to None.
        subtract_mean : boolean, optional
            Whether or not to subtract the mean density from the DTFE result to obtain the overdensity. 
            Defaults to True.    
        skip_sdens : boolean, optional
            Whether or not to attempt to skip the DTFE calculation by reading density outputs
            from destination pointed to by `inp` as `self.dtfe_path`. If `False`, call the DTFE 
            and write result to file. If `True`, but no density file is found for read in, will 
            implicitly call `read_cutout_particles()` if needed and call the DTFE and write result 
            to file. Defaults to `True`.
        image_out : boolean, optional
            Whether or not to have the STDFE output tiff images of the density field. Defaults to False.

        Returns
        -------
        list of numpy arrays
            The resulting lensing quantities in the following form (all but the first) two
            elements of the list are arrays of length `ngp^2`:
            [lens redshift, source redshift, convergence, deflection component 1, deflection component 2, 
             shear component 1, shear component 2]

        '''
        
        
        # write out/read in denisty file
        if(self.multiplane):
            self.print('\n---------- working on plane {}/{} ----------'.format(idx+1, self.inp.num_lens_planes))
            dtfe_file = '{}/lensplane{}_dtfe_input.bin'.format(self.inp.dtfe_path, idx)
        else: 
            dtfe_file = '{}/lensplane_dtfe_input.bin'.format(self.inp.dtfe_path)
        
        # read in particle data, masking by desired lens plane boundaries in the multi-plane case
        # Note: zp, zl, zs for particles, lens, sources
        if(self.multiplane):
            plane_mask = np.logical_and(self.zp_los > lens_plane_bounds[0], self.zp_los <= lens_plane_bounds[1])
        else:
            plane_mask = np.ones(len(self.zp_los), dtype=bool)
        
        zp = self.zp_los[plane_mask]
        zl_median = np.median(zp)
        zs = self.inp.zs0
        mpin = np.ones(len(zp)) * self.inp.mpp
        ncc = self.inp.nnn
        bsz_mpc = self.inp.bsz_mpc
        bsz_arc = self.inp.bsz_arc
        
        # manually toggle the noskip boolean to force density calculation and ignore skip_sdens
        noskip = False
 
        # ----------------------------------- handle special cases --------------------------------------------

        if len(zp) == 1 and (self.tp_los[plane_mask]-324000 + self.pp_los[plane_mask])[0] == 0:
            # single point-mass perfectly centered at origin; solution is analytic
            # note that this explicitly assumes the current plane is the "halo plane", 
            # so of this happens by coincidence then big F
            kappa = np.zeros((ncc, ncc))
            sigma_proper = np.zeros((ncc, ncc))
            alpha1, alpha2 = cm.schwarzschild_deflection(mpin[0], self.inp.bsz_mpc, ncc)
            shear1, shear2 = self._shear_from_alphas(alpha1, alpha2)
            return zl_median, zs, kappa, alpha1, alpha2, shear1, shear2, sigma_proper
        
        elif len(zp) < 10:
            # very sparse particle distribution; conside empty
            empty_plane = np.zeros((ncc, ncc), dtype=np.float32)
            return zl_median, zs, empty_plane, empty_plane, empty_plane, empty_plane, empty_plane     

        # ---------------------- call SPH func or DTFE exec for density estimation ---------------------------
        
        if(skip_sdens == True and noskip == False and self.density_estimator=='dtfe'): 
            try:
                # read in density result
                self.print('reading DTFE density')
                sdens_comoving = np.fromfile('{}.rho.bin'.format(dtfe_file))
                sdens_comoving = sdens_comoving.reshape(ncc, ncc)
            except FileNotFoundError: 
                self.print('Cant skip density calculation for plane {}; doesnt exist'.format(idx))
                noskip = True
       
        if(skip_sdens == False or noskip == True):

            self.print('extrcting lens plane from LOS')
            xxp = self.xxp_los[plane_mask]
            yyp = self.yyp_los[plane_mask]
            zzp = self.zzp_los[plane_mask]
            tp = self.tp_los[plane_mask]
            pp = self.pp_los[plane_mask]
            self.print('{} particles'.format(len(xxp)))
   
            # compute fov-centric coordinates:
            # x1 azimuthal projected distance in comoving Mpc
            # x2 coaltitude projected distance in comoving Mpc
            # x3 in comoving Mpc along the LOS

            rp = np.linalg.norm(np.vstack([xxp, yyp, zzp]), axis=0)
            rp_center = (np.max(rp)+np.min(rp))*0.5
            plane_width = 0.95 * (np.tan(bsz_arc/cm.apr/2) * rp_center * 2)
            x3in = rp - rp_center
            
            tp_centered = tp - (np.max(tp)+np.min(tp))*0.5 
            x1in = np.sin((tp_centered)/cm.apr) * rp
            
            pp_centered = pp - (np.max(pp)+np.min(pp))*0.5 
            x2in = np.sin((pp_centered)/cm.apr) * rp
    
            # ------ do density estiamtion via SPH cfunc ------
            if(self.density_estimator == 'sph'):
                self.print('doing SPH density estimation')
                sdens_comoving = cf.call_sph_sdens_weight_omp(x1in, x2in, x3in, mpin, bsz_mpc, ncc)
            
            # ------ do density estiamtion via system call to SDTFE exe ------
            elif(self.density_estimator == 'dtfe'):
                self.print('doing DTFE density estimation')
                
                # x, y, z in column major
                dtfe_input_array = np.ravel(np.vstack([x1in, x2in, x3in]))
                dtfe_input_array.astype('f').tofile(dtfe_file)
                
                if(image_out == True): image_out = 1
                else: image_out = 0

                mc_box_width = plane_width/ncc/4
                plane_depth = np.max(x3in)-np.min(x3in)
                
                # Usage: dtfe [ path_to_file n_particles grid_dim center_x center_y center_z 
                #               field_width(Mpc) field_depth(Mpc) particle_mass mc_box_width 
                #               n_mc_samples sample_factor image_out? ]
                dtfe_args = ["%s"%s for s in 
                             [self.dtfe_exe,
                              dtfe_file, len(mpin), ncc, 0, 0, 0, 
                              plane_width, plane_depth, self.inp.mpp, 
                              mc_box_width, 4, 1.0, image_out
                             ]
                            ]
                self.print(dtfe_args)
               
                # call process
                with cd(self.inp.dtfe_path):
                    subprocess.run(dtfe_args, stdout=self.c_out)
                
                # read in result
                # sdens_comoving is comoving surface density of this lens plane in M_sun/Mpc^2
                sdens_comoving = np.fromfile('{}.rho.bin'.format(dtfe_file))
                sdens_comoving = sdens_comoving.reshape(ncc, ncc)

                # make sure we can recover particle mass from the density to 10%
                inferred_mpp = np.sum(sdens_comoving * (plane_width/ncc)**2) / len(x1in)
                fdiff_mpp = (self.inp.mpp - inferred_mpp) / inferred_mpp
                #assert abs(fdiff_mpp) <= 0.1, "particle mass not recoverable from density estimation!"
 
        # subtract mean density from sdens_comoving
        if(subtract_mean):
            rho_mean = cm.projected_rho_mean(np.min(zp), np.max(zp))
            mean_diff = np.mean(sdens_comoving) / rho_mean
            sdens_comoving -= rho_mean
            self.print('Measured/theory mean is {}'.format(mean_diff))

        # ----------------------- convergence maps and positions -----------------------------
        
        # sdens_comoving expected in comoving M_sun/Mpc^2
        # sigma_crit in proper M_sun/Mpc^2
        # factor of a^-2 in kappa to get sdens_comoving in proper area
        # convergence dimensionless 
        self.print('computing convergence')
        sigma_proper = sdens_comoving * (1+zl_median)**2
        kappa = sigma_proper / cm.sigma_crit(zl_median, zs)
         
        # ----------------------- defelection maps ------------------------------

        if(self.multiplane):
            if lens_plane_bounds[0] < self.inp.halo_redshift and \
               lens_plane_bounds[1] > self.inp.halo_redshift:
                # The Snapshot contains the main halo.
                BoundaryCondition = "Isolated"
            else:
                BoundaryCondition = "Periodic"
        else:
                # Single-lens plane case; plane is assumed to contain a halo.
                BoundaryCondition = "Isolated"

        if BoundaryCondition == "Periodic":
            # if mean(mass) ~ mean(Universe)
            alpha1, alpha2 = cf.call_kappa0_to_alphas_p(kappa, self.inp.bsz_arc, ncc)
        elif BoundaryCondition == "Isolated":
            # if mean(mass) >> mean(Universe)
            alpha1, alpha2 = cf.call_kappa0_to_alphas(kappa, self.inp.bsz_arc, ncc)
        
        # ------------------ higher-order lensing maps and return ----------------------------
        shear1, shear2 = self._shear_from_alphas(alpha1, alpha2)
        return zl_median, zs, kappa, alpha1, alpha2, shear1, shear2, sigma_proper
    
    
    # ------------------------------------------------------------------------------------------------------


    def _shear_from_alphas(self, alpha1, alpha2):
        '''
        Compute shear via gradient components of deflection maps
        
        Parameters
        ----------
        alpha1, alpha2 : 2D float arrays
            the deflection angles at each grid point; should be of shape (N,N), if N is
            the number of pixels on a side of the field of view

        Returns
        -------
        [shear1, shear2]: 2D float arrays
            the shears at each grid point, in shape (N,N)
        '''
        
        self.print('computing defelctions and shears')
        al11, al12 = np.gradient(alpha1, self.inp.dsx_arc)
        al21, al22 = np.gradient(alpha2, self.inp.dsx_arc)
        # mua = 1.0/(1.0 - (al11 + al22) + al11*al22 - al12*al21)
        shear1 = 0.5*(al11 - al22)
        shear2 = 0.5*(al12 + al21)
        
        return shear1, shear2
        
