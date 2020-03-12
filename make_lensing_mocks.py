import pdb
import sys
import h5py
import glob
import halo_inputs
import numpy as np
import pylab as pl
import cfuncs as cf
import halo_inputs as inp
from astropy.table import Table
from cosmology import Nz_Chang2014

# ------------------------- utilities ----------------------------------

def printflush(s):
    print(s)
    sys.stdout.flush()

# ======================================================================================================


class lensing_mock_generator():

    def __init__(self, inp, overwrite=False, stdout=True):
        '''
        This class implements functions for creating lensing mocks via interpolation on ray-traced maps.
        After initializing with a `halo_inputs` object, the raytraced lensing maps computed from the lens 
        plane density data should be read with `read_raytrace_planes()`, and the interpolation performed with 
        `make_lensing_mocks()`

        Parameters
        ----------
        inp : halo_inputs instance
            An object instance of a class from halo_inputs (either single_plane_inputs or multi_plane_inputs),
            giving run parameters and read/write directories
        overwrite : bool, optional
            Whether or not to overwrite old outputs. Defaults to False (will crash if HDF5 file exists)
        stdout : bool, optional
            Whether or not to supress print statements (useful for parallel runs). `False` means all
            print statements will be suppressed. Defaults to `True`.
        '''
        
        if(stdout == False): self.print = lambda s: None
        else: self.print = printflush

        self.inp = inp
        self.multiplane = isinstance(inp, halo_inputs.multi_plane_inputs)
        
        # grid points
        self.xx1 = self.inp.xi1
        self.xx2 = self.inp.xi2
        
        # point to raytrace result and define output file
        mode = 'w' if overwrite else 'a'
        self.raytrace_path = glob.glob('{}/{}_raytraced_maps.hdf5'.format(self.inp.outputs_path, self.inp.halo_id))
        self.out_file = h5py.File('{}/{}_lensing_mocks.hdf5'.format(self.inp.outputs_path, self.inp.halo_id), mode)
        self.print('created out file {}'.format(self.out_file.filename))

        self.raytrace_file = None
        self.source_planes = None
        self.source_plane_keys = None
    
    
    # ------------------------------------------------------------------------------------------------------
        
        
    def read_raytrace_planes(self):
        '''
        Read in raytraced maps pointed to by `self.inp.outputs_path`
        '''

        # read raytrace result
        self.raytrace_file = h5py.File(self.raytrace_path[0])
        
        # for multiplane case, make sure planes are in order of increasing redshift
        if(self.multiplane):
            source_plane_keys = np.array(list(self.raytrace_file.keys()))
            source_planes = np.array([int(s.split('plane')[-1]) for s in source_plane_keys])
            plane_order = np.argsort(source_planes) 
            self.source_plane_keys = source_plane_keys[plane_order]
        else:
            self.source_plane_keys = np.array(list(self.raytrace_file.keys()))
    
    
    # ------------------------------------------------------------------------------------------------------


    def make_lensing_mocks(self, nsrcs=Nz_Chang2014, n_places='rand', zs=None, vis_shears=False):
        """
        Performs interpolation on ray tracing maps, writing the result to an HDF file, the location
        of which is specified in `self.inp.outputs_path`. Note: the output files will be closed after
        this function executes, so if needed to be called again, this object will need to be re-initialized.
        
        Parameters
        ----------
        nsrcs : int or callable or None
            If passed as: 
            -- an int, then this is the number of sources to place in the fov. This is 
               useful if creating a mock at one source plane; if using many source planes, then the lower
               redshift will effectively be weighted more strongly (nsrcs is constant while the angular scale
               grows).
            -- a callable, then a function to compute the number distribution N(z) should be given, and
               will be called at each source plane redshift, where galaxies will be populated randomly in 
               angular space, with their count inferred from N(z). The callable is expected to return N(z)
               in armin^-2. Defaults to Nz_Chang2014().
        n_places:
            How to place the sources if nsrcs is an integer; options are:
            -- 'grid', which places the sources on a grid (this is effectively just a lower resolution version 
                of the ray-traced maps)
            -- 'rand', which places the sources by a uniform random ditribution in the angular coordinates
        zs : float array or None
            Redshifts of source planes to include. If None, use all source planes given in ray-tracing outputs
            for the halo specified in self.inp. Defaults to None.
        theta1 : float array or None **DEPRECATED**
            azimuthal angular coordinate of the sources, in arcsec. If None, then nsrcs is assumed to have
            been passed, and theta1 will be generated from a uniform random distribution on the fov. If not
            None, then nsrcs should be None, else will throw an error. Defaults to None.
        theta2 : float array or None **DEPRECATED**
            coalitude angular coordinate of the sources, in arcsec. If None, then nsrcs is assumed to have
            been passed, and theta1 will be generated from a uniform random distribution on the fov. If not
            None, then nsrcs should be None, else will throw an error. Defaults to None.
        """
        
        self.print('\n ---------- creating lensing mocks for halo {} ---------- '.format(self.inp.halo_id))
        assert self.raytrace_file is not None, "read_raytrace_planes() must be called before interpolation"
       
        # get source plane redshift edges if not passed
        if(zs is None): 
            zs = np.array([self.raytrace_file[key]['zs'][0] for key in self.source_plane_keys])
        
        # define interpolation points
        # if nsrcs is callable, calculate N(z) and randomly populate source planes
        # if nsrcs is an int and n_places=grid, then place the sources on a uniform grid over the fov
        # if nsrcs is an int and n_places=rand, then randomly populate source planes with uniform density
        if hasattr(nsrcs, '__call__'): 
            box_arcmin2 = (self.inp.bsz_arc / 60) ** 2
            Nz = (nsrcs(zs) * box_arcmin2).astype(int)
            ys1_arrays = np.array([np.random.random(nn)*self.inp.bsz_arc-self.inp.bsz_arc*0.5 for nn in Nz])
            ys2_arrays = np.array([np.random.random(nn)*self.inp.bsz_arc-self.inp.bsz_arc*0.5 for nn in Nz])
        
        elif(type(nsrcs)==int and n_places=='grid'):
            grid = np.meshgrid(np.linspace(-self.inp.bsz_arc/2, self.inp.bsz_arc, np.sqrt(nsrcs)), 
                               np.linspace(-self.inp.bsz_arc/2, self.inp.bsz_arc, np.sqrt(nsrcs)))
            ys1_arrays = np.array([np.ravel(grid[0]) for i in range(len(zs))])
            ys2_arrays = np.array([np.ravel(grid[1]) for i in range(len(zs))])
        
        elif(type(nsrcs)==int and n_places=='rand'):
            ys1_arrays = np.array([np.random.random(nsrcs)*self.inp.bsz_arc-self.inp.bsz_arc*0.5 
                                   for i in range(len(zs))])
            ys2_arrays = np.array([np.random.random(nsrcs)*self.inp.bsz_arc-self.inp.bsz_arc*0.5 
                                   for i in range(len(zs))])
       
        self.print('created out file {}'.format(self.out_file.filename))
        self.print('reading {}'.format(self.raytrace_file.filename))

        # loop over source planes
        for i in range(len(self.source_plane_keys)):
           
            zsp =zs[i]
            zkey = self.source_plane_keys[i]

            self.print('-------- placing {} sources at source plane {} --------'.format(len(ys1_arrays[i]), zkey))
           

            # get positions at this source plane
            ys1_array = ys1_arrays[i]
            ys2_array = ys2_arrays[i]

            af1 = self.raytrace_file[zkey]['alpha1'][:]
            af2 = self.raytrace_file[zkey]['alpha2'][:]
            sf1 = self.raytrace_file[zkey]['shear1'][:]
            sf2 = self.raytrace_file[zkey]['shear2'][:]
            kf0 = self.raytrace_file[zkey]['kappa0'][:]
            
            # Deflection Angles and lensed Positions
            yf1 = self.xx1 - af1
            yf2 = self.xx2 - af2
            xr1_array, xr2_array = cf.call_mapping_triangles_arrays_omp(
                                   ys1_array,ys2_array,self.xx1,self.xx2,yf1,yf2)
            # xr1_array, xr2_array = ys1_array, ys2_array
            
            # Update Lensing Signals of Lensed Positions
            sr1_array = cf.call_inverse_cic_single(sf1,0.0,0.0,xr1_array,xr2_array,self.inp.dsx_arc)
            sr2_array = cf.call_inverse_cic_single(sf2,0.0,0.0,xr1_array,xr2_array,self.inp.dsx_arc)
            kr0_array = cf.call_inverse_cic_single(kf0,0.0,0.0,xr1_array,xr2_array,self.inp.dsx_arc)

            mfa = cf.alphas_to_mu(af1, af2, self.inp.bsz_arc, self.inp.nnn)
            mra_array = cf.call_inverse_cic_single(mfa,0.0,0.0,xr1_array,xr2_array,self.inp.dsx_arc)
            
            # Save Outputs
            self.out_file.create_group(zkey)
            self.out_file[zkey]['zs'] = np.atleast_1d(zsp).astype('float32')
            self.out_file[zkey]['x1'] = xr1_array.astype('float32')
            self.out_file[zkey]['x2'] = xr2_array.astype('float32')
            self.out_file[zkey]['shear1'] = sr1_array.astype('float32')
            self.out_file[zkey]['shear2'] = sr2_array.astype('float32')
            self.out_file[zkey]['kappa0'] = kr0_array.astype('float32')
            #self.out_file[zkey]['mra'] = mra_array.astype('float32') <-- probs dont need
           
            if vis_shears:
                self.shear_vis_mocks(xr1_array, xr2_array, sr1_array, sr2_array, kf0)
        
        self.raytrace_file.close()
        self.out_file.close()
        self.print('done')
    
    
    # ------------------------------------------------------------------------------------------------------
   

    def shear_vis_gmaps(self, x1, x2, shear1, shear2, kappa):

        nnx, nny = np.shape(kappa)

        # scale_reduced = (1.0-kappa)
        # idx = kappa >= 0.5
        # scale_reduced[idx] = 1.0

        g1 = shear1#/scale_reduced
        g2 = shear2#/scale_reduced
        
        pl.figure(figsize=(10,10),dpi=80)
        pl.imshow(np.log10(kappa.T),aspect='equal',cmap=pl.cm.jet,origin='higher',
                  extent=[-self.inp.bsz_arc/2.0,
                           self.inp.bsz_arc/2.0,
                          -self.inp.bsz_arc/2.0,
                           self.inp.bsz_arc/2.0,])

        ndiv = 8
        scale_shear = 80

        for i in range(int(ndiv/2),nnx,ndiv):
            for j in range(int(ndiv/2),nny,ndiv):
                gt1 = g1[i, j]
                gt2 = g2[i, j]

                ampli = np.sqrt(gt1*gt1+gt2*gt2)
                alph = np.arctan2(gt2,gt1)/2.0

                st_x = x1[i, j]-ampli*np.cos(alph)*scale_shear
                md_x = x1[i, j]
                ed_x = x1[i, j]+ampli*np.cos(alph)*scale_shear

                st_y = x2[i, j]-ampli*np.sin(alph)*scale_shear
                md_y = x2[i, j]
                ed_y = x2[i, j]+ampli*np.sin(alph)*scale_shear

                pl.plot([md_x,ed_x],[md_y,ed_y],'w-',linewidth=1.0)
                pl.plot([md_x,st_x],[md_y,st_y],'w-',linewidth=1.0)

        pl.xlim(-self.inp.bsz_arc/2.0, self.inp.bsz_arc/2.0)
        pl.ylim(-self.inp.bsz_arc/2.0, self.inp.bsz_arc/2.0)
        pl.show()
        return 0
    
    
    # ------------------------------------------------------------------------------------------------------


    def shear_vis_mocks(self, x1, x2, shear1, shear2, kappa, zs=None, log=True):
        
        g1 = shear1
        g2 = shear2
        if(log): 
            kappa = np.log10(kappa)
            kappa[np.isnan(kappa)] = 0
        mink = min(np.ravel(kappa)[np.ravel(kappa) != 0])
        
        #---------------------------------------------------------------------
        pl.figure(figsize=(10,10),dpi=80)
        pl.imshow(kappa.T,aspect='equal',cmap=pl.cm.viridis,origin='higher',
                  extent=[-self.inp.bsz_arc/2.0,
                           self.inp.bsz_arc/2.0,
                          -self.inp.bsz_arc/2.0,
                           self.inp.bsz_arc/2.0,])

        scale_shear = 100
        ampli = np.sqrt(g1**2 + g2**2)
        alph = np.arctan2(g2, g1) / 2.0
        st_x = x1 - ampli * np.cos(alph) * scale_shear
        ed_x = x1 + ampli * np.cos(alph) * scale_shear
        st_y = x2 - ampli * np.sin(alph) * scale_shear
        ed_y = x2 + ampli * np.sin(alph) * scale_shear

        if(zs is not None):
            plt_alpha = np.min(np.array([np.ones(len(zs)), 1-( (zs - self.inp.halo_redshift)/3)]).T, axis=1)
        else:
            plt_alpha = np.ones(len(g1))

        print('plotting')
        for i in range(len(g1)):    
            a, c = plt_alpha[i], [1, plt_alpha[i], plt_alpha[i]] 
            pl.plot([st_x[i],ed_x[i]],[st_y[i],ed_y[i]],'w-',linewidth=1.0, alpha=a)
        
        pl.xlim(-self.inp.bsz_arc/2.0, self.inp.bsz_arc/2.0)
        pl.ylim(-self.inp.bsz_arc/2.0, self.inp.bsz_arc/2.0)
        pl.show()
        return 0
