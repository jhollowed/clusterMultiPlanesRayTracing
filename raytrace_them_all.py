import sys
import pdb
import glob
import inps
import h5py
import numpy as np
import cfuncs as cf
from astropy.table import Table

def printflush(s):
    print(s)
    sys.stdout.flush()

class ray_tracer():

    def __init__(self, inp, overwrite=False, stdout=True):
        '''
        This class implements functions for ray tracing through lensing grid maps constructed from 
        a particle lightcone cutout. After initializing with a `halo_inputs` object, the lensing maps
        computed from the LOS particle data should be read with `read_grid_maps_zs0()`, and the ray-tracing
        performed with `raytrace_grid_maps_for_zs()`

        Parameters
        ----------
        inp : halo_inputs instance
            A class instance of halo_inputs giving run parameters and read/write directories
        overwrite : bool
            Whether or not to overwrite old outputs. Defaults to False (will crash if HDF5 file exists)
        stdout : bool
            Whether or not to supress print statements (useful for parallel runs). `False` means all
            print statements will be suppressed. Defaults to `True`.
        '''
        
        if(stdout == False): self.print = lambda s: None
        else: self.print = printflush
        
        self.inp = inp
        self.kappa_zs0 = None
        self.alpha1_zs0 = None
        self.alpha2_zs0 = None
        self.shear1_zs0 = None
        self.shear2_zs0 = None
        mode = 'w' if overwrite else 'a'
        self. out_file = h5py.File('{}/{}_raytraced_maps.hdf5'.format(
                                    self.inp.outputs_path, self.inp.halo_id), mode)
        self.print('created out file {}'.format(self.out_file.filename))
   

    def read_grid_maps_zs0(self):
        '''
        Read in lensing maps computed from z ~infinity pointed to by `self.inp.outputs_path`
        '''
       
        # read grid maps and sort into ascending redshift order
        gmaps_path = glob.glob('{}/{}*gmaps.hdf5'.format(self.inp.outputs_path, self.inp.halo_id))
        assert len(gmaps_path)==1, "Exactly one raytrace file must be present in the target path"
        self.print('read lensing maps from {}'.format(gmaps_path[0]))
        
        gmaps_file = h5py.File(gmaps_path[0])
        lens_planes = np.array(list(gmaps_file.keys()))
        lens_zs = np.atleast_1d(np.squeeze([gmaps_file[plane]['zl'][:] for plane in lens_planes]))
        idx = np.argsort(lens_zs)
        self.lens_planes = lens_planes[idx]
        
        self.kappa_zs0 = np.array([gmaps_file[plane]['kappa0'][:] for plane in self.lens_planes])
        self.alpha1_zs0 = np.array([gmaps_file[plane]['alpha1'][:] for plane in self.lens_planes])
        self.alpha2_zs0 = np.array([gmaps_file[plane]['alpha2'][:] for plane in self.lens_planes])
        self.shear1_zs0 = np.array([gmaps_file[plane]['shear1'][:] for plane in self.lens_planes])
        self.shear2_zs0 = np.array([gmaps_file[plane]['shear2'][:] for plane in self.lens_planes])

        gmaps_file.close()
    
        # get higher redshift bound of each lens plane
        self.z_lens_planes = self.inp.lens_plane_edges[1:]
        self.zmedian_lens_planes = lens_zs
        assert len(self.z_lens_planes) == len(self.zmedian_lens_planes), ( 
               "mismatch between cutout and gmaps lens planes!")
   

    def raytrace_grid_maps_for_zs(self, ZS=None):
        """
        Performs ray tracing at the source planes at redshift ZS. Note: the output files will be closed after
        this function executes, so if needed to be called again, this object will need to be re-initialized.

        Parameters
        ----------
        ZS : float array
            redshifts at which to perform ray tracing on the grid points. If None, then use the 
            higher redshift edge of each lens plane in the grid maps as a source plane. Defaults 
            to None.
        max_planes : int
            maximum number of planes about the halo plane to include in the ray tracing (e.g. if
            max_planes = 4, then include at most four foreground planes and four background planes
            in the calculation. Sources will still be placed to the full depth of the cutout, but
            will be separated by empty space beyond the nine total included selected planes.
        """
        
        self.print('\n ---------- creating ray trace maps for halo {} ---------- '.format(self.inp.halo_id))        
        assert self.z_lens_planes is not None, "read_grid_maps_zs0() must be called before ray-tracing"
        
        ZS0 = self.inp.zs0
        ncc = self.inp.nnn
        if(ZS is None):
            ZS = self.z_lens_planes

        # loop over source redshifts
        for i in range(len(ZS)):
            
            # Rescale Lens Data (zs0->zs)
            zs = ZS[i]
            zl_array = self.zmedian_lens_planes[self.zmedian_lens_planes<=(zs)]
            nzlp = len(zl_array)
            
            alpha1_array = np.zeros((nzlp,ncc,ncc))
            alpha2_array = np.zeros((nzlp,ncc,ncc))
            kappa0_array = np.zeros((nzlp,ncc,ncc))
            shear1_array = np.zeros((nzlp,ncc,ncc))
            shear2_array = np.zeros((nzlp,ncc,ncc))

            for j in range(nzlp):
                rescale = cf.Da(ZS0)/cf.Da2(zl_array[j],ZS0)*cf.Da2(zl_array[j],zs)/cf.Da(zs)
                kappa0_array[j] = self.kappa_zs0[j]*rescale
                alpha1_array[j] = self.alpha1_zs0[j]*rescale
                alpha2_array[j] = self.alpha2_zs0[j]*rescale
                shear1_array[j] = self.shear1_zs0[j]*rescale
                shear2_array[j] = self.shear2_zs0[j]*rescale

            # Ray-tracing
            self.print('-------- ray tracing at source plane {:.3f} --------'.format(zs))
            af1, af2, kf0, sf1, sf2 = self._ray_tracing_all(alpha1_array, alpha2_array, kappa0_array,
                                                           shear1_array, shear2_array, zl_array,zs)
            self.print('max values:')
            self.print("kf0 = {}".format(np.max(kf0)))
            self.print("af1 = {}".format(np.max(af1)))
            self.print("af2 = {}".format(np.max(af2)))
            self.print("sf1 = {}".format(np.max(sf1)))
            self.print("sf2 = {}".format(np.max(sf2)))
            
            # Save Outputs
            zs_group = 'plane{}'.format(i)
            self.out_file.create_group(zs_group)
            self.out_file[zs_group]['zs'] = np.atleast_1d(zs).astype('float32')
            self.out_file[zs_group]['kappa0'] = kf0.astype('float32')
            self.out_file[zs_group]['alpha1'] = af1.astype('float32')
            self.out_file[zs_group]['alpha2'] = af2.astype('float32')
            self.out_file[zs_group]['shear1'] = sf1.astype('float32')
            self.out_file[zs_group]['shear2'] = sf2.astype('float32')

        self.out_file.close()
        self.print('done')


    def _ray_tracing_all(self, alpha1_array,alpha2_array, kappas_array, shear1_array, shear2_array, zl_array, zs):
        
        xx1 = self.inp.xi1
        xx2 = self.inp.xi2
        dsi = self.inp.dsx_arc

        nlpl = len(zl_array)
        af1 = xx1*0.0
        af2 = xx2*0.0
        kf0 = xx1*0.0
        sf1 = xx1*0.0
        sf2 = xx2*0.0

        for i in range(nlpl):
            xj1,xj2 = self._rec_read_xj(alpha1_array,alpha2_array,zl_array,zs,i+1)
            #------------------
            alpha1_tmp = cf.call_inverse_cic(alpha1_array[i],0.0,0.0,xj1,xj2,dsi)
            af1 = af1 + alpha1_tmp
            alpha2_tmp = cf.call_inverse_cic(alpha2_array[i],0.0,0.0,xj1,xj2,dsi)
            af2 = af2 + alpha2_tmp
            #------------------
            kappa_tmp = cf.call_inverse_cic(kappas_array[i],0.0,0.0,xj1,xj2,dsi)
            kf0 = kf0 + kappa_tmp
            #------------------
            shear1_tmp = cf.call_inverse_cic(shear1_array[i],0.0,0.0,xj1,xj2,dsi)
            sf1 = sf1 + shear1_tmp
            shear2_tmp = cf.call_inverse_cic(shear2_array[i],0.0,0.0,xj1,xj2,dsi)
            sf2 = sf2 + shear2_tmp
            #------------------

        pad = self.inp.npad
        kf0[:pad,:] = 0.0;kf0[-pad:,:] = 0.0;kf0[:,:pad] = 0.0;kf0[:,-pad:] = 0.0;
        sf1[:pad,:] = 0.0;sf1[-pad:,:] = 0.0;sf1[:,:pad] = 0.0;sf1[:,-pad:] = 0.0;
        sf2[:pad,:] = 0.0;sf2[-pad:,:] = 0.0;sf2[:,:pad] = 0.0;sf2[:,-pad:] = 0.0;

        return af1, af2, kf0, sf1, sf2


    def _rec_read_xj(self, alpha1_array,alpha2_array,zln,zs,n):

        xx1 = self.inp.xi1
        xx2 = self.inp.xi2
        dsi = self.inp.dsx_arc
        nx1 = self.inp.nnn
        nx2 = self.inp.nnn

        if n == 0 :
            return xx1*0.0,xx2*0.0

        if n == 1 :
            xx1.astype('double').tofile(self.inp.xj_path+str(n-1)+"_xj1.bin")
            xx2.astype('double').tofile(self.inp.xj_path+str(n-1)+"_xj2.bin")
            return xx1,xx2

        if n == 2:
            z2 = zln[1]
            z1 = zln[0]
            z0 = 0

            x01 = xx1*0.0
            x02 = xx2*0.0

            try:
                x11 = np.fromfile(self.inp.xj_path+str(n-2)+"_xj1.bin", dtype='double').reshape((nx1, nx2))
                x12 = np.fromfile(self.inp.xj_path+str(n-2)+"_xj2.bin", dtype='double').reshape((nx1, nx2))
            except:
                self.print("No {} files, recalculate XJ...".format(self.inp.xj_path+str(n-2)+"_xj1.bin"))
                x11 = xx1
                x12 = xx2

            aim11 = alpha1_array[n-2]
            aim12 = alpha2_array[n-2]

            ahm11 = cf.ai_to_ah(aim11,z1,zs)
            ahm12 = cf.ai_to_ah(aim12,z1,zs)

            bij = cf.Da(z1)*cf.Da2(z0,z2)/(cf.Da(z2)*cf.Da2(z0,z1))
            x21 = x11*bij-(bij-1)*x01-ahm11*cf.Da2(z1,z2)/cf.Da(z2)
            x22 = x12*bij-(bij-1)*x02-ahm12*cf.Da2(z1,z2)/cf.Da(z2)

            x21.astype('double').tofile(self.inp.xj_path+str(n-1)+"_xj1.bin")
            x22.astype('double').tofile(self.inp.xj_path+str(n-1)+"_xj2.bin")

            return x21,x22

        if n > 2:
            zi = zln[n-1]
            zim1 = zln[n-2]
            zim2 = zln[n-3]

            try:
                xjm21 = np.fromfile(self.inp.xj_path+str(n-3)+"_xj1.bin", dtype='double').reshape((nx1, nx2))
                xjm22 = np.fromfile(self.inp.xj_path+str(n-3)+"_xj2.bin", dtype='double').reshape((nx1, nx2))
            except:
                self.print("No {} files, recalculate XJ...".format(self.inp.xj_path+str(n-3)+"_xj1.bin"))
                xjm21,xjm22 = self._rec_read_xj(alpha1_array,alpha2_array,zln,zs,n-2)

            try:
                xjm11 = np.fromfile(self.inp.xj_path+str(n-2)+"_xj1.bin", dtype='double').reshape((nx1, nx2))
                xjm12 = np.fromfile(self.inp.xj_path+str(n-2)+"_xj2.bin", dtype='double').reshape((nx1, nx2))
            except:
                self.print("No {} files, recalculate XJ...".format(self.inp.xj_path+str(n-2)+"_xj1.bin"))
                xjm11,xjm12 = self._rec_read_xj(alpha1_array,alpha2_array,zln,zs,n-1)

            aijm11 = cf.call_inverse_cic(alpha1_array[n-2],0.0,0.0,xjm11,xjm12,dsi)
            aijm12 = cf.call_inverse_cic(alpha2_array[n-2],0.0,0.0,xjm11,xjm12,dsi)

            ahjm11 = cf.ai_to_ah(aijm11,zim1,zs)
            ahjm12 = cf.ai_to_ah(aijm12,zim1,zs)

            bij = cf.Da(zim1)*cf.Da2(zim2,zi)/cf.Da(zi)/cf.Da2(zim2,zim1)
            xj1 = xjm11*bij-(bij-1)*xjm21-ahjm11*cf.Da2(zim1,zi)/cf.Da(zi)
            xj2 = xjm12*bij-(bij-1)*xjm22-ahjm12*cf.Da2(zim1,zi)/cf.Da(zi)

            xj1.astype('double').tofile(self.inp.xj_path+str(n-1)+"_xj1.bin")
            xj2.astype('double').tofile(self.inp.xj_path+str(n-1)+"_xj2.bin")

            return xj1,xj2
