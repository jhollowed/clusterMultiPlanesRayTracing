import os
import sys
import pdb
import time
import glob
import h5py as h
import numpy as np
#from mpi4py import MPI

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inps
import create_grid_maps as gm
import raytrace_them_all  as rt
import make_lensing_mocks as mk

def halo_raytrace(cutout_dir = os.path.abspath('./nfw_particle_realization'), 
                  out_dir = os.path.abspath('./lensing_output')):
    
        # crate inputs instance
        print('reading inputs...')
        inp = inps.inputs(cutout_dir, out_dir, mean_lens_width = 70, mpp = None, 
                          halo_id='nfw_realization', min_depth=0.275)        
       
        # make grid maps
        print('making grid maps...')
        gm_gen = gm.grid_map_generator(inp, sdtfe_exe = '/home/hollowed/repos/SDTFE/cooley/dtfe', 
                                       overwrite=True)
        gm_gen.read_cutout_particles()
        gm_gen.create_grid_maps_for_zs0(skip_sdens=True, output_dens_tiffs=True)
 
        print('raytracing from z=1...')
        rt_gen = rt.ray_tracer(inp, overwrite=True)
        rt_gen.read_grid_maps_zs0()
        rt_gen.raytrace_grid_maps_for_zs(ZS=1)
            
        mock_gen = mk.lensing_mock_generator(inp, overwrite=True)
        mock_gen.read_raytrace_planes()
        mock_gen.make_lensing_mocks(vis_shears = False, nsrcs = 1000)


def vis_outputs(cutout_dir = os.path.abspath('./nfw_particle_realization'), 
                lensing_dir = os.path.abspath('./lensing_output')):
    
    lensed_cutout = lensing_dir
    particles = cutout_dir 
    inp = inps.inputs(cutout_dir, lensing_dir, mean_lens_width = 70, mpp = None, 
                      halo_id='nfw_realization', min_depth=0.275)        
    
    gmaps = h.File(glob.glob('{}/*gmaps*.hdf5'.format(inp.outputs_path))[0])
    raytrace = h.File(glob.glob('{}/*raytrace*.hdf5'.format(inp.outputs_path))[0])
    raytrace_highz = sorted(list(raytrace.keys()), key=lambda s: int(s.split('plane')[-1]))[-1]
    gmaps_halo = sorted(list(gmaps.keys()))[0]

    mock = glob.glob('{}/*mock*hdf5'.format(lensing_dir))[0]
    zgroups = list(mock.keys())

    zs, x1, x2, s1, s2, k0 = [], [], [], [], [], []        
    for z in zgroups:    
        zsource = mock[z]['zs'][:][0]
        if(zsource < inp.halo_redshift): continue
        
        x1.extend(mock[z]['xr1'][:])
        x2.extend(mock[z]['xr2'][:])
        s1.extend(mock[z]['sr1'][:])
        s2.extend(mock[z]['sr2'][:])
        zs.extend(list(np.ones(len(mock[z]['xr1'][:]))*zsource))
    
    select_sources = np.random.choice(np.arange(len(x1)), int(len(x1)*df_sources), replace=False)
    x1 = np.array(x1)[select_sources]
    x2 = np.array(x2)[select_sources]
    s1 = np.array(s1)[select_sources]
    s2 = np.array(s2)[select_sources]
    zs = np.array(zs)[select_sources]
    
    raytrace_k0 = raytrace[raytrace_highz]['kappa0'][:]
    gmaps_k0 = gmaps[gmaps_halo]['kappa0'][:]
    
    shear_vis_mocks(x1, x2, s1, s2, gmaps_k0, zs=zs, log=True)


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

if __name__ == '__main__':
    halo_raytrace()
