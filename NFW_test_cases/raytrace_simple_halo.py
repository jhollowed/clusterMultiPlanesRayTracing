import sys
import pdb
import inps
import time
import glob
import h5py as h
import numpy as np
from mpi4py import MPI
import create_grid_maps as gm
import raytrace_them_all  as rt
import make_lensing_mocks as mk

def halo_raytrace(cutout_dir = './nfw_particle_realization', 
                  out_dir = './raytrace_output'):
    
        # crate inputs instance
        inp = inps.inputs(cutout_dir, out_dir, mean_lens_width = 70, mpp = None)        
       
        # make grid maps
        gm_gen = gm.grid_map_generator(inp, sdtfe_exe = '/home/hollowed/repos/SDTFE/cooley/dtfe', 
                                       overwrite=True, stdout=(rank==0))
        gm_gen.read_cutout_particles()
        gm_gen.create_grid_maps_for_zs0(skip_sdens=True, output_dens_tiffs=14.7)

        # 
        rt_gen = rt.ray_tracer(inp, overwrite=True, stdout=(rank==0))
        rt_gen.read_grid_maps_zs0()
        rt_gen.raytrace_grid_maps_for_zs()

        mock_gen = mk.lensing_mock_generator(inp, overwrite=True, stdout=(rank==0))
        mock_gen.read_raytrace_planes()
        mock_gen.make_lensing_mocks(vis_shears = False)


def vis_outputs(cutout_dir = './data/lenses/prtcls', lensing_dir = './output', 
                df_lenses=1.0, df_sources=1.0):
    
    # --------------------------------------
    # ---------- find all cutouts ----------
    #all_mocks = np.array(glob.glob('{}/zbin*/halo*'.format(lensing_dir)))
    all_mocks = np.array(glob.glob('{}/halo*'.format(lensing_dir)))
    #all_mocks = np.array(glob.glob('{}/halo_103815185219_0'.format(lensing_dir)))
    print('found {} total halos'.format(len(all_mocks)))
  
    # ---------------------------------------------------
    # ---------- remove cutouts missing shells ----------
    print('downsampling and removing cutouts with missing or truncated mocks')  
    empty_grids_mask = np.array([len(glob.glob('{}/*gmaps*.hdf5'.format(c))) != 0 for c in all_mocks])
    all_mocks = all_mocks[empty_grids_mask]
    empty_mocks_mask = np.array([len(glob.glob('{}/*mock*.hdf5'.format(c))) != 0 for c in all_mocks])
    all_mocks = all_mocks[empty_mocks_mask]
    all_mocks = np.random.choice(all_mocks, int(len(all_mocks)*df_lenses), replace=False)

    gmaps_planes = np.zeros(len(all_mocks))
    mocks_planes = np.zeros(len(all_mocks))
    for i in range(len(all_mocks)):
        c = all_mocks[i]
        gf = h.File(glob.glob('{}/*gmaps*.hdf5'.format(c))[0])
        mf = h.File(glob.glob('{}/*mocks*.hdf5'.format(c))[0])
        gmaps_planes[i] = len(list(gf.keys())) 
        mocks_planes[i] = len(list(mf.keys()))
        gf.close()
        mf.close()
    truncated_mask = gmaps_planes == mocks_planes
    all_mocks = all_mocks[truncated_mask]
    
    
    print('{} halos to visualize'.format(len(all_mocks)))
    
    #-----------------------------------------------------
    #------------- make input objects and plot ------------
    for i in range(len(all_mocks)):
        
        lensed_cutout = all_mocks[i]
        particles = glob.glob('{}/{}'.format(cutout_dir, lensed_cutout.split('/')[-1]))[0] 
        inp = inps.inputs(particles, lensed_cutout, mean_lens_width = 85)
        
        print('\n\n---------- working on halo {}/{} ----------'.format(i+1, len(all_mocks)))
       
        gmaps = h.File(glob.glob('{}/*gmaps*.hdf5'.format(inp.outputs_path))[0])
        raytrace = h.File(glob.glob('{}/*raytrace*.hdf5'.format(inp.outputs_path))[0])
        raytrace_highz = sorted(list(raytrace.keys()), key=lambda s: int(s.split('plane')[-1]))[-1]
        gmaps_halo = sorted(list(gmaps.keys()))[0]

        mock_gen = mk.lensing_mock_generator(inp, overwrite=False, stdout=True)
        mock = mock_gen.out_file
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
        
        mock_gen.shear_vis_mocks(x1, x2, s1, s2, gmaps_k0, zs=zs, log=True)


if __name__ == '__main__':
    parallel_raytrace(sys.argv[1], sys.argv[2])
