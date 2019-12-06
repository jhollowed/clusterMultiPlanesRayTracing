import sys
import pdb
import time
import glob
import h5py as h
import numpy as np
import halo_inputs
from mpi4py import MPI

import create_grid_maps as gm
import raytrace  as rt
import make_lensing_mocks as mk

def parallel_raytrace(cutout_dir = './data/lenses/prtcls', 
                      out_dir = './output'):
    
    # toggle this on to test communication without actually creating lensing maps
    dry_run = False
    
    # toggle these on to rebuild lensing maps even if files already exist
    overwrite_gmaps = True
    overwrite_raytrace = True
    overwrite_mocks = True

    # -----------------------------------------
    # ---------- define communicator ----------
    comm= MPI.COMM_WORLD
    rank = comm.Get_rank()
    numranks = comm.Get_size()
    if(rank==0):
        print('\n---------- starting with {} MPI processes ----------'.format(numranks))
        sys.stdout.flush()
    comm.Barrier()


    # --------------------------------------
    # ---------- find all cutouts ----------
    #all_cutouts = np.array(glob.glob('{}/zbin*/halo*'.format(cutout_dir)))
    #all_cutouts = np.array(glob.glob('{}/halo*'.format(cutout_dir)))
    all_cutouts = np.array(glob.glob('{}/halo_244960324069_0'.format(cutout_dir)))
    if( rank==0 ):
        print('found {} total halos'.format(len(all_cutouts)))
  

    # ---------------------------------------------------
    # ---------- remove cutouts missing shells ----------
    if(rank == 0): 
        print('removing empty and truncated cutouts') 
    empty_mask = np.array([len(glob.glob('{}/*Cutout*'.format(c))) != 0 for c in all_cutouts])
    all_cutouts = all_cutouts[empty_mask]

    cutout_shells = np.array([np.array([int(shell.split('Cutout')[-1])
                              for shell in glob.glob('{}/*Cutout*'.format(c))])
                             for c in all_cutouts])
    truncated_mask = np.array([min(sc) == 124 and max(sc)==487 for sc in cutout_shells])
    all_cutouts = all_cutouts[truncated_mask]
    if(rank == 0):
        print('distributing {} cutouts to {} ranks'.format(len(all_cutouts), numranks))
        sys.stdout.flush()

    # -------------------------------------------------
    # ---------- distribute cutouts to ranks ----------
    this_rank_halos = np.array_split(all_cutouts, numranks)[rank]
    print("rank {} gets {} cutouts".format(rank, len(this_rank_halos)))
    comm.Barrier()
    sys.stdout.flush()
    if(rank == 0): print('\n')
    comm.Barrier()
   
    
    # ----------------------------------------------------------------------------
    # ---------- compute lensing maps, ray tracing, mocks for this rank ----------
    start = time.time()
    for i in range(len(this_rank_halos)):
        
        cutout = this_rank_halos[i]
        output = '{}/{}'.format(out_dir, cutout.split('/')[-1])
        inp = inps.multi_plane_inputs(cutout, output, mean_lens_width = 85)
        
        if(rank==0): print('\n\n---------- working on halo {}/{} ----------'.format(i+1, len(this_rank_halos)))
        
        #if( (len(glob.glob('{}/*gmaps.hdf5'.format(inp.outputs_path))) == 0 or overwrite) and not dry_run):
        if( (len(glob.glob('{}/*gmaps*hdf5'.format(inp.outputs_path))) == 0 or overwrite_gmaps) and not dry_run):
            if(rank==0): 
                print('\n--- gridding ---')
                sys.stdout.flush()
            gm_gen = gm.grid_map_generator(inp, sdtfe_exe = '/home/hollowed/repos/SDTFE/cooley/dtfe', 
                                           overwrite=True, stdout=(rank==0))
            gm_gen.read_cutout_particles()
            gm_gen.create_grid_maps_for_zs0(skip_sdens=True, output_dens_tiffs=14.7)

        if( (len(glob.glob('{}/*ray*.hdf5'.format(inp.outputs_path))) == 0 or overwrite_raytrace) and not dry_run):
            if(rank==0): 
                print('\n--- raytracing--- ')
                sys.stdout.flush()
            rt_gen = rt.ray_tracer(inp, overwrite=True, stdout=(rank==0))
            rt_gen.read_grid_maps_zs0()
            rt_gen.raytrace_grid_maps_for_zs()

        if( (len(glob.glob('{}/*mock*.hdf5'.format(inp.outputs_path))) == 0 or overwrite_mocks) and not dry_run):
            if(rank==0): 
                print('\n--- mocking ---')
                sys.stdout.flush()
            mock_gen = mk.lensing_mock_generator(inp, overwrite=True, stdout=(rank==0))
            mock_gen.read_raytrace_planes()
            mock_gen.make_lensing_mocks(vis_shears = False)

    end = time.time()

    # all done
    comm.Barrier()
    if(rank == 0): print('\n')
    comm.Barrier()
    print('rank {} finished {} halos in {:.2f} s'.format(
          rank, len(this_rank_halos), end-start))


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
        inp = inps.multi_plane_inputs(particles, lensed_cutout, mean_lens_width = 85)
        
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
