import sys
import pdb
import inps
import time
import glob
import numpy as np
from mpi4py import MPI
import create_grid_maps as gm
import raytrace_them_all  as rt
import make_lensing_mocks as mock

def parallel_raytrace(cutout_dir = './data/lenses/prtcls', 
                      out_dir = './output'):
    
    # toggle this on to test communication without actually creating lensing maps
    dry_run = False
    # toggle this on to rebuild lensing maps even if files already exist
    overwrite = False

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
    all_cutouts = np.array(glob.glob('{}/halo*'.format(cutout_dir)))
  

    # ---------------------------------------------------
    # ---------- remove cutouts missing shells ----------
    if(rank == 0): 
        print('removing empty and truncated cutouts') 
    empty_mask = np.array([len(glob.glob('{}/*Cutout*'.format(c))) != 0 for c in all_cutouts])
    all_cutouts = all_cutouts[empty_mask]

    cutout_shells = np.array([np.array([int(shell.split('Cutout')[-1])
                              for shell in glob.glob('{}/*Cutout*'.format(c))])
                             for c in all_cutouts])
    truncated_mask = np.array([min(sc) == 148 and max(sc)==487 for sc in cutout_shells])
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
        inp = inps.inputs(cutout, output, mean_lens_width = 85)
        
        if(rank==0): print('\n\n---------- working on halo {}/{} ----------'.format(i+1, len(this_rank_halos)))
        
        #if( (len(glob.glob('{}/*gmaps.hdf5'.format(inp.outputs_path))) == 0 or overwrite) and not dry_run):
        if( (len(glob.glob('{}/*mock*.hdf5'.format(inp.outputs_path))) == 0 or overwrite) and not dry_run):
            if(rank==0): 
                print('\n--- gridding ---')
                sys.stdout.flush()
            gm_gen = gm.grid_map_generator(inp, sdtfe_exe = '/home/hollowed/repos/SDTFE/cooley/dtfe', 
                                           overwrite=True, stdout=(rank==0))
            gm_gen.read_cutout_particles()
            gm_gen.create_grid_maps_for_zs0(skip_sdens=True, output_dens_tiffs=14.7)

        if( (len(glob.glob('{}/*ray*.hdf5'.format(inp.outputs_path))) == 0 or overwrite) and not dry_run):
            if(rank==0): 
                print('\n--- raytracing--- ')
                sys.stdout.flush()
            rt_gen = rt.ray_tracer(inp, overwrite=True, stdout=(rank==0))
            rt_gen.read_grid_maps_zs0()
            rt_gen.raytrace_grid_maps_for_zs()

        if( (len(glob.glob('{}/*mock*.hdf5'.format(inp.outputs_path))) == 0 or overwrite) and not dry_run):
            if(rank==0): 
                print('\n--- mocking ---')
                sys.stdout.flush()
            mock_gen = mock.lensing_mock_generator(inp, overwrite=True, stdout=(rank==0))
            mock_gen.read_raytrace_planes()
            mock_gen.make_lensing_mocks(vis_shears = False)

    end = time.time()

    # all done
    comm.Barrier()
    if(rank == 0): print('\n')
    comm.Barrier()
    print('rank {} finished {} halos in {:.2f} s'.format(
          rank, len(this_rank_halos), end-start))


if __name__ == '__main__':
    parallel_raytrace(sys.argv[1], sys.argv[2])
