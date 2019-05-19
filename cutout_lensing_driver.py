import pdb
import numpy as np
import glob
import create_grid_maps as gm
import raytrace_them_all  as rt
import make_lensing_mocks as mock
import inps

cutouts_avail = glob.glob('./data/lenses/prtcls/halo*')

for cutout in cutouts_avail:
    
    inp = inps.inputs(cutout, '/projects/DarkUniverse_esp/jphollowed/outerRim/cutouts_raytracing/{}'.format(
                              cutout.split('/')[-1]), mean_lens_width = 70) 
    
    if(len(glob.glob('{}/*gmaps.hdf5'.format(inp.outputs_path))) == 0 or 1):
        print('gridding')
        gm_gen = gm.grid_map_generator(inp, overwrite=True)
        gm_gen.read_cutout_particles()
        gm_gen.create_grid_maps_for_zs0(skip_sdens=False, output_dens_tiffs=14.7)

    if(len(glob.glob('{}/*raytraced_maps.hdf5'.format(inp.outputs_path))) == 0 or 1):
        print('raytracing')
        rt_gen = rt.ray_tracer(inp, overwrite=True)
        rt_gen.read_grid_maps_zs0()
        rt_gen.raytrace_grid_maps_for_zs()

    if(len(glob.glob('{}/*raytraced_maps.hdf5'.format(inp.outputs_path))) == 0 or 1):
        print('mocking')
        mock_gen = mock.lensing_mock_generator(inp, overwrite=True)
        mock_gen.read_raytrace_planes()
        mock_gen.make_lensing_mocks(vis_shears = False)



