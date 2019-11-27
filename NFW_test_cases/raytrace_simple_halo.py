import os
import sys
import pdb
import time
import glob
import h5py as h
import numpy as np
import matplotlib
#from mpi4py import MPI
from matplotlib import rc
import matplotlib.pyplot as plt
rc('text', usetex=True)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import inps
#import create_grid_maps as gm
#import raytrace_them_all  as rt
#import make_lensing_mocks as mk

def halo_raytrace(halo_dir = os.path.abspath('./nfw_particle_realization'), 
                  out_dir = os.path.abspath('./lensing_output')):
    
        # crate inputs instance
        print('reading inputs...') 
        halo_prop_file = '{}/properties.csv'.format(halo_dir)
        halo_props = np.genfromtxt(self.halo_prop_file, delimiter=',', names=True)
        inp = inps.inputs(halo_dir, out_dir, mean_lens_width = 70, 
                          halo_id='nfw_realization', min_depth=0.275, 
                          sim={'mpp':halo_props['mpp']})
       
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


def vis_outputs(halo_dir = os.path.abspath('./nfw_particle_realization'), 
                lensing_dir = os.path.abspath('./lensing_output')):
    
    lensed_cutout = lensing_dir
    particles = halo_dir 
    inp = inps.inputs(halo_dir, lensing_dir, mean_lens_width = 70, mpp = None, 
                      halo_id='nfw_realization', min_depth=0.275)        
    
    gmaps = h.File(glob.glob('{}/*gmaps*.hdf5'.format(inp.outputs_path))[0])
    raytrace = h.File(glob.glob('{}/*raytrace*.hdf5'.format(inp.outputs_path))[0])
    raytrace_highz = sorted(list(raytrace.keys()), key=lambda s: int(s.split('plane')[-1]))[-1]
    gmaps_halo = sorted(list(gmaps.keys()))[0]

    mock = h.File(glob.glob('{}/*mock*.hdf5'.format(inp.outputs_path))[0])
    zgroups = list(mock.keys())

    zs, a1, a2, x1, x2, s1, s2, k0 = [], [], [], [], [], [], [], []
    for z in zgroups:    
        zsource = mock[z]['zs'][:][0]
        #if(zsource < inp.halo_redshift): continue

        x1.extend(mock[z]['xr1'][:])
        x2.extend(mock[z]['xr2'][:])
        s1.extend(mock[z]['sr1'][:])
        s2.extend(mock[z]['sr2'][:])
        zs.extend(list(np.ones(len(mock[z]['sr1'][:]))*zsource))
        a1.extend(raytrace[z]['alpha1'])
        a2.extend(raytrace[z]['alpha2'])
    
    raytrace_k0 = raytrace[raytrace_highz]['kappa0'][:]
    gmaps_k0 = gmaps[gmaps_halo]['kappa0'][:]
    
    my_norm = matplotlib.colors.Normalize(vmin=.25, vmax=.75, clip=False)
    fontsize=9
    f = plt.figure()
    ax1 = f.add_subplot(221)
    ax2 = f.add_subplot(222)
    ax3 = f.add_subplot(223)
    titles = [r'$\alpha_\theta$', r'$\alpha_\phi$', r'$\kappa,\>\>\gamma_{1,2}$']
    extent=np.array([-1, 1, -1, 1])*inp.bsz_arc/2
    ax1.imshow(a1, aspect='equal', extent=extent, origin='higher')
    ax2.imshow(a2, aspect='equal', extent=extent, origin='higher')
    
    shear_vis_mocks(inp, np.array(x1), np.array(x2), np.array(s1), np.array(s2), 
                    np.array(gmaps_k0), ax=ax3, zs=np.array(zs), log=False)
    axes = [ax1, ax2, ax3]
    for i in range(len(axes)):
        ax = axes[i]
        ax.set_xlabel(r'$\theta\>[\mathrm{arcmin}]$', fontsize=fontsize)
        ax.set_ylabel(r'$\phi\>[\mathrm{arcmin}]$', fontsize=fontsize)
        ax.title.set_text(titles[i])
        ax.title.set_fontsize(fontsize*1.25)
        ax.set_xlim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
        ax.set_ylim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
    ax3.set_xlim(-inp.bsz_arc/5.0, inp.bsz_arc/5.0)
    ax3.set_ylim(-inp.bsz_arc/5.0, inp.bsz_arc/5.0)
    plt.tight_layout()
    #plt.savefig('{}/lensed_nfw.png'.format(lensing_dir), dpi=600)
    plt.show()


def shear_vis_mocks(inp, x1, x2, shear1, shear2, kappa, ax, zs=None, log=True):
     
    g1 = shear1
    g2 = shear2
    if(log): 
        kappa = np.log10(kappa)
        kappa[np.isnan(kappa)] = 0
    mink = min(np.ravel(kappa)[np.ravel(kappa) != 0])
    
    #---------------------------------------------------------------------
    extent=np.array([-1, 1, -1, 1])*inp.bsz_arc/2
    ax.imshow(np.log((kappa.clip(min=1e-3)).T),aspect='equal',cmap=plt.cm.viridis, origin='higher',
              extent=extent)
    
    scale_shear = 250
    ampli = np.sqrt(g1**2 + g2**2)
    alph = np.arctan2(g2, g1) / 2.0
    st_x = x1 - ampli * np.cos(alph) * scale_shear
    ed_x = x1 + ampli * np.cos(alph) * scale_shear
    st_y = x2 - ampli * np.sin(alph) * scale_shear
    ed_y = x2 + ampli * np.sin(alph) * scale_shear

    if(zs is not None):
        plt_alpha = np.min(np.array([np.ones(len(zs)), 1-( (zs - inp.halo_redshift)/3)]).T, axis=1)
    else:
        plt_alpha = np.ones(len(g1))

    print('plotting (gimmie a minute)')
    for i in range(len(g1)):
        ax.plot([st_x[i],ed_x[i]],[st_y[i],ed_y[i]],'w-',linewidth=0.75)
    
if __name__ == '__main__':
    halo_raytrace()
