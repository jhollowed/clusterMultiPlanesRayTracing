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
import halo_inputs as inps
import create_grid_maps as gm
import raytrace  as rt
import make_lensing_mocks as mk

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula = LinearSegmentedColormap.from_list('parula', cm_data)

# ======================================================================================


def halo_raytrace(halo_dir = os.path.abspath('./nfw_particle_realization'), 
                  out_dir = os.path.abspath('./lensing_output'), 
                  sdtfe_exe = '/Users/joe/repos/SDTFE/bin/dtfe', 
                  zs = [1.0]):
    
        for i in range(len(zs)):
            # crate inputs instance
            print('reading inputs...') 
            halo_prop_file = '{}/properties.csv'.format(halo_dir)
            halo_props = np.genfromtxt(halo_prop_file, delimiter=',', names=True)
            inp = inps.single_plane_inputs(halo_dir, '{}/lensmaps_zs{}'.format(out_dir, zs[i]), 
                                           halo_id='nfw_realization', sim={'mpp':halo_props['mpp']})
            inp.dtfe_path = out_dir + "/dtfe_dens/"

            # make grid maps
            print('making grid maps...')
            gm_gen = gm.grid_map_generator(inp, sdtfe_exe, overwrite=True)
            gm_gen.read_cutout_particles()
            gm_gen.create_grid_maps_for_zs0(subtract_mean=False, skip_sdens=True, 
                                            output_dens_tiffs=True, output_density=True, 
                                            output_positions=True)
           
            print('raytracing from z={}...'.format(zs[i]))
            rt_gen = rt.ray_tracer(inp, overwrite=True)
            rt_gen.read_grid_maps_zs0()
            rt_gen.raytrace_grid_maps_for_zs(ZS=[zs[i]])
                
            mock_gen = mk.lensing_mock_generator(inp, overwrite=True)
            mock_gen.read_raytrace_planes()
            mock_gen.make_lensing_mocks(vis_shears = False, nsrcs = 2000, n_places='rand')


# ------------------------------------------------------------------------------------------


def vis_outputs(halo_dir = os.path.abspath('./nfw_particle_realization'), 
                lensing_dir = os.path.abspath('./lensing_output')):
    
    lensed_cutout = lensing_dir
    particles = halo_dir 
    halo_prop_file = '{}/properties.csv'.format(halo_dir)
    halo_props = np.genfromtxt(halo_prop_file, delimiter=',', names=True)
    inp = inps.single_plane_inputs(halo_dir, lensing_dir, halo_id='nfw_realization',
                                   sim={'mpp':halo_props['mpp']})
    #cm = plt.cm.rainbow
    cm = parula
    
    gmaps = h.File(glob.glob('{}/*gmaps*.hdf5'.format(inp.outputs_path))[0])
    raytrace = h.File(glob.glob('{}/*raytrace*.hdf5'.format(inp.outputs_path))[0])
    raytrace_highz = sorted(list(raytrace.keys()), key=lambda s: raytrace[s]['zs'][0])[-1]
    gmaps_halo = sorted(list(gmaps.keys()))[0]

    mock = h.File(glob.glob('{}/*mock*.hdf5'.format(inp.outputs_path))[0])
    zgroups = list(mock.keys())

    zs, a1, a2, x1, x2, s1, s2, sm, k0 = [], [], [], [], [], [], [], [], []
    for z in zgroups:    
        zsource = mock[z]['zs'][:][0]

        x1.extend(mock[z]['x1'][:])
        x2.extend(mock[z]['x2'][:])
        s1.extend(mock[z]['shear1'][:])
        s2.extend(mock[z]['shear2'][:])
        zs.extend(list(np.ones(len(mock[z]['shear1'][:]))*zsource))
        a1.extend(raytrace[z]['alpha1'])
        a2.extend(raytrace[z]['alpha2'])
    
    raytrace_k0 = raytrace[raytrace_highz]['kappa0'][:]
    gmaps_k0 = gmaps[gmaps_halo]['kappa0'][:]
    
    my_norm = matplotlib.colors.Normalize(vmin=.25, vmax=.75, clip=False)
    fontsize=12
    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax2 = f.add_subplot(212)
    extent=np.array([-1, 1, -1, 1])*inp.bsz_arc/2
    aim1 = ax1.imshow(a1, aspect='equal', extent=extent, origin='higher', cmap=cm)
    aim2 = ax2.imshow(a2, aspect='equal', extent=extent, origin='higher', cmap=cm)
    cbar = f.colorbar(aim2, ax=[ax1, ax2], shrink=1.0)
    cbar.set_label(r'$\alpha_{\theta,\phi}$')
    
    f2 = plt.figure()
    ax3 = f2.add_subplot(111)
    shear_vis_mocks(inp, np.array(x1), np.array(x2), -np.array(s1), np.array(s2), 
                    np.array(gmaps_k0), fig=f2, ax=ax3, zs=np.array(zs), cm=cm, log=False)
    axes = [ax1, ax2, ax3]
    for i in range(len(axes)):
        axi = axes[i]
        axi.set_xlabel(r'$\theta\>[\mathrm{arcmin}]$', fontsize=fontsize)
        axi.set_ylabel(r'$\phi\>[\mathrm{arcmin}]$', fontsize=fontsize)
        if(axi == ax1):
            axi.set_xticks([])
            axi.set_xlabel('')
        axi.set_xlim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
        axi.set_ylim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
    plt.tight_layout()
    
    f.savefig('{}/alphas.png'.format(lensing_dir), dpi=300)
    f2.savefig('{}/shear.png'.format(lensing_dir), dpi=300)
    plt.show()


# ------------------------------------------------------------------------------------------


def shear_vis_mocks(inp, x1, x2, shear1, shear2, kappa, fig, ax, cm, zs=None, log=True, cut_core=30):
     
    g1 = shear1
    g2 = shear2
    if(log): 
        kappa = np.log10(kappa)
        kappa[np.isnan(kappa)] = 0
    mink = min(np.ravel(kappa)[np.ravel(kappa) != 0])
    
    extent=np.array([-1, 1, -1, 1])*inp.bsz_arc/2
    aimk = ax.imshow(np.log(kappa),aspect='equal',cmap=cm, origin='higher',
              extent=extent)
    cbar1 = fig.colorbar(aimk, ax=ax)
    cbar1.set_label(r'$\mathrm{ln}(\kappa)$', fontsize=12)
    
    scale_shear = 500
    ampli = np.sqrt(g1**2 + g2**2)
    alph = np.arctan2(g2, g1) / 2.0
    
    mask = np.sqrt((x1-20)**2+(x2-10)**2) > cut_core
    alph = alph[mask]
    ampli = ampli[mask]
    x1 = x1[mask]
    x2 = x2[mask]
    
    st_x = x1 - ampli * np.cos(alph) * scale_shear
    ed_x = x1 + ampli * np.cos(alph) * scale_shear
    st_y = x2 - ampli * np.sin(alph) * scale_shear
    ed_y = x2 + ampli * np.sin(alph) * scale_shear

    if(zs is not None):
        plt_alpha = np.min(np.array([np.ones(len(zs)), 1-( (zs - inp.halo_redshift)/3)]).T, axis=1)
    else:
        plt_alpha = np.ones(len(g1))

    print('plotting (gimmie a minute)')
    for i in range(len(g1[mask])):
        ax.plot([st_x[i],ed_x[i]],[st_y[i],ed_y[i]],'w-',linewidth=0.75)
    

# ======================================================================================


if __name__ == '__main__':
    
    halo_raytrace(zs = [float(sys.argv[1])])
    #vis_outputs()
