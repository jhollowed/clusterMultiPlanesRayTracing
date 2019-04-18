import pdb
import inps
import h5py
import glob
import numpy as np
import pylab as pl
import inps as inp
import cfuncs as cf
from astropy.table import Table

def Nz_Chang2014(z, case='conservative', sys='masking'):
    """
    Computes the forecasted LSST lensing source density, n_eff, in arcmin^-2
    """

    fiducial = {'a':1.24, 'z0':0.51, 'B':1.01, 'neff_raw':37, 'neff_blending':31, 'neff_masking':26}
    optimistic = {'a':1.23, 'z0':0.59, 'B':1.05, 'neff_raw':48, 'neff_blending':36, 'neff_masking':31}
    conservative = {'a':1.28, 'z0':0.41, 'B':0.97, 'neff_raw':24, 'neff_blending':22, 'neff_masking':18}
    p = {'fiducial':fiducial, 'optimistic':optimistic, 'conservative':conservative}

    a = p[case]['a']
    z0 = p[case]['z0']
    B = p[case]['B']
    neff = p[case]['neff_{}'.format(sys)]

    Pz = z**a * np.exp(-(z/z0)**B)
    return neff*Pz

    
def shear_vis_gmaps(x1, x2, shear1, shear2, kappa):

    nnx, nny = np.shape(kappa)

    # scale_reduced = (1.0-kappa)
    # idx = kappa >= 0.5
    # scale_reduced[idx] = 1.0

    g1 = shear1#/scale_reduced
    g2 = shear2#/scale_reduced
    #---------------------------------------------------------------------
    pl.figure(figsize=(10,10),dpi=80)
#     pl.axes([0.0,0.0,1.0,1.0])
#     pl.axis("off")
    pl.imshow(kappa.T,aspect='equal',cmap=pl.cm.jet,origin='higher',
              extent=[-inp.bsz_arc/2.0,
                       inp.bsz_arc/2.0,
                      -inp.bsz_arc/2.0,
                       inp.bsz_arc/2.0,])

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

    pl.xlim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
    pl.ylim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
    pl.show()
    return 0


def shear_vis_mocks(x1, x2, shear1, shear2, kappa):
    g1 = shear1
    g2 = shear2
    #---------------------------------------------------------------------
    pl.figure(figsize=(10,10),dpi=80)
    pl.imshow(kappa.T,aspect='equal',cmap=pl.cm.viridis,origin='higher',
              extent=[-inp.bsz_arc/2.0,
                       inp.bsz_arc/2.0,
                      -inp.bsz_arc/2.0,
                       inp.bsz_arc/2.0,])

    scale_shear = 1000
    
    for i in range(len(g1)):
        gt1 = g1[i]
        gt2 = g2[i]

        ampli = np.sqrt(gt1*gt1+gt2*gt2)
        alph = np.arctan2(gt2,gt1)/2.0

        st_x = x1[i]-ampli*np.cos(alph)*scale_shear
        md_x = x1[i]
        ed_x = x1[i]+ampli*np.cos(alph)*scale_shear

        st_y = x2[i]-ampli*np.sin(alph)*scale_shear
        md_y = x2[i]
        ed_y = x2[i]+ampli*np.sin(alph)*scale_shear

        pl.plot([md_x,ed_x],[md_y,ed_y],'w-',linewidth=1.0)
        pl.plot([md_x,st_x],[md_y,st_y],'w-',linewidth=1.0)

    pl.xlim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
    pl.ylim(-inp.bsz_arc/2.0, inp.bsz_arc/2.0)
    pl.show()
    return 0


def make_lensing_mocks(haloID, nsrcs=Nz_Chang2014, zs=None, plot_shears=False):
    """
    Performs interpolation on ray tracing maps, writing the result to an HDF file, the location
    of which is specified in inps.py
    
    Parameters
    ----------
    haloID : string
        ID of the target halo (must match subdirectory name at ray-tracing output location)
    nsrcs : int or callable or None
        If passed as an int, this is the number of sources to randomly place in the fov. This is 
        useful if creating a mock at one source plane; if using many source planes, then the lower
        redshift will effectively be weighted more strongly (nsrcs is constant while the angular scale
        grows).
        If a callable, then a function to compute the number distribution N(z) should be given, and
        will be called at each source plane redshift, where galaxies will be populated randomly in 
        angular space, with their count inferred from N(z). The callable is expected to return N(z)
        in armin^-2
        Defaults to Nz_Chang2014().
    zs : float array or None
        Source plane redshifts to include. If None, use all source plaes given in ray-tracing outputs
        for the halo spcified in haloID. Defaults to None
    theta1 : float array or None **DEPRECATED**
        azimuthal angular coordinate of the sources, in arcsec. If None, then nsrcs is assumed to have
        been passed, and theta1 will be generated from a uniform random distribution on the fov. If not
        None, then nsrcs should be None, else will throw an error. Defaults to None.
    theta2 : float array or None **DEPRECATED**
        coalitude angular coordinate of the sources, in arcsec. If None, then nsrcs is assumed to have
        been passed, and theta1 will be generated from a uniform random distribution on the fov. If not
        None, then nsrcs should be None, else will throw an error. Defaults to None.
    plot_shears : boolean
        Whether or not to visualize the resulting mock at each source plane. Defaults to False.
    """
    
    print('\n ---------- creating lensing mocks for halo {} ---------- '.format(haloID))
    
    xx1 = inp.xi1
    xx2 = inp.xi2
    
    # read in raytrace result
    raytrace_path = glob.glob('{}/*raytraced_maps.hdf5'.format(inp.outputs_path, haloID))
    assert len(raytrace_path)==1, "Exactly one raytrace file must be present in the target path"

    raytrace_file = h5py.File(raytrace_path[0])
    source_plane_keys = list(raytrace_file.keys())
    source_planes = np.array([float(s.split('_')[-1]) for s in source_plane_keys])
    if(zs is None): zs = source_planes
    
    # define interpolation points and output file
    # if nsrcs is callable, calulate N(z) and randomly populate source planes
    # if nsrcs is an int, randomly populate source planes with uniform density
    if hasattr(nsrcs, '__call__'): 
        box_arcmin2 = (inp.bsz_arc / 60)**2
        Nz = (nsrcs(zs) * box_arcmin2).astype(int)
        ys1_arrays = np.array([np.random.random(nn)*inp.bsz_arc-inp.bsz_arc*0.5 for nn in Nz])
        ys2_arrays = np.array([np.random.random(nn)*inp.bsz_arc-inp.bsz_arc*0.5 for nn in Nz])
    else:
        ys1_arrays = np.array([np.random.random(nsrcs)*inp.bsz_arc-inp.bsz_arc*0.5 for i in range(len(zs))])
        ys2_arrays = np.array([np.random.random(nsrcs)*inp.bsz_arc-inp.bsz_arc*0.5 for i in range(len(zs))])
    
    out_file = h5py.File('{}/{}_lensing_mocks.hdf5'.format(inp.outputs_path, haloID), 'w')
    
    print('created out file {}'.format(out_file.filename))
    print('reading {}'.format(raytrace_file.filename))

    # loop over source planes
    for i in range(len(source_planes)):
        zsp = source_planes[i]
        if zsp not in zs: continue
        zkey = source_plane_keys[i]
        print('-------- placing {} sources at source plane {} --------'.format(len(ys1_arrays[i]), zkey))
        
        # get positions at this source plane
        ys1_array = ys1_arrays[i]
        ys2_array = ys2_arrays[i]

        af1 = raytrace_file[zkey]['alpha1'].value
        af2 = raytrace_file[zkey]['alpha2'].value
        kf0 = raytrace_file[zkey]['kappa0'].value
        sf1 = raytrace_file[zkey]['shear1'].value
        sf2 = raytrace_file[zkey]['shear2'].value
        
        # Deflection Angles and lensed Positions
        yf1 = xx1 - af1
        yf2 = xx2 - af2
        xr1_array, xr2_array = cf.call_mapping_triangles_arrays_omp(ys1_array,ys2_array,xx1,xx2,yf1,yf2)
        # xr1_array, xr2_array = ys1_array, ys2_array
        
        # Update Lensing Signals of Lensed Positions
        kr0_array = cf.call_inverse_cic_single(kf0,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)
        sr1_array = cf.call_inverse_cic_single(sf1,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)
        sr2_array = cf.call_inverse_cic_single(sf2,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)

        mfa = cf.alphas_to_mu(af1, af2, inp.bsz_arc,inp.nnn)
        mra_array = cf.call_inverse_cic_single(mfa,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)
        
        # Save Outputs
        out_file.create_group(zkey)
        out_file[zkey]['xr1'] = xr1_array.astype('float32')
        out_file[zkey]['xr2'] = xr2_array.astype('float32')
        out_file[zkey]['kr0'] = kr0_array.astype('float32')
        out_file[zkey]['sr1'] = sr1_array.astype('float32')
        out_file[zkey]['sr2'] = sr2_array.astype('float32')
        out_file[zkey]['mra'] = mra_array.astype('float32')
        
        if plot_shears:
            shear_vis_mocks(xr1_array, xr2_array, sr1_array, sr2_array, kf0)
    
    raytrace_file.close()
    out_file.close()
    print('done')


if __name__ == '__main__':

    halo_ids_avail = [s.split('/')[-1]+'/' for s in glob.glob('./data/lenses/prtcls/halo*')]
    for halo_id in halo_ids_avail:
        inp = inps.inputs(halo_id)
        halo_id = inp.halo_info[:-1]
        make_lensing_mocks(halo_id, nsrcs=Nz_Chang2014, plot_shears=False)
