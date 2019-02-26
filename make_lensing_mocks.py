import numpy as np
import pylab as pl
from astropy.table import Table

import cfuncs as cf
import inps as inp


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

    for i in range(ndiv/2,nnx,ndiv):
        for j in range(ndiv/2,nny,ndiv):
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

    scale_shear = 80.0
    ngals = 5000
    index = np.random.choice(np.linspace(0, len(x1)-1, len(x1)).astype('int'), ngals)

    for i in index:
        gt1 = g1[i]
        gt2 = g2[i]

        ampli = np.sqrt(gt1*gt1+gt2*gt2)
        if(i%100 == 0): print(ampli)
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

#------------------------------------------------------------------------------
def make_lensing_mocks(haloID, zs, DATA=1, plot_shears=False):
    xx1 = inp.xi1
    xx2 = inp.xi2

    #------------------------------------------------------
    # gtheta = np.fromfile(inp.gals_path+"theta_1.5_gals.bin", dtype='float32')
    # gphi = np.fromfile(inp.gals_path+"phi_1.5_gals.bin", dtype='float32')
    # ys1_array = (gtheta/3600.0-87.5)*3600.
    # ys2_array = (gphi/3600.0-2.5)*3600.

    nsrcs = 6000
    ys1_array = np.random.random(nsrcs)*inp.bsz_arc-inp.bsz_arc*0.5
    ys2_array = np.random.random(nsrcs)*inp.bsz_arc-inp.bsz_arc*0.5

    if type(DATA)==int:
        data = Table.read(inp.outputs_path + haloID + '_' + str(zs) + '_raytraced_maps.hdf5', path='/raytraced_maps')

        af1 = data['af1']
        af2 = data['af2']
        kf0 = data['kf0']
        sf1 = data['sf1']
        sf2 = data['sf2']

        del data
    else:
        af1 = DATA['af1']#.data
        af2 = DATA['af2']#.data
        kf0 = DATA['kf0']#.data
        sf1 = DATA['sf1']#.data
        sf2 = DATA['sf2']#.data

        del DATA

    #------------------------------------------------------
    # Deflection Angles and lensed Positions
    #

    yf1 = xx1 - af1
    yf2 = xx2 - af2
    xr1_array, xr2_array = cf.call_mapping_triangles_arrays_omp(ys1_array,ys2_array,xx1,xx2,yf1,yf2)
    # xr1_array, xr2_array = ys1_array, ys2_array
    #------------------------------------------------------
    # Update Lensing Signals of Lensed Positions
    #

    kr0_array = cf.call_inverse_cic_single(kf0,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)
    sr1_array = cf.call_inverse_cic_single(sf1,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)
    sr2_array = cf.call_inverse_cic_single(sf2,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)

    mfa = cf.alphas_to_mu(af1, af2, inp.bsz_arc,inp.nnn)
    mra_array = cf.call_inverse_cic_single(mfa,0.0,0.0,xr1_array,xr2_array,inp.dsx_arc)
    #------------------------------------------------------
    # Save Outputs
    #

#     xr1_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_xr1.bin")
#     xr2_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_xr2.bin")
#     kr0_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_kr0.bin")
#     sr1_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_sr1.bin")
#     sr2_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_sr2.bin")
#     mra_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_mra.bin")

    data = Table()

    data['xr1'] = xr1_array.astype('float32')
    data['xr2'] = xr2_array.astype('float32')
    data['kr0'] = kr0_array.astype('float32')
    data['sr1'] = sr1_array.astype('float32')
    data['sr2'] = sr2_array.astype('float32')
    data['mra'] = mra_array.astype('float32')

    data.write(inp.outputs_path + haloID + '_' + str(zs) + '_lensing_mocks.hdf5',
               path="/lensing_mocks", append=True, overwrite=True)#, compression=True)

    if plot_shears:
        shear_vis_mocks(xr1_array, xr2_array, sr1_array, sr2_array, kf0)
        #shear_vis_gmaps(xx1, xx2, sf1, sf2, kf0)
    else:
        pass

    return data


if __name__ == '__main__':
    halo_id = inp.halo_info[:-1]
    zs_t = 1.5
    make_lensing_mocks(halo_id, zs_t, plot_shears=True)
