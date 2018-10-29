import numpy as np
import inps as inp
import cfuncs as cf
import pylab as pl

def shear_vis_grids(x1, x2, shear1, shear2, kappa):

    nnx, nny = np.shape(shear1)

    # scale_reduced = (1.0-kappa)
    # idx = kappa >= 0.5
    # scale_reduced[idx] = 1.0

    g1 = shear1#/scale_reduced
    g2 = shear2#/scale_reduced
    #---------------------------------------------------------------------
    pl.figure(figsize=(10,10),dpi=80)
    pl.axes([0.0,0.0,1.0,1.0])
    pl.axis("off")
    pl.imshow(kappa.T,aspect='auto',cmap=pl.cm.jet,origin='higher')
    #pl.imshow(kappa.T)

    ndiv = 8
    scale_shear = 80

    for i in xrange(ndiv/2,nnx,ndiv):
       for j in xrange(ndiv/2,nny,ndiv):
           gt1 = g1[i,j]
           gt2 = g2[i,j]

           ampli = np.sqrt(gt1*gt1+gt2*gt2)
           alph = np.arctan2(gt2,gt1)/2.0

           st_x = i-ampli*np.cos(alph)*scale_shear
           md_x = i
           ed_x = i+ampli*np.cos(alph)*scale_shear

           st_y = j-ampli*np.sin(alph)*scale_shear
           md_y = j
           ed_y = j+ampli*np.sin(alph)*scale_shear

           pl.plot([md_x,ed_x],[md_y,ed_y],'w-',linewidth=1.0)
           pl.plot([md_x,st_x],[md_y,st_y],'w-',linewidth=1.0)

    return 0


def shear_vis_mocks(xi1, xi2, x1, x2, shear1, shear2, kappa):

    g1 = shear1
    g2 = shear2
    #---------------------------------------------------------------------
    pl.figure(figsize=(10,10),dpi=80)
    pl.imshow(kappa.T,aspect='auto',cmap=pl.cm.jet,origin='higher',
              extent=[-inp.bsz_arc/2.0,
                       inp.bsz_arc/2.0,
                      -inp.bsz_arc/2.0,
                       inp.bsz_arc/2.0,])
    # pl.contourf(xi1, xi2, kappa, aspect='auto',cmap=pl.cm.jet,origin='lower')
    # pl.contourf(xi1, xi2, kappa, aspect='auto',cmap=pl.cm.jet,origin='image')
    # pl.contourf(xi1, xi2, kappa.T)
    # pl.axes([0.0,0.0,1.0,1.0])
    # pl.axis("off")

    scale_shear = 80.0
    # nskip = len(x1)/500
    ngals = 5000
    index = np.random.choice(np.linspace(0, len(x1)-1, len(x1)).astype('int'), ngals)

    for i in index:
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

    return 0


#------------------------------------------------------------------------------
def main(zs, plot_shears=True):

    xi1,xi2 = cf.make_c_coor(inp.bsz_arc,inp.nnn)
    #------------------------------------------------------

    # gtheta = np.fromfile(inp.gals_path+"theta_1.5_gals.bin", dtype='float32')
    # gphi = np.fromfile(inp.gals_path+"phi_1.5_gals.bin", dtype='float32')

    # ys1_array = (gtheta/3600.0-87.5)*3600.
    # ys2_array = (gphi/3600.0-2.5)*3600.

    ys1_array = np.random.random(6000)*inp.bsz_arc-inp.bsz_arc*0.5
    ys2_array = np.random.random(6000)*inp.bsz_arc-inp.bsz_arc*0.5

    af1 = np.fromfile(inp.rmaps_path+"cl_"+'%.6f'%(zs)+"_af1.bin", dtype='float32').reshape((inp.nnn, inp.nnn))
    af2 = np.fromfile(inp.rmaps_path+"cl_"+'%.6f'%(zs)+"_af2.bin", dtype='float32').reshape((inp.nnn, inp.nnn))
    kf0 = np.fromfile(inp.rmaps_path+"cl_"+'%.6f'%(zs)+"_kf0.bin", dtype='float32').reshape((inp.nnn, inp.nnn))
    sf1 = np.fromfile(inp.rmaps_path+"cl_"+'%.6f'%(zs)+"_sf1.bin", dtype='float32').reshape((inp.nnn, inp.nnn))
    sf2 = np.fromfile(inp.rmaps_path+"cl_"+'%.6f'%(zs)+"_sf2.bin", dtype='float32').reshape((inp.nnn, inp.nnn))

    #------------------------------------------------------
    # Deflection Angles and lensed Positions
    #

    yf1 = xi1 - af1
    yf2 = xi2 - af2
    xr1_array, xr2_array = cf.call_mapping_triangles_arrays_omp(ys1_array,ys2_array,xi1,xi2,yf1,yf2)
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

    xr1_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_xr1.bin")
    xr2_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_xr2.bin")
    kr0_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_kr0.bin")
    sr1_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_sr1.bin")
    sr2_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_sr2.bin")
    mra_array.astype('float32').tofile(inp.mocks_path + "cl_" + '%.6f'%(zs) + "_mra.bin")

    if plot_shears:
        shear_vis_mocks(xi1, xi2, xr1_array, xr2_array, sr1_array, sr2_array, kf0)
        shear_vis_grids(xi1, xi2, sf1, sf2, kf0)
        # pl.savefig(str(i)+".png")

    else:
        pass

    return 0

#------------------------------------------------------------------------------
if __name__ == '__main__':
    zs_t = 1.5
    main(zs_t)
    pl.show()
