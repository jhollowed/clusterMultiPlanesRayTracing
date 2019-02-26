import numpy as np
from astropy.table import Table

import cfuncs as cf
import inps
import h5py
import pdb
import glob

def loadin_lens_data_zs0_hdf5(haloID):
    
    data = h5py.File(inp.outputs_path + haloID + '_' + str(inp.zs0) + '_gmaps.hdf5')
    lens_shells = np.array(list(data.keys()))
    lens_zs = np.array([data[shell]['zl'].value for shell in lens_shells])
    idx = np.argsort(lens_zs)
    zl_array_zs0 = lens_zs[idx]

    kappa0_array_zs0 = np.array([data[shell]['kappa0'].value for shell in lens_shells[idx]])
    alpha1_array_zs0 = np.array([data[shell]['alpha1'].value for shell in lens_shells[idx]])
    alpha2_array_zs0 = np.array([data[shell]['alpha2'].value for shell in lens_shells[idx]])
    shear1_array_zs0 = np.array([data[shell]['shear1'].value for shell in lens_shells[idx]])
    shear2_array_zs0 = np.array([data[shell]['shear2'].value for shell in lens_shells[idx]])

    return alpha1_array_zs0, alpha2_array_zs0, kappa0_array_zs0, shear1_array_zs0, shear2_array_zs0, zl_array_zs0

#--------------------------------------------------------------------
def ray_tracing_all(alpha1_array,alpha2_array, kappas_array, shear1_array, shear2_array, zl_array, zs):
    xx1 = inp.xi1
    xx2 = inp.xi2
    dsi = inp.dsx_arc

    nlpl = len(zl_array)
    af1 = xx1*0.0
    af2 = xx2*0.0
    kf0 = xx1*0.0
    sf1 = xx1*0.0
    sf2 = xx2*0.0

    for i in range(nlpl):
        xj1,xj2 = rec_read_xj(alpha1_array,alpha2_array,zl_array,zs,i+1)
        #------------------
        alpha1_tmp = cf.call_inverse_cic(alpha1_array[i],0.0,0.0,xj1,xj2,dsi)
        af1 = af1 + alpha1_tmp
        alpha2_tmp = cf.call_inverse_cic(alpha2_array[i],0.0,0.0,xj1,xj2,dsi)
        af2 = af2 + alpha2_tmp
        #------------------
        kappa_tmp = cf.call_inverse_cic(kappas_array[i],0.0,0.0,xj1,xj2,dsi)
        kf0 = kf0 + kappa_tmp
        #------------------
        shear1_tmp = cf.call_inverse_cic(shear1_array[i],0.0,0.0,xj1,xj2,dsi)
        sf1 = sf1 + shear1_tmp
        shear2_tmp = cf.call_inverse_cic(shear2_array[i],0.0,0.0,xj1,xj2,dsi)
        sf2 = sf2 + shear2_tmp
        #------------------

    kf0[:inp.npad,:] = 0.0;kf0[-inp.npad:,:] = 0.0;kf0[:,:inp.npad] = 0.0;kf0[:,-inp.npad:] = 0.0;
    sf1[:inp.npad,:] = 0.0;sf1[-inp.npad:,:] = 0.0;sf1[:,:inp.npad] = 0.0;sf1[:,-inp.npad:] = 0.0;
    sf2[:inp.npad,:] = 0.0;sf2[-inp.npad:,:] = 0.0;sf2[:,:inp.npad] = 0.0;sf2[:,-inp.npad:] = 0.0;

    return af1, af2, kf0, sf1, sf2

#--------------------------------------------------------------------
def rec_read_xj(alpha1_array,alpha2_array,zln,zs,n):

    xx1 = inp.xi1
    xx2 = inp.xi2
    dsi = inp.dsx_arc
    nx1 = inp.nnn
    nx2 = inp.nnn

    if n == 0 :
        return xx1*0.0,xx2*0.0

    if n == 1 :
        xx1.astype('double').tofile(inp.xj_path+str(n-1)+"_xj1.bin")
        xx2.astype('double').tofile(inp.xj_path+str(n-1)+"_xj2.bin")
        return xx1,xx2

    if n == 2:
        z2 = zln[1]
        z1 = zln[0]
        z0 = 0

        x01 = xx1*0.0
        x02 = xx2*0.0

        try:
            x11 = np.fromfile(inp.xj_path+str(n-2)+"_xj1.bin", dtype='double').reshape((nx1, nx2))
            x12 = np.fromfile(inp.xj_path+str(n-2)+"_xj2.bin", dtype='double').reshape((nx1, nx2))
        except:
            print("No {} files, recalculate XJ...".format(inp.xj_path+str(n-2)+"_xj1.bin"))
            x11 = xx1
            x12 = xx2

        aim11 = alpha1_array[n-2]
        aim12 = alpha2_array[n-2]

        ahm11 = cf.ai_to_ah(aim11,z1,zs)
        ahm12 = cf.ai_to_ah(aim12,z1,zs)

        bij = cf.Da(z1)*cf.Da2(z0,z2)/(cf.Da(z2)*cf.Da2(z0,z1))
        x21 = x11*bij-(bij-1)*x01-ahm11*cf.Da2(z1,z2)/cf.Da(z2)
        x22 = x12*bij-(bij-1)*x02-ahm12*cf.Da2(z1,z2)/cf.Da(z2)

        x21.astype('double').tofile(inp.xj_path+str(n-1)+"_xj1.bin")
        x22.astype('double').tofile(inp.xj_path+str(n-1)+"_xj2.bin")

        return x21,x22

    if n > 2:
        zi = zln[n-1]
        zim1 = zln[n-2]
        zim2 = zln[n-3]

        try:
            xjm21 = np.fromfile(inp.xj_path+str(n-3)+"_xj1.bin", dtype='double').reshape((nx1, nx2))
            xjm22 = np.fromfile(inp.xj_path+str(n-3)+"_xj2.bin", dtype='double').reshape((nx1, nx2))
        except:
            print("No {} files, recalculate XJ...".format(inp.xj_path+str(n-3)+"_xj1.bin"))
            xjm21,xjm22 = rec_read_xj(alpha1_array,alpha2_array,zln,zs,n-2)

        try:
            xjm11 = np.fromfile(inp.xj_path+str(n-2)+"_xj1.bin", dtype='double').reshape((nx1, nx2))
            xjm12 = np.fromfile(inp.xj_path+str(n-2)+"_xj2.bin", dtype='double').reshape((nx1, nx2))
        except:
            print("No {} files, recalculate XJ...".format(inp.xj_path+str(n-2)+"_xj1.bin"))
            xjm11,xjm12 = rec_read_xj(alpha1_array,alpha2_array,zln,zs,n-1)

        aijm11 = cf.call_inverse_cic(alpha1_array[n-2],0.0,0.0,xjm11,xjm12,dsi)
        aijm12 = cf.call_inverse_cic(alpha2_array[n-2],0.0,0.0,xjm11,xjm12,dsi)

        ahjm11 = cf.ai_to_ah(aijm11,zim1,zs)
        ahjm12 = cf.ai_to_ah(aijm12,zim1,zs)

        bij = cf.Da(zim1)*cf.Da2(zim2,zi)/cf.Da(zi)/cf.Da2(zim2,zim1)
        xj1 = xjm11*bij-(bij-1)*xjm21-ahjm11*cf.Da2(zim1,zi)/cf.Da(zi)
        xj2 = xjm12*bij-(bij-1)*xjm22-ahjm12*cf.Da2(zim1,zi)/cf.Da(zi)

        xj1.astype('double').tofile(inp.xj_path+str(n-1)+"_xj1.bin")
        xj2.astype('double').tofile(inp.xj_path+str(n-1)+"_xj2.bin")

        return xj1,xj2


def raytrace_grid_maps_for_zs(haloID, ZS):
    
    print('Doing ray-tracing for halo {} at zs={}'.format(haloID, ZS))
    #------------------------------------------------------
    # Load in lensing maps
    #

    alpha1_array_zs0, \
    alpha2_array_zs0, \
    kappa0_array_zs0, \
    shear1_array_zs0, \
    shear2_array_zs0, \
    zl_array_zs0 = loadin_lens_data_zs0_hdf5(haloID)
    
    # print type(alpha1_array_zs0)
    # print type(zl_array_zs0)
    ZS0 = inp.zs0
    ncc = inp.nnn
    #------------------------------------------------------
    # Rescale Lens Data (zs0->zs)
    #

    zl_array = zl_array_zs0[zl_array_zs0<(ZS)]
    nzlp = len(zl_array)

    alpha1_array = np.zeros((nzlp,ncc,ncc))
    alpha2_array = np.zeros((nzlp,ncc,ncc))
    kappa0_array = np.zeros((nzlp,ncc,ncc))
    shear1_array = np.zeros((nzlp,ncc,ncc))
    shear2_array = np.zeros((nzlp,ncc,ncc))

    print("-----------------------------------------------", ZS, nzlp)

    for i in range(nzlp):
        rescale = cf.Da(ZS0)/cf.Da2(zl_array[i],ZS0)*cf.Da2(zl_array[i],ZS)/cf.Da(ZS)
        alpha1_array[i] = alpha1_array_zs0[i]*rescale
        alpha2_array[i] = alpha2_array_zs0[i]*rescale
        kappa0_array[i] = kappa0_array_zs0[i]*rescale
        shear1_array[i] = shear1_array_zs0[i]*rescale
        shear2_array[i] = shear2_array_zs0[i]*rescale

    #------------------------------------------------------
    # Ray-tracing
    #

    af1, af2, kf0, sf1, sf2 = ray_tracing_all(alpha1_array,alpha2_array,
                                              kappa0_array,shear1_array,shear2_array,
                                              zl_array,ZS)
    print("kf0 = ", np.max(kf0))
    print("af1 = ", np.max(af1))
    print("af2 = ", np.max(af2))
    print("sf1 = ", np.max(sf1))
    print("sf2 = ", np.max(sf2))
    
    #------------------------------------------------------
    # Save Outputs
    #
    
    data = h5py.File(inp.outputs_path + haloID + '_' + str(ZS) + '_raytraced_maps.hdf5', 'w')
    ZS = np.atleast_1d(ZS)

    for i in range(len(ZS)):
        source_plane = data.create_group('{}'.format(ZS[i]))
        source_plane.create_dataset('kappa0', data=kappa0_array[i], dtype='float32')
        source_plane.create_dataset('alpha1', data=alpha1_array[i], dtype='float32')
        source_plane.create_dataset('alpha2', data=alpha2_array[i], dtype='float32')
        source_plane.create_dataset('shear1', data=shear1_array[i], dtype='float32')
        source_plane.create_dataset('shear2', data=shear2_array[i], dtype='float32')


if __name__ == '__main__':
    
    halo_ids_avail = [s.split('/')[-1]+'/' for s in glob.glob('./data/lenses/prtcls/halo*')]
    
    for halo_id in halo_ids_avail:
        inp = inps.inputs(halo_id)
        halo_id = inp.halo_info[:-1]
        zs_t = np.linspace(1.0/201.0,1.0,500)[inp.halo_shell] + 0.7
        raytrace_grid_maps_for_zs(halo_id, zs_t)
