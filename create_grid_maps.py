import numpy as np
from astropy.table import Table
import h5py

import cfuncs as cf
import inps as inp
import pdb
import matplotlib.pyplot as plt
import subprocess
import os
import pdb

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def convert_one_from_dir(file_name_base, snapid, skip_sdens=False):
    
    #---------------------------------------
    # Load in particles of one lens plane
    #
   
    #if(int(snapid) > 430): return
        
    # write out/read in denisty file
    write_dir = '{}/dtfe_dens'.format(inp.outputs_path)
    if not os.path.exists(write_dir):
            os.makedirs(write_dir)
    dtfe_file = '{}/{}_dtfe_input.bin'.format(write_dir, snapid)
    
    # read in particle data
    print('reading redshift')
    zl_array = np.fromfile(file_name_base+str(snapid)+"/"+"redshift."+str(snapid)+".bin", dtype = "f")
    zl_median = np.median(zl_array)     
    mp_array = np.ones(len(zl_array))*inp.mpp
        
    zs = inp.zs0
    ncc = inp.nnn
    bsz_mpc = inp.bsz_arc*cf.Dc(zl_median)/cf.apr
    # dsx_mpc = bsz_mpc/ncc
    
    if(skip_sdens == False):    
        
        print('---------- working on {} ----------'.format(snapid))
        print('reading x, y, z')
        xxl_array = np.fromfile(file_name_base+str(snapid)+"/"+"x."+str(snapid)+".bin", dtype = "f")
        yyl_array = np.fromfile(file_name_base+str(snapid)+"/"+"y."+str(snapid)+".bin", dtype = "f")
        zzl_array = np.fromfile(file_name_base+str(snapid)+"/"+"z."+str(snapid)+".bin", dtype = "f")
        #xo3 = cf.Dc(zl_array)
        xo3 = np.linalg.norm(np.vstack([xxl_array, yyl_array, zzl_array]), axis=0)
        xc3 = (np.max(xo3)+np.min(xo3))*0.5
        x3_array = xo3 - xc3
        
        print('reading theta')
        xo1 = np.fromfile(file_name_base+str(snapid)+"/"+"theta."+snapid+".bin",dtype = "f")
        xc1 = (np.max(xo1)+np.min(xo1))*0.5
        #x1_array = (xo1-xc1)*cf.Dc(zl_array)/cf.apr
        x1_array = np.sin((xo1-xc1)/cf.apr) * xo3
        
        print('reading phi')
        xo2 = np.fromfile(file_name_base+str(snapid)+"/"+"phi."+snapid+".bin",dtype = "f")
        xc2 = (np.max(xo2)+np.min(xo2))*0.5
        #x2_array = (xo2-xc2)*cf.Dc(zl_array)/cf.apr
        x2_array = np.sin((xo2-xc2)/cf.apr) * xo3
        
        
        #---------------------------------------
        # Calculate convergence maps
        #

        idx1 = x1_array > -0.5*bsz_mpc
        idx2 = x1_array <= 0.5*bsz_mpc
        idx3 = x2_array > -0.5*bsz_mpc
        idx4 = x2_array <= 0.5*bsz_mpc

        idx = idx1&idx2&idx3&idx4

        x1in = x1_array[idx]
        x2in = x2_array[idx]
        x3in = x3_array[idx]
        mpin = mp_array[idx]

        npp = len(mpin)
        if npp < 20:
            return 1
        
        # ------ do density estiamtion via system call to SDTFE exe ------
 
        # x, y, z in column major
        dtfe_input_array = np.ravel(np.vstack([x1in, x2in, x3in]))
        dtfe_input_array.astype('f').tofile(dtfe_file)
        
        # Usage: dtfe [ path_to_file n_particles grid_dim center_x center_y center_z 
        #               field_width field_depth particle_mass mc_box_width 
        #               n_mc_samples sample_factor ]
        dtfe_args = ["%s"%s for s in 
                     ['/home/hollowed/repos/SDTFE/bin/dtfe',
                      dtfe_file, len(mpin), inp.nnn, 0, 0, 0, 
                      bsz_mpc*0.95, np.max(x3in)-np.min(x3in), inp.mpp, 
                      bsz_mpc/inp.nnn/4, 4, 1.0
                     ]
                    ]
        
        # call process
        with cd(write_dir):
            subprocess.run(dtfe_args)
        
        # read in result
        sdens_cmpch = np.fromfile('{}.rho.bin'.format(dtfe_file))
    
    else:
        # read in result

        print('---------- reading density at {} ----------'.format(snapid))
        sdens_cmpch = np.fromfile('{}.rho.bin'.format(dtfe_file))

    sdens_cmpch = sdens_cmpch.reshape(inp.nnn, inp.nnn)
    kappa = sdens_cmpch*(1.0+zl_median)**2.0/cf.sigma_crit(zl_median,zs)
    
    # -----------------------------------------------------------------

    #print(snapid)
    #sdens_cmpch2 = cf.call_sph_sdens_weight_omp(x1in,x2in,x3in,mpin,bsz_mpc,ncc)
    #plt.imshow(np.log10(sdens_cmpch))
    #plt.plot(x1in, x2in, '.')
    #plt.show()
    #pdb.set_trace()
    #return
    #kappa2 = sdens_cmpch*(1.0+zl_median)**2.0/cf.sigma_crit(zl_median,zs)
    #pdb.set_trace()
    
    #---------------------------------------
    # Calculate deflection maps
    #

    if snapid == inp.halo_shell: # The Snapshot contains the main halo.
        BoundaryCondition = "Isolated"
    else:
        BoundaryCondition = "Periodic"

    #BoundaryCondition = "Isolated"
#     BoundaryCondition = "Periodic"

    if BoundaryCondition == "Periodic":
        # if mean(mass) ~ mean(Universe)
        alpha1, alpha2 = cf.call_kappa0_to_alphas_p(kappa, inp.bsz_arc, inp.nnn)
    elif BoundaryCondition == "Isolated":
        # if mean(mass) >> mean(Universe)
        alpha1, alpha2 = cf.call_kappa0_to_alphas(kappa, inp.bsz_arc, inp.nnn)
    else:
        print("You should define the Boundary Condition first!!!")
    #---------------------------------------
    # Calculate higher order lensing maps
    #

    al11, al12 = np.gradient(alpha1, inp.dsx_arc)
    al21, al22 = np.gradient(alpha2, inp.dsx_arc)
    # mua = 1.0/(1.0 - (al11 + al22) + al11*al22 - al12*al21)
    shear1 = 0.5*(al11 - al22)
    shear2 = 0.5*(al12 + al21)

    return snapid, zl_median, zs, kappa, alpha1, alpha2, shear1, shear2


def create_grid_maps_for_zs0(haloID, skip_sdens=False):
    
    data = h5py.File(inp.outputs_path + haloID + '_' + str(inp.zs0) + '_gmaps.hdf5', 'w')

    nslices = len(inp.snapid_list)

    for i in range(nslices):
        output_ar = convert_one_from_dir(inp.input_prtcls_dir+"STEPCutout",
                                         inp.snapid_list[i], skip_sdens=skip_sdens)
        if output_ar == 1:
            continue
        else:
            snapid_ar = output_ar[0]
            zl_ar = output_ar[1]
            zs_ar = output_ar[2]
            kappa0_ar = output_ar[3]
            alpha1_ar = output_ar[4]
            alpha2_ar = output_ar[5]
            shear1_ar = output_ar[6]
            shear2_ar = output_ar[7]

        del output_ar
 
        lens_plane = data.create_group('{}'.format(inp.snapid_list[i]))
        lens_plane.create_dataset('zl', data=zl_ar, dtype='float32')
        lens_plane.create_dataset('zs', data=zs_ar, dtype='float32')
        lens_plane.create_dataset('kappa0', data=kappa0_ar, dtype='float32')
        lens_plane.create_dataset('alpha1', data=alpha1_ar, dtype='float32')
        lens_plane.create_dataset('alpha2', data=alpha2_ar, dtype='float32')
        lens_plane.create_dataset('shear1', data=shear1_ar, dtype='float32')
        lens_plane.create_dataset('shear2', data=shear2_ar, dtype='float32')

    del snapid_ar, zl_ar, zs_ar
    del kappa0_ar, alpha1_ar, alpha2_ar, shear1_ar, shear2_ar

    data.close()


if __name__ == '__main__':
    halo_id = inp.halo_info[:-1]
    create_grid_maps_for_zs0(halo_id, skip_sdens=True)
