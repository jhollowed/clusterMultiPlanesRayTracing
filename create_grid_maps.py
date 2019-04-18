import numpy as np
from astropy.table import Table
import h5py

import cfuncs as cf
import inps
import pdb
import matplotlib.pyplot as plt
import subprocess
import os
import pdb
import glob

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
    
    print('\n---------- working on {} ----------'.format(snapid))
    
    # Load in particles of one lens plane    
    # write out/read in denisty file
    write_dir = '{}/dtfe_dens'.format(inp.outputs_path)
    if not os.path.exists(write_dir):
            os.makedirs(write_dir)
    dtfe_file = '{}/{}_dtfe_input.bin'.format(write_dir, snapid)
    
    # read in particle data
    print('finding shell redshift')
    zl_array = np.fromfile(file_name_base+str(snapid)+"/"+"redshift."+str(snapid)+".bin", dtype = "f")
    zl_median = np.median(zl_array)     
    mp_array = np.ones(len(zl_array))*inp.mpp
        
    zs = inp.zs0
    ncc = inp.nnn
    bsz_mpc = inp.bsz_arc*cf.Dc(zl_median)/cf.apr
    # dsx_mpc = bsz_mpc/ncc
    
    noskip = False
    if(skip_sdens == True and noskip == False):    
        try:
            # read in result
            print('reading density'.format(snapid))
            sdens_cmpch = np.fromfile('{}.rho.bin'.format(dtfe_file))
        except FileNotFoundError: 
            print('Cant skip density calculation for {}; doesnt exist'.format(snapid))
            noskip = True
    
    if(skip_sdens == False or noskip == True):
        print('reading x, y, z')
        xxl_array = np.fromfile(file_name_base+str(snapid)+"/"+"x."+str(snapid)+".bin", dtype = "f")
        yyl_array = np.fromfile(file_name_base+str(snapid)+"/"+"y."+str(snapid)+".bin", dtype = "f")
        zzl_array = np.fromfile(file_name_base+str(snapid)+"/"+"z."+str(snapid)+".bin", dtype = "f")
       
        if len(xxl_array) < 10:
            return 1
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
        x1in = x1_array
        x2in = x2_array
        x3in = x3_array
        mpin = mp_array
        npp = len(mpin)
        if npp < 10:
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
    
    
    print('computing convergence')
    sdens_cmpch = sdens_cmpch.reshape(inp.nnn, inp.nnn)
    kappa = sdens_cmpch*(1.0+zl_median)**2.0/cf.sigma_crit(zl_median,zs)
     
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

    print('computing defelctions and shears')
    al11, al12 = np.gradient(alpha1, inp.dsx_arc)
    al21, al22 = np.gradient(alpha2, inp.dsx_arc)
    # mua = 1.0/(1.0 - (al11 + al22) + al11*al22 - al12*al21)
    shear1 = 0.5*(al11 - al22)
    shear2 = 0.5*(al12 + al21)

    return snapid, zl_median, zs, kappa, alpha1, alpha2, shear1, shear2


def create_grid_maps_for_zs0(haloID, skip_sdens=False):
    
    print('\n ---------- creating grid maps for halo {} ---------- '.format(haloID))

    out_file = h5py.File('{}/{}_{}_gmaps.hdf5'.format(inp.outputs_path, haloID, inp.zs0), 'w')
    print('created out file at {}'.format(out_file.filename))
    print('reading lightcone cutout files at {}'.format(inp.input_prtcls_dir))

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
 
        zl_group = '{}'.format(inp.snapid_list[i])
        out_file.create_group(zl_group)
        out_file[zl_group]['zl'] = np.atleast_1d(zl_ar).astype('float32')
        out_file[zl_group]['zs'] = np.atleast_1d(zs_ar).astype('float32')
        out_file[zl_group]['kappa0'] = kappa0_ar.astype('float32')
        out_file[zl_group]['alpha1'] = alpha1_ar.astype('float32')
        out_file[zl_group]['alpha2'] = alpha2_ar.astype('float32')
        out_file[zl_group]['shear1'] = shear1_ar.astype('float32')
        out_file[zl_group]['shear2'] = shear2_ar.astype('float32')

    out_file.close()
    print('done')


if __name__ == '__main__':
    
    halo_ids_avail = [s.split('/')[-1]+'/' for s in glob.glob('./data/lenses/prtcls/halo*')]
    for halo_id in halo_ids_avail:
        inp = inps.inputs(halo_id)
        halo_id = inp.halo_info[:-1]
        create_grid_maps_for_zs0(halo_id, skip_sdens=True)


