def convert_one_from_dir(file_name_base, snapid):
    #---------------------------------------
    # Load in particles of one lens plane
    #
    
    zl_array = np.fromfile(file_name_base+str(snapid)+"/"+"redshift."+str(snapid)+".bin", dtype = "f")
    zl_median = np.median(zl_array)
    mp_array = zl_array*0.0+inp.mpp

    xo1 = np.fromfile(file_name_base+str(snapid)+"/"+"theta."+snapid+".bin",dtype = "f")
    xc1 = (xo1.max()+xo1.min())*0.5
    x1_array = (xo1-xc1)*cf.Dc(zl_array)/cf.apr

    xo2 = np.fromfile(file_name_base+str(snapid)+"/"+"phi."+snapid+".bin",dtype = "f")
    xc2 = (xo2.max()+xo2.min())*0.5
    x2_array = (xo2-xc2)*cf.Dc(zl_array)/cf.apr

    xo3 = cf.Dc(zl_array)
    xc3 = (xo3.max()+xo3.min())*0.5
    x3_array = xo3 - xc3
    #---------------------------------------
    # Need talk to Joe, how to conver x, y, z to ra, dec, z
    #
    #
    # xo1 = np.fromfile(file_name_base+str(snapid)+"/"+"x."+snapid+".bin",dtype = ">f")
    # xo2 = np.fromfile(file_name_base+str(snapid)+"/"+"y."+snapid+".bin",dtype = ">f")
    # xo3 = np.fromfile(file_name_base+str(snapid)+"/"+"z."+snapid+".bin",dtype = ">f")
    # xpol3, xpol1, xpol2 =  cart2pol3d(xo1, xo2, xo3)

    # x1_array = np.rad2deg((xpol1 - (xpol1.max()+xpol1.min())*0.5))*3600.*cf.Dc(zl)/cf.apr
    # x2_array = np.rad2deg((xpol2 - (xpol2.max()+xpol2.min())*0.5))*3600.*cf.Dc(zl)/cf.apr
    # x3_array =  xpol3 - (xpol3.max()+xpol3.min())*0.5
    #---------------------------------------
    # Calculate convergence maps
    #
    
    zs = inp.zs0
    ncc = inp.nnn
    bsz_mpc = inp.bsz_arc*cf.Dc(zl_median)/cf.apr
    dsx_mpc = bsz_mpc/ncc

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
    if npp < 200:
        return 1

    sdens_cmpch = cf.call_sph_sdens_weight_omp(x1in,x2in,x3in,mpin,bsz_mpc,ncc)

    file_out_sdens =  inp.sdens_path+str(snapid)+"_"+'%.6f'%(zl_median)+"_sdens.bin"
    sdens_cmpch.astype(np.float32).tofile(file_out_sdens)

    kappa = sdens_cmpch*(1.0+zl_median)**2.0/cf.sigma_crit(zl_median,zs)
    #---------------------------------------
    # Calculate deflection maps
    #
    
    if snapid == '382': # The Snapshot contains the main halo.
        BoundaryCondition = "Isolated"
    else:
        BoundaryCondition = "Periodic"

    BoundaryCondition = "Isolated"
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
    mua = 1.0/(1.0 - (al11 + al22) + al11*al22 - al12*al21)
    shear1 = 0.5*(al11 - al22)
    shear2 = 0.5*(al12 + al21)

    return snapid, zl_median, zs, kappa, alpha1, alpha2, shear1, shear2


def main_create_grid_maps_hdf5(haloID):
    data = Table()
    
    nslices = len(inp.snapid_list)
    
    snapid_ar = []
    zl_ar = []
    zs_ar = []
    kappa0_ar = []
    alpha1_ar = []
    alpha2_ar = []
    shear1_ar = []
    shear2_ar = []
    
    for i in range(nslices):
        output_ar = convert_one_from_dir(inp.input_prtcls_dir+"lcCutout", 
                                         inp.snapid_list[i])
        if output_ar == 1:
            continue
        else:
            snapid_ar.append(output_ar[0])
            zl_ar.append(output_ar[1])
            zs_ar.append(output_ar[2])
            kappa0_ar.append(output_ar[3])
            alpha1_ar.append(output_ar[4])
            alpha2_ar.append(output_ar[5])
            shear1_ar.append(output_ar[6])
            shear2_ar.append(output_ar[7])
        
        del output_ar

    data['snapID'] = np.array(snapid_ar, dtype="int")
    data['zl']     = np.array(zl_ar, dtype="float32")
    data['zs']     = np.array(zs_ar, dtype="float32")

    data['kappa0'] = np.array(kappa0_ar, dtype="float32")
    data['alpha1'] = np.array(alpha1_ar, dtype="float32")
    data['alpha2'] = np.array(alpha2_ar, dtype="float32")
    data['shear1'] = np.array(shear1_ar, dtype="float32")
    data['shear2'] = np.array(shear2_ar, dtype="float32")

    del snapid_ar, zl_ar, zs_ar
    del kappa0_ar, alpha1_ar, alpha2_ar, shear1_ar, shear2_ar

    data.write('./outputs/' + haloID + '_gmaps.hdf5',
                       path="/grids_maps", append=True, overwrite=True)#, compression=True)

    return 0