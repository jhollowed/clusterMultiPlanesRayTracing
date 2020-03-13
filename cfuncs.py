#---------------------------------------------------------------------------------
# Call C functions
#
import sys,os
import halo_inputs as inp
import numpy as np
import ctypes as ct
import cosmology as cm

# this will have to be manually toggled for running on mac with gcc9, or running on alcf
gcc9 = False
gcc9_str = ''
if gcc9: gcc9_str = '9'

here = os.path.abspath(os.path.dirname(__file__))
lib_path = os.path.join(here, 'lib/lib_gcc{}/'.format(gcc9_str))
vc = cm.vc
G = cm.G
apr = cm.apr

#---------------------------------------------------------------------------------
sps = ct.CDLL(lib_path+"lib_so_sph_w_omp/libsphsdens.so")

sps.cal_sph_sdens_weight.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    ct.c_float,ct.c_long,ct.c_float,ct.c_long,ct.c_long, \
                                    ct.c_float,ct.c_float,ct.c_float, \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_float)]

sps.cal_sph_sdens_weight.restype  = ct.c_int

def call_sph_sdens_weight(x1,x2,x3,mp,Bsz,Ncc):

    x1_in = np.array(x1,dtype=ct.c_float)
    x2_in = np.array(x2,dtype=ct.c_float)
    x3_in = np.array(x3,dtype=ct.c_float)
    mp_in = np.array(mp,dtype=ct.c_float)
    dcl = ct.c_float(Bsz/Ncc)
    Ngb = ct.c_long(16)
    xc1 = ct.c_float(0.0)
    xc2 = ct.c_float(0.0)
    xc3 = ct.c_float(0.0)
    Np  = len(mp)
    posx1 = np.zeros((Ncc,Ncc),dtype=ct.c_float)
    posx2 = np.zeros((Ncc,Ncc),dtype=ct.c_float)
    sdens = np.zeros((Ncc,Ncc),dtype=ct.c_float)

    sps.cal_sph_sdens_weight(x1_in,x2_in,x3_in,mp_in,ct.c_float(Bsz),ct.c_long(Ncc),dcl,Ngb,ct.c_long(Np),xc1,xc2,xc3,posx1,posx2,sdens);
    return sdens*mp.sum()/(sdens.sum()*dcl*dcl)
#---------------------------------------------------------------------------------


sps.cal_sph_sdens_weight_omp.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        ct.c_float,ct.c_long,ct.c_float,ct.c_long,ct.c_long, \
                                        ct.c_float,ct.c_float,ct.c_float, \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                        np.ctypeslib.ndpointer(dtype = ct.c_float)]

sps.cal_sph_sdens_weight_omp.restype  = ct.c_int

def call_sph_sdens_weight_omp(x1,x2,x3,mp,Bsz,Nc):

    x1_in = np.array(x1,dtype=ct.c_float)
    x2_in = np.array(x2,dtype=ct.c_float)
    x3_in = np.array(x3,dtype=ct.c_float)
    mp_in = np.array(mp,dtype=ct.c_float)
    dcl = ct.c_float(Bsz/Nc)
    Ngb = ct.c_long(16)
    xc1 = ct.c_float(0.0)
    xc2 = ct.c_float(0.0)
    xc3 = ct.c_float(0.0)
    Np  = len(mp)
    posx1 = np.zeros((Nc,Nc),dtype=ct.c_float)
    posx2 = np.zeros((Nc,Nc),dtype=ct.c_float)
    sdens = np.zeros((Nc,Nc),dtype=ct.c_float)

    sps.cal_sph_sdens_weight_omp(x1_in,x2_in,x3_in,mp_in,ct.c_float(Bsz),ct.c_long(Nc),dcl,Ngb,ct.c_long(Np),xc1,xc2,xc3,posx1,posx2,sdens);
    return sdens*mp.sum()/(sdens.sum()*dcl*dcl)

#---------------------------------------------------------------------------------
gls = ct.CDLL(lib_path+"lib_so_cgls/libglsg.so")
gls.kappa0_to_alphas.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                 ct.c_int,ct.c_double,\
                                 np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                 np.ctypeslib.ndpointer(dtype = ct.c_double)]
gls.kappa0_to_alphas.restype  = ct.c_void_p

def call_kappa0_to_alphas(Kappa, Bsz, Ncc):
    kappa0 = np.array(Kappa, dtype=ct.c_double)
    alpha1 = np.zeros((Ncc,Ncc),dtype=ct.c_double)
    alpha2 = np.zeros((Ncc,Ncc),dtype=ct.c_double)
    gls.kappa0_to_alphas(kappa0,Ncc,Bsz,alpha1,alpha2)
    return alpha1,alpha2

#--------------------------------------------------------------------
gls_p = ct.CDLL(lib_path+"lib_so_cgls_p/libglsg_p.so")

gls_p.kappa0_to_alphas.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                   ct.c_int,ct.c_double, \
                                   np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                   np.ctypeslib.ndpointer(dtype = ct.c_double)]
gls_p.kappa0_to_alphas.restype  = ct.c_void_p

def call_kappa0_to_alphas_p(Kappa, Bsz, Ncc):
    kappa0 = np.array(Kappa, dtype=ct.c_double)
    alpha1 = np.zeros((Ncc,Ncc),dtype=ct.c_double)
    alpha2 = np.zeros((Ncc,Ncc),dtype=ct.c_double)

    gls_p.kappa0_to_alphas(kappa0, ct.c_int(Ncc), ct.c_double(Bsz), alpha1, alpha2)

    return alpha1, alpha2

#--------------------------------------------------------------------
lzos = ct.CDLL(lib_path+"lib_so_lzos/liblzos.so")
lzos.lanczos_diff_2_tag.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                    ct.c_double,ct.c_int,ct.c_int]
lzos.lanczos_diff_2_tag.restype  = ct.c_void_p

def call_lanczos_derivative(alpha1,alpha2,Bsz,Ncc):

    dif_tag = 2

    dcl = Bsz/Ncc

    m1 = np.array(alpha1,dtype=ct.c_double)
    m2 = np.array(alpha2,dtype=ct.c_double)

    m11 = m1*0.0
    m12 = m1*0.0
    m21 = m2*0.0
    m22 = m2*0.0

    lzos.lanczos_diff_2_tag(m1,m2,m11,m12,m21,m22,ct.c_double(dcl),ct.c_int(Ncc),ct.c_int(dif_tag))

    return m11,m12,m21,m22

#---------------------------------------------------------------------------------
sngp = ct.CDLL(lib_path + "lib_so_omp_ngp/libngp.so")
sngp.ngp_w_rebin.argtypes =[np.ctypeslib.ndpointer(dtype = ct.c_float), \
                            np.ctypeslib.ndpointer(dtype = ct.c_float), \
                            np.ctypeslib.ndpointer(dtype = ct.c_float), \
                            ct.c_int,ct.c_float,ct.c_float,ct.c_float, \
                            ct.c_int,ct.c_int, \
                            np.ctypeslib.ndpointer(dtype = ct.c_float)]

sngp.ngp_w_rebin.restype  = ct.c_int

def call_ngp_w_rebin(x1,x2,mpp,Bsz,Nc):

    Np  = ct.c_int(mpp.size)
    dcl = ct.c_float(Bsz/Nc)
    xc1 = ct.c_float(0.0)
    xc2 = ct.c_float(0.0)
    x1 = np.array(x1,dtype=ct.c_float)
    x2 = np.array(x2,dtype=ct.c_float)
    mpp= np.array(mpp,dtype=ct.c_float)

    sdens = np.zeros((Nc,Nc),dtype=ct.c_float)

    sngp.ngp_w_rebin(x1,x2,mpp,Np,xc1,xc2,dcl,ct.c_int(Nc),ct.c_int(Nc),sdens)
    return sdens
#---------------------------------------------------------------------------------
rtf = ct.CDLL(lib_path + "lib_so_icic/librtf.so")

rtf.inverse_cic.argtypes = [np.ctypeslib.ndpointer(dtype =  ct.c_double),\
                            np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                            np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                            ct.c_double,ct.c_double,ct.c_double, \
                            ct.c_int,ct.c_int,ct.c_int,ct.c_int,\
                            np.ctypeslib.ndpointer(dtype = ct.c_double)]
rtf.inverse_cic.restype  = ct.c_void_p

def call_inverse_cic(img_in, yc1, yc2, yi1, yi2, dsi):
    ny1,ny2 = np.shape(img_in)
    nx1,nx2 = np.shape(yi1)

    img_in = np.array(img_in,dtype=ct.c_double)

    yi1 = np.array(yi1,dtype=ct.c_double)
    yi2 = np.array(yi2,dtype=ct.c_double)

    img_out = np.zeros((nx1,nx2))

    rtf.inverse_cic(img_in,yi1,yi2,ct.c_double(yc1),ct.c_double(yc2),ct.c_double(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nx1),ct.c_int(nx2),img_out)
    return img_out.reshape((nx1,nx2))

#--------------------------------------------------------------------
rtf.inverse_cic_omp.argtypes = [np.ctypeslib.ndpointer(dtype =  ct.c_double),\
                                np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                                np.ctypeslib.ndpointer(dtype =  ct.c_double), \
                                ct.c_double,ct.c_double,ct.c_double, \
                                ct.c_int,ct.c_int,ct.c_int,ct.c_int,\
                                np.ctypeslib.ndpointer(dtype = ct.c_double)]
rtf.inverse_cic_omp.restype  = ct.c_void_p

def call_inverse_cic_omp(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    nx1,nx2 = np.shape(yi1)
    img_in = np.array(img_in,dtype=ct.c_double)
    yi1 = np.array(yi1,dtype=ct.c_double)
    yi2 = np.array(yi2,dtype=ct.c_double)
    img_out = np.zeros((nx1,nx2))

    rtf.inverse_cic_omp(img_in,yi1,yi2,ct.c_double(yc1),ct.c_double(yc2),ct.c_double(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nx1),ct.c_int(nx2),img_out)
    return img_out.reshape((nx1,nx2))

#--------------------------------------------------------------------
rtf.inverse_cic_single.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                   ct.c_float,ct.c_float,ct.c_float,ct.c_int,ct.c_int,ct.c_int, \
                                   np.ctypeslib.ndpointer(dtype = ct.c_float)]
rtf.inverse_cic_single.restype  = ct.c_void_p

def call_inverse_cic_single(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    img_in = np.array(img_in,dtype=ct.c_float)
    yi1 = np.array(yi1,dtype=ct.c_float)
    yi2 = np.array(yi2,dtype=ct.c_float)
    nlimgs = len(yi1)
    img_out = np.zeros((nlimgs),dtype=ct.c_float)

    rtf.inverse_cic_single(img_in,yi1,yi2,ct.c_float(yc1),ct.c_float(yc2),ct.c_float(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nlimgs),img_out)
    return img_out

#--------------------------------------------------------------------
rtf.inverse_cic_omp_single.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float), \
                                       ct.c_float,ct.c_float,ct.c_float,ct.c_int,ct.c_int,ct.c_int, \
                                       np.ctypeslib.ndpointer(dtype = ct.c_float)]
rtf.inverse_cic_omp_single.restype  = ct.c_void_p

def call_inverse_cic_single_omp(img_in,yc1,yc2,yi1,yi2,dsi):
    ny1,ny2 = np.shape(img_in)
    img_in = np.array(img_in,dtype=ct.c_float)
    yi1 = np.array(yi1,dtype=ct.c_float)
    yi2 = np.array(yi2,dtype=ct.c_float)
    nlimgs = len(yi1)
    img_out = np.zeros((nlimgs),dtype=ct.c_float)

    rtf.inverse_cic_omp_single(img_in,yi1,yi2,ct.c_float(yc1),ct.c_float(yc2),ct.c_float(dsi),ct.c_int(ny1),ct.c_int(ny2),ct.c_int(nlimgs),img_out)
    return img_out
#--------------------------------------------------------------------
tri = ct.CDLL(lib_path+"lib_so_tri_roots/libtri.so")
tri.PIT.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                    np.ctypeslib.ndpointer(dtype = ct.c_double), \
                    np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.PIT.restype  = ct.c_bool

def call_PIT(pt,v0,v1,v2):

    pt_in = np.array(pt,dtype=ct.c_double)
    v0_in = np.array(v0,dtype=ct.c_double)
    v1_in = np.array(v1,dtype=ct.c_double)
    v2_in = np.array(v2,dtype=ct.c_double)

    res = tri.PIT(pt_in,v0_in,v1_in,v2_in)
    return res

#--------------------------------------------------------------------
tri.Cart2Bary.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.Cart2Bary.restype  = ct.c_void_p

def call_cart_to_bary(pt,v0,v1,v2):

    pt_in = np.array(pt,dtype=ct.c_double)
    v0_in = np.array(v0,dtype=ct.c_double)
    v1_in = np.array(v1,dtype=ct.c_double)
    v2_in = np.array(v2,dtype=ct.c_double)

    bary_out = np.array([0,0,0],dtype=ct.c_double)
    tri.Cart2Bary(pt_in,v0_in,v1_in,v2_in,bary_out)
    return bary_out

#--------------------------------------------------------------------
tri.bary2cart.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double), \
                          np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.bary2cart.restype  = ct.c_void_p

def call_bary_to_cart(v0,v1,v2,bary):
    v0_in = np.array(v0,dtype=ct.c_double)
    v1_in = np.array(v1,dtype=ct.c_double)
    v2_in = np.array(v2,dtype=ct.c_double)
    bary_in = np.array(bary,dtype=ct.c_double)

    pt_out = np.array([0,0],dtype=ct.c_double)

    tri.bary2cart(v0_in,v1_in,v2_in,bary_in,pt_out)
    return pt_out

#--------------------------------------------------------------------
tri.mapping_triangles.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                  ct.c_int, \
                                  np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.mapping_triangles.restype  = ct.c_void_p

def call_mapping_triangles(pys,xi1,xi2,yi1,yi2):

    pys_in = np.array(pys,dtype=ct.c_double)
    xi1_in = np.array(xi1,dtype=ct.c_double)
    xi2_in = np.array(xi2,dtype=ct.c_double)
    yi1_in = np.array(yi1,dtype=ct.c_double)
    yi2_in = np.array(yi2,dtype=ct.c_double)
    nc_in = ct.c_int(np.shape(xi1)[0])

    xroots_out = np.zeros((10),dtype=ct.c_double)

    tri.mapping_triangles(pys_in,xi1_in,xi2_in,yi1_in,yi2_in,nc_in,xroots_out)

    return xroots_out

#--------------------------------------------------------------------
tri.mapping_triangles_arrays.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                         np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                         ct.c_int,
                                         np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                         np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                         np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                         np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                         ct.c_int, \
                                         np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                         np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.mapping_triangles_arrays.restype  = ct.c_void_p

def call_mapping_triangles_arrays(ys1,ys2,xi1,xi2,yi1,yi2):

    ys1_in = np.array(ys1,dtype=ct.c_double)
    ys2_in = np.array(ys2,dtype=ct.c_double)
    xi1_in = np.array(xi1,dtype=ct.c_double)
    xi2_in = np.array(xi2,dtype=ct.c_double)
    yi1_in = np.array(yi1,dtype=ct.c_double)
    yi2_in = np.array(yi2,dtype=ct.c_double)

    ngals_in = ct.c_int(len(ys1_in))
    nc_in = ct.c_int(np.shape(xi1)[0])

    xr1_out = ys1_in*0.0
    xr2_out = ys2_in*0.0

    tri.mapping_triangles_arrays(ys1_in,ys2_in,ngals_in,xi1_in,xi2_in,yi1_in,yi2_in,nc_in,xr1_out,xr2_out)

    return xr1_out, xr2_out

#--------------------------------------------------------------------
tri.mapping_triangles_arrays_omp.argtypes = [np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                             np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                             ct.c_int,
                                             np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                             np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                             np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                             np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                             ct.c_int, \
                                             np.ctypeslib.ndpointer(dtype = ct.c_double), \
                                             np.ctypeslib.ndpointer(dtype = ct.c_double)]
tri.mapping_triangles_arrays_omp.restype  = ct.c_void_p

def call_mapping_triangles_arrays_omp(ys1,ys2,xi1,xi2,yi1,yi2):

    ys1_in = np.array(ys1,dtype=ct.c_double)
    ys2_in = np.array(ys2,dtype=ct.c_double)
    xi1_in = np.array(xi1,dtype=ct.c_double)
    xi2_in = np.array(xi2,dtype=ct.c_double)
    yi1_in = np.array(yi1,dtype=ct.c_double)
    yi2_in = np.array(yi2,dtype=ct.c_double)

    ngals_in = ct.c_int(len(ys1_in))
    nc_in = ct.c_int(np.shape(xi1)[0])

    xr1_out = ys1_in*0.0
    xr2_out = ys2_in*0.0

    tri.mapping_triangles_arrays_omp(ys1_in,ys2_in,ngals_in,xi1_in,xi2_in,yi1_in,yi2_in,nc_in,xr1_out,xr2_out)

    return xr1_out, xr2_out
#--------------------------------------------------------------------

def cart2pol3d(x, y, z):
    r_pol = np.sqrt(x**2 + y**2 + z**2)
    theta_pol = np.arccos(z/r_pol)
    phi_pol = np.arctan2(y, x)
    return r_pol, theta_pol, phi_pol

def ai_to_ah(ai,zl,zs):
    res = cm.Da(zs)/cm.Da2(zl,zs)*ai
    return res

def ah_to_ai(ah,zl,zs):
    res = cm.Da2(zl,zs)/cm.Da(zs)*ah
    return res

def al_zs1_to_zs2(ai,zl,zs1,zs2):
    res = cm.Da(zs1)/cm.Da2(zl,zs1)*cm.Da2(zl,zs2)/cm.Da(zs2)*ai
    return res

def alphas_to_mu(alpha1, alpha2, Bsz, Ncc):
    al11,al12,al21,al22 = call_lanczos_derivative(alpha1,alpha2,Bsz,Ncc)

    al11[:2, :] = 0.0;al11[-2:,:] = 0.0;al11[:, :2] = 0.0;al11[:,-2:] = 0.0
    al12[:2, :] = 0.0;al12[-2:,:] = 0.0;al12[:, :2] = 0.0;al12[:,-2:] = 0.0
    al21[:2, :] = 0.0;al21[-2:,:] = 0.0;al21[:, :2] = 0.0;al21[:,-2:] = 0.0
    al22[:2, :] = 0.0;al22[-2:,:] = 0.0;al22[:, :2] = 0.0;al22[:,-2:] = 0.0

    res = 1.0/(al11*al22-(al11+al22)-al12*al21+1.0)

    return res
