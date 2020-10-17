import os
import sys
import pdb
import scipy
import pathlib
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib import rc
from astropy import units as u
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from colossus.cosmology import cosmology as colcos
from astropy.cosmology import WMAP7, z_at_value
from halotools.empirical_models import NFWProfile
from colossus.halo.concentration import concentration as mass_conc
rc('text', usetex=True)

sys.path.append('{}/..'.format(pathlib.Path(__file__).parent.absolute()))
import cosmology as cm


# =========================================================================================


class NFW:
    def __init__(self, z, m200c=None, r200c=None, c=None, cM_err=False, cosmo=cm.OuterRim_params, seed=None):
        """
        Class for generating NFW test-case input files for the ray tracing modules supplied in
        the directory above. This class is constructed with a halo mass, redshift, and 
        cosmological model, and builds a HaloTools NFWProfile object. The methods provided here 
        can then populate the profile with a particle distribution realization in 3 dimensions, 
        and output the result in the form expected by the raytracing modules.
        
        Parameters
        ----------
        z : float 
            The redshift of the halo.
        m200c : float
            The mass of the halo within a radius containing 200*rho_crit, in M_sun. If not passed, 
            then r200c must be supplied
        r200c : float
            The radius of the halo enclosing a mean density of 200*rho_crit, in Mpc. If not passed,
            then m200c must be supplied.
        c : float, optional 
            The concentration of the halo. If not given, samples from a Gaussian 
            with location and scale suggested by the M-c relation of Child+2018
        cM_err : bool, optional 
            Whether or not to impose scatter on the cM relation used to draw a concentration, 
            in the case that the argument c is not passed. If False, the concentration drawn 
            will always lie exactly on the cM relation used (currently Child+ 2018). Defaults
            to True. In either case, the 'sod_halo_cdelta_error' quntity in the output halo propery
            csv file will be zero.
        cosmo : object, optional
            An AstroPy cosmology object. Defaults to OuterRim parameters.
        seed : float, optional
            Random seed to pass to HaloTools for generation of radial particle positions, and
            use for drawing concentrations and angular positions of particles. Defaults to None
            (giving stochastic output)
        
        Methods
        -------
        populate_halo(r)
            Uses HaloTools to generate a MonteCarlo realization of discrete tracers of the density
            profile (particles)
        output_particles():
            Writes out the particle positions generated by populate_halo() to a form that is prepped 
            for input to the ray tracing modules of this package.
        """
       
        assert (m200c is not None or r200c is not None), "Either m200c (in M_sun) or r200c (in Mpc) must be supplied"
        self.m200c = m200c
        self.r200c = r200c
        self.redshift = z
        self.cosmo = cosmo
        self.seed = seed

        self.profile = NFWProfile(cosmology=self.cosmo, redshift=self.redshift, mdef = '200c')
        
        # HaloTools and Colossus expect masses,radii with h dependence, so scale accordingly on input and output
        if(r200c is None):
            self.m200ch = m200c * cosmo.h # M_sun/h
            self.r200ch = self.profile.halo_mass_to_halo_radius(self.m200ch) #proper Mpc/h
            self.r200c = self.r200ch / cosmo.h # proper Mpc
        if(m200c is None):
            self.r200ch = r200c * cosmo.h # proper Mpc/h
            self.m200ch = self.profile.halo_radius_to_halo_mass(self.r200ch) # M_sun/h
            self.m200c = self.m200ch / cosmo.h # M_sun
            
        # these to be filled by populate_halo()
        self.r = None
        self.theta = None
        self.phi = None
        self.mpp = None
        self.max_rfrac = None
        self.populated = False
        
        if c is not None:
            self.c = c
            self.c_err = 0
        else:
            # if cM_err=True, draw a concentration from gaussian, otherwise use Child+2018 cM relation scatter-free
            rand = np.random.RandomState(self.seed)
            
            cosmo_colossus = colcos.setCosmology('OuterRim',
                             {'Om0':cosmo.Om0, 'Ob0':cosmo.Ob0, 'H0':cosmo.H0.value, 'sigma8':0.8, 
                              'ns':0.963, 'relspecies':False})
            c_u = mass_conc(self.m200ch, '200c', z, model='child18')
            if(cM_err):
                c_sig = c_u/3
                self.c = rand.normal(loc=c_u, scale=c_sig) 
            else:
                self.c = c_u
            self.c_err = 0
     
    
    # -----------------------------------------------------------------------------------------------


    def populate_halo(self, N=10000, rfrac=1, rfrac_los=None):
        """
        Generates a 3-dimensional relization of the discreteley-sampled NFW mass distribution for
        this halo. The radial positions are obtained with the HaloTools 
        mc_generate_nfw_radial_positions module. The angular positions are drawn from a uniform 
        random distribution, the azimuthal coordiante ranging from 0 to 2pi, and the coaltitude 
        from 0 to pi.

        Parameters
        ----------
        N : int
            The number of particles to drawn
        rfrac: float, optional
            Multiplier of r200c which sets the maximum radial extent of the population
            (concentration will be scaled as well, as c=r200c/r_s). Defaults to 1
        rfrac_los: float, optional
            Multiplier of r200c which sets the maaximum extent of the population in the line-of-sight (LOS)
            dimension, i.e. it clips the halo along the LOS. If rfrac_los = 0.5 the LOS dimension of the halo 
            will be clipped 0.5*r200c toward the observer, and again away from the observer, with respect to 
            the halo center. Note that this does *not* rescale mpp, and is therefore almost totally useless...
            the argument is kept for rather specific debugging purposes, but probably should not be used. 
            Default is None, in which case no clipping is performed. Also if rfrac_los > rfrac, obviously 
            nothing will happen.
        """
       
        self.populated = True
        
        # the radial positions in proper Mpc
        r = self.profile.mc_generate_nfw_radial_positions(num_pts = N, conc = rfrac * self.c, 
                                                          halo_radius = rfrac * self.r200ch, seed=self.seed+1)
        self.r = r / self.cosmo.h
        self.max_rfrac = rfrac

        # compute mass enclosed to find mass per particle
        # (this is the analytic integration of the NFW profile in terms of m_200c, assuming c=c_200c)
        rs = self.r200c / self.c
        rmax = rfrac * self.r200c
        n = np.log((rs+rmax)/rs) - rmax/(rmax+rs)
        d = np.log(1+self.c) - self.c/(1+self.c)
        M_enc = self.m200c * n/d
        self.mpp = M_enc / N

        # radial positions need to be in comoving comoving coordiantes, as the kappa maps in the raytracing
        # modules expect the density estimation to be done on a comoving set of particles        
        self.r = self.r * (1+self.redshift)
        
        # now let's add in uniform random positions in the angular coordinates as well
        # Note that this is not the same as a uniform distribution in theta and phi 
        # over [0, pi] and [0, 2pi], since the area element on a sphere is a function of 
        # the coaltitude! See http://mathworld.wolfram.com/SpherePointPicking.html 
        rand = np.random.RandomState(self.seed) 
        v = rand.uniform(low=0, high=1, size = len(r))
        self.phi = rand.uniform(low=0, high=2*np.pi, size = len(r))
        self.theta = np.arccos(2*v-1)
        
        # finally, do los clipping if user requested (in self.output_particles below, the los dimension 
        # is assumed to be the cartesian x)
        x = self.r *  np.sin(self.theta) * np.cos(self.phi)
        if(rfrac_los is not None):
            los_mask = (np.abs(x)/self.r200c) <= rfrac_los
            self.r, self.theta, self.phi = self.r[los_mask], self.theta[los_mask], self.phi[los_mask]
    
    
    # -----------------------------------------------------------------------------------------------
    


    def populate_halo_fov(self, N=10000, rfrac=1, depth=None):
        """
        eventually merge with function above...
        """
       
        self.populated = True
        
        # modify requested radius to fill the encapsulating fov volume
        if(depth < rfrac or depth is None): depth = rfrac
        radius = max([rfrac, depth])
        halo_radius = np.sqrt(2) * radius * self.r200ch
        conc = np.sqrt(2) * radius * self.c
        
        # the radial positions in proper Mpc
        r = self.profile.mc_generate_nfw_radial_positions(num_pts = N, conc = conc, 
                                                          halo_radius = halo_radius, seed=self.seed+1)
        self.r = r / self.cosmo.h
        self.max_rfrac = rfrac

        # compute mass enclosed to find mass per particle
        # (this is the analytic integration of the NFW profile in terms of m_200c, assuming c=c_200c)
        rs = self.r200c / self.c
        rmax = rfrac * self.r200c
        n = np.log((rs+rmax)/rs) - rmax/(rmax+rs)
        d = np.log(1+self.c) - self.c/(1+self.c)
        M_enc = self.m200c * n/d
        self.mpp = M_enc / N

        # radial positions need to be in comoving comoving coordiantes, as the kappa maps in the raytracing
        # modules expect the density estimation to be done on a comoving set of particles        
        self.r = self.r * (1+self.redshift)
        
        # now let's add in uniform random positions in the angular coordinates as well
        # Note that this is not the same as a uniform distribution in theta and phi 
        # over [0, pi] and [0, 2pi], since the area element on a sphere is a function of 
        # the coaltitude! See http://mathworld.wolfram.com/SpherePointPicking.html 
        rand = np.random.RandomState(self.seed) 
        v = rand.uniform(low=0, high=1, size = len(r))
        self.phi = rand.uniform(low=0, high=2*np.pi, size = len(r))
        self.theta = np.arccos(2*v-1)
        
        # trim the particle population to the fov
        # move to polar coordinates, with x the los dimension
        x = self.r *  np.sin(self.theta) * np.cos(self.phi)
        y = self.r *  np.sin(self.theta) * np.sin(self.phi)
        z = self.r *  np.cos(self.theta)
        fov_mask = np.logical_and.reduce((np.abs(x) <= depth * self.r200c, 
                                          np.abs(y) <= rmax * self.r200c,
                                          np.abs(z) <= rmax * self.r200c))
        self.r, self.theta, self.phi = self.r[fov_mask], self.theta[fov_mask], self.phi[fov_mask] 
    
    
    # -----------------------------------------------------------------------------------------------
         
        
    def output_particles(self, output_dir='./nfw_particle_realization', vis_debug=False, vis_output_dir=None, 
                         fov_multiplier = None):
        """
        Computes three dimensional quantities for particles sampled along radial dimension. Each 
        quantity is output as little-endian binary files (expected input for ray-tracing modules
        in this package). The output quantities are x, y, z, theta, phi, redshift. In 
        cartesian space, the distribution is placed at a distance along the x-axis computed as 
        the comoving distance to the halo redshift by the input cosmology.

        Parameters
        ----------
        output_dir : string
            The desired output location for the binary files
        vis_debug : bool
            If True, display a 3d plot of the particles to be output for visual inspection
        vis_output_dir : string
            The desired output location for matplotlib figures images, if vis_debug is True
        fov_multiplier : float, optional
            The size of the field of view, which sets the scale of the eventual density 
            estimation, in projected comving Mpc, as a fraction of the largest radial particle 
            dispalcement. Defaults to None, in which case the field of view is set to be the 
            largest square that can fit entirely inside the projected halo.
        """
        
        if(self.populated == False):
            raise RuntimeError('populate_halo must be called before output_particles')
        if(vis_output_dir is None): vis_output_dir = output_dir
        if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

        # now find projected positions wrt origin after pushing halo down x-axis (Mpc and arcsec)
        self.halo_r = self.cosmo.comoving_distance(self.redshift).value
        x = self.r *  np.sin(self.theta) * np.cos(self.phi) + self.halo_r
        y = self.r *  np.sin(self.theta) * np.sin(self.phi)
        z = self.r *  np.cos(self.theta)
        r_fov = np.linalg.norm([y, z], axis=0)
        r_sky = np.linalg.norm([x,y,z], axis=0)
        theta_sky = np.arccos(z/r_sky) * 180/np.pi * 3600
        phi_sky = np.arctan(y/x) * 180/np.pi * 3600
       
        # get particle redshifts
        zmin = z_at_value(self.cosmo.comoving_distance, ((r_sky.min()-0.1)*u.Mpc))
        zmax = z_at_value(self.cosmo.comoving_distance, ((r_sky.max()+0.1)*u.Mpc))
        z_samp = np.linspace(zmin, zmax, 10)
        x_samp = self.cosmo.comoving_distance(z_samp).value
        invfunc = scipy.interpolate.interp1d(x_samp, z_samp)
        redshift = invfunc(r_sky)

        if(vis_debug):
            print(vis_output_dir)
            f = plt.figure(figsize=(12,6))
            ax = f.add_subplot(121, projection='3d')
            ax2 = f.add_subplot(122)

            ax.scatter(x, y, z, c='k', alpha=0.25)
            max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
            mid_x = (x.max()+x.min()) * 0.5
            mid_y = (y.max()+y.min()) * 0.5
            mid_z = (z.max()+z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.set_xlabel(r'$x\>[Mpc/h]$', fontsize=16)
            ax.set_ylabel(r'$y\>[Mpc/h]$', fontsize=16)
            ax.set_zlabel(r'$z\>[Mpc/h]$', fontsize=16)

            ax2.scatter(theta_sky, phi_sky, c='k', alpha=0.2)
            ax2.set_xlabel(r'$\theta\>[\mathrm{arsec}]$', fontsize=16)
            ax2.set_ylabel(r'$\phi\>[\mathrm{arcsec}]$', fontsize=16)
            plt.savefig('{}/nfw_particles.png'.format(vis_output_dir), dpi=300)

        # write out all to binary
        x.astype('f').tofile('{}/x.bin'.format(output_dir))
        y.astype('f').tofile('{}/y.bin'.format(output_dir))
        z.astype('f').tofile('{}/z.bin'.format(output_dir))
        theta_sky.astype('f').tofile('{}/theta.bin'.format(output_dir))
        phi_sky.astype('f').tofile('{}/phi.bin'.format(output_dir))
        redshift.astype('f').tofile('{}/redshift.bin'.format(output_dir))
    
        if(fov_multiplier is not None):
            fov_size = fov_multiplier * np.max(self.r)
        else:
            # the halo prop file records half the radius of the square FOV, which sets the scale for the density
            # estimation... we don't want the FOV to include any space outside of the region we have populated with
            # halos, else the density estiamtion will plummet at the boundary. Above, we populated the halo with
            # particles out to rfrac * r200c. The largest square that can fit inside the projection of this NFW sphere
            # then has a side length of 2*(rfrac*r200c)/sqrt(2) --> radius = (rfrac*r200c)/sqrt(2). 
            # Replace rfrac*r200c by the radial distance to the furthest particle and trim by 5%, to be safe.
            # Also note that self.r is a comoving distance, which is correct
            fov_size = 0.95 * (np.max(r_fov) / np.sqrt(2))
        self._write_prop_file(fov_size, output_dir)
    
    
    # -----------------------------------------------------------------------------------------------


    def _write_prop_file(self, fov_radius, output_dir):
        """
        Writes a csv file contining the halo properties needed by this package's ray tracing modules
        The boxRadius can really be anything, since the space around the NFW ball is empty-- here, we
        set it to correspond to a transverse comoving distance equal to R*r200 at the redshift of the
        halo.

        Parameters
        ----------
        fov_radius : float
            Half of the square FOV side length (this scale will be used in later calls to the density estaimtor)

        output_dir : string, optional
            The desired output location for the property file. Defaults to a subdir created at the 
            location of this module.
        """

        # find the angular scale corresponding to fov_r200c * r200c in proper Mpc at the redshift of the halo
        boxRadius_Mpc = fov_radius
        trans_Mpc_per_arcsec = (self.cosmo.kpc_proper_per_arcmin(self.redshift).value/1e3)/60 * (self.redshift+1)
        boxRadius_arcsec = boxRadius_Mpc / trans_Mpc_per_arcsec

        cols = '#halo_redshift, sod_halo_mass, sod_halo_radius, sod_halo_cdelta, sod_halo_cdelta_error, '\
               'halo_lc_x, halo_lc_y, halo_lc_z, boxRadius_Mpc, boxRadius_arcsec, mpp'
        props = np.array([self.redshift, self.m200c, self.r200c, self.c, self.c_err, 
                          0, 0, 0, boxRadius_Mpc, boxRadius_arcsec, self.mpp])
        
        np.savetxt('{}/properties.csv'.format(output_dir), [props],
                   fmt='%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f', 
                   delimiter=',',header=cols)




# ======================================================================================================


# example usage
#hh = simple_halo(m200c = 1e14, z = 0.3)
#hh.populate_halo(N = 10000, rfrac = 6)
#hh.output_particles(vis_debug=True)