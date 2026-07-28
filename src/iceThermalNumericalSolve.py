import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spr
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
import warnings

class ice_thermal_numeric:
    """
    Author: Donglai Yang
    Email: dyang379@gatech.edu

    This class implement a 1D numerical solver for time dependent heat transfer (advection-diffusion with source)
    within an ice column
    We use operator splitting method to solve the advection diffusion equation
    - Advection: upwind scheme
    - Diffusion: Crank-Nicolson scheme

    """
    def __init__(self, T_init, final_year, intSave = 100, nz = 1e3, dt = None, L = 40e3, splitting='AD', Tpmp=273.15):
        # initialize constants
        self.secinyear = 3600*24*365
        self.rhoi = 920
        self.g = 9.81
        self.Cp_i = 2009 # heat capacity of ice (J/kg/K)
        self.k_i = 2.1 # thermal conductivity of ice (W/m/K)

        # compute mixing model
        self.alpha = self.k_i/(self.rhoi*self.Cp_i) # thermal diffusivity

        self.intSave=intSave
        self.w_flag = False # unknown vertical velocity when initialized
        self.solved = False
        self.debug  = False
        self.Q_flag = False
        self.splitting = splitting

        # ---- pressure melting point ----
        self.Tpmp = Tpmp
        self.basal_melting = False  # flag: has basal T reached PMP?

        # ----- parameters ------
        self.L = L # domain depth (should be semi-inifinite in theory, but just a deep domain numerically)
        self.nz = int(nz+1)
        self.z = np.linspace(0,self.L,num=self.nz)
        self.dz = self.z[1]-self.z[0]

        # time stepping and non-dimensional coefficient
        if dt is None: # determine automatically
            self.C_D = 1 # alpha*dt/(2*dz^2)
            self.C_Dd2 = 0.5 # 0.5*alpha*dt/(2*dz^2) or 0.5*C_D, for Strang splitting half step
            self.dt = 2*(self.dz**2)*self.C_D/self.alpha # computed using full step C_D
            self.dt_hs = 2*(self.dz**2)*self.C_Dd2/self.alpha # half step dt
        else: # user supplied dt    
            self.dt = dt
            self.dt_hs = dt/2 # half step dt
            self.C_D   = self.alpha*self.dt/(2*(self.dz**2))
            self.C_Dd2 = 0.5*self.alpha*self.dt/(2*(self.dz**2)) 

        self.tf = self.secinyear*final_year
        self.nt = int(np.round(self.tf/self.dt) + 1)
        self.t = np.linspace(0,self.tf,num=self.nt)
        self.T_init = T_init # initial temperature

        # intialize internal source
        self.Q = np.zeros(self.nz)

        # pre-allocate Temperature over time
        self.T = T_init*np.ones(self.nz)
        self.T_all = []
        self.t_all = []
        self.T_all.append(self.T.copy())
        self.t_all.append(0)

        # ------ initialize crank nicolson matrices
        if self.splitting == 'ADA':
            self._crank_nicolson_matrix_fullstep()
        elif self.splitting == 'DAD':
            self._crank_nicolson_matrix_halfstep()
        elif self.splitting == 'DA':
            self._crank_nicolson_matrix_fullstep()
        elif self.splitting == 'AD':
            self._crank_nicolson_matrix_fullstep()

        # also pre-build the Dirichlet-Dirichlet matrices for basal melting
        self._crank_nicolson_matrix_fullstep_dd()
        if self.splitting == 'DAD':
            self._crank_nicolson_matrix_halfstep_dd()

   
    def model_restart(self, T=None, z=None, dt=None, final_year=None, show_plot=False, intSave=None, splitting=None, Tpmp=None):
        """
        Restart the model after steady state solve

        Optional: change grid resolution, time step, and final time
        """
        if intSave is not None:
            self.intSave = intSave

        if Tpmp is not None:
            self.Tpmp = Tpmp

        # a user provided temperature profile is given
        if T is not None and z is not None:
            T_init = T
            z_init = z
        else:
            # using the previous self.T
            T_init = self.T
            z_init = self.z

        # if new dt is provided
        if dt is not None:
            self.dt = dt
            self.dt_hs = dt/2 # half step dt
            # recompute C_D and C_Dd2
            self.C_D   = self.alpha*self.dt/(2*(self.dz**2))
            self.C_Dd2 = self.alpha*self.dt_hs/(2*(self.dz**2))
        else:
            self.C_D = 1 # alpha*dt/(2*dz^2)
            self.C_Dd2 = 0.5 # 0.5*alpha*dt/(2*dz^2) or 0.5*C_D, for Strang splitting half step
            self.dt = 2*(self.dz**2)*self.C_D/self.alpha # computed using full step C_D
            self.dt_hs = 2*(self.dz**2)*self.C_Dd2/self.alpha # half step dt

        # if there is internal source
        if hasattr(self, 'Q_flag') and self.Q_flag:
            # interpolate Q onto new grid
            self.Q = np.interp(self.z, z_init, self.Q)

        # new final time is provided
        if final_year is not None:
            self.tf = self.secinyear*final_year
        self.nt = int(np.round(self.tf/self.dt) + 1)
        self.t = np.linspace(0,self.tf,num=self.nt)

        # re-compute crank-nicolson matrices
        if splitting is not None: # if none, we are not rewriting crank nicolson matrices
            self.splitting = splitting
        if self.splitting == 'ADA': # advection half step, diffusion full step
            self._crank_nicolson_matrix_fullstep()
        elif self.splitting == 'DAD':
            self._crank_nicolson_matrix_halfstep()
        elif self.splitting == 'DA':
            self._crank_nicolson_matrix_fullstep()
        elif self.splitting == 'AD':
            self._crank_nicolson_matrix_fullstep()

        # also rebuild DD matrices
        self._crank_nicolson_matrix_fullstep_dd()
        if self.splitting == 'DAD':
            self._crank_nicolson_matrix_halfstep_dd()

        # check if basal temperature is already at PMP
        if self.T[-1] >= self.Tpmp:
            self.basal_melting = True
            self.T[-1] = self.Tpmp
            print(f"Basal temperature already at PMP ({self.Tpmp} K). Using Dirichlet BC at base.")
        else:
            self.basal_melting = False

        # re-initialize arrays
        self.solved = False
        self.T_all = []
        self.t_all = []
        self.T_all.append(self.T.copy())
        self.t_all.append(0)

        print("Model restarted from the previous solution. \n")
        self._print_model_info()


        if show_plot:
            plt.figure(figsize=(6,8))
            plt.plot(self.T, self.z/1e3, label=f'Restarted Profile')
            plt.gca().invert_yaxis()
            plt.xlabel('Temperature (K)')
            plt.ylabel('Depth (km)')
            plt.title('Restarted Temperature Profile')
            plt.legend()
            plt.grid()
            plt.show()

        return
    
    
    def apply_surface_bc(self, Tbc_sfc, t=None):
        """
        Apply time-dependent boundary condition at the surface
        Tbc_sfc: array of surface temperature boundary condition (K)
        t: array of time points corresponding to Tbc_sfc (s)
        """
        # if no t is given, assume cconstant
        if t is None:
            t = np.array([0, self.tf])
            Tbc_sfc = np.array([Tbc_sfc, Tbc_sfc])
            print("Constant surface T: ", Tbc_sfc[0] ," K")
        else:
            print("Time-dependent surface T applied.")
        self.Tbc_sfc = Tbc_sfc
        self.t_bc_sfc = t
        self.bc_sfc_flag = True
        return
    
    def apply_base_bc(self, Tbc_base, t=None):
        """
        Apply time-dependent boundary condition at the base
        Tbc_base: array of basal heat flux condition (W/m^2); note, not the gradient!
        t: array of time points corresponding to Tbc_base (s)
        """
        if t is None:
            t = np.array([0, self.tf])
            Tbc_base = -1*np.array([Tbc_base, Tbc_base]) # negative: flux into the base
            print("Constant base flux: ", Tbc_base[0] ," W/m^2")
            self.Tbc_base = Tbc_base
            self.t_bc_base = t
        else:
            # check that the time dimension of Tbc_base 
            # interpolate to self.t
            Tbc_base = np.interp(self.t, t, Tbc_base)
            self.Tbc_base = -1 * Tbc_base
            self.t_bc_base = self.t

        self.bc_base_flag = True
        return
    
    def load_advection(self, z, w, show_plot=False):
        self.w = np.interp(self.z, z, w)
        self.w_flag = True

        self.CFL = np.max(np.abs(self.w)) * self.dt / self.dz
        self.Pe  = np.max(np.abs(self.w)) * self.dz / (2 * self.alpha)

        print(f'Maximum element Peclet number: {self.Pe:.3f}')
        print(f'Maximum CFL number: {self.CFL:.3f}')
        print(f'Maximum vertical velocity: {np.max(self.w)*self.secinyear:.3f} m/a')
        print(f'Minimum vertical velocity: {np.min(self.w)*self.secinyear:.3f} m/a')

        if np.max(np.abs(self.w)) != 0:
            dt_cfl = self.dz / np.max(np.abs(self.w)) * 0.5
            if dt_cfl < self.dt:
                self.dt    = dt_cfl
                self.dt_hs = self.dt / 2
                self.C_D   = self.alpha * self.dt    / (2 * self.dz**2)
                self.C_Dd2 = self.alpha * self.dt_hs / (2 * self.dz**2)
                print(f'dt reduced to {self.dt:.4f} s to satisfy CFL. Rebuilding CN matrices...')
                if self.splitting == 'ADA':
                    self._crank_nicolson_matrix_fullstep()
                elif self.splitting == 'DAD':
                    self._crank_nicolson_matrix_halfstep()
                elif self.splitting in ('DA', 'AD'):
                    self._crank_nicolson_matrix_fullstep()
                self._crank_nicolson_matrix_fullstep_dd()
                if self.splitting == 'DAD':
                    self._crank_nicolson_matrix_halfstep_dd()
                self.nt = int(np.round(self.tf / self.dt) + 1)
                self.t  = np.linspace(0, self.tf, num=self.nt)

        if show_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(self.w * self.secinyear, self.z / 1e3)
            plt.gca().invert_yaxis()
            plt.xlabel('Vertical Velocity (m/a)')
            plt.ylabel('Depth (km)')
            plt.grid()
            plt.show()

    def solve_advection(self, t, halfstep=False):
        """
        Solve the advection step using upwind scheme
        note:
            domain top is [0], bottom is [-1]
            w > 0 means flow upward
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try: 
                if self.w_flag == False:
                    raise ValueError("Exfiltration velocity w not set. Please apply_advection() first.")
                T_new = np.copy(self.T)

                if self.debug:
                    print("Top 5 T_new before advection:", T_new[:5])
                
                # Interior points
                for iz in range(1, self.nz-1):
                    if self.w[iz] >= 0:  # Upward flow, use point below (iz+1)
                        advect_flux = self.w[iz] * (self.T[iz] - self.T[iz+1]) / self.dz
                    else:  # Downward flow, use point above (iz-1)
                        advect_flux = self.w[iz] * (self.T[iz-1] - self.T[iz]) / self.dz

                    # add flux and heat source
                    T_new[iz] -= advect_flux * self.dt
                    T_new[iz] += self.Q_advection_frac * self.Q[iz] * self.dt / (self.rhoi * self.Cp_i) if self.Q_flag else 0.0

                # Base (iz=self.nz-1)
                iz_base = self.nz - 1

                # If basal melting, clamp base to Tpmp and skip advection update there
                if self.basal_melting:
                    T_new[iz_base] = self.Tpmp
                else:
                    # Original Neumann-style base advection logic
                    if self.w[iz_base] >= 0:  # Upward flow at base
                        pass
                    else:  # Downward flow at base, use point above (iz-1)
                        advect_flux = self.w[iz_base] * (self.T[iz_base-1] - self.T[iz_base]) / self.dz
                        T_new[iz_base] -= advect_flux * self.dt
                        T_new[iz_base] += self.Q_advection_frac * self.Q[iz_base] * self.dt / (self.rhoi * self.Cp_i) if self.Q_flag else 0.0

                if self.debug:
                    print("Top 5 T_new after advection:", T_new[:5])
                
                self.T = T_new

                if self.debug:
                    raise ValueError("Debug mode: exiting after advection step.")
                return
            except RuntimeWarning as e:
                print("Runtime warning during advection step: ", e)
                self._print_model_info()
                raise e


    def solve_diffusion(self, t, halfstep=True):
        """
        Solve the diffusion step with Crank-Nicolson scheme
        Internal source is added here
        """
        self.b.fill(0) # reset b vector
        # update boundary conditions if applicable
        if hasattr(self, 'bc_sfc_flag') and self.bc_sfc_flag:
            T_sfc = np.interp(t, self.t_bc_sfc, self.Tbc_sfc)
            self.T[0] = T_sfc
        else:
            raise ValueError("Surface boundary condition not set. Please apply_surface_bc() first.")

        # Choose matrices based on basal_melting flag
        if self.basal_melting:
            # Dirichlet at base: clamp T[-1] = Tpmp
            self.T[-1] = self.Tpmp
            iMl = self._get_iMl_dd(halfstep)
            Mr  = self._get_Mr_dd(halfstep)
        else:
            # Neumann at base: apply flux BC
            iMl = self.iMl
            Mr  = self.Mr
            if hasattr(self, 'bc_base_flag') and self.bc_base_flag:
                flux_base = np.interp(t, self.t_bc_base, self.Tbc_base)
                if halfstep:
                    self.b[-1] = -flux_base*self.dz*self.C_Dd2/self.k_i
                else:
                    self.b[-1] = -flux_base*self.dz*self.C_D/self.k_i
            else:
                raise ValueError("Base boundary condition not set. Please apply_base_bc() first.")
        
        # add heat source
        if hasattr(self, 'Q_flag') and self.Q_flag:
            if halfstep:
                self.b += self.Q_diffusion_frac * self.Q * self.dt_hs / (self.rhoi * self.Cp_i)
            else: # original dt is computed with half step, so here we multiply by 2
                self.b += self.Q_diffusion_frac * self.Q * self.dt / (self.rhoi * self.Cp_i)

        # For Dirichlet base, zero out b[-1] since the row is identity
        if self.basal_melting:
            self.b[-1] = 0.0

        # compute right-hand side
        A = Mr @ self.T + self.b

        # solve for new temperature profile
        self.T = iMl @ A
        
        return
    
    def load_source_term(self, z, Q, show_plot=False):
        """
        Load internal heat source term
        Q: array of internal heat source (W/m^3)
        """
        self.Q = np.interp(self.z, z, Q)
        self.Q_flag = True
        # for diffusion step, the fraction of heat source needs to integrated
        # ----- halfstep specific -----------
        self.Q_diffusion_frac = 1 
        self.Q_advection_frac = 0
        # -----------------------------------
        
        if show_plot:
            plt.figure(figsize=(6,4))
            plt.plot(self.Q, self.z/1e3)
            plt.gca().invert_yaxis()
            plt.xlabel('Internal Heat Source (W/m^3)')
            plt.ylabel('Depth (km)')
            plt.title('Internal Heat Source Profile')
            plt.grid()
            plt.show()
        return
    
    def _check_basal_melting(self):
        """
        Check if basal temperature has reached or exceeded the pressure melting point.
        If so, switch to Dirichlet BC at the base and clamp T[-1] to Tpmp.
        Returns True if the switch just happened this call.
        """
        if not self.basal_melting and self.T[-1] >= self.Tpmp:
            self.basal_melting = True
            self.T[-1] = self.Tpmp
            print(f"  *** Basal temperature reached PMP ({self.Tpmp} K). Switching to Dirichlet BC at base. ***")
            return True
        return False

    def solve_steady_state(self, tol=1e-8, max_iter=1e5, show_plot=False, debug=False):
        """
        Transient solve until steady state
        """
        self._print_model_info()
        self.debug = debug

        halfstep = True
        err = np.inf
        iter_count = 0

        self.t_ss = []
        self.T_ss = []
        # initial vals
        self.T_ss.append(self.T.copy())
        self.t_ss.append(0.0)

        while err > tol and iter_count < max_iter:
            iter_count += 1
            T_old = self.T.copy()

            if self.splitting == 'ADA':
                self.solve_advection(0, halfstep)
                self._check_basal_melting()
                self.solve_diffusion(0, halfstep=False) 
                self._check_basal_melting()
                self.solve_advection(0, halfstep)
                self._check_basal_melting()
            if self.splitting == 'DAD':
                self.solve_diffusion(0, halfstep)
                self._check_basal_melting()
                self.solve_advection(0, halfstep=False)
                self._check_basal_melting()
                self.solve_diffusion(0, halfstep)
                self._check_basal_melting()
            if self.splitting == 'DA':
                self.solve_diffusion(0, halfstep=False) 
                self._check_basal_melting()
                self.solve_advection(0, halfstep=False)
                self._check_basal_melting()
            if self.splitting == 'AD':
                self.solve_advection(0, halfstep=False)
                self._check_basal_melting()
                self.solve_diffusion(0, halfstep=False)
                self._check_basal_melting()

            # compute change in temperature
            err = np.max(np.abs(self.T - T_old)) / np.max(np.abs(T_old))

            # save results at specified intervals
            if iter_count % self.intSave == 0:
                idx_save = iter_count // self.intSave
                print("idx_save:", idx_save)
                self.T_ss.append(self.T.copy())
                self.t_ss.append(iter_count*self.dt)
                print(f'Steady state iteration {iter_count}, relative error: {err:.2e}')


            if debug:
                return
        
        # add the last one
        self.T_ss.append(self.T.copy())
        self.t_ss.append(iter_count*self.dt)

        self.solved = True

            
        if show_plot:
            # only show the last temperature profile
            plt.figure(figsize=(6,8))
            plt.plot(self.T_ss[-1], self.z/1e3, label=f'Steady State Profile')
            plt.gca().invert_yaxis()
            plt.xlabel('Temperature (K)')
            plt.ylabel('Depth (km)')
            plt.title('Steady State Temperature Profile at time = {:.2f} years'.format(self.t_ss[-1]/self.secinyear))
            plt.legend()
            plt.grid()
            plt.show()

        return

    def solve_transient(self, show_plot=False, debug=False):
        """
        Solve the advection-diffusion equation with Strang splitting

        Here we consider half stepping for diffusion, and full stepping for advection
        then another half stepping for diffusion
        """
        self._print_model_info()
        self.debug = debug

        halfstep = True
        for it in range(1, self.nt):
            # passing time to acquire time-dependent velocity or boundary conditions
            if self.splitting == 'ADA':
                self.solve_advection(self.t[it], halfstep)
                self._check_basal_melting()
                self.solve_diffusion(self.t[it], halfstep=False) 
                self._check_basal_melting()
                self.solve_advection(self.t[it], halfstep)
                self._check_basal_melting()
            if self.splitting == 'DAD':
                self.solve_diffusion(self.t[it], halfstep)
                self._check_basal_melting()
                self.solve_advection(self.t[it], halfstep=False)
                self._check_basal_melting()
                self.solve_diffusion(self.t[it], halfstep)
                self._check_basal_melting()
            if self.splitting == 'DA':
                self.solve_diffusion(self.t[it], halfstep=False) 
                self._check_basal_melting()
                self.solve_advection(self.t[it], halfstep=False)
                self._check_basal_melting()
            if self.splitting == 'AD':
                self.solve_advection(self.t[it], halfstep=False)
                self._check_basal_melting()
                self.solve_diffusion(self.t[it], halfstep=False)
                self._check_basal_melting()

            # save results at specified intervals
            if it % self.intSave == 0:
                idx_save = it // self.intSave
                self.T_all.append(self.T.copy())
                self.t_all.append(self.t[it])
            
            # if it is less than intSave, save the last one
            if it == self.nt - 1 and it % self.intSave != 0:
                idx_save = it // self.intSave + 1
                self.T_all.append(self.T.copy())
                self.t_all.append(self.t[it])

            if debug:
                return

            # print progress
            if it % (self.nt // 10) == 0:
                print(f'Solving progress: {it/self.nt*100:.1f}%')

        self.solved = True

        if show_plot:
            self.plot_T_change()

        return
    
    def get_flux(self, t=None, z=0):

        """
        get heat flux
        optional: depth and timeslices
        """
        if not self.solved:
            raise ValueError("Model not yet solved. Please run solve_transient() first.")
        
        if t is None:
            # only look at the last solution 
            T_profile = self.T.copy()
            dTdz = np.gradient(T_profile, self.dz)
            q = -self.k_i * dTdz
            # interpolate to the depth z
            q_z = np.interp(z, self.z, q)
            return q_z
        else:   
            raise ValueError("Time-dependent flux extraction not yet implemented.")        

    
    def plot_results(self, time_indices=None):
        """
        Plot temperature profiles at specified time indices
        time_indices: list of time indices to plot
        """
        plt.figure(figsize=(6,8))
        if time_indices is None:
            time_indices = range(len(self.T_all))

        for idx in time_indices:
            plt.plot(self.T_all[idx], self.z/1e3, label=f'Time = {self.t_all[idx]/self.secinyear:.2f} years')
        plt.gca().invert_yaxis()
        plt.xlabel('Temperature (K)')
        plt.ylabel('Depth (km)')
        plt.title('Temperature Profiles Over Time')
        plt.legend()
        plt.grid()
        plt.show()
        return
    
    def plot_T_change(self):
        """
        Plot two subplots:
        1. Initial temperature profile
        2. Temperature profiles at later time - Initial temperature profile
        """

        plt.figure(figsize=(12,6))
        plt.subplot(1,2,1)
        plt.plot(self.T_all[0], self.z/1e3, label='Initial Profile')
        plt.gca().invert_yaxis()
        plt.xlabel('Temperature (K)')
        plt.ylabel('Depth (km)')
        plt.title('Initial Temperature Profile')
        plt.legend()
        plt.grid()
        
        plt.subplot(1,2,2)
        for idx in range(1, len(self.T_all)):
            plt.plot(self.T_all[idx]-self.T_all[0], self.z/1e3,
                     label=f'Time = {self.t_all[idx]/self.secinyear:.2f} years')
        plt.gca().invert_yaxis()
        plt.xlabel('Temperature Change (K)')
        plt.ylabel('Depth (km)')
        plt.title('Temperature Change Over Time')
        plt.legend()
        plt.grid()
        plt.show()
        return
    

    def _print_model_info(self):
        print('----- Ice Thermal Numerical Solver Info -----')
        print(f'Domain Depth L: {self.L/1e3} km')
        print(f'Number of Grid Points nz: {self.nz}')
        print(f'Grid Spacing dz: {self.dz} m')
        print(f'Time Step dt: {self.dt} s ({self.dt/self.secinyear} years)')
        print(f'Nondimensional C_D: {self.C_D}')
        print(f'Final Time tf: {self.tf} s ({self.tf/self.secinyear} years)')
        print(f'Number of Time Steps nt: {self.nt}')
        print(f'Output Saving Interval intSave: every {self.intSave} time steps')
        print(f'Pressure Melting Point Tpmp: {self.Tpmp} K')
        print(f'Basal Melting Active: {self.basal_melting}')
        print('----------------------------------------------')

        return
    
    def _crank_nicolson_matrix_fullstep(self):
        """
        Create the Crank-Nicolson matrices for 1D diffusion equation
        with Neumann boundary condition at the bottom and Dirichlet
        boundary condition at the top
        """
        # create sparse matrix components
        main_diag_Ml = (1 + 2*self.C_D)*np.ones(self.nz)
        off_diag_Ml = -self.C_D*np.ones(self.nz-1)
        main_diag_Mr = (1 - 2*self.C_D)*np.ones(self.nz)
        off_diag_Mr = self.C_D*np.ones(self.nz-1)

        main_diag_Ml[-1] = 1 + self.C_D
        off_diag_Ml[-1] = -self.C_D
        main_diag_Mr[-1] = 1 - self.C_D
        off_diag_Mr[-1] = self.C_D

        # create sparse matrices
        Ml = spr.diags([main_diag_Ml, off_diag_Ml, off_diag_Ml],
                       [0, -1, 1], format='csc')
        Mr = spr.diags([main_diag_Mr, off_diag_Mr, off_diag_Mr],
                       [0, -1, 1], format='csc')
        
        # replace first row with [1,0,0...]
        Ml = Ml.tolil()
        Mr = Mr.tolil()
        Ml[0,:] = 0
        Ml[0,0] = 1
        Mr[0,:] = 0
        Mr[0,0] = 1
        Ml = Ml.tocsc()
        self.Mr = Mr.tocsc()
        # create empty b vector (due to the Neumann BC at the bottom)
        self.b = np.zeros(self.nz)

        # compute inverse of Ml
        self.iMl = inv(Ml)

        return
    
    def _crank_nicolson_matrix_halfstep(self):
        """
        Create the Crank-Nicolson matrices for 1D diffusion equation
        with Neumann boundary condition at the bottom and Dirichlet
        boundary condition at the top

        Consider only half step for Strang splitting
        """
        # create sparse matrix components
        main_diag_Ml = (1 + 2*self.C_Dd2)*np.ones(self.nz)
        off_diag_Ml = -self.C_Dd2*np.ones(self.nz-1)
        main_diag_Mr = (1 - 2*self.C_Dd2)*np.ones(self.nz)
        off_diag_Mr = self.C_Dd2*np.ones(self.nz-1)

        main_diag_Ml[-1] = 1 + self.C_Dd2
        off_diag_Ml[-1] = -self.C_Dd2  # Standard coefficient, not doubled
        main_diag_Mr[-1] = 1 - self.C_Dd2
        off_diag_Mr[-1] = self.C_Dd2

        # create sparse matrices
        Ml = spr.diags([main_diag_Ml, off_diag_Ml, off_diag_Ml],
                       [0, -1, 1], format='csc')
        Mr = spr.diags([main_diag_Mr, off_diag_Mr, off_diag_Mr],
                       [0, -1, 1], format='csc')
                
        # replace first row with [1,0,0...]
        Ml = Ml.tolil()
        Mr = Mr.tolil()
        Ml[0,:] = 0
        Ml[0,0] = 1
        Mr[0,:] = 0
        Mr[0,0] = 1
        Ml = Ml.tocsc()
        self.Mr = Mr.tocsc()

        # create empty b vector (b is needed due to the Neumann BC at the bottom)
        self.b = np.zeros(self.nz)

        # compute inverse of Ml
        self.iMl = inv(Ml)

        return

    # =====================================================================
    #  Dirichlet-Dirichlet Crank-Nicolson matrices (for basal melting)
    # =====================================================================

    def _crank_nicolson_matrix_fullstep_dd(self):
        """
        Crank-Nicolson matrices with Dirichlet BC at BOTH top and bottom.
        Used when basal temperature is clamped at Tpmp.
        Full time-step version.
        """
        main_diag_Ml = (1 + 2*self.C_D)*np.ones(self.nz)
        off_diag_Ml  = -self.C_D*np.ones(self.nz-1)
        main_diag_Mr = (1 - 2*self.C_D)*np.ones(self.nz)
        off_diag_Mr  = self.C_D*np.ones(self.nz-1)

        Ml = spr.diags([main_diag_Ml, off_diag_Ml, off_diag_Ml],
                       [0, -1, 1], format='csc')
        Mr = spr.diags([main_diag_Mr, off_diag_Mr, off_diag_Mr],
                       [0, -1, 1], format='csc')

        Ml = Ml.tolil()
        Mr = Mr.tolil()
        # Top row: Dirichlet
        Ml[0, :] = 0;  Ml[0, 0] = 1
        Mr[0, :] = 0;  Mr[0, 0] = 1
        # Bottom row: Dirichlet
        Ml[-1, :] = 0; Ml[-1, -1] = 1
        Mr[-1, :] = 0; Mr[-1, -1] = 1

        Ml = Ml.tocsc()
        self.Mr_dd = Mr.tocsc()
        self.iMl_dd = inv(Ml)
        return

    def _crank_nicolson_matrix_halfstep_dd(self):
        """
        Crank-Nicolson matrices with Dirichlet BC at BOTH top and bottom.
        Half time-step version (for DAD Strang splitting).
        """
        main_diag_Ml = (1 + 2*self.C_Dd2)*np.ones(self.nz)
        off_diag_Ml  = -self.C_Dd2*np.ones(self.nz-1)
        main_diag_Mr = (1 - 2*self.C_Dd2)*np.ones(self.nz)
        off_diag_Mr  = self.C_Dd2*np.ones(self.nz-1)

        Ml = spr.diags([main_diag_Ml, off_diag_Ml, off_diag_Ml],
                       [0, -1, 1], format='csc')
        Mr = spr.diags([main_diag_Mr, off_diag_Mr, off_diag_Mr],
                       [0, -1, 1], format='csc')

        Ml = Ml.tolil()
        Mr = Mr.tolil()
        Ml[0, :] = 0;  Ml[0, 0] = 1
        Mr[0, :] = 0;  Mr[0, 0] = 1
        Ml[-1, :] = 0; Ml[-1, -1] = 1
        Mr[-1, :] = 0; Mr[-1, -1] = 1

        Ml = Ml.tocsc()
        self.Mr_dd_hs = Mr.tocsc()
        self.iMl_dd_hs = inv(Ml)
        return

    def _get_iMl_dd(self, halfstep):
        """Return the appropriate DD inverse matrix."""
        if halfstep and hasattr(self, 'iMl_dd_hs'):
            return self.iMl_dd_hs
        return self.iMl_dd

    def _get_Mr_dd(self, halfstep):
        """Return the appropriate DD right-hand-side matrix."""
        if halfstep and hasattr(self, 'Mr_dd_hs'):
            return self.Mr_dd_hs
        return self.Mr_dd
