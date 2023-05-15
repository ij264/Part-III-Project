# imports
import numpy as np
import constants as ct
import matplotlib.pyplot as plt
import os
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
import multiprocessing
import itertools
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

# data from Wilson et al (2009)
# we will fit a 5th order polynomial to this function, as done in Wilson.

eq_dist = np.linspace(3, 10, 29)
v_phi = np.array([29.3, 29.8, 29.8, 30.2, 30.4, 32.1, 35.1, 38.4, 40.9, 42.8, 45.3, 47.4, 48.5, 49.1, 50., 52., 53.4, 55.3, 57., 57.2, 57.8, 58.2, 60.7, 65., 68.2, 71., 72.8, 76.7, 80.4]) * 1e3
Omega_data = v_phi/(ct.R_S * eq_dist)

def Omega(L):
    '''
    Returns the angular frequency as a function of the normalised equatorial distance. This is based off of Wilson et al (2009) [doi:10.1029/2009GL040225]

    Args:
        L (scalar): Equatorial distance, normalised to a saturnian radius.
    
    Returns: 
        Omega (float): The angular velocity evaluated at the equatorial radius.
    '''    
    eq_dist_sine = eq_dist[6:]
    Omega_data_sine = Omega_data[6:]
    # Fit a polynomial to data of order 5
    z = np.polyfit(eq_dist[:7], Omega_data[:7], 5)
    
    # Define the function to fit (a sine wave)
    def sine(x, a, b, c, d):
        return a*np.sin(b*x + c) + d

    # Fit the function to the data
    popt, pcov = curve_fit(sine, eq_dist_sine, Omega_data_sine)

    # Extract the parameters of the fitted curve
    a, b, c, d = popt

    return np.piecewise(L, [L < 4.5, L >= 4.5], [lambda L: np.poly1d(z)(L), lambda L: sine(L, a, b, c, d)])

def Omega(L):
    '''
    Returns the angular frequency as a function of the normalised equatorial distance. This is based off of Wilson et al (2009) [doi:10.1029/2009GL040225]

    Args:
        L (scalar): Equatorial distance, normalised to a saturnian radius.
    
    Returns: 
        Omega (float): The angular velocity evaluated at the equatorial radius.
    '''    
    # Equatorial distance (no units)
    eq_dist = np.linspace(3, 10, 29) 

    # Corresponding speeds (km/s)
    v_phi = np.array([29.3, 29.8, 29.8, 30.2, 30.4, 32.1, 35.1, 38.4, 40.9, 42.8, 45.3, 47.4, 48.5, 49.1, 50., 52., 53.4, 55.3, 57., 57.2, 57.8, 58.2, 60.7, 65., 68.2, 71., 72.8, 76.7, 80.4]) * 1e3

    # Converting to angular velocites (rad/s)
    Omega_data = v_phi / (eq_dist * ct.R_S)

    # Fit a polynomial to data of order 5
    z = np.polyfit(eq_dist, Omega_data, 5)

    # Convert to function and evaluate it at L
    return(np.poly1d(z)(L))

# def Omega(L):
#     return np.full_like(L, 0.00013)

def electric_field(t, phi):
    '''
    Returns the radial and azimuthal components of the electric field model.

    Args:
        t (np.array of floats): np array of time values. This is useful for time-dependent electric fields.
        L (np.array of floats): np array of radial distances, normalised to a saturnian radius. This is useful for an electric field that varies with radius.
        phi (np.array of floats): np array of azimuthal angle. This is useful for an electric field that varies with azimuthal angle.
    
    Returns: 
        E_r (np.array of floats): np array of the radial component electric field evaluated at every time value.
        E_r (np.array of floats): np array of the azimuthal component electric field evaluated at every time value.
    '''
    # constants

    # electric field amplitude (V)
    E_0 = 0.3e-3

    # e-folding time (hours)
    tau = 150 * 3600

    # scaling factor
    L_0 = 5

    # unitless index controlling the electric field's spacial variation
    gamma_bar = 0.5

    # radial component of electric field
    E_r = -E_0 * np.sin(phi) * np.exp(-(t%tau)/tau)

    # azimuthal component of electric field
    E_phi = -E_0 * np.cos(phi) * np.exp(-(t%tau)/tau)

    return(E_r, E_phi)


def pol2cart(pos):
    '''
    Converts polar coordinates to cartesian. 

    Args:
        pos (np.array of floats, dim=(2,N)): pos[0, :] is a list of the L values, and pos[1, :] is a list of the phi values.
    
    Returns: 
        np.array of floats, dim=(2,N): pos[0, :] is a list of the x values, and pos[1, :] is a list of the y values.
    '''
    return np.array([pos[0, :] * np.cos(pos[1, :]), pos[0, :] * np.sin(pos[1, :])])

def energy(L, mu):
    '''
    Returns the energy of the electron using the formular E = gamma * m * c**2

    Args:
        L (np.array of floats): This is an np.array of L coordinates.
        mu (float): The first adiabatic index.

    Returns:
        E (np.array): An np.array of energy values evaluated at each L coordinate. 
    '''

    # Speed squared
    v_squared = 2 * ct.B_S * mu / (ct.m * L**3  + 2 * ct.B_S * mu / ct.c**2)

    # Relativistic factor 
    gamma = 1 / np.sqrt(1 - v_squared / ct.c**2)
    
    # Calculate energy
    E = (gamma -1) * ct.m * ct.c**2

    return(E/ct.MeV)

# finds where a wanted point is in array
def find_idx(arr, wanted_arr):
    ''''
    inputs:
    arr: array that is being searched through
    wanted_arr: the list of values we are interested in
    
    returns:
    np.array of indices
    '''
    # sort arr numerically
    arrsorted = np.argsort(arr)
    # find indices of wanted_arr in the new sorted array
    ypos = np.searchsorted(arr[arrsorted], wanted_arr)
    # return array corresponding to the indices
    idx = arrsorted[ypos]
    return(idx)

def plot_vector(x, y, dx, dy, x_0=np.array([]), scale=1, color='b', norm=False):
    if x_0.size != 0:
        # evaluate x at x_0
        idx = find_idx(x, x_0)
        x = x[idx]
        y = y[idx]
        dx = dx[idx]
        dy = dy[idx]
        if norm == True:
            norm = np.hypot(dy, dx)
            dx /= norm
            dy /= norm
    plt.quiver(x, y, dx, dy, angles='xy', scale_units='xy', scale=scale, color=color, width=0.005)


def f(t, y, mu):
    '''
    Returns a state vector of the time derivatives of L and phi. This function is needed for solve_ivp.

    Args:
        t (np.array of floats): np.array of time values
        y (np.array of floats, dim=(2,N)): np.array of L (y[0]) and phi (y[1]) values. 

    Returns: 
        List containg the time derivatives of L and phi respectively.
    '''
    L = y[0]
    phi = y[1]

    # Speed squared (m^2/s^2)
    v_squared = 2 * ct.B_S * mu / (ct.m * L**3  + 2 * ct.B_S * mu / ct.c**2)

    # Beta = v/c (m/s)
    beta = np.sqrt(v_squared)/ct.c

    # Relativistic factor 
    gamma = 1/np.sqrt(1-beta**2)
    
    # Use appropriate electric field
    E_r, E_phi = electric_field(t, phi)

    # Time derivative of L, taken from Hao et al (2020)
    L_dot = E_phi/(ct.R_S * ct.B_S) * L**3

    # Time derivative of phi, taken from Hao et al (2020)
    phi_dot = Omega(L) - 3 * gamma * ct.m * beta**2 * ct.c**2 / (2 * ct.e * ct.B_S * ct.R_S**2) * L - E_r / (ct.B_S * ct.R_S) * L**2

    return np.array([L_dot, phi_dot])

def solve_for_L_E_k(L_0, E_k_0, phi_0=0, N=1, t_f=20, met='DOP853', rtol=1e-8):
    '''
    Calculates the orbit for a given starting L and kinetic energy and saves the orbit to the appropriate folder. 

    Args:
        L_0 (float): This is the starting L value. 
        E_k_0 (float): This is the starting kinetic energy.
        N (int): The number of integration points per second
        t_f (float): The final time to integrate up to in hours.
        met (str): The integration method used in solve_ivp
    '''
    
    t_0 = 1e-100
    t_f = int(t_f * 3600)
    N = int(N * np.abs(t_f))
    t = np.linspace(t_0, t_f, N)

    # initial relativistic momentum
    p_0 = np.sqrt(E_k_0**2 + 2 * E_k_0 * ct.m * ct.c**2)/ct.c
    # first adiabatic invariant
    mu = p_0**2/(2 * ct.m * ct.B_S) * L_0**3

    y_0 = np.array([L_0, phi_0])

    # define function with constants pre-calculated
    f_ = lambda t, y: f(t, y, mu)

    # Solve for orbit
    sol = solve_ivp(f_, [t_0, t_f], y_0, method=met, t_eval=t, rtol=rtol)
    if sol.success:
        L, _ = sol.y

        fig = plt.figure()
        ax = fig.add_subplot()
        theta = np.linspace(0, 2 * np.pi, 100)

        circle1 = plt.Circle((0, 0), 1, color='black', fill=False)
        ax.add_patch(circle1)
        max_L = np.max(L)
        if max_L > 1: 
            for r in range(2, int(max_L) + 2):
                plt.plot(r*np.cos(theta), r*np.sin(theta), alpha=0, c='black')

        plt.xlabel('$x/R_S$')
        plt.ylabel('$y/R_S$')
        ax.set_aspect('equal', adjustable='box')

        # Convert to cartesian coordinates
        L, phi = sol.y
        x, y = pol2cart(sol.y)

        # Output a fraction of the original solution
        step = int(len(x)/10000)
        x_short = x[::step]
        y_short = y[::step]
        t_short = t[::step]
        phi_short = phi[::step]
        color_idx = np.linspace(0, 1, len(x_short))
        path = '/Users/issac/Library/CloudStorage/OneDrive-UniversityofCambridge/GitHub/III-project/figures/E_damped_spikes_24hrs'
        #path = 'C:/Users/ij264/OneDrive - University of Cambridge/GitHub/III-project/figures/E_damped_spikes_24hrs'
        path = path + f'{phi_0/np.pi}/L_0 = {L_0}/'
        
        color_idx = np.linspace(0, 1, len(x))
        plt.scatter(x, y, c=color_idx, cmap=plt.cm.RdYlBu_r, s=1)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + f'orbit, E_k_0 = {E_k_0/(ct.MeV):.2f} MeV.png', dpi=1000)
        
    else: return(sol.success)

def orbits(L_min, L_max, L_step_size=0.25, E_k_min = 0.1, E_k_max=1.1, E_k_step_size=0.1, phi_0_min=0, phi_0_max=0, phi_step=np.pi/8, batch_size=10, N=1, t_f=20, method='DOP853', rtol=1e-6):

    # set up parallel processing
    num_cores = multiprocessing.cpu_count()

    # divide the range of L_0 and E_k_0 into batches of size batch_size
    L_batches = np.array_split(np.arange(L_min, L_max + L_step_size, L_step_size), np.ceil((L_max - L_min + L_step_size) / batch_size))
    E_k_batches = np.array_split(np.arange(E_k_min, E_k_max + E_k_step_size, E_k_step_size) * ct.MeV, np.ceil((E_k_max - E_k_min + E_k_step_size) * ct.MeV / batch_size))
    phi_batches = np.array_split(np.arange(phi_0_min, phi_0_max + phi_step, phi_step), np.ceil((phi_0_max - phi_0_min + phi_step) / batch_size))
    # Give each core an orbit in batches
    for L_batch, E_k_batch, phi_batch in itertools.product(L_batches, E_k_batches, phi_batches):
        print(L_batch, E_k_batch/ct.MeV, phi_batch)
        Parallel(n_jobs=num_cores)(delayed(solve_for_L_E_k)(L_0, E_k_0, phi_0, N, t_f, method, rtol) 
                                             for L_0, E_k_0, phi_0 in itertools.product(L_batch, E_k_batch, phi_batch))

def E_CDR(L):
    alpha = 2 * ct.e * ct.B_S * ct.R_S**2 / (3 * ct.m)
    v = np.sqrt((-alpha ** 2 * (Omega(L)) ** 2 / (L * ct.c) ** 2+ np.sqrt(alpha ** 4 * (Omega(L)) ** 4 / (L **4 * ct.c ** 4) + 4 * alpha ** 2 * (Omega(L)) ** 2 / L ** 2 )) / 2)
    gamma = 1/np.sqrt(1-v**2/ct.c**2)
    E_k = (gamma - 1) * ct.m * ct.c**2/ct.MeV
    return(E_k)

def speed(x, y, t):
    v_x = np.gradient(x, t)
    v_y = np.gradient(y, t)
    
    return np.sqrt(v_x**2 + v_y**2)