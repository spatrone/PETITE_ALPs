# Initialization of the tutorial
from platform import python_version
print("Python version: ", python_version())

import numpy as np
print("Numpy version: ", np.__version__)

import vegas
print("Vegas version: ", vegas.__version__)

import os
current_path = os.getcwd()
# Get the parent directory
parent_dir = os.path.dirname(current_path)

PETITE_home_dir= parent_dir.split('examples')[0]

print("PETITE home directory:", PETITE_home_dir)
# folder where VEGAS dictionaries are stored
# dictionary_dir = "data/VEGAS_dictionaries/"
dictionary_dir = "/data/"

from numpy.random import random
from PETITE.shower import *
import pickle as pk
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
import scipy.integrate as integrate
from scipy.stats import loguniform
import sys
import time
from scipy.optimize import fsolve

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, MaxNLocator
import cProfile
profile = cProfile.Profile()
import pstats

font0 = FontProperties()
font = font0.copy()
font.set_size(24)
font.set_family('serif')
labelfont=font0.copy()
labelfont.set_size(20)
labelfont.set_weight('bold')
legfont=font0.copy()
legfont.set_size(18)
legfont.set_weight('bold')


def set_size(w,h, ax=None):
    """ Helper function to set figure size.
        Input:
            w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
    
def checkandsave(filename,stuff):
    # Check if the file already exists
    if os.path.exists(filename):
        # If the file exists, generate a new filename
        i = 1
        while True:
            new_filename = f'{os.path.splitext(filename)[0]}_{i}{os.path.splitext(filename)[1]}'  # Append _1, _2, etc.
            if not os.path.exists(new_filename):
                break
            i += 1
    else:
        new_filename = filename

    # Open the file with the new filename for writing
    with open(new_filename, 'wb') as f:
        pickle.dump(stuff, f)
    print(f"Data saved on {new_filename}!")
    return None

def PL(a,delta):
    return (4 * np.pi**4 * a**4 * (a*delta +1) * np.log(1/(a*delta)+1))/(4 * np.pi**4 * a**4 * (a*delta +1) * np.log(1/(a*delta) + 1) + np.pi**4 * a**4-delta**4)

def PR(a,delta):
    return 1 - PL(a,delta)

def coverL(theta,a,delta):
    return a/(delta*(theta+delta**2))

def coverR(theta,a,delta):
    return delta**3/(a**3*(theta**5)*(a*delta+1))

def covering_func(theta,a,delta):
    if theta<delta/a:
        #normL=integrate.quad(lambda x: coverL(x,a,delta), 0, a/delta)[0]
        return coverL(theta,a,delta)
    else:
        #normR=integrate.quad(lambda x: coverR(x,a,delta), a/delta, np.pi)[0]
        return coverR(theta,a,delta)

def invCDFL(a, delta, u):
    return delta**2 * (((a * delta)/(a*delta + 1))**(-u) - 1)

def invCDFR(a, delta, u):
    return (np.pi * delta)/(delta**4 * u - np.pi**4 * a**4 * (u-1))**0.25

def theta_gen_cover(a, delta):
    u=random.random()
    if random.random()<PL(a,delta):
        theta=invCDFL(a, delta, u)
    else:
        theta=invCDFR(a, delta, u)
    return theta

def qsq(theta,a,delta):
    return 2*np.cos(theta) * np.sqrt(1-delta**2) + delta**2-2

def dsigmadtheta(theta,a,delta):
    return (np.sin(theta)**3 * (1-delta**2)**1.5)/(2*qsq(theta,a,delta)*(1-qsq(theta,a,delta)*a**2/delta**2))**2

def theta_gen(a, delta):
    while True:
        theta_sample=theta_gen_cover(a, delta)
        x=random.random()*covering_func(theta_sample, a, delta)
        if x<dsigmadtheta(theta_sample,a,delta):
            break
    return theta_sample 

def theta_gen_fast(a, delta):
    theta_max=dsigmaptlikemax(delta)
    const=dsigmadtheta(theta_max,a,delta)*20/covering_func(theta_max, a, delta)
    while True:
        theta_sample=theta_gen_cover(a, delta)
        x=random.random()*covering_func(theta_sample, a, delta)
        if x<dsigmadtheta(theta_sample,a,delta)/const:
            break
    return theta_sample

def dsigmaptlikemax(delta):
    costheta= (np.sqrt(4 - 8*delta**2 + 13*delta**4 - 9*delta**6)-np.sqrt(-((-1 + delta)*(1 + delta)))*(6 - 3*delta**2))/ (2.*(-2 + 2*delta**2))
    return np.arccos(costheta)

def axion_gen_3mom(omega, ma, invR):
    delta=ma/omega
    a=ma/invR
    theta=theta_gen_fast(a, delta)
    phi=random.random()*2*np.pi
    return np.sqrt(omega**2-ma**2)*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])

def invGeV_to_cm(invGev):
    return 0.197*1e-13*invGev

def cm_to_invGeV(cm):
    return cm/(0.197*1e-13)

def invGeV_to_sec(invGev):
    return 0.197*1e-23/3*invGev

def sigma_P(a,delta):
    return -0.25*np.sqrt(1 - delta**2) - ((2 + (-1 + a**2)*delta**2)*
         (np.log(2 - delta**2 - 2*np.sqrt(1 - delta**2)) - np.log(2 - delta**2 + 2*np.sqrt(1 - delta**2)) + 
           np.log(delta**2 + a**2*(2 - delta**2 + 2*np.sqrt(1 - delta**2))) - 
           np.log(delta**2 - a**2*(-2 + delta**2 + 2*np.sqrt(1 - delta**2)))))/16.
    

def Nsigma_P(gayy,ndensity,a,delta,Z):
    return (1./137) * (invGeV_to_cm(gayy))**2 * Z**2 * ndensity * sigma_P(a,delta)

def GammaDecay(ma,gayy):
    #[ma]=GeV, [gayy]=GeV**-1
    return gayy**2*ma**3/(64*np.pi)

def decay_weight(w_LLP_gone, gayy, l0, Lpipe):
    w_LLP=w_LLP_gone*gayy**2
    w_tilde=w_LLP*l0/Lpipe
    return np.exp(-w_tilde)*(1-np.exp(-w_LLP))

def params_dict(shower,ma,gayy=1,Lpipe=50,l0=45, A_detector=50):
    #[ma]=GeV, [gayy]=GeV**-1, [Lpipe]=[l0]=m, [A_detector]=m**2
    (Z,A,rho,_)=shower.get_material_properties()
    ndensity=rho/A*6.022*1e23 #cm^-3
    d=(0.197/(1.2*A**0.333)) #GeV
    angle_accept=np.sqrt(A_detector)/(2*(Lpipe+l0))
    params={'Z':Z,'A':A,'rho':rho,'ndensity':ndensity,'d':d,'ma':ma,'gayy':gayy,'l0':l0,'Lpipe':Lpipe,'angle_accept':angle_accept}
    return params

def photons_from_Nprimary(shower, p0, N, omega_min=0, plothisto=True):
    i = 0
    photons = []
    n = 0
    start_time = time.time()  # Start timing
    
    while i < N:
        # Generate a new shower
        particles = shower.generate_shower(p0)
        
        # Select photons (pid=22) with energy above the threshold omega_min
        shower_photons = [p for p in particles if p.get_pid() == 22 and p.get_p0()[0] > omega_min]
        
        # Add these photons to the main list
        photons += shower_photons
        
        # Update counters
        i += 1
        n = len(photons)
        
        # Calculate elapsed time and ETA
        elapsed_time = time.time() - start_time
        avg_time_per_shower = elapsed_time / i
        eta = avg_time_per_shower * (N - i)
        
        # Display progress with elapsed time and ETA
        progress = f"Generated {i}/{N} showers ({n} total photons with E > {omega_min} GeV) | "
        progress += f"Elapsed: {elapsed_time:.2f}s | ETA: {eta:.2f}s\n"
        sys.stdout.write(f'\r{progress}')
        sys.stdout.flush()
    
    # Plot a histogram of the photon energies if requested
    if plothisto:
        omega = [photon.get_p0()[0] for photon in photons]
        
        # Set log-spaced bins for the histogram
        min_energy = min(omega) if omega else 1e-3  # Use a small value if no photons
        max_energy = max(omega) if omega else 1e3   # Use a large value if no photons
        bins = np.logspace(np.log10(min_energy), np.log10(max_energy), 50)
        
        plt.figure(figsize=(8, 6))
        plt.hist(omega, bins=bins, color='blue', alpha=0.7)
        plt.xscale('log')  # Set x-axis to log scale
        plt.title(f'Photon Energy Distribution (E > {omega_min} GeV)')
        plt.xlabel('Photon Energy (GeV)')
        plt.ylabel('Counts')
        plt.show()
    
    return photons
    
def count_photons_above_ma(photons,ma):
    photons_above_ma=0
    for photon in photons:
        if  photon.get_p0()[0]>ma: photons_above_ma+=1
    return photons_above_ma
    
def axion_shower_primakhoff(shower, photons, params):
    #[ma]=[d]=GeV
    ma, d = params['ma'], params['d']
    a = ma / d
    axions = []
    cnt = 0
    photons_above_ma=count_photons_above_ma(photons,ma)
    
    # Start timing
    start_time = time.time()
    
    for i, photon in enumerate(photons):
        R = photon.rotation_matrix()
        omega = photon.get_p0()[0]
        delta = ma / omega
        
        if delta < 1:
            axion_p = R @ axion_gen_3mom(omega, ma, d)
            axions.append({'m_a': ma, 'p_a': np.array(np.hstack([omega, axion_p]))})
            cnt += 1
        
        # Calculate elapsed time and ETA
        elapsed_time = time.time() - start_time
        avg_time_per_photon = elapsed_time / (cnt + 1)
        eta = avg_time_per_photon * (photons_above_ma - (cnt + 1))
        
        if cnt%10 == 0:
            # Display progress with elapsed time and ETA
            progress = (f"Generated {cnt}/{photons_above_ma} axions of mass ma={ma} GeV | "
                        f"Elapsed: {elapsed_time:.2f}s | ETA: {eta:.2f}s")
            sys.stdout.write(f'\r{progress}')
            sys.stdout.flush()
    
    sys.stdout.write('\n')
    return axions

def axions_weights_update(shower,axions,params):
    Z,ndensity,ma, d,angle_accept,Lpipe = params['Z'],params['ndensity'],params['ma'], params['d'],params['angle_accept'],params['Lpipe']
    a = ma / d
    for i,axion in enumerate(axions):
        axion_p=axion['p_a'][-3:]
        omega=axion['p_a'][0]
        delta=ma/omega

        if np.arccos(axion_p[2]/np.sqrt(axion_p@axion_p))<angle_accept:
            angle_weight=1
        else: 
            angle_weight=0
            
        prod_weight=Nsigma_P(1,ndensity,a,delta,Z)/shower._NSigmaPhoton(omega)
            
        betagamma=np.sqrt(omega**2-ma**2)/ma
        decay_length=invGeV_to_cm(betagamma/GammaDecay(ma,1))
        decay_weight_LLP=Lpipe*100/decay_length
            
        axions[i]['w_angle']= angle_weight
        axions[i]['w_prod']= prod_weight
        axions[i]['w_LLP_gone']=decay_weight_LLP
    return axions
        
def gayy_LLP_approx(axions,N_primary,N_gamma=5*1e20,N_discovery=5):
    W_tot=0
    for axion in axions:
        W_tot+=axion['w_angle']*axion['w_prod']*axion['w_LLP_gone']
    W_tot/=N_primary
    return (W_tot*N_gamma/N_discovery)**(-0.25)

def gayy_exact(axions,N_primary,N_gamma=5*1e20,N_discovery=5,l0=45,Lpipe=50,plot=True,gayy_initial_guess=1e-8):
    R=N_discovery/N_gamma
    N_axions=len(axions)
    func = lambda gayy: R - gayy**2/N_primary*sum(axions[i]['w_angle']*axions[i]['w_prod']*decay_weight(axions[i]['w_LLP_gone'], gayy, l0, Lpipe) for i in range(N_axions))
    if plot:
        gayy = np.logspace(-8, -3, num=1000)
        plt.plot(gayy, R-func(gayy))
        plt.plot(gayy,np.full(1000,R))
        plt.xlabel("gayy")
        plt.ylabel("W_tot")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()
    sol=fsolve(func, gayy_initial_guess)
    return sol.tolist()

