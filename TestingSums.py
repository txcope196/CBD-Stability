# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 13:00:54 2023

@author: txcop
"""

import numpy as np 
import matplotlib.pyplot as plt #for plotting
import scipy
import scipy.special as spec
import multiprocessing as mp
import time
from scipy.optimize import fsolve


r0 = np.linspace(1.1,3,1000)
def l0(r):
    return np.sqrt(r)
#l0 = np.sqrt(r0)
def omega(r):
    return np.sqrt(1/r**3)-1
#omega = np.sqrt(1/r0**3)-1
cphi = np.linspace(0,2*np.pi,100)
Mtot = 1.0
q = 0.5

m = np.arange(1,21, 1).astype(float)
l = np.arange(1,21 ,1).astype(float)
mu = np.arange(1,21 ,1).astype(float)

G = 1.0
a = 1.0

dlm = 1.0
def Wlm(l, m):
    return (-1)**((l + m) / 2) * np.sqrt(spec.factorial(l - m) * spec.factorial(l + m)) / (
            2**l * spec.factorial((l - m) / 2) * spec.factorial((l + m) / 2))


def Wlmu(l,mu):
    return (-1)**((l + mu) / 2) * np.sqrt(spec.factorial(l - mu) * spec.factorial(l + mu)) / (
            2**l * spec.factorial((l - mu) / 2) * spec.factorial((l + mu) / 2))

def Ml(l):
    return (q * (1 - q)**l + (-1)**l * (1 - q) * q**l) * Mtot

def phi0(r, l,m,mu):
    return -2 * (G / a) * Ml(l) * Wlm(l, m) * Wlmu(l,mu) * dlm * (r / a)**-(l + 1)

def phiprime(r,l,m,mu):
    return -2 * (-l-1) * (G/a) * Ml(l) * Wlm(l,m) * Wlmu(l,mu) * dlm * a**(l+1) * (r)**-(l+2)

def phidobprime(r,l,m,mu):
    return -2 * (l+1) * (l+2) * (G/a) * Ml(l) * Wlm(l,m) * Wlmu(l,mu) * dlm * a**(l+1) * (r)**-(l-3)

def l1(r, phi):
    phisum = np.sum(phi0(r, l,m,mu))
    return np.sum(-(phisum / omega(r)) * np.cos(m * phi))

def r1(r, phi):
    phisum = np.sum(phi0(r, l,m,mu))
    phisumprime = np.sum(phiprime(r,l,m,mu))
    return np.sum(-np.cos(m * phi) / np.abs((1 + omega(r))**2 - m**2 * omega(r)**2) *
                  (phisumprime + (2 * (1 + omega(r)) * phisum) / (omega(r) * r)))

def Phim(r, phi):
    phisum = np.sum(phi0(r, l,m,mu))
    #print(phisum)
    return np.sum(m**2 * phisum * np.cos(m * phi))

def Phimprime(r, phi):
    phisum = np.sum(phiprime(r, l,m,mu))
    return np.sum(m * phisum * np.sin(m  * phi))

def Phimdoubprime(r, phi):
    phisum = np.sum(phidobprime(r, l,m,mu))
    return np.sum(phisum * np.cos(m *phi))

def deltaK(r, phi):
    return np.array([[0, 0, 0, 0],
                     [-2*(l1(r,phi)/r**3 + (3*l0(r)*r1(r,phi))/r**4), 0, 0, -2*r1(r,phi)/r**3],
                     [6*(r1(r,phi)/r-(l0(r)*l1(r,phi))/r**4) - Phimdoubprime(r,phi), Phimprime(r,phi), 0, 2*(l1(r,phi)/r**3-(3*l0(r)*r1(r,phi))/r**4)],
                     [Phimprime(r,phi), Phim(r,phi), 0, 0]])

def compute_eigenvalue0(r):
    eigarray = np.array([])
    for t in cphi:
        eigi = max(scipy.linalg.eigvals(K0(r) + deltaK(r, t)).real)
        eigarray = np.append(eigarray, eigi)
        maxfind = max(eigarray)
    return maxfind

def compute_eigenvalue1(r):
    eigarray = np.array([])
    eigvals_func = np.vectorize(lambda t: max(scipy.linalg.eigvals(deltaK(r, t)).real))
    eigarray = eigvals_func(cphi)
    maxfind = np.max(eigarray)
    return maxfind

def compute_eigenvalue2(r):
    eigvals = np.max(np.real(np.linalg.eigvals(np.stack([deltaK(r, t) for t in cphi]))), axis=1)
    maxfind = np.max(eigvals)
    return maxfind

def K0(r):
    return np.array([[0, 0, 1, 0],
                     [-(2*l0(r)/r**3), 0, 0, 1/r**2],
                     [-1/r**3, 0, 0, 2*l0(r)/r**3],
                     [0, 0, 0, 0]])

lindblad = np.arange(1,7,1).astype(float)
def resonance_equation(r, m):
    return (1 + omega(r))**2 - m**2 * omega(r)**2

initial_guess = 1.0
r0list = r0.tolist()
if __name__ == '__main__':
    start = time.time()
    with mp.Pool() as pool:
        results = pool.map(compute_eigenvalue2,r0list)
        #print(results)
        #print(1/np.array(results),r0)
    end = time.time()
    #res = []
    #for mtest in lindblad:
        #resonance = fsolve(resonance_equation, initial_guess, args=(mtest,))
        #res.append(resonance)
    print("Time to complete:")
    print(end -start)
    fig, ax = plt.subplots()
    #ax.scatter(r0,1/np.array(results),s = 2,marker='o')
    ax.plot(r0,1/np.array(results))
    #for re in res:
        #ax.axvline(re)
    ax.set_yscale("log")
    ax.set_title("Tau vs Radius")
    ax.set_xlabel('r/a')
    ax.set_ylabel('tau')