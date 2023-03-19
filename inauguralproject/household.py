
from types import SimpleNamespace

import numpy as np
from scipy import optimize

import pandas as pd 
import matplotlib.pyplot as plt

from scipy.optimize import minimize, minimize_scalar

class HouseholdSpecializationModelClass:

    def __init__(self):
        """ setup model """

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # c. household production
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        # H = min virker ikke, skal der lige kigges på 
        if par.sigma==0:
            H = min(HM, HF)
        elif par.sigma==1:
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma)/par.sigma))**(par.sigma/(par.sigma-1))


        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """ solve model continously """

        pass    

    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """

        pass

    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        pass

    def __init__(self, a, w, p, v, wm, wf, sigma_m, sigma_f, beta0, beta1, L_M, W_M, H_M, epsilon):
        """ setup model """
        #Defining the float values
        self.a_values = [0.25, 0.5, 0.75] #These values represent the productivity in home production for females relative to males.
        self.sigma_values = [0.5, 1.0, 1.5] #These values represent the elasticity of substitution between the home production of males and females in the household.
        self.H_M_values = np.zeros((len(self.a_values), len(self.sigma_values), 49))
        self. W_F_values = np.array([0.8, 0.9, 1.0, 1.1, 1.2])

        #Question 5
        # Define the parameters as instance variables
        self.a = a
        self.w = w
        self.p = p
        self.v = v
        self.wm = wm
        self.wf = wf
        self.sigma_m = sigma_m
        self.sigma_f = sigma_f
        self.beta0 = beta0
        self.beta1 = beta1
        self.W_M = W_M
        self.H_M = H_M
        self.L_M = L_M
        self.T_M = L_M+H_M
        self.epsilon=epsilon
      
      
# Create an instance of the HouseholdSpecializationModel1Class
model = HouseholdSpecializationModel1Class(a=0.5, w=0.5, p=2, v=0.001, wm=1, wf=1, sigma_m=1.5, sigma_f=2, beta0=0.4, beta1=-0.1, L_M=0, W_M=1, H_M=0, epsilon=1)
