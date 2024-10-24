import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from iminuit import cost, Minuit
import pandas
from iminuit.util import describe, make_func_code
from matplotlib import colors
from scipy.stats import norm
from scipy.special import expit, logit
from scipy import stats
from scipy import signal
from scipy import integrate
from ztfidr.sample import Sample
from scipy.stats import multivariate_normal
from scipy.interpolate import CubicSpline

class ColorFit:
    def __init__(self, data):
        self._data = data         
    
    @classmethod
    def from_dataset(cls, data):
        return cls(data)
    
    def get_data(self, column = False):
        if column == False:
            return self.data
        else:
            return self.data[column]
    
    def full_fit(self):
        if self.dust_only == True:
            print("Intrisic color distribution is blocked: either define one or change the value of 'dust_only' to False")
        m_col = Minuit(get_loglikelihood_tot_err(self.data['c'], self.data['c_err']), a=0.12, mu=-0.06, sig=0.02)
        m_col.errordef = Minuit.LIKELIHOOD
        m_col.migrad()
        self.set_tau_dust(m_col.values[0])
        self.set_tau_dust_err(m_col.errors[0])
        self.set_mu_int(m_col.values[1])
        self.set_mu_int_err(m_col.errors[1])
        self.set_sig_int(m_col.values[2])
        self.set_sig_int_err(m_col.errors[2])
        self.set_minuit(m_col)
    
    def fit_dust(self):
        m_dust = Minuit(get_loglikelihood_dustcut_err(self.data['c'], self.data['c_err'], self.mu_int, self.sig_int), a=0.5)
        m_dust.errordef = Minuit.LIKELIHOOD
        m_dust.migrad()
        self.set_tau_dust(m_dust.values[0])
        self.set_tau_dust_err(m_dust.errors[0])
        self.set_minuit(m_dust)
        
    def plot_int_fit(self, ax, color='tab:blue', linestyle='dotted', title='', data_color='lightblue', axis=[0], alpha=1, data=True):
        if len(axis)==1:
            minmax = max(abs(min(self.data['c'])), abs(max(self.data['c'])))
            xc = np.linspace(-minmax*1.05, minmax*1.05, 200)
        else:
            xc = axis
        ax.plot(xc, fit_function_sne(xc, self.mu_int, self.sig_int), 
             color=color, linestyle=linestyle, label=title, alpha=alpha)
        if data==True:
            alldata.plot_data(ax, xc, self.data['c'], self.data['c_err'], colour=data_color, title='')
        
    def plot_dust_fit(self, ax, color='tab:blue', linestyle='dotted', title='', data_color='lightblue', axis=[0], alpha=1, data=True):
        if len(axis)==1:
            minmax = max(abs(min(self.data['c'])), abs(max(self.data['c'])))
            xc = np.linspace(-minmax*1.05, minmax*1.05, 200)
        else:
            xc = axis        
        ax.plot(xc, fit_function_dust(xc, self.tau_dust), 
             color=color, linestyle=linestyle, label=title, alpha=alpha)
        if data==True:
            alldata.plot_data(ax, xc, self.data['c'], self.data['c_err'], colour=data_color, title='')
        
    def plot_full_fit(self, ax, color=['tab:blue', 'r'], title='', data_color='lightblue', axis=[0], data=True, bootstrap=False):
        if len(axis)==1:
            minmax = max(abs(min(self.data['c'])), abs(max(self.data['c'])))
            xc = np.linspace(-minmax*1.05, minmax*1.05, 200)
        else:
            xc = axis  
        y = fit_function_tot(xc, self.tau_dust, self.mu_int, self.sig_int)
        ax.plot(xc, y, color=color[0], label=title, zorder=2, lw=1)
        if bootstrap==True:
            rng = np.random.default_rng(1)
            par_b = rng.multivariate_normal(self.minuit.values, self.minuit.covariance, size=1000)
            y_b = [fit_function_tot(xc, p[0], p[1], p[2]) for p in par_b]
            yerr_boot = np.std(y_b, axis=0)
            ax.fill_between(xc, y - yerr_boot, y + yerr_boot, facecolor=color[1], alpha=0.5, zorder=1)
        if data==True:
            alldata.plot_data(ax, xc, self.data['c'], self.data['c_err'], colour=data_color, title='Data')
        
    @property
    def tau_dust(self):
        if not hasattr(self, "_tau_dust"):
            if self.dust_only == True:
                self.fit_dust()
            else:
                self.full_fit()
        return self._tau_dust
    
    def set_tau_dust(self, tau_dust):
        self._tau_dust = tau_dust
        
    @property
    def tau_dust_err(self):
        if not hasattr(self, "_tau_dust_err"):
            if self.dust_only == True:
                self.fit_dust()
            else:
                self.full_fit()
        return self._tau_dust_err
    
    def set_tau_dust_err(self, tau_dust_err):
        self._tau_dust_err = tau_dust_err
        
    @property
    def mu_int(self):
        if self.dust_only == False:
            if not hasattr(self, "_mu_int"):
                self.full_fit()
            return self._mu_int
        else:
            if not hasattr(self, "_mu_int"):
                print("Intrisic color distribution is blocked: either define one or change the value of 'dust_only' to False")
            else:
                return self._mu_int
    
    def set_mu_int(self, mu_int):
        self._mu_int = mu_int
       
    @property
    def mu_int_err(self):
        if self.dust_only == False:
            if not hasattr(self, "_mu_int_err"):
                self.full_fit()
            return self._mu_int_err
        else:
            if not hasattr(self, "_mu_int_err"):
                print("Intrisic color distribution is blocked: either define one or change the value of 'dust_only' to False")
            else:
                return self._mu_int_err
    
    def set_mu_int_err(self, mu_int_err):
        self._mu_int_err = mu_int_err
        
        
    @property
    def sig_int(self):
        if self.dust_only == False:
            if not hasattr(self, "_sig_int"):
                self.full_fit()
            return self._sig_int
        else:
            if not hasattr(self, "_sig_int"):
                print("Intrisic color distribution is blocked: either define one or change the value of 'dust_only' to False")
            else:
                return self._sig_int
    
    def set_sig_int(self, sig_int):
        self._sig_int = sig_int
        
    @property
    def sig_int_err(self):
        if self.dust_only == False:
            if not hasattr(self, "_sig_int_err"):
                self.full_fit()
            return self._sig_int_err
        else:
            if not hasattr(self, "_sig_int_err"):
                print("Intrisic color distribution is blocked: either define one or change the value of 'dust_only' to False")
            else:
                return self._sig_int_err
    
    def set_sig_int_err(self, sig_int_err):
        self._sig_int_err = sig_int_err
        
    
    @property
    def minuit(self):
        if not hasattr(self, "_minuit"):
            self.full_fit() ##CHANGE
        return self._minuit
    
    def set_minuit(self, minuit):
        self._minuit = minuit
    
    @property
    def data(self):
        return self._data
    
    @property
    def dust_only(self):
        if not hasattr(self, "_dust_only"):
            self._dust_only = True
        return self._dust_only
    
    def set_dust_only(self, dust_only):
        self._dust_only = dust_only
    

def get_loglikelihood_dust(data):
    def f(a):
        if a < 0:
            return 1e99
        else:
            data_cut = data[data>0]
            return np.sum(data_cut/a) + len(data_cut)*np.log(a)
    return f

def get_loglikelihood_sne(data, data_err):
    def f(mu, sig):
        sig_tot = np.sqrt(sig**2+data_err**2)
        return - np.sum(np.log(norm.pdf(data, loc=mu, scale=sig_tot)))
    return f

def get_loglikelihood_tot(data, xmin=-1, step=1e-3):
    def f(a, mu, sig):
        if (a < 0) :
            return 1e99
        else:
            xrange = np.arange(xmin, -xmin, step)
            dust = np.zeros(len(xrange))
            dust[xrange>0] = np.exp(-xrange[xrange>0]/a)/a
            full = np.convolve(dust, norm.pdf(xrange, loc=mu, scale=sig), mode='same')
            area = integrate.simps(y=full, x=xrange)
            prob_func = full/area
            ind = np.array((data-xmin)/step)
            return - np.sum(np.log(prob_func[ind.astype(int)]))
    return f

def get_loglikelihood_dustcut(data, mu, sig, xmin=-1, step=1e-3):
    def f(a):
        if (a < 0) :
            return 1e99
        else:
            xrange = np.arange(xmin, -xmin, step)
            dust = np.zeros(len(xrange))
            dust[xrange>0] = np.exp(-xrange[xrange>0]/a)/a
            full = np.convolve(dust, norm.pdf(xrange, loc=mu, scale=sig), mode='same')
            area = integrate.simps(y=full, x=xrange)
            prob_func = full/area
            ind = np.array((data-xmin)/step)
            return - np.sum(np.log(prob_func[ind.astype(int)]))
    return f

def colour_prob(a, mu, sig, xmax, step):
    xrange_pos = np.arange(0, xmax, step)
    xrange_neg = np.arange(-xmax, 0, step)
    xrange = np.concatenate([xrange_neg, xrange_pos])
    dust_neg = np.zeros(len(xrange_neg))
    dust_pos = np.exp(-xrange_pos/a)/a
    dust = np.concatenate([dust_neg, dust_pos])
    prob = np.convolve(dust, norm.pdf(xrange, loc=mu, scale=sig), mode='same')
    return CubicSpline(xrange, 1e3*prob/np.sum(prob))

def get_loglikelihood_tot_err(data, data_err, xmax=1, step=1e-3):
    def f(a, mu, sig):
        if (a < 0) or (sig < 0):
            return 1e99
        else:
            logprob = 0
            for i in range(len(data)):
                distrib = colour_prob(a, mu, np.sqrt(sig**2+data_err.iloc[i]**2), xmax, step)
                logprob += np.log(distrib(data.iloc[i]))
            return - logprob
    return f

def get_loglikelihood_dustcut_err(data, data_err, mu, sig, xmax=1, step=1e-3):
    def f(a):
        if (a < 0):
            return 1e99
        else:
            logprob = 0
            for i in range(len(data)):
                distrib = colour_prob(a, mu, np.sqrt(sig**2+data_err.iloc[i]**2), xmax, step)
                logprob += np.log(distrib(data.iloc[i]))
            return - logprob
    return f

def fit_function_dust(data, a):
    fit_func = np.zeros(len(data))
    fit_func[data>0] = np.exp(-data[data>0]/a)/a
    return fit_func

def fit_function_sne(data, mu, sig):
    return norm.pdf(data, loc=mu, scale=sig)

def fit_function_tot(data, a, mu, sig): ##this might return the correct solution offsetted in x by half the step, can be negliged by using a small step in data
    dust = np.zeros(len(data))
    dust[data>0] = np.exp(-data[data>0]/a)/a
    full = np.convolve(dust, norm.pdf(data, loc=mu, scale=sig), mode='same')
    area = integrate.simps(y=full, x=data)
    return full/area

