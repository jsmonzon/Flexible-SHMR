import numpy as np
import matplotlib.pyplot as plt
from numpy.random import poisson
from scipy.stats import ks_2samp
from scipy.special import gamma, loggamma, factorial
from scipy import stats

def pdf(data):
    index, counts = np.unique(data, return_counts=True)
    full = np.zeros(700) # the max number of unique counts across the models
    # needs to be set sufficiently high such that even extreme models can populate the Pnsat matrix
    full[index.astype("int")] = counts/data.shape[0]
    return full

def cumulative(lgMs_1D:np.ndarray, mass_bins, return_bins=False):
    N = np.histogram(lgMs_1D, bins=mass_bins)[0]
    if return_bins:
        return np.cumsum(N[::-1])[::-1], (mass_bins[:-1] + mass_bins[1:]) / 2
    else:
        return np.cumsum(N[::-1])[::-1]
    
def count(lgMs_1D:np.ndarray, mass_bins, return_bins=False):
    N = np.histogram(lgMs_1D, bins=mass_bins)[0]
    if return_bins:
        return N, (mass_bins[:-1] + mass_bins[1:]) / 2
    else:
        return N
    
def correlation(stat1, stat2):
    return stats.pearsonr(stat1, stat2)[0]

def ecdf(data):
    return np.arange(1, data.shape[0]+1)/float(data.shape[0])

def N_rank(arr, threshold, fillval=np.nan):
    sorted_arr = np.sort(arr, axis=1) # sort the masses
    mask = (sorted_arr > threshold) & (~np.isnan(sorted_arr)) # are they above the threshold? cut out the nan values
    masked_sorted_arr = np.where(mask, sorted_arr, np.nan)
    uneven = list(map(lambda row: row[~np.isnan(row)], masked_sorted_arr)) #setting up a list of lists
    lens = np.array(list(map(len, uneven))) # which list has the most elements?
    shift = lens[:,None] > np.arange(lens.max())[::-1] #flipping so it starts with the largest
    even = np.full(shift.shape, fillval)
    even[shift] = np.concatenate(uneven)
    full_rank = even[:, ::-1]
    return full_rank[~np.isnan(full_rank).all(axis=1)] # this automatically removes all rows that are filled with nans 


class SatStats_D:

    def __init__(self, lgMs, min_mass):
        self.lgMs = lgMs
        self.min_mass = min_mass

        self.mass_rank = N_rank(self.lgMs, threshold=self.min_mass)
        self.Nsat_perhost = np.sum(~np.isnan(self.mass_rank), axis=1)
        self.PNsat = pdf(self.Nsat_perhost)
        self.Nsat_unibin, self.Nsat_perbin = np.unique(self.Nsat_perhost, return_counts=True)

        #self.Nsat_completeness = np.sum(~np.isnan(self.mass_rank), axis=0)
        #self.N_grtM = np.arange(0, self.mass_rank.shape[1])

        self.Nsat_index = np.insert(np.cumsum(self.Nsat_perbin),0,0)
        self.maxmass = self.mass_rank[:,0] # this is where you can toggle through frames! the second most massive and so on
        self.max_split = np.split(self.maxmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]

        self.sigma_N = np.nanstd(self.Nsat_perhost)
        self.correlation = correlation(self.Nsat_perhost[self.Nsat_perhost>0], self.maxmass[self.Nsat_perhost>0])

        self.totmass = np.log10(np.nansum(10**self.mass_rank, axis=1))
        self.tot_split = np.split(self.totmass[np.argsort(self.Nsat_perhost)], self.Nsat_index)[1:-1]

        self.Neff_mask = self.Nsat_perbin > 4 # need to feed this to the models in the KS test step
        self.model_mask = self.Nsat_unibin[self.Neff_mask].tolist() 
        self.clean_max_split = list(map(self.max_split.__getitem__, np.where(self.Neff_mask)[0].tolist()))
        self.clean_tot_split = list(map(self.tot_split.__getitem__, np.where(self.Neff_mask)[0].tolist()))

        #just for plotting!
        self.PNsat_range = np.arange(self.PNsat.shape[0])

        self.Msmax_sorted = np.sort(self.maxmass)
        self.ecdf_Msmax = ecdf(self.Msmax_sorted)

        self.Mstot_sorted = np.sort(self.totmass)
        self.ecdf_Mstot = ecdf(self.Mstot_sorted)


        # self.mass_bins = np.linspace(4.5,10.5,45)
        # self.CSMF_counts = np.apply_along_axis(cumulative, 1, self.lgMs, mass_bins=self.mass_bins) 
        # self.quant = np.percentile(self.CSMF_counts, np.array([5, 50, 95]), axis=0, method="closest_observation")
        # self.D23_quant = np.sum(self.CSMF_counts, axis=0)

            # def CSMF_plot(self):
    #     plt.figure(figsize=(6,6))
    #     plt.plot(self.mass_bins, self.quant[1], label="median")
    #     plt.fill_between(self.mass_bins, y1=self.quant[0], y2=self.quant[2], alpha=0.2, label="5% - 95%")
    #     plt.xlabel("log m$_{stellar}$ (M$_\odot$)", fontsize=15)
    #     plt.ylabel("N (> m$_{stellar}$)", fontsize=15)
    #     plt.xlim(6.5, 11)
    #     plt.yscale("log")
    #     plt.legend()
    #     plt.show()

    def Pnsat_plot(self):
        plt.figure(figsize=(6,6))
        plt.plot(self.PNsat_range, self.PNsat)
        plt.xlabel("N satellites > $10^{"+str(self.min_mass)+"} \mathrm{M_{\odot}}$", fontsize=15)
        plt.ylabel("PDF", fontsize=15)
        plt.xlim(0,35)
        plt.show()

    def Msmax_plot(self):
        plt.figure(figsize=(6,6))
        plt.plot(np.sort(self.maxmass), ecdf(np.sort(self.maxmass)))
        plt.xlabel("max (M$_*$) ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.show()

    def Mstot_plot(self):
        plt.figure(figsize=(6,6))
        plt.plot(np.sort(self.totmass), ecdf(np.sort(self.totmass)))
        plt.xlabel("max (M$_*$) ($\mathrm{log\ M_{\odot}}$)", fontsize=15)
        plt.ylabel("CDF", fontsize=15)
        plt.show()
