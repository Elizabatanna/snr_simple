# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 13:53:08 2025

@author: basharova
"""
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import numpy as np
import inspect
#from statsmodels.tsa.stattools import acf

def load_file_raman(path):
    #gr_path = r"D:/earthcarekit_new/ec_data/"  + str(filename)
    ds = xr.open_dataset(os.path.join(path))
    
    bsc_data = ds["aerBsc_raman_355"].values
    bsc_1064 = ds["aerBsc_raman_1064"].values
    ext_data = ds["aerExt_raman_355"].values
    bsc_err = ds["uncertainty_aerBsc_raman_355"]
    lr_data  = ds["aerLR_raman_355"].values
    lr_data_error  = ds["uncertainty_aerLR_raman_532"].values
    dep_data = ds["parDepol_raman_532"].values
    height = ds["height"].values
    return bsc_data,ext_data, lr_data, dep_data, height
def load_file_klett(path):
    #gr_path = r"D:/earthcarekit_new/ec_data/"  + str(filename)
    ds = xr.open_dataset(os.path.join(path))
    
    bsc_data = ds["aerBsc_klett_355"].values
    bsc_1064 = ds["aerBsc_klett_1064"].values
  
    

    dep_data = ds["parDepol_klett_532"].values
    height = ds["height"].values
    return bsc_data, dep_data, height
def load_file_jap(path):
    #gr_path = r"D:/earthcarekit_new/ec_data/"  + str(filename)
    ds = xr.open_dataset(os.path.join(path))

    bsc_data = ds['AEROSOL.BACKSCATTER.COEFFICIENT'].values#[200:]
    ext_data = ds['AEROSOL.EXTINCTION.COEFFICIENT'].values#[200:]
    
    lr_data  = ds['AEROSOL.BACKSCATTER.RATIO'].values#[200:]
    dep_data = ds['AEROSOL.DEPOLARIZATION.RATIO.LINEAR'].values#[200:]
    height = ds["ALTITUDE"].values#[200:]
    return bsc_data, ext_data, lr_data, dep_data, height


def smooth(data, sigma):
    return gaussian_filter1d(data, sigma)

def savgol(data, window, polyorder):
    return savgol_filter(data, window, polyorder=polyorder)
    

def snr_calc(signal, smoothed,window): ####apply to smoothed data
        residues = signal - smoothed
        #noise   = np.sqrt(np.convolve(residues**2, np.ones(window)/window, mode="valid"))
        noise_valid = np.sqrt(np.convolve(
        residues**2, np.ones(window)/window, mode="valid"
        ))
        # pad with nan to restore original length
        pad = len(signal) - len(noise_valid)
        pad_top = pad // 2
        pad_bot = pad - pad_top
        noise = np.concatenate([
            np.full(pad_top, np.nan),
            noise_valid,
            np.full(pad_bot, np.nan)
        ])
        
        
        return np.abs(smoothed/noise)

#def acf_f(data, nlags, fft, missing):
#    return acf(data, nlags=nlags, fft=fft, missing=missing)

def resid(signal, smoothed): ####apply to smoothed data
        return signal - smoothed
        
            
            
def mask_low_snr(snr, threshold, data):
    return np.where(snr > threshold, data, np.nan)



def overlap_cut(data, overlap_h, height):   ####masked data used
    max_over = np.nanmax(data[height < overlap_h])
    print(max_over)
    overlap_height = height[np.argwhere(data==max_over)]
    print(overlap_height)
    cut=int(overlap_height[0][0])
    #print(cut)
    mask = (height < cut)
    return mask

def plot( data,height, color=None, alpha=None):
    frame = inspect.currentframe().f_back
    var_name = None
    for name, val in frame.f_locals.items():
        if id(val) == id(data):
            var_name = name
            break
    
    if var_name is None:
        var_name = "data"
    plt.plot(data, height, label=var_name, color=color, alpha=alpha)
    plt.legend()
    #plt.show()
    
def fill_value_mask(data, fill_value):
    return np.where(data != fill_value,data, np.nan)

def check_bsc_status(ds):
    if "aerBsc_raman_355" not in ds:
        return "missing"


    bsc = ds["aerBsc_raman_355"]
    
    
    # Force minimal evaluation
    all_nan = bsc.isnull().all().item()
    all_zero = ((bsc == 0).all()).item()
    
    
    if all_nan:
        return "all_nan"
    if all_zero:
        return "all_zero"
    
    
    return "ok"

import numpy as np

def cand_mask_to_layers(cand_mask):
    """
    Convert boolean candidate mask into list of (bottom_idx, top_idx) layer pairs
    """
    layers = []
    # Find where cand_mask changes (True ↔ False)
    diff = np.diff(cand_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1  # start of True region
    ends   = np.where(diff == -1)[0]     # end of True region

    # Handle edge cases
    if cand_mask[0]:
        starts = np.r_[0, starts]
    if cand_mask[-1]:
        ends = np.r_[ends, len(cand_mask)-1]

    # Pair them
    for s, e in zip(starts, ends):
        if e-s != 1:
            layers.append((s, e))
    return layers
