# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:40:43 2026

@author: basharova
"""

# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.ndimage import label

from noise_mod import load_file_raman, snr_calc, smooth


# ==========================================================
# SETTINGS
# ==========================================================


PATH = r'C:/Users/basharova/Downloads/TJK/nc/2019/'


SNR_THRESHOLD = 10
MIN_THICKNESS = 500

OVERLAP_MAX_HEIGHT = 2000

SMOOTH_WINDOW = 9
GROW_SMOOTH_WINDOW = 15

ZERO_THRESHOLD = 1

YMIN = 0
YMAX = 11000

SETTINGS = {

    "bsc": {
        "snr_threshold": 15,
        "zero_threshold": 1e-6,
        "xlim": (0, 1e-6),
        "overlap": 1500
    },

    "ext": {
        "snr_threshold": 10,
        "zero_threshold": 5,
        "xlim": (0, 0.3),
        "overlap": 1500
    },

    "lr": {
        "snr_threshold": 5,
        "zero_threshold": 5,
        "xlim": (0, 120),
        "overlap": 1500
    },

    "dep": {
        "snr_threshold": 25,
        "zero_threshold": 0.02,
        "xlim": (0, 0.5),
        "overlap": 1500
    }
}

# ==========================================================
# REMOVE THIN LAYERS
# ==========================================================

# def remove_thin_fragments(profile, height, min_thickness=500):

#     valid = ~np.isnan(profile)

#     labeled, n = label(valid)

#     result = profile.copy()

#     for i in range(1, n + 1):

#         idx = np.where(labeled == i)[0]

#         if len(idx) == 0:
#             continue

#         thickness = height[idx[-1]] - height[idx[0]]

#         if thickness < min_thickness:
#             result[idx] = np.nan

#     return result

def enforce_min_thickness(mask, height, min_thickness):
    labeled, n = label(mask)

    cleaned = np.zeros_like(mask, dtype=bool)

    for i in range(1, n + 1):
        idx = np.where(labeled == i)[0]

        if len(idx) < 2:
            continue

        thickness = height[idx[-1]] - height[idx[0]]

        if thickness >= min_thickness:
            cleaned[idx] = True

    return cleaned
# ==========================================================
# REMOVE OVERLAP REGION
# ==========================================================
def remove_overlap(profile, height, cutoff=1500):

    out = profile.copy()

    out[height < cutoff] = np.nan

    return out

def discard_overlap_region(profile,
                           height,
                           search_max_height=1000,
                           smooth_window=11):

    result = profile.copy()

    # region below max search height
    mask = height < search_max_height

    idx_all = np.where(mask)[0]

    prof = savgol_filter(
        np.nan_to_num(profile[mask]),
        smooth_window,
        3
    )

    # start from top and move downward
    for k in range(len(prof) - 2, 1, -1):

        # local minimum
        if (
            prof[k] < prof[k - 1]
            and prof[k] < prof[k + 1]
        ):

            result[:idx_all[k]] = np.nan

            return result

    # fallback:
    # remove everything below search height
    result[height < search_max_height] = np.nan

    return result
#########################################################################
#####  bottom and top for all layers it can identify date | bottom1 | top1 | bottom2 | top2 | .....
#####  march 2015 - august 2016 pollyxt_tropos, 2019 - 2025 tjk
######################################################


# ==========================================================
# GROW TO LOCAL MINIMA
# ==========================================================
def merge_close_layers(profile,
                       height,
                       max_gap=500):
    """
    Merge vertically close layer fragments.

    max_gap in meters.
    """

    valid = ~np.isnan(profile)

    labeled, n = label(valid)

    result = profile.copy()

    if n <= 1:
        return result

    for i in range(1, n):

        idx1 = np.where(labeled == i)[0]
        idx2 = np.where(labeled == i + 1)[0]

        if len(idx1) == 0 or len(idx2) == 0:
            continue

        top1 = height[idx1[-1]]
        bottom2 = height[idx2[0]]

        gap = bottom2 - top1

        # merge if gap small
        if gap < max_gap:

            fill_idx = np.arange(idx1[-1], idx2[0] + 1)

            result[fill_idx] = np.nanmean(profile[idx1])

    return result
def grow_to_minima(signal,
                   masked_signal,
                   snr_profile,
                   height,
                   snr_floor=5,
                   smooth_window=15,
                   zero_threshold=1):

    smooth_sig = savgol_filter(
        np.nan_to_num(signal),
        smooth_window,
        3
    )

    valid = ~np.isnan(masked_signal)

    filled = valid.copy()

    labeled, n = label(valid)

    for i in range(1, n + 1):

        region = np.where(labeled == i)[0]

        # --------------------------
        # upward growth
        # --------------------------

        j = region[-1] + 1

        while j < len(signal) - 2:

            if snr_profile[j] < settings['snr_threshold']:
                break

            filled[j] = True

            # local minimum near baseline
            if (
                smooth_sig[j] < smooth_sig[j - 1]
                and smooth_sig[j] < smooth_sig[j + 1]
                and smooth_sig[j] < zero_threshold
            ):
                break

            j += 1

        # --------------------------
        # downward growth
        # --------------------------

        j = region[0] - 1

        while j > 1:
        
            if snr_profile[j] < snr_floor:
                break
        
            # strong decay from peak
            if smooth_sig[j] < 0.2 * smooth_sig[region].max():
                break
        
            filled[j] = True
        
            if (
                smooth_sig[j] < smooth_sig[j - 1]
                and smooth_sig[j] < smooth_sig[j + 1]
                and smooth_sig[j] < zero_threshold
            ):
                break
        
            j -= 1

    result = np.where(filled, signal, np.nan)

    return result


# ==========================================================
# MAIN SNR DETECTION
# ==========================================================

def detect_layers(var, height, settings):

    # --------------------------
    # smooth background
    # --------------------------

    bg = smooth(var, 4)

    # --------------------------
    # SNR
    # --------------------------

    snr_full = snr_calc(var, bg, window=70)
    print("SNR min/max:", np.nanmin(snr_full), np.nanmax(snr_full))
    print("fraction above threshold:",
      np.mean(snr_full >= SNR_THRESHOLD))
    # --------------------------
    # threshold
    # --------------------------

    detected = np.where(
        snr_full >= settings['snr_threshold'],
        var,
        np.nan
    )

    # --------------------------
    # cleanup
    # --------------------------

   


    # --------------------------
    # grow full hill
    # --------------------------
    mask = ~np.isnan(detected)
    
    mask = enforce_min_thickness(mask, height, 1000)
    
    detected = np.where(mask, detected, np.nan)
    #detected = discard_overlap_region(detected, height)
    #detected = merge_close_layers(detected, height)
    detected = grow_to_minima(
        signal=var,
        masked_signal=detected,
        snr_profile=snr_full,
        height=height,
        snr_floor=settings["snr_threshold"],
        smooth_window=GROW_SMOOTH_WINDOW,
        zero_threshold=settings["zero_threshold"]
    )
    detected = discard_overlap_region(detected, height)
    #detected = merge_close_layers(detected, height)
    
    #remove_overlap(
    #detected,
    #height,
    #cutoff=settings["overlap"]
#)
    return detected, snr_full
def extract_layer_properties(detected, signal, height, filename):

    valid = ~np.isnan(detected)

    labeled, n = label(valid)

    layers = []

    for i in range(1, n + 1):

        idx = np.where(labeled == i)[0]

        if len(idx) == 0:
            continue

        bottom = height[idx[0]]
        top = height[idx[-1]]

        thickness = top - bottom
        #thickness = top - bottom

        if thickness < MIN_THICKNESS:
            continue
        
        valid_points = np.sum(~np.isnan(signal[idx]))
        
        if valid_points < 5:
            continue
        center = np.mean(height[idx])

        mean_signal = np.nanmean(signal[idx])

        max_signal = np.nanmax(signal[idx])

        integrated_signal = np.trapz(signal[idx], height[idx])

        layer_info = {

            "file": filename,
            "layer_id": i,

            "bottom_m": bottom,
            "top_m": top,

            "center_m": center,
            "thickness_m": thickness,

            "mean_signal": mean_signal,
            "max_signal": max_signal,
            "integrated_signal": integrated_signal
        }

        layers.append(layer_info)

    return layers

# ==========================================================
# PLOT
# ==========================================================
def plot_result(raw, detected, height, xlim, title=''):

    plt.figure(figsize=(5, 8))

    plt.plot(raw, height,
             color='green',
             alpha=0.4,
             label='raw')

    plt.plot(detected, height,
             color='red',
             linewidth=2,
             label='detected')

    plt.ylim(YMIN, YMAX)
    plt.xlim(xlim)
    plt.title(title)

    plt.xlabel('Signal')
    plt.ylabel('Height (m)')

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()
    plt.ylim(0,20000)
    plt.xlim(0, 3e-6)
    plt.show()


# ==========================================================
# RUN
# ==========================================================

mean_values = []
month = []
top = []
bottom = []
name = []
top_lr = []
bottom_lr = []
top_dep = []
bottom_dep = []
top_bsc = []
bottom_bsc = []
for file in os.listdir(PATH):
    
    if not file.endswith('.nc'):
        continue
    name.append(file)
    print('\nProcessing:', file)

    bsc, ext, lr, dep, height = load_file_raman(
        os.path.join(PATH, file)
    )

    # choose variable
    SIGNAL_NAME = "bsc"

    SIGNALS = {
        "bsc": bsc,
        "ext": ext,
        "lr": lr,
        "dep": dep
    }
    settings = SETTINGS[SIGNAL_NAME]
    signal = SIGNALS[SIGNAL_NAME]
    if SIGNAL_NAME == "lr":

        signal = np.where(
            (signal > 0) & (signal < 120),
            signal,
            np.nan
        )
    if SIGNAL_NAME == "bsc":

        signal = np.where(
           (height < 8200),
           signal,
           np.nan
        )
    detected, snr = detect_layers(
        signal,
        height,
        settings
    )
    
    #mean_values.append(np.nanmean(detected))
    layer_results = extract_layer_properties(
    detected,
    signal,
    height,
    file
)

    for layer in layer_results:
    
        print("\nLayer detected:")
        print(layer)
    
        mean_values.append(layer["mean_signal"])
        month.append(layer["file"][:10])
        top.append(layer["top_m"])
        bottom.append(layer["bottom_m"])
        # if SIGNAL_NAME == "lr":
        #     top_lr.append(layer["top_m"])
        #     bottom_lr.append(layer["bottom_m"])
        # if SIGNAL_NAME == "dep":
        #     top_dep.append(layer["top_m"])
        #     bottom_dep.append(layer["bottom_m"])
        # if SIGNAL_NAME == "bsc":
        #     top_bsc.append(layer["top_m"])
        #     bottom_bsc.append(layer["bottom_m"])
    plot_result(
    raw=signal,
    detected=detected,
    height=height,
    xlim=settings["xlim"],
    title=f"{file} | {SIGNAL_NAME}"
    )

print('\nOverall mean:')
print(np.nanmean(mean_values))
# %%


import pandas as pd

df = pd.DataFrame({
    "month": month,
    "mean": mean_values
})
#df = df.drop_duplicates(subset='month',ignore_index=True)

plt.figure(figsize=(12,4))
monthly = df.groupby("month")["mean"].mean()
#plt.scatter(monthly.index, monthly.values, marker='o')

# plt.scatter(df['month'], df['mean'])
# monthly_mean = df.groupby("month")["mean"].mean()
# plt.xticks(rotation=90)

# #plt.tight_layout()

# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

df1 = pd.DataFrame({
    "date": month,
    "bottom": bottom,
    "top": top
})
#df1 = df1.drop_duplicates(subset='date')


# # ----------------------------
# # monthly averages
# # ----------------------------
import pandas as pd
plt.scatter(df1.date, df1.top,s=4)
# Convert your date column to datetime
df1['date'] = pd.to_datetime(df1['date'], format='%Y_%m_%d')

# Group by the integer month component (1 to 12)
#monthly_grouped = df.groupby(df['date'].dt.month).sum()

monthly = df1.groupby(df1['date'].dt.month).agg({
    "bottom": "mean",
    "top": "mean"
}).reset_index()

# # ----------------------------
# # plot monthly mean layers
# # ----------------------------
#plt.ylim(0,10000)

plt.scatter(monthly.date, monthly.top)
plt.xticks(rotation=90)
# %%
PATH = r'C:/Users/basharova/Downloads/dush_RF/SNR/'


import pandas as pd

df_manual = pd.read_excel(PATH + "manual_lay.xlsx")



df_manual["date"] = df_manual["date"].astype(str).str[:10]
#df_manual["date"] = df_manual["date"].dt.strftime("%Y_%m_%d")


df_merged = pd.merge(
    df1,
    df_manual,
    on=["date"],
    suffixes=("_model", "_manual")
)

df_merged["top_error"] = df_merged["layer_top"] - df_merged["top"]
df_merged["bottom_error"] = df_merged["layer_bot"] - df_merged["bottom"]

df_merged["thickness_model"] = df_merged["layer_top"] - df_merged["layer_bot"]
df_merged["thickness_manual"] = df_merged["top"] - df_merged["bottom"]

df_merged["thickness_error"] = df_merged["thickness_model"] - df_merged["thickness_manual"]


print("Mean absolute top error:",
      df_merged["top_error"].abs().mean())

print("Mean absolute bottom error:",
      df_merged["bottom_error"].abs().mean())

print("Top bias:", df_merged["top_error"].mean())
print("Bottom bias:", df_merged["bottom_error"].mean())
# %%



import matplotlib.pyplot as plt

plt.figure(figsize=(28,6))

plt.scatter(df1["date"],df1["top"],s=24, label="auto")
plt.scatter(df_merged["date"],df_merged["layer_top"],s=84, label="manual")
plt.ylim(9000,20000)
plt.legend()
plt.xticks(rotation=90)
plt.show()
# %%



fig, ax = plt.subplots(figsize=(12,6))

ax.vlines(
    monthly["date"],
    monthly["bottom"],
    monthly["top"],
    color='red',
    linewidth=3,
    label='autom',
    alpha=0.5
)

# optional points
ax.scatter(monthly["date"], monthly["top"])
ax.scatter(monthly["date"], monthly["bottom"])

# ax.vlines(
#     df_merged["date"],
#     df_merged["layer_bot"],
#     df_merged["layer_top"],
#     color='blue',
#     linewidth=3,
#     label='manual'
# )


# ax.scatter(df_merged['date'], df_merged['layer_top'])
# ax.scatter(df_merged['date'], df_merged['layer_bot'])


ax.set_ylabel("Height (m)")
ax.set_xlabel("Month")

plt.ylim(0, 20000)

plt.xticks(rotation=90)
plt.legend()
plt.tight_layout()
plt.show()


#plt.plot([0,11000],[0,11000],'k--')

# plt.xlabel("Model height (m)")
# plt.ylabel("Manual height (m)")
# plt.legend()
# plt.tight_layout()
# plt.show()

# %%

# Keep rows where the 'error' column is less than or equal to 500
df_merged = df_merged[abs(df_merged['top_error']) <= 700]
# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(28,6))

plt.scatter(df1["date"],df1["top"],s=24, label="auto")
plt.scatter(df_merged["date"],df_merged["layer_top"],s=84, label="manual")
plt.ylim(8000,20000)
plt.legend()
plt.xticks(rotation=90,fontsize=15)
plt.show()

fig, ax = plt.subplots(figsize=(6,6))

ax.scatter(
    df_merged["layer_top"],
    df_merged["top"],
    s=80,
    alpha=0.7
)
########  
# 1:1 line
lims = [1000, 20000]

ax.plot(lims, lims, 'k--')

ax.set_xlim(lims)
ax.set_ylim(lims)

ax.set_xlabel("Manual layer top (m)",fontsize=25)
ax.set_ylabel("SNR layer top (m)",fontsize=25)

# statistics
mae = df_merged["top_error"].abs().mean()
bias = df_merged["top_error"].mean()

ax.text(
    2000,
    7000,
    
    f"MAE = {mae:.0f} m\nBias = {bias:.0f} m",
    fontsize=16
)

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7,8))

x = np.arange(len(df_merged))

# manual
ax.vlines(
    x-0.1,
    df_merged["layer_bot"],
    df_merged["layer_top"],
    linewidth=4,
    color='red',
    label="Manual"
)

# SNR
ax.vlines(
    x+0.1,
    df_merged["bottom"],
    df_merged["top"],
    linewidth=4,
    alpha=0.6,
    label="SNR"
)

ax.set_xticks(x)
ax.set_xticklabels(df_merged["date"], rotation=90)
ax.set_ylim(0,20000)
ax.set_ylabel("Height (m)")
ax.set_title("layer thickness comparison")
ax.legend()

plt.tight_layout()
plt.show()

# %%


import matplotlib.pyplot as plt

plt.figure(figsize=(28,6))

plt.scatter(df_merged["date"],df_merged["top"],s=24, label="auto")
plt.scatter(df_merged["date"],df_merged["layer_top"],s=24, label="manual")
#plt.scatter(df_merged["date"],df_merged["bottom"],s=84, label="auto")
#plt.scatter(df_merged["date"],df_merged["layer_bot"],s=84, label="manual")
plt.title("manual vs automatic layer tops comparison")
plt.ylim(100,15000)
plt.legend()

plt.xticks(rotation=90)
plt.show()


