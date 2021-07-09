#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:34:57 2021

@author: sam
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as stats
import glob
import os

%matplotlib inline

from pathlib import Path
''' prepare hazard data '''
# load obs data
DP = Path('/Users/sam/OneDrive - ETH Zurich/WCR/Projects/Heat/Data/')
ZRH_mort = pd.read_csv(DP.joinpath('Swiss/ZRH_heat_mortality.csv'))

# load model data
DP_LE = Path('/Users/sam/OneDrive - ETH Zurich/WCR/Projects/Heat/Data/ESM/CESM12-LE_ZRH/')
start_date = ZRH_mort.date.min()
end_date = ZRH_mort.date.max()
lat_ZRH = 47.3
lon_ZRH = 8.5

### load all ZRH CESM-LS data, adjust to Celsisus, create DF

def load_CESM_LS_data(data_path, lon, lat, start_date, end_date):

    file_list = [f for f in glob.glob(os.path.join(data_path, "*.nc"))]
    for i in range(len(file_list)):
        print(i)
        # load data
        data_ESM = xr.open_dataset(file_list[i])
        dat = data_ESM.sel(
            time=slice(start_date, end_date),
            lon=slice(lon-1., lon+1.),
            lat=slice(lat-1., lat+1.))

        dat_df = dat.to_dataframe()
        dat_df.TREFHT = dat_df.TREFHT-273.14
        dat_df = dat_df.reset_index()
        
        if i==0:
            data_LS = dat_df
            data_LS = data_LS.rename(columns={'TREFHT': "ens_"+str(i+1)})
        else:
            data_LS["ens_"+str(i+1)] = dat_df.TREFHT
    return(data_LS)

CESM_LS_data = load_CESM_LS_data(DP_LE, lon_ZRH, lat_ZRH, start_date, end_date)
CESM_LS_data_T_only = CESM_LS_data.drop(['lat', 'lon', 'time'], axis=1)

# do bias correction (quantile mapping)
# see functions from bias correction class
ZRH_LE_REF = bias_correct_ensemble(CESM_LS_data_T_only, ZRH_mort.Tabsd,
                                   CESM_LS_data_T_only)
# check bias v no bias
bins=np.linspace(-20, 35, 56)
for i in range(ZRH_LE_REF.shape[1]):
    density = stats.gaussian_kde(ZRH_LE_REF.iloc[:,i])
    plt.plot(bins, density(bins), color="green", alpha=0.2)
for i in range(CESM_LS_data_T_only.shape[1]):
    density = stats.gaussian_kde(CESM_LS_data_T_only.iloc[:,i])
    plt.plot(bins, density(bins), color="blue", alpha=0.2)
density = stats.gaussian_kde(ZRH_mort.Tabsd)
plt.plot(bins, density(bins), color="black", linewidth=2)
plt.xlabel("Temperature [C]", fontsize=12)
plt.ylabel("Density", fontsize=12)

p1 = plt.plot(np.NaN, np.NaN, color="black", linewidth=2)
p2 = plt.plot(np.NaN, np.NaN, '-', color="blue", alpha=0.9)
p3 = plt.plot(np.NaN, np.NaN, color="green")

plt.legend([p1[0], p2[0], p3[0]],
           ['Measurement timeseries, ZRH Fluntern, 1989-2017',
            'Nearest gridpoint to ZRH, 82 Large ensemble runs, 1989-2017',
            'Nearest gridpoint to ZRH, 82 Large ensemble runs, 1989-2017, bias corrected'],
           bbox_to_anchor=(0.4, -0.5), loc='lower center',
            fontsize=11, frameon=False)
plt.show()

# load climate projections
start_date_2000 = '1990-01-01'
end_date_2000 = '2010-12-30'
CESM_LS_data_2000 = load_CESM_LS_data(DP_LE, lon_ZRH, lat_ZRH, start_date_2000, end_date_2000)
CESM_LS_data_2000_T_only = CESM_LS_data_2000.drop(['lat', 'lon', 'time'], axis=1)
ZRH_LE_2000 = bias_correct_ensemble(CESM_LS_data_T_only, ZRH_mort.Tabsd,
                                   CESM_LS_data_2000_T_only)

start_date_now = '2010-01-01'
end_date_now = '2029-12-30'
CESM_LS_data_now = load_CESM_LS_data(DP_LE, lon_ZRH, lat_ZRH, start_date_now, end_date_now)
CESM_LS_data_now_T_only = CESM_LS_data_now.drop(['lat', 'lon', 'time'], axis=1)
ZRH_LE_NOW = bias_correct_ensemble(CESM_LS_data_T_only, ZRH_mort.Tabsd,
                                   CESM_LS_data_now_T_only)

start_date_2030 = '2020-01-01'
end_date_2030 = '2039-12-30'
CESM_LS_data_2030 = load_CESM_LS_data(DP_LE, lon_ZRH, lat_ZRH, start_date_2030, end_date_2030)
CESM_LS_data_2030_T_only = CESM_LS_data_2030.drop(['lat', 'lon', 'time'], axis=1)
ZRH_LE_2030 = bias_correct_ensemble(CESM_LS_data_T_only, ZRH_mort.Tabsd,
                                   CESM_LS_data_2030_T_only)

# check projections
bins=np.linspace(-20, 35, 56)
for i in range(ZRH_LE_REF.shape[1]):
    density = stats.gaussian_kde(ZRH_LE_REF.iloc[:,i])
    plt.plot(bins, density(bins), color="green", alpha=0.2)
for i in range(ZRH_LE_NOW.shape[1]):
    density = stats.gaussian_kde(ZRH_LE_2000.iloc[:,i])
    plt.plot(bins, density(bins), color="khaki", alpha=0.2)   
for i in range(ZRH_LE_NOW.shape[1]):
    density = stats.gaussian_kde(ZRH_LE_NOW.iloc[:,i])
    plt.plot(bins, density(bins), color="olive", alpha=0.2)
for i in range(ZRH_LE_2030.shape[1]):
    density = stats.gaussian_kde(ZRH_LE_2030.iloc[:,i])
    plt.plot(bins, density(bins), color="goldenrod", alpha=0.2)

density = stats.gaussian_kde(ZRH_mort.Tabsd)
plt.plot(bins, density(bins), color="black", linewidth=2)
plt.xlabel("Temperature [C]", fontsize=12)
plt.ylabel("Density", fontsize=12)

p1 = plt.plot(np.NaN, np.NaN, color="black", linewidth=2)
p2 = plt.plot(np.NaN, np.NaN, color="green")
p3 = plt.plot(np.NaN, np.NaN, color="khaki")
p4 = plt.plot(np.NaN, np.NaN, color="olive")
p5 = plt.plot(np.NaN, np.NaN, color="goldenrod")


plt.legend([p1[0], p2[0], p3[0], p4[0], p5[0]],
           ['Measurement timeseries, ZRH Fluntern, 1989-2017',
            'Nearest gridpoint to ZRH, 82 Large ensemble runs, 1989-2017, bias corrected',
            'Nearest gridpoint to ZRH, 82 Large ensemble runs, 1990-2010, bias corrected',
            'Nearest gridpoint to ZRH, 82 Large ensemble runs, 2010-2029, bias corrected',
            'Nearest gridpoint to ZRH, 82 Large ensemble runs, 2020-2039, bias corrected'],
           bbox_to_anchor=(0.4, -0.6), loc='lower center',
            fontsize=11, frameon=False)
plt.show()

''' load vulnerability and exposure data '''
ZRH_RR = pd.read_csv(DP.joinpath('Swiss/RR_ZRH.csv'))
ZRH_RR = pd.read_csv(DP.joinpath('Swiss/RR_ZRH_hack.csv')) # we're taking the pseude-extrapolation
ZRH_pop = pd.read_csv(DP.joinpath('Swiss/ZRH_pop.csv'))

def plot_RR_space(RR, col='blue'):
    
    fig, ax = plt.subplots()
    ax.plot(RR.temp.values, RR.RRfit.values, '-', color=col)
    ax.fill_between(RR.temp.values, RR.RRlow.values, RR.RRhigh.values,
                    color=col, alpha=0.2)
    p1 = ax.plot(np.NaN, np.NaN, color=col)
    p2 = ax.fill(np.NaN, np.NaN, color=col, alpha=0.2)
    plt.plot([35.0,-15.], [1.0, 1.0], 'k-',linewidth=1.)
    plt.ylim([0.8,4.0])
    plt.xlim(-15,35)    
    plt.xlabel("Temperature [C]", fontsize=12)
    plt.ylabel("Relative risk", fontsize=12)

plot_RR_space(ZRH_RR, col='blue')

# get covered years from mort
ZRH_mort["date"] = pd.to_datetime(ZRH_mort["date"])
ZRH_mort["year"] = ZRH_mort["date"].dt.year
ZRH_base = pd.DataFrame()
ZRH_base["deaths"] = ZRH_mort.groupby('year')['deaths'].sum()
ZRH_base["Year"] = np.arange(ZRH_mort["year"].min(),ZRH_mort["year"].max()+1)
ZRH_base = pd.merge(ZRH_base, ZRH_pop, on="Year")
ZRH_base["mortality"] = ZRH_base["deaths"]/ZRH_base["Pop"]

# express impactfunction as #death per million pop per day vs temp
daily_mort = ZRH_base.mortality.mean()/365.25
ZRH_mort_IF = ZRH_RR
ZRH_mort_IF.RRfit = ZRH_mort_IF.RRfit*daily_mort*10**6
ZRH_mort_IF.RRlow = ZRH_mort_IF.RRlow*daily_mort*10**6
ZRH_mort_IF.RRhigh = ZRH_mort_IF.RRhigh*daily_mort*10**6
ZRH_mort_IF = ZRH_mort_IF.rename(columns={"RRfit": "Rfit", "RRlow": "Rlow", "RRhigh": "Rhigh"}) # rename T

def plot_MC_space(MC, col='red'):
    
    fig, ax = plt.subplots()
    ax.plot(MC.temp.values, MC.Rfit.values, '-', color=col)
    ax.fill_between(MC.temp.values, MC.Rlow.values, MC.Rhigh.values,
                    color=col, alpha=0.2)
    p1 = ax.plot(np.NaN, np.NaN, color=col)
    p2 = ax.fill(np.NaN, np.NaN, color=col, alpha=0.2)
    #plt.ylim([0.8,2.0])
    plt.xlim(-15,35)    
    plt.xlabel("Temperature [C]", fontsize=12)
    plt.ylabel("All-age mortality (per million)", fontsize=12)

plot_MC_space(ZRH_mort_IF, col='salmon')

# indicate heat mortality space
TMM = ZRH_mort_IF.temp.values[ZRH_mort_IF.Rfit.argmin()]
def plot_HM_space(MC, TMM, col='lightcoral', col2='blueviolet'):
    
    fig, ax = plt.subplots()
    ax.plot(MC.temp.values, MC.Rfit.values, '-', color=col)
    ax.fill_between(MC.temp.values, MC.Rlow.values, MC.Rhigh.values,
                    color=col, alpha=0.2)
    ax.fill_between(MC.temp.values[MC.temp.values>=TMM],
                    np.repeat(MC.Rfit.values[TMM==MC.temp.values],
                              len(MC.temp.values[MC.temp.values>=TMM])),
                    MC.Rfit.values[MC.temp.values>=TMM],
                    color=col2, alpha=0.2)
    p1 = ax.plot(np.NaN, np.NaN, color=col)
    p2 = ax.fill(np.NaN, np.NaN, color=col, alpha=0.2)
    plt.xlim(-15,35)
    plt.ylim(15,80)
    plt.xlabel("Temperature [C]", fontsize=12)
    plt.ylabel("All-age mortality (per million)", fontsize=12)

plot_HM_space(ZRH_mort_IF, TMM)


''' calculate impact '''
# make mortality calcs all ensembles
def calc_mort_ens(ens, exp, IF_mort, IF_temp, TMM):
    out_all_mort = np.zeros(ens.shape)
    out_temp_mort = np.zeros(ens.shape)
    out_heat_mort = np.zeros(ens.shape)

    m = np.interp(ens, IF_temp, IF_mort)
    out_all_mort = m * exp / 10**6
    out_temp_mort = out_all_mort - (IF_mort.min() / 10**6 * exp)
    
    out_heat_mort = out_temp_mort.copy()
    out_heat_mort[TMM>ens] = 0
    
    return out_all_mort, out_temp_mort, out_heat_mort

exp = pop_z = ZRH_base.Pop.values[-1]
a,b,c = calc_mort_ens(CESM_LS_data_T_only.to_numpy(), exp, ZRH_mort_IF.Rfit.values, ZRH_mort_IF.temp.values, TMM)
a_00,b_00,c_00 = calc_mort_ens(ZRH_LE_2000.to_numpy(), exp, ZRH_mort_IF.Rfit.values, ZRH_mort_IF.temp.values, TMM)
a_now, b_now, c_now = calc_mort_ens(ZRH_LE_NOW.to_numpy(), exp, ZRH_mort_IF.Rfit.values, ZRH_mort_IF.temp.values, TMM)
a_30, b_30, c_30 = calc_mort_ens(ZRH_LE_2030.to_numpy(), exp, ZRH_mort_IF.Rfit.values, ZRH_mort_IF.temp.values, TMM)
a_obs, b_obs, c_obs = calc_mort_ens(ZRH_mort.Tabsd.values, exp, ZRH_mort_IF.Rfit.values, ZRH_mort_IF.temp.values, TMM)

# check impacts on years
def calc_imp_year(ens_mort):
    n_years = int((ens_mort.shape[0]+1)/365)
    try: n_ens = ens_mort.shape[1]
    except IndexError: n_ens = 1
    annual_mort = np.zeros([n_years, n_ens])
    if n_ens > 1:
        for i in range(n_years):
            annual_mort[i,:] = np.sum(ens_mort[i*365:(i+1)*365,:], axis=0)
    else:
        for i in range(n_years):
            annual_mort[i] = np.sum(ens_mort[i*365:(i+1)*365], axis=0)
    return annual_mort

d = calc_imp_year(c)
d_00 = calc_imp_year(c_00)
d_now = calc_imp_year(c_now)
d_30 = calc_imp_year(c_30)
d_obs = calc_imp_year(c_obs)

def plot_hist_ens_annual_mort(mort, bins=np.linspace(0, 600, 21),
                              col="salmon", legend=True):
    for i in range(mort.shape[1]):
        density = stats.gaussian_kde(mort[:,i])
        plt.plot(bins, density(bins), color=col, alpha=0.2)
        
    plt.xlabel("Excess mortality [#]", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    if legend:
        p1 = plt.plot(np.NaN, np.NaN, color=col)
        plt.legend([p1[0]],
               ['Excess mortality ZRH, 82 Large ensemble runs'],
               bbox_to_anchor=(0.4, -0.3), loc='lower center',
                fontsize=11, frameon=False)  
    
plot_hist_ens_annual_mort(d_00)
plot_hist_ens_annual_mort(d_now, col='goldenrod')
plot_hist_ens_annual_mort(d_30, col='blue')
plot_hist_ens_annual_mort(d_obs, col='black')


''' imapct freqeuncy curve '''
def calc_ImpFreqCurve(mort_imp, freq=None):
    event = mort_imp.ravel()
    if freq==None:
        freq = np.zeros(len(event))
        freq[:] = 1./(len(event))
    sort_idxs = np.argsort(event)[::-1]
    # Calculate exceedence frequency
    exceed_freq = np.cumsum(freq[sort_idxs])
    # Set return period and imact exceeding frequency
    return_per = 1 / exceed_freq[::-1]
    impact = event[sort_idxs][::-1]
    
    return impact, return_per


# plt
def plot_ImpFreqCurve(impact, return_per, axis=None, log_frequency=False, **kwargs):
    if not axis:
        _, axis = plt.subplots(1, 1)
    axis.set_title("")
    axis.set_ylabel('Impact (# lifes lost)')
    if log_frequency:
        axis.set_xlabel('Exceedance frequency (1/year)')
        axis.set_xscale('log')
        axis.plot(return_per**-1, impact, **kwargs)
    else:
        axis.set_xlabel('Return period (year)')
        axis.plot(return_per, impact, **kwargs)
    return axis

i_00, r_00 = calc_ImpFreqCurve(d_00)
i_now, r_now = calc_ImpFreqCurve(d_now)
i_30, r_30 = calc_ImpFreqCurve(d_30)
i_obs, r_obs = calc_ImpFreqCurve(d_obs)

# plot impact curves
plot_ImpFreqCurve(i_00, r_00, log_frequency=True)
plot_ImpFreqCurve(i_now, r_now, log_frequency=True)
plot_ImpFreqCurve(i_30, r_30, log_frequency=True)
plot_ImpFreqCurve(i_obs, r_obs, log_frequency=True)


plt.plot(r_00**-1, i_00, color='goldenrod')
plt.plot(r_now**-1, i_now, color='khaki')
plt.plot(r_30**-1, i_30, color='darkkhaki')
plt.plot(r_obs**-1, i_obs, color='black', linewidth=2)

plt.xscale('log')
plt.xlabel("Exceedance frequency (1/year)", fontsize=12)
plt.ylabel("Impact [# lifes]", fontsize=12)
p1 = plt.plot(np.NaN, np.NaN, color="goldenrod")
p2 = plt.plot(np.NaN, np.NaN, color="khaki")
p3 = plt.plot(np.NaN, np.NaN, color="darkkhaki")
p4 = plt.plot(np.NaN, np.NaN, color="black")

plt.legend([p1[0], p2[0],p3[0], p4[0]],
           ['Risk 2000 ZRH - 82 Large ensemble runs - 1990-2009',
            'Risk now ZRH - 82 Large ensemble runs - 2010-2029',
            'Risk 2030 ZRH - 82 Large ensemble runs - 2020-2039',
            'Observational estimate, 1989-2017'],
           bbox_to_anchor=(0.3, -0.5), loc='lower center',
            fontsize=11, frameon=False)


''' include spread in vulnerability '''
# use low and high Rfit
a_00_low,b_00_low,c_00_low = calc_mort_ens(ZRH_LE_2000.to_numpy(), exp, ZRH_mort_IF.Rlow.values, ZRH_mort_IF.temp.values, TMM)
a_now_low, b_now_low, c_now_low = calc_mort_ens(ZRH_LE_NOW.to_numpy(), exp, ZRH_mort_IF.Rlow.values, ZRH_mort_IF.temp.values, TMM)
a_30_low, b_30_low, c_30_low = calc_mort_ens(ZRH_LE_2030.to_numpy(), exp, ZRH_mort_IF.Rlow.values, ZRH_mort_IF.temp.values, TMM)
a_obs_low, b_obs_low, c_obs_low = calc_mort_ens(ZRH_mort.Tabsd.values, exp, ZRH_mort_IF.Rlow.values, ZRH_mort_IF.temp.values, TMM)

a_00_high,b_00_high,c_00_high = calc_mort_ens(ZRH_LE_2000.to_numpy(), exp, ZRH_mort_IF.Rhigh.values, ZRH_mort_IF.temp.values, TMM)
a_now_high, b_now_high, c_now_high = calc_mort_ens(ZRH_LE_NOW.to_numpy(), exp, ZRH_mort_IF.Rhigh.values, ZRH_mort_IF.temp.values, TMM)
a_30_high, b_30_high, c_30_high = calc_mort_ens(ZRH_LE_2030.to_numpy(), exp, ZRH_mort_IF.Rhigh.values, ZRH_mort_IF.temp.values, TMM)
a_obs_high, b_obs_high, c_obs_high = calc_mort_ens(ZRH_mort.Tabsd.values, exp, ZRH_mort_IF.Rhigh.values, ZRH_mort_IF.temp.values, TMM)

d_00_low = calc_imp_year(c_00_low)
d_now_low = calc_imp_year(c_now_low)
d_30_low = calc_imp_year(c_30_low)
d_obs_low = calc_imp_year(c_obs_low)

d_00_high = calc_imp_year(c_00_high)
d_now_high = calc_imp_year(c_now_high)
d_30_high = calc_imp_year(c_30_high)
d_obs_high = calc_imp_year(c_obs_high)

i_00_low, r_00_low = calc_ImpFreqCurve(d_00_low)
i_now_low, r_now_low = calc_ImpFreqCurve(d_now_low)
i_30_low, r_30_low = calc_ImpFreqCurve(d_30_low)
i_obs_low, r_obs_low = calc_ImpFreqCurve(d_obs_low)

i_00_high, r_00_high = calc_ImpFreqCurve(d_00_high)
i_now_high, r_now_high = calc_ImpFreqCurve(d_now_high)
i_30_high, r_30_high = calc_ImpFreqCurve(d_30_high)
i_obs_high, r_obs_high = calc_ImpFreqCurve(d_obs_high)

# plot impact curves
plt.plot(r_00**-1, i_00, color='goldenrod')
plt.plot(r_now**-1, i_now, color='sandybrown')
plt.plot(r_30**-1, i_30, color='peru')
plt.plot(r_obs**-1, i_obs, color='black', linewidth=2)

plt.fill_between(r_00**-1, i_00_low, i_00_high,
                    color='goldenrod', alpha=0.2)
plt.fill_between(r_now**-1, i_now_low, i_now_high,
                    color='sandybrown', alpha=0.2)
plt.fill_between(r_30**-1, i_30_low, i_30_high,
                    color='peru', alpha=0.2)
plt.fill_between(r_obs**-1, i_obs_low, i_obs_high,
                    color='black', alpha=0.1)
plt.xscale('log')
plt.xlim([0.0008,1.0])
plt.xlabel("Exceedance frequency (1/year)", fontsize=12)
plt.ylabel("Impact [# lifes]", fontsize=12)
p1 = plt.plot(np.NaN, np.NaN, color="goldenrod")
p2 = plt.plot(np.NaN, np.NaN, color="sandybrown")
p3 = plt.plot(np.NaN, np.NaN, color="peru")
p4 = plt.plot(np.NaN, np.NaN, color="black")

plt.legend([p1[0], p2[0],p3[0], p4[0]],
           ['Risk 2000 ZRH - 82 Large ensemble runs - 1990-2009',
            'Risk now ZRH - 82 Large ensemble runs - 2010-2029',
            'Risk 2030 ZRH - 82 Large ensemble runs - 2020-2039',
            'Observational estimate, 1989-2017'],
           bbox_to_anchor=(0.3, -0.5), loc='lower center',
            fontsize=11, frameon=False)

# boxplot analysis
sns.boxplot([d_obs,d_00])