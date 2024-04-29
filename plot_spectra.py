"""
plot-spectra.py
Author: Toby Dixon (toby.dixon.23@ucl.ac.uk), Sofia Calgaro
"""
from legend_plot_style import LEGENDPlotStyle as lps
lps.use('legend_talks')
import shutil
import subprocess
from collections import OrderedDict
import uproot
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tol_colors as tc
from hist import Hist
import hist
import argparse
from datetime import datetime, timezone
import utils
import os
import sys
import re
import json
from legendmeta import LegendMetadata
plt.rcParams.update({'font.size': 24}) 

from matplotlib.backends.backend_pdf import PdfPages
vset = tc.tol_cset('vibrant')
mset = tc.tol_cset('muted')
plt.rc('axes', prop_cycle=plt.cycler('color', list(vset)))
plt.rcParams.update({
    'font.size': 20,          # Main font size
    'axes.labelsize': 20,     # Font size of x and y labels
    'axes.titlesize': 20,     # Font size of titles
    'xtick.labelsize': 16,    # Font size of x-axis ticks
    'ytick.labelsize': 16,    # Font size of y-axis ticks
    'legend.fontsize': 14,    # Font size of legend
    'figure.titlesize': 20    # Font size of figure titles
})

gamma_isotope = {   "K42_1525"    :    "K-42",
                    "K40_1461"    :    "K-40",
                    "Co60_1332"   :    "Co-60",
                    "Co60_1173"   :    "Co-60",
                    "Ac228_911"   :    "Ac-228",
                    "Bi212_727"   :    "Bi-212",
                    "Tl208_2614"  :    "Tl-208",
                    "Tl208_583"   :    "Tl-208",
                    "Tl208_861"   :    "Tl-208",
                    "Pa234m_1001" :   "Pa-234m",
                    "Pb214_352"   :   "Pb-214",
                    "Pb214_295"   :    "Pb-214",
                    "Bi214_609"   :    "Bi-214",
                    "Bi214_1378"  :    "Bi-214",
                    "Bi214_1730"  :    "Bi-214",
                    "Bi214_1764"  :    "Bi-214",
                    "Bi214_1238"  :    "Bi-214",
                    "Bi214_2204"  :    "Bi-214",
                    "Bi214_2448"  :    "Bi-214",
                    "Ac228_1588"  :    "Ac-228",
                    "e+e-"        :    "e$^+$e$^-$",
                    "e+e-_511"    :    "e$^+$e$^-$",
                    "Kr85_514"    :    "Kr-85",
                    "Pb212_239"   :    "Pb-212",
                    "Pb214_242"   :    "Pb-214",
                    "Ac228_338"   :    "Ac-228",
                    "Pb214_352"   :    "Pb-214",
                    "Ac228_965"   :    "Ac-228",
                    "Ac228_969"   :    "Ac-228",
                    "Bi214_1120"  :    "Bi-214",
                    "Zn65_1125"   :    "Zn-65"}
gamma_ranges = {    "K42_1525"    :    [[1480,1570], [None,None]],
                    "K40_1461"    :    [[1440,1480], [None,None]],
                    "Co60_1332"   :    [[None,None], [None,None]],
                    "Co60_1173"   :    [[None,None], [None,None]],
                    "Ac228_911"   :    [[None,None], [None,None]],
                    "Bi212_727"   :    [[None,None], [None,None]],
                    "Tl208_2614"  :    [[2580,2650], [None,None]],
                    "Tl208_583"   :    [[553,613], [None,None]],
                    "Tl208_861"   :    [[None,None], [None,None]],
                    "Pa234m_1001" :    [[None,None], [None,None]],
                    "Pb214_352"   :    [[None,None], [None,None]],
                    "Pb214_295"   :    [[None,None], [None,None]],
                    "Bi214_609"   :    [[584,634], [None,None]],
                    "Bi214_1378"  :    [[None,None], [None,None]],
                    "Bi214_1730"  :    [[None,None], [None,None]],
                    "Bi214_1764"  :    [[1740,1790], [None,None]],
                    "Bi214_1238"  :    [[None,None], [None,None]],
                    "Bi214_2204"  :    [[2174,2234], [None,None]],
                    "Bi214_2448"  :    [[None,None], [None,None]],
                    "Ac228_1588"  :    [[None,None], [None,None]],
                    "e+e-"        :    [[None,None], [None,None]],
                    "e+e-_511"    :    [[None,None], [None,None]],
                    "Kr85_514"    :    [[None,None], [None,None]],
                    "Pb212_239"   :    [[None,None], [None,None]],
                    "Pb214_242"   :    [[None,None], [None,None]],
                    "Ac228_338"   :    [[None,None], [None,None]],
                    "Pb214_352"   :    [[None,None], [None,None]],
                    "Ac228_965"   :    [[None,None], [None,None]],
                    "Ac228_969"   :    [[None,None], [None,None]],
                    "Bi214_1120"  :    [[1090,1150], [None,None]],
                    "Zn65_1125"   :    [[None,None], [None,None]],
                    "ROI"         :    [[1930,2190], [None,None]],
                    "all"         :    [[0,5200], [None,None]]}
gamma_energy = {    "K42_1525"    :    1524.7,
                    "K40_1461"    :    1460.8,
                    "Co60_1332"   :    1332.5,
                    "Co60_1173"   :    1173.2,
                    "Ac228_911"   :    911.2,
                    "Bi212_727"   :    727.3,
                    "Tl208_2614"  :    2614.5,
                    "Tl208_583"   :    583.2,
                    "Tl208_861"   :    860.6,
                    "Pa234m_1001" :    1001.0,
                    "Pb214_352"   :    351.9,
                    "Pb214_295"   :    295.2,
                    "Bi214_609"   :    609.3,
                    "Bi214_1378"  :    1377.7,
                    "Bi214_1730"  :    1730,
                    "Bi214_1764"  :    1764.5,
                    "Bi214_1238"  :    1238.1,
                    "Bi214_2204"  :    2204.1,
                    "Bi214_2448"  :    2447.9,
                    "Ac228_1588"  :    1588,
                    "e+e-"        :    511,
                    "e+e-_511"    :    511,
                    "Kr85_514"    :    514,
                    "Pb212_239"   :    238.6,
                    "Pb214_242"   :    242,
                    "Ac228_338"   :    338.3,
                    "Pb214_352"   :    351.9,
                    "Ac228_965"   :    964.8,
                    "Ac228_969"   :    969,
                    "Bi214_1120"  :    1120.3,
                    "Zn65_1125"   :    1125}
# something needed for plotting later on
list_custom_lines = [k for k in gamma_isotope.keys()]
list_custom_lines = sorted(list_custom_lines, key=lambda label: gamma_energy[label])

style = {
  "yerr":False,
    "flow": None,
    "lw": 0.6,
}

def get_hist(obj,range:tuple=(132,4195),bins:int=10):
    """                                                                                                                                                                                                    
    Extract the histogram (hist package object) from the uproot histogram                                                                                                                                  
    Parameters:                                                                                                                                                                                            
        - obj: the uproot histogram                                                                                                                                                                        
        - range: (tuple): the range of bins to select (in keV)                                                                                                                                             
        - bins (int): the (constant) rebinning to apply                                                                                                                                                    
    Returns:                                                                                                                                                                                               
        - hist                                                                                                                                                                                             
    """
    return obj.to_hist()[range[0]:range[1]][hist.rebin(bins)]


def old_vs_p10():
    """Compare previous periods with current p10."""
    parser = argparse.ArgumentParser(description="Script to plot the time dependence of counting rates in L200")
    parser.add_argument("--folder", "-f",type=str,help="Name of output folder",default="")
    parser.add_argument("--output", "-o",type=str,help="Name of output pdf file",default="test")
    parser.add_argument("--input_1", "-i",type=str,help="Name of 1st input root file",default = "/data1/users/tdixon/build_pdf/outputs/l200a-p34678-dataset-v1.0.root")
    parser.add_argument("--input_2", "-I",type=str,help="Name of 2nd input root file (p10)",default ="/data1/users/calgaro/legen-bkg-dev-toby/outputs/l200a-p10-0134-tmp-auto-20240323T101238Z-recomputed-v3.root")
    parser.add_argument("--prodenv_1", "-p1",type=str,help="Path to prodenv of 1st input file (default: same as p10)",default ="/data2/public/prodenv/prod-blind/tmp-auto/inputs")
    parser.add_argument("--prodenv_2", "-p2",type=str,help="Path to prodenv of 2nd input file (p10)",default ="/data2/public/prodenv/prod-blind/tmp-auto/inputs")
    parser.add_argument("--energy", "-e",type=str,help="Energy range to plot",default="0,4000")
    parser.add_argument("--binning", "-b",type=int,help="Binning",default=5)
    parser.add_argument("--spectrum","-s",type=str,help="Spectrum to plot",default="mul_surv")
    parser.add_argument("--dataset","-d",type=str,help="Which group of detectors to plot",default="all")
    parser.add_argument("--variable","-V",type=str,help="Variable binning, argument is the path to the cfg file defualt 'None' and flat binning is used",default=None)
    parser.add_argument("--scale","-S",type=str,help="scale to use, default 'linear'",default="linear")
    parser.add_argument("--gamma","-g",help="line to inspect, default 'None'",default=None)
    parser.add_argument("--usability","-u",help="change detector statuses, default 'None'",default=None)

    args = parser.parse_args()

    folder = args.folder
    path_all = args.input_1
    path = args.input_2
    ref_1 = args.prodenv_1
    ref_2 = args.prodenv_2
    output =args.output
    binning=args.binning
    spectrum =args.spectrum
    scale=args.scale
    energy=args.energy
    dataset=args.dataset
    variable = args.variable
    gamma=args.gamma
    usability = args.usability

    energy_low = int(energy.split(",")[0]) if gamma is None else gamma_ranges[gamma][0][0]
    energy_high = int(energy.split(",")[1]) if gamma is None else gamma_ranges[gamma][0][1]
    if energy_low is None and energy_high is None:
        energy_low = round(gamma_energy[gamma])-20
        energy_high = round(gamma_energy[gamma])+20
    os.makedirs(f"plots_{folder}",exist_ok=True)

    # use the same metadata of p10 to load 
    metadb = LegendMetadata(ref_1)
    chmap = metadb.channelmap(datetime.now())
    runs=metadb.dataprod.config.analysis_runs

    run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)
    exp_0=0
    tot_mass_0=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if p in ["p03","p04","p05","p06","p07","p08"]:
                exp_0+=(times[1]-times[0])*times[2]/(60*60*24*365)

    metadb_2 = LegendMetadata(ref_2)
    chmap_2 = metadb_2.channelmap(datetime.now())
    runs_2=metadb_2.dataprod.config.analysis_runs
    runs_2['p10']=['r006']

    usability_path = "cfg/usability_changes.json"
    with Path(usability_path).open() as f:
        usability = json.load(f)

    if usability is not None:
        run_times_2=utils.get_run_times(metadb_2,runs_2,ac=usability["ac"],off=usability["ac_to_off"],verbose=True)
    else:
        run_times_2=utils.get_run_times(metadb_2,runs_2,ac=[],off=[],verbose=True)
    exp_1=0
    tot_mass_1=0
    for p, _dict in run_times_2.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_1+=(times[1]-times[0])*times[2]/(60*60*24*365)

    print(f"Total exposure in old:", exp_0, f"kg-yr")
    print(f"Total exposure in p10:", exp_1, f"kg-yr")

    ## get binning edges for now the ones for ICPC, this can be also repalced with any other binning
    edges =None
    if (variable is not None):
        with open(variable, 'r') as json_file:
            edges=np.unique(utils.string_to_edges(json.load(json_file)["icpc"]))

    with uproot.open(path_all) as f2:
        h0=utils.get_hist(f2[f"{spectrum}/all"],(energy_low,energy_high),binning,edges)
    with uproot.open(path) as f2:
        h1=utils.get_hist(f2[f"{spectrum}/all"],(energy_low,energy_high),binning,edges)

    for i in range(h0.size-2):
        h0[i]/=exp_0
    for i in range(h1.size-2):
        h1[i]/=exp_1

    fig, axes_full = lps.subplots(1, 1, figsize=(7,5), sharex=True)

    h1.plot(ax=axes_full,**style,color=vset.blue,label="p10 6")
    h0.plot(ax=axes_full,**style,color=vset.orange,label="p3-8")
    axes_full.set_xlabel("Energy [keV]")
    if (variable is None):
        axes_full.set_ylabel(f"counts/({binning} keV kg yr)")
    else:
        axes_full.set_ylabel(f"counts/(keV kg yr)")
    axes_full.set_yscale(scale)

    os.makedirs(f"gammas_{folder}",exist_ok=True)
    if gamma is not None:
        energy_low = gamma_ranges[gamma][0][0]
        energy_high = gamma_ranges[gamma][0][1]
        axes_full.set_xlim(energy_low, energy_high)
        
        if gamma == "ROI":
            g_1st = 2104
            g_2nd = 2119
            plt.axvline(x=g_1st, color='gray', linestyle="--")
            plt.axvline(x=g_2nd, color='gray', linestyle="--")
            plt.axvspan(g_1st-5,g_1st+5, color='grey', alpha=0.3)
            plt.axvspan(2039.06-25,2040+25, color='grey', alpha=1)
            plt.axvspan(g_2nd-5,g_2nd+5, color='grey', alpha=0.3)
            dx=0.8
            plt.text(g_1st+dx, 0.05, "Tl-208", fontsize=9, rotation=90)
            plt.text(g_2nd+dx, 0.05, "Bi-214", fontsize=9, rotation=90)
            axes_full.set_ylim(3e-2, 0.8)

        elif gamma == "all":
            print(energy_low, energy_high)
            for k_idx,k in enumerate(list_custom_lines):
                if gamma_energy[k] < energy_low or gamma_energy[k] > energy_high: continue
                if k == "e+e-_511": continue
                if k == "Pb214_242": 
                    plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                    continue

                plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                scarto = 0
                if k_idx % 2 == 0:
                    scarto = 60
                if k == "e+e-": scarto = 0
                dx = 0.8
                dy = 0.1
                if gamma_energy[k] in [238.6, 511, 295.2, 338.3, 238.6]:
                    dx = -18
                if gamma_energy[k] in [964.8, 1120.3, 1588, 1460.8, 1524.7]:
                    dx = -18
                if gamma_energy[k] in [2204.1,2447.9,1730,1764.5,2614.5]:
                    dx = -33
                    dy = 10
                plt.text(gamma_energy[k]+dx, dy, gamma_isotope[k], fontsize=9, rotation=90)

        else:
            axes_full.set_xlim(energy_low, energy_high)
            plt.axvline(x=gamma_energy[gamma], color='gray', linestyle="--")
            if gamma=="Ac228_965": plt.axvline(x=969, color='gray', linestyle="--")
            if gamma=="Pb212_239": plt.axvline(x=242, color='gray', linestyle="--")
            if gamma=="Ac228_338": plt.axvline(x=351.9, color='gray', linestyle="--")
            if gamma=="e+e-_511":  plt.axvline(x=514, color='gray', linestyle="--")
            if gamma=="Bi214_1120": plt.axvline(x=1125, color='gray', linestyle="--")
        
        axes_full.set_title(f"{spectrum} - {dataset} - {gamma}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"gammas_{folder}/"+output+f"_{gamma}.png")

    axes_full.set_yscale(scale)
    axes_full.set_title(f"{spectrum} - {dataset}")
    axes_full.set_xlim(energy_low,energy_high)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"plots_{folder}/"+output+f"_{energy_low}_{energy_high}.png")


def p10_compare_two_versions():
    """Compare for p10 two versions of files."""
    parser = argparse.ArgumentParser(description="Script to plot the time dependence of counting rates in L200")
    parser.add_argument("--folder", "-f",type=str,help="Name of output folder",default="")
    parser.add_argument("--output", "-o",type=str,help="Name of output pdf file",default="test")
    parser.add_argument("--input_1", "-i",type=str,help="Name of 1st input root file",default = "/data1/users/calgaro/legen-bkg-dev-toby/outputs/l200a-p10-0-tmp-auto-20240323T101238Z-1121603.root")
    parser.add_argument("--input_2", "-I",type=str,help="Name of 2nd input root file",default = "/data1/users/calgaro/legen-bkg-dev-toby/outputs/l200a-p10-0-tmp-auto-20240323T101238Z-recomputed-1121603.root")
    parser.add_argument("--energy", "-e",type=str,help="Energy range to plot",default="0,4000")
    parser.add_argument("--binning", "-b",type=int,help="Binning",default=5)
    parser.add_argument("--spectrum","-s",type=str,help="Spectrum to plot",default="mul_surv")
    parser.add_argument("--dataset","-d",type=str,help="Which group of detectors to plot",default="all")
    parser.add_argument("--variable","-V",type=str,help="Variable binning, argument is the path to the cfg file defualt 'None' and flat binning is used",default=None)
    parser.add_argument("--scale","-S",type=str,help="scale to use, default 'linear'",default="linear")
    parser.add_argument("--gamma","-g",help="line to inspect, default 'None'",default=None)

    args = parser.parse_args()

    folder = args.folder
    path =args.input_2
    path_all = args.input_1
    output =args.output
    binning=args.binning
    spectrum =args.spectrum
    scale=args.scale
    energy=args.energy
    dataset=args.dataset
    variable = args.variable
    gamma=args.gamma
    energy_low = int(energy.split(",")[0]) if gamma is None else gamma_ranges[gamma][0][0]
    energy_high = int(energy.split(",")[1]) if gamma is None else gamma_ranges[gamma][0][1]
    if energy_low is None and energy_high is None:
        energy_low = round(gamma_energy[gamma])-20
        energy_high = round(gamma_energy[gamma])+20
    os.makedirs(f"plots_{folder}",exist_ok=True)

    metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")

    chmap = metadb.channelmap(datetime.now())
    runs=metadb.dataprod.config.analysis_runs

    runs['p10']=['r000']
    run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)
    exp_0=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_0+=(times[1]-times[0])*times[2]/(60*60*24*365)

    runs['p10']=['r000']
    run_times=utils.get_run_times(metadb,runs,verbose=True) # ac=[1108800, 1080005, 1089600],off=[1080000, 1121603]
    exp_1=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_1+=(times[1]-times[0])*times[2]/(60*60*24*365)

    print(f"Total exposure in p10 after:", exp_0, "kg-yr")
    print(f"Total exposure in p10 before:", exp_1, "kg-yr")

    ## get binning edges for now the ones for ICPC, this can be also repalced with any other binning
    edges =None
    if (variable is not None):
        with open(variable, 'r') as json_file:
            edges=np.unique(utils.string_to_edges(json.load(json_file)["icpc"]))

    with uproot.open(path_all) as f2:
        h0=utils.get_hist(f2[f"{spectrum}/all"],(energy_low,energy_high),binning,edges)
    with uproot.open(path) as f2:
        h1=utils.get_hist(f2[f"{spectrum}/all"],(energy_low,energy_high),binning,edges)

    for i in range(h0.size-2):
        h0[i]
    for i in range(h1.size-2):
        h1[i]

    fig, axes_full = lps.subplots(1, 1, figsize=(7,5), sharex=True)

    h0.plot(ax=axes_full,**style,color='orange',label="p10 after")
    h1.plot(ax=axes_full,**style,color=vset.blue,label="p10 before")
    axes_full.set_xlabel("Energy [keV]")
    if (variable is None):
        axes_full.set_ylabel(f"counts/({binning} keV)")
    else:
        axes_full.set_ylabel(f"counts/(keV)")
    axes_full.set_yscale(scale)

    os.makedirs(f"gammas_{folder}",exist_ok=True)
    if gamma is not None:
        energy_low = gamma_ranges[gamma][0][0]
        energy_high = gamma_ranges[gamma][0][1]
        axes_full.set_xlim(energy_low, energy_high)
        
        if gamma == "ROI":
            g_1st = 2104
            g_2nd = 2119
            plt.axvline(x=g_1st, color='gray', linestyle="--")
            plt.axvline(x=g_2nd, color='gray', linestyle="--")
            plt.axvspan(g_1st-5,g_1st+5, color='grey', alpha=0.3)
            plt.axvspan(2039.06-25,2040+25, color='grey', alpha=1)
            plt.axvspan(g_2nd-5,g_2nd+5, color='grey', alpha=0.3)
            dx=0.8
            plt.text(g_1st+dx, 0.05, "Tl-208", fontsize=9, rotation=90)
            plt.text(g_2nd+dx, 0.05, "Bi-214", fontsize=9, rotation=90)
            axes_full.set_ylim(3e-2, 0.8)

        elif gamma == "all":
            for k_idx,k in enumerate(list_custom_lines):
                if gamma_energy[k] < energy_low or gamma_energy[k] > energy_high: continue
                if k == "e+e-_511": continue
                if k == "Pb214_242": 
                    plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                    continue

                plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                if k=="Ac228_965": plt.axvline(x=969, color='gray', linestyle="--")
                if k=="Pb212_239": plt.axvline(x=242, color='gray', linestyle="--")
                if k=="Ac228_338": plt.axvline(x=351.9, color='gray', linestyle="--")
                if k=="e+e-_511":  plt.axvline(x=514, color='gray', linestyle="--")
                if k=="Bi214_1120": plt.axvline(x=1125, color='gray', linestyle="--")
                scarto = 0
                if k_idx % 2 == 0:
                    scarto = 60
                if k == "e+e-": scarto = 0
                dx = 0.8
                dy = 300
                if gamma_energy[k] in [238.6, 511, 964.8, 1120.3, 295.2, 338.3, 1588, 1460.8, 1524.7, 238.6]:
                    dx = -18
                    dy = 30
                if gamma_energy[k] in [2614.5]:
                    dx = -33
                    dy = 2
                plt.text(gamma_energy[k]+dx, dy, gamma_isotope[k], fontsize=9, rotation=90)

        else:
            axes_full.set_xlim(energy_low, energy_high)
            plt.axvline(x=gamma_energy[gamma], color='gray', linestyle="--")
        
        axes_full.set_title(f"{spectrum} - {dataset} - {gamma}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"gammas_{folder}/"+output+f"_{gamma}.png")
        exit()

    axes_full.set_yscale(scale)
    axes_full.set_title(f"{spectrum} - {dataset}")
    axes_full.set_xlim(energy_low,energy_high)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"plots_{folder}/"+output+f"_{energy_low}_{energy_high}.png")

    fig, axes_full = lps.subplots(1, 1, figsize=(7,5), sharex=True)
    diff = h1 - h0
    bin_contents, bin_edges = diff.to_numpy()
    plt.bar(bin_edges[:-1], bin_contents, width=np.diff(bin_edges), color="red")
    plt.xlabel('Energy [keV]')
    plt.ylabel('Difference of spectra')
    plt.yscale('log')
    plt.title('Difference in p10: before - after')
    plt.savefig(f"plots_{folder}/"+output+f"_difference_{energy_low}_{energy_high}.png")


def all_dets():
    """Compare previous periods with current p10."""
    parser = argparse.ArgumentParser(description="Script to plot the time dependence of counting rates in L200")
    parser.add_argument("--output", "-o",type=str,help="Name of output pdf file",default="test")
    parser.add_argument("--input_1", "-i",type=str,help="Name of 1st input root file",default = "/data1/users/calgaro/legend-bkg-scripts-dev-sofia/outputs/l200a-p10-013-tmp-auto-all.root")
    parser.add_argument("--input_2", "-I",type=str,help="Name of 2nd input root file",default ="/data1/users/calgaro/legen-bkg-dev-toby/outputs/l200a-p10-4-tmp-auto-20240323T101238Z-recomputed-v3.root")
    parser.add_argument("--energy", "-e",type=str,help="Energy range to plot",default="0,4000")
    parser.add_argument("--binning", "-b",type=int,help="Binning",default=5)
    parser.add_argument("--spectrum","-s",type=str,help="Spectrum to plot",default="mul_surv")
    parser.add_argument("--dataset","-d",type=str,help="Which group of detectors to plot",default="all")
    parser.add_argument("--variable","-V",type=str,help="Variable binning, argument is the path to the cfg file defualt 'None' and flat binning is used",default=None)
    parser.add_argument("--scale","-S",type=str,help="scale to use, default 'linear'",default="linear")
    parser.add_argument("--gamma","-g",type=str,help="line to inspect, default 'None'",default=None)

    args = parser.parse_args()

    path =args.input_2
    path_all = args.input_1
    output =args.output
    binning=args.binning
    spectrum =args.spectrum
    scale=args.scale
    energy=args.energy
    dataset=args.dataset
    variable = args.variable
    gamma=args.gamma
    energy_low = int(energy.split(",")[0]) if gamma is None else gamma_ranges[gamma][0][0]
    energy_high = int(energy.split(",")[1]) if gamma is None else gamma_ranges[gamma][0][1]
    os.makedirs("plots",exist_ok=True)

    metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")

    chmap = metadb.channelmap(datetime.now())
    runs=metadb.dataprod.config.analysis_runs

    runs['p10']=['r000']
    run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)
    exp_0=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_0+=(times[1]-times[0])*times[2]/(60*60*24*365)

    runs['p10']=['r001']
    run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)
    exp_1=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_1+=(times[1]-times[0])*times[2]/(60*60*24*365)

    runs['p10']=['r000','r001','r003','r004']#['r004']
    run_times=utils.get_run_times(metadb,runs,ac=[1108800, 1080005, 1089600],off=[1080000, 1121603],verbose=True)
    exp_2=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_2+=(times[1]-times[0])*times[2]/(60*60*24*365)

    runs['p10']=['r003']
    run_times=utils.get_run_times(metadb,runs,verbose=True)
    exp_3=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_3+=(times[1]-times[0])*times[2]/(60*60*24*365)

    print(f"Total exposure in r000:", exp_0, "kg-yr")
    print(f"Total exposure in r001:", exp_1, "kg-yr")
    print(f"Total exposure in r003:", exp_3, "kg-yr")
    print(f"Total exposure in r004:", exp_2, "kg-yr")

    ## get binning edges for now the ones for ICPC, this can be also repalced with any other binning
    edges =None
    if (variable is not None):
        with open(variable, 'r') as json_file:
            edges=np.unique(utils.string_to_edges(json.load(json_file)["icpc"]))

    with uproot.open(path_all) as f2:
        h0=utils.get_hist(f2[f"{spectrum}/p10_r000"],(energy_low,energy_high),binning,edges)
    with uproot.open(path_all) as f2:
        h1=utils.get_hist(f2[f"{spectrum}/p10_r001"],(energy_low,energy_high),binning,edges)
    with uproot.open(path) as f2:
        h2=utils.get_hist(f2[f"{spectrum}/all"],(energy_low,energy_high),binning,edges)
    with uproot.open(path_all) as f2:
        h3=utils.get_hist(f2[f"{spectrum}/p10_r003"],(energy_low,energy_high),binning,edges)

    for i in range(h0.size-2):
        h0[i]/=exp_0
    for i in range(h1.size-2):
        h1[i]/=exp_1
    for i in range(h2.size-2):
        h2[i]/=exp_2
    for i in range(h3.size-2):
        h3[i]/=exp_3

    fig, axes_full = lps.subplots(1, 1, figsize=(7,5), sharex=True)

    h0.plot(ax=axes_full,**style,color='g',label="p10 0")
    h1.plot(ax=axes_full,**style,color='r',label="p10 1")
    h3.plot(ax=axes_full, **style,color='orange',label=f"p10 3")
    h2.plot(ax=axes_full, **style,color=vset.blue,label=f"p10 4")
    axes_full.set_xlabel("Energy [keV]")
    if (variable is None):
        axes_full.set_ylabel(f"counts/({binning} keV kg yr)")
    else:
        axes_full.set_ylabel(f"counts/(keV kg yr)")
    axes_full.set_yscale(scale)

    os.makedirs("gammas",exist_ok=True)
    if gamma is not None:
        energy_low = gamma_ranges[gamma][0][0]
        energy_high = gamma_ranges[gamma][0][1]
        axes_full.set_xlim(energy_low, energy_high)
        
        if gamma == "ROI":
            g_1st = 2104
            g_2nd = 2119
            plt.axvline(x=g_1st, color='gray', linestyle="--")
            plt.axvline(x=g_2nd, color='gray', linestyle="--")
            plt.axvspan(g_1st-5,g_1st+5, color='grey', alpha=0.3)
            plt.axvspan(2039.06-25,2040+25, color='grey', alpha=1)
            plt.axvspan(g_2nd-5,g_2nd+5, color='grey', alpha=0.3)
            dx=0.8
            plt.text(g_1st+dx, 0.05, "Tl-208", fontsize=9, rotation=90)
            plt.text(g_2nd+dx, 0.05, "Bi-214", fontsize=9, rotation=90)
            axes_full.set_ylim(3e-2, 0.8)

        elif gamma == "all":
            for k_idx,k in enumerate(list_custom_lines):
                if gamma_energy[k] < energy_low or gamma_energy[k] > energy_high: continue
                if k == "e+e-_511": continue
                if k == "Pb214_242": 
                    plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                    continue

                plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                if k=="Ac228_965": plt.axvline(x=969, color='gray', linestyle="--")
                if k=="Pb212_239": plt.axvline(x=242, color='gray', linestyle="--")
                if k=="Ac228_338": plt.axvline(x=351.9, color='gray', linestyle="--")
                if k=="e+e-_511":  plt.axvline(x=514, color='gray', linestyle="--")
                if k=="Bi214_1120": plt.axvline(x=1125, color='gray', linestyle="--")
                scarto = 0
                if k_idx % 2 == 0:
                    scarto = 60
                if k == "e+e-": scarto = 0
                dx = 0.8
                dy = 300
                if gamma_energy[k] in [238.6, 511, 964.8, 1120.3, 295.2, 338.3, 1588, 1460.8, 1524.7, 238.6]:
                    dx = -18
                    dy = 30
                if gamma_energy[k] in [2614.5]:
                    dx = -33
                    dy = 2
                plt.text(gamma_energy[k]+dx, dy, gamma_isotope[k], fontsize=9, rotation=90)

        else:
            plt.axvline(x=gamma_energy[gamma], color='gray', linestyle="--")
        
        axes_full.set_title(f"{spectrum} - {dataset} - {gamma}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("gammas/"+output+f"_{gamma}.png")
        exit()

    axes_full.set_yscale(scale)
    axes_full.set_title(f"{spectrum} - {dataset}")
    axes_full.set_xlim(energy_low,energy_high)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/"+output+f"_{energy_low}_{energy_high}.png")


def all_dets_per_run():
    """Compare one run of a given period with previous period."""
    parser = argparse.ArgumentParser(description="Script to plot the time dependence of counting rates in L200")
    parser.add_argument("--folder", "-f",type=str,help="Name of output folder",default="")
    parser.add_argument("--output", "-o",type=str,help="Name of output pdf file",default="test")
    parser.add_argument("--input_1", "-i",type=str,help="Name of 1st input root file",default = "/data1/users/calgaro/legend-bkg-scripts-dev-sofia/outputs/l200a-p10-013-tmp-auto-all.root")
    parser.add_argument("--input_2", "-I",type=str,help="Name of 2nd input root file",default ="/data1/users/calgaro/legend-bkg-scripts-dev-sofia/outputs/l200a-p10-4-tmp-auto.root")
    parser.add_argument("--energy", "-e",type=str,help="Energy range to plot",default="0,4000")
    parser.add_argument("--binning", "-b",type=int,help="Binning",default=5)
    parser.add_argument("--spectrum","-s",type=str,help="Spectrum to plot",default="mul_surv")
    parser.add_argument("--dataset","-d",type=str,help="Which group of detectors to plot",default="all")
    parser.add_argument("--variable","-V",type=str,help="Variable binning, argument is the path to the cfg file defualt 'None' and flat binning is used",default=None)
    parser.add_argument("--scale","-S",type=str,help="scale to use, default 'log'",default="linear")
    parser.add_argument("--gamma","-g",type=str,help="line to inspect, default 'None'",default=None)

    args = parser.parse_args()

    path =args.input_2
    path_all = args.input_1
    output = args.output
    folder = args.folder
    binning=args.binning
    spectrum =args.spectrum
    scale=args.scale
    energy=args.energy
    dataset=args.dataset
    variable = args.variable
    gamma=args.gamma
    energy_low = int(energy.split(",")[0]) if gamma is None else gamma_ranges[gamma][0][0]
    energy_high = int(energy.split(",")[1]) if gamma is None else gamma_ranges[gamma][0][1]


    metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")

    chmap = metadb.channelmap(datetime.now())
    runs=metadb.dataprod.config.analysis_runs

    runs['p10']=['r000']
    run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)
    exp_0=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_0+=(times[1]-times[0])*times[2]/(60*60*24*365)

    runs['p10']=['r001']
    run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)
    exp_1=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_1+=(times[1]-times[0])*times[2]/(60*60*24*365)

    runs['p10']=['r004']
    run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)#ac=[1108800, 1080005, 1089600],off=[1080000, 1121603],verbose=True)
    exp_2=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_2+=(times[1]-times[0])*times[2]/(60*60*24*365)

    runs['p10']=['r003']
    run_times=utils.get_run_times(metadb,runs,verbose=True)
    exp_3=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_3+=(times[1]-times[0])*times[2]/(60*60*24*365)


    ########### manual inclusion of the last part of p10 #################
    runs['p10']=['r005']
    run_times=utils.get_run_times(metadb,runs,verbose=True)
    exp_5=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_5+=(times[1]-times[0])*times[2]/(60*60*24*365)
    runs['p10']=['r006']
    run_times=utils.get_run_times(metadb,runs,verbose=True)
    exp_6=0
    for p, _dict in run_times.items():
        for r, times in _dict.items():
            if (p=="p10"):
                exp_6+=(times[1]-times[0])*times[2]/(60*60*24*365)

    print(f"Total exposure in r000:", exp_0, "kg-yr")
    print(f"Total exposure in r001:", exp_1, "kg-yr")
    print(f"Total exposure in r003:", exp_3, "kg-yr")
    print(f"Total exposure in r004:", exp_2, "kg-yr")
    print(f"Total exposure in r005:", exp_5, "kg-yr")
    print(f"Total exposure in r006:", exp_6, "kg-yr")

    ## get binning edges for now the ones for ICPC, this can be also repalced with any other binning
    edges =None
    if (variable is not None):
        with open(variable, 'r') as json_file:
            edges=np.unique(utils.string_to_edges(json.load(json_file)["icpc"]))

    with uproot.open(path_all) as f2:
        h0=utils.get_hist(f2[f"{spectrum}/p10_r000"],(energy_low,energy_high),binning,edges)
    with uproot.open(path_all) as f2:
        h1=utils.get_hist(f2[f"{spectrum}/p10_r001"],(energy_low,energy_high),binning,edges)
    with uproot.open(path) as f2:
        h2=utils.get_hist(f2[f"{spectrum}/p10_r004"],(energy_low,energy_high),binning,edges)
    with uproot.open(path_all) as f2:
        h3=utils.get_hist(f2[f"{spectrum}/p10_r003"],(energy_low,energy_high),binning,edges)

    with uproot.open("outputs/l200a-p10-5-tmp-auto.root") as f2:
        h5=utils.get_hist(f2[f"{spectrum}/p10_r005"],(energy_low,energy_high),binning,edges)
    with uproot.open("outputs/l200a-p10-6-tmp-auto.root") as f2:
        h6=utils.get_hist(f2[f"{spectrum}/p10_r006"],(energy_low,energy_high),binning,edges)

    for i in range(h0.size-2):
        h0[i]/=exp_0
    for i in range(h1.size-2):
        h1[i]/=exp_1
    for i in range(h2.size-2):
        h2[i]/=exp_2
    for i in range(h3.size-2):
        h3[i]/=exp_3
    for i in range(h5.size-2):
        h5[i]/=exp_5
    for i in range(h6.size-2):
        h6[i]/=exp_6

    fig, axes_full = lps.subplots(1, 1, figsize=(7,5), sharex=True)

    h0.plot(ax=axes_full,**style,color='g',label="p10 0")
    h1.plot(ax=axes_full,**style,color='r',label="p10 1")
    h3.plot(ax=axes_full, **style,color='orange',label=f"p10 3")
    h2.plot(ax=axes_full, **style,color=vset.blue,label=f"p10 4")
    h5.plot(ax=axes_full, **style,color='k',label=f"p10 5")
    h6.plot(ax=axes_full, **style,color='pink',label=f"p10 6")
    axes_full.set_xlabel("Energy [keV]")
    if (variable is None):
        axes_full.set_ylabel(f"counts/({binning} keV kg yr)")
    else:
        axes_full.set_ylabel(f"counts/(keV kg yr)")
    axes_full.set_yscale(scale)

    os.makedirs(folder,exist_ok=True)
    if gamma is not None:
        energy_low = gamma_ranges[gamma][0][0]
        energy_high = gamma_ranges[gamma][0][1]
        
        if gamma == "ROI":
            axes_full.set_xlim(energy_low, energy_high)
            g_1st = 2104
            g_2nd = 2119
            plt.axvline(x=g_1st, color='gray', linestyle="--")
            plt.axvline(x=g_2nd, color='gray', linestyle="--")
            plt.axvspan(g_1st-5,g_1st+5, color='grey', alpha=0.3)
            plt.axvspan(2039.06-25,2040+25, color='grey', alpha=1)
            plt.axvspan(g_2nd-5,g_2nd+5, color='grey', alpha=0.3)
            dx=0.8
            plt.text(g_1st+dx, 0.05, "Tl-208", fontsize=9, rotation=90)
            plt.text(g_2nd+dx, 0.05, "Bi-214", fontsize=9, rotation=90)
            axes_full.set_ylim(3e-2, 0.8)

        elif gamma == "all":
            axes_full.set_xlim(energy_low, energy_high)
            for k_idx,k in enumerate(list_custom_lines):
                if gamma_energy[k] < energy_low or gamma_energy[k] > energy_high: continue
                if k == "e+e-_511": continue
                if k == "Pb214_242": 
                    plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                    continue

                plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                if k=="Ac228_965": plt.axvline(x=969, color='gray', linestyle="--")
                if k=="Pb212_239": plt.axvline(x=242, color='gray', linestyle="--")
                if k=="Ac228_338": plt.axvline(x=351.9, color='gray', linestyle="--")
                if k=="e+e-_511":  plt.axvline(x=514, color='gray', linestyle="--")
                if k=="Bi214_1120": plt.axvline(x=1125, color='gray', linestyle="--")
                scarto = 0
                if k_idx % 2 == 0:
                    scarto = 60
                if k == "e+e-": scarto = 0
                dx = 0.8
                dy = 300
                if gamma_energy[k] in [238.6, 511, 964.8, 1120.3, 295.2, 338.3, 1588, 1460.8, 1524.7, 238.6]:
                    dx = -18
                    dy = 30
                if gamma_energy[k] in [2614.5]:
                    dx = -33
                    dy = 2
                plt.text(gamma_energy[k]+dx, dy, gamma_isotope[k], fontsize=9, rotation=90)

        else:
            if energy_low is None and energy_high is None:
                energy_low = round(gamma_energy[gamma])-10
                energy_high = round(gamma_energy[gamma])+10
            axes_full.set_xlim(energy_low, energy_high)
            plt.axvline(x=gamma_energy[gamma], color='gray', linestyle="--")
        
        axes_full.set_title(f"{spectrum} - {dataset} - {gamma}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{folder}/"+output+f"_{gamma}.png")
        exit()

    axes_full.set_yscale(scale)
    axes_full.set_title(f"{spectrum} - {dataset}")
    axes_full.set_xlim(energy_low,energy_high)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{folder}/"+output+f"_{energy_low}_{energy_high}.png")


def single_dets():
    """Compare individual detectors per period."""
    parser = argparse.ArgumentParser(description="Script to plot the time dependence of counting rates in L200")
    parser.add_argument("--output", "-o",type=str,help="Name of output pdf file",default="test")
    parser.add_argument("--input_1", "-i",type=str,help="Name of 1st input root file",default = "/data1/users/calgaro/legend-bkg-scripts-dev-sofia/outputs/l200a-p10-013-tmp-auto-all.root")
    parser.add_argument("--input_2", "-I",type=str,help="Name of 2nd input root file",default ="/data1/users/calgaro/legend-bkg-scripts-dev-sofia/outputs/l200a-p10-4-tmp-auto-20240323T101238Z-recomputed-v2.root")
    parser.add_argument("--energy", "-e",type=str,help="Energy range to plot",default="0,4000")
    parser.add_argument("--binning", "-b",type=int,help="Binning",default=5)
    parser.add_argument("--spectrum","-s",type=str,help="Spectrum to plot",default="mul_surv")
    parser.add_argument("--variable","-V",type=str,help="Variable binning, argument is the path to the cfg file defualt 'None' and flat binning is used",default=None)
    parser.add_argument("--scale","-S",type=str,help="scale to use, default 'linear'",default="linear")
    parser.add_argument("--gamma","-g",type=str,help="line to inspect, default 'None'",default=None)

    args = parser.parse_args()

    path =args.input_2
    path_all = args.input_1
    output =args.output
    binning=args.binning
    spectrum =args.spectrum
    scale=args.scale
    energy=args.energy
    variable = args.variable
    gamma=args.gamma
    energy_low = int(energy.split(",")[0]) if gamma is None else gamma_ranges[gamma][0][0]
    energy_high = int(energy.split(",")[1]) if gamma is None else gamma_ranges[gamma][0][1]
    os.makedirs("single_dets",exist_ok=True)

    metadb = LegendMetadata("/data2/public/prodenv/prod-blind/tmp-auto/inputs")

    chmap = metadb.channelmap(datetime.now())
    runs=metadb.dataprod.config.analysis_runs

    runinfo = json.load(open('/data1/users/calgaro/runinfo_new.json'))
    ch = metadb.channelmap(runinfo['p10']['r001']["phy"]["start_key"])
    geds_list = [ _dict["daq"]["rawid"] for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and 
                    ch[_name]["analysis"]["usability"] in ["on","no_psd"]]
    geds_name = [ _name for _name, _dict in ch.items() if ch[_name]["system"] == "geds" and 
                    ch[_name]["analysis"]["usability"] in ["on","no_psd"]]

    for idx,det in enumerate(geds_list):
        fig, axes_full = lps.subplots(1, 1, figsize=(7,5), sharex=True)

        runs['p10']=['r000','r001','r003']
        run_times=utils.get_run_times(metadb,runs,ac=[],off=[],verbose=True)
        exp_0=0
        for p, _dict in run_times.items():
            for r, times in _dict.items():
                if p=='p10':
                    exp_0+=(times[1]-times[0])*times[2]/(60*60*24*365)
        print(f"Total exposure in r000/1/3:", exp_0, "kg-yr")

        runs['p10']=['r004']
        run_times=utils.get_run_times(metadb,runs,ac=[1108800, 1080005, 1089600],off=[1080000, 1121603],verbose=True) 
        exp_1=0
        for p, _dict in run_times.items():
            for r, times in _dict.items():
                if p=='p10':
                    exp_1+=(times[1]-times[0])*times[2]/(60*60*24*365)
        print(f"Total exposure in r004:", exp_1, "kg-yr")

        ## get binning edges for now the ones for ICPC, this can be also repalced with any other binning
        edges =None
        if (variable is not None):
            with open(variable, 'r') as json_file:
                edges=np.unique(utils.string_to_edges(json.load(json_file)["icpc"]))

        # 0+1+3
        with uproot.open(path_all) as f2:
            h0=utils.get_hist(f2[f"{spectrum}/ch{det}"],(energy_low,energy_high),binning,edges)
        for i in range(h0.size-2):
            h0[i]/=exp_0

        # 4 only
        with uproot.open(path) as f2:
            h1=utils.get_hist(f2[f"{spectrum}/ch{det}"],(energy_low,energy_high),binning,edges)
        for i in range(h0.size-2):
            h1[i]/=exp_1

        h0.plot(ax=axes_full,**style,color='g',label="p10 0-1-3")
        h1.plot(ax=axes_full,**style,color='red',label="p10 4")
        
        axes_full.set_xlabel("Energy [keV]")
        if (variable is None):
            axes_full.set_ylabel(f"counts/({binning} keV kg yr)")
        else:
            axes_full.set_ylabel(f"counts/(keV kg yr)")
        axes_full.set_yscale(scale)

        axes_full.set_yscale(scale)
        axes_full.set_title(f"{spectrum} - {geds_name[idx]}")
        axes_full.set_xlim(energy_low,energy_high)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("single_dets/"+output+f"_{geds_name[idx]}_{energy_low}_{energy_high}_{det}.png")

    
    """
    if gamma is not None:
        energy_low = gamma_ranges[gamma][0][0]
        energy_high = gamma_ranges[gamma][0][1]
        axes_full.set_xlim(energy_low, energy_high)
        
        if gamma == "ROI":
            g_1st = 2104
            g_2nd = 2119
            plt.axvline(x=g_1st, color='gray', linestyle="--")
            plt.axvline(x=g_2nd, color='gray', linestyle="--")
            plt.axvspan(g_1st-5,g_1st+5, color='grey', alpha=0.3)
            plt.axvspan(2039.06-25,2040+25, color='grey', alpha=1)
            plt.axvspan(g_2nd-5,g_2nd+5, color='grey', alpha=0.3)
            dx=0.8
            plt.text(g_1st+dx, 0.05, "Tl-208", fontsize=9, rotation=90)
            plt.text(g_2nd+dx, 0.05, "Bi-214", fontsize=9, rotation=90)
            axes_full.set_ylim(3e-2, 0.8)

        elif gamma == "all":
            for k_idx,k in enumerate(list_custom_lines):
                if gamma_energy[k] < energy_low or gamma_energy[k] > energy_high: continue
                if k == "e+e-_511": continue
                if k == "Pb214_242": 
                    plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                    continue

                plt.axvline(x=gamma_energy[k], color='gray', linestyle="--")
                if k=="Ac228_965": plt.axvline(x=969, color='gray', linestyle="--")
                if k=="Pb212_239": plt.axvline(x=242, color='gray', linestyle="--")
                if k=="Ac228_338": plt.axvline(x=351.9, color='gray', linestyle="--")
                if k=="e+e-_511":  plt.axvline(x=514, color='gray', linestyle="--")
                if k=="Bi214_1120": plt.axvline(x=1125, color='gray', linestyle="--")
                scarto = 0
                if k_idx % 2 == 0:
                    scarto = 60
                if k == "e+e-": scarto = 0
                dx = 0.8
                dy = 300
                if gamma_energy[k] in [238.6, 511, 964.8, 1120.3, 295.2, 338.3, 1588, 1460.8, 1524.7, 238.6]:
                    dx = -18
                    dy = 30
                if gamma_energy[k] in [2614.5]:
                    dx = -33
                    dy = 2
                plt.text(gamma_energy[k]+dx, dy, gamma_isotope[k], fontsize=9, rotation=90)

        else:
            plt.axvline(x=gamma_energy[gamma], color='gray', linestyle="--")
        
        axes_full.set_title(f"{spectrum} - {dataset} - {gamma}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig("gammas/"+output+f"_{gamma}.png")
        exit()
    """
    