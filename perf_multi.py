#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, math, shutil, subprocess, threading, concurrent.futures as cf, numpy as np, ROOT
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # batch mode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter  # for % ticks


# ============================================================
# === Plotting style & configuration =========================
# ============================================================

PLOT_CFG = {
    "Pt": {
        "ylabel": r"$\sigma(p_T)/p_T$",
        "xlabel_pt": r"$p_T$ [GeV]",
        "xlabel_p_mag": r"$p$ [GeV]",
        "xlabel_t": r"$\theta$ [deg]",
        "ymin": 0e-3,
        # "ymax": 3e-2,
        "ymax": 2e-1,
        "logx_pt": True,
        "logx_p_mag": True,
        "logx_t": False,
        "logy": True,
        "ratio_ymin_pt": 0.5,
        "ratio_ymax_pt": 1.5,
        "ratio_ymin_p_mag": 0.7,
        "ratio_ymax_p_mag": 1.1,
        "ratio_ymin_t": 0.5,
        "ratio_ymax_t": 1.5,
    },

    "D": {
        "ylabel": r"$\sigma(d_0)$ [$\mu$m]",
        "xlabel_pt": r"$p_T$ [GeV]",
        "xlabel_p_mag": r"$p$ [GeV]",
        "xlabel_t": r"$\theta$ [deg]",
        "ymin": 1,
        "ymax": 400.0,
        "logx_pt": False,
        "logx_p_mag": False,
        "logx_t": False,
        "logy": True,
        "ratio_ymin_pt": 0.5,
        "ratio_ymax_pt": 1.5,
        "ratio_ymin_p_mag": 0.7,
        "ratio_ymax_p_mag": 1.1,
        "ratio_ymin_t": 0.5,
        "ratio_ymax_t": 1.5,
    },

    "Z0": {
        "ylabel": r"$\sigma(z_0)$ [$\mu$m]",
        "xlabel_pt": r"$p_T$ [GeV]",
        "xlabel_p_mag": r"$p$ [GeV]",
        "xlabel_t": r"$\theta$ [deg]",
        "ymin": 1,
        "ymax": 2000.0,
        "logx_pt": False,
        "logx_p_mag": False,
        "logx_t": False,
        "logy": True,
        "ratio_ymin_pt": 0.5,
        "ratio_ymax_pt": 1.5,
        "ratio_ymin_p_mag": 0.7,
        "ratio_ymax_p_mag": 1.1,
        "ratio_ymin_t": 0.5,
        "ratio_ymax_t": 1.5,
    },

    "Theta": {
        "ylabel": r"$\sigma(\theta)$",
        "xlabel_pt": r"$p_T$ [GeV]",
        "xlabel_p_mag": r"$p$ [GeV]",
        "xlabel_t": r"$\theta$ [deg]",
        "ymin": 1e-6,
        "ymax": 3e-3,
        "logx_pt": False,
        "logx_p_mag": False,
        "logx_t": False,
        "logy": True,
        "ratio_ymin_pt": 0.5,
        "ratio_ymax_pt": 1.5,
        "ratio_ymin_p_mag": 0.5,
        "ratio_ymax_p_mag": 1.5,
        "ratio_ymin_t": 0.5,
        "ratio_ymax_t": 1.5,
    },

    "Phi": {
        "ylabel": r"$\sigma(\varphi_0)$",
        "xlabel_pt": r"$p_T$ [GeV]",
        "xlabel_p_mag": r"$p$ [GeV]",
        "xlabel_t": r"$\theta$ [deg]",
        "ymin": 5e-6,
        "ymax": 3e-2,
        "logx_pt": False,
        "logx_p_mag": False,
        "logx_t": False,
        "logy": True,
        "ratio_ymin_pt": 0.5,
        "ratio_ymax_pt": 1.5,
        "ratio_ymin_p_mag": 0.5,
        "ratio_ymax_p_mag": 1.5,
        "ratio_ymin_t": 0.5,
        "ratio_ymax_t": 1.5,
    },

}

# Geometry (r-z) plot configuration (in-file, not via argparse)
GEO_CFG = {
    "zmax": 3,    # override z-axis limit [m]. None -> auto (from layer extents + margin)
    "rmax": 3,    # override r-axis limit [m]. None -> auto (from layer extents + margin)
    "dpi":  200,
}

# Zoomed vertex region geometry plot configuration
GEO_VTX_CFG = {
    "zmax": 0.40,  # zoom to vertex region in z [m]
    "rmax": 0.06,  # zoom to vertex region in r [m]
    "dpi":  200,
}

# Material-budget plot configuration (in-file, not via argparse)
MAT_CFG = {
    "bins": 100,              # number of cos(theta) bins
    "cmin": -0.99,            # min cos(theta)
    "cmax": 0.99,             # max cos(theta)
    "ymax": 30,             # optional y-limit (percent). None -> auto
    "ylabel": r"Material budget [$\%\;X_0$]",
    "xlabel": r"$\cos\theta$",
    "legend_title": "Layers",
    "legend_ncol": 1,
    "legend_bbox": (0.5, 0.90),  # (x,y) in axes fraction
    "ignore_labels_containing": ["MAG", "ARC", "ECAL"],  # case-insensitive substrings to ignore
    "percent_ticks": True,     # show % in tick labels
    "dpi": 200,
}

# LaTeX aesthetics (applies to ALL plots, including geometry & material)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.4,
})

# ============================================================
# === Scan grids (edit here) =================================
# ============================================================

# theta values used in the plots vs pT (fixed list)
THETA_SET_FOR_PTSCAN = [15.0, 45.0, 90.0]  # deg
# THETA_SET_FOR_PTSCAN = [90.0]  # deg

# theta values used in the plots vs |p| (fixed list)
THETA_SET_FOR_PSCAN = [15.0, 45.0, 90.0]  # deg

# pT scan (for plots vs pT): range + number of bins (spacing auto from PLOT_CFG[*].logx_pt)
PT_RANGE_PTSCAN  = (0.1, 100.0)   # GeV
PT_NBINS_PTSCAN  = 396

# |p| scan (for plots vs |p|): range + number of bins (spacing auto from PLOT_CFG[*].logx_p_mag)
P_RANGE_PSCAN  = (0.1, 190.0)   # GeV
P_NBINS_PSCAN  = 396

# theta scan (for plots vs theta): range + number of bins (spacing auto from PLOT_CFG[*].logx_t; default linear)
THETA_RANGE_TSCAN = (10.0, 90.0)  # deg
THETA_NBINS_TSCAN = 320

# pT values used IN theta-scan (for plots vs theta): explicit list
PT_LIST_TSCAN = [1.0, 10.0, 100]   # GeV

# |p| values used IN theta-scan (for plots vs theta): explicit list
P_LIST_TSCAN = [1.0, 45.6, 80.0, 120.0, 182.5]   # GeV

PARAMS = ["Pt", "D", "Z0", "Theta", "Phi"]  # custom order
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1, 1, 1))]  # cycles for multiple detectors
progress_lock = threading.Lock()


# choose a stable palette
TAB10 = plt.get_cmap("tab10").colors

# fixed color per theta (used in vs-p plots)
COLOR_BY_THETA = {th: TAB10[i % len(TAB10)]
                  for i, th in enumerate(sorted(THETA_SET_FOR_PTSCAN))}

# fixed color per pT value (used in vs-theta plots)
COLOR_BY_PTVAL  = {ptv: TAB10[i % len(TAB10)]
                  for i, ptv in enumerate(sorted(PT_LIST_TSCAN))}

# fixed color per |p| value (used in vs-theta plots)
COLOR_BY_PVAL  = {p_magv: TAB10[i % len(TAB10)]
                  for i, p_magv in enumerate(sorted(P_LIST_TSCAN))}

# ============================================================
# === ROOT utilities =========================================
# ============================================================

def ensure_dir(path): os.makedirs(path, exist_ok=True)
def set_root_batch(): ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)

def load_cpp(inc_dir, compile_units):
    inc_dir = os.path.abspath(os.path.expanduser(inc_dir))
    # Detect project root if inc_dir ends with .../examples/classes
    proj_root = inc_dir
    parent = os.path.dirname(inc_dir)
    if os.path.basename(inc_dir) == "classes" and os.path.basename(parent) == "examples":
        proj_root = os.path.dirname(parent)

    # Add both include roots
    ROOT.gSystem.AddIncludePath(f"-I{inc_dir}")
    ROOT.gSystem.AddIncludePath(f"-I{proj_root}")
    ROOT.gROOT.ProcessLine(f'.I {inc_dir}')
    ROOT.gROOT.ProcessLine(f'.I {proj_root}')

    if compile_units:
        # Compile the .cc files using absolute paths
        ROOT.gROOT.ProcessLine(f'.L {os.path.join(inc_dir,"SolGeom.cc")}+')
        ROOT.gROOT.ProcessLine(f'.L {os.path.join(inc_dir,"TrkUtil.cc")}+')
        ROOT.gROOT.ProcessLine(f'.L {os.path.join(inc_dir,"SolTrack.cc")}+')
        ROOT.gROOT.ProcessLine(f'.L {os.path.join(inc_dir,"KalmanCk.cc")}+')

    ROOT.gInterpreter.Declare('#include "SolGeom.h"')
    ROOT.gInterpreter.Declare('#include "SolTrack.h"')

def polar_to_components(p_mag, theta_deg):
    th = math.radians(theta_deg)
    return (p_mag*math.sin(th), 0.0, p_mag*math.cos(th))


def eval_point(p_mag, theta_deg, geom, npoints, minmeas, doKalman, doRes, doMS, verbose=False):
    if verbose: print(f"    -> p={p_mag:.3g} GeV, theta={theta_deg:.1f}°")
    x0 = ROOT.TVector3(0.,0.,0.)
    px,py,pz = polar_to_components(p_mag, theta_deg)
    pvec = ROOT.TVector3(px,py,pz)
    pT = max(1e-12, pvec.Perp())
    cot = pz/pT
    acc = {k: [] for k in PARAMS}
    trk = ROOT.SolTrack(x0,pvec,geom)
    for _ in range(npoints):
        if trk.nMeas()>=minmeas:

            if doKalman:
                trk.KalmanCov(doRes, doMS)
            else:
                trk.OldCovCalc(doRes, doMS)
            acc["D"].append(trk.s_D()*1e6)
            acc["Phi"].append(trk.s_phi0())
            acc["Pt"].append(trk.s_pt())
            acc["Z0"].append(trk.s_z0()*1e6)
            acc["Theta"].append(trk.s_ct()/(1.+cot*cot))
    mean=lambda v:float(np.mean(v)) if v else float("nan")
    return {k:mean(v) for k,v in acc.items()}

def eval_point_samples(p_mag, theta_deg, geom, npoints, minmeas, doKalman, doRes, doMS, verbose=False):
    """
    Like eval_point, but returns the full list of single-track resolutions
    for all PARAMS instead of their mean.
    """
    if verbose:
        print(f"    [samples] p={p_mag:.3g} GeV, theta={theta_deg:.1f}°")

    x0 = ROOT.TVector3(0., 0., 0.)
    px, py, pz = polar_to_components(p_mag, theta_deg)
    pvec = ROOT.TVector3(px, py, pz)
    pT = max(1e-12, pvec.Perp())
    cot = pz / pT

    acc = {k: [] for k in PARAMS}
    trk = ROOT.SolTrack(x0, pvec, geom)

    for _ in range(npoints):
        if trk.nMeas() >= minmeas:
            if doKalman:
                trk.KalmanCov(doRes, doMS)
            else:
                trk.OldCovCalc(doRes, doMS) 
            acc["D"].append(trk.s_D() * 1e6)
            acc["Phi"].append(trk.s_phi0())
            acc["Pt"].append(trk.s_pt())
            acc["Z0"].append(trk.s_z0() * 1e6)
            acc["Theta"].append(trk.s_ct() / (1. + cot * cot))

    return acc


def collect_hist_samples_for_detector(
    geom,
    p_mag_values: List[float],
    theta_values: List[float],
    npoints: int,
    minmeas: int,
    doKalman: bool,
    doRes: bool,
    doMS: bool,
    verbose: bool = False,
):
    samples = {
        param: {p_mag: {th: [] for th in theta_values} for p_mag in p_mag_values}
        for param in PARAMS
    }

    for p_mag in p_mag_values:
        for th in theta_values:
            acc = eval_point_samples(p_mag, th, geom, npoints, minmeas, doKalman, doRes, doMS, verbose)
            for param in PARAMS:
                samples[param][p_mag][th].extend(acc[param])

    return samples


# ============================================================
# === Grid builders ==========================================
# ============================================================

def build_pt_grid():
    ptmin,ptmax = PT_RANGE_PTSCAN
    nbins       = PT_NBINS_PTSCAN
    if any(PLOT_CFG[k].get("logx_pt",False) for k in PARAMS):
        if ptmin<=0 or ptmax<=0: raise ValueError("logx_pt requested but ptmin/ptmax <= 0")
        grid=list(np.logspace(np.log10(ptmin),np.log10(ptmax),nbins)); mode="log"
    else:
        grid=list(np.linspace(ptmin,ptmax,nbins)); mode="linear"
    print(f"pT-grid: {mode}-spaced {nbins} pts in [{ptmin},{ptmax}] GeV")
    return grid

def build_p_mag_grid():
    p_mag_min,p_mag_max = P_RANGE_PSCAN
    nbins       = P_NBINS_PSCAN
    if any(PLOT_CFG[k].get("logx_p_mag",False) for k in PARAMS):
        if p_mag_min<=0 or p_mag_max<=0: raise ValueError("logx_p_mag requested but p_mag_min/p_mag_max <= 0")
        grid=list(np.logspace(np.log10(p_mag_min),np.log10(p_mag_max),nbins)); mode="log"
    else:
        grid=list(np.linspace(p_mag_min,p_mag_max,nbins)); mode="linear"
    print(f"p-grid: {mode}-spaced {nbins} pts in [{p_mag_min},{p_mag_max}] GeV")
    return grid

def build_theta_grid():
    tmin,tmax = THETA_RANGE_TSCAN
    nbins     = THETA_NBINS_TSCAN
    if any(PLOT_CFG[k].get("logx_t",False) for k in PARAMS):
        if tmin<=0 or tmax<=0: raise ValueError("logx_t requested but theta min/max <= 0")
        grid=list(np.logspace(np.log10(tmin),np.log10(tmax),nbins)); mode="log"
    else:
        grid=list(np.linspace(tmin,tmax,nbins)); mode="linear"
    print(f"theta-grid: {mode}-spaced {nbins} pts in [{tmin},{tmax}] deg")
    return grid

def get_pt_for_tscan():
    vals = list(PT_LIST_TSCAN)
    print(f"pT (for theta-scan): explicit list with {len(vals)} values -> {vals}")
    return vals

def get_p_mag_for_tscan():
    vals = list(P_LIST_TSCAN)
    print(f"p (for theta-scan): explicit list with {len(vals)} values -> {vals}")
    return vals

# ============================================================
# === Parallel scans (per detector) ==========================
# ============================================================

def parallel_scan_vs_pt_for_detector(pt_grid, thetas, geom, npoints, minmeas, doKalman, doRes, doMS, workers, verbose):
    """Scan over pT values (pt_grid) at fixed theta angles. For each (pT, theta) point,
    the total momentum p = pT / sin(theta) is passed to eval_point."""
    results={param:{th:[np.nan]*len(pt_grid) for th in thetas} for param in PARAMS}
    tasks=[(ith,ipt,th,pt) for ith,th in enumerate(thetas) for ipt,pt in enumerate(pt_grid)]
    total=len(tasks); done=0
    print(f"  [pT-scan] {total} points | {workers} threads")
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        def _submit(ith, ipt, th, pt):
            sinth = math.sin(math.radians(th))
            p_mag = pt / max(sinth, 1e-12)
            return ex.submit(eval_point, p_mag, th, geom, npoints, minmeas, doKalman, doRes, doMS, verbose)
        futs={_submit(ith,ipt,th,pt):(ith,ipt,th,pt)
              for ith,ipt,th,pt in tasks}
        step=max(1,total//20)
        for fut in cf.as_completed(futs):
            ith,ipt,th,pt=futs[fut]; vals=fut.result()
            for param in PARAMS: results[param][th][ipt]=vals[param]
            with progress_lock:
                done+=1
                if done%step==0 or done==total:
                    print(f"    progress: {done}/{total} ({100*done/total:.1f}%)")
    return results

def parallel_scan_vs_p_mag_for_detector(p_mag_grid, thetas, geom, npoints, minmeas, doKalman, doRes, doMS, workers, verbose):
    """Scan over |p| values (p_mag_grid) at fixed theta angles. For each (|p|, theta) point,
    the total momentum |p| is passed to eval_point."""
    results={param:{th:[np.nan]*len(p_mag_grid) for th in thetas} for param in PARAMS}
    tasks=[(ith,ip_mag,th,p_mag) for ith,th in enumerate(thetas) for ip_mag,p_mag in enumerate(p_mag_grid)]
    total=len(tasks); done=0
    print(f"  [|p|-scan] {total} points | {workers} threads")
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        def _submit(ith, ip_mag, th, p_mag):
            return ex.submit(eval_point, p_mag, th, geom, npoints, minmeas, doKalman, doRes, doMS, verbose)
        futs={_submit(ith,ip_mag,th,p_mag):(ith,ip_mag,th,p_mag)
              for ith,ip_mag,th,p_mag in tasks}
        step=max(1,total//20)
        for fut in cf.as_completed(futs):
            ith,ip_mag,th,p_mag=futs[fut]; vals=fut.result()
            for param in PARAMS: results[param][th][ip_mag]=vals[param]
            with progress_lock:
                done+=1
                if done%step==0 or done==total:
                    print(f"    progress: {done}/{total} ({100*done/total:.1f}%)")
    return results

def parallel_scan_vs_theta_by_pt_for_detector(theta_grid, pts, geom, npoints, minmeas, doKalman, doRes, doMS, workers, verbose):
    """Scan over theta values at fixed pT values (pts). For each (pT, theta) point,
    the total momentum p = pT / sin(theta) is passed to eval_point."""
    results={param:{ptv:[np.nan]*len(theta_grid) for ptv in pts} for param in PARAMS}
    tasks=[(ipt,it,ptv,th) for ipt,ptv in enumerate(pts) for it,th in enumerate(theta_grid)]
    total=len(tasks); done=0
    print(f"  [theta-scan] {total} points | {workers} threads")
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        def _submit(ipt, it, ptv, th):
            sinth = math.sin(math.radians(th))
            p_mag = ptv / max(sinth, 1e-12)
            return ex.submit(eval_point, p_mag, th, geom, npoints, minmeas, doKalman, doRes, doMS, verbose)
        futs={_submit(ipt,it,ptv,th):(ipt,it,ptv,th)
              for ipt,it,ptv,th in tasks}
        step=max(1,total//20)
        for fut in cf.as_completed(futs):
            ipt,it,ptv,th=futs[fut]; vals=fut.result()
            for param in PARAMS: results[param][ptv][it]=vals[param]
            with progress_lock:
                done+=1
                if done%step==0 or done==total:
                    print(f"    progress: {done}/{total} ({100*done/total:.1f}%)")
    return results

def parallel_scan_vs_theta_by_p_mag_for_detector(theta_grid, pts, geom, npoints, minmeas, doKalman, doRes, doMS, workers, verbose):
    """Scan over theta values at fixed |p| values (pts). For each (|p|, theta) point,
    the total momentum |p| is passed to eval_point."""
    results={param:{p_magv:[np.nan]*len(theta_grid) for p_magv in pts} for param in PARAMS}
    tasks=[(ip_mag,it,p_magv,th) for ip_mag,p_magv in enumerate(pts) for it,th in enumerate(theta_grid)]
    total=len(tasks); done=0
    print(f"  [theta-scan] {total} points | {workers} threads")
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        def _submit(ip_mag, it, p_magv, th):
            p_mag = p_magv
            return ex.submit(eval_point, p_mag, th, geom, npoints, minmeas, doKalman, doRes, doMS, verbose)
        futs={_submit(ip_mag,it,p_magv,th):(ip_mag,it,p_magv,th)
              for ip_mag,it,p_magv,th in tasks}
        step=max(1,total//20)
        for fut in cf.as_completed(futs):
            ip_mag,it,p_magv,th=futs[fut]; vals=fut.result()
            for param in PARAMS: results[param][p_magv][it]=vals[param]
            with progress_lock:
                done+=1
                if done%step==0 or done==total:
                    print(f"    progress: {done}/{total} ({100*done/total:.1f}%)")
    return results

# ============================================================
# === Geometry description (r–z) =============================
# ============================================================

SCALE_ALPHA_WITH_FRACTION = True  # use meas_frac to modulate alpha if available

@dataclass
class LayerGeo:
    ltype: int
    label: str
    min_loc: float
    max_loc: float
    r_or_z: float
    thickness: float
    x0: float
    nmeas: int
    stereo_u: float
    stereo_l: float
    res_u: float
    res_l: float
    meas_frac: float
    measuring_flag: Optional[int] = None
    raw_line_no: Optional[int] = None
    raw_text: Optional[str] = None

def _to_float(tok: str) -> float:
    return float(tok.replace(",", "."))

def parse_geometry_geo(path: str) -> List[LayerGeo]:
    layers: List[LayerGeo] = []
    with open(path, "r") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                ltype = int(parts[0])
                label = parts[1]
                min_loc   = _to_float(parts[2])
                max_loc   = _to_float(parts[3])
                r_or_z    = _to_float(parts[4])
                thickness = _to_float(parts[5])
                x0        = _to_float(parts[6])
                nmeas     = int(parts[7])

                measuring_flag = None
                def looks_like_int01(s: str) -> bool:
                    try:
                        v = int(s); return v in (0, 1)
                    except Exception:
                        return False

                if len(parts) >= 14 and looks_like_int01(parts[8]):
                    measuring_flag = int(parts[8])
                    stereo_u = _to_float(parts[9])
                    stereo_l = _to_float(parts[10])
                    res_u    = _to_float(parts[11])
                    res_l    = _to_float(parts[12])
                    meas_frac = _to_float(parts[13]) if len(parts) >= 14 else 0.0
                else:
                    stereo_u = _to_float(parts[8])
                    stereo_l = _to_float(parts[9])
                    res_u    = _to_float(parts[10])
                    res_l    = _to_float(parts[11])
                    meas_frac = _to_float(parts[12])

                layers.append(LayerGeo(
                    ltype, label, min_loc, max_loc, r_or_z, thickness, x0, nmeas,
                    stereo_u, stereo_l, res_u, res_l, meas_frac,
                    measuring_flag, i, line
                ))
            except Exception as e:
                print(f"[GEO][WARN] L{i}: {e}\n  {line}")
    return layers

def validate_layers_geo(layers: List[LayerGeo]) -> int:
    issues = 0
    for L in layers:
        here = []
        if L.thickness <= 0: here.append("non-positive thickness")
        if L.x0 <= 0: here.append("non-positive X0")
        if L.min_loc > L.max_loc: here.append("min_loc>max_loc")
        if L.nmeas not in (0,1,2): here.append(f"nmeas={L.nmeas} not in {{0,1,2}}")
        if here:
            issues += len(here)
            print(f"[ISSUE] L{L.raw_line_no:>3} {L.label:>7}: " + "; ".join(here))
    print(f"[geom] validation: {issues} issue(s)")
    return issues

_GEO_SPECIAL_COLORS = {
    "mag":  ("#7f7f7f", 0.45),   # magnet  → gray
    "ecal": ("#d62728", 0.50),   # ECAL    → red
    "arc":  ("#2ca02c", 0.50),   # ARC     → green
}

def _style_for_layer_geo(L: LayerGeo) -> Tuple[str, float]:
    label_low = L.label.lower()
    for key, (face, alpha) in _GEO_SPECIAL_COLORS.items():
        if key in label_low:
            return face, alpha
    active = (L.meas_frac > 0.0)
    twoD   = (L.nmeas == 2)
    if active:
        face = "#1f77b4"
        base_alpha = 0.85 if twoD else 0.45
        if SCALE_ALPHA_WITH_FRACTION:
            min_alpha = 0.25 if twoD else 0.15
            alpha = min_alpha + (base_alpha - min_alpha) * max(0.0, min(1.0, L.meas_frac))
        else:
            alpha = base_alpha
    else:
        face = "#ff7f0e"
        alpha = 0.75 if twoD else 0.35
    return face, alpha

def plot_rz_geometry(layers: List[LayerGeo], title: str, out_pdf: str, cfg: dict = None):
    if cfg is None:
        cfg = GEO_CFG
    fig, ax = plt.subplots(figsize=(10, 6))
    r_min, r_max = math.inf, -math.inf
    z_min, z_max = math.inf, -math.inf
    subsys_boxes: Dict[str, dict] = {}  # key -> {zlo,zhi,rlo,rhi}
    for L in layers:
        color, alpha = _style_for_layer_geo(L)
        is_pipe = ("pipe" in L.label.lower())
        if L.ltype == 1:
            z0, z1 = sorted([L.min_loc, L.max_loc])
            r0, r1 = L.r_or_z - 0.5*L.thickness, L.r_or_z + 0.5*L.thickness
            rect = Rectangle((z0, r0), z1 - z0, r1 - r0,
                             facecolor=color, edgecolor="k", linewidth=0.7, alpha=alpha)
            ax.add_patch(rect)
            if not is_pipe:
                r_min = min(r_min, r0); r_max = max(r_max, r1)
                z_min = min(z_min, z0); z_max = max(z_max, z1)
            for key in _GEO_SPECIAL_COLORS:
                if key in L.label.lower():
                    b = subsys_boxes.setdefault(key, {"zlo": math.inf, "zhi": -math.inf, "rlo_bar": math.inf, "rhi_bar": -math.inf})
                    b["zlo"] = min(b["zlo"], z0); b["zhi"] = max(b["zhi"], z1)
                    b["rlo_bar"] = min(b["rlo_bar"], r0); b["rhi_bar"] = max(b["rhi_bar"], r1)
        elif L.ltype == 2:
            R0, R1 = sorted([L.min_loc, L.max_loc])
            zc = L.r_or_z
            z0, z1 = zc - 0.5*L.thickness, zc + 0.5*L.thickness
            rect = Rectangle((z0, R0), z1 - z0, R1 - R0,
                             facecolor=color, edgecolor="k", linewidth=0.7, alpha=alpha)
            ax.add_patch(rect)
            if not is_pipe:
                r_min = min(r_min, R0); r_max = max(r_max, R1)
                z_min = min(z_min, z0); z_max = max(z_max, z1)
            for key in _GEO_SPECIAL_COLORS:
                if key in L.label.lower():
                    b = subsys_boxes.setdefault(key, {"zlo": math.inf, "zhi": -math.inf, "rlo_bar": math.inf, "rhi_bar": -math.inf})
                    b["zlo"] = min(b["zlo"], z0); b["zhi"] = max(b["zhi"], z1)
    ax.set_xlabel(r"$z$ [m]"); ax.set_ylabel(r"$r$ [m]"); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    if not math.isinf(r_min):
        dr = 0.02*(r_max - r_min) if (r_max > r_min) else 0.01
        dz = 0.02*(z_max - z_min) if (z_max > z_min) else 0.01
        xlo = z_min - dz; xhi = z_max + dz
        ylo = max(0, r_min - dr); yhi = r_max + dr
        if cfg["zmax"] is not None:
            xlo, xhi = -cfg["zmax"], cfg["zmax"]
        if cfg["rmax"] is not None:
            yhi = cfg["rmax"]
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        # --- subsystem labels ---
        for key, b in subsys_boxes.items():
            face_hex, _ = _GEO_SPECIAL_COLORS[key]
            dark = tuple(c * 0.55 for c in mcolors.to_rgb(face_hex))
            zc = -0.5  # fixed offset from centre, within barrel volumes
            rc = 0.5 * (b["rlo_bar"] + b["rhi_bar"])
            # only label if position is within the current view
            if xlo <= zc <= xhi and ylo <= rc <= yhi:
                ax.text(zc, rc, key.upper(),
                        color=dark, fontsize=11, fontweight="bold",
                        ha="center", va="center", clip_on=True)
        # --- theta reference lines ---
        for th_deg in [10, 25, 45, 60, 90]:
            th_rad = math.radians(th_deg)
            if abs(th_deg - 90) < 1e-9:
                ax.axvline(x=0, color="lightgrey", linestyle="--", lw=1.0, zorder=0)
                ax.text(0, 0.97 * yhi, r"$90^\circ$",
                        color="darkgrey", fontsize=9, ha="right", va="top",
                        rotation=90, clip_on=True)
            elif xhi > 0:
                slope = math.tan(th_rad)
                r_at_xhi = slope * xhi
                if r_at_xhi <= yhi:
                    z_end, r_end = xhi, r_at_xhi
                else:
                    z_end, r_end = yhi / slope, yhi
                ax.plot([0, z_end], [0, r_end],
                        color="lightgrey", linestyle="--", lw=1.0, zorder=0)
                lz, lr = 0.90 * z_end, 0.90 * r_end
                ax.text(lz, lr, rf"${th_deg}^\circ$",
                        color="darkgrey", fontsize=9, ha="center", va="center",
                        clip_on=True,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=MAT_CFG["dpi"])
    png = out_pdf.replace(".pdf", ".png")
    fig.savefig(png, dpi=MAT_CFG["dpi"])
    plt.close(fig)
    print(f"[geom] saved {out_pdf} and {png}")

# ============================================================
# === Material budget (%/X0) =================================
# ============================================================

@dataclass
class LayerMat:
    ltype: int
    label: str
    min_loc: float
    max_loc: float
    r_or_z: float
    thickness: float
    x0: float

def parse_geometry_mat(path: str) -> List[LayerMat]:
    layers: List[LayerMat] = []
    ignore_subs = [s.lower() for s in MAT_CFG["ignore_labels_containing"]]
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            try:
                ltype = int(parts[0])
                label = parts[1]
                # ignore labels with any configured substring (case-insensitive)
                if any(sub in label.lower() for sub in ignore_subs):
                    continue
                min_loc = float(parts[2]); max_loc = float(parts[3])
                r_or_z  = float(parts[4]); thickness = float(parts[5]); x0 = float(parts[6])
                layers.append(LayerMat(ltype, label, min_loc, max_loc, r_or_z, thickness, x0))
            except Exception as e:
                print(f"[MAT][WARN] skip line: {line}\n  -> {e}")
    return layers

def _interval_intersection(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lo = max(min(a), min(b)); hi = min(max(a), max(b))
    return max(0.0, hi - lo)

def _path_len_barrel(L: LayerMat, costh: float, sinth: float) -> float:
    if sinth <= 0.0: return 0.0
    R = L.r_or_z; dR = max(0.0, L.thickness)
    z0, z1 = sorted([L.min_loc, L.max_loc])
    t_r = ((R - 0.5*dR)/sinth, (R + 0.5*dR)/sinth)
    if costh == 0.0:
        return (t_r[1]-t_r[0]) if (z0 <= 0.0 <= z1) else 0.0
    t_z_edges = (z0/costh, z1/costh); t_z = (min(t_z_edges), max(t_z_edges))
    return _interval_intersection(t_r, t_z)

def _path_len_endcap(L: LayerMat, costh: float, sinth: float) -> float:
    R0, R1 = sorted([L.min_loc, L.max_loc]); zc = L.r_or_z; dz = max(0.0, L.thickness)
    if costh == 0.0:
        if (zc - 0.5*dz) <= 0.0 <= (zc + 0.5*dz): t_z = (-math.inf, math.inf)
        else: return 0.0
    else:
        e = ((zc - 0.5*dz)/costh, (zc + 0.5*dz)/costh); t_z = (min(e), max(e))
    if sinth == 0.0:
        if R0 == 0.0: t_r = (-math.inf, math.inf)
        else: return 0.0
    else:
        t_r = (R0/sinth, R1/sinth)
    return _interval_intersection(t_r, t_z)

def material_vs_costh_grouped(layers: List[LayerMat], bins=100, cmin=-0.99, cmax=0.99):
    edges = np.linspace(cmin, cmax, bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    width = edges[1]-edges[0]
    labels_order: List[str] = []
    groups: Dict[str, List[LayerMat]] = {}
    for L in layers:
        if L.label not in groups:
            labels_order.append(L.label); groups[L.label] = []
        groups[L.label].append(L)
    nG = len(labels_order); stack = np.zeros((nG, bins), dtype=float)
    for gi, label in enumerate(labels_order):
        Ls = groups[label]
        for b, cth in enumerate(centers):
            cth = float(np.clip(cth, -0.999999, 0.999999))
            sth = math.sqrt(max(0.0, 1.0 - cth*cth))
            pct = 0.0
            for L in Ls:
                if L.x0 <= 0.0: continue
                tlen = _path_len_barrel(L, cth, sth) if L.ltype == 1 else \
                       _path_len_endcap(L, cth, sth) if L.ltype == 2 else 0.0
                if tlen > 0.0: pct += 100.0*(tlen / L.x0)
            stack[gi, b] = pct
    return centers, width, stack, labels_order

def _latex_percent_formatter():
    # Works with text.usetex=True (escapes %)
    return FuncFormatter(lambda x, pos: rf"{x:.0f}\%")

def plot_material_budget(geom_path: str, out_pdf: str):
    layers = parse_geometry_mat(geom_path)
    if not layers:
        print(f"[mat] no valid layers for {geom_path}, skipping material plot")
        return
    centers, width, stack, labels_order = material_vs_costh_grouped(
        layers,
        bins=MAT_CFG["bins"],
        cmin=MAT_CFG["cmin"],
        cmax=MAT_CFG["cmax"]
    )
    nG = len(labels_order)
    tab20 = plt.get_cmap("tab20").colors
    colors = [tab20[i % len(tab20)] for i in range(nG)]
    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros_like(centers)
    for i, label in enumerate(labels_order):
        ax.bar(centers, stack[i], width=width, bottom=bottom, align="center",
               edgecolor="none", label=label, color=colors[i])
        bottom += stack[i]
    base = os.path.splitext(os.path.basename(geom_path))[0]
    ax.set_xlabel(MAT_CFG["xlabel"])
    ax.set_ylabel(MAT_CFG["ylabel"])  # LaTeX-safe percent sign
    ax.set_title(fr"{base}: ($\%/X_0$) vs $\cos\theta$")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(centers[0]-0.5*width, centers[-1]+0.5*width)
    if MAT_CFG["ymax"] is not None:
        ax.set_ylim(0, MAT_CFG["ymax"])
    if MAT_CFG["percent_ticks"]:
        ax.yaxis.set_major_formatter(_latex_percent_formatter())
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=MAT_CFG["legend_bbox"],
        frameon=True,
        fancybox=True,
        shadow=False,
        ncol=MAT_CFG["legend_ncol"],
        fontsize=9,
        title=MAT_CFG["legend_title"],
        title_fontsize=10,
    )
    legend.get_frame().set_alpha(0.9)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=MAT_CFG["dpi"])
    # also export PNG for quick browsing
    png = out_pdf.replace(".pdf", ".png")
    fig.savefig(png, dpi=MAT_CFG["dpi"])
    plt.close(fig)
    print(f"[mat] saved {out_pdf} and {png}")

# ============================================================
# === Plotting (multi-detector overlays for res) =============
# ============================================================
def _save_both(figpath_png: Path):
    png = str(figpath_png)
    pdf = str(figpath_png.with_suffix(".pdf"))
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    print(f"[plot] saved {png} and {pdf}")


_RATIO_FIG_SIZE   = (8.2, 7.0)
_RATIO_HEIGHT_KW  = {"height_ratios": [3, 1], "hspace": 0.06}
# _SINGLE_FIG_SIZE  = (8.2, 5.8)
_SINGLE_FIG_SIZE  = (8.2, 7.0)


def make_plot_vs_pt_multi(param, pt_grid, curves_list, labels, bfields, outdir):
    cfg = PLOT_CFG[param]
    multi = len(curves_list) > 1

    fig, (ax_main, ax_rat) = plt.subplots(
        2, 1, figsize=_RATIO_FIG_SIZE, sharex=True,
        gridspec_kw=_RATIO_HEIGHT_KW,
    )

    if not multi:
        ax_rat.axis('off')

    for idx, curves in enumerate(curves_list):
        ls = LINESTYLES[idx % len(LINESTYLES)]
        lab_prefix = f"{labels[idx]} ({bfields[idx]:g}T)" if labels[idx] else f"Card {idx+1} ({bfields[idx]:g}T)"
        for th in sorted(curves.keys()):
            color = COLOR_BY_THETA.get(th, None)
            ax_main.plot(
                pt_grid, curves[th],
                linestyle=ls, lw=1.8, color=color,
                label=fr"{lab_prefix}, $\theta={th:.0f}^\circ$"
            )

    # ratio panel: config[i] / config[0] for i >= 1
    if multi and ax_rat is not None:
        ref_curves = curves_list[0]
        for idx in range(1, len(curves_list)):
            ls = LINESTYLES[idx % len(LINESTYLES)]
            for th in sorted(curves_list[idx].keys()):
                color = COLOR_BY_THETA.get(th, None)
                ref = np.array(ref_curves[th], dtype=float)
                cur = np.array(curves_list[idx][th], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref != 0, cur / ref, np.nan)
                ax_rat.plot(pt_grid, ratio, linestyle=ls, lw=1.5, color=color)
        ax_rat.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_rat.set_ylabel("Ratio")
        ax_rat.set_xlabel(cfg["xlabel_pt"])
        ax_rat.set_ylim(cfg.get("ratio_ymin_pt", cfg.get("ratio_ymin", 0.0)),
                        cfg.get("ratio_ymax_pt", cfg.get("ratio_ymax", 2.0)))
        if cfg["logx_pt"]:
            ax_rat.set_xscale("log")
        ax_rat.grid(True, which="both", alpha=0.3)

    ax_main.set_ylabel(cfg["ylabel"])
    if not multi:
        ax_main.set_xlabel(cfg["xlabel_pt"])
        ax_main.tick_params(axis='x', labelbottom=True)
    if cfg["logx_pt"]: ax_main.set_xscale("log")
    if cfg["logy"]:   ax_main.set_yscale("log")
    ax_main.set_ylim(cfg["ymin"], cfg["ymax"])
    ax_main.grid(True, which="both", alpha=0.3)
    n_det = len(curves_list)
    ax_main.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                   ncol=n_det if multi else 1, borderaxespad=0, frameon=True)
    fig.tight_layout()
    _save_both(Path(outdir) / f"{param}_vs_pt_by_theta.png")
    plt.close(fig)


def make_plot_vs_p_mag_multi(param, p_mag_grid, curves_list, labels, bfields, outdir):
    cfg = PLOT_CFG[param]
    multi = len(curves_list) > 1

    fig, (ax_main, ax_rat) = plt.subplots(
        2, 1, figsize=_RATIO_FIG_SIZE, sharex=True,
        gridspec_kw=_RATIO_HEIGHT_KW,
    )

    if not multi:
        ax_rat.axis('off')

    for idx, curves in enumerate(curves_list):
        ls = LINESTYLES[idx % len(LINESTYLES)]
        lab_prefix = f"{labels[idx]} ({bfields[idx]:g}T)" if labels[idx] else f"Card {idx+1} ({bfields[idx]:g}T)"
        for th in sorted(curves.keys()):
            color = COLOR_BY_THETA.get(th, None)
            ax_main.plot(
                p_mag_grid, curves[th],
                linestyle=ls, lw=1.8, color=color,
                label=fr"{lab_prefix}, $\theta={th:.0f}^\circ$"
            )

    # ratio panel: config[i] / config[0] for i >= 1
    if multi and ax_rat is not None:
        ref_curves = curves_list[0]
        for idx in range(1, len(curves_list)):
            ls = LINESTYLES[idx % len(LINESTYLES)]
            for th in sorted(curves_list[idx].keys()):
                color = COLOR_BY_THETA.get(th, None)
                ref = np.array(ref_curves[th], dtype=float)
                cur = np.array(curves_list[idx][th], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref != 0, cur / ref, np.nan)
                ax_rat.plot(p_mag_grid, ratio, linestyle=ls, lw=1.5, color=color)
        ax_rat.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_rat.set_ylabel("Ratio")
        ax_rat.set_xlabel(cfg["xlabel_p_mag"])
        ax_rat.set_ylim(cfg.get("ratio_ymin_p_mag", cfg.get("ratio_ymin", 0.0)),
                        cfg.get("ratio_ymax_p_mag", cfg.get("ratio_ymax", 2.0)))
        if cfg["logx_p_mag"]:
            ax_rat.set_xscale("log")
        ax_rat.grid(True, which="both", alpha=0.3)

    ax_main.set_ylabel(cfg["ylabel"])
    if not multi:
        ax_main.set_xlabel(cfg["xlabel_p_mag"])
        ax_main.tick_params(axis='x', labelbottom=True)
    if cfg["logx_p_mag"]: ax_main.set_xscale("log")
    if cfg["logy"]:   ax_main.set_yscale("log")
    ax_main.set_ylim(cfg["ymin"], cfg["ymax"])
    ax_main.grid(True, which="both", alpha=0.3)
    n_det = len(curves_list)
    ax_main.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                   ncol=n_det if multi else 1, borderaxespad=0, frameon=True)
    fig.tight_layout()
    _save_both(Path(outdir) / f"{param}_vs_p_mag_by_theta.png")
    plt.close(fig)


def make_plot_vs_theta_by_pt_multi(param, theta_grid, curves_list, labels, bfields, outdir):
    cfg = PLOT_CFG[param]
    multi = len(curves_list) > 1

    fig, (ax_main, ax_rat) = plt.subplots(
        2, 1, figsize=_RATIO_FIG_SIZE, sharex=True,
        gridspec_kw=_RATIO_HEIGHT_KW,
    )

    if not multi:
        ax_rat.axis('off')

    for idx, curves in enumerate(curves_list):
        ls = LINESTYLES[idx % len(LINESTYLES)]
        lab_prefix = f"{labels[idx]} ({bfields[idx]:g}T)" if labels[idx] else f"Card {idx+1} ({bfields[idx]:g}T)"
        for ptval in sorted(curves.keys()):
            color = COLOR_BY_PTVAL.get(ptval, None)
            ax_main.plot(
                theta_grid, curves[ptval],
                linestyle=ls, lw=1.8, color=color,
                label=fr"{lab_prefix}, $p_T={ptval:.0f}\,\mathrm{{GeV}}$"
            )

    # ratio panel: config[i] / config[0] for i >= 1
    if multi and ax_rat is not None:
        ref_curves = curves_list[0]
        for idx in range(1, len(curves_list)):
            ls = LINESTYLES[idx % len(LINESTYLES)]
            for ptval in sorted(curves_list[idx].keys()):
                color = COLOR_BY_PTVAL.get(ptval, None)
                ref = np.array(ref_curves[ptval], dtype=float)
                cur = np.array(curves_list[idx][ptval], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref != 0, cur / ref, np.nan)
                ax_rat.plot(theta_grid, ratio, linestyle=ls, lw=1.5, color=color)
        ax_rat.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_rat.set_ylabel("Ratio")
        ax_rat.set_xlabel(cfg["xlabel_t"])
        ax_rat.set_ylim(cfg.get("ratio_ymin_t", cfg.get("ratio_ymin", 0.0)),
                        cfg.get("ratio_ymax_t", cfg.get("ratio_ymax", 2.0)))
        if cfg["logx_t"]:
            ax_rat.set_xscale("log")
        ax_rat.grid(True, which="both", alpha=0.3)

    ax_main.set_ylabel(cfg["ylabel"])
    if not multi:
        ax_main.set_xlabel(cfg["xlabel_t"])
        ax_main.tick_params(axis='x', labelbottom=True)
    if cfg["logx_t"]: ax_main.set_xscale("log")
    if cfg["logy"]:   ax_main.set_yscale("log")
    ax_main.set_ylim(cfg["ymin"], cfg["ymax"])
    ax_main.grid(True, which="both", alpha=0.3)
    n_det = len(curves_list)
    ax_main.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                   ncol=n_det if multi else 1, borderaxespad=0, frameon=True)
    fig.tight_layout()
    _save_both(Path(outdir) / f"{param}_vs_theta_by_pt.png")
    plt.close(fig)

def make_plot_vs_theta_by_p_mag_multi(param, theta_grid, curves_list, labels, bfields, outdir):
    cfg = PLOT_CFG[param]
    multi = len(curves_list) > 1

    fig, (ax_main, ax_rat) = plt.subplots(
        2, 1, figsize=_RATIO_FIG_SIZE, sharex=True,
        gridspec_kw=_RATIO_HEIGHT_KW,
    )

    if not multi:
        ax_rat.axis('off')

    for idx, curves in enumerate(curves_list):
        ls = LINESTYLES[idx % len(LINESTYLES)]
        lab_prefix = f"{labels[idx]} ({bfields[idx]:g}T)" if labels[idx] else f"Card {idx+1} ({bfields[idx]:g}T)"
        for p_magval in sorted(curves.keys()):
            color = COLOR_BY_PVAL.get(p_magval, None)
            ax_main.plot(
                theta_grid, curves[p_magval],
                linestyle=ls, lw=1.8, color=color,
                label=fr"{lab_prefix}, $p={p_magval:.0f}\,\mathrm{{GeV}}$"
            )

    # ratio panel: config[i] / config[0] for i >= 1
    if multi and ax_rat is not None:
        ref_curves = curves_list[0]
        for idx in range(1, len(curves_list)):
            ls = LINESTYLES[idx % len(LINESTYLES)]
            for p_magval in sorted(curves_list[idx].keys()):
                color = COLOR_BY_PVAL.get(p_magval, None)
                ref = np.array(ref_curves[p_magval], dtype=float)
                cur = np.array(curves_list[idx][p_magval], dtype=float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(ref != 0, cur / ref, np.nan)
                ax_rat.plot(theta_grid, ratio, linestyle=ls, lw=1.5, color=color)
        ax_rat.axhline(1.0, color="k", ls="--", lw=0.8)
        ax_rat.set_ylabel("Ratio")
        ax_rat.set_xlabel(cfg["xlabel_t"])
        ax_rat.set_ylim(cfg.get("ratio_ymin_t", cfg.get("ratio_ymin", 0.0)),
                        cfg.get("ratio_ymax_t", cfg.get("ratio_ymax", 2.0)))
        if cfg["logx_t"]:
            ax_rat.set_xscale("log")
        ax_rat.grid(True, which="both", alpha=0.3)

    ax_main.set_ylabel(cfg["ylabel"])
    if not multi:
        ax_main.set_xlabel(cfg["xlabel_t"])
        ax_main.tick_params(axis='x', labelbottom=True)
    if cfg["logx_t"]: ax_main.set_xscale("log")
    if cfg["logy"]:   ax_main.set_yscale("log")
    ax_main.set_ylim(cfg["ymin"], cfg["ymax"])
    ax_main.grid(True, which="both", alpha=0.3)
    n_det = len(curves_list)
    ax_main.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                   ncol=n_det if multi else 1, borderaxespad=0, frameon=True)
        
    fig.tight_layout()
    _save_both(Path(outdir) / f"{param}_vs_theta_by_p_mag.png")
    plt.close(fig)


def make_hist_plots_multi(hist_samples_all, labels, bfields, outdir):
    """
    hist_samples_all: list over detectors:
        hist_samples_all[i][param][p][theta] -> list of values

    Produces overlaid step histograms for each (param, p, theta) combination,
    comparing all detector geometries.

    Filenames: PARAM_hist_pX_theta_thY.(png|pdf)
    """
    nbins_hist = 60
    outdir = Path(outdir)

    # All detectors share the same set of p and theta by construction
    for param in PARAMS:
        # get p values from first detector
        p_keys = sorted(hist_samples_all[0][param].keys())
        for p in p_keys:
            theta_keys = sorted(hist_samples_all[0][param][p].keys())
            for theta in theta_keys:
                arrays = []
                descs = []

                for det_idx, samples in enumerate(hist_samples_all):
                    vals = np.array(samples[param][p][theta], dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        continue
                    arrays.append(vals)
                    descs.append(f"{labels[det_idx]} ({bfields[det_idx]:g}T)")

                if not arrays:
                    continue

                vmin = min(a.min() for a in arrays)
                vmax = max(a.max() for a in arrays)
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    continue

                margin = 0.05 * (vmax - vmin) if vmax > vmin else 0.1 * max(abs(vmax), 1.0)
                bins = np.linspace(vmin - margin, vmax + margin, nbins_hist + 1)

                plt.figure(figsize=(8.2, 5.8))
                ax = plt.gca()

                text_lines = []
                for vals, desc in zip(arrays, descs):
                    counts, edges = np.histogram(vals, bins=bins)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.step(
                        centers,
                        counts,
                        where="mid",
                        linewidth=3,        # solid, thick line
                        label=desc,
                    )
                    mu = np.mean(vals)
                    rms = np.std(vals)
                    text_lines.append(
                        rf"{desc}: $\mu={mu:.3g}$, $\mathrm{{RMS}}={rms:.3g}$"
                    )

                ax.set_xlabel(PLOT_CFG[param]["ylabel"])
                ax.set_ylabel("Entries")
                ax.set_title(
                    rf"{param} resolution, $p={p:.0f}$ GeV, "
                    rf"$\theta={theta:.0f}^\circ$"
                )
                ax.grid(True, alpha=0.3)
                ax.legend()

                ax.text(
                    0.97,
                    0.97,
                    "\n".join(text_lines),
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=11,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor="gray",
                    ),
                )

                plt.tight_layout()
                # ---- filename: PARAM_hist_p_magX_theta_thY ----
                p_tag = int(round(p))
                th_tag = int(round(theta))
                base = f"{param}_hist_p_mag{p_tag}_theta_th{th_tag}"
                _save_both(outdir / f"{base}.png")
                plt.close()


# ============================================================
# === LaTeX report builder ===================================
# ============================================================

LATEX_TPL_HEADER = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.5cm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{hyperref}
\usepackage{pdflscape}  % <— landscape pages
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}
\setlength{\parskip}{0.5em}
\begin{document}
\begin{titlepage}
  \centering
  {\LARGE Tracker Performance Report\par}
  \vspace{0.5cm}
  {\large \today\par}
  \vfill
  \begin{table}[H]\centering
  \caption*{Geometry cards and magnetic fields}
  \begin{tabular}{ll}
  \toprule
  \textbf{Card} & \textbf{B [T]} \\
  \midrule
"""

LATEX_TPL_HEADER_END = r"""
  \bottomrule
  \end{tabular}
  \end{table}
  \vfill
\end{titlepage}
"""

LATEX_SECTION_GEOM = r"""
\section*{Geometry (r--z)}
"""

LATEX_SECTION_MAT = r"""
\section*{Material budget ($\%/X_0$) vs $\cos\theta$}
"""

LATEX_SECTION_RES = r"""
\section*{Resolution scans}
"""

LATEX_DOC_END = r"""
\end{document}
"""

LATEX_BEAMER_HEADER = r"""
\documentclass[aspectratio=169,11pt]{beamer}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{hyperref}
\usepackage{ragged2e}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}

% Clean, plain frames
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{}
\setbeamertemplate{headline}{}

% A bit less white border on the page
\setlength{\parskip}{0pt}
\setlength{\parindent}{0pt}

\begin{document}
"""

LATEX_BEAMER_FOOTER = r"""
\end{document}
"""


def _latex_escape(s: str) -> str:
    # minimal escape for underscores, percent, ampersand
    return s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")

def write_beamer_report(outdir: Path, cards: List[str], bfields: List[float], labels: List[str], scan_variable: str) -> Path:
    tex_path = outdir / "report.tex"

    def esc_text(s: str) -> str:
        return (s.replace("\\", r"\textbackslash{}")
                 .replace("_", r"\_")
                 .replace("%", r"\%")
                 .replace("&", r"\&"))

    with open(tex_path, "w") as f:
        f.write(LATEX_BEAMER_HEADER)

        # Title slide
        f.write(r"\begin{frame}[plain]" "\n")
        f.write(r"\centering" "\n")
        f.write(r"{\LARGE Tracker Performance Report}\par" "\n")
        f.write(r"\vspace{0.5cm}" "\n")
        f.write(r"{\large\today}\par" "\n")
        f.write(r"\vspace{0.8cm}" "\n")
        f.write(r"\begin{tabular}{ll}" "\n")
        f.write(r"\toprule" "\n")
        f.write(r"\textbf{Card} & \textbf{B [T]} \\" "\n")
        f.write(r"\midrule" "\n")
        for c, b, lab in zip(cards, bfields, labels):
            f.write(f"{esc_text(lab)} ({esc_text(os.path.basename(c))}) & {b:g} " + r"\\" + "\n")
        f.write(r"\bottomrule" "\n")
        f.write(r"\end{tabular}" "\n")
        f.write(r"\end{frame}" "\n")

        # ============ Geometry ============
        if len(cards) == 1:
            base = os.path.splitext(os.path.basename(cards[0]))[0]
            fig  = f"{base}_geom.pdf"
            fig  = os.path.join(".", f"{base}_geom.pdf")
            f.write(r"\begin{frame}[plain]" "\n")
            f.write(r"\frametitle{Geometry (r--z)}" "\n")
            f.write(r"\centering\vfill" "\n")
            f.write(rf"\includegraphics[width=0.98\paperwidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{fig}}}}}" "\n")
            f.write(r"\vfill\end{frame}" "\n")
            # zoomed vertex slide (single card)
            fig_vtx = os.path.join(".", f"{base}_geom_vtx.pdf")
            f.write(r"\begin{frame}[plain]" "\n")
            f.write(r"\frametitle{Geometry (r--z) — vertex region}" "\n")
            f.write(r"\centering\vfill" "\n")
            f.write(rf"\includegraphics[width=0.98\paperwidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{fig_vtx}}}}}" "\n")
            f.write(r"\vfill\end{frame}" "\n")
        else:
            for i in range(0, len(cards), 2):
                pair_cards  = cards[i:i+2]
                pair_labels = labels[i:i+2]
                title = "Geometry (r--z): " + " \\& ".join(esc_text(L) for L in pair_labels)
                f.write(r"\begin{frame}[plain]" "\n")
                f.write(rf"\frametitle{{{title}}}" "\n")
                f.write(r"\centering\vfill" "\n")
                f.write(r"\begin{columns}[T,totalwidth=0.9\paperwidth]" "\n")
                for c in pair_cards:
                    base = os.path.splitext(os.path.basename(c))[0]
                    fig  = f"{base}_geom.pdf"
                    f.write(r"\begin{column}{0.42\paperwidth}\centering" "\n")
                    f.write(rf"\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{fig}}}}}" "\n")
                    f.write(r"\end{column}" "\n")
                f.write(r"\end{columns}" "\n")
                f.write(r"\vfill\end{frame}" "\n")
            # zoomed vertex slides (multi-card, same 2-per-slide layout)
            for i in range(0, len(cards), 2):
                pair_cards  = cards[i:i+2]
                pair_labels = labels[i:i+2]
                title = "Geometry (r--z) vertex region: " + " \\& ".join(esc_text(L) for L in pair_labels)
                f.write(r"\begin{frame}[plain]" "\n")
                f.write(rf"\frametitle{{{title}}}" "\n")
                f.write(r"\centering\vfill" "\n")
                f.write(r"\begin{columns}[T,totalwidth=0.9\paperwidth]" "\n")
                for c in pair_cards:
                    base = os.path.splitext(os.path.basename(c))[0]
                    fig  = f"{base}_geom_vtx.pdf"
                    f.write(r"\begin{column}{0.42\paperwidth}\centering" "\n")
                    f.write(rf"\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{fig}}}}}" "\n")
                    f.write(r"\end{column}" "\n")
                f.write(r"\end{columns}" "\n")
                f.write(r"\vfill\end{frame}" "\n")

        # ============ Material ============
        if len(cards) == 1:
            base = os.path.splitext(os.path.basename(cards[0]))[0]
            fig  = f"{base}_mat.pdf"
            f.write(r"\begin{frame}[plain]" "\n")
            f.write(r"\frametitle{Material budget (\%/$X_0$) vs $\cos\theta$}" "\n")
            f.write(r"\centering\vfill" "\n")
            f.write(rf"\includegraphics[width=0.98\paperwidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{fig}}}}}" "\n")
            f.write(r"\vfill\end{frame}" "\n")
        else:
            for i in range(0, len(cards), 2):
                pair_cards  = cards[i:i+2]
                pair_labels = labels[i:i+2]
                title = "Material budget (\\%/$X_0$) vs $\\cos\\theta$: " + " \\& ".join(esc_text(L) for L in pair_labels)
                f.write(r"\begin{frame}[plain]" "\n")
                f.write(rf"\frametitle{{{title}}}" "\n")
                f.write(r"\centering\vfill" "\n")
                f.write(r"\begin{columns}[T,totalwidth=0.9\paperwidth]" "\n")
                for c in pair_cards:
                    base = os.path.splitext(os.path.basename(c))[0]
                    fig  = f"{base}_mat.pdf"
                    f.write(r"\begin{column}{0.42\paperwidth}\centering" "\n")
                    f.write(rf"\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{fig}}}}}" "\n")
                    f.write(r"\end{column}" "\n")
                f.write(r"\end{columns}" "\n")
                f.write(r"\vfill\end{frame}" "\n")

        # ============ Resolution (always two per slide) ============
        for param in PARAMS:
            left = f"{param}_vs_pt_by_theta.pdf" if scan_variable in ("pt", "both") else f"{param}_vs_p_mag_by_theta.pdf" 
            right = f"{param}_vs_theta_by_pt.pdf" if scan_variable in ("pt", "both") else f"{param}_vs_theta_by_p_mag.pdf" 
            f.write(r"\begin{frame}[plain]" "\n")
            f.write(rf"\frametitle{{Resolution — {esc_text(param)} (left: vs $p$, right: vs $\theta$)}}" "\n")
            f.write(r"\centering\vfill" "\n")
            f.write(r"\begin{columns}[T,totalwidth=0.9\paperwidth]" "\n")
            f.write(r"\begin{column}{0.42\paperwidth}\centering" "\n")
            f.write(rf"\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{left}}}}}" "\n")
            f.write(r"\end{column}" "\n")
            f.write(r"\begin{column}{0.42\paperwidth}\centering" "\n")
            f.write(rf"\includegraphics[width=\linewidth,height=0.86\textheight,keepaspectratio]{{\detokenize{{{right}}}}}" "\n")
            f.write(r"\end{column}" "\n")
            f.write(r"\end{columns}" "\n")
            f.write(r"\vfill\end{frame}" "\n")

        f.write(LATEX_BEAMER_FOOTER)

    print(f"[latex] wrote {tex_path}")
    return tex_path

# --- add near the top (helpers) ---
from pathlib import Path
import re

def slugify(s: str) -> str:
    s = s.strip().lower().replace(" ", "-")
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    return s or "card"

# --- bool parsing helper (place near other helpers) ---
def _to_bool(x) -> bool:
    if isinstance(x, bool): 
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):  return True
    if s in ("0", "false", "f", "no", "n", "off"): return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {x}")


# --- replace your build_pdf() with this version ---
def build_pdf(tex_path: Path, out_pdf: Path):
    """Run pdflatex twice and move the resulting PDF to out_pdf."""
    if shutil.which("pdflatex") is None:
        print("[latex] pdflatex not found in PATH. Skipping compilation. You can compile manually:")
        print(f"        (cd {tex_path.parent} && pdflatex -interaction=nonstopmode -halt-on-error {tex_path.name})")
        return None

    cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
           "-output-directory", str(tex_path.parent), str(tex_path)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        built_pdf = tex_path.with_suffix(".pdf")
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(built_pdf), str(out_pdf))
        print(f"[latex] built {out_pdf}")
        return out_pdf
    except subprocess.CalledProcessError as e:
        print("[latex] pdflatex error:\n", e.stdout.decode(errors="ignore"))
        return None



# ============================================================
# === Main ===================================================
# ============================================================

def main():
    import argparse
    set_root_batch()

    ap=argparse.ArgumentParser(description="Track resolution scans + geometry & material plots (multi-detector)")
    ap.add_argument("--card", action="append", required=True,
                    help="Path to a detector geometry card (repeatable)")
    ap.add_argument("--bfield", action="append", type=float,
                    help="Magnetic field [T] per card (repeatable). Defaults to --bfield-default for missing entries.")
    ap.add_argument("--label", action="append",
                    help="Legend label per card (repeatable). Defaults to basename(card).")
    ap.add_argument("--bfield-default", type=float, default=3.0, help="Default B for cards without an explicit --bfield")

    ap.add_argument("--doKalman", action="append", type=_to_bool,
                help="(repeatable) Enable Kalman filter *for this card* (default: --doKalman-default)")
    ap.add_argument("--doRes",   action="append", type=_to_bool,
                help="(repeatable) Enable measurement-resolution term *for this card* (default: --doRes-default)")
    ap.add_argument("--doMS",    action="append", type=_to_bool,
                    help="(repeatable) Enable multiple-scattering term *for this card* (default: --doMS-default)")

    ap.add_argument("--doKalman-default", type=_to_bool, default=False,
                    help="Default doKalman for cards without explicit --doKalman")
    ap.add_argument("--doRes-default", type=_to_bool, default=True,
                    help="Default doRes for cards without explicit --doRes")
    ap.add_argument("--doMS-default",  type=_to_bool, default=True,
                    help="Default doMS for cards without explicit --doMS")
    
    
    ap.add_argument("--inc", default="examples/classes", help="Directory with SolGeom.h / SolTrack.h")
    ap.add_argument("--outdir", default="plots_effres_multi", help="Output directory for PNG/PDFs")
    ap.add_argument("--npoints", type=int, default=1000, help="#throws per (p,theta) point")
    ap.add_argument("--minmeas", type=int, default=6, help="Minimum #measurements to accept")
    ap.add_argument("--compile", action="store_true", help="(Re)compile C++ sources")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel threads per detector")
    ap.add_argument("--verbose-points", action="store_true", help="Print each (p,theta) point as it starts")
    ap.add_argument("--latex", action="store_true", help="Produce a LaTeX report (report.tex + report.pdf if pdflatex is available)")
    ap.add_argument("--scan_variable", choices=["pt", "p", "both"], default="both", help="Momentum scan type: 'pt' (transverse), 'p' (total), or 'both'")
    args=ap.parse_args()

    # Validate that all specified card files exist
    missing = [c for c in args.card if not os.path.isfile(c)]
    if missing:
        ap.error("card file(s) not found:\n  " + "\n  ".join(missing))

    load_cpp(args.inc,args.compile)


    def format_b_for_name(b: float) -> str:
        """Format magnetic field value for filenames: 2.0 -> 2T, 1.5 -> 1p5T."""
        if abs(b - round(b)) < 1e-9:
            s = str(int(round(b)))
        else:
            s = f"{b:.3g}".rstrip("0").rstrip(".").replace(".", "p")
        return f"{s}T"

    # Normalize list args
    cards = args.card
    nb = len(cards)
    bfields = (args.bfield or [])
    if len(bfields) < nb:
        bfields = bfields + [args.bfield_default] * (nb - len(bfields))
    labels = (args.label or [])
    if len(labels) < nb:
        labels = labels + [os.path.splitext(os.path.basename(c))[0] for c in cards[len(labels):]]

    # Normalize doRes/doMS per card
    doRes_list = (args.doRes or [])
    if len(doRes_list) < nb:
        doRes_list = doRes_list + [args.doRes_default] * (nb - len(doRes_list))

    doMS_list = (args.doMS or [])
    if len(doMS_list) < nb:
        doMS_list = doMS_list + [args.doMS_default] * (nb - len(doMS_list))

    doKalman_list = (args.doKalman or [])
    if len(doKalman_list) < nb:
        doKalman_list = doKalman_list + [args.doKalman_default] * (nb - len(doKalman_list))

    parts = []
    for lab, b, dRes, dMS, dKalman in zip(labels, bfields, doRes_list, doMS_list, doKalman_list):
        ms_tag  = "MSon" if dMS else "MSoff"
        res_tag = "ResOn" if dRes else "ResOff"
        kalman_tag = "KalmanOn" if dKalman else "KalmanOff"
        parts.append(f"{slugify(lab)}-{ms_tag}-{res_tag}-{kalman_tag}-{format_b_for_name(b)}")

    stem   = "report_" + "_".join(parts) if parts else "report"
    outdir = Path(args.outdir) / stem
    ensure_dir(outdir)            # <<< YOU MUST ADD THIS BACK
    report_pdf_path = Path(outdir) / f"{stem}.pdf"


    # Build grids for scans
    pt_grid      = build_pt_grid()
    p_mag_grid      = build_p_mag_grid()
    theta_grid   = build_theta_grid()
    pt_for_tscan = get_pt_for_tscan()
    p_mag_for_tscan = get_p_mag_for_tscan()

    # Per-card outputs (for LaTeX inclusion)
    geom_pdfs = []
    mat_pdfs  = []

    # Accumulate scan results across detectors
    res_vs_pt_all = []
    res_vs_p_mag_all = []
    res_vs_theta_by_pt_all = []
    res_vs_theta_by_p_mag_all = []

    # Per-detector hist samples for resolution distributions
    hist_samples_all = []

    for i,(card,bf,lab) in enumerate(zip(cards,bfields,labels)):
        dRes = doRes_list[i]
        dMS  = doMS_list[i]
        dKalman = doKalman_list[i]
        print(f"\n=== Detector {i+1}/{nb}: {lab} | card={card} | B={bf}T | doRes={dRes} doMS={dMS} doKalman={dKalman} ===")

        # ... geometry & material as before ...

        G = ROOT.SolGeom(card.encode()); G.SetBfield(bf)

        # --- Geometry and material plots for this detector ---
        base = os.path.splitext(os.path.basename(card))[0]

        # Parse and validate geometry for r–z view
        layers_geo = parse_geometry_geo(card)
        validate_layers_geo(layers_geo)
        geom_pdf_path = Path(outdir) / f"{base}_geom.pdf"
        plot_rz_geometry(layers_geo, title=f"{lab} (r–z view)", out_pdf=str(geom_pdf_path))

        geom_vtx_pdf_path = Path(outdir) / f"{base}_geom_vtx.pdf"
        plot_rz_geometry(layers_geo, title=f"{lab} (vertex region)", out_pdf=str(geom_vtx_pdf_path), cfg=GEO_VTX_CFG)

        # Material budget plot (%/X0)
        mat_pdf_path = Path(outdir) / f"{base}_mat.pdf"
        plot_material_budget(card, str(mat_pdf_path))

        if args.scan_variable in ("pt", "both"):

            print("Scanning vs pT ...")
            res_pt = parallel_scan_vs_pt_for_detector(
                pt_grid, THETA_SET_FOR_PTSCAN, G,
                args.npoints, args.minmeas, dKalman, dRes, dMS, args.workers, args.verbose_points
            )
            res_vs_pt_all.append(res_pt)

            print("Scanning vs theta by pT ...")
            res_t_by_pt = parallel_scan_vs_theta_by_pt_for_detector(
                theta_grid, pt_for_tscan, G,
                args.npoints, args.minmeas, dKalman, dRes, dMS, args.workers, args.verbose_points
            )
            res_vs_theta_by_pt_all.append(res_t_by_pt)

        if args.scan_variable in ("p", "both"):


            print("Scanning vs |p| ...")
            res_p_mag = parallel_scan_vs_p_mag_for_detector(
                p_mag_grid, THETA_SET_FOR_PSCAN, G,
                args.npoints, args.minmeas, dKalman, dRes, dMS, args.workers, args.verbose_points
            )
            res_vs_p_mag_all.append(res_p_mag)

            print("Scanning vs theta by |p| ...")
            res_t_by_p_mag = parallel_scan_vs_theta_by_p_mag_for_detector(
                theta_grid, p_mag_for_tscan, G,
                args.npoints, args.minmeas, dKalman, dRes, dMS, args.workers, args.verbose_points
            )
            res_vs_theta_by_p_mag_all.append(res_t_by_p_mag)


        # Histograms (note that these are in |p| not pT, same chosen p/theta as before)
        p_mag_for_hists = [100]
        theta_for_hists = [90]
        hist_samples = collect_hist_samples_for_detector(
            G,
            p_mag_for_hists,
            theta_for_hists,
            args.npoints,
            args.minmeas,
            dKalman,
            dRes,
            dMS,
            args.verbose_points,
        )
        hist_samples_all.append(hist_samples)


    # 4) Overlay plots (resolution)
    for param in PARAMS:

        if args.scan_variable in ("pt", "both"):

            make_plot_vs_pt_multi(
                param, pt_grid,
                [det_res[param] for det_res in res_vs_pt_all],
                labels, bfields, str(outdir)
            )

            make_plot_vs_theta_by_pt_multi(
                param, theta_grid,
                [det_res[param] for det_res in res_vs_theta_by_pt_all],
                labels, bfields, str(outdir)
            )

        if args.scan_variable in ("p", "both"):

            make_plot_vs_p_mag_multi(
                param, p_mag_grid,
                [det_res[param] for det_res in res_vs_p_mag_all],
                labels, bfields, str(outdir)
            )


            make_plot_vs_theta_by_p_mag_multi(
                param, theta_grid,
                [det_res[param] for det_res in res_vs_theta_by_p_mag_all],
                labels, bfields, str(outdir)
            )

    # 4b) Histogram plots of resolution distributions (multi-detector)
    make_hist_plots_multi(hist_samples_all, labels, bfields, str(outdir))


    # 5) LaTeX report (optional)
    if args.latex:
        tex_path = write_beamer_report(outdir, cards, bfields, labels, args.scan_variable)
        build_pdf(tex_path, report_pdf_path)

    print(f"\nDone. Outputs in {outdir.resolve()}")

if __name__=="__main__":
    main()
