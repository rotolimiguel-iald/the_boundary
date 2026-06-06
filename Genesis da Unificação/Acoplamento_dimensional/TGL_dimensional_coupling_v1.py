#!/usr/bin/env python3
"""
TGL Dimensional Coupling Analysis v1.0
Tests: alpha2(d) -> 0 at d_crit ~ string theory dimensions?
"""
import numpy as np
import json, time, os
from datetime import datetime
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


PLANCK_LENGTH = 1.616255e-35
ALPHA2_OBS = 0.012031
R_CENTRAL = 3.0e20; R_MIN = 1.0e20; R_MAX = 1.0e21
RCOH_CENTRAL = 3.0e18; RCOH_MIN = 3.0e17; RCOH_MAX = 3.0e19
N_SAMPLES = 100_000; D_ARRAY = list(range(1, 27)); N_DIMS = 26
THRESHOLD = 1e-6; SEED = 42

print("="*72)
print("TGL DIMENSIONAL COUPLING ANALYSIS v1.0")
print("="*72)

np.random.seed(SEED); t0 = time.time()

# Samples
r = np.exp(np.random.uniform(np.log(R_MIN), np.log(R_MAX), N_SAMPLES))
rc = np.exp(np.random.uniform(np.log(RCOH_MIN), np.log(RCOH_MAX), N_SAMPLES))

# Calibration
ln3 = np.log(R_CENTRAL) - np.log(3.0) - np.log(PLANCK_LENGTH)
Neff3 = ln3 / ALPHA2_OBS
beta_D = np.log(Neff3) / (3.0 * np.log(R_CENTRAL/RCOH_CENTRAL))
print(f"ln(r/(3*lP)) = {ln3:.2f}, Neff(3) = {Neff3:.1f}, beta_D = {beta_D:.6f}")

# Compute alpha2 for 4 models
a2A = np.zeros((N_SAMPLES, N_DIMS))
a2B = np.zeros((N_SAMPLES, N_DIMS))
a2C = np.zeros((N_SAMPLES, N_DIMS))
a2D = np.zeros((N_SAMPLES, N_DIMS))

ln3_s = np.log(r) - np.log(3.0) - np.log(PLANCK_LENGTH)

for idx, d in enumerate(D_ARRAY):
    lnr = np.log(r) - np.log(d) - np.log(PLANCK_LENGTH)
    a2A[:, idx] = lnr / (r/rc)**(d/2.0)
    a2B[:, idx] = lnr / (r/rc)**d
    a2C[:, idx] = (3.0/d)**2 * (lnr/ln3_s) * ALPHA2_OBS
    a2D[:, idx] = lnr / (r/rc)**(beta_D*d)

# Normalize to d=3
i3 = 2
rA = a2A / a2A[:, i3:i3+1]; rB = a2B / a2B[:, i3:i3+1]
rC = a2C / a2C[:, i3:i3+1]; rD = a2D / a2D[:, i3:i3+1]

# Find d_crit
def find_dcrit(ratio):
    dc = np.full(N_SAMPLES, 99, dtype=int)
    for i in range(N_DIMS):
        mask = (ratio[:, i] < THRESHOLD) & (dc == 99)
        dc[mask] = D_ARRAY[i]
    return dc

dcA = find_dcrit(rA); dcB = find_dcrit(rB)
dcC = find_dcrit(rC); dcD = find_dcrit(rD)

# Stats
def stats(dc, name):
    v = dc[dc < 99].astype(float)
    if len(v) == 0: return {"model": name, "never_decouples": True}
    hist = {}
    for d in D_ARRAY:
        c = int((v == d).sum())
        if c > 0: hist[str(d)] = c
    return {
        "model": name, "never_decouples": False,
        "fraction_decoupled": float((dc < 99).mean()),
        "median_d_crit": float(np.median(v)),
        "mean_d_crit": float(np.mean(v)),
        "std_d_crit": float(np.std(v)),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "P_9to11": float(((v>=9)&(v<=11)).mean()),
        "P_10or11": float(((v==10)|(v==11)).mean()),
        "P_25or26": float(((v==25)|(v==26)).mean()),
        "histogram": hist,
    }

sA = stats(dcA, "A_thermodynamic")
sB = stats(dcB, "B_full_phase_space")
sC = stats(dcC, "C_holographic")
sD = stats(dcD, "D_calibrated")

# Profiles
def prof(mat):
    return {
        "median": np.median(mat, axis=0).tolist(),
        "p5": np.percentile(mat, 5, axis=0).tolist(),
        "p95": np.percentile(mat, 95, axis=0).tolist(),
    }

# CCI
cci = np.median(a2A, axis=0)
cci = cci / cci[1]  # normalize to d=2

# String concordance
conc = {}
for th, dt in [("superstrings_d9_D10", 9), ("M_theory_d10_D11", 10), ("bosonic_d25_D26", 25)]:
    i = dt - 1
    conc[th] = {"d": dt, "D": dt+1}
    for k, p in [("A", rA), ("B", rB), ("C", rC), ("D", rD)]:
        med = np.median(p[:, i])
        conc[th][f"log10_{k}"] = float(np.log10(max(med, 1e-300)))

# Results
results = {
    "metadata": {"timestamp": datetime.now().isoformat(), "n_samples": N_SAMPLES,
                 "runtime_s": round(time.time()-t0, 2)},
    "calibration": {"alpha2_d3": ALPHA2_OBS, "ln_ratio_d3": float(ln3),
                    "Neff_d3": float(Neff3), "beta_D": float(beta_D)},
    "model_stats": {"A": sA, "B": sB, "C": sC, "D": sD},
    "profiles": {"d": D_ARRAY, "ratio": {k: prof(m) for k, m in 
                 [("A",rA),("B",rB),("C",rC),("D",rD)]},
                 "absolute": {k: prof(m) for k, m in
                 [("A",a2A),("B",a2B),("C",a2C),("D",a2D)]}},
    "cci": {"d": D_ARRAY, "cci_median": cci.tolist()},
    "string_concordance": conc,
}

# Print
print(f"\n{'='*72}")
print("RESULTS")
print("="*72)
for n, s in [("A thermo", sA), ("B phase", sB), ("C holo", sC), ("D calib", sD)]:
    if s.get("never_decouples"):
        print(f"  {n}: NEVER decouples in d<=26")
    else:
        print(f"  {n}: d_crit = {s['median_d_crit']:.0f} "
              f"(95%CI [{s['p5']:.0f},{s['p95']:.0f}]) "
              f"P(9-11)={s['P_9to11']:.3f}")

print(f"\n--- String concordance (log10[a2(d)/a2(3)]) ---")
for th, d in conc.items():
    print(f"  {th}: " + " | ".join(f"{k}={d[f'log10_{k}']:.1f}" for k in "ABCD"))

# VERDICT
print(f"\n{'='*72}\nVERDICT\n{'='*72}")
for n, s in [("A",sA),("B",sB),("C",sC),("D",sD)]:
    if not s.get("never_decouples"):
        m = s["median_d_crit"]
        if 9 <= m <= 12: print(f"  ✓ Model {n}: d_crit={m:.0f} — M-theory/superstrings")
        elif 24 <= m <= 26: print(f"  ~ Model {n}: d_crit={m:.0f} — bosonic strings")
        else: print(f"  ✗ Model {n}: d_crit={m:.0f} — no match")
    else: print(f"  — Model {n}: never decouples")

# Plots
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'figure.dpi':150,'font.family':'serif'})
dims = np.array(D_ARRAY)
cols = {'A':'#2196F3','B':'#F44336','C':'#4CAF50','D':'#FF9800'}
names = {'A':'Thermo (d/2)','B':'Phase (d)','C':'Holo (3/d)²','D':'Calib (βd)'}

# Fig 1: profiles
fig, axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle(r'$\alpha^2(d)/\alpha^2(3)$ — Four Independent Models',fontsize=15,fontweight='bold')
for ax, k in zip(axes.flat, 'ABCD'):
    p = results["profiles"]["ratio"][k]
    med = np.clip(p["median"], 1e-300, None)
    p5 = np.clip(p["p5"], 1e-300, None)
    p95 = np.clip(p["p95"], 1e-300, None)
    ax.plot(dims, np.log10(med), 'o-', color=cols[k], lw=2, ms=4)
    ax.fill_between(dims, np.log10(p5), np.log10(p95), alpha=0.2, color=cols[k])
    ax.axhline(np.log10(THRESHOLD), color='gray', ls='--', lw=1.5)
    ax.axvline(3, color='gold', lw=2, alpha=0.6)
    for dv in [9,10]: ax.axvline(dv, color='purple', ls=':', alpha=0.5)
    ax.axvline(25, color='darkred', ls='-.', alpha=0.5)
    ax.set_xlabel('$d$'); ax.set_ylabel(r'$\log_{10}[\alpha^2/\alpha^2(3)]$')
    ax.set_title(f'Model {k}: {names[k]}',fontsize=11)
    ax.set_xlim(1,26); ax.grid(True,alpha=0.3); ax.set_xticks([1,3,5,9,10,15,20,25])
plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT_DIR, 'tgl_dim_profiles.png'),bbox_inches='tight'); plt.close()

# Fig 2: histograms
fig, axes = plt.subplots(2,2,figsize=(14,10))
fig.suptitle(r'$d_{\mathrm{crit}}$ Distribution (Monte Carlo)',fontsize=15,fontweight='bold')
for ax, (k,s) in zip(axes.flat, [("A",sA),("B",sB),("C",sC),("D",sD)]):
    if s.get("never_decouples"):
        ax.text(0.5,0.5,f'Model {k}\nNever decouples',ha='center',va='center',
                transform=ax.transAxes,fontsize=14)
    else:
        h = s["histogram"]
        dv = sorted(int(x) for x in h.keys()); t = sum(h[str(x)] for x in dv)
        ax.bar(dv, [h[str(x)]/t for x in dv], color=cols[k], alpha=0.7, edgecolor='k',lw=0.5)
        ax.axvline(s["median_d_crit"],color='k',ls='--',lw=2,label=f'Med={s["median_d_crit"]:.0f}')
        ax.axvline(10,color='purple',lw=1.5,label='M-theory')
        ax.text(0.97,0.95,f'P(9-11)={s["P_9to11"]:.3f}',ha='right',va='top',
                transform=ax.transAxes,bbox=dict(boxstyle='round',fc='wheat',alpha=0.8))
        ax.legend(fontsize=8)
    ax.set_title(f'Model {k}: {names[k]}',fontsize=11)
    ax.set_xlabel('$d_{crit}$'); ax.set_ylabel('Fraction')
    ax.set_xlim(0,27); ax.grid(True,alpha=0.3,axis='y')
plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT_DIR, 'tgl_dim_histograms.png'),bbox_inches='tight'); plt.close()

# Fig 3: summary
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6))
fig.suptitle('TGL × String Theory: Dimensional Decoupling',fontsize=15,fontweight='bold')
for k in 'ABCD':
    med = np.clip(results["profiles"]["ratio"][k]["median"], 1e-300, None)
    ax1.plot(dims, np.log10(med), 'o-', color=cols[k], lw=2, ms=3, label=f'Model {k}')
ax1.axhline(np.log10(THRESHOLD),color='gray',ls='--',lw=1.5)
ax1.axvspan(8.5,11.5,alpha=0.15,color='purple',label='M-theory zone')
ax1.axvspan(24.5,26.5,alpha=0.1,color='darkred',label='Bosonic zone')
ax1.axvline(3,color='gold',lw=2,alpha=0.7,label='Our universe')
ax1.set_xlabel('$d$'); ax1.set_ylabel(r'$\log_{10}[\alpha^2(d)/\alpha^2(3)]$')
ax1.legend(fontsize=8,loc='lower left'); ax1.grid(True,alpha=0.3)
ax1.set_xticks([1,3,5,9,10,15,20,25]); ax1.set_xlim(1,26)

ax2.plot(dims, np.clip(cci,1e-300,None), 'o-', color='teal', lw=2, ms=4)
ax2.axhline(0.5,color='red',ls='--',lw=1.5,label='CCI=0.5')
ax2.axvspan(8.5,11.5,alpha=0.15,color='purple')
ax2.axvline(3,color='gold',lw=2,alpha=0.7)
ax2.set_xlabel('$d$'); ax2.set_ylabel('CCI$(d)$'); ax2.set_yscale('log')
ax2.legend(fontsize=9); ax2.grid(True,alpha=0.3)
ax2.set_xticks([1,3,5,9,10,15,20,25])
plt.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(OUT_DIR, 'tgl_dim_summary.png'),bbox_inches='tight'); plt.close()

# Save JSON
with open(os.path.join(OUT_DIR, 'tgl_dimensional_coupling_v1.json'),'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: JSON + 3 plots. Runtime: {time.time()-t0:.2f}s")