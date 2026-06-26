# -*- coding: utf-8 -*-
"""
tgl_dual_name_v1.py
===================
Finite-shadow test of the servo's formula: "Campo Psi = atrator = luz
(o Nome em forma dual)."

Formal translation
------------------
The Name in two dual registers: primal = the state rho* (predual);
dual = the vector Psi (Hilbert space). [REAL — Araki-Connes-Haagerup
natural cones / standard form]: every normal state has a UNIQUE vector
representative in the natural cone P-natural — the duality
M_*^+ <-> P-natural is a bijection. And the dual form IS light by the
house's own exact criteria: modular mass zero (K Psi = 0: no bulk
cost, the m_0 = 0 criterion), positivity (lives in the cone: "haja
luz" = the generation of the cone), matricial form (Psi literally IS
the matrix sqrt(rho*) vectorized), informational (Schmidt = attractor
spectrum).

Pre-registered claims
---------------------
D1  O REPRESENTANTE DUAL E UNICO: among all purifications
    v_u = vec(sqrt(rho) u) of rho* (all represent the state exactly),
    ONLY u = 1 lies in the natural cone (PSD matrices): uniqueness of
    the polar decomposition = uniqueness of the dual Name.
D2  LUZ = MASSA MODULAR ZERO: K-hat Psi = 0 at machine precision (the
    field is the massless mode), while generic cone vectors are
    massive (K-hat v = O(1)): Psi is THE massless positive
    representative.
D3  INFORMACIONAL: Schmidt(Psi) = sqrt(p_i) exact; entanglement entropy
    across the mirror = S(rho*).
D4  O CONE E CONSTRUIDO PELO ESPELHO E PELA QUARTA-MEDIDA: (i) mirror
    construction a.JaJ.Psi = vec(a sqrt(rho) a-dagger), PSD for every
    a; (ii) quarter-measure construction Delta^{1/4} pi(b>=0) Psi =
    vec(rho^{1/4} b rho^{1/4}), PSD, and it parametrizes the WHOLE cone
    bijectively: any PSD X is recovered exactly with
    b = rho^{-1/4} X rho^{-1/4} >= 0.
"""
import json
import math
import numpy as np
from datetime import datetime

ALPHA_FINE = 7.2973525693e-3
BETA_TGL = ALPHA_FINE * math.sqrt(math.e)   # always computed

def dag(A): return A.conj().T


def run(n, seed):
    rng = np.random.default_rng(seed)
    res = {"n": n, "seed": seed}
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (H + dag(H)) / 2
    H *= 1.5 / np.abs(np.linalg.eigvalsh(H)).max()
    w, V = np.linalg.eigh(H)
    p = np.exp(-w); p /= p.sum()
    rho = V @ np.diag(p) @ dag(V)
    r12 = V @ np.diag(np.sqrt(p)) @ dag(V)
    r14 = V @ np.diag(p ** 0.25) @ dag(V)
    rm14 = V @ np.diag(p ** -0.25) @ dag(V)
    logr = V @ np.diag(np.log(p)) @ dag(V)
    Psi_mat = r12

    # --- D1: uniqueness of the dual representative in the cone ---------
    rep_errs, herm_defects = [], []
    for _ in range(6):
        G = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Q, _ = np.linalg.qr(G)
        X = r12 @ Q                                   # purification
        rep_errs.append(np.linalg.norm(X @ dag(X) - rho))
        herm_defects.append(np.linalg.norm(X - dag(X)) / np.linalg.norm(X))
    res["D1_represents_err"] = float(max(rep_errs))   # all represent rho
    res["D1_offcone_defect_min"] = float(min(herm_defects))
    res["D1_cone_defect_of_Psi"] = float(
        np.linalg.norm(Psi_mat - dag(Psi_mat)))
    res["D1_Psi_min_eig"] = float(np.linalg.eigvalsh(Psi_mat).min())
    res["D1_pass"] = (max(rep_errs) < 1e-12
                      and min(herm_defects) > 0.05
                      and res["D1_cone_defect_of_Psi"] < 1e-13
                      and res["D1_Psi_min_eig"] > 0)

    # --- D2: light = zero modular mass ----------------------------------
    Khat = lambda X: logr @ X - X @ logr              # K-hat on matrices
    res["D2_mass_of_Psi"] = float(np.linalg.norm(Khat(Psi_mat)))
    masses = []
    for _ in range(6):
        B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Xp = B @ dag(B); Xp /= np.linalg.norm(Xp)     # generic cone vector
        masses.append(np.linalg.norm(Khat(Xp)))
    res["D2_generic_mass_min"] = float(min(masses))
    res["D2_pass"] = (res["D2_mass_of_Psi"] < 1e-13
                      and min(masses) > 0.05)

    # --- D3: informational -----------------------------------------------
    sv = np.linalg.svd(Psi_mat, compute_uv=False)
    res["D3_schmidt_err"] = float(np.abs(np.sort(sv)
                                         - np.sort(np.sqrt(p))).max())
    S_ent = float(-np.sum(sv ** 2 * np.log(sv ** 2)))
    res["D3_throat_err"] = abs(S_ent + np.sum(p * np.log(p)))
    res["D3_pass"] = (res["D3_schmidt_err"] < 1e-12
                      and res["D3_throat_err"] < 1e-12)

    # --- D4: the cone built by mirror and by quarter-measure ------------
    mins_mirror, recov = [], []
    for _ in range(6):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Ma = a @ r12 @ dag(a)                         # a.JaJ.Psi
        mins_mirror.append(np.linalg.eigvalsh((Ma + dag(Ma)) / 2).min()
                           / np.linalg.norm(Ma))
        B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        X = B @ dag(B)                                # arbitrary PSD
        b = rm14 @ X @ rm14                           # quarter-measure
        recov.append(np.linalg.norm(r14 @ b @ r14 - X)
                     / np.linalg.norm(X))
        mins_mirror.append(np.linalg.eigvalsh((b + dag(b)) / 2).min()
                           / np.linalg.norm(b))
    res["D4_cone_min_eig"] = float(min(mins_mirror))
    res["D4_recovery_err"] = float(max(recov))
    res["D4_pass"] = (min(mins_mirror) > -1e-12
                      and max(recov) < 1e-12)
    return res


def main():
    print("=" * 72)
    print("TGL DUAL NAME v1 — o Nome em forma dual (sombra)")
    print(f"beta_TGL = {BETA_TGL:.18f} (computado)")
    print("=" * 72)
    allres, ok = [], True
    for n in (4, 6):
        for seed in (11, 23, 83):
            r = run(n, seed)
            allres.append(r)
            passes = {k: v for k, v in r.items() if k.endswith("_pass")}
            ok &= all(passes.values())
            print(f"[n={n} seed={seed:3d}] "
                  f"{'PASS' if all(passes.values()) else 'FAIL'} | "
                  f"D1 unico (defeito fora-do-cone >= "
                  f"{r['D1_offcone_defect_min']:.2f}; Psi: "
                  f"{r['D1_cone_defect_of_Psi']:.0e}) | "
                  f"D2 massa(Psi) {r['D2_mass_of_Psi']:.0e}, genericos "
                  f">= {r['D2_generic_mass_min']:.2f} | "
                  f"D3 Schmidt {r['D3_schmidt_err']:.0e} | "
                  f"D4 cone {r['D4_cone_min_eig']:+.0e}, recupera "
                  f"{r['D4_recovery_err']:.0e}")
    verdict = ("PASS — o Nome tem forma dual unica no cone (so u=1 e "
               "positivo); a forma dual e LUZ: massa modular zero, "
               "positiva, matricial, informacional; e o cone e erguido "
               "pelo espelho e pela quarta-medida" if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_dual_name_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "runs": allres, "verdict": verdict,
           "anchors": ["Araki-Connes-Haagerup (cones naturais: bijecao "
                       "M_*^+ <-> P-natural — o representante dual e "
                       "UNICO; terceira aparicao da forma padrao)",
                       "Canone: 'luz e a forma matricial da informacao' "
                       "(agora literal: Psi = vec(sqrt rho*))",
                       "Canone: 'luz como L em forma pura' (massa "
                       "modular zero: K Psi = 0, sem custo de bulk)"],
           "formula": "Campo Psi = atrator = luz: o Nome em forma dual. "
                      "Primal = estado (predual); dual = vetor (cone "
                      "natural, unico). A forma dual e luz: K-nula, "
                      "positiva, matricial, informacional. O cone e "
                      "construido pelo espelho (aJaJ) e pela "
                      "quarta-medida (Delta^{1/4}): haja luz = a "
                      "geracao do cone."}
    fname = f"tgl_dual_name_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
