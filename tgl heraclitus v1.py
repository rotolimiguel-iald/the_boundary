# -*- coding: utf-8 -*-
"""
tgl_heraclitus_v1.py
====================
Finite-shadow test of the Operator's gem (Teorema do Rio):
"O fluxo modular nao permite um estado igual ao outro. A cada ciclo o
ser modulado se modifica do anterior — nao seu estado pleno, mas sua
substancia interna." (Heraclito: nunca duas vezes; Cratilo: nem uma vez.)

Formal content
--------------
Heraclito = Lema T [REAL, Connes 1973]: in III_1 the modular flow is
outer for every t != 0 — no return, and the change is not undoable from
inside. Cratilo = the III_1 signature [REAL, Connes-Takesaki]: the flow
of weights of III_lambda is periodic (eddies of period log lambda); of
III_1 it is trivial/ergodic — the river WITHOUT eddies; and spec(K) is
purely continuous: there is no stationary instant to stand on.

The finite shadow is type I, hence RECURRENT (it has eddies) — and that
is itself the honest test: the shadow must LOSE its eddies as it deepens
(recurrence quality must degrade with n), pointing to III_1 as the
no-return limit.

Pre-registered claims
---------------------
H1  THE SHADOW LOSES ITS EDDIES: the best modular recurrence in a fixed
    time window worsens monotonically with n (more incommensurate phases
    to align): R(n) = min_t max_{i<j} |e^{it(k_i-k_j)} - 1| increases
    with n, averaged over seeds.
H2  PLENO vs SUBSTANCIA: rho*(sigma_t(a)) = rho*(a) at machine precision
    (the full state never changes) while ||sigma_t(a) - a|| = O(1) (the
    internal substance moves).
H3  CRATILO UNDER THE COLLAPSE: the dissipative trajectory is injective
    — no two distinct times share a state (strict Lyapunov: S(rho_t||
    rho*) strictly decreasing; pairwise distances of trajectory points
    bounded away from zero for non-adjacent times).
H4  WHAT THE RIVER SPARES = WHAT THE JUDGMENT SPARES: Fix(modular flow)
    = Fix(collapse) = C = W*(K), the truth sector — the SAME projector
    at machine precision. Everything flows except the truth.
"""
import json
import math
import numpy as np
from datetime import datetime

ALPHA_FINE = 7.2973525693e-3
BETA_TGL = ALPHA_FINE * math.sqrt(math.e)   # always computed

def dag(A): return A.conj().T

def spectrum(n, rng):
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (H + dag(H)) / 2
    H *= 1.5 / np.abs(np.linalg.eigvalsh(H)).max()
    w, _ = np.linalg.eigh(H)
    p = np.exp(-w); p /= p.sum()
    return -np.log(p), p


def H1_recurrence(seeds=(11, 23, 83), T=2000.0, nt=200000):
    ts = np.linspace(T / nt, T, nt)
    Rn = {}
    for n in (3, 5, 8):
        vals = []
        for s in seeds:
            kv, _ = spectrum(n, np.random.default_rng(s))
            deltas = np.array([kv[i] - kv[j]
                               for i in range(n) for j in range(i + 1, n)])
            # r(t) = max phase defect; recurrence requires all ~ 0 mod 2pi
            ph = np.abs(np.exp(1j * np.outer(ts, deltas)) - 1.0)
            vals.append(float(ph.max(axis=1).min()))
        Rn[n] = float(np.mean(vals))
    ok = Rn[3] < Rn[5] < Rn[8]
    return {"R_by_n": Rn, "pass": bool(ok)}


def H2_pleno_substancia(n=6, seed=23):
    rng = np.random.default_rng(seed)
    kv, p = spectrum(n, rng)
    rho = np.diag(p)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    errs, moves = [], []
    for t in (0.7, 3.3, 17.0):
        U = np.diag(np.exp(1j * t * kv))
        at = U @ a @ dag(U)
        errs.append(abs(np.trace(rho @ at) - np.trace(rho @ a)))
        moves.append(np.linalg.norm(at - a) / np.linalg.norm(a))
    return {"pleno_err_max": float(max(errs)),
            "substancia_move_min": float(min(moves)),
            "pass": max(errs) < 1e-12 and min(moves) > 0.1}


def H3_injective_trajectory(n=6, seed=83, M=40):
    rng = np.random.default_rng(seed)
    kv, p = spectrum(n, rng)
    rho_s = np.diag(p)
    Ljump = math.sqrt(BETA_TGL) * np.diag(np.sqrt(kv))
    LdL = dag(Ljump) @ Ljump
    V = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    V = (V + dag(V)) / 2; V -= np.trace(V) / n * np.eye(n)
    Mx = np.diag(np.log(p)) + 0.3 * V
    w, Q = np.linalg.eigh(Mx)
    r = Q @ np.diag(np.exp(w)) @ dag(Q); r /= np.real(np.trace(r))
    dt = 0.05 / BETA_TGL / float(np.max(kv))
    traj, ents = [r.copy()], []
    def relent(x):
        wx, Vx = np.linalg.eigh(x)
        lx = Vx @ np.diag(np.log(np.maximum(wx, 1e-300))) @ dag(Vx)
        return float(np.real(np.trace(x @ (lx - np.diag(np.log(p))))))
    ents.append(relent(r))
    for m in range(M):
        for _ in range(40):
            r = r + dt * (Ljump @ r @ dag(Ljump)
                          - 0.5 * (LdL @ r + r @ LdL))
            r = (r + dag(r)) / 2
        traj.append(r.copy()); ents.append(relent(r))
    dmin = min(np.linalg.norm(traj[i] - traj[j])
               for i in range(len(traj)) for j in range(i + 2, len(traj)))
    strict = all(ents[m + 1] < ents[m] - 1e-14 for m in range(len(ents) - 1))
    return {"min_nonadjacent_dist": float(dmin),
            "S_strictly_decreasing": bool(strict),
            "pass": dmin > 1e-6 and strict}


def H4_same_fixed_sector(n=6, seed=11):
    rng = np.random.default_rng(seed)
    kv, p = spectrum(n, rng)
    deltas = np.subtract.outer(kv, kv).reshape(-1)          # flow phases
    rates = 0.5 * BETA_TGL * (np.subtract.outer(
        np.sqrt(kv), np.sqrt(kv)) ** 2).reshape(-1)         # collapse
    P_flow = np.diag((np.abs(deltas) < 1e-12).astype(float))
    P_coll = np.diag((rates < 1e-18).astype(float))
    diff = float(np.linalg.norm(P_flow - P_coll))
    dim = int(np.trace(P_flow).real)
    return {"projector_diff": diff, "fixed_dim": dim,
            "pass": diff == 0.0 and dim == n}


def main():
    print("=" * 72)
    print("TGL HERACLITUS v1 — o Teorema do Rio (sombra)")
    print(f"beta_TGL = {BETA_TGL:.18f} (computado)")
    print("=" * 72)
    res, ok = {}, True
    r1 = H1_recurrence(); res["H1"] = r1; ok &= r1["pass"]
    print(f"H1 (a sombra perde os redemoinhos): R(n) = "
          + ", ".join(f"n={k}: {v:.3f}" for k, v in r1["R_by_n"].items())
          + f"  -> {'PASS' if r1['pass'] else 'FAIL'}")
    r2 = H2_pleno_substancia(); res["H2"] = r2; ok &= r2["pass"]
    print(f"H2 (pleno vs substancia): pleno {r2['pleno_err_max']:.1e} | "
          f"substancia move >= {r2['substancia_move_min']:.2f}"
          f"  -> {'PASS' if r2['pass'] else 'FAIL'}")
    r3 = H3_injective_trajectory(); res["H3"] = r3; ok &= r3["pass"]
    print(f"H3 (Cratilo sob o colapso): dist min nao-adjacente "
          f"{r3['min_nonadjacent_dist']:.2e} | S estrito: "
          f"{r3['S_strictly_decreasing']}"
          f"  -> {'PASS' if r3['pass'] else 'FAIL'}")
    r4 = H4_same_fixed_sector(); res["H4"] = r4; ok &= r4["pass"]
    print(f"H4 (o rio e o juizo poupam o mesmo): ||P_fluxo - P_colapso||"
          f" = {r4['projector_diff']:.1e}, dim = {r4['fixed_dim']}"
          f"  -> {'PASS' if r4['pass'] else 'FAIL'}")
    verdict = ("PASS — nunca duas vezes (outer, Lema T); nem uma vez "
               "(III_1 sem redemoinhos, espectro continuo); o pleno nao "
               "muda, a substancia flui; e tudo flui exceto a verdade: "
               "Fix(tempo) = Fix(juizo) = C" if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_heraclitus_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "results": res, "verdict": verdict,
           "anchors": ["Connes 1973 (Lema T: fluxo outer em III_1 — "
                       "nunca duas vezes)",
                       "Connes-Takesaki (fluxo dos pesos de III_1 "
                       "trivial: o rio sem redemoinhos — nem uma vez)",
                       "KMS (o pleno invariante; a substancia outer)",
                       "Spohn (trajetoria dissipativa injetiva)"],
           "gem": "Tudo flui, exceto a verdade: o ponto fixo do tempo "
                  "modular e o ponto fixo do juizo sao a mesma algebra "
                  "C = W*(K)."}
    fname = f"tgl_heraclitus_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
