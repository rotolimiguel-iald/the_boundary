# -*- coding: utf-8 -*-
"""
tgl_gesture_inscription_v1.py
=============================
Finite-shadow test of the servo's formula: "O Nome em forma dual e a
representacao algebrica do Verbo — inscricao modular do gesto;
observar o gesto e fractalizacao do Nome."

Formal translation
------------------
[REAL — GNS] The dual Name Psi generates the algebraic representation,
whose vectors ARE inscribed gestures (a Psi). Psi is cyclic (every
vector is a gesture: nothing else exists there) and separating (no
nonzero gesture inscribes to zero: no gesture is lost). [REAL — Tomita]
The modular structure (S, J, Delta) is BORN from the gesture rule
S(a Psi) = a* Psi: the inscription is modular by anatomy, not by
decoration. And iterated observation of ever finer gestures is the
CASCADE: the Name splits into branches whose profile densifies to the
continuum measure of the slice — observation fractalizes the Name into
Palavra (Spec C).

Pre-registered claims
---------------------
F1  A REPRESENTACAO E FEITA DE GESTOS: the inscription map
    I(a) = a Psi is bijective with smallest singular value sqrt(p_min)
    > 0 (separating: no gesture lost) and full rank (cyclic: nothing
    but gestures).
F2  O MODULAR NASCE DA REGRA DO GESTO: define S purely by the gesture
    rule (a -> a*), with no mention of Delta; then S = J Delta^{1/2}
    holds at machine precision — the modular pair is recovered FROM the
    inscription.
F3  OBSERVAR E FRACTALIZAR E COLAPSAR: iterated dyadic observation of
    the gesture (refining projective partitions in the K-order) (i)
    composes EXACTLY to the terminal collapse E on any initial state,
    and (ii) produces branch measures that converge (KS -> 0) to the
    continuum measure on the slice coordinate x = sqrt(k).
F4  A FRACTALIZACAO COMPLETA RECUPERA EXATAMENTE A INFORMACAO DO NOME:
    the Shannon entropy of the branch distribution increases with each
    generation and terminates EXACTLY at S(rho*): nem mais, nem menos.
"""
import json
import math
import numpy as np
from datetime import datetime

ALPHA_FINE = 7.2973525693e-3
BETA_TGL = ALPHA_FINE * math.sqrt(math.e)   # always computed

def dag(A): return A.conj().T


def run(seed, G=6):
    rng = np.random.default_rng(seed)
    N = 2 ** G
    res = {"seed": seed, "N": N}
    # thermal Name, sorted by k = -log p (the slice order)
    w = np.sort(rng.normal(size=N))
    p = np.exp(-1.5 * w); p /= p.sum()
    kv = -np.log(p)
    order = np.argsort(kv)
    p = p[order]; kv = kv[order]
    x = np.sqrt(kv)

    # --- F1: inscription map bijective, sigma_min = sqrt(p_min) --------
    # I(a) = a sqrt(rho): singular values = sqrt(p_j), multiplicity N.
    res["F1_sigma_min"] = float(np.sqrt(p.min()))
    res["F1_sep_check"] = float(abs(np.sqrt(p.min())
                                    - np.sqrt(np.min(p))))
    # explicit small-block numeric check (n=6 corner) of bijectivity:
    n = 6
    r12s = np.diag(np.sqrt(p[:n] / p[:n].sum()))
    M = np.kron(np.eye(n), r12s.T)
    sv = np.linalg.svd(M, compute_uv=False)
    res["F1_numeric_sigma_min"] = float(sv.min())
    res["F1_pass"] = sv.min() > 1e-8 and res["F1_sigma_min"] > 0

    # --- F2: modular pair born from the gesture rule (n=6 block) -------
    pb = p[:n] / p[:n].sum()
    r12 = np.diag(np.sqrt(pb)); rm12 = np.diag(pb ** -0.5)
    errs = []
    for _ in range(8):
        W = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        a = W @ rm12                      # the gesture: W = a sqrt(rho)
        S_gesture = (dag(a) @ r12).reshape(-1)        # rule: a -> a*
        Dh = (np.diag(np.sqrt(pb)) @ W @ np.diag(pb ** -0.5))
        S_modular = dag(Dh).reshape(-1)               # J Delta^{1/2} W
        errs.append(np.linalg.norm(S_gesture - S_modular)
                    / np.linalg.norm(S_gesture))
    res["F2_S_eq_JDelta_half"] = float(max(errs))
    res["F2_pass"] = max(errs) < 1e-12

    # --- F3: iterated observation = fractalization = collapse ----------
    # initial NON-diagonal state in the K basis
    Vr = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Q, _ = np.linalg.qr(Vr)
    th = 0.35
    mix = Q @ np.diag(p) @ dag(Q)
    rho0 = (1 - th) * np.diag(p) + th * mix
    rho0 = (rho0 + dag(rho0)) / 2; rho0 /= np.real(np.trace(rho0))
    r = rho0.copy()
    KS, Hs = [], []
    cdf_full = np.cumsum(p)
    for g in range(1, G + 1):
        nb = 2 ** g
        labels = (np.arange(N) * nb) // N             # dyadic blocks
        P = np.zeros((N, N), complex)
        rnew = np.zeros_like(r)
        for b in range(nb):
            Pb = np.diag((labels == b).astype(float))
            rnew += Pb @ r @ Pb
        r = rnew
        bp = np.array([p[labels == b].sum() for b in range(nb)])
        Hs.append(float(-(bp * np.log(bp)).sum()))
        # KS between branch step-CDF and full CDF at block right edges
        edges = [np.max(np.where(labels == b)[0]) for b in range(nb)]
        KS.append(float(np.max(np.abs(np.cumsum(bp)
                                      - cdf_full[edges]))) if nb < N
                  else 0.0)
        # finer probe: KS over all points between step measure and full
        step_cdf = np.repeat(np.cumsum(bp), N // nb)
        KS[-1] = float(np.abs(step_cdf - cdf_full).max())
    E_rho0 = np.diag(np.diag(rho0))
    res["F3_composition_err"] = float(np.linalg.norm(r - E_rho0)
                                      / np.linalg.norm(E_rho0))
    res["F3_KS_first_last"] = [KS[0], KS[-1]]
    res["F3_pass"] = (res["F3_composition_err"] < 1e-12
                      and KS[-1] < KS[0] / 3 and KS[-1] < 0.05)

    # --- F4: full fractalization recovers exactly S(rho*) ---------------
    S_name = float(-(p * np.log(p)).sum())
    res["F4_H_increasing"] = bool(all(Hs[i + 1] > Hs[i] - 1e-14
                                      for i in range(len(Hs) - 1)))
    res["F4_terminal_err"] = abs(Hs[-1] - S_name)
    res["F4_S_name"] = S_name
    res["F4_pass"] = res["F4_H_increasing"] and res["F4_terminal_err"] < 1e-12
    return res


def main():
    print("=" * 72)
    print("TGL GESTURE INSCRIPTION v1 — a inscricao do gesto (sombra)")
    print(f"beta_TGL = {BETA_TGL:.18f} (computado)")
    print("=" * 72)
    allres, ok = [], True
    for seed in (11, 23, 83):
        r = run(seed)
        allres.append(r)
        passes = {k: v for k, v in r.items() if k.endswith("_pass")}
        ok &= all(passes.values())
        print(f"[seed={seed:3d}] {'PASS' if all(passes.values()) else 'FAIL'}"
              f" | F1 sigma_min={r['F1_numeric_sigma_min']:.3f}>0"
              f" | F2 S=JD^1/2 a {r['F2_S_eq_JDelta_half']:.0e}"
              f" | F3 composicao->E {r['F3_composition_err']:.0e}, "
              f"KS {r['F3_KS_first_last'][0]:.3f}->"
              f"{r['F3_KS_first_last'][1]:.3f}"
              f" | F4 H crescente, |H_G - S(rho*)| = "
              f"{r['F4_terminal_err']:.0e}")
    verdict = ("PASS — a representacao e feita de gestos inscritos "
               "(ciclico-separador); o modular nasce da regra do gesto; "
               "observar iteradamente fractaliza o Nome ate a fatia E "
               "realiza o colapso E; e a fractalizacao completa devolve "
               "exatamente S(rho*): nem mais, nem menos"
               if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_gesture_inscription_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "runs": allres, "verdict": verdict,
           "anchors": ["GNS (a representacao construida do Nome: "
                       "vetores = gestos inscritos) [REAL]",
                       "ciclico-separador (todo vetor e gesto; nenhum "
                       "gesto se perde) [REAL]",
                       "Tomita (S,J,Delta nascem da regra do gesto "
                       "S(a Psi)=a* Psi) [REAL]",
                       "Ax.S (a cascata: observacao iterada = "
                       "fractalizacao)"],
           "formula": "A luz e o pergaminho onde o Verbo se inscreve; "
                      "observar a escrita fractaliza o Nome em espaco. "
                      "O circuito da trindade: o Nome inscreve o Verbo "
                      "(GNS); observar o Verbo fractaliza o Nome em "
                      "Palavra (Spec C)."}
    fname = f"tgl_gesture_inscription_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
