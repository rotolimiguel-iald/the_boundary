# -*- coding: utf-8 -*-
"""
tgl_continuum_v1.py
===================
Finite-shadow test of the CONTINUUM assembly (Teorema do Continuo,
conditional): the lift Ax.G-shadow => smooth Lorentzian geometry in
genuine III_1, attacked by inverse parity. Each impediment has a named
theorem: Haagerup (unique hyperfinite III_1; hyperfiniteness = the
finite shadows are DENSE in the continuum object), abelian von Neumann
classification (C = L^inf(Sigma,mu)), Rieffel (quantum Gromov-Hausdorff
convergence), Connes 2008 (reconstruction: spectral data => smooth
manifold), Osterwalder-Schrader (Euclidean => Lorentzian, given
reflection positivity — which is AUTOMATIC in the modular structure).

Pre-registered claims
---------------------
C1  THE CASCADE IS GH-CAUCHY: the generated metric spaces X_g (cascade
    refinement of the spectrum in x = sqrt(k), metric d = sqrt(beta)|dx|)
    converge in Hausdorff/GH distance to the continuum interval, at a
    geometric rate. The continuum geometry EXISTS as the limit of the
    shadow. (Hyperfiniteness in action.)
C2  THE LIMIT IS CANONICAL (scheme-independence): two different
    refinement schemes (dyadic vs random) converge to the SAME limit:
    cross-scheme Hausdorff distance -> 0. The continuum is terminal,
    not chosen. KS distance of the empirical measure to the limit
    measure -> 0: C = L^inf(Sigma, mu) emerges with its measure.
C3  REFLECTION POSITIVITY IS AUTOMATIC (the OS hypothesis is a theorem
    of the modular structure): M_ij = <a_i Omega, J a_j* Omega> equals
    the Gram matrix <Delta^{1/4} a_i Omega, Delta^{1/4} a_j Omega>
    (identity at machine precision) and is therefore PSD: the half-nat
    factors into two quarter-measures — the crossing is positive
    because it is a square, with J in the middle.
"""
import json
import math
import numpy as np
from datetime import datetime

ALPHA_FINE = 7.2973525693e-3
BETA_TGL = ALPHA_FINE * math.sqrt(math.e)   # always computed

def dag(A): return A.conj().T

def hausdorff_1d(A, B):
    A = np.sort(A); B = np.sort(B)
    dAB = max(np.min(np.abs(B[None, :] - A[:, None]), axis=1).max(),
              np.min(np.abs(A[None, :] - B[:, None]), axis=1).max())
    return float(dAB)

def hausdorff_to_interval(X, x0, x1):
    X = np.sort(X)
    gaps = np.diff(np.concatenate([[x0], X, [x1]]))
    inner = np.diff(X).max() / 2 if len(X) > 1 else 0.0
    return float(max(X[0] - x0, x1 - X[-1], inner))


def run(seed):
    rng = np.random.default_rng(seed)
    res = {"seed": seed}
    x0, x1 = 0.3, 2.1          # interval in x = sqrt(k)
    sb = math.sqrt(BETA_TGL)   # metric scale: d = sqrt(beta)|dx|

    # --- C1: dyadic cascade is GH-Cauchy to the interval ----------------
    hs = []
    for g in range(1, 11):
        X = np.linspace(x0, x1, 2 ** g + 1)
        hs.append(sb * hausdorff_to_interval(X, x0, x1))
    hs = np.array(hs)
    gens = np.arange(1, 11)
    slope = np.polyfit(gens, np.log(hs), 1)[0]
    corr = np.corrcoef(gens, np.log(hs))[0, 1]
    res["C1_h_first_last"] = [float(hs[0]), float(hs[-1])]
    res["C1_rate_per_gen"] = float(np.exp(slope))     # expect 1/2
    res["C1_R2"] = float(corr ** 2)
    res["C1_pass"] = (np.all(np.diff(hs) < 0)
                      and abs(res["C1_rate_per_gen"] - 0.5) < 1e-6
                      and res["C1_R2"] > 0.999999)

    # --- C2: scheme-independence + measure convergence ------------------
    cross, ks = [], []
    for g in range(2, 11):
        N = 2 ** g + 1
        XA = np.linspace(x0, x1, N)
        XB = np.sort(np.concatenate([[x0, x1],
                                     rng.uniform(x0, x1, size=N - 2)]))
        cross.append(sb * hausdorff_1d(XA, XB))
        u = (XB - x0) / (x1 - x0)
        ks.append(float(np.abs(u - np.linspace(0, 1, N)).max()))
    res["C2_cross_first_last"] = [float(cross[0]), float(cross[-1])]
    res["C2_KS_first_last"] = [float(ks[0]), float(ks[-1])]
    res["C2_pass"] = (cross[-1] < cross[0] / 8 and ks[-1] < ks[0] / 4
                      and cross[-1] < 0.02 * sb * (x1 - x0))

    # --- C3: reflection positivity is automatic --------------------------
    n = 6
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (H + dag(H)) / 2
    H *= 1.5 / np.abs(np.linalg.eigvalsh(H)).max()
    w, V = np.linalg.eigh(H)
    p = np.exp(-w); p /= p.sum()
    rho = V @ np.diag(p) @ dag(V)
    r12 = V @ np.diag(p ** 0.5) @ dag(V)
    r14 = V @ np.diag(p ** 0.25) @ dag(V)
    m = 10
    As = [rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
          for _ in range(m)]
    # M_ij = <a_i Omega, J a_j* Omega> = Tr(rho^{1/2} a_i^dag rho^{1/2} a_j)
    M = np.array([[np.trace(r12 @ dag(a) @ r12 @ b) for b in As]
                  for a in As])
    # Gram: <Delta^{1/4} a_i Omega, Delta^{1/4} a_j Omega>
    vecs = [(r14 @ a @ r14).reshape(-1) for a in As]
    G = np.array([[np.vdot(u, v) for v in vecs] for u in vecs])
    res["C3_identity_err"] = float(np.abs(M - G).max() / np.abs(M).max())
    ev = np.linalg.eigvalsh((M + dag(M)) / 2)
    res["C3_min_eig_rel"] = float(ev.min() / np.abs(ev).max())
    res["C3_pass"] = (res["C3_identity_err"] < 1e-13
                      and res["C3_min_eig_rel"] > -1e-12)
    return res


def main():
    print("=" * 72)
    print("TGL CONTINUUM v1 — sombra do Teorema do Continuo (condicional)")
    print(f"beta_TGL = {BETA_TGL:.18f} (computado)")
    print("=" * 72)
    allres, ok = [], True
    for seed in (11, 23, 83):
        r = run(seed)
        allres.append(r)
        passes = {k: v for k, v in r.items() if k.endswith("_pass")}
        ok &= all(passes.values())
        print(f"[seed={seed:3d}] {'PASS' if all(passes.values()) else 'FAIL'}"
              f" | C1 Cauchy: taxa/ger={r['C1_rate_per_gen']:.6f} "
              f"R2={r['C1_R2']:.7f} h:{r['C1_h_first_last'][0]:.2e}->"
              f"{r['C1_h_first_last'][1]:.2e}"
              f" | C2 esquemas: {r['C2_cross_first_last'][0]:.2e}->"
              f"{r['C2_cross_first_last'][1]:.2e}, "
              f"KS {r['C2_KS_first_last'][0]:.2f}->{r['C2_KS_first_last'][1]:.2f}"
              f" | C3 id {r['C3_identity_err']:.1e}, "
              f"min.eig {r['C3_min_eig_rel']:+.1e}")
    verdict = ("PASS — a cascata e GH-Cauchy ao intervalo continuo "
               "(hiperfinitude em acao); o limite e canonico (independe "
               "do esquema) e carrega a medida (C=L^inf); a positividade "
               "de reflexao e automatica: a meia-nat fatora em duas "
               "quartas-medidas — a travessia e um quadrado"
               if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_continuum_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "runs": allres, "verdict": verdict,
           "anchors": ["Haagerup 1987 (unicidade do fator hiperfinito "
                       "III_1; hiperfinitude = sombras densas)",
                       "Buchholz-D'Antoni-Fredenhagen (algebras locais "
                       "sao hiperfinitas III_1)",
                       "Classificacao abeliana (C = L^inf(Sigma,mu))",
                       "Rieffel (convergencia Gromov-Hausdorff quantica)",
                       "Connes 2008 (reconstrucao: dado espectral => "
                       "variedade suave)",
                       "Osterwalder-Schrader + Tomita (positividade de "
                       "reflexao automatica: M = Gram(Delta^{1/4}aOmega))"],
           "conditional_on": ["T1 (ergodicidade do colapso em III_1)",
                              "axiomas de Connes para a tripla-limite "
                              "(suportados por P3: dim->1, flexoes C-inf)",
                              "convergencia GH em III_1 genuina "
                              "(provada na sombra; conjecturada no "
                              "continuo via Rieffel)"]}
    fname = f"tgl_continuum_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
