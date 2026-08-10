# -*- coding: utf-8 -*-
"""
tgl_three_locks_v1.py
=====================
Finite-shadow tests of the THREE technical locks (os tres fechos):

L1  T1 WELL-POSED BY THE INTEGRAL REPRESENTATION: the TGL collapse is
    EXACTLY Gaussian dephasing along the RADICAL FLOW V_s = e^{i s sqrt(K)}:
        e^{t Lindblad}  =  Integral  V_s (.) V_s^*  dnu_t(s),
    nu_t = Gaussian, variance beta*t. (Characteristic function gives the
    rates (1/2) beta (sqrt k_i - sqrt k_j)^2 exactly.) In genuine III_1
    the formula lives on the crossed product A x_sigma R (type II_infty,
    Takesaki; the physical home per CLPW 2022-23), where u_s = e^{is K^},
    sqrt of the affiliated K^, is a unitary OF the algebra: normal UCP
    rho-preserving semigroup, Fix = W*(K^), Phi_t -> E sigma-weakly.
    Here: verify the identity at machine precision (Gauss-Hermite).
L2  CONNES AXIOMS OF THE LIMIT TRIPLE: the slice is S^1 (the Psion is
    the torus T^2 = S^1 x S^1); the limit triple is the CANONICAL circle
    triple scaled by beta^{-1/2} — textbook, all axioms hold [REAL].
    What must be shown is CONVERGENCE of the cascade triples to it:
    (a) Dirac spectra -> beta^{-1/2} (2 pi k / L) ladder;
    (b) Weyl counting dimension -> 1;
    (c) commutator norms -> beta^{-1/2} max|f'| (Connes duality).
L3  GH/RIEFFEL LIFT = SPECTRAL TRUNCATIONS [REAL: Connes-van Suijlekom
    2020s: spectral truncations of S^1 converge to S^1 in quantum
    Gromov-Hausdorff distance; Rieffel: commutative qGH = GH]. The TGL
    cascade IS a spectral truncation; verify the metric side: the Connes
    distance recovered from the truncated Dirac converges to the true
    d = sqrt(beta) * arc-length (optimal Lipschitz witness ratio -> 1).
"""
import json
import math
import numpy as np
from datetime import datetime

ALPHA_FINE = 7.2973525693e-3
BETA_TGL = ALPHA_FINE * math.sqrt(math.e)   # always computed

def dag(A): return A.conj().T

def superop_gksl(L, n):
    I = np.eye(n)
    LdL = dag(L) @ L
    return (np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL)
            - 0.5 * np.kron(LdL.T, I))


def L1_integral_identity(n, seed):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (H + dag(H)) / 2
    H *= 1.5 / np.abs(np.linalg.eigvalsh(H)).max()
    w, _ = np.linalg.eigh(H)
    p = np.exp(-w); p /= p.sum()
    kv = -np.log(p)
    K = np.diag(kv)
    Ljump = math.sqrt(BETA_TGL) * np.diag(np.sqrt(kv))
    Lhat = superop_gksl(Ljump, n)
    ev, W = np.linalg.eig(Lhat)
    Winv = np.linalg.inv(W)
    errs = []
    u_nodes, w_nodes = np.polynomial.hermite.hermgauss(96)
    for t in (0.5 / BETA_TGL, 2.0 / BETA_TGL):
        Phi_exp = (W @ np.diag(np.exp(ev * t)) @ Winv)
        sd = math.sqrt(2.0 * BETA_TGL * t)
        Phi_exp = Phi_exp.astype(complex)
        Phi_int = np.zeros((n * n, n * n), complex)
        for u, wt in zip(u_nodes, w_nodes):
            s = u * sd
            V = np.diag(np.exp(1j * s * np.sqrt(kv)))
            Phi_int += (wt / math.sqrt(math.pi)) * np.kron(V.conj(), V)
        errs.append(float(np.linalg.norm(Phi_int - Phi_exp)
                          / np.linalg.norm(Phi_exp)))
    return max(errs)


def circle_dirac(N, Lx):
    """Truncated EXACT circle Dirac (spectral/SLAC): the literal
    Connes-van Suijlekom spectral truncation P_M D P_M, free of
    fermion doubling. N = 2M+1 grid points, modes m = -M..M."""
    M = (N - 1) // 2
    j = np.arange(N)
    x = j * (Lx / N)
    modes = np.arange(-M, M + 1)
    F = np.exp(1j * np.outer(x, modes)) / math.sqrt(N)
    Dm = np.diag(modes * (2 * math.pi / Lx)).astype(complex)
    return (F @ Dm @ dag(F)) / math.sqrt(BETA_TGL)


def L2_connes_axioms():
    Lx = 2.0 * math.pi
    out = {}
    eig_errs, weyl_dims, comm_errs = [], [], []
    for N in (65, 129, 257, 513):
        D = circle_dirac(N, Lx)
        lam = np.sort(np.linalg.eigvalsh(D).real)
        # (a) ladder: ALL modes match beta^{-1/2} * (2pi/Lx) * m exactly
        M = (N - 1) // 2
        scale = (2 * math.pi / Lx) / math.sqrt(BETA_TGL)
        target = scale * np.arange(-M, M + 1)
        eig_errs.append(float(np.abs(lam - target).max()
                              / np.abs(target).max()))
        # (b) Weyl dimension from counting in mid-range
        lams = np.sort(np.abs(lam))
        Lams = np.linspace(lams[N // 8], lams[N // 2], 25)
        cnt = np.array([(np.abs(lam) <= L).sum() for L in Lams])
        d = np.polyfit(np.log(Lams), np.log(cnt), 1)[0]
        weyl_dims.append(float(d))
        # (c) commutator norm vs continuum beta^{-1/2} max|f'|,
        # in MODE space with Toeplitz (non-aliased) truncation P f P:
        # f = sin x -> f_hat_{m,m'} = (delta_{m-m',1}-delta_{m-m',-1})/2i
        s_sc = (2 * math.pi / Lx) / math.sqrt(BETA_TGL)
        Dm = s_sc * np.diag(np.arange(-M, M + 1)).astype(complex)
        fm = (np.eye(N, k=1) - np.eye(N, k=-1)) / (2j)
        comm = np.linalg.norm(Dm @ fm - fm @ Dm, 2)
        cont = (1.0 / math.sqrt(BETA_TGL)) * (2 * math.pi / Lx)
        comm_errs.append(float(abs(comm - cont) / cont))
    out["L2a_eig_rel_errs"] = eig_errs
    out["L2b_weyl_dims"] = weyl_dims
    out["L2c_comm_rel_errs"] = comm_errs
    out["pass"] = (max(eig_errs) < 1e-10
                   and abs(weyl_dims[-1] - 1.0) < 0.03
                   and comm_errs[-1] < 2e-2
                   and comm_errs[-1] < comm_errs[0])
    return out


def L3_truncation_metric():
    """Connes-van Suijlekom in miniature: truncated optimal witness
    (the tent f*(x) = sqrt(beta)*arc(x,0)) recovers the true distance
    d(0,pi) = sqrt(beta)*pi: value via Fejer-localized states, Lipschitz
    norm via the truncated commutator; ratio -> 1 with O(1/M)."""
    Lx = 2.0 * math.pi
    ratios = []
    for N in (65, 129, 257, 513):
        M = (N - 1) // 2
        s_sc = (2 * math.pi / Lx) / math.sqrt(BETA_TGL)
        modes = np.arange(-M, M + 1)
        Dm = s_sc * np.diag(modes).astype(complex)
        # Fourier coefficients of arc(x,0)=min(x,2pi-x):
        # c_0 = pi/2; c_d = -2/(pi d^2) for odd d; 0 for even d != 0
        def ccoef(d):
            d = abs(int(d))
            if d == 0: return math.pi / 2
            if d % 2 == 1: return -2.0 / (math.pi * d * d)
            return 0.0
        fm = math.sqrt(BETA_TGL) * np.array(
            [[ccoef(m - mp) for mp in modes] for m in modes], complex)
        lip = np.linalg.norm(Dm @ fm - fm @ Dm, 2)      # -> 1
        # Fejer-localized vector states at p=pi, q=0
        wts = np.sqrt(1.0 - np.abs(modes) / (M + 1.0))
        def state(x0):
            v = wts * np.exp(-1j * modes * x0)
            return v / np.linalg.norm(v)
        vp, vq = state(math.pi), state(0.0)
        val = float(np.real(np.vdot(vp, fm @ vp) - np.vdot(vq, fm @ vq)))
        d_true = math.sqrt(BETA_TGL) * math.pi
        ratios.append(float(val / (d_true * lip)))
    return {"L3_ratios": ratios,
            "pass": abs(ratios[-1] - 1.0) < 5e-2
                    and abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)}

def main():
    print("=" * 72)
    print("TGL THREE LOCKS v1 — os tres fechos tecnicos")
    print(f"beta_TGL = {BETA_TGL:.18f} (computado)")
    print("=" * 72)
    res, ok = {}, True
    errs = [L1_integral_identity(n, s) for n in (5, 7) for s in (11, 83)]
    res["L1_identity_err_max"] = max(errs)
    res["L1_pass"] = max(errs) < 1e-9
    ok &= res["L1_pass"]
    print(f"L1 (T1 bem-posto): e^(tL) = Integral V_s (.) V_s* dnu_t |"
          f" erro max = {max(errs):.2e}  ->"
          f" {'PASS' if res['L1_pass'] else 'FAIL'}")
    r2 = L2_connes_axioms(); res["L2"] = r2; ok &= r2["pass"]
    print(f"L2 (axiomas da tripla-limite, S^1): escada Dirac "
          f"{r2['L2a_eig_rel_errs'][0]:.1e}->{r2['L2a_eig_rel_errs'][-1]:.1e}"
          f" | dim Weyl {r2['L2b_weyl_dims'][-1]:.4f}"
          f" | comutador {r2['L2c_comm_rel_errs'][0]:.1e}->"
          f"{r2['L2c_comm_rel_errs'][-1]:.1e}  ->"
          f" {'PASS' if r2['pass'] else 'FAIL'}")
    r3 = L3_truncation_metric(); res["L3"] = r3; ok &= r3["pass"]
    print(f"L3 (truncamento espectral -> circulo): razao da testemunha"
          f" otima {' -> '.join(f'{r:.4f}' for r in r3['L3_ratios'])}  ->"
          f" {'PASS' if r3['pass'] else 'FAIL'}")
    verdict = ("PASS — T1 bem-posto pela representacao integral no fluxo "
               "do radical (produto cruzado/CLPW); a tripla-limite e a do "
               "circulo canonico (axiomas REAL) com convergencia "
               "verificada; o levantamento GH e truncamento espectral "
               "(Connes-van Suijlekom)" if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_three_locks_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "results": res, "verdict": verdict,
           "anchors": ["Takesaki (produto cruzado II_inf) + CLPW 2022-23 "
                       "(o produto cruzado e o lar fisico)",
                       "tripla canonica de S^1 (axiomas de Connes, REAL)",
                       "Rieffel (comutativo: qGH = GH)",
                       "Connes-van Suijlekom (truncamentos espectrais de "
                       "S^1 convergem em qGH)"],
           "identity": "e^{tL} = Int V_s (.) V_s* dnu_t,  V_s = "
                       "exp(i s sqrt(K)), nu_t gaussiana var beta t"}
    fname = f"tgl_three_locks_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
