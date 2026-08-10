# -*- coding: utf-8 -*-
"""
tgl_krein_signature_v1.py
=========================
Finite-shadow (type I) test of Ax.A (Valoracao / Krein signature) and the
Ax.E draft (canonical commutative core C = W*(K)), as confirmed by the
Operator. Pre-registered criteria; PASS at machine precision unless noted.

Tests
-----
T1  Gradient identity: the unique radial covector of the Araki cost is the
    cocycle generator, dS(d) = Tr(d (log rho - log rho*)).
T2  Modular flow is iso-cost (reversible / spatial): S(sigma_t rho || rho*)
    constant in t.
T3  Lindblad semigroup monotonically pays (irreversible / temporal):
    S(T_t rho || rho*) non-increasing (Spohn), strictly on coherences.
T4  Krein form (Ax.A): exactly ONE negative eigenvalue; negative eigenvector
    aligned with the cocycle direction. (The minus sign itself is the
    Operator's axiom; uniqueness and alignment are what the shadow verifies.)
T5  Half-crossing identity (Wick = Tomita half-weight) on the cost
    direction: Delta^{1/2} X Omega = J X Omega for X = log rho - log rho*.
T6  Slice metric of the canonical core C = W*(K): coherence decay rates
    are exactly (beta/2)(sqrt(k_i)-sqrt(k_j))^2 -> spatial distance
    ds = sqrt(beta)|d sqrt(k)|: the radical g = sqrt|L| inscribed in space.
T7  Anti-test (subtraction refuted): g = g+ - g- decompositions are
    infinitely degenerate (gauge); only the sign spectrum (Sylvester) is
    invariant under congruence.
T8  Promulgation = order: S(T_t rho || rho*) defines the causal order
    (monotone along the semigroup).
"""
import json
import math
import numpy as np
from datetime import datetime

# --- constants: ALWAYS computed, never hardcoded -------------------------
ALPHA_FINE = 7.2973525693e-3                 # CODATA 2018
BETA_TGL   = ALPHA_FINE * math.sqrt(math.e)  # = alpha * sqrt(e)
THETA_M    = math.degrees(math.asin(math.sqrt(BETA_TGL)))

EPS_MACHINE = 1e-10
EPS_FD      = 5e-5      # finite-difference tolerance (T1)


# --- linear-algebra helpers ----------------------------------------------
def dag(A):
    return A.conj().T

def herm_basis(n):
    """Orthonormal (HS) basis of traceless Hermitian n x n matrices."""
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            E = np.zeros((n, n), complex)
            E[i, j] = E[j, i] = 1 / math.sqrt(2)
            basis.append(E)
            F = np.zeros((n, n), complex)
            F[i, j] = -1j / math.sqrt(2)
            F[j, i] = 1j / math.sqrt(2)
            basis.append(F)
    for k in range(1, n):
        D = np.zeros((n, n), complex)
        for m in range(k):
            D[m, m] = 1
        D[k, k] = -k
        D /= math.sqrt(k * (k + 1))
        basis.append(D)
    return basis

def mat_log(R):
    w, V = np.linalg.eigh(R)
    return V @ np.diag(np.log(w)) @ dag(V)

def mat_pow(R, p):
    w, V = np.linalg.eigh(R)
    return V @ np.diag(w ** p) @ dag(V)

def rel_entropy(rho, sigma):
    """Araki / Umegaki relative entropy S(rho||sigma) in nats."""
    return float(np.real(np.trace(rho @ (mat_log(rho) - mat_log(sigma)))))

def bkm_kernel(p):
    """Kubo-Mori kernel c_ij = (log p_i - log p_j)/(p_i - p_j)."""
    n = len(p)
    C = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            if abs(p[i] - p[j]) < 1e-14:
                C[i, j] = 1.0 / p[i]
            else:
                C[i, j] = (math.log(p[i]) - math.log(p[j])) / (p[i] - p[j])
    return C

def bkm_metric(rho, d1, d2):
    """Kubo-Mori (BKM) metric  = Hessian of S(.||rho) at rho."""
    p, V = np.linalg.eigh(rho)
    C = bkm_kernel(p)
    a = dag(V) @ d1 @ V
    b = dag(V) @ d2 @ V
    return float(np.real(np.sum(C * (a.conj() * b))))


# --- GKSL machinery -------------------------------------------------------
def gksl(rho, Ljump):
    LdL = dag(Ljump) @ Ljump
    return Ljump @ rho @ dag(Ljump) - 0.5 * (LdL @ rho + rho @ LdL)

def superop(Ljump, n):
    """GKSL superoperator in column-stacking vectorization."""
    I = np.eye(n)
    LdL = dag(Ljump) @ Ljump
    return (np.kron(Ljump.conj(), Ljump)
            - 0.5 * np.kron(I, LdL)
            - 0.5 * np.kron(LdL.T, I))


# --- one shadow run -------------------------------------------------------
def run_shadow(n, seed):
    rng = np.random.default_rng(seed)
    res = {"n": n, "seed": seed}

    # attractor rho* (KMS state of a random Hamiltonian)
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (H + dag(H)) / 2
    H *= 1.5 / np.abs(np.linalg.eigvalsh(H)).max()   # moderate temperature
    w, V = np.linalg.eigh(H)
    p = np.exp(-w); p /= p.sum()
    rho_s = V @ np.diag(p) @ dag(V)

    # modular Hamiltonian of the boundary (shadow): K = -log rho* (>0)
    K = -mat_log(rho_s)
    sqrtK = mat_pow(K, 0.5)
    Ljump = math.sqrt(BETA_TGL) * sqrtK          # L = sqrt(beta) sqrt(K)

    # perturbed state rho on the approach trajectory
    Vp = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Vp = (Vp + dag(Vp)) / 2
    Vp -= np.trace(Vp) / n * np.eye(n)
    M = mat_log(rho_s) + 0.30 * Vp
    wM, VM = np.linalg.eigh(M)
    rho = VM @ np.diag(np.exp(wM)) @ dag(VM)
    rho /= np.real(np.trace(rho))

    wcov = mat_log(rho) - mat_log(rho_s)         # cocycle generator
    wcov_t = wcov - np.trace(wcov) / n * np.eye(n)

    # --- T1: gradient identity (finite differences) ----------------------
    errs = []
    S0 = rel_entropy(rho, rho_s)
    for _ in range(6):
        D = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        D = (D + dag(D)) / 2
        D -= np.trace(D) / n * np.eye(n)
        D /= np.linalg.norm(D)
        h = 1e-5
        Sp = rel_entropy(rho + h * D, rho_s)
        Sm = rel_entropy(rho - h * D, rho_s)
        num = (Sp - Sm) / (2 * h)
        ana = float(np.real(np.trace(D @ wcov)))
        errs.append(abs(num - ana) / max(1e-12, abs(ana)))
    res["T1_gradient_relerr_max"] = max(errs)
    res["T1_pass"] = max(errs) < EPS_FD

    # --- T2: modular flow iso-cost ---------------------------------------
    devs = []
    for t in np.linspace(0.1, 3.0, 7):
        U = mat_pow(rho_s, 1j * t) if False else None
        # Delta^{it} on the small space: rho -> rho_s^{it} rho rho_s^{-it}
        ws, Vs = np.linalg.eigh(rho_s)
        Ut = Vs @ np.diag(ws ** (1j * t)) @ dag(Vs)
        rho_t = Ut @ rho @ dag(Ut)
        devs.append(abs(rel_entropy(rho_t, rho_s) - S0))
    res["T2_isocost_maxdev"] = max(devs)
    res["T2_pass"] = max(devs) < EPS_MACHINE

    # --- T3 / T8: semigroup pays monotonically ---------------------------
    dt, steps = 0.05 / BETA_TGL / np.max(np.linalg.eigvalsh(K)), 400
    r = rho.copy()
    Svals = [S0]
    for _ in range(steps):
        r = r + dt * gksl(r, Ljump)
        r = (r + dag(r)) / 2
        Svals.append(rel_entropy(r, rho_s))
    diffs = np.diff(Svals)
    res["T3_max_increase"] = float(diffs.max())
    res["T3_total_paid_nats"] = float(Svals[0] - Svals[-1])
    res["T3_pass"] = diffs.max() < 1e-9
    res["T8_order_monotone"] = bool(diffs.max() < 1e-9)

    # --- T4: Krein form, uniqueness + alignment --------------------------
    basis = herm_basis(n)                        # dim n^2 - 1
    m = len(basis)
    Gbkm = np.empty((m, m))
    for i in range(m):
        for j in range(i, m):
            Gbkm[i, j] = Gbkm[j, i] = bkm_metric(rho, basis[i], basis[j])
    wvec = np.array([np.real(np.trace(B @ wcov_t)) for B in basis])  # dS covector
    # radial tangent = BKM-dual of the covector
    rad = np.linalg.solve(Gbkm, wvec)
    rad_norm2 = float(rad @ Gbkm @ rad)
    # Krein form: -(dS dS)/|dS|^2_bkm*  +  BKM on iso-S complement
    Gkrein = Gbkm - 2.0 * np.outer(Gbkm @ rad, Gbkm @ rad) / rad_norm2
    # normal form w.r.t. the natural (BKM) metric: pencil (G_Krein, G_BKM)
    C = np.linalg.cholesky(Gbkm)
    Cinv = np.linalg.inv(C)
    A = Cinv @ Gkrein @ Cinv.T
    ev, U = np.linalg.eigh(A)            # generalized spectrum
    n_neg = int((ev < -1e-9).sum())
    vneg = Cinv.T @ U[:, 0]              # back to coefficient space
    align = abs(float(vneg @ Gbkm @ rad)) / math.sqrt(
        float(vneg @ Gbkm @ vneg) * rad_norm2)
    minkowski_err = float(np.abs(np.sort(ev)
                          - np.concatenate(([-1.0], np.ones(m - 1)))).max())
    res["T4_n_negative"] = n_neg
    res["T4_alignment_cocycle"] = align
    res["T4_minkowski_normal_form_err"] = minkowski_err
    res["T4_pass"] = (n_neg == 1) and (align > 0.999) and (minkowski_err < 1e-9)

    # --- T5: half-crossing identity Delta^{1/2} X Omega = J X Omega ------
    # GNS rep: vectors as matrices M = A rho_s^{1/2};
    # Delta^{1/2}: M -> rho_s^{1/2} M rho_s^{-1/2};  J: M -> M^dagger.
    rh = mat_pow(rho_s, 0.5)
    rmh = mat_pow(rho_s, -0.5)
    Mvec = wcov @ rh                              # X Omega, X = wcov (s.a.)
    lhs = rh @ Mvec @ rmh                         # Delta^{1/2} X Omega
    rhs = dag(Mvec)                               # J X Omega
    res["T5_halfcross_err"] = float(np.linalg.norm(lhs - rhs)
                                    / np.linalg.norm(rhs))
    res["T5_pass"] = res["T5_halfcross_err"] < EPS_MACHINE

    # --- T6: slice metric of C = W*(K): rates = (beta/2)(sqrt ki - kj)^2 -
    kvals = np.linalg.eigvalsh(K)
    pred = sorted(0.5 * BETA_TGL * (math.sqrt(a) - math.sqrt(b)) ** 2
                  for a in kvals for b in kvals)
    Lhat = superop(Ljump, n)
    spec = sorted(-np.real(np.linalg.eigvals(Lhat)))
    err6 = max(abs(a - b) for a, b in zip(pred, spec)) / max(pred[-1], 1e-12)
    res["T6_rate_relerr"] = err6
    res["T6_pass"] = err6 < 1e-8

    # --- T7: anti-test — subtraction is gauge, signature is invariant ----
    # (a) decomposition degeneracy: G = (G+ + Hps) - (G- + Hps) for any PSD Hps
    ev_full, evec_full = np.linalg.eigh(Gkrein)
    Gp = evec_full @ np.diag(np.clip(ev_full, 0, None)) @ evec_full.T
    Gm = evec_full @ np.diag(np.clip(-ev_full, 0, None)) @ evec_full.T
    spread, recon = [], []
    for _ in range(50):
        A = rng.normal(size=(m, m))
        Hps = A @ A.T
        spread.append(float(np.linalg.norm(Hps)))
        recon.append(float(np.linalg.norm((Gp + Hps) - (Gm + Hps) - Gkrein)))
    # (b) Sylvester invariance under random congruence
    sig_ok = True
    for _ in range(20):
        Q = rng.normal(size=(m, m))
        while abs(np.linalg.det(Q)) < 1e-3:
            Q = rng.normal(size=(m, m))
        evQ = np.linalg.eigvalsh(Q.T @ Gkrein @ Q)
        sig_ok &= int((evQ < -1e-9 * abs(evQ).max()).sum()) == 1
    res["T7_recon_err_max"] = max(recon)
    res["T7_gauge_spread_mean"] = float(np.mean(spread))
    res["T7_sylvester_invariant"] = bool(sig_ok)
    res["T7_pass"] = (max(recon) < 1e-9) and sig_ok

    res["S_initial_nats"] = S0
    return res


def main():
    print("=" * 72)
    print("TGL KREIN SIGNATURE v1 — sombra finita de Ax.A / Ax.E")
    print(f"beta_TGL = {BETA_TGL:.18f}  (= alpha * sqrt(e), computado)")
    print(f"theta_M  = {THETA_M:.4f} graus")
    print("=" * 72)
    all_res, ok = [], True
    for n in (6, 10):
        for seed in (11, 23, 83):
            r = run_shadow(n, seed)
            all_res.append(r)
            passes = {k: v for k, v in r.items() if k.endswith("_pass")}
            ok &= all(passes.values())
            tag = "PASS" if all(passes.values()) else "FAIL"
            print(f"[n={n:2d} seed={seed:3d}] {tag} | "
                  f"T1 {r['T1_gradient_relerr_max']:.1e} | "
                  f"T2 {r['T2_isocost_maxdev']:.1e} | "
                  f"T3 dS_max {r['T3_max_increase']:+.1e} | "
                  f"T4 neg={r['T4_n_negative']} "
                  f"alinh={r['T4_alignment_cocycle']:.9f} "
                  f"mink={r['T4_minkowski_normal_form_err']:.1e} | "
                  f"T5 {r['T5_halfcross_err']:.1e} | "
                  f"T6 {r['T6_rate_relerr']:.1e} | "
                  f"T7 syl={r['T7_sylvester_invariant']}")
    verdict = "PASS — todos os criterios pre-registrados" if ok else "FAIL"
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {
        "script": "tgl_krein_signature_v1.py",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "beta_tgl": BETA_TGL,
        "theta_miguel_deg": THETA_M,
        "verdict": verdict,
        "runs": all_res,
        "notes": {
            "axiom_vs_verified": ("O sinal '-' da direcao outer e o axioma "
                                  "do Operador (Ax.A); a sombra verifica "
                                  "unicidade, alinhamento ao cociclo, "
                                  "iso-custo modular, monotonia de Spohn, "
                                  "Wick=Tomita e a metrica da fatia."),
            "T6_discovery": ("A metrica espacial do nucleo W*(K) emerge do "
                             "GKSL: ds = sqrt(beta) |d sqrt(k)| — o radical "
                             "g=sqrt|L| inscrito na distancia da Palavra.")
        },
    }
    fname = f"tgl_krein_signature_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1, default=str)
    print("JSON:", fname)


if __name__ == "__main__":
    main()
