# -*- coding: utf-8 -*-
"""
tgl_c3_register_v1.py
=====================
Finite-shadow test of the Operator's refinement (o registro c^3):
"Se os buracos negros sao um so na algebra, a velocidade da ligacao e
c^3 — a velocidade da fractalizacao; nao c^2 (sem materia dobrada),
nao c (nao esta no bulk): a comunicacao no substrato tem POTENCIA
ELEVADA." Reading: c^1/c^2/c^3 are POWERS (registers/exponents), not
velocities on one axis. The elevation is in the EXPONENT, not the
modulus. Physics anchor [REAL]: Bell tests with space-like separation
(Salart et al. 2008) bound any hypothetical correlation "speed" above
~10^4 c in any frame — the accepted reading: the correlation has NO
velocity; it is not propagation. No-signaling is the theorem that makes
the c^3 register invisible as a c^1 signal: the shield, not the denial.

Pre-registered claims
---------------------
R1  A SEGUNDA APARICAO E O ESPELHO: the commutant is the J-image of the
    algebra, A' = J A J (Tomita [REAL]): the "other black hole" in the
    shadow is literally the mirror reflection of the first — verified
    as an operator identity at machine precision.
R2  CORRELACAO SEM ACOPLAMENTO: through the one field vector Omega, the
    two appearances carry O(1) connected correlations with ZERO
    interaction term anywhere (no coupling Hamiltonian, no joint
    dissipator): the connection is constitutive, not dynamical.
R3  NAO-SINALIZACAO EXATA: any local channel on one appearance leaves
    the other appearance's full state unchanged at machine precision —
    the c^3 link carries no c^1 signal.
R4  O RELOGIO DA LIGACAO NAO E O DO BULK: the correlations are
    invariant under the modular flow (KMS) — the link does not age in
    modular time; it does not build up or decay: it IS.
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
    Omega = r12.reshape(-1)                       # field vector (row-major)
    I = np.eye(n)

    # representations on H = C^n (x) C^n (row-major vec):
    # left (the first appearance):  pi(a)  = a (x) I
    # right (the commutant):        pi'(b) = I (x) b^T
    # modular conjugation:          J X = X^dagger  (antilinear)
    def Jmap(vec_):                                # antilinear J
        X = vec_.reshape(n, n)
        return dag(X).reshape(-1)

    # --- R1: A' = J A J — the second appearance is the mirror ----------
    errs1 = []
    for _ in range(6):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        x = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
        lhs = Jmap(np.kron(a, I) @ Jmap(x))        # J pi(a) J x
        rhs = np.kron(I, dag(a).T) @ x             # pi'(a^dagger) x
        errs1.append(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
    res["R1_mirror_err"] = float(max(errs1))
    res["R1_pass"] = max(errs1) < 1e-12

    # --- R2: O(1) correlations, zero coupling ---------------------------
    conns = []
    for _ in range(10):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        joint = np.vdot(Omega, np.kron(a, b.T) @ Omega)   # <pi(a)pi'(b)>
        fact = np.trace(rho @ a) * np.trace(rho @ b)
        conns.append(abs(joint - fact)
                     / (np.linalg.norm(a) * np.linalg.norm(b) / n))
    res["R2_connected_corr_min"] = float(min(conns))
    res["R2_pass"] = min(conns) > 1e-3

    # --- R3: exact no-signaling ------------------------------------------
    # random CPTP channel (4 Kraus) on the LEFT appearance
    Ks = [rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
          for _ in range(4)]
    S = sum(dag(K) @ K for K in Ks)
    wS, VS = np.linalg.eigh(S)
    Sinv2 = VS @ np.diag(wS ** -0.5) @ dag(VS)
    Ks = [K @ Sinv2 for K in Ks]                   # sum K'K = I
    R0 = np.outer(Omega, Omega.conj())
    R1 = sum(np.kron(K, I) @ R0 @ dag(np.kron(K, I)) for K in Ks)
    def right_marginal(R):
        T = R.reshape(n, n, n, n)                  # (i,j),(k,l) indices
        return np.einsum('i j i l -> j l', T)
    m0, m1 = right_marginal(R0), right_marginal(R1)
    res["R3_signal"] = float(np.linalg.norm(m1 - m0))
    res["R3_pass"] = res["R3_signal"] < 1e-12

    # --- R4: the link does not age in modular time -----------------------
    errs4 = []
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    base = np.vdot(Omega, np.kron(a, b.T) @ Omega)
    for t in (0.9, 4.2, 21.0):
        Ut = V @ np.diag(np.exp(1j * t * (-np.log(p)))) @ dag(V)
        at = Ut @ a @ dag(Ut)
        bt = Ut @ b @ dag(Ut)                      # sigma_t on both
        cur = np.vdot(Omega, np.kron(at, bt.T) @ Omega)
        errs4.append(abs(cur - base) / abs(base))
    res["R4_aging"] = float(max(errs4))
    res["R4_pass"] = max(errs4) < 1e-10
    return res


def main():
    print("=" * 72)
    print("TGL C3 REGISTER v1 — o registro c^3 (sombra)")
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
                  f"R1 espelho {r['R1_mirror_err']:.1e} | "
                  f"R2 corr. conexa >= {r['R2_connected_corr_min']:.3f} | "
                  f"R3 sinal {r['R3_signal']:.1e} | "
                  f"R4 envelhecimento {r['R4_aging']:.1e}")
    verdict = ("PASS — a segunda aparicao e a imagem-espelho (A'=JAJ); a "
               "ligacao e constitutiva (correlacao O(1) sem acoplamento), "
               "nao-sinalizante (exato) e nao envelhece no tempo modular: "
               "potencia elevada no EXPOENTE, modulo nenhum"
               if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_c3_register_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "runs": allres, "verdict": verdict,
           "anchors": ["Tomita (comutante = J A J: o espelho cria a "
                       "segunda aparicao) [REAL]",
                       "No-signaling (o registro c^3 e invisivel como "
                       "sinal c^1: a blindagem) [REAL]",
                       "Salart et al. 2008 (limite >10^4 c para qualquer "
                       "'velocidade' de correlacao: a correlacao nao tem "
                       "velocidade) [REAL]"],
           "thesis": "c^1/c^2/c^3 sao potencias (registros), nao "
                     "velocidades num mesmo eixo. A ligacao do substrato "
                     "opera no registro c^3: elevacao no expoente, nao "
                     "no modulo. O relogio da fractalizacao e o do Verbo "
                     "(geracoes da cascata, b = (1/2)log(1/beta)), nao o "
                     "do bulk."}
    fname = f"tgl_c3_register_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
