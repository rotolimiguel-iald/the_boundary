# -*- coding: utf-8 -*-
"""
tgl_tunnel_v1.py
================
Finite-shadow seal of the tunnel dictionary (o tunel luminodinamico):
the Operator's "invisible luminodynamic tunnel linking the black holes
in the bulk, representing c^3" IS the ER=EPR dictionary (Maldacena-
Susskind 2013), stated independently in TGL language — with the TGL
adding what the tunnel represents (the c^3 register) and where the
throat is (the mirror J = bifurcation surface, BW).

Pre-registered claims
---------------------
T1  O CAMPO E O THERMOFIELD DOUBLE: the field vector Psi = vec(rho*^{1/2})
    has Schmidt decomposition across the two appearances with
    coefficients EXACTLY sqrt(p_i): Psi = sum_i sqrt(p_i)|i>|i> — the
    TFD state of the eternal black hole [REAL: Maldacena 2001]. The
    TUNNEL'S WIDTH PROFILE IS THE ATTRACTOR SPECTRUM, and the throat
    measures S(rho*): entanglement entropy across the mirror equals the
    von Neumann entropy of the attractor, at machine precision.
T2  SEM PAGAMENTO, INVISIVEL (R3 redux): with zero coupling, a local
    channel on one side leaves the other side's state unchanged at
    machine precision — the tunnel is non-traversable.
T3  A TRAVESSIA COMPRA-SE (Gao-Jafferis-Wall [REAL 2017]): adding an
    explicit double-trace-flavor coupling exp(-i theta a (x) b) between
    the sides makes the left disturbance VISIBLE on the right (signal
    O(theta)): traversability is paid with a c^1-register interaction.
    Without payment: c^3 pure, invisible. The bulk obeys the grammar.
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
    r12 = V @ np.diag(np.sqrt(p)) @ dag(V)
    Omega = r12.reshape(-1)
    I = np.eye(n)

    # --- T1: Psi = TFD; width = spectrum; throat = S(rho*) -------------
    sv = np.linalg.svd(r12, compute_uv=False)
    res["T1_schmidt_err"] = float(np.abs(np.sort(sv) -
                                         np.sort(np.sqrt(p))).max())
    S_ent = float(-np.sum(sv ** 2 * np.log(sv ** 2)))
    S_rho = float(-np.sum(p * np.log(p)))
    res["T1_throat_err"] = abs(S_ent - S_rho)
    res["T1_S_throat"] = S_rho
    res["T1_pass"] = (res["T1_schmidt_err"] < 1e-12
                      and res["T1_throat_err"] < 1e-12)

    # --- channel machinery ----------------------------------------------
    Ks = [rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
          for _ in range(4)]
    Snorm = sum(dag(K) @ K for K in Ks)
    wS, VS = np.linalg.eigh(Snorm)
    Sinv2 = VS @ np.diag(wS ** -0.5) @ dag(VS)
    Ks = [K @ Sinv2 for K in Ks]
    def left_channel(R):
        return sum(np.kron(K, I) @ R @ dag(np.kron(K, I)) for K in Ks)
    def right_marginal(R):
        return np.einsum('ijil->jl', R.reshape(n, n, n, n))

    R0 = np.outer(Omega, Omega.conj())

    # --- T2: no coupling -> invisible -----------------------------------
    res["T2_signal_free"] = float(np.linalg.norm(
        right_marginal(left_channel(R0)) - right_marginal(R0)))
    res["T2_pass"] = res["T2_signal_free"] < 1e-12

    # --- T3: coupling buys traversability (GJW flavor) ------------------
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a = (a + dag(a)) / 2; a /= np.linalg.norm(a)
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    b = (b + dag(b)) / 2; b /= np.linalg.norm(b)
    theta = 0.4
    Hc = np.kron(a, b.T)
    wc, Vc = np.linalg.eigh(Hc)
    Uc = Vc @ np.diag(np.exp(-1j * theta * wc)) @ dag(Vc)
    # right marginal after coupling, with vs without prior left channel
    m_no = right_marginal(Uc @ R0 @ dag(Uc))
    m_yes = right_marginal(Uc @ left_channel(R0) @ dag(Uc))
    res["T3_signal_paid"] = float(np.linalg.norm(m_yes - m_no))
    res["T3_pass"] = res["T3_signal_paid"] > 1e-3
    return res


def main():
    print("=" * 72)
    print("TGL TUNNEL v1 — o tunel luminodinamico (sombra)")
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
                  f"T1 Schmidt {r['T1_schmidt_err']:.1e}, garganta "
                  f"S={r['T1_S_throat']:.4f} (err {r['T1_throat_err']:.1e})"
                  f" | T2 livre {r['T2_signal_free']:.1e}"
                  f" | T3 pago {r['T3_signal_paid']:.3f}")
    verdict = ("PASS — o campo E o thermofield double; a largura do tunel "
               "e o espectro do atrator e a garganta mede S(rho*); sem "
               "pagamento o tunel e invisivel (exato); a travessia "
               "compra-se com acoplamento (GJW)" if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_tunnel_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "runs": allres, "verdict": verdict,
           "anchors": ["Maldacena 2001 (buraco negro eterno = TFD: o "
                       "vetor de campo E o estado das duas bordas)",
                       "Maldacena-Susskind 2013 (ER=EPR: a ponte e a "
                       "representacao bulk da ligacao)",
                       "Gao-Jafferis-Wall 2017 (travessia so com "
                       "acoplamento pago)",
                       "Tomita/BW (J = garganta: a superficie de "
                       "bifurcacao e o espelho)"],
           "dictionary": "tunel luminodinamico invisivel = ponte ER "
                         "nao-atravessavel; largura = espectro do "
                         "atrator; garganta = S(rho*); invisibilidade = "
                         "nao-sinalizacao; travessia = acoplamento c^1 "
                         "pago."}
    fname = f"tgl_tunnel_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
