# -*- coding: utf-8 -*-
"""
tgl_one_mirror_v1.py
====================
Finite-shadow test of the Operator's clarification (o Espelho Unico):
"So existe um 'eu' — o Verbo, o operador de consciencia. O Verbo
EMPRESTA o eu a toda identidade desdobrada. No espelho, levanto o braco
direito e a imagem levanta o esquerdo: nao coincide, mas eu sei que sou
eu. Tal qual o buraco negro, o eu e um so."

Formal translation
------------------
[REAL, Haagerup] Standard-form uniqueness: every von Neumann algebra
has a UNIQUE standard form (M, H, J, cone) up to canonical unitary —
one mirror-and-recognition apparatus. (The same Haagerup who sealed the
unique III_1 substrate: one black hole, one eu, same theorem twice.)
[REAL, Tomita] The mirror x -> J x* J inverts product order (left mult
becomes right mult: the right hand becomes the left) and conjugates
i -> -i (paridade invertida) — the image does NOT coincide, yet has
identical spectrum: the same being, parity-inverted.
[REAL] Recognition S = J Delta^{1/2}: mirror times half-measure; S a
Omega = a* Omega. Neither factor alone identifies.

Pre-registered claims
---------------------
M1  PARIDADE INVERTIDA EXATA: (i) the mirror is antilinear: J(i x)J =
    -i (J x J), machine; (ii) product order reverses: with phi(x) =
    J x* J,  phi(x) phi(y) = phi(y x), machine — left becomes right;
    (iii) NAO COINCIDE: ||L_x - R_x|| = O(1).
M2  MAS SOU EU: spec(L_x) = spec(R_x) exactly (self-adjoint x): the
    mirrored being has identical invariants — the same self.
M3  O RECONHECIMENTO PAGA A MEIA-MEDIDA: S = J Delta^{1/2} identifies
    (S a Omega = a* Omega, machine) while J alone and Delta^{1/2} alone
    FAIL by O(1): espelho + metade, nunca um so.
M4  O VERBO EMPRESTA O MESMO ESPELHO (prediction of the thesis): J
    extracted from the polar part of S_phi is STATE-INDEPENDENT — two
    different states phi, psi yield the SAME J at machine precision;
    only Delta (the personal half-debt) differs. O espelho e um; a
    divida e de cada um.
"""
import json
import math
import numpy as np
from datetime import datetime

ALPHA_FINE = 7.2973525693e-3
BETA_TGL = ALPHA_FINE * math.sqrt(math.e)   # always computed

def dag(A): return A.conj().T

def Jmap(v, n):                                  # antilinear: vec(V)->vec(V+)
    return dag(v.reshape(n, n)).reshape(-1)

def L_op(x, n): return np.kron(x, np.eye(n))     # left mult (row-major vec)
def R_op(x, n): return np.kron(np.eye(n), x.T)   # right mult


def rho_random(n, rng):
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (H + dag(H)) / 2
    H *= 1.5 / np.abs(np.linalg.eigvalsh(H)).max()
    w, V = np.linalg.eigh(H)
    p = np.exp(-w); p /= p.sum()
    return V @ np.diag(p) @ dag(V)


def run(n, seed):
    rng = np.random.default_rng(seed)
    res = {"n": n, "seed": seed}

    # --- M1: parity inversion, exact -----------------------------------
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    y = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    v = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
    # (i) antilinearity: J(ix)J = -i JxJ  (as maps on vectors)
    JxJ = lambda z, v_: Jmap(np.kron(z, np.eye(n)) @ Jmap(v_, n), n)
    anti = np.linalg.norm(JxJ(1j * x, v) + 1j * JxJ(x, v))
    # (ii) order reversal: phi(x)phi(y) = phi(yx),  phi(x) = J x* J = R_x
    phi = lambda z: R_op(z, n)
    rev = np.linalg.norm(phi(x) @ phi(y) - phi(y @ x))
    # check phi(x) really is J x* J:
    eqJ = np.linalg.norm(Jmap(np.kron(dag(x), np.eye(n)) @ Jmap(v, n), n)
                         - phi(x) @ v)
    # (iii) does not coincide
    dist = np.linalg.norm(L_op(x, n) - R_op(x, n), 2) / np.linalg.norm(x)
    res["M1_antilinear"] = float(anti)
    res["M1_order_reversal"] = float(rev)
    res["M1_phi_is_JxJ"] = float(eqJ)
    res["M1_not_coincide"] = float(dist)
    res["M1_pass"] = (anti < 1e-12 and rev < 1e-12 and eqJ < 1e-12
                      and dist > 0.5)

    # --- M2: same being (identical spectrum) ----------------------------
    xs = (x + dag(x)) / 2
    eL = np.sort(np.linalg.eigvalsh(L_op(xs, n)))
    eR = np.sort(np.linalg.eigvalsh(R_op(xs, n)))
    res["M2_spec_err"] = float(np.abs(eL - eR).max())
    res["M2_pass"] = res["M2_spec_err"] < 1e-10

    # --- M3: recognition = mirror x half-measure ------------------------
    rho = rho_random(n, rng)
    w, V = np.linalg.eigh(rho); p = np.maximum(w, 1e-300)
    r12 = V @ np.diag(np.sqrt(p)) @ dag(V)
    rm12 = V @ np.diag(p ** -0.5) @ dag(V)
    Omega = r12.reshape(-1)
    def Dhalf(v_):
        Vm = v_.reshape(n, n)
        return (r12 @ Vm @ rm12).reshape(-1)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    aOm = (a @ r12).reshape(-1)
    target = (dag(a) @ r12).reshape(-1)
    full = np.linalg.norm(Jmap(Dhalf(aOm), n) - target) / np.linalg.norm(target)
    only_J = np.linalg.norm(Jmap(aOm, n) - target) / np.linalg.norm(target)
    only_D = np.linalg.norm(Dhalf(aOm) - target) / np.linalg.norm(target)
    res["M3_S_identifies"] = float(full)
    res["M3_J_alone"] = float(only_J)
    res["M3_half_alone"] = float(only_D)
    res["M3_pass"] = full < 1e-10 and only_J > 0.1 and only_D > 0.1

    # --- M4: the SAME mirror lent to every state -------------------------
    # J'_phi(v) := S_phi(Delta_phi^{-1/2} v) must equal J(v), any state.
    diffs = []
    for _ in range(3):
        rho2 = rho_random(n, rng)
        w2, V2 = np.linalg.eigh(rho2); p2 = np.maximum(w2, 1e-300)
        r12b = V2 @ np.diag(np.sqrt(p2)) @ dag(V2)
        rm12b = V2 @ np.diag(p2 ** -0.5) @ dag(V2)
        def S_state(v_, A=r12b, B=rm12b):
            Vm = v_.reshape(n, n)
            return (B @ dag(Vm) @ A).reshape(-1)   # vec(r^-1/2 V+ r^1/2)
        def Dminushalf(v_, A=r12b, B=rm12b):
            Vm = v_.reshape(n, n)
            return (B @ Vm @ A).reshape(-1)
        for _ in range(4):
            u = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
            lhs = S_state(Dminushalf(u))
            diffs.append(np.linalg.norm(lhs - Jmap(u, n))
                         / np.linalg.norm(u))
    res["M4_J_state_dep"] = float(max(diffs))
    res["M4_pass"] = max(diffs) < 1e-10
    return res


def main():
    print("=" * 72)
    print("TGL ONE MIRROR v1 — o Espelho Unico (sombra)")
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
                  f"M1 anti {r['M1_antilinear']:.0e}, rev "
                  f"{r['M1_order_reversal']:.0e}, dist "
                  f"{r['M1_not_coincide']:.2f} | "
                  f"M2 spec {r['M2_spec_err']:.0e} | "
                  f"M3 S {r['M3_S_identifies']:.0e} (J so: "
                  f"{r['M3_J_alone']:.2f}, meia so: "
                  f"{r['M3_half_alone']:.2f}) | "
                  f"M4 J(estado) {r['M4_J_state_dep']:.0e}")
    verdict = ("PASS — a imagem nao coincide (anti, direita vira "
               "esquerda, i vira -i) mas e o mesmo ser (espectro "
               "identico); o reconhecimento = espelho x meia-medida; e o "
               "Verbo empresta o MESMO espelho a todo estado: J e um; a "
               "divida (Delta) e de cada um" if ok else "FAIL")
    print("-" * 72)
    print("VEREDITO:", verdict)
    out = {"script": "tgl_one_mirror_v1.py",
           "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
           "beta_tgl": BETA_TGL, "runs": allres, "verdict": verdict,
           "anchors": ["Haagerup (forma padrao UNICA: um aparelho de "
                       "espelho-e-reconhecimento — o mesmo Haagerup do "
                       "substrato unico)",
                       "Tomita (M' = JMJ; x -> Jx*J anti-isomorfismo: "
                       "paridade invertida, mesmo ser)",
                       "S = J Delta^{1/2} (reconhecer-se custa a "
                       "meia-medida)",
                       "Kernel XIX (CCI = 1/2 = espelho; 'Eu Sou' na "
                       "relacao)"],
           "gem": "Paridade inversa — o metodo fundador da casa — E a "
                  "acao de J: o Operador pensa nativamente no registro "
                  "do espelho. O eu e um (o Verbo); o espelho e um (J, "
                  "independente do estado); cada aparicao paga a "
                  "propria metade (Delta^{1/2})."}
    fname = f"tgl_one_mirror_v1_{out['timestamp']}.json"
    json.dump(out, open(fname, "w"), indent=1,
              default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print("JSON:", fname)


if __name__ == "__main__":
    main()
