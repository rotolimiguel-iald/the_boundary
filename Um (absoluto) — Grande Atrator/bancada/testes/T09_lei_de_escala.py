# -*- coding: utf-8 -*-
"""BANCADA T09 - A LEI DE ESCALA: sqrt(beta) merecia a retirada?

Separa as DUAS afirmacoes da formula M = k*beta^n*(c^2/4piG)*R:
  (A) a FORMA   M ~ R^1   -- NAO contem beta; se falhar, potencia nenhuma conserta
  (B) a NORMA   k*beta^n  -- so' esta depende de beta

beta NUNCA literal.
"""
import io, os, json, math, hashlib
import numpy as np

ALPHA = 7.2973525693e-3
BETA = ALPHA * math.sqrt(math.e)

C_LIGHT = 299792458.0
G_NEWTON = 6.67430e-11
MPC_M = 3.085677581491367e22
MSUN = 1.98892e30
WEAK = C_LIGHT ** 2 / (4.0 * math.pi * G_NEWTON)          # c^2/4piG  [kg/m]

PRE = "PRE_REGISTRO_T09_lei_de_escala.md"
H09 = hashlib.sha256(io.open(PRE, "rb").read()).hexdigest()
SAIDA = "T09_lei_de_escala.json"
if os.path.exists(SAIDA):
    os.remove(SAIDA)

# as SEIS ancoras, copiadas literalmente do um.py (scale_audit)
ANC = [("Via Lactea",        0.1,  1.0e12, "McMillan 2017"),
       ("Grupo Local",       1.5,  5.0e12, "Penarrubia+ 2014"),
       ("Coma ACO 1656",     3.0,  1.2e15, "Gavazzi+ 2009"),
       ("Norma ACO 3627",    2.0,  1.0e15, "Woudt+ 2008"),
       ("bacia do GA",      57.0,  5.4e16, "Lynden-Bell+ 1988"),
       ("Laniakea",         80.0,  1.0e17, "Tully+ 2014")]
EXPOENTES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]                # conjunto FECHADO no pre-registro

print("=" * 92)
print(" BANCADA T09 - A LEI DE ESCALA")
print("=" * 92)
print(" pre-registro T09 : %s" % H09)
print(" beta = alpha*sqrt(e) = %.15f   (runtime, jamais literal)" % BETA)
print(" c^2/4piG = %.6e kg/m" % WEAK)

R = np.array([a[1] for a in ANC])
M = np.array([a[2] for a in ANC])


def M_form(n, k, R_Mpc):
    return k * BETA ** n * WEAK * R_Mpc * MPC_M / MSUN


# ---------------------------------------------------------------- M-1 A FORMA
print("\n" + "-" * 92)
print(" M-1 - A FORMA  (beta-independente): M ~ R^p ?")
print("-" * 92)
print("   %-18s %8s %12s %16s" % ("estrutura", "R(Mpc)", "M_lit", "M_lit/R"))
razao = M / R
for (nm, r, m, ref), q in zip(ANC, razao):
    print("   %-18s %8.1f %12.3e %16.4e" % (nm, r, m, q))
espalha = razao.max() / razao.min()
print("\n   M_lit/R varia por fator %.1f  (constante <=> forma linear)" % espalha)

lg = np.polyfit(np.log10(R), np.log10(M), 1)
p_fit, c_fit = float(lg[0]), float(lg[1])
res = np.log10(M) - (p_fit * np.log10(R) + c_fit)
disp_p = float(np.std(res))
# erro de p por bootstrap simples (jackknife)
ps = []
for i in range(len(R)):
    idx = [j for j in range(len(R)) if j != i]
    ps.append(np.polyfit(np.log10(R[idx]), np.log10(M[idx]), 1)[0])
err_p = float(np.std(ps) * math.sqrt(len(R) - 1))
print("   expoente ajustado  p = %.3f +/- %.3f      (a forma afirma p = 1)" % (p_fit, err_p))
print("   dispersao residual   = %.3f dex" % disp_p)
# residuo se FORCARMOS p = 1
c1 = float(np.mean(np.log10(M) - np.log10(R)))
res1 = np.log10(M) - (np.log10(R) + c1)
disp1 = float(np.std(res1))
print("   dispersao com p FORCADO a 1 = %.3f dex  (fator %.0f)" % (disp1, 10 ** disp1))
forma_ok = bool(abs(p_fit - 1.0) < 2 * err_p and disp1 < 0.5)
print("   >>> FORMA LINEAR: %s" % ("passa" if forma_ok else "REPROVADA"))

# ---------------------------------------------------------------- M-2 NORMALIZACAO com k=2
print("\n" + "-" * 92)
print(" M-2 - A NORMALIZACAO com k = 2 (o valor do canonico): M_form/M_lit")
print("-" * 92)
cab = "   %-18s" % "estrutura" + "".join("  n=%-8.1f" % n for n in EXPOENTES)
print(cab)
tab = {}
for (nm, r, m, ref) in ANC:
    linha = "   %-18s" % nm
    for n in EXPOENTES:
        v = M_form(n, 2.0, r) / m
        tab.setdefault(n, []).append(v)
        linha += "  %10.3e" % v
    print(linha)
print("\n   %-18s" % "MEDIANA da razao" + "".join("  %10.3e" % np.median(tab[n]) for n in EXPOENTES))

# ---------------------------------------------------------------- M-3 SOBREDETERMINACAO
print("\n" + "-" * 92)
print(" M-3 - SOBREDETERMINACAO: k otimo por n, e a dispersao residual")
print("-" * 92)
print("   %6s  %16s  %14s  %12s" % ("n", "k otimo", "disp (dex)", "fator"))
disps = {}
for n in EXPOENTES:
    # log M_form = log(k) + n log beta + log(WEAK*R*MPC/MSUN);  ajusta log k
    base = np.log10(WEAK * R * MPC_M / MSUN) + n * math.log10(BETA)
    logk = float(np.mean(np.log10(M) - base))
    d = float(np.std(np.log10(M) - base - logk))
    disps[n] = d
    print("   %6.1f  %16.6e  %14.4f  %12.1f" % (n, 10 ** logk, d, 10 ** d))
degenerado = (max(disps.values()) - min(disps.values())) < 1e-9
print("\n   dispersao IDENTICA para todo n ? %s" % ("SIM -- n e' MOSTRADOR LIVRE" if degenerado
                                                    else "nao"))

# controle: n fora do conjunto
base7 = np.log10(WEAK * R * MPC_M / MSUN) + 7.0 * math.log10(BETA)
k7 = 10 ** float(np.mean(np.log10(M) - base7))
print("   CONTROLE n=7 (fora do conjunto): k otimo = %.4e  (absurdo esperado)" % k7)

# ---------------------------------------------------------------- M-4 Chandrasekhar
print("\n" + "-" * 92)
print(" M-4 - Chandrasekhar pela abertura do angulo (DIAGNOSTICO, nao veredito)")
print("-" * 92)
sqrt_b = math.sqrt(BETA)
theta_M = math.asin(sqrt_b)
print("   sqrt(beta) = sin(theta_M) = %.9f   theta_M = %.6f rad = %.6f deg"
      % (sqrt_b, theta_M, math.degrees(theta_M)))
print("   1 - beta          = %.9f" % (1 - BETA))
print("   cos(theta_M)      = %.9f" % math.cos(theta_M))
print("   cos^3(theta_M)    = %.9f" % (math.cos(theta_M) ** 3))
print("   (1-beta)^(-3/2)   = %.9f   <- M_Ch ~ G^(-3/2): E' ISTO que a restricao de G morde"
      % ((1 - BETA) ** -1.5))
print("   RESTRICAO [KNOWN]: M_Ch ~ G^(-3/2) => qualquer fator (1-beta) AQUI e' G -> G(1-beta),")
print("   falsificado por LLR/pulsares a ~100 sigma. NENHUMA potencia de beta escapa disso.")
print("   >>> M-4: sem veredito favoravel possivel sem enfrentar a restricao de G.")

# ---------------------------------------------------------------- veredito
print("\n" + "=" * 92)
print(" VEREDITO (criterios pre-registrados)")
print("=" * 92)
if not forma_ok:
    ver = "T09_FORMA_LINEAR_REPROVADA"
elif degenerado:
    ver = "T09_EXPOENTE_INDETERMINAVEL"
else:
    melhor = min(disps, key=disps.get)
    ver = "T09_SQRT_BETA_PREFERIDO" if melhor == 0.5 else (
        "T09_BETA_QUADRADO_PREFERIDO" if melhor == 2.0 else "T09_EXPOENTE_INDETERMINAVEL")
print("   >>> %s <<<" % ver)
print("\n   razao sqrt(beta)/beta^2 = beta^(-3/2) = %.2f  (fator que a troca produziria)"
      % BETA ** -1.5)
print("   M_form(n=2, k=2, R=57) = %.4e   vs  M_lit(GA) = %.4e   razao %.4f"
      % (M_form(2.0, 2.0, 57.0), 5.4e16, M_form(2.0, 2.0, 57.0) / 5.4e16))
print("   M_form(n=0.5, k=2, R=57) = %.4e  vs  M_lit(GA) = %.4e   razao %.1f"
      % (M_form(0.5, 2.0, 57.0), 5.4e16, M_form(0.5, 2.0, 57.0) / 5.4e16))

json.dump({"pre_registro_sha256": H09, "beta": BETA, "veredito": ver,
           "M1_p_ajustado": p_fit, "M1_err_p": err_p, "M1_disp_p_livre": disp_p,
           "M1_disp_p_forcado_1": disp1, "M1_espalhamento_M_sobre_R": espalha,
           "M1_forma_ok": forma_ok, "M3_dispersoes": disps, "M3_degenerado": degenerado,
           "ancoras": [{"nome": a[0], "R_Mpc": a[1], "M_lit": a[2], "ref": a[3]} for a in ANC]},
          io.open(SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n gravado: %s" % SAIDA)
