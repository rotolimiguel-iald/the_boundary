# -*- coding: utf-8 -*-
"""recalculo_ao_lado_v388.py -- AO LADO do RESULTADO congelado da Fase 9 (que NAO se reescreve; R7 do V1): a afericao adversarial da v388 (06/10) mediu que a
quadratura n = 25 do lnB(TGL/beta-livre) nao convergiu (integrando estreito: sigma_beta ~ 0,002 contra o prior de 0,1). Aqui: as MESMAS formulas do runner
(rodar_fase9_v1.py, sha256 f3108bc2...) sobre a MESMA entrada (o V1 por hash), com n = 25, 35, 60, 100, 200; e o z_disc nas duas formas (forma quadratica, a do
runner; e com o log-det, a variante que a afericao mediu). NAO e' veredito; nao muda o veredito (a regra 1 decide antes). Grava RECALCULO_AO_LADO_v388.json."""
import hashlib, json, math, os, re, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
V1J = r"C:\IALD\Bancada_Um\investigacao\preregistro_fase9_hubble\PREREGISTRO_FASE9_RAZAO_DE_HUBBLE_V1.json"
raw = open(V1J, "rb").read(); D = json.loads(raw.decode("utf-8")); SP = D["spec"]
assert hashlib.sha256(json.dumps(SP, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest() == D["spec_sha256"]
src = open(r"C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um.py", "rb").read()
ALPHA = float(re.search(rb"^ALPHA_FINE_CODATA_2018\s*=\s*([0-9.eE+-]+)", src, re.M).group(1)); BETA = ALPHA * math.sqrt(math.e)
K = SP["lei"]["K_no_fundo_nu"]; LN_E = 1.5 * math.log(K) / BETA; Kb = lambda b: math.exp(2.0 * b / 3.0 * LN_E)
LEIT, FAM, BG = SP["leitores"], SP["familias"], SP["fundos"]; S_SN, S_4258 = SP["covariancia"]["off_diagonal_sigma_SN"], SP["covariancia"]["off_diagonal_ancora_4258_L1xL2"]
ids = FAM["P_primaria"]["ids"]; y = np.array([LEIT[i]["H0"] for i in ids]); n = len(ids)
C = np.zeros((n, n))
for i, a in enumerate(ids):
    C[i, i] = LEIT[a]["sigma"] ** 2
    for j, b in enumerate(ids):
        if i != j:
            if LEIT[a]["grupo_SN"] and LEIT[b]["grupo_SN"]: C[i, j] += S_SN ** 2
            if LEIT[a]["grupo_4258"] and LEIT[b]["grupo_4258"]: C[i, j] += S_4258 ** 2
mu, s = BG["BG1"]["H0"], BG["BG1"]["sigma"]; one = np.ones(n)


def lnL(f):
    f = np.asarray(f, float); S = C + s ** 2 * np.outer(f, f); r = y - mu * f; q = float(r @ np.linalg.solve(S, r)); _, ld = np.linalg.slogdet(2 * math.pi * S)
    return -0.5 * (q + ld), q, ld


def lnZ(nq):
    x, w = np.polynomial.legendre.leggauss(nq); t = 0.05 * x
    v = np.array([lnL(Kb(b) * one)[0] for b in t]); m = v.max()
    return m + math.log(np.sum(w * np.exp(v - m)) * 0.5)


LT, qT, ldT = lnL(K * one); LL, qL, ldL = lnL(one)
Z = {nq: lnZ(nq) for nq in (25, 35, 60, 100, 200)}
lnB = {str(nq): LT - z for nq, z in Z.items()}
zq = math.copysign(math.sqrt(abs(qL - qT)), qL - qT); zld = math.copysign(math.sqrt(abs((qL + ldL) - (qT + ldT))), (qL + ldL) - (qT + ldT))
res = {"nota": __doc__.split("\n")[0], "runner_original_sha256": hashlib.sha256(open(os.path.join(HERE, "rodar_fase9_v1.py"), "rb").read()).hexdigest(),
       "v1_spec_sha256": D["spec_sha256"], "lnB_TGL_vs_beta_livre_por_n": lnB, "convergido_n200": lnB["200"], "erro_n25_n35": abs(lnB["25"] - lnB["35"]),
       "z_disc_forma_quadratica_runner": zq, "z_disc_com_logdet": zld,
       "estatuto": "[REAL — recalculo ao lado, mesma entrada e mesmas formulas; NAO e' veredito; o RESULTADO congelado fica como esta]"}
b = json.dumps(res, ensure_ascii=False, indent=1).encode("utf-8"); p = os.path.join(HERE, "RECALCULO_AO_LADO_v388.json")
open(p + ".tmp", "wb").write(b); os.replace(p + ".tmp", p)
print(json.dumps({k: res[k] for k in ("lnB_TGL_vs_beta_livre_por_n", "convergido_n200", "erro_n25_n35", "z_disc_forma_quadratica_runner", "z_disc_com_logdet")}, ensure_ascii=False))
print("sha256", hashlib.sha256(b).hexdigest())
