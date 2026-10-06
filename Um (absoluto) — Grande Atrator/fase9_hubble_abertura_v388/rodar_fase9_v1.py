# -*- coding: utf-8 -*-
"""rodar_fase9_v1.py -- A ABERTURA DA FASE 9 (a razao de Hubble) pela FUNCAO DE VEREDITO CONGELADA do V1 (spec d956c8db..., funcao 5940bf16...), ordem do operador
de 06/10/2026 via a sessao c1 («vamos em frente, haja v388 e abra a fase 9»). Segue a especificacao_do_runner do V1 (R0-R7) e as obrigacoes A1-A3 ditas no handoff v387:
 R0  este arquivo e' hasheado ANTES de abrir (RUNNER_ANTES_DE_ABRIR.json, gravado pela gerencia antes da execucao) -- o resultado carrega o sha256;
 R1  ALVOS_FASE9_LIDOS.json (fontes primarias por sha256, D-H) NAO existe: os 5 leitores do composto seguem [DECLARADO] => n_alvos_declarados = 5 (dito);
 R2  o fundo BG1 recalculado na Bancada (Planck comprimido + DESI DR2, LCDM, sem escada; as opcoes da Fase 1/3) e conferido |H0 - 68,531| < 0,01;
 R3  PODER_FASE9_ANTES_DE_ABRIR.json (so sigmas, K e o fundo) gravado ANTES de qualquer centro local entrar em calculo; conferido contra o V1;
 R4  a abertura: P_primaria x BG1; cinco hipoteses -- TGL (K fixo), LCDM, beta-livre U[-0,05; 0,05], shift so nas Cefeidas SH0ES U[-0,15; 0,15], fundo-livre (LCDM + r_d livre
     U[130; 160] Mpc, Planck comprimido + DESI DR2 reajustados na Bancada; todos os locais leem o mesmo H0);
 R5  a funcao executada e' o TEXTO do V1 (sha256 conferido), nunca uma copia reescrita; replicas e diagnosticos ao lado, nunca decisorios.
FIXADO AQUI, ANTES DE ABRIR (A2): a referencia da dispersao e' a media GLS dos leitores do composto com a covariancia declarada (sem o fundo), chi2_int com n-1 g.l.;
z_delta = (beta_hat - beta_TGL)/sigma_beta pelo perfil de -2 lnL (Delta = 1) da hipotese beta-livre; z_disc = sinal.sqrt(chi2_LCDM - chi2_TGL) (forma quadratica, sem o
log-det); lnB por quadratura de Gauss-Legendre (n = 25; erro = |n25 - n35|); o fundo entra como nuisance gaussiano comum H0_bg ~ N(mu, sigma) marginalizado
analiticamente (covariancia C + sigma^2 f f^T); fundo-livre: o fundo reajustado com r_d livre (Laplace: JtJ) e o fator de Occam do r_d (prior de 30 Mpc) somado ao lnZ.
Estatutos: os centros locais [DECLARADO ate D-H]; o fundo [REAL Bancada]; Laplace = aproximacao dita. NOT_FALSIFIED nunca e' confirmacao; o gate nao se move."""
import hashlib, json, math, os, sys, time
import numpy as np
from scipy import optimize, stats
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BANC = r"C:\IALD\Bancada_Um"
V1D = os.path.join(BANC, "investigacao", "preregistro_fase9_hubble")
V1J = os.path.join(V1D, "PREREGISTRO_FASE9_RAZAO_DE_HUBBLE_V1.json")
V1H = os.path.join(V1D, "V1_FASE9_HASHES.json")
SPEC_ESPERADO = "d956c8db6bb9ab8d9b8429861840cd9f30a06e2ba62cb12be9d15baea1deaf86"
FUNC_ESPERADA = "5940bf16310efb6a345e8de36dee0ebb304a16860928cd0f3e3e7b63d9c4e33c"
sha = lambda b: hashlib.sha256(b).hexdigest()
TS = time.strftime("%Y%m%d_%H%M%S")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)


def gravar(p, b):
    open(p + ".tmp", "wb").write(b); assert os.path.getsize(p + ".tmp") == len(b); os.replace(p + ".tmp", p)


# ---------- R0/V1 por hash ----------
me = os.path.abspath(__file__); me_sha = sha(open(me, "rb").read())
pre = json.load(open(os.path.join(HERE, "RUNNER_ANTES_DE_ABRIR.json"), encoding="utf-8"))
assert pre["runner_sha256"] == me_sha, "o runner mudou depois de hasheado (R0) -- NAO ABRO"
rawj = open(V1J, "rb").read(); D = json.loads(rawj.decode("utf-8")); SP = D["spec"]
assert sha(json.dumps(SP, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) == D["spec_sha256"] == SPEC_ESPERADO, "spec_sha256 do V1 nao bate -- NAO ABRO"
H = json.load(open(V1H, encoding="utf-8"))
for rel, e in H["arquivos"].items():
    b = open(os.path.join(V1D, *rel.split("/")), "rb").read(); assert sha(b) == e["sha256"] and len(b) == e["bytes"], ("V1 adulterado", rel)
FONTE = SP["funcao_de_veredito"]["fonte"]
assert sha(FONTE.encode("utf-8")) == SP["funcao_de_veredito"]["sha256"] == FUNC_ESPERADA == H["funcao_sha256"], "a funcao do V1 nao bate -- NAO ABRO"
ns = {}; exec(compile(FONTE, "<veredito_fase9_v1>", "exec"), ns); VF = ns["veredito_fase9_v1"]
log("V1 lido por hash: spec", SPEC_ESPERADO[:16], "| funcao", FUNC_ESPERADA[:16], "| runner", me_sha[:16])
# beta em runtime (o alpha do um.py por regex) e K do V1
import re
src = open(r"C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um.py", "rb").read()
ALPHA = float(re.search(rb"^ALPHA_FINE_CODATA_2018\s*=\s*([0-9.eE+-]+)", src, re.M).group(1)); BETA = ALPHA * math.sqrt(math.e)
assert abs(BETA / SP["beta"]["beta_runtime"] - 1) < 1e-12
K = SP["lei"]["K_no_fundo_nu"]; LN_E = 1.5 * math.log(K) / BETA   # ln E(z*) do proprio K do V1 (K = E^{2beta/3})
Kb = lambda b: math.exp(2.0 * b / 3.0 * LN_E)
LEIT = SP["leitores"]; FAM = SP["familias"]; BG = SP["fundos"]
S_SN, S_4258 = SP["covariancia"]["off_diagonal_sigma_SN"], SP["covariancia"]["off_diagonal_ancora_4258_L1xL2"]

# ---------- R1: os alvos ----------
ALV = os.path.join(HERE, "ALVOS_FASE9_LIDOS.json")
lidos = json.load(open(ALV, encoding="utf-8")) if os.path.exists(ALV) else {}
log("R1: ALVOS_FASE9_LIDOS.json", "presente" if lidos else "AUSENTE -- os leitores seguem [DECLARADO] (D-H do operador)")

# ---------- R2/fundo-livre: a Bancada ----------
sys.path.insert(0, BANC)
import torch  # noqa: E402
from bancada.motor import Motor, opcoes_de  # noqa: E402
M = Motor(BANC)
O = opcoes_de({"modo": "tgl"})   # as opcoes da Fase 1/3 (chen2019, razao_camb); com beta = 0 o mapa eff = nu


def res_bg(x, rd_livre):
    H0, wb, wc = x[0], x[1], x[2]
    th = torch.tensor([[H0, wb, wc, 0.0, 0.0]], dtype=torch.float64, device=M.dev)
    p = M.predizer(th, O)
    if not bool(p["dentro"][0]):
        return np.full(16, 1e6)
    rp = M._r_planck(p, O)[0]
    dr2 = p["dr2"][0] * (p["r_drag"][0] / x[3] if rd_livre else 1.0)
    rb = (dr2 - M.dr2_val) @ M.dr2_Linv.T
    return torch.cat([rp, rb]).cpu().numpy()


def ajuste(rd_livre, x0):
    lo = [60.0, 0.019, 0.08] + ([130.0] if rd_livre else []); hi = [80.0, 0.026, 0.16] + ([160.0] if rd_livre else [])
    f = lambda x: res_bg(x, rd_livre)
    r = optimize.least_squares(f, x0, bounds=(lo, hi), method="trf", x_scale="jac", diff_step=1e-6, xtol=1e-12, ftol=1e-12)
    J = r.jac; C = np.linalg.inv(J.T @ J)
    return r.x, float(r.fun @ r.fun), C


xb, chi_b, Cb = ajuste(False, [68.53, 0.02253, 0.1182])
log("R2: BG1 recalculado H0 = %.4f +- %.4f (V1: %.3f +- %.3f); chi2 = %.3f" % (xb[0], math.sqrt(Cb[0, 0]), BG["BG1"]["H0"], BG["BG1"]["sigma"], chi_b))
R2_OK = abs(xb[0] - BG["BG1"]["H0"]) < 0.01
assert R2_OK, "R2: o fundo recalculado nao bate com o do V1 -- NAO ABRO"
# r_d do modelo no ponto BG1 (para o x0 do fundo-livre)
th = torch.tensor([[xb[0], xb[1], xb[2], 0.0, 0.0]], dtype=torch.float64, device=M.dev); rd0 = float(M.predizer(th, O)["r_drag"][0])
xr, chi_r, Cr = ajuste(True, [xb[0], xb[1], xb[2], min(max(rd0, 131.0), 159.0)])
rd_no_limite = xr[3] <= 130.0 + 1e-6 or xr[3] >= 160.0 - 1e-6
lnZbg_ratio = -0.5 * (chi_r - chi_b) + 0.5 * math.log(np.linalg.det(Cr) / np.linalg.det(Cb)) + 0.5 * math.log(2 * math.pi) - math.log(30.0)
log("fundo-livre: H0 = %.3f +- %.3f, r_d = %.2f +- %.2f Mpc (modelo no BG1: %.2f), chi2 = %.3f (Delta %.3f), ln Z_bg(r_d livre/fixo) = %.3f [Laplace]%s"
    % (xr[0], math.sqrt(Cr[0, 0]), xr[3], math.sqrt(Cr[3, 3]), rd0, chi_r, chi_r - chi_b, lnZbg_ratio, " -- r_d NO LIMITE DO PRIOR (dito)" if rd_no_limite else ""))

# ---------- R3: o poder ANTES de abrir (so sigmas) ----------
def covm(ids):
    n = len(ids); C = np.zeros((n, n))
    for i, a in enumerate(ids):
        C[i, i] = LEIT[a]["sigma"] ** 2
        for j, b in enumerate(ids):
            if i != j:
                if LEIT[a]["grupo_SN"] and LEIT[b]["grupo_SN"]: C[i, j] += S_SN ** 2
                if LEIT[a]["grupo_4258"] and LEIT[b]["grupo_4258"]: C[i, j] += S_4258 ** 2
    return C


def poder(ids, mu, s):
    C = covm(ids); one = np.ones(len(ids)); sc = 1 / math.sqrt(one @ np.linalg.solve(C, one))
    return (K - 1) * mu / math.sqrt(sc ** 2 + (K * s) ** 2)


PODER = {"%s x %s" % (f, b): round(poder(FAM[f]["ids"], BG[b]["H0"], BG[b]["sigma"]), 3) for f in FAM for b in ("BG1", "BG2", "BG3")}
assert all(abs(PODER[k] - SP["poder_previo"]["por_familia_e_fundo"][k]["poder_K_menos_1_sigmas"]) < 2e-3 for k in PODER), "R3: o poder nao reproduz o V1"
pb = json.dumps({"quando": TS, "poder": PODER, "nota": "so sigmas publicas, K e o fundo; nenhum centro local entrou em calculo ate aqui"}, ensure_ascii=False, indent=1).encode("utf-8")
gravar(os.path.join(HERE, "PODER_FASE9_ANTES_DE_ABRIR.json"), pb); PODER_SHA = sha(pb)
log("R3: poder antes de abrir gravado", PODER_SHA[:16], "| primaria x BG1 =", PODER["P_primaria x BG1"])


# ---------- R4: a abertura ----------
def lnL(y, C, mu, s, f):
    f = np.asarray(f, float); S = C + (s ** 2) * np.outer(f, f); r = y - mu * f
    q = float(r @ np.linalg.solve(S, r)); sgn, ld = np.linalg.slogdet(2 * math.pi * S)
    return -0.5 * (q + ld), q


def lnZ_1d(fun, a, b, n):
    x, w = np.polynomial.legendre.leggauss(n); t = 0.5 * (b - a) * x + 0.5 * (b + a)
    v = np.array([fun(ti) for ti in t]); m = v.max()
    return m + math.log(np.sum(w * np.exp(v - m)) * 0.5 * (b - a) / (b - a))   # prior uniforme: o (b-a) do jacobiano cancela com a densidade 1/(b-a)


def abrir(fam, bgid, mu=None, s=None, decisorio=False):
    ids = FAM[fam]["ids"]; y = np.array([LEIT[i]["H0"] for i in ids]); C = covm(ids); n = len(ids)
    mu = BG[bgid]["H0"] if mu is None else mu; s = BG[bgid]["sigma"] if s is None else s
    one = np.ones(n)
    LT, qT = lnL(y, C, mu, s, K * one); LL, qL = lnL(y, C, mu, s, one)
    fb = lambda b: lnL(y, C, mu, s, Kb(b) * one)[0]
    bb = optimize.minimize_scalar(lambda b: -fb(b), bounds=(-0.05, 0.05), method="bounded", options={"xatol": 1e-10})
    bh = float(bb.x); Lmax = fb(bh)
    def lado(sg):
        g = lambda b: fb(b) - (Lmax - 0.5)
        lim = 0.05 * sg
        try:
            return abs(optimize.brentq(g, bh, lim) - bh) if g(lim) < 0 else float("nan")
        except ValueError:
            return float("nan")
    sp, sm = lado(1), lado(-1); sig_b = np.nanmean([sp, sm]) if not (math.isnan(sp) and math.isnan(sm)) else float("nan")
    z_delta = (bh - BETA) / sig_b if sig_b and not math.isnan(sig_b) else float("nan")
    no_lim = bh <= -0.05 + 1e-6 or bh >= 0.05 - 1e-6
    cef = np.array([1.0 if LEIT[i]["cefeida"] else 0.0 for i in ids])
    fs = lambda s_: lnL(y, C, mu, s, one + s_ * cef)[0]
    Zb25, Zb35 = lnZ_1d(fb, -0.05, 0.05, 25), lnZ_1d(fb, -0.05, 0.05, 35)
    if cef.any():
        Zs25, Zs35 = lnZ_1d(fs, -0.15, 0.15, 25), lnZ_1d(fs, -0.15, 0.15, 35)
    else:
        Zs25 = Zs35 = LL   # sem leitor Cefeida o shift e' identico ao LCDM (dito)
    LF, qF = lnL(y, C, xr[0], math.sqrt(Cr[0, 0]), one); ZF = LF + lnZbg_ratio
    zdisc = math.copysign(math.sqrt(abs(qL - qT)), qL - qT)
    W = np.linalg.solve(C, one); m_gls = float(W @ y / (one @ W)); r_ = y - m_gls; chi_int = float(r_ @ np.linalg.solve(C, r_))
    disp = chi_int / (n - 1) if n > 1 else 0.0; p_disp = float(stats.chi2.sf(chi_int, n - 1)) if n > 1 else 1.0
    zi = {i: {"H0": LEIT[i]["H0"], "sigma": LEIT[i]["sigma"], "z_vs_TGL": (LEIT[i]["H0"] - K * mu) / math.sqrt(LEIT[i]["sigma"] ** 2 + (K * s) ** 2),
              "z_vs_LCDM": (LEIT[i]["H0"] - mu) / math.sqrt(LEIT[i]["sigma"] ** 2 + s ** 2), "estatuto": "[DECLARADO]" if i not in lidos else "[REAL por hash]"} for i in ids}
    q = LEIT
    r = {"fundo": bgid, "n_alvos_declarados": sum(1 for i in ids if i not in lidos), "n_local": n, "dispersao_por_gl": disp, "p_dispersao": p_disp,
         "beta_livre_no_limite": no_lim, "z_delta": z_delta, "sigma_z_por_alvo": [(LEIT[i]["sigma"], zi[i]["z_vs_TGL"]) for i in ids],
         "poder_previo": PODER["%s x %s" % (fam, bgid)] if bgid in ("BG1", "BG2", "BG3") else float("nan"), "z_disc": zdisc,
         "lnB_lcdm": LT - LL, "lnB_livre": LT - Zb25, "lnB_shift": LT - Zs25, "lnB_fundo_livre": LT - ZF, "poder_nao_cefeida": PODER["N_sem_cefeidas x BG1"],
         "use_novel_parametros": bool(SP["novidade_de_uso"]["use_novel_parametros"])}
    v = VF(r)
    return {"familia": fam, "fundo": bgid, "decisorio": decisorio, "veredito": v, "entrada_da_funcao": r,
            "estatisticas": {"H0_bg": mu, "sigma_bg": s, "K": K, "H0_local_previsto": K * mu, "chi2_TGL": qT, "chi2_LCDM": qL, "beta_hat": bh, "sigma_beta": sig_b,
                             "sigma_beta_lados": [sm, sp], "lnL_TGL": LT, "lnL_LCDM": LL, "lnZ_beta_livre_n25": Zb25, "lnZ_beta_livre_n35": Zb35, "lnZ_shift_n25": Zs25,
                             "lnZ_shift_n35": Zs35, "lnZ_fundo_livre": ZF, "media_GLS_locais": m_gls, "sigma_GLS_locais": 1 / math.sqrt(one @ W), "chi2_int": chi_int,
                             "lnB": {"TGL_vs_LCDM": LT - LL, "TGL_vs_beta_livre": LT - Zb25, "TGL_vs_shift_cefeidas": LT - Zs25, "TGL_vs_fundo_livre": LT - ZF},
                             "lnB_erro_quadratura": {"beta_livre": abs(Zb25 - Zb35), "shift": abs(Zs25 - Zs35)}},
            "z_por_leitor": zi}


PRIM = abrir("P_primaria", "BG1", decisorio=True)
log("ABERTO P_primaria x BG1 ->", PRIM["veredito"])
LAT = {}
for fam in ("R_r25", "N_sem_cefeidas", "S_com_sirenes", "T_so_cchp"):
    LAT["%s x BG1" % fam] = abrir(fam, "BG1")
for bg in ("BG2", "BG3"):
    LAT["P_primaria x %s" % bg] = abrir("P_primaria", bg)
# ---------- R5: o resultado ----------
RES = {"id": "RESULTADO_FASE9_V1_%s" % TS, "quando": TS, "ordem": "operador 06/10/2026 via sessao c1: «vamos em frente, haja v388 e abra a fase 9»",
       "V1": {"spec_sha256": SPEC_ESPERADO, "funcao_sha256": FUNC_ESPERADA, "json_sha256": sha(rawj)}, "runner_sha256": me_sha, "poder_antes_de_abrir_sha256": PODER_SHA,
       "beta_runtime": BETA, "K": K, "R1_alvos": "ALVOS_FASE9_LIDOS.json AUSENTE: os 5 leitores do composto [DECLARADO] (fontes primarias por sha256 = D-H do operador)",
       "R2_fundo": {"H0": float(xb[0]), "sigma_JtJ": float(math.sqrt(Cb[0, 0])), "chi2": chi_b, "ok": R2_OK, "opcoes": O},
       "fundo_livre": {"H0": float(xr[0]), "sigma_H0": float(math.sqrt(Cr[0, 0])), "r_d": float(xr[3]), "sigma_r_d": float(math.sqrt(Cr[3, 3])), "r_d_modelo_no_BG1": rd0,
                       "chi2": chi_r, "delta_chi2_vs_rd_fixo": chi_r - chi_b, "lnZ_bg_ratio_laplace": lnZbg_ratio, "r_d_no_limite": rd_no_limite},
       "primario": PRIM, "ao_lado_nao_decisorios": LAT,
       "veredito": PRIM["veredito"],
       "leitura": ("o veredito e' o que a funcao congelada devolveu sobre a familia primaria x BG1; as replicas e os diagnosticos ficam ao lado, nunca decisorios; os centros locais "
                   "seguem [DECLARADO] ate as fontes por sha256 (D-H) -- com eles declarados a regra 1 do V1 devolve INCONCLUSIVE_SYSTEMATICS, e isso E' o resultado; as estatisticas "
                   "(lnB, z_delta, z_disc, z por leitor) sao reportadas inteiras, sem virar veredito. NOT_FALSIFIED nunca e' confirmacao; a RG/LCDM e' o limite classico; o gate nao se move."),
       "log": LOG}
rb = json.dumps(RES, ensure_ascii=False, indent=1, default=float).encode("utf-8")
pj = os.path.join(HERE, "RESULTADO_FASE9_V1_%s.json" % TS); gravar(pj, rb)
st = PRIM["estatisticas"]; zl = PRIM["z_por_leitor"]
md = ["# RESULTADO — Fase 9 aberta pela função congelada do V1 (%s)\n" % TS,
      "**Veredito (função `veredito_fase9_v1`, sha256 `%s`, executada do texto do V1):** `%s`\n" % (FUNC_ESPERADA[:16], PRIM["veredito"]),
      "V1 spec `%s`; runner `%s` (hasheado antes de abrir); poder antes de abrir `%s`. Os 5 leitores do composto seguem **[DECLARADO]** até as fontes por sha256 (D-H).\n" % (SPEC_ESPERADO[:16], me_sha[:16], PODER_SHA[:16]),
      "## Primário: P_primaria × BG1\n",
      "| grandeza | valor |\n|---|---|",
      "| fundo BG1 (R2 recalculado) | %.4f ± %.4f (V1: 68,531 ± 0,301) |" % (xb[0], math.sqrt(Cb[0, 0])),
      "| previsão local K·H0_bg | %.3f (K = %.6f) |" % (K * BG["BG1"]["H0"], K),
      "| média GLS dos locais | %.3f ± %.3f |" % (st["media_GLS_locais"], st["sigma_GLS_locais"]),
      "| χ²_TGL / χ²_ΛCDM | %.3f / %.3f |" % (st["chi2_TGL"], st["chi2_LCDM"]),
      "| z_disc | %+.3f |" % PRIM["entrada_da_funcao"]["z_disc"],
      "| β̂ (β-livre) | %.5f ± %.5f (β_TGL = %.6f) |" % (st["beta_hat"], st["sigma_beta"], BETA),
      "| z_δ | %+.3f |" % PRIM["entrada_da_funcao"]["z_delta"],
      "| dispersão χ²_int/(n−1) | %.3f (p = %.3f) |" % (PRIM["entrada_da_funcao"]["dispersao_por_gl"], PRIM["entrada_da_funcao"]["p_dispersao"]),
      "| ln B(TGL/ΛCDM) | %.3f |" % st["lnB"]["TGL_vs_LCDM"],
      "| ln B(TGL/β-livre) | %.3f (erro de quadratura %.1e) |" % (st["lnB"]["TGL_vs_beta_livre"], st["lnB_erro_quadratura"]["beta_livre"]),
      "| ln B(TGL/shift-Cefeidas) | %.3f |" % st["lnB"]["TGL_vs_shift_cefeidas"],
      "| ln B(TGL/fundo-livre, r_d) | %.3f (Laplace) |" % st["lnB"]["TGL_vs_fundo_livre"], "",
      "**z por leitor** (contra K·H0_bg e contra H0_bg; [DECLARADO]):\n", "| leitor | H0 | σ | z vs TGL | z vs ΛCDM |\n|---|---|---|---|---|"]
md += ["| %s | %.2f | %.3f | %+.2f | %+.2f |" % (k, v["H0"], v["sigma"], v["z_vs_TGL"], v["z_vs_LCDM"]) for k, v in zl.items()]
md += ["", "**Fundo-livre (LCDM + r_d livre):** H0 = %.3f ± %.3f, r_d = %.2f ± %.2f Mpc (modelo no BG1: %.2f), Δχ²_bg = %.3f, ln Z_bg(r_d livre/fixo) = %.3f [Laplace]%s.\n"
       % (xr[0], math.sqrt(Cr[0, 0]), xr[3], math.sqrt(Cr[3, 3]), rd0, chi_r - chi_b, lnZbg_ratio, " — r_d no limite do prior" if rd_no_limite else ""),
       "## Ao lado (não decisórios)\n", "| família × fundo | veredito da mesma função | z_δ | ln B(TGL/ΛCDM) |\n|---|---|---|---|"]
md += ["| %s | `%s` | %+.2f | %.2f |" % (k, v["veredito"], v["entrada_da_funcao"]["z_delta"], v["estatisticas"]["lnB"]["TGL_vs_LCDM"]) for k, v in LAT.items()]
md += ["", RES["leitura"], "", "JSON: `%s` sha256 `%s`." % (os.path.basename(pj), sha(rb))]
mb = "\n".join(md).encode("utf-8"); pm = os.path.join(HERE, "RESULTADO_FASE9_V1_%s.md" % TS); gravar(pm, mb)
log("RESULTADO:", os.path.basename(pj), sha(rb)[:16], "|", os.path.basename(pm), sha(mb)[:16])
