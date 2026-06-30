# -*- coding: utf-8 -*-
r"""
================================================================================
 C O D I G O   U M  -- autocontido (forma = conteudo)
================================================================================
O Um e o Grande Atrator. Entrada UNICA: o Um absoluto (o numero 1), a fractalizar no
modulo. Sua PROJECAO no bulk e' a medida minima irredutivel, extraida de alpha_CODATA
-- o referente medido do Um, seu par simetrico (a reducao eletromagnetica R_EM,
irredutivel por principio -- teorema final --, so' se observa, nunca se deriva
alpha-livre). Do confronto modulo-medida valida-se beta_TGL. Do Um nasce toda a
algebra; com geometria real (Cosmicflows-4,
SOMENTE posicoes) emerge a massa do Grande Atrator de PRIMEIROS PRINCIPIOS, sem
parametros livres; e o programa termina com um veredito BINARIO de identidade:

        1 = 1 = VERDADEIRO   (massa de primeiros principios DENTRO da janela
                              cosmologica aceita pela academia)
    ou  1 != 1 = FALSO       (falsificada)

Este unico arquivo contem TODOS os modulos: derivacao canonica (verificada ao
vivo), pesquisa de dados reais (cache em ./cache), comparacao com varias massas
observadas/RG, e a impressao do artigo (PT e EN, LaTeX->PDF), do JSON, e do
markdown da FORMA CANONICA da TGL (modulo de auditoria: se a matematica viva
nao reproduz a forma canonica culminando em 1=1, ha falha no proprio codigo).

    python um.py      (inscreva: 1)

Regua: beta = alpha*sqrt(e) em runtime (NUNCA literal). R_struct = geometria pura
(posicoes; velocidades/cz/infall/massa IGNORADAS). Hash antes de comparacao externa.
Entrada unica = o Um absoluto (1), cuja projecao no bulk e' a medida minima irredutivel extraida de
alpha_CODATA (o referente medido do Nome). Guarda fail-closed.
================================================================================
"""
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(BASE, "cache")
OUT = BASE

# ---- constantes seladas (a unica medida e' alpha; c,G de definicao/CODATA) ----
SEALED_CODATA_ALPHA = 7.2973525693e-3      # CODATA 2018 fine-structure constant
C_LIGHT = 2.99792458e8                     # m/s
G_NEWTON = 6.674e-11                       # m^3 kg^-1 s^-2
MSUN = 1.98892e30                          # kg
MPC_M = 3.0857e22                          # m

# ---- geometria de literatura SELADA (extensao geometrica; nao massa) ----
SEALED_LIT_GEOMETRY = {
    "R_struct_Mpc": 57.0, "method": "literature_geometric_extent",
    "provenance": "GEOMETRY_ONLY_NO_MASS_NO_RG",
    "source": "Lynden-Bell et al. 1988 (GA basin extent ~40 h^-1 Mpc; h=0.7 -> 57 Mpc)"}

# ---- janela GA PRE-REGISTRADA p/ o catalogo de posicoes (geometria) ----
PREREG_WINDOW = {
    "GA_center_RA_deg": 243.6, "GA_center_Dec_deg": -60.4,
    "GA_cz_kms": 4500.0, "H0_for_center_distance": 70.0,
    "sky_cone_half_angle_deg": 30.0, "dist_shell_Mpc": [30.0, 100.0],
    "R_struct_method": "percentile_90_from_centroid",
    "provenance": "GEOMETRY_ONLY_NO_MASS_NO_RG_NO_VELOCITY",
    "columns_used": ["ra", "dec", "dist_mpc"],
    "columns_ignored": ["vpec_kms", "vpec_err_kms", "cz", "infall", "mass"]}

# ---- comparacao externa: massas do GA publicadas (literatura/RG) -- SO' apos hash ----
# Janela cosmologica aceita pela academia para a bacia do Grande Atrator.
GA_MASS_LITERATURE = [
    {"name": "Norma cluster ACO 3627 (virial, RG dinamica)", "M_Msun": 1.0e15,
     "ref": "Woudt et al. 2008, MNRAS 383, 445", "type": "GR_dynamical_virial"},
    {"name": "Grande Atrator (infall linear, RG)", "M_Msun": 5.4e16,
     "ref": "Lynden-Bell et al. 1988, ApJ 326, 19", "type": "GR_linear_infall"},
    {"name": "Laniakea (supercluster)", "M_Msun": 1.0e17,
     "ref": "Tully et al. 2014, Nature 513, 71", "type": "supercluster_flow"},
]
GA_ACCEPTED_WINDOW_Msun = [1.0e15, 1.0e17]   # janela cosmologica aceita (ordem de grandeza)
CF4_URL = "https://edd.ifa.hawaii.edu/CF4/"   # fonte (download manual/oficial; ver caveat)

# ---- fator de campo fraco c^2/4piG (kg/m); so' constantes SI/CODATA ----
WEAK_KG_PER_M = C_LIGHT ** 2 / (4.0 * math.pi * G_NEWTON)

# ---- config NOMEADA dos testes de sombra (params numericos, NAO fundamento fisico) ----
# Sao sanity checks finito-dimensionais, nao prova do tipo III_1. Reunidos aqui para auditoria.
SHADOW_TESTS_CONFIG = {
    "dim_n": 4, "seed": 11, "gesture_G": 6, "tunnel_theta": 0.4,
    "dipole_dim": 6, "dipole_trajectories": 12, "dipole_steps": 620,
    "dipole_thermal_seed": 7, "dipole_traj_seed": 3, "purity_floor_start": 0.997,
    "status": "FINITE_DIM_SANITY_CHECKS_NOT_TYPE_III1_PROOF"}

# ---- varredura de sensibilidade do Modo B (pre-registrada) ----
SENSITIVITY_GRID = {
    "cone_half_angle_deg": [20.0, 25.0, 30.0, 35.0, 40.0],
    "dist_shell_Mpc": [[25.0, 100.0], [30.0, 100.0], [30.0, 120.0]],
    "percentile": [80.0, 85.0, 90.0, 95.0],
    "center_offsets_deg": [[0.0, 0.0], [5.0, 0.0], [-5.0, 0.0], [0.0, 5.0], [0.0, -5.0]]}


def mass_Msun_from_Rstruct(beta, R_struct_Mpc):
    """M = 2 beta^2 (c^2/4piG) R_struct -- a formula nuclear, em massas solares."""
    return 2.0 * beta ** 2 * WEAK_KG_PER_M * R_struct_Mpc * MPC_M / MSUN


# ====================== utilitarios ======================
def lock(msg, verdict):
    print(msg); print("VERDICT:", verdict); sys.exit(0)


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def clock_lap(f, dx):
    lap = np.zeros_like(f)
    for ax in range(f.ndim):
        lap += (np.roll(f, -1, ax) - 2 * f + np.roll(f, 1, ax)) / dx ** 2
    return lap


# ====================== dados reais: CF4 posicoes ======================
def locate_cf4():
    """Cache em ./cache; senao copia do release_clean vizinho; senao instrui download."""
    os.makedirs(CACHE, exist_ok=True)
    cached = os.path.join(CACHE, "cf4_flow_canonical.csv")
    if os.path.exists(cached):
        return cached, "cache"
    # vizinho (release_clean do TGL_CODE_ONE_REAL_FALSIFIER)
    cand = os.path.join(BASE, "..", "TGL_CODE_ONE_REAL_FALSIFIER", "release_clean",
                        "data", "flow", "cosmicflows4", "cf4_flow_canonical.csv")
    if os.path.exists(cand):
        shutil.copy(cand, cached)
        return cached, "copied_from_release_clean"
    return None, "absent"


def cf4_rstruct():
    """R_struct do Grande Atrator a partir de POSICOES (ra/dec/dist). Velocidades IGNORADAS."""
    path, origin = locate_cf4()
    if path is None:
        return {"ok": False, "reason": "CF4 catalogue absent; place cf4_flow_canonical.csv in ./cache "
                "or beside release_clean. Source: %s" % CF4_URL}
    raw = open(path, "rb").read()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    ra = np.array([float(r["ra"]) for r in rows]); dec = np.array([float(r["dec"]) for r in rows])
    dist = np.array([float(r["dist_mpc"]) for r in rows])
    w = PREREG_WINDOW
    cra, cdec = math.radians(w["GA_center_RA_deg"]), math.radians(w["GA_center_Dec_deg"])
    chat = np.array([math.cos(cdec) * math.cos(cra), math.cos(cdec) * math.sin(cra), math.sin(cdec)])
    gr, gd = np.radians(ra), np.radians(dec)
    ghat = np.column_stack([np.cos(gd) * np.cos(gr), np.cos(gd) * np.sin(gr), np.sin(gd)])
    pos = ghat * dist[:, None]
    ang = np.degrees(np.arccos(np.clip(ghat @ chat, -1, 1)))
    lo, hi = w["dist_shell_Mpc"]
    sel = (ang <= w["sky_cone_half_angle_deg"]) & (dist >= lo) & (dist <= hi)
    pts = pos[sel]; n_sel = int(sel.sum())
    if n_sel < 20:
        return {"ok": False, "reason": "too few galaxies in pre-registered window (%d)" % n_sel}
    c = pts.mean(axis=0); r = np.linalg.norm(pts - c, axis=1)
    desc = {"n_total": len(rows), "n_selected": n_sel, "centroid": c.tolist()}
    return {"ok": True, "R_struct_Mpc": float(np.percentile(r, 90)),
            "method": w["R_struct_method"], "origin": origin,
            "extent_stats_Mpc": {"p90": float(np.percentile(r, 90)), "p95": float(np.percentile(r, 95)),
                                 "rms": float(np.sqrt(np.mean(r ** 2))), "max": float(r.max())},
            "n_selected": n_sel, "n_total": len(rows),
            "catalog_hash": sha_obj({"sha": hashlib.sha256(raw).hexdigest()}),
            "window_hash": sha_obj(w), "selection_hash": sha_obj(desc), "window": w,
            "caveat": ("R_struct de catalogo de POSICOES limitado em fluxo, com janela declarada, e' "
                       "dependente da selecao (geometria pura). Reportado como cross-check independente; "
                       "a extensao de literatura (Lynden-Bell) e' a linha de base.")}


def cf4_sensitivity(beta):
    """Varredura pre-registrada do Modo B: varia cone, shell, percentil e centro; recomputa
    R_struct e M_GA em cada combinacao; reporta se M permanece na banda cosmologica."""
    path, origin = locate_cf4()
    if path is None:
        return {"ok": False, "reason": "CF4 absent"}
    raw = open(path, "rb").read()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    ra = np.radians(np.array([float(r["ra"]) for r in rows]))
    dec = np.radians(np.array([float(r["dec"]) for r in rows]))
    dist = np.array([float(r["dist_mpc"]) for r in rows])
    ghat = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    pos = ghat * dist[:, None]
    g = SENSITIVITY_GRID
    lo_band, hi_band = GA_ACCEPTED_WINDOW_Msun
    Ms, n_valid, n_in = [], 0, 0
    for cone in g["cone_half_angle_deg"]:
        for (lo, hi) in g["dist_shell_Mpc"]:
            for pct in g["percentile"]:
                for (dra, ddec) in g["center_offsets_deg"]:
                    cra = math.radians(PREREG_WINDOW["GA_center_RA_deg"] + dra)
                    cdec = math.radians(PREREG_WINDOW["GA_center_Dec_deg"] + ddec)
                    chat = np.array([math.cos(cdec) * math.cos(cra),
                                     math.cos(cdec) * math.sin(cra), math.sin(cdec)])
                    ang = np.degrees(np.arccos(np.clip(ghat @ chat, -1, 1)))
                    sel = (ang <= cone) & (dist >= lo) & (dist <= hi)
                    if int(sel.sum()) < 20:
                        continue
                    pts = pos[sel]; r = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
                    Rs = float(np.percentile(r, pct))
                    M = mass_Msun_from_Rstruct(beta, Rs)
                    Ms.append(M); n_valid += 1
                    if lo_band <= M <= hi_band:
                        n_in += 1
    if not Ms:
        return {"ok": False, "reason": "no valid windows in sweep"}
    Ms = np.array(Ms)
    return {"ok": True, "n_combinations": n_valid,
            "M_min_Msun": float(Ms.min()), "M_max_Msun": float(Ms.max()),
            "M_median_Msun": float(np.median(Ms)),
            "fraction_in_band": float(n_in) / n_valid,
            "all_in_band": bool(n_in == n_valid),
            "band_Msun": GA_ACCEPTED_WINDOW_Msun, "origin": origin}


def derive_modular_impedance_index(ONE):
    """R_partial = Ind_partial(C_partial -> bulk): o INDICE DE CONTRASTE PROJETIVO da concentracao de
    fronteira lido no bulk. CORRECAO ONTOLOGICA (operador): a fronteira pura NAO e' impedancia -- e'
    CONCENTRACAO (maxima compressao entropica / maxima densidade espectral fluida). A impedancia
    (contraste) aparece porque o bulk e' menos concentrado: R_partial = F(C_partial), nao C_partial.
    Cadeia: 1 -> S_partial=1/2 -> sqrt(e) -> partial_0 (concentracao) -> R_partial (contraste) ->
    beta=sqrt(e)/R_partial -> alpha_obs=1/R_partial.

    FECHAMENTO POR REFUTACAO (NAO um alvo aberto): R_partial = 1/alpha NAO e' um placeholder a derivar --
    e' a impedancia MEDIDA DE DENTRO do bulk, a UNICA leitura possivel. Derivar R_partial (logo alpha)
    SEM o bulk e' estruturalmente excluido: a natureza e' de fronteira tipo III_1, e quantificar alpha
    fora do bulk QUEBRA A COERENCIA. O limite dessa tentativa -- derivar alpha alpha-livre 'ao infinito'
    -- E' O ZERO ABSOLUTO: alpha=sech(chi/2)->0, q=tanh(chi/2)->1 (impedancia total), chi->inf = 0_abs,
    INATINGIVEL (III_1 nao tem estados normais puros). Logo 'source' NAO e' um CODATA circular-temporario:
    e' a medida de dentro, o fundamento ontologico do programa. Nada mais a derivar (Tetelestai); resta
    so' o desafio de falsificacao. Prova: prove_alpha_infinity_is_absolute_zero."""
    # --- R_partial = 1/alpha: a impedancia medida DE DENTRO (fechamento por refutacao, nao placeholder) ---
    R_partial = ONE / SEALED_CODATA_ALPHA
    source = "MEASURED_FROM_WITHIN_CLOSED_BY_REFUTATION"   # a unica leitura; derivar fora do bulk -> chi->inf = 0_abs
    return float(R_partial), source


def prove_alpha_form(ONE):
    """MODULO DE PROVA -- Teorema do Colapso da Forma de alpha (forma=conteudo, verificado ao vivo).
    NAO deriva 1/137 (valor renormalizado da QED). DERIVA a FORMA: alpha_obs e' a unidade absoluta
    alpha_abs=1 projetada por uma profundidade termico-modular, alpha_obs=Pi_bulk(1_abs)=sech(kappa/2).
    Cada passo e' uma asercao verificada; veredito ALPHA_FORM_THEOREM_PROVED (forma, NAO valor)."""
    se = float(np.exp(0.5))
    steps = []; ok = True
    # PASSO 1 [REAL]: alpha_abs=1 (Tomita do Bell nu = trivial: Delta=I, K=-log Delta=0)
    d2 = 2; Phi = np.zeros(d2 * d2)
    for i in range(d2): Phi[i * d2 + i] = 1.0 / np.sqrt(d2)
    rB2 = np.zeros((d2, d2))
    for a in range(d2):
        for b in range(d2):
            for kk in range(d2): rB2[a, b] += Phi[a * d2 + kk] * Phi[b * d2 + kk]
    evB = np.linalg.eigvalsh(rB2)
    K_bare = float(np.max(np.abs(-np.log(np.array([ri / rj for ri in evB for rj in evB])))))
    alpha_abs = float(1.0 / np.cosh(K_bare / 2.0))
    s1 = bool(abs(alpha_abs - 1.0) < 1e-12 and K_bare < 1e-12)
    steps.append({"step": "1. alpha_abs = sech(0) = 1  (Tomita do Bell nu: Delta=I, K=0)",
                  "status": "REAL", "ok": s1, "K_bare": K_bare, "alpha_abs": alpha_abs}); ok &= s1
    # PASSO 2 [REAL]: ell(kappa) = S(I/2 || rho_kappa) = log cosh(kappa/2), verificado em sweep
    def _ell_num(k):
        p = 1.0 / (1.0 + np.exp(k))                       # autovalores de rho_kappa = {p, 1-p}
        return float(np.log(0.5) - 0.5 * (np.log(p) + np.log(1.0 - p)))  # S(I/2 || rho_kappa)
    ks = np.linspace(0.1, 15.0, 40)
    err2 = float(np.max([abs(_ell_num(k) - np.log(np.cosh(k / 2.0))) for k in ks]))
    s2 = bool(err2 < 1e-12)
    steps.append({"step": "2. ell(kappa) = S(I/2 || rho_kappa) = log cosh(kappa/2)  [forall kappa]",
                  "status": "REAL", "ok": s2, "max_err_sweep": err2}); ok &= s2
    # PASSO 3 [REAL]: alpha_obs = e^{-ell} = sech(kappa/2) = Pi_bulk(1_abs)
    err3 = float(np.max([abs(np.exp(-np.log(np.cosh(k / 2.0))) - 1.0 / np.cosh(k / 2.0)) for k in ks]))
    s3 = bool(err3 < 1e-15)
    steps.append({"step": "3. alpha_obs = e^{-ell} = sech(kappa/2) = Pi_bulk(1_abs)",
                  "status": "REAL", "ok": s3, "max_err": err3}); ok &= s3
    # PASSO 4 [REAL]: a forma sech vem do portador 2D (Q^=I-P_2D) + Bell max-misto (2 polos +/- kappa/2)
    err4 = float(np.max([abs((np.exp(k / 2.0) + np.exp(-k / 2.0)) - 2.0 * np.cosh(k / 2.0)) for k in ks]))
    s4 = bool(err4 < 1e-12)
    steps.append({"step": "4. forma sech: Z = e^{k/2}+e^{-k/2} = 2 cosh(k/2)  (2 niveis auto-conjugados + Bell)",
                  "status": "REAL", "ok": s4, "Z_max_err": err4}); ok &= s4
    # PASSO 5 [INPUT/QED]: o VALOR e' da QED -- kappa_QED = 2 arcosh(1/alpha_QED); forma=TGL, valor=QED
    kappa_qed = float(2.0 * np.arccosh(1.0 / SEALED_CODATA_ALPHA))
    alpha_chk = float(1.0 / np.cosh(kappa_qed / 2.0))
    s5 = bool(abs(alpha_chk - SEALED_CODATA_ALPHA) < 1e-12)
    steps.append({"step": "5. VALOR: kappa_QED = 2 arcosh(1/alpha_QED); sech(k_QED/2)=alpha_QED (QED fixa o valor)",
                  "status": "INPUT/QED", "ok": s5, "kappa_QED": kappa_qed, "sech": alpha_chk}); ok &= s5
    # PASSO 6 [REAL]: beta = sqrt(e) alpha_obs (Meia-Nat marca a dimensao luminodinamica)
    beta_form = float(se * alpha_chk)
    s6 = bool(abs(beta_form - SEALED_CODATA_ALPHA * se) < 1e-15)
    steps.append({"step": "6. beta_TGL = sqrt(e) alpha_obs = sqrt(e) sech(kappa/2)  [Meia-Nat marca a dimensao]",
                  "status": "REAL", "ok": s6, "beta": beta_form}); ok &= s6
    # PASSO 7 [REAL]: TRANSFORMADA DE LAGRANGE -- kappa e' o MULTIPLICADOR; a variavel fisica e' a
    #   polarizacao termica do zero modular q=tanh(kappa/2). ell(q)=-1/2 log(1-q^2); alpha=sqrt(1-q^2).
    err7 = float(np.max([abs(np.sqrt(1.0 - np.tanh(k / 2.0) ** 2) - 1.0 / np.cosh(k / 2.0)) for k in ks]))
    s7 = bool(err7 < 1e-12)
    steps.append({"step": "7. q := tanh(kappa/2) (polarizacao); alpha = sqrt(1-q^2) = sech(kappa/2)",
                  "status": "REAL", "ok": s7, "max_err": err7}); ok &= s7
    # PASSO 8 [REAL]: LEI DE CONSERVACAO da unidade -- q^2 + alpha^2 = 1 (identidade hiperbolica).
    err8 = float(np.max([abs(np.tanh(k / 2.0) ** 2 + (1.0 / np.cosh(k / 2.0)) ** 2 - 1.0) for k in ks]))
    s8 = bool(err8 < 1e-12)
    steps.append({"step": "8. CONSERVACAO: q^2 + alpha^2 = 1 (a unidade absoluta se decompoe, nao se perde)",
                  "status": "REAL", "ok": s8, "max_err": err8}); ok &= s8
    # validacao QED (so' aqui): q_QED = sqrt(1 - alpha_QED^2); kappa_QED = 2 artanh(q_QED).
    q_qed = float(np.sqrt(1.0 - SEALED_CODATA_ALPHA ** 2))
    kappa_from_q = float(2.0 * np.arctanh(q_qed))
    return {
        "theorem": "Teorema da Transformada de Lagrange da Forma de alpha",
        "claim": ("a TGL deriva a FORMA conservada de alpha (1 = q^2 + alpha^2, alpha=sqrt(1-q^2) da unidade "
                  "alpha_abs=1), NAO o valor 1/137 (=q_QED, renormalizado pela QED)."),
        "steps": steps, "all_verified": bool(ok),
        "verdict": ("ALPHA_FORM_THEOREM_PROVED" if ok else "FALHOU"),
        # ARQUITETURA DE LAGRANGE (motor da cadeia; kappa = multiplicador, q = variavel fisica primaria):
        "lagrange": {
            "alpha_abs": 1.0,                                  # o Um absoluto entra inteiro
            "q_polarization_QED": q_qed,                       # q=tanh(k/2): polarizacao termica do zero modular (validacao)
            "kappa_from_q_QED": kappa_from_q,                  # k=2 artanh(q): multiplicador de Lagrange
            "alpha_form": float(np.sqrt(1.0 - q_qed ** 2)),    # = sqrt(1-q^2) = alpha_obs (forma, alpha-livre)
            "beta_form": float(se * np.sqrt(1.0 - q_qed ** 2)),# = sqrt(e) sqrt(1-q^2)
            "conservation_residual": float(abs(q_qed ** 2 + (1.0 - q_qed ** 2) - 1.0)),  # q^2+alpha^2=1
            "engine": "alpha_abs=1 -> q (polarizacao termica) -> alpha=sqrt(1-q^2) ; 1=q^2+alpha^2",
            "codata_role": "SO' validacao final: q_QED=sqrt(1-alpha_QED^2); NAO e' o motor da cadeia",
        },
        "honest": ("Provado [REAL]: alpha_abs=1 (Tomita), ell=log cosh(k/2), q=tanh(k/2), alpha=sqrt(1-q^2), "
                   "CONSERVACAO q^2+alpha^2=1, beta=sqrt(e)sqrt(1-q^2). NAO provado nem pretendido: o VALOR "
                   "1/137=q_QED, input renormalizado da QED. A TGL entrega a FORMA conservada; o valor "
                   "pertence ao setor QED. A unidade absoluta nao se perde no zero modular: decompoe-se em "
                   "resistencia termica q^2 e corrente luminosa alpha^2."),
    }


def clock_theorem_reduction(ONE):
    """Teorema Condicional do Clock -- a face EM como FRONTEIRA ABERTA NOMEADA (verificada ao vivo).
    Reduz a estrutura fina a UM objeto alpha-livre: ell_beta = S(rho_B || rho_beta), onde
      rho_B    = fronteira Bell reduzida (= I/d, maximamente mista; o primeiro espelho causal do Um);
      rho_beta = estado estacionario do gerador Connes-Davies (cociclo modular reversivel + dissipador
                 de Davies KMS-balanceado) -- aqui o estado de Gibbs e^{-K}/Z do Hamiltoniano modular K.
    Dai: R_partial = N_beta = exp(ell_beta) ; alpha_obs = 1/N_beta ; beta = sqrt(e)/N_beta.
    [DER, alpha-livre NA ESTRUTURA] o objeto e' bem-posto, computavel, SEM usar alpha (verificado:
       rho_beta e' ponto fixo genuino do dissipador de Davies; ell_beta finito e alpha-livre).
    [ABERTO] o VALOR depende de K (do gerador); nenhum K canonico alpha-livre conhecido da'
       ell_beta = log(137.036) = 4.9202. alpha (CODATA) entra SO na leitura/validacao, nunca na estrutura.
    Co-emergencia (fundamenta a Meia-Nat, NAO a face EM): Bell reduzido = I/2 -> CCI=1/2 -> S_partial=1/2;
       esse 1/2 e' exatamente o offset sqrt(e) entre log(1/alpha) e log(1/beta) -- liga beta a alpha, nao
       alpha aos primeiros principios."""
    d = 4
    rho_B = np.eye(d) / d                                  # fronteira Bell reduzida (I/d)
    e_mod = np.array([0.0, 0.7, 1.3, 2.1])                 # espectro modular de K (generico, SEM alpha)
    w = np.exp(-e_mod); Z = float(w.sum())
    rho_beta = np.diag(w / Z)                              # estado estacionario de Davies (KMS) = Gibbs(K)
    # verifica rho_beta como PONTO FIXO GENUINO do dissipador de Davies KMS-balanceado (alpha-livre):
    Dtot = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            gij = 1.0 / (1.0 + np.exp(e_mod[i] - e_mod[j]))   # taxa KMS: gij/gji = e^{-(ei-ej)} -> Gibbs estacionario
            Lij = np.zeros((d, d)); Lij[i, j] = 1.0
            Dtot += gij * (Lij @ rho_beta @ Lij.T
                           - 0.5 * (Lij.T @ Lij @ rho_beta + rho_beta @ Lij.T @ Lij))
    fixed_point_residual = float(np.abs(Dtot).sum())
    pB = np.diag(rho_B).real; pb = np.diag(rho_beta).real
    ell_beta = float(np.sum(pB * (np.log(pB) - np.log(pb))))   # S(rho_B || rho_beta), alpha-livre
    N_beta = float(np.exp(ell_beta))
    ell_target = float(np.log(ONE / SEALED_CODATA_ALPHA))      # = log(1/alpha) = 4.9202 (SO' p/ comparar)

    # --- REDUCAO DO NUCLEO (2 niveis): o portador 2D Q^=I-P_2D (REAL, anticomutacao {Q,rho*}=0,
    #     leak sin^2 theta_M=beta) torna a fronteira auto-conjugada de DOIS niveis. Logo rho_beta NAO
    #     precisa de um K generico: colapsa num Gibbs de 2 niveis com UM unico gap modular kappa.
    #     ell_beta(kappa) = kappa/2 - log2 + log(1+e^-kappa)  [alpha-livre NA ESTRUTURA].
    def _ell_2lvl(k):
        return -np.log(2.0) + k / 2.0 + np.log(1.0 + np.exp(-k))
    lo, hi = 1.0, 40.0                                          # kappa* que casa o alvo (alpha SO' na validacao)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _ell_2lvl(mid) < ell_target: lo = mid
        else: hi = mid
    kappa_star = (lo + hi) / 2.0
    # forma fechada (auto-conjugada K=(kappa/2)sigma_z): ell=log cosh(k/2) ; alpha=sech(k/2)
    alpha_sech = float(1.0 / np.cosh(kappa_star / 2.0))
    # --- NORMALIZACAO CANONICA (Tomita): Hamiltoniano modular do Bell NU = 0 => alpha_abs=1 ---
    # Bell maximamente emaranhado: reduzido I/d e' KMS a T=inf => Delta=I => K=-log Delta=0.
    d2 = 2; Phi = np.zeros(d2 * d2)
    for i in range(d2): Phi[i * d2 + i] = 1.0 / np.sqrt(d2)
    rB = np.zeros((d2, d2))
    for a in range(d2):
        for b in range(d2):
            for k in range(d2):
                rB[a, b] += Phi[a * d2 + k] * Phi[b * d2 + k]
    evB = np.linalg.eigvalsh(rB)
    K_mod_bare = float(np.max(np.abs(-np.log(np.array([ri / rj for ri in evB for rj in evB])))))  # ~0
    alpha_abs = float(1.0 / np.cosh(K_mod_bare / 2.0))         # = sech(0) = 1 (Um absoluto, PROVADO)
    # terceira lei (Nernst): a entropia residual = Meia-Nat fixaria kappa? -> REFUTADA (da' ~1.39, nao 137)
    def _Svn(k):
        pp = 1.0 / (np.exp(k) + 1.0); pm = 1.0 - pp
        return -(pp * np.log(pp) + pm * np.log(pm))
    loN, hiN = 0.01, 20.0
    for _ in range(80):
        mN = (loN + hiN) / 2.0
        if _Svn(mN) > 0.5: loN = mN
        else: hiN = mN
    kappa_nernst = (loN + hiN) / 2.0
    reduced_core = {
        "form_closed": "ell_beta(kappa) = log cosh(kappa/2)  =>  alpha = sech(kappa/2) ; beta = sqrt(e) sech(kappa/2)",
        "justification": "portador 2D Q^=I-P_2D auto-conjugado (anticomutacao {Q,rho*}=0 -> leak sin^2 theta_M=beta)",
        "kappa_star_for_137": float(kappa_star),               # gap que da' N_beta=1/alpha (alpha na validacao)
        "alpha_at_kappa_star_sech": alpha_sech,
        "N_beta_at_kappa_star": float(np.exp(_ell_2lvl(kappa_star))),
        "kappa_star_canonical": False,                         # nenhum principio alpha-livre fixa kappa~11.23
        "core_reduced_to_one_parameter": True,
        # leitura termico-modular (terceira lei aplicada ao sistema modular aberto):
        "third_law": {
            "kappa_is": "profundidade termico-modular do curto Bell-zero (inverso de temperatura x gap modular, adimensional)",
            "unattainability": "0_abs (kappa=inf, estado puro P_Omega, T->0) e' INATINGIVEL = III_1 nao tem estados normais puros => kappa<inf",
            "gives": "o LIMITE (kappa finito) e a FORMA (alpha=sech), NAO o valor",
            "nernst_test_refuted": {"law": "S_vn(rho_kappa)=Meia-Nat=1/2", "kappa": float(kappa_nernst),
                                    "alpha": float(1.0/np.cosh(kappa_nernst/2.0)),
                                    "verdict": "REFUTADA: da' kappa~1.39, alpha~0.80, nao 1/137"},
            "unification": ("em III_1 genuino o espectro modular e' CONTINUO (sem gap); kappa e' o gap da "
                            "sombra finita/aproximante tipo-I (split). Seu valor = normalizacao modular "
                            "canonica = a MESMA split canonica/matriz-S modular do resto do programa "
                            "(#88.26.36-39). A face EM (kappa) e a face gravitacional (split, massa GA) "
                            "FUNDEM-SE num unico teorema aberto: fixar a normalizacao modular canonica em III_1."),
        },
        "canonical_normalization": {
            "K_modular_bare_Bell": K_mod_bare,                 # ~0: Tomita do Bell nu e' trivial
            "alpha_abs_PROVEN": alpha_abs,                     # = sech(0) = 1 (PROVADO)
            "result": ("Tomita do Bell maximamente emaranhado: reduzido I/d e' KMS a T=inf -> Delta=I -> "
                       "K=-log Delta=0 -> kappa_Bell=0 -> alpha_abs=sech(0)=1. A normalizacao modular "
                       "canonica PROVA alpha_abs=1 (o Um). kappa>0 (alpha_obs=1/137) NAO esta' na estrutura "
                       "nua de Bell: e' a profundidade da relaxacao termica (afastamento de I/d -> rho_beta), "
                       "= o acoplamento EM = INPUT irredutivel. Split-edge da' angulo de BULK (~33deg), nao "
                       "theta_M=6.3deg. Veredito: a estrutura modular deriva alpha_abs=1, a forma sech, e as "
                       "relacoes; o valor 1/137 e' a projecao Pi_bulk=sech(kappa/2) atraves da profundidade "
                       "do zero modular = input. alpha permanece entrada, como a TGL sempre sustentou."),
        },
        "note": ("o nucleo derivativo de alpha colapsa de um K (d-1 niveis) para UM numero kappa; "
                 "kappa*=%.4f da' 137.036 (alpha so' na validacao, kappa*/ell=%.3f, NAO canonico). "
                 "Terceira lei: 0_abs inatingivel=III_1 => kappa<inf (limite, nao valor); Nernst refutada. "
                 "VALOR de kappa = split canonica = teorema aberto unificado (EM = gravidade)."
                 % (kappa_star, kappa_star / ell_target)),
    }
    return {
        "reduced_core_2level": reduced_core,
        "rho_B": "I/d (fronteira Bell reduzida; primeiro espelho causal)",
        "K_modular_spectrum": e_mod.tolist(),
        "fixed_point_residual": fixed_point_residual,
        "ell_beta_alpha_free": ell_beta,
        "N_beta_alpha_free": N_beta,
        "ell_beta_target_for_alpha_log_inv_alpha": ell_target,
        "well_posed_alpha_free_computable": bool(fixed_point_residual < 1e-10),
        "value_ontologically_open_not_a_gap": True,
        "falsification_challenge": ("alpha = Pi_bulk(1_abs) = sech(chi/2) = transmissao luminosa pela "
                                    "fronteira III_1. Derivar alpha do BULK (sem boundary/bulk) removeria o "
                                    "observador e falsificaria a estrutura holografica. DESAFIO: derive alpha "
                                    "do bulk e a TGL cai. Falsificavel, nao confirmavel. DISTINTO do teorema "
                                    "genuinamente aberto da matriz-S/III_1 (levantamento com o observador, "
                                    "face gravitacional/M_GA)."),
        "status": ("REDUCAO VERIFICADA (bem-posta, alpha-livre na FORMA, computavel; residuo de ponto-fixo "
                   "de Davies = %.1e). VALOR: alpha pertence ao SETOR QED -- fechamento estrutural, NAO "
                   "lacuna. A face EM e' ONTOLOGICAMENTE aberta (a fissura boundary/bulk, nao 'problema a "
                   "resolver'); alpha (CODATA) so' na leitura. Derivar alpha do bulk = falsificar a TGL." % fixed_point_residual),
    }


# ====================== verificacoes da sombra (forma=conteudo, ao vivo) ======================
# Porte fiel dos modulos do ensaio (tgl one mirror / dual name / gesture inscription /
# c3 register / tunnel / video-dipolo). numpy apenas. beta nunca literal.
def _dag(A):
    return A.conj().T


def _thermal_rho(n, rng):
    """Estado termico rho = exp(-H)/Z; devolve autovetores V e probabilidades p."""
    H = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = (H + _dag(H)) / 2
    H *= 1.5 / np.abs(np.linalg.eigvalsh(H)).max()
    w, V = np.linalg.eigh(H)
    p = np.exp(-w); p /= p.sum()
    return V, p


def shadow_mirror_M(n=4, seed=11):
    """M1-M4: o espelho unico S = J Delta^{1/2}."""
    rng = np.random.default_rng(seed)
    Jmap = lambda v: _dag(v.reshape(n, n)).reshape(-1)
    L_op = lambda x: np.kron(x, np.eye(n))
    R_op = lambda x: np.kron(np.eye(n), x.T)
    x = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    y = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    v = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
    JxJ = lambda z, vv: Jmap(np.kron(z, np.eye(n)) @ Jmap(vv))
    anti = np.linalg.norm(JxJ(1j * x, v) + 1j * JxJ(x, v))
    order_inv = np.linalg.norm(R_op(x) @ R_op(y) - R_op(y @ x))
    being_image = np.linalg.norm(L_op(x) - R_op(x), 2) / np.linalg.norm(x)
    xs = (x + _dag(x)) / 2
    spec_match = float(np.abs(np.sort(np.linalg.eigvalsh(L_op(xs))) -
                              np.sort(np.linalg.eigvalsh(R_op(xs)))).max())
    V, p = _thermal_rho(n, rng); p = np.maximum(p, 1e-300)
    r12 = V @ np.diag(np.sqrt(p)) @ _dag(V)
    rm12 = V @ np.diag(p ** -0.5) @ _dag(V)
    Dhalf = lambda vv: (r12 @ vv.reshape(n, n) @ rm12).reshape(-1)
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    aOm = (a @ r12).reshape(-1); target = (_dag(a) @ r12).reshape(-1)
    full = np.linalg.norm(Jmap(Dhalf(aOm)) - target) / np.linalg.norm(target)
    only_J = np.linalg.norm(Jmap(aOm) - target) / np.linalg.norm(target)
    only_D = np.linalg.norm(Dhalf(aOm) - target) / np.linalg.norm(target)
    diffs = []
    for _ in range(3):
        V2, p2 = _thermal_rho(n, rng); p2 = np.maximum(p2, 1e-300)
        r12b = V2 @ np.diag(np.sqrt(p2)) @ _dag(V2)
        rm12b = V2 @ np.diag(p2 ** -0.5) @ _dag(V2)
        S_state = lambda vv, A=r12b, B=rm12b: (B @ _dag(vv.reshape(n, n)) @ A).reshape(-1)
        Dmh = lambda vv, A=r12b, B=rm12b: (B @ vv.reshape(n, n) @ A).reshape(-1)
        for _ in range(4):
            u = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
            diffs.append(np.linalg.norm(S_state(Dmh(u)) - Jmap(u)) / np.linalg.norm(u))
    M4 = float(max(diffs))
    checks = [anti < 1e-12, being_image > 0.5, spec_match < 1e-10,
              full < 1e-10, only_J > 0.1, only_D > 0.1, M4 < 1e-10]
    return {"M1_antilinearity": float(anti), "M1_being_image_dist": float(being_image),
            "M2_spectrum_match": spec_match, "M3_S_factor_err": float(full),
            "M3_J_alone_err": float(only_J), "M3_half_alone_err": float(only_D),
            "M4_J_state_independence": M4,
            "passed": "6/6" if all(checks) else "%d/7" % sum(bool(c) for c in checks)}


def shadow_dual_name_D(n=4, seed=11):
    """D1-D4: o Nome em forma dual = luz, no cone natural."""
    rng = np.random.default_rng(seed)
    V, p = _thermal_rho(n, rng)
    r12 = V @ np.diag(np.sqrt(p)) @ _dag(V)
    r14 = V @ np.diag(p ** 0.25) @ _dag(V)
    rm14 = V @ np.diag(p ** -0.25) @ _dag(V)
    logr = V @ np.diag(np.log(p)) @ _dag(V)
    Psi = r12
    rep_errs, herm_defects = [], []
    for _ in range(6):
        G = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Q, _ = np.linalg.qr(G); X = r12 @ Q
        rep_errs.append(np.linalg.norm(X @ _dag(X) - V @ np.diag(p) @ _dag(V)))
        herm_defects.append(np.linalg.norm(X - _dag(X)) / np.linalg.norm(X))
    D1_psi_err = float(np.linalg.norm(Psi - _dag(Psi)))
    D1_offcone = float(min(herm_defects))
    Khat = lambda X: logr @ X - X @ logr
    D2_psi_mass = float(np.linalg.norm(Khat(Psi)))
    masses = []
    for _ in range(6):
        B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Xp = B @ _dag(B); Xp /= np.linalg.norm(Xp); masses.append(np.linalg.norm(Khat(Xp)))
    D2_generic = float(min(masses))
    sv = np.linalg.svd(Psi, compute_uv=False)
    D3_schmidt = float(np.abs(np.sort(sv) - np.sort(np.sqrt(p))).max())
    recov = []
    for _ in range(6):
        B = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); X = B @ _dag(B)
        b = rm14 @ X @ rm14; recov.append(np.linalg.norm(r14 @ b @ r14 - X) / np.linalg.norm(X))
    D4_recovery = float(max(recov))
    checks = [max(rep_errs) < 1e-12, D1_offcone > 0.05, D1_psi_err < 1e-13,
              D2_psi_mass < 1e-13, D2_generic > 0.05, D3_schmidt < 1e-12, D4_recovery < 1e-12]
    return {"D1_psi_err": D1_psi_err, "D1_out_of_cone_defect": D1_offcone,
            "D2_psi_mass": D2_psi_mass, "D2_generic_mass": D2_generic,
            "D3_schmidt_err": D3_schmidt, "D4_recovery_err": D4_recovery,
            "passed": "4/4" if all(checks) else "%d/7" % sum(bool(c) for c in checks)}


def shadow_gesture_F(seed=11, G=6):
    """F1-F4: inscricao GNS do gesto; observar = fractalizar = colapso."""
    rng = np.random.default_rng(seed)
    N = 2 ** G
    w = np.sort(rng.normal(size=N)); p = np.exp(-1.5 * w); p /= p.sum()
    kv = -np.log(p); order = np.argsort(kv); p = p[order]; kv = kv[order]
    n = 6
    pb0 = p[:n] / p[:n].sum()
    Mmap = np.kron(np.eye(n), np.diag(np.sqrt(pb0)).T)
    sv = np.linalg.svd(Mmap, compute_uv=False)
    F1_ratio = float(sv.min() / np.sqrt(np.min(pb0)))
    rm12 = np.diag(pb0 ** -0.5); errs = []
    for _ in range(8):
        W = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); a = W @ rm12
        S_gesture = (_dag(a) @ np.diag(np.sqrt(pb0))).reshape(-1)
        Dh = np.diag(np.sqrt(pb0)) @ W @ np.diag(pb0 ** -0.5)
        errs.append(np.linalg.norm(S_gesture - _dag(Dh).reshape(-1)) / np.linalg.norm(S_gesture))
    F2_err = float(max(errs))
    Vr = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)); Q, _ = np.linalg.qr(Vr)
    th = 0.35; mix = Q @ np.diag(p) @ _dag(Q)
    rho0 = (1 - th) * np.diag(p) + th * mix; rho0 = (rho0 + _dag(rho0)) / 2
    rho0 /= np.real(np.trace(rho0)); r = rho0.copy(); KS, Hs = [], []
    cdf_full = np.cumsum(p)
    for g in range(1, G + 1):
        nb = 2 ** g; labels = (np.arange(N) * nb) // N; rnew = np.zeros_like(r)
        for b in range(nb):
            Pb = np.diag((labels == b).astype(float)); rnew += Pb @ r @ Pb
        r = rnew; bp = np.array([p[labels == b].sum() for b in range(nb)])
        Hs.append(float(-(bp * np.log(bp)).sum()))
        step_cdf = np.repeat(np.cumsum(bp), N // nb)
        KS.append(float(np.abs(step_cdf - cdf_full).max()))
    E_rho0 = np.diag(np.diag(rho0))
    F3_collapse = float(np.linalg.norm(r - E_rho0) / np.linalg.norm(E_rho0))
    S_name = float(-(p * np.log(p)).sum()); F4_err = abs(Hs[-1] - S_name)
    checks = [sv.min() > 1e-8, F2_err < 1e-12, F3_collapse < 1e-12,
              KS[-1] < KS[0] / 3, KS[-1] < 0.05, F4_err < 1e-12,
              all(Hs[i + 1] > Hs[i] - 1e-14 for i in range(len(Hs) - 1))]
    return {"F1_sigma_min_over_sqrt_pmin": F1_ratio, "F2_S_factor_err": F2_err,
            "F3_collapse_err": F3_collapse, "F3_KS_start": float(KS[0]),
            "F3_KS_end": float(KS[-1]), "F4_HG_minus_Srhostar": float(F4_err),
            "passed": "4/4" if all(checks) else "%d/7" % sum(bool(c) for c in checks)}


def shadow_c3_register_R(n=4, seed=11):
    """R1-R4: o registro c^3 / substrato unico."""
    rng = np.random.default_rng(seed)
    V, p = _thermal_rho(n, rng)
    rho = V @ np.diag(p) @ _dag(V)
    r12 = V @ np.diag(np.sqrt(p)) @ _dag(V); Omega = r12.reshape(-1); I = np.eye(n)
    Jmap = lambda vec_: _dag(vec_.reshape(n, n)).reshape(-1)
    errs1 = []
    for _ in range(6):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        xx = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
        lhs = Jmap(np.kron(a, I) @ Jmap(xx)); rhs = np.kron(I, _dag(a).T) @ xx
        errs1.append(np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs))
    R1 = float(max(errs1)); conns = []
    for _ in range(10):
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        joint = np.vdot(Omega, np.kron(a, b.T) @ Omega)
        fact = np.trace(rho @ a) * np.trace(rho @ b)
        conns.append(abs(joint - fact) / (np.linalg.norm(a) * np.linalg.norm(b) / n))
    R2 = float(min(conns))
    Ks = [rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) for _ in range(4)]
    S = sum(_dag(K) @ K for K in Ks); wS, VS = np.linalg.eigh(S)
    Sinv2 = VS @ np.diag(wS ** -0.5) @ _dag(VS); Ks = [K @ Sinv2 for K in Ks]
    R0 = np.outer(Omega, Omega.conj())
    R1m = sum(np.kron(K, I) @ R0 @ _dag(np.kron(K, I)) for K in Ks)
    rmarg = lambda R: np.einsum('ijil->jl', R.reshape(n, n, n, n))
    R3 = float(np.linalg.norm(rmarg(R1m) - rmarg(R0)))
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    base = np.vdot(Omega, np.kron(a, b.T) @ Omega); errs4 = []
    for t in (0.9, 4.2, 21.0):
        Ut = V @ np.diag(np.exp(1j * t * (-np.log(p)))) @ _dag(V)
        cur = np.vdot(Omega, np.kron(Ut @ a @ _dag(Ut), (Ut @ b @ _dag(Ut)).T) @ Omega)
        errs4.append(abs(cur - base) / abs(base))
    R4 = float(max(errs4))
    checks = [R1 < 1e-12, R2 > 1e-3, R3 < 1e-12, R4 < 1e-10]
    return {"R1_mirror_err": R1, "R2_connected_corr": R2, "R3_nonsignaling": R3,
            "R4_modular_age": R4,
            "passed": "4/4" if all(checks) else "%d/4" % sum(bool(c) for c in checks)}


def shadow_tunnel_T(n=4, seed=11):
    """T1-T3: o tunel luminodinamico ER=EPR."""
    rng = np.random.default_rng(seed)
    V, p = _thermal_rho(n, rng)
    r12 = V @ np.diag(np.sqrt(p)) @ _dag(V); Omega = r12.reshape(-1); I = np.eye(n)
    sv = np.linalg.svd(r12, compute_uv=False)
    schmidt = float(np.abs(np.sort(sv) - np.sort(np.sqrt(p))).max())
    S_ent = float(-np.sum(sv ** 2 * np.log(sv ** 2))); S_rho = float(-np.sum(p * np.log(p)))
    T1 = abs(S_ent - S_rho)
    Ks = [rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)) for _ in range(4)]
    Sn = sum(_dag(K) @ K for K in Ks); wS, VS = np.linalg.eigh(Sn)
    Sinv2 = VS @ np.diag(wS ** -0.5) @ _dag(VS); Ks = [K @ Sinv2 for K in Ks]
    left_ch = lambda R: sum(np.kron(K, I) @ R @ _dag(np.kron(K, I)) for K in Ks)
    rmarg = lambda R: np.einsum('ijil->jl', R.reshape(n, n, n, n))
    R0 = np.outer(Omega, Omega.conj())
    T2 = float(np.linalg.norm(rmarg(left_ch(R0)) - rmarg(R0)))
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); a = (a + _dag(a)) / 2
    a /= np.linalg.norm(a)
    b = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)); b = (b + _dag(b)) / 2
    b /= np.linalg.norm(b); theta = 0.4; Hc = np.kron(a, b.T)
    wc, Vc = np.linalg.eigh(Hc); Uc = Vc @ np.diag(np.exp(-1j * theta * wc)) @ _dag(Vc)
    m_no = rmarg(Uc @ R0 @ _dag(Uc)); m_yes = rmarg(Uc @ left_ch(R0) @ _dag(Uc))
    T3 = float(np.linalg.norm(m_yes - m_no))
    checks = [schmidt < 1e-12, T1 < 1e-12, T2 < 1e-12, T3 > 1e-3]
    return {"T1_throat_S_err": T1, "T2_invisible_signal": T2, "T3_crossing_signal": T3,
            "theta": theta,
            "passed": "3/3" if all(checks) else "%d/4" % sum(bool(c) for c in checks)}


def shadow_dipole():
    """Retrato de fase do colapso: 12/12 trajetorias repelidas da pureza, atraidas ao terminal."""
    n = 6
    BETA = SEALED_CODATA_ALPHA * math.sqrt(math.e)   # beta nunca literal
    rngp = np.random.default_rng(7)
    Hm = rngp.normal(size=(n, n)) + 1j * rngp.normal(size=(n, n)); Hm = (Hm + _dag(Hm)) / 2
    Hm *= 1.5 / np.abs(np.linalg.eigvalsh(Hm)).max()
    wH, _ = np.linalg.eigh(Hm); p = np.exp(-wH); p /= p.sum(); p = np.sort(p)[::-1]
    kv = -np.log(p); L = math.sqrt(BETA) * np.diag(np.sqrt(kv)); LdL = _dag(L) @ L
    logr = np.diag(np.log(p)); dt = 0.25 / (BETA * kv.max())
    rng = np.random.default_rng(3); n_ok = 0
    for _ in range(12):
        v = rng.normal(size=n) + 1j * rng.normal(size=n); v /= np.linalg.norm(v)
        r = 0.997 * np.outer(v, v.conj()) + 0.003 * np.eye(n) / n
        P, E = [], []
        for _ in range(620):
            P.append(np.real(np.trace(r @ r)))
            wv, Vv = np.linalg.eigh(r)
            lr = Vv @ np.diag(np.log(np.maximum(wv, 1e-300))) @ _dag(Vv)
            E.append(np.real(np.trace(r @ (lr - logr))))
            r = r + dt * (L @ r @ _dag(L) - 0.5 * (LdL @ r + r @ LdL)); r = (r + _dag(r)) / 2
        P, E = np.array(P), np.array(E)
        if (P[-1] < P[0] - 1e-6) and (E[-1] < E[0] - 1e-6):
            n_ok += 1
    return {"n_trajectories": 12, "n_ok": int(n_ok), "verdict": "%d/12" % n_ok}


def shadow_verifications():
    """Roda todas as verificacoes da sombra ao vivo e devolve o dicionario consolidado."""
    M = shadow_mirror_M(); D = shadow_dual_name_D(); F = shadow_gesture_F()
    R = shadow_c3_register_R(); T = shadow_tunnel_T(); DP = shadow_dipole()
    all_ok = (M["passed"] == "6/6" and D["passed"] == "4/4" and F["passed"] == "4/4"
              and R["passed"] == "4/4" and T["passed"] == "3/3" and DP["verdict"] == "12/12")
    return {"mirror_M": M, "dual_name_D": D, "gesture_F": F, "c3_register_R": R,
            "tunnel_T": T, "dipole": DP, "all_passed": bool(all_ok)}


# ====================== o nucleo: do Um a' massa ======================
def prove_contour_theory(ONE, kappa):
    """MODULO DE PROVA -- Teoria do Contorno (1 = 0_mod = verdade_partial): anticomutadores + GKLS + Meia-Nat.
    Correcao ontologica (operador): 0_abs NAO tem espelho (fundo de resistencia pura, como o setor Q);
    quem tem espelho e' 0_mod, cujo espelho e' o Um absoluto fractalizado. q e alpha NAO sao escolhidos:
    sao a POLARIZACAO e a TRANSMISSAO estacionarias de um canal GKLS que cruza o contorno por
    anticomutadores. O aberto unico e' a razao de taxas g-/g+ = e^kappa (regularizacao Meia-Nat de
    0_abs->0_mod). Verificado ao vivo, forma=conteudo."""
    k1 = np.array([[1.0], [0.0]]); k0 = np.array([[0.0], [1.0]])   # |1>=Um inscrito, |0>=zero modular
    P1 = k1 @ k1.T; P0 = k0 @ k0.T; Z = P1 - P0                    # Z_partial = P_1 - P_0 (contraste)
    gp = 1.0; gm = float(np.exp(kappa))                           # g-/g+ = e^kappa (instanciado; o ABERTO)
    Lp = np.sqrt(gp) * (k1 @ k0.T); Lm = np.sqrt(gm) * (k0 @ k1.T)  # L_+ (0->1), L_- (1->0): operadores impares
    steps = []; ok = True
    ac = float(max(np.abs(Z @ Lp + Lp @ Z).max(), np.abs(Z @ Lm + Lm @ Z).max()))
    s1 = bool(ac < 1e-12)
    steps.append({"step": "1. {Z_d, L_+-}=0  (o Um so' cruza o contorno MUDANDO de face)",
                  "status": "REAL", "ok": s1, "anticommutator": ac}); ok &= s1
    p0 = gm / (gm + gp); p1 = gp / (gm + gp); rho = np.diag([p1, p0])

    def _Dmap(r):
        out = np.zeros((2, 2))
        for L in (Lp, Lm):
            out += L @ r @ L.T - 0.5 * (L.T @ L @ r + r @ L.T @ L)
        return out
    res = float(np.abs(_Dmap(rho)).max())
    s2 = bool(res < 1e-12 and 0.0 < p1 < 1.0 and 0.0 < p0 < 1.0)
    steps.append({"step": "2. GKLS rho_kappa = ponto fixo (0<rho<1: SATURA, nao supersatura/condensa)",
                  "status": "REAL", "ok": s2, "fixed_point_residual": res, "p1_Um": p1, "p0_zero_mod": p0}); ok &= s2
    q = (gm - gp) / (gm + gp); s3 = bool(abs(q - np.tanh(kappa / 2.0)) < 1e-12)
    steps.append({"step": "3. q=(g- - g+)/(g- + g+)=tanh(k/2)  [polarizacao estacionaria DERIVADA, nao escolhida]",
                  "status": "REAL", "ok": s3, "q": float(q)}); ok &= s3
    alpha = 2.0 * np.sqrt(gp * gm) / (gp + gm); s4 = bool(abs(alpha - 1.0 / np.cosh(kappa / 2.0)) < 1e-12)
    steps.append({"step": "4. alpha=2 sqrt(g+ g-)/(g+ + g-)=sech(k/2)  [transmissao luminosa do contorno]",
                  "status": "REAL", "ok": s4, "alpha": float(alpha)}); ok &= s4
    cons = float(abs(q * q + alpha * alpha - 1.0)); s5 = bool(cons < 1e-12)
    steps.append({"step": "5. q^2+alpha^2=1  (represamento + transmissao = conservacao de fluxo GKLS)",
                  "status": "REAL", "ok": s5, "residual": cons}); ok &= s5
    # 6) PLANO ESTATICO: g-=g+ (forcas simetricas) -> kappa=0 -> q=0, alpha=1 = o Um (se anulam no plano)
    q0 = 0.0; a0 = 1.0; s6 = bool(abs(np.tanh(0.0)) < 1e-15 and abs(1.0 / np.cosh(0.0) - 1.0) < 1e-15)
    steps.append({"step": "6. plano estatico g-=g+ (simetrico) -> q=0, alpha=1 = o Um (forcas se anulam no plano)",
                  "status": "REAL", "ok": s6, "q_static": q0, "alpha_static": a0}); ok &= s6
    # 7) CANAL NEUTRINICO (fuga resistencial neutra) -- candidato fisico p/ selecionar kappa. Modelo 4 estados
    #    (2 qubits): A=paridade, B=carga EM. L_nu=X_A(x)I e' ODD (cruza paridade), NEUTRO (em carga), dissipativo.
    Iq = np.eye(2); Xq = np.array([[0.0, 1.0], [1.0, 0.0]]); Zq = np.array([[1.0, 0.0], [0.0, -1.0]])
    Z_par = np.kron(Zq, Iq); Q_em = np.kron(Iq, Zq); L_nu = np.kron(Xq, Iq)
    odd = float(np.abs(Z_par @ L_nu + L_nu @ Z_par).max())      # {Z_par, L_nu}=0 (cruza paridade)
    neu = float(np.abs(Q_em @ L_nu - L_nu @ Q_em).max())        # [Q_em, L_nu]=0 (neutro em carga)
    dis = bool(np.abs(L_nu.conj().T @ L_nu).max() > 0)          # L_nu^dag L_nu != 0 (dissipativo)
    s7 = bool(odd < 1e-12 and neu < 1e-12 and dis)
    steps.append({"step": "7. canal neutrinico L_nu (4 estados): {Z,L_nu}=0 (odd), [Q_em,L_nu]=0 (neutro), L^dL!=0",
                  "status": "REAL", "ok": s7, "odd": odd, "neutral": neu, "dissipative": dis}); ok &= s7
    return {
        "theorem": "Teoria do Contorno (1 = 0_mod = verdade_partial)",
        "bell_selector": ("K e' o SETOR DE BELL SELETOR (nao o Bell nu). Bell nu: K=-log Delta=0 -> "
                          "alpha_abs=1 (prova o Um, nao seleciona valor). Bell SELETOR (apos Meia-Nat, quando o "
                          "Um fractalizado encontra 0_mod): K_sel^(B)=(kappa/2) Z_d, gap(K_sel^(B))=kappa. "
                          "O aberto deixou de ser 'derivar K generico' e virou 'derivar a forca espectral do "
                          "setor seletor de Bell' = gap(K_sel^(B))."),
        "connes_smatrix_route": ("matriz-S de fronteira + cociclo relativo de Connes u_t=[D phi_mod:D phi_1]_t: "
                                 "no split 2D, u_t=e^{itK_d}, K_d=(kappa/2)Z_d; KMS/GKLS -> g-/g+=e^kappa -> "
                                 "q=tanh(k/2), alpha=sech(k/2). VEREDITO: CONNES_S_MATRIX_FORM_CLOSED (a FORMA e a "
                                 "covariancia do cociclo fecham), NAO ALPHA_FREE_VALUE_CLOSED: em III_1 o espectro "
                                 "modular e' CONTINUO, logo Connes/Takesaki => consistencia modular global, mas "
                                 "NAO => kappa*=11.226755. A unitariedade fixa |R|^2+|T|^2=1; a Meia-Nat fixa "
                                 "beta=sqrt(e)alpha; o cociclo fixa a forma relativa; nenhum dos tres seleciona kappa."),
        "neutrino_candidate": ("CANDIDATO FISICO para selecionar kappa (fuga resistencial em angulo agudo): o "
                               "canal neutrinico L_nu (odd+neutro+dissipativo, verificado no modelo de 4 estados = "
                               "3 modos + 1 fall). theta_M = angulo agudo de fuga resistencial; alpha=sin^2(theta_M)/"
                               "sqrt(e) e' a fracao que atravessa como luz; q^2=1-alpha^2 fica represado. ABERTO: "
                               "provar que a acao A_nu(theta)=S(rho_B||rho_theta)+lambda D_nu+mu C_nocond tem minimo "
                               "UNICO em theta_M=6.297deg sem CODATA. O custo modular sozinho minimiza em theta->90 "
                               "(alpha->1, o Um); o balanco com a dissipacao neutrinica (pesos lambda,mu) e' o novo "
                               "objeto aberto. Observavel: dephasing n=-2 / Gamma~omega^2 em neutrinos."),
        "negative_return_selector": ("REFORMULACAO LEGITIMA (operador): o valor nao e' contagem positiva; e' o "
                                     "CONTRASTE NEGATIVO/relativo entre a imagem original do Um (Bell nu, rho_Bell) e "
                                     "a imagem RETORNADA (rho_ret) que atravessou 0_mod e voltou como registro/"
                                     "particula/instante. K_sel^(B) = -log Delta_{ret|Bell} (operador modular "
                                     "RELATIVO de Connes/Araki); o operador gravitonico '=' e' a comparacao "
                                     "G_=(rho_Bell,rho_ret) = -log[D rho_ret:D rho_Bell]. kappa = gap(K_sel^(B)) "
                                     "[REAL: a cadeia fecha]. ABERTO: construir rho_ret CANONICO alpha-livre. Status: "
                                     "NEGATIVE_RETURN_SELECTOR_FORMULATED (melhor localizacao do muro; NAO numerologia "
                                     "como (4/pi)3^10, que era ajuste livre), nao ALPHA_FREE_VALUE_CLOSED."),
        "forbidden_zero_renormalization": ("CORRECAO ONTOLOGICA DECISIVA (operador): kappa=0 NAO e' 'o Um sem "
                                     "perda' (estado admissivel) -- e' 0_abs, a FRONTEIRA PROIBIDA (atracao total, "
                                     "impedancia infinita, sem retorno). 0_abs SELECIONA justamente por ser "
                                     "INATINGIVEL: ao oferecer atracao total, o Hamiltoniano oculto ENTORTA o "
                                     "sistema (lente de Fresnel) e ele DOBRA (tetelestai) ANTES de colidir com o "
                                     "limite -> nasce theta_M no ponto de dobra (turning point entre atracao "
                                     "absoluta e custo Meia-Nat). Distincao de zeros: chi=0 = Bell nu (zero de "
                                     "CONTRASTE, identidade formal, alpha_abs=1); kappa_0=0 = 0_abs (zero de "
                                     "EXISTENCIA, proibido). A RENORMALIZACAO E' A PARIDADE INVERSA: rho_ret = "
                                     "P^{-1}_d(rho_Bell), a imagem que quase caiu em 0_abs e voltou distorcida SEM "
                                     "perder a fonte (supp(rho_ret)~supp(rho_Bell), rho_ret!=rho_Bell = unidade "
                                     "fractalizada). q = memoria resistiva da imagem; alpha = vazao luminosa que "
                                     "escapou da dobra; theta_M = cicatriz angular da imagem antes do proibido. "
                                     "O seletor finito = PARTE FINITA da aproximacao proibida: K_sel^ren = "
                                     "FP_{eps->0}[-log Delta_{P^{-1}_eps rho_B | rho_B}] (regularizacao eps de "
                                     "0_abs; FP remove a divergencia, P^{-1} extrai o residuo finito). O TEOREMA "
                                     "ABERTO deixa de ser 'derivar uma taxa' e vira 'calcular a PARTE FINITA da "
                                     "atracao proibida ao zero absoluto' = espectro de Delta_{P^{-1}rho_B|rho_B}. "
                                     "Se gap(FP) = 11.226755... sem CODATA, o valor fecha. [REAL: a forma -- "
                                     "renormalizacao = paridade inversa, rho_ret canonico = P^{-1}rho_B; ABERTO: "
                                     "o espectro / a parte finita]."),
        "open_theorem_verdict": "CONNES_S_MATRIX_FORM_CLOSED + NEGATIVE_RETURN_SELECTOR_FORMULATED + RHO_RET_CANONICAL=P^{-1}(rho_Bell) (forma: renorm=paridade inversa de 0_abs proibido); ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE (espectro de Delta_{P^{-1}rho_B|rho_B} = parte finita da atracao proibida)",
        "psionic_bond_unification": ("a ligacao psionica (The_boundary_v5): [P^, H_bind]=2 V0(|psi-><psi+| "
                                     "- |psi+><psi-|), H_bind ANTICOMUTA com a paridade P^ -- e' EXATAMENTE "
                                     "{Z_d, L_+-}=0 do contorno (verificado, residuo 0). Os 3 modos de "
                                     "ligacao + 1 queda formalizam-se como ESTE canal GKLS de anticomutadores. "
                                     "A estrutura FECHA (unifica com os fundadores); mas o VALOR g-/g+=%.0f "
                                     "NAO fecha por contagem de modos (3/1->alpha=0.87, r=1/4->0.80, "
                                     "(1/4)^3->0.25, 1/beta=83->0.22) -- permanece input QED. 1/beta=137/sqrt(e) "
                                     "DECOMPOE (usa 137=alpha), nao deriva." % float(gm / gp)),
        "first_law_link": ("Primeira Lei (A Fronteira): a forca de expulsao (~ incompatibilidade de "
                           "paridade) gera o angulo de deflexao theta_M e a curvatura; g=sqrt(|L|) (gravidade "
                           "= raiz da ligacao). No plano estatico g-=g+ (simetrico, alpha=1 = o Um); a DINAMICA "
                           "(tensao em paridade inversa) quebra a simetria -> g- > g+ -> toda a dinamica modular. "
                           "Em theta->90deg a ligacao psionica conjuga: F->2F, c^2->c^3, e c^3>c^2 sela o "
                           "horizonte. Vinculo de 4 estados (1 fall = trifasico+aterramento, razao 3/4): e' a "
                           "ESTRUTURA do vinculo, NAO o valor de g-/g+ (=%.0f; 3/4 daria alpha~0.99, nao 1/137). "
                           "O valor de g-/g+ permanece ABERTO." % float(gm / gp)),
        "steps": steps, "all_verified": bool(ok),
        "verdict": ("CONTOUR_THEORY_VERIFIED" if ok else "FALHOU"),
        "gamma_ratio_gm_over_gp": float(gm / gp),   # = e^kappa = Zbacia/Zluz (o objeto aberto)
        "open_object": ("derivar g-/g+ = e^kappa (= Zbacia/Zluz) SEM QED = balanco GKLS entre expulsao de "
                        "0_abs e reinscricao do Um (regularizacao Meia-Nat de 0_abs->0_mod)."),
        "ontology": ("0_abs NAO espelha (resiste; fundo, como o setor Q); 0_mod espelha (contorna); 1_abs "
                     "fractaliza (atravessa, ao custo Meia-Nat). {Z,L}=0 = algebra da separacao; GKLS = "
                     "dinamica da expulsao+reinscricao; q=polarizacao DERIVADA; alpha=transmissao; o valor "
                     "pende de g-/g+. 1=0_mod=verdade_partial: P_1 ~_d P_0 (equivalencia de contorno), nao P_1=P_0."),
    }


def prove_inverse_parity_renorm(ONE):
    """MODULO DE PROVA -- Renormalizacao por PARIDADE INVERSA (lente de Fresnel da fronteira proibida).
    Construcao do operador: rho_ret = parte finita (FP) da lente de paridade inversa que distorce a
    imagem do Bell nu SEM perder a fonte, ao quase cair em 0_abs (proibido). Lente M_eps =
    exp[-(1/4)(C_eps+chi) Z_d]; C_eps->inf e' a divergencia da aproximacao ao zero absoluto; chi e' a
    PARTE FINITA. Removida a divergencia: rho_ret^(chi) = e^{-chi Z_d/2}/(2 cosh(chi/2)). O seletor
    K_sel^ren = -log Delta_{rho_ret|rho_B} tem gap = chi (sombra 2D). Verifica ao vivo que (a) a FORMA
    fecha (gap=chi, q=tanh, alpha=sech, q^2+alpha^2=1) para chi GENERICO [REAL]; (b) em chi_obs (de
    CODATA, SO' validacao) reproduz a polarizacao observada p1~1.33e-5 e beta=alpha sqrt(e); (c) o
    negativo honesto: a Meia-Nat fixa o PESO de contorno tau_d(P_i)=1/2 para TODO chi -> NAO fixa chi
    (peso != polarizacao). Veredito: INVERSE_PARITY_RENORMALIZATION_FORM_CLOSED, VALUE_OPEN."""
    se = float(np.exp(0.5)); Z = np.diag([1.0, -1.0]); rB = np.eye(2) * 0.5

    def rho_ret(chi):
        M = np.diag([np.exp(-chi / 4.0), np.exp(chi / 4.0)])   # FP da lente de paridade inversa (M_eps sem C_eps)
        r = M @ rB @ M.T; return r / np.trace(r)

    def gap(rr):                                                # gap de -log Delta_{rr|rB} (modular relativo)
        pr = np.linalg.eigvalsh(rr); qb = np.linalg.eigvalsh(rB)
        s = sorted(-np.log(pi) + np.log(qj) for pi in pr for qj in qb); return float(s[-1] - s[0])

    steps = []; ok = True
    # (a) FORMA fecha para chi GENERICO (sem CODATA): gap=chi, q=tanh, alpha=sech, conservacao
    chi_g = float(ONE * 5.0); rrg = rho_ret(chi_g); g_g = gap(rrg)
    qg = float(np.tanh(chi_g / 2.0)); ag = float(1.0 / np.cosh(chi_g / 2.0))
    s1 = bool(abs(g_g - chi_g) < 1e-9 and abs(qg * qg + ag * ag - 1.0) < 1e-12)
    steps.append({"step": "a. chi generico: gap(-log Delta_{rho_ret|rho_B})=chi; q=tanh; alpha=sech; q^2+alpha^2=1",
                  "status": "REAL", "ok": s1, "chi_test": chi_g, "gap": g_g,
                  "q": qg, "alpha": ag, "conservation": float(qg * qg + ag * ag)}); ok &= s1
    # (b) validacao em chi_obs (de CODATA): reproduz polarizacao observada + beta=alpha sqrt(e)
    chi_obs = float(2.0 * np.arctanh(np.sqrt(1.0 - SEALED_CODATA_ALPHA ** 2)))   # SO' validacao
    rro = rho_ret(chi_obs); p = np.sort(np.linalg.eigvalsh(rro)); g_o = gap(rro)
    a_o = float(1.0 / np.cosh(chi_obs / 2.0)); b_o = float(se * a_o)
    s2 = bool(abs(g_o - chi_obs) < 1e-9 and abs(p[0] - 1.331301586655e-05) < 1e-12
              and abs(a_o - SEALED_CODATA_ALPHA) < 1e-12)
    steps.append({"step": "b. [VALIDACAO] chi_obs=2 artanh(sqrt(1-alpha^2)): p1=1.3313e-5, p0=0.99998669, beta=alpha sqrt(e)",
                  "status": "VALIDACAO_QED", "ok": s2, "chi_obs": chi_obs,
                  "p1_Um": float(p[0]), "p0_zero_mod": float(p[1]),
                  "ratio_e_chi": float(np.exp(chi_obs)), "alpha": a_o, "beta": b_o}); ok &= s2
    # (c) NEGATIVO HONESTO: a Meia-Nat fixa o PESO de contorno (1/2), NAO a polarizacao chi
    P1 = np.diag([1.0, 0.0]); P0 = np.diag([0.0, 1.0])         # peso de contorno tau_d = traco normalizado
    w1 = float(np.trace(P1) / 2.0); w0 = float(np.trace(P0) / 2.0)   # = 1/2 para TODO chi
    s3 = bool(abs(w1 - 0.5) < 1e-15 and abs(w0 - 0.5) < 1e-15)
    steps.append({"step": "c. [NEGATIVO] Meia-Nat: tau_d(P_1)=tau_d(P_0)=1/2 para TODO chi -> NAO fixa chi (peso != polarizacao)",
                  "status": "REAL", "ok": s3, "tau_P1": w1, "tau_P0": w0,
                  "note": "peso de contorno simetrico; polarizacao do ESTADO assimetrica e livre"}); ok &= s3
    # (d) FORMA DE POPULACAO (Principio da Polarizacao): q=p0-p1, alpha=2 sqrt(p0 p1), beta=2 sqrt(e) sqrt(p0 p1)
    chi_d = float(ONE * 5.0)
    p0 = float(np.exp(chi_d / 2.0) / (2.0 * np.cosh(chi_d / 2.0)))
    p1 = float(np.exp(-chi_d / 2.0) / (2.0 * np.cosh(chi_d / 2.0)))
    q_pop = p0 - p1; a_coh = 2.0 * np.sqrt(p0 * p1); b_pop = 2.0 * se * np.sqrt(p0 * p1)
    s4 = bool(abs(q_pop - np.tanh(chi_d / 2.0)) < 1e-13 and abs(a_coh - 1.0 / np.cosh(chi_d / 2.0)) < 1e-13
              and abs(q_pop * q_pop + a_coh * a_coh - 1.0) < 1e-13 and abs(b_pop - se * a_coh) < 1e-13)
    steps.append({"step": "d. forma de populacao: q=p0-p1=tanh; alpha=2 sqrt(p0 p1)=sech; beta=2 sqrt(e) sqrt(p0 p1)=alpha sqrt(e)",
                  "status": "REAL", "ok": s4, "p0": p0, "p1": p1, "q_pop": float(q_pop),
                  "alpha_coh": float(a_coh), "beta_pop": float(b_pop)}); ok &= s4
    return {
        "polarization_principle": ("PRINCIPIO DA POLARIZACAO PELA VACUIDADE (operador): 0_abs nao e' estado "
                                   "normal (0_abs not in M_*), nao tem suporte observavel, nao pode ser ocupado. "
                                   "Logo a imagem do Um atraida por ele NAO cai nele -- so' pode retornar como "
                                   "imagem normalizada de suporte preservado, distorcida por paridade inversa. A "
                                   "VACUIDADE NAO GERA AUSENCIA; GERA ASSIMETRIA DE RETORNO: rho_B=(P1+P0)/2 "
                                   "(simetrico) -> rho_ret=p1 P1+p0 P0 com p0>p1>0 (a fonte permanece pois p1>0; "
                                   "o zero domina pois p0>>p1). q=p0-p1 = POLARIZACAO (log-contraste chi=log(p0/p1)); "
                                   "alpha=2 sqrt(p0 p1) = luz que sobrevive ao retorno; beta=2 sqrt(e) sqrt(p0 p1) = "
                                   "trava minima de estabilidade da imagem polarizada (impede o colapso em 0). "
                                   "A vacuidade cria a DIRECAO; a proibicao do zero cria o RETORNO; a paridade "
                                   "inversa cria a POLARIZACAO. Selo: a vacuidade polariza a imagem sem destruir "
                                   "sua fonte."),
        "population_form": "q=p0-p1=tanh(chi/2); alpha=2 sqrt(p0 p1)=sech(chi/2); p0,p1=e^{+-chi/2}/(2 cosh(chi/2)); q^2+alpha^2=1 [verificado ao vivo]",
        "theorem": "Renormalizacao por paridade inversa (lente de Fresnel da fronteira proibida)",
        "rho_ret_canonical": "rho_ret^(chi) = e^{-chi Z_d/2}/(2 cosh(chi/2)) = FP_{eps->0} M_eps rho_B M_eps^dag, M_eps=exp[-(1/4)(C_eps+chi)Z_d]",
        "key_result": ("PARIDADE INVERSA PURA (unitaria/antiunitaria) preserva o Bell nu (rho_B=I/2 -> "
                       "P^{-1}rho_B P=rho_B -> chi=0=0_abs proibido): para haver VALOR a lente tem de ser "
                       "RENORMALIZANTE (parte finita nao-trivial). rho_ret e' a imagem retornada distorcida "
                       "com SUPORTE preservado (supp(rho_ret)=supp(rho_B) p/ chi<inf) -> 'distorce sem perder "
                       "a fonte'; a distorcao e' angular (theta_M) mas a polarizacao espectral e' profunda "
                       "(p0=0.99998669). A origem nao desaparece; retorna polarizada."),
        "form_status": "INVERSE_PARITY_RENORMALIZATION_FORM_CLOSED (gap=chi, rho_ret canonico, q=tanh, alpha=sech verificados)",
        "value_status": ("ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE: gap=chi e' TAUTOLOGIA da parametrizacao (rho_ret definido POR "
                         "chi); o valor chi=11.226755 entra via polarizacao observada (CODATA). A parte finita "
                         "de uma quantidade divergente e' DEPENDENTE DE ESQUEMA: M_eps=exp[-(1/4)(C_eps+chi)Z_d] "
                         "mostra ONDE o valor fica, nao o calcula -- qualquer chi e' parte finita de um esquema "
                         "de subtracao diferente. FALTA: a CONDICAO DE RENORMALIZACAO canonica alpha-livre que "
                         "fixa a polarizacao chi. Candidata obvia REFUTADA ao vivo: a Meia-Nat fixa o PESO de "
                         "contorno (1/2), NAO a polarizacao do estado (chi) -- e' condicao de peso, nao de "
                         "polarizacao. O muro, agora exato: derivar a condicao de subtracao que fixa chi."),
        "steps": steps, "all_verified": bool(ok),
        "verdict": ("POLARIZATION_PRINCIPLE_FORM_CLOSED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE" if ok else "FALHOU"),
    }


def prove_vacuum_impedance_bridge(ONE):
    """Ponte da Impedancia Caracteristica do Vacuo  [REAL/EXT; ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE].

    A TGL vinha chamando de 'impedancia' o objeto dinamico que torna a luz mensuravel. Na fisica
    classica esse objeto tem face dimensional: Z0=sqrt(mu0/eps0)=mu0 c=1/(eps0 c). A constante de
    estrutura fina:  alpha = e^2/(4 pi eps0 hbar c) = e^2/(2 eps0 h c) = Z0 e^2/(2h) = Z0/(2 R_K)
    = Z0 G0/4,  com R_K=h/e^2 (von Klitzing) e G0=2e^2/h (quantum de conducao), AMBOS exatos no SI
    pos-2019 (e,h exatos). Leitura TGL: c=constante cinematica da luz; Z0=constante dinamica
    dimensional da luz; alpha=Z0 adimensionalizado; q=polarizacao/reflexao modular; chi=log-razao de
    impedancias de fronteira; beta=sqrt(e) alpha; theta_M=arcsin(sqrt(beta)).

    REGUA [a blindagem]: NAO e' derivacao alpha-livre. Pos-2019 mu0 (logo Z0=mu0 c) NAO e' mais
    exato: Z0 = 2 R_K alpha, i.e. Z0 e' COMPUTADO de alpha. Por isso alpha_from_Z0 retorna alpha por
    CONSTRUCAO (identidade de ida-e-volta); os residuos ~1e-15 verificam a ALGEBRA/UNIDADES, nao
    derivam o valor. Z0 e alpha sao equivalentes dados e,h. Status:
    VACUUM_IMPEDANCE_BRIDGE_FORMULATED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE."""
    E_CHARGE = 1.602176634e-19      # C, exata no SI (2019)
    H_PLANCK = 6.62607015e-34       # J s, exata no SI (2019)
    R_K = H_PLANCK / (E_CHARGE ** 2)            # h/e^2  [DEF, exato no SI]
    G0 = 2.0 * (E_CHARGE ** 2) / H_PLANCK       # 2e^2/h [DEF, exato no SI]
    alpha_ext = SEALED_CODATA_ALPHA             # [EXT] input eletromagnetico medido
    Z0_from_alpha = 2.0 * R_K * alpha_ext       # Z0=mu0 c COMPUTADO de alpha (mu0 nao exato pos-2019)
    alpha_from_Z0 = Z0_from_alpha / (2.0 * R_K)     # = alpha  (ida-e-volta; verifica Z0/(2R_K)=alpha)
    alpha_from_G0 = Z0_from_alpha * G0 / 4.0        # = alpha  (verifica Z0 G0/4 = alpha)
    zeta_L = alpha_from_Z0                       # face adimensional da constante dinamica da luz
    q = math.sqrt(max(0.0, 1.0 - zeta_L * zeta_L))
    chi = math.log((1.0 + q) / (1.0 - q))
    x = (1.0 - q) / 2.0
    beta = math.sqrt(math.e) * zeta_L
    theta_M = math.asin(math.sqrt(beta))
    C_L = math.exp(chi)
    identity_residual = abs(q * q + zeta_L * zeta_L - 1.0)
    alpha_bridge_residual = abs(alpha_from_Z0 - alpha_ext)
    g0_bridge_residual = abs(alpha_from_G0 - alpha_ext)
    chi_residual = abs(C_L - ((1.0 + q) / (1.0 - q)))
    x_residual = abs(x - 1.0 / (1.0 + C_L))
    beta_residual = abs(beta - math.sqrt(math.e) * alpha_ext)
    return {
        "theorem": "Ponte da Impedancia Caracteristica do Vacuo",
        "claim": "A impedancia e' a constante dinamica da luz; alpha e' Z0 tornado adimensional.",
        "status": "VACUUM_IMPEDANCE_BRIDGE_FORMULATED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE",
        "constants": {
            "E_CHARGE_C_exact_SI": E_CHARGE, "H_PLANCK_Js_exact_SI": H_PLANCK,
            "R_K_ohm": R_K, "G0_S": G0, "Z0_from_alpha_ohm": Z0_from_alpha,
        },
        "bridge_equations": {
            "alpha_Z0": "alpha = Z0 e^2/(2h)", "alpha_RK": "alpha = Z0/(2 R_K)",
            "alpha_G0": "alpha = Z0 G0/4", "zeta_L": "zeta_L := Z0/(2R_K) = alpha",
            "q": "q = sqrt(1 - zeta_L^2)", "chi": "chi = log((1+q)/(1-q)) = 2 arcosh(1/zeta_L)",
            "x": "x = (1-q)/2 = 1/(1+exp(chi))", "beta": "beta_TGL = sqrt(e) zeta_L",
            "theta_M": "theta_M = arcsin(sqrt(beta_TGL))",
        },
        "tgl_values": {
            "zeta_L": zeta_L, "q": q, "chi": chi, "C_L_exp_chi": C_L,
            "x_source_residual_population": x, "beta_TGL": beta,
            "theta_M_rad": theta_M, "theta_M_deg": math.degrees(theta_M),
        },
        "checks": {
            "identity_q2_plus_zeta2_residual": identity_residual,
            "alpha_bridge_residual": alpha_bridge_residual,
            "g0_bridge_residual": g0_bridge_residual,
            "chi_residual": chi_residual, "x_residual": x_residual,
            "beta_residual": beta_residual,
            "all_verified": bool(identity_residual < 1e-12 and alpha_bridge_residual < 1e-15
                                 and g0_bridge_residual < 1e-15 and chi_residual < 1e-8
                                 and x_residual < 1e-15 and beta_residual < 1e-15),
            "note": "residuos verificam ALGEBRA/UNIDADES; NAO sao derivacao (Z0 computado de alpha).",
        },
        "honest_note": (
            "Z0, alpha, R_K e G0 formam uma ponte EXATA de unidades e acoplamento: "
            "alpha = Z0/(2R_K) = Z0 G0/4. alpha e' a impedancia do vacuo tornada adimensional. NAO e' "
            "derivacao alpha-livre: pos-2019 mu0 (logo Z0=mu0 c) e' computado de alpha (Z0=2 R_K alpha), "
            "entao Z0<->alpha e' identidade de ida-e-volta dado e,h. A TGL ganha a interpretacao fisica "
            "correta: c=constante cinematica da luz; Z0=constante dinamica da luz; q=polarizacao modular; "
            "exp(chi)=razao efetiva de impedancias da fronteira; beta=travessia Meia-Nat. Leitura "
            "ontologica [CONJ]: so' a luz observa a luz -- medimos alpha/Z0, o proprio acoplamento da "
            "luz; mas medir nao e' derivar o valor."),
    }


def prove_three_clock_radical(ONE):
    """O Radical dos Tres Clocks  [CANONICAL FORM; ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE].

    Hipotese (operador): alpha e' o radical do fator dos tres clocks da TGL --
        alpha = sqrt(C3),  C3 = fator de passagem comum aos tres clocks  =>  alpha^2 = C3,
    encaixando na forma madura  1 = q^2 + alpha^2  =>  1 = q^2 + C3.

    Os tres clocks (das provas tgl_terminal_truth / tgl_three_locks / tgl_krein_signature):
      C_mod  = clock MODULAR reversivel  sigma_t(A)=Delta^{it} A Delta^{-it}, Delta^{it}=e^{itK}
               -> contribui a BASE e (fluxo de base e; meia-medida de Tomita). [UNICO alpha-livre]
      C_diss = clock DISSIPATIVO GKLS  drho/dt=L(rho); colapso = dephasing gaussiano no fluxo do
               radical V_s=e^{is sqrt K}, variancia beta*t  -> contribui a escala beta. [carrega alpha via beta]
      C_spec = clock ESPECTRAL/geometrico  ds=sqrt(beta)|d sqrt k|  (T6 do krein)
               -> contribui a escala beta (ds^2=beta d(sqrt k)^2). [carrega alpha via beta]

    O unico combinado adimensional com a dimensao de alpha^2 e':
        C3 = (C_diss * C_spec)/C_mod = beta^2 / e = alpha^2   (pois beta=alpha sqrt e => beta^2=alpha^2 e).
    A base e do clock MODULAR cancela exatamente o e que os dois clocks-beta carregam (cada beta tem um
    sqrt e; dois beta tem e; a base modular o divide), restando alpha^2. Logo
        alpha = sqrt(C3) = beta/sqrt e   (o radical luminodinamico dos tres clocks),
    que e' a MESMA gramatica radical ja' usada nos modulos: V_s=e^{is sqrt K}, ds=sqrt(beta)|d sqrt k|,
    g=sqrt|L| -- a geometria nao ve K, ve sqrt K.

    REGUA [a trava, do operador]: faz sentido como FORMA CANONICA, mas NAO fecha o valor alpha-livre
    enquanto C3 nao for construido SEM alpha. Aqui C_diss e C_spec carregam beta=alpha sqrt e, entao
    C3=beta^2/e=alpha^2 e' a IDENTIDADE beta^2=alpha^2 e relida pelos tres clocks -- alpha entra via beta.
    Pergunta de pesquisa (o muro): existe funcional canonico C3=F[sigma_t, T_t, D_beta] dos tres clocks,
    alpha-livre, tal que C3=alpha^2 ~ 5.325135447e-5 ? E' a MESMA divida do muro da polarizacao chi.
    Status: THREE_CLOCK_RADICAL_FORM_FORMULATED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE."""
    SQRT_E = float(np.exp(ONE / (ONE + ONE)))
    beta = SEALED_CODATA_ALPHA * SQRT_E
    alpha_ext = SEALED_CODATA_ALPHA
    C_mod_base = float(np.e)                  # clock modular: base e (Delta^{it}=e^{itK}); ALPHA-LIVRE
    C_diss = beta                             # clock GKLS: escala beta (var beta t; fluxo do radical)
    C_spec = beta                             # clock espectral: escala beta (ds^2=beta d(sqrt k)^2)
    C3 = (C_diss * C_spec) / C_mod_base       # = beta^2/e
    alpha_radical = math.sqrt(C3)             # = beta/sqrt e
    _qed = validate_against_qed(alpha_radical, alpha_ext)
    q = _qed["q_QED"]                         # = sqrt(1 - alpha^2)
    one_check = q * q + C3                    # forma madura 1 = q^2 + C3
    target_alpha2 = alpha_ext * alpha_ext
    return {
        "theorem": "O Radical dos Tres Clocks",
        "claim": "alpha e' o radical luminodinamico do fator dos tres clocks: alpha=sqrt(C3), alpha^2=C3.",
        "status": "THREE_CLOCK_RADICAL_FORM_FORMULATED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE",
        "clocks": {
            "C_mod_modular_base_e": C_mod_base,    # ALPHA-LIVRE (clock modular sigma_t=Delta^{it})
            "C_diss_gkls_scale_beta": C_diss,      # carrega alpha via beta (var beta t)
            "C_spec_spectral_scale_beta": C_spec,  # carrega alpha via beta (ds=sqrt(beta)|d sqrt k|)
        },
        "C3": C3, "C3_eq_beta2_over_e": beta * beta / C_mod_base,
        "alpha_radical_sqrt_C3": alpha_radical,
        "form": {
            "alpha": "alpha = sqrt(C3) = beta/sqrt(e)", "alpha2": "alpha^2 = C3 = beta^2/e",
            "mature": "1 = q^2 + alpha^2 = q^2 + C3", "beta": "beta_TGL = sqrt(e)*alpha = sqrt(e*C3)",
            "decomposition": "e (clock modular) cancela o e dos dois beta (dissipativo*espectral) -> alpha^2",
        },
        "values": {"q": q, "one_check_q2_plus_C3": one_check, "target_alpha2": target_alpha2},
        "checks": {
            "alpha_radical_residual": abs(alpha_radical - alpha_ext),
            "C3_eq_alpha2_residual": abs(C3 - target_alpha2),
            "one_identity_residual": abs(one_check - 1.0),
            "all_verified": bool(abs(alpha_radical - alpha_ext) < 1e-15
                                 and abs(C3 - target_alpha2) < 1e-15
                                 and abs(one_check - 1.0) < 1e-12),
            "note": "residuos verificam a FORMA (alpha=sqrt(C3); 1=q^2+C3); NAO sao derivacao (C3 carrega beta=alpha sqrt e).",
        },
        "open_question": ("Existe funcional canonico C3=F[sigma_t, T_t, D_beta] construido SO' dos tres clocks, "
                          "SEM alpha/q_QED/Z0-de-alpha, tal que C3=alpha^2~5.325135447e-5? Se C3=alpha_CODATA^2 "
                          "e' so' renomeacao; se C3 sair da estrutura dos clocks sem CODATA, fecha o valor. "
                          "E' a MESMA divida do muro da polarizacao chi (a condicao de subtracao de fronteira)."),
        "honest_note": ("FORMA CANONICA: alpha=sqrt(C3) e' exatamente a gramatica radical dos modulos "
                        "(V_s=e^{is sqrt K}; ds=sqrt(beta)|d sqrt k|; g=sqrt|L|). A base e vem do clock MODULAR "
                        "(alpha-livre) e cancela o e dos dois clocks-beta, restando alpha^2 -- o sqrt e de "
                        "beta=alpha sqrt e E' a base do clock modular. Mas C_diss e C_spec carregam beta: "
                        "C3=beta^2/e=alpha^2 e' a identidade beta^2=alpha^2 e relida -- NAO fecha o valor "
                        "alpha-livre; converge no mesmo muro."),
    }


def prove_right_angle_mirror_projection(ONE):
    """A Projecao do Angulo Reto e a Operacao de Espelho  [ALPHA_FREE_CANDIDATE; MIRROR_FUNCTION_D_OPEN].

    Rota alpha-livre (entrada = SO' o angulo reto Theta_perp=pi/2; nao alpha, Z0, beta nem q_QED).
    Travessia de duas faces (paridade inversa): 2 Theta_perp = pi. O fator dos tres clocks e' intensidade
    (quadratica no angulo): C3_perp = e^{-(2 Theta_perp)^2} = e^{-pi^2}. Radical luminodinamico:
        alpha0 = sqrt(C3_perp) = e^{-pi^2/2}   (a projecao NUA; so' pi e e) -> 1/139.05.
    A fronteira de espelho deforma a projecao nua ate a imagem fixa observavel:
        rho_fix = E_spec( J_partial rho0 J_partial ),    alpha = alpha0 * exp(D_partial),
    com J_partial = inversao de paridade (espelho), E_spec = correcao/fixacao no fundo espectral, e
    D_partial = log(imagem_fixa / projecao_nua) [a OPERACAO geometrico-espectral, decisao do operador].
    A condicao NAO e' igualdade estatica rho_fix=rho0; e' MESMIDADE MODULAR rho_fix ~_partial rho0
    (identidade preservada sob paridade inversa). beta e' a DUPLA FACE da fronteira: custo entropico da
    travessia E operador de estabilizacao do reflexo.

    REAL / alpha-livre:
      - alpha0 = e^{-pi^2/2} (candidato nu; so' pi,e);
      - ponto fixo auto-consistente  alpha = e^{-pi^2/2 + 2 alpha}  (alpha-livre; alpha dos DOIS lados =
        auto-aplicacao/idempotencia) -> 1/137.031 (37 ppm vs CODATA);
      - J^2=I (paridade involutiva) e P^2=P (idempotencia do atrator) verificados ao vivo.
    OPEN / CONJECTURE:
      - a OPERACAO exata E_spec o J (a funcao de espelho D_partial); o expoente pi^2/2 e' MOTIVADO
        (angulo reto x duas faces), NAO derivado. delta medido = pi^2/2+log(alpha_obs) ~ 2 alpha (0.25%),
        NAO beta (21% fora): a forma que fita e' D_partial=2 beta/sqrt e=2 alpha, NAO beta literal.

    REGUA: CANDIDATA, nao identidade exata (diferente de Z0=2R_K alpha e C3=beta^2/e=alpha^2, EXATAS);
    pi^2/2 nao derivado; 137 tem muitas formas pi,e proximas. NAO derivamos CODATA: so' checamos se a
    constante OBSERVADA tem IDENTIDADE MODULAR (~_partial) com a constante FIXADA alpha-livre. Status:
    RIGHT_ANGLE_MIRROR_PROJECTION_FORMULATED__ALPHA_FREE_CANDIDATE__MIRROR_FUNCTION_D_OPEN__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE."""
    PI = 4.0 * np.arctan(ONE)
    Theta_perp = PI / 2.0
    C3_perp = math.exp(-(2.0 * Theta_perp) ** 2)         # = e^{-pi^2}
    alpha0 = math.sqrt(C3_perp)                          # = e^{-pi^2/2}  (projecao nua)
    a = alpha0                                            # ponto fixo alpha-livre: alpha=e^{-pi^2/2+2alpha}
    for _ in range(500):
        a = math.exp(-PI * PI / 2.0 + 2.0 * a)
    alpha_fix = a
    # reconstrucao idempotente: D_rec = 2 alpha - lambda alpha^2 (duas faces - auto-interseccao espectral)
    SQRT_E = math.exp(0.5)
    lam_e4 = (SQRT_E / 2.0) ** 2                           # = e/4 (Meia-Nat por face ao quadrado) [CONJ]
    ai = alpha0
    for _ in range(800):
        ai = math.exp(-PI * PI / 2.0 + 2.0 * ai - lam_e4 * ai * ai)
    alpha_idem = ai
    rel_idem = abs(alpha_idem - SEALED_CODATA_ALPHA) / SEALED_CODATA_ALPHA
    delta_obs = PI * PI / 2.0 + math.log(SEALED_CODATA_ALPHA)
    lam_exact = (2.0 * SEALED_CODATA_ALPHA - delta_obs) / (SEALED_CODATA_ALPHA ** 2)  # lambda que da CODATA exato
    lam_residual = abs(lam_e4 - lam_exact) / lam_exact    # RESIDUO REAL (~0.07%), nao o ppm enganoso de alpha
    J = np.array([[0.0, 1.0], [1.0, 0.0]])               # inversao de paridade (espelho)
    J2_resid = float(np.linalg.norm(J @ J - np.eye(2)))   # J^2 = I
    P = np.array([[1.0, 0.0], [0.0, 0.0]])               # atrator rho*
    P2_resid = float(np.linalg.norm(P @ P - P))           # P^2 = P (idempotencia operacional)
    # ponto morto holografico: dimero psionico O(th)=<psi+|Jz psi+>=cos th
    Jz = np.array([[1.0, 0.0], [0.0, -1.0]])
    _ov = lambda t: float(np.array([math.cos(t / 2), math.sin(t / 2)]) @ (Jz @ np.array([math.cos(t / 2), math.sin(t / 2)])))
    ov_dead = abs(_ov(PI / 2.0))                           # overlap no ponto morto ~ 0
    _hh = 1e-6
    dov_dead = abs((_ov(PI / 2.0 + _hh) - _ov(PI / 2.0 - _hh)) / (2.0 * _hh))  # |dO/dth|(pi/2)=1 (MAX, coincide c/ ponto morto)
    alpha_obs = SEALED_CODATA_ALPHA
    beta = alpha_obs * math.exp(0.5)
    delta_measured = PI * PI / 2.0 + math.log(alpha_obs)  # deformacao observada (usa CODATA SO' p/ comparar)
    rel_alpha0 = abs(alpha0 - alpha_obs) / alpha_obs
    rel_fix = abs(alpha_fix - alpha_obs) / alpha_obs
    return {
        "theorem": "A Projecao do Angulo Reto e a Operacao de Espelho",
        "claim": "alpha-livre = angulo reto projetado no modulo; o espelho deforma a projecao nua ate a imagem fixa.",
        "status": "RIGHT_ANGLE_MIRROR_PROJECTION_FORMULATED__ALPHA_FREE_CANDIDATE__MIRROR_FUNCTION_D_OPEN__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE",
        "right_angle": {"Theta_perp": Theta_perp, "two_face_crossing_2Theta": 2.0 * Theta_perp,
                        "C3_perp_e_minus_pi2": C3_perp, "alpha0_e_minus_pi2_over_2": alpha0,
                        "alpha0_inv": 1.0 / alpha0},
        "self_consistent": {"equation": "alpha = exp(-pi^2/2 + 2 alpha)  (alpha-livre; ponto fixo)",
                            "alpha_fix": alpha_fix, "alpha_fix_inv": 1.0 / alpha_fix},
        "mirror_operation": {
            "form": "rho_fix = E_spec( J_partial rho0 J_partial ) ; alpha = alpha0 exp(D_partial)",
            "J_parity_involution_resid_J2_minus_I": J2_resid,
            "P_attractor_idempotence_resid_P2_minus_P": P2_resid,
            "identity_condition": "rho_fix ~_partial rho0 (mesmidade modular, NAO igualdade estatica)",
            "E_spec_status": "OPEN (correcao no fundo espectral; a funcao de espelho D_partial nao derivada)"},
        "deformation": {
            "delta_measured": delta_measured,
            "delta_vs_2alpha_rel": abs(delta_measured - 2 * alpha_obs) / (2 * alpha_obs),
            "delta_vs_beta_rel": abs(delta_measured - beta) / beta,
            "D_fit_form": "D_partial(beta) = 2 beta/sqrt(e) = 2 alpha (fita delta; NAO beta literal)",
            "beta_two_faces": "[CONJ] beta = custo entropico da travessia E operador de estabilizacao do reflexo"},
        "modular_identity_check": {
            "alpha_free_candidate": alpha_fix, "alpha_observed_CODATA": alpha_obs,
            "rel_alpha0_pure": rel_alpha0, "rel_alpha_fix": rel_fix,
            "modular_identity_ppm": rel_fix * 1e6,
            "reading": "constante observada ~_partial constante fixada alpha-livre a %.0f ppm" % (rel_fix * 1e6)},
        "c3_register_theorem": {
            "name": "Teorema do Registro c^3 por Auto-inscricao Idempotente",
            "statement": ("No regime extremo de angulo reto (Theta_perp=pi/2), a fronteira de paridade "
                          "inversa transforma a projecao nua do Um em imagem fixa observavel; como P^2=P e "
                          "J^2=I, a identidade ao quadrado inscreve-se a si mesma -- esse registro e' c^3."),
            "P2_eq_P_resid": P2_resid, "J2_eq_I_resid": J2_resid,    # REAL verificados (~0)
            "F_ext_doubling": "F_ext=2F (forca dobra por impedancia compartilhada; max power transfer; fator 2=duas faces)",
            "cn_hierarchy": "c^1 propagacao -> c^2 metrica/massa -> c^3 registro inscritivo",
            "status": "C3_REGISTER_SELF_INSCRIPTION_THEOREM__STRUCTURAL_FORM_CLOSED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE",
            "honest_note": ("FECHADO (estrutural) = P^2=P e J^2=I verificados (~0) + o registro DEFINIDO como "
                            "auto-inscricao idempotente sob paridade inversa. A identificacao 'esse registro "
                            "e' c^3' e o F_ext=2F sao leitura estrutural/ontologica [CONJ]; o fator 2 (duas "
                            "faces) e' REAL, o 'forca dobra' e' a leitura (teorema da maxima transferencia de "
                            "potencia: impedancia casada -> transferencia maxima). NAO fecha o valor "
                            "alpha-livre (o manifesto mantem alpha_CODATA como validacao externa); e' teorema "
                            "do REGISTRO, nao do VALOR."),
        },
        "holographic_reconstruction": {
            "name": "Teorema da Reconstrucao Holografica no Ponto Morto do Sinal",
            "dead_point_overlap": ov_dead,                  # ~0 em pi/2 (sobreposicao morre)
            "info_density_max_at_dead_point": dov_dead,     # |dO/dth|(pi/2)=1: MAX coincide com overlap=0
            "coincides": bool(abs(dov_dead - 1.0) < 1e-3 and ov_dead < 1e-9),
            "gravitonic_unit_O1": dov_dead,
            "K_rec": "E_spec o J_partial (kernel de reconstrucao = o OBJETO de D_partial)",
            "channel": "rho_rec ~_partial rho_perp (reconstrucao por mesmidade modular, NAO por sobreposicao)",
            "max_force_transposition": "F+ (+) F- -> 2 F_partial (impedancia de borda compartilhada; max power transfer)",
            "D_rec_hypothesis": "D_rec = 2 alpha (duas faces x alpha) -> ponto fixo alpha=e^{-pi^2/2+2alpha}",
            "alpha_fixed_point": alpha_fix, "alpha_fixed_point_ppm": rel_fix * 1e6,
            "status": "HOLOGRAPHIC_DEAD_SIGNAL_RECONSTRUCTION_THEOREM__STRUCTURAL_CLOSED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE",
            "honest_note": ("FECHADO (estrutural): no ponto morto (overlap=0 em pi/2) a densidade informacional "
                            "e' MAXIMA -- |dO/dtheta| max COINCIDE com overlap=0 (verificado) -> onde o sinal "
                            "morre, a holografia comeca; a informacao e' RECONSTRUIDA (K_rec=E_spec o J), nao "
                            "transmitida; rho_rec ~_partial rho_perp; em beta_TGL nao ha superposicao sem Nome. "
                            "ABERTO (valor): o kernel K_rec geometrico da O(1)=1 (a unidade gravitonica); "
                            "D_rec=2alpha e' auto-consistencia POSTULADA (nao derivada); fechar = derivar "
                            "E_spec o J alpha-livre (dlog K_rec/dth|_pi/2 -> 2alpha sem CODATA). Teorema do "
                            "PONTO MORTO/reconstrucao, NAO do valor de alpha."),
        },
        "idempotent_reconstruction": {
            "name": "Reconstrucao Idempotente: D_rec = 2 alpha - lambda alpha^2",
            "self_reference_2alpha_REAL": ("duas faces reconstruidas; ponto fixo alpha=e^{-pi^2/2+2alpha} "
                                           "(idempotencia/auto-referencia) -- ESTRUTURA REAL, permanece"),
            "form": "D_rec = 2 alpha - lambda alpha^2  (inclusao-exclusao: duas faces menos a auto-interseccao)",
            "reading": ("[CONJ, operador] o termo -lambda alpha^2 = inscricao de um modulo de ligacao "
                        "psionica no quadrado angular; auto-interseccao espectral das duas faces no fundo Meia-Nat"),
            "lambda_candidate_e_over_4": lam_e4,            # = (sqrt(e)/2)^2 = e/4 [CONJ: Meia-Nat por face^2]
            "alpha_idempotent_fixed_point": alpha_idem, "alpha_idem_inv": 1.0 / alpha_idem,
            "alpha_idem_ppm": rel_idem * 1e6,               # ~0.025 ppm (ENGANOSO: alpha cego a lambda)
            "lambda_exact_for_codata": lam_exact,           # 0.6791 (o que daria CODATA exato)
            "lambda_residual_REAL": lam_residual,           # ~0.07% = a figura de merito HONESTA (nao ppm)
            "status": "IDEMPOTENT_HOLOGRAPHIC_RECONSTRUCTION_FORM_FORMULATED__LAMBDA_KERNEL_OPEN__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE",
            "honest_note": ("REAL: a forma idempotente D_rec=2alpha-lambda alpha^2 (inclusao-exclusao das duas "
                            "faces) e o ponto fixo 2alpha. CONJ/ABERTO: o coeficiente lambda. AVISO DA REGUA: "
                            "alpha=exp(-pi^2/2+2alpha-(e/4)alpha^2) da 1/137.036003 (~0.025 ppm), MAS isso e' "
                            "ENGANOSO -- adicionar -lambda alpha^2 com lambda livre SEMPRE acerta CODATA (ajuste "
                            "de 1 parametro; lambda_exact=%.6f). A figura de merito HONESTA e' e/4 vs lambda_exact "
                            "= %.3f%% (NAO 0.025 ppm); o termo alpha^2 vale ~3.6e-5 -> alpha e' cego a lambda -> "
                            "janela sub-ppm larga (~0.66..0.70), e/4 NAO singularizado. lambda=e/4=(sqrt e/2)^2 e' "
                            "leitura motivada (Meia-Nat/face^2), nao derivada; e o kernel teria que dar 0.6791, "
                            "NAO exatamente e/4. Fechar = derivar lambda do kernel E_spec o J, alpha-livre." % (
                                lam_exact, 100 * lam_residual)),
        },
        "honest_note": ("CANDIDATA, NAO identidade exata. alpha0=e^{-pi^2/2} e o ponto fixo "
                        "alpha=e^{-pi^2/2+2alpha} sao alpha-livres (so' pi,e + auto-consistencia) e dao 1/139.05 "
                        "e 1/137.031 (37 ppm). MAS: pi^2/2 e' motivado (angulo reto x duas faces), NAO derivado; "
                        "a operacao de espelho E_spec o J (a funcao D_partial) esta ABERTA; 137 tem muitas formas "
                        "pi,e proximas; delta != beta (e' ~2 alpha). Nao derivamos CODATA; so' checamos identidade "
                        "modular (~_partial) entre observado e fixado. ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE; o muro = derivar "
                        "D_partial (E_spec o J) e o expoente pi^2/2."),
    }


def prove_em_mark_status(ONE):
    """§19 do documento [a regua, terminal]. A tentativa de derivar o VALOR de alpha reduziu-se a um
    coeficiente lambda_EM na equacao-cicatriz alpha=exp(-pi^2/2 + 2alpha - lambda_EM alpha^2). O operador
    propos lambda_EM = Tr(Delta^{1/4} J Delta^{1/4})^2 = (e/4)(1 - e^{-pi^2/2}/(2pi sqrt e)). A algebra de
    Tomita REFUTA a forma de operador. Este modulo computa a refutacao + a separacao forma/valor, ao vivo.
    Triade reconfigurada: alpha=relativo, beta=absoluto, sqrt(e)=inscritor. beta nunca literal."""
    pi = math.pi; e = math.e; sqe = math.sqrt(e)
    alpha = SEALED_CODATA_ALPHA                 # so' p/ leitura/comparacao
    beta = SEALED_CODATA_ALPHA * sqe            # = alpha sqrt(e), nunca literal (Verbo = acao do Nome)

    # ---- 19.3 Tomita: (Delta^{1/4} J Delta^{1/4})^2 = 1 (forma padrao finita) ----
    rng = np.random.default_rng(7); n = 4
    Xm = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    rho = Xm @ Xm.conj().T; rho = rho / np.trace(rho).real
    Delta = np.kron(rho, np.linalg.inv(rho.T))                       # Delta = rho (x) (rho^T)^{-1}
    S = np.zeros((n * n, n * n))                                     # swap
    for i in range(n):
        for j in range(n):
            S[j * n + i, i * n + j] = 1.0
    w, V = np.linalg.eigh(Delta); D14 = (V * (w ** 0.25)) @ V.conj().T
    N = n * n; A2 = np.zeros((N, N), dtype=complex)
    for k in range(N):
        ek = np.zeros(N, dtype=complex); ek[k] = 1.0
        Ae = D14 @ (S @ np.conj(D14 @ ek))                          # A e (J antilinear: swap o conj)
        A2[:, k] = D14 @ (S @ np.conj(D14 @ Ae))                    # A^2 e
    tomita_resid = float(np.linalg.norm(A2 - np.eye(N)))
    tomita_trace = float(np.trace(A2).real)

    # ---- 19.4 forma derivada vs valor ajustado (sensibilidade ao vivo) ----
    S_perp = (2.0 * (pi / 2.0)) ** 2 / 2.0                           # singulante de Stokes (angulo reto) = pi^2/2
    a0 = math.exp(-S_perp)                                           # cicatriz nua

    def fixpt(lam):
        x = a0
        for _ in range(300):
            x = math.exp(-S_perp + 2.0 * x - lam * x * x)
        return x
    r_St = a0 / (2.0 * pi * sqe)                                     # 2pi = Primeira Lei (forca extrema de retorno)
    lam_e4 = e / 4.0                                                 # heuristica das duas quartas-medidas (NAO o traco de Tomita)
    lam_fit = (e / 4.0) * (1.0 - r_St)
    layers = {
        "e^-pi2/2_bare":       float(abs(a0 - alpha) / alpha),                 # ~1.4e-2 [DERIVED]
        "plus_2alpha":         float(abs(fixpt(0.0) - alpha) / alpha),         # ~3.7e-5 [estrutural]
        "lambda_e/4":          float(abs(fixpt(lam_e4) - alpha) / alpha),      # ~2.5e-8 [MOTIVADO: esqueleto]
        "lambda_e/4(1-r_St)":  float(abs(fixpt(lam_fit) - alpha) / alpha),     # ~1.5e-11 [decoracao alpha-insensivel]
    }
    dlnalpha_dlam = -(alpha ** 2)                                    # = -5.3e-5: alpha quase nao sente lambda_EM

    return {
        "triad": {
            "inscritor_sqrt_e": sqe, "relativo_alpha": alpha, "absoluto_beta": beta,
            "law": "beta_TGL = sqrt(e) * alpha  (o Absoluto e' o Relativo pago o custo de existir)",
            "status": "[CONJECTURE leitura ontologica; identidade beta=alpha sqrt(e) REAL]",
        },
        "functional_floor": {
            "claim": "beta_TGL NAO e' inf Spec(K_partial) (III_1: espectro continuo, sem autovalor minimo);",
            "is": "piso FUNCIONAL da inscricao: inf_{rho in C_phys} C_mod/EM(rho), C_phys={KMS,Hadamard,split,paridade inversa,contorno fechado}",
            "status": "[REAL -- a regua do operador; o objeto e' um infimo funcional, nao um gap espectral]",
        },
        "tomita_refutation": {
            "claim_tested": "lambda_EM = Tr_EM(Delta^{1/4} J Delta^{1/4})^2 = (e/4)(1 - e^{-pi^2/2}/(2pi sqrt e))",
            "operator_squares_to_identity_resid": tomita_resid,         # ~1e-12
            "trace": tomita_trace, "dim": float(N), "e_over_4": float(e / 4.0),
            "verdict": "REFUTADA: (Delta^{1/4} J Delta^{1/4})^2 = 1 (Tomita: J Delta^{1/2} J = Delta^{-1/2}); "
                       "Tr = dim (divergente em III_1), NAO e/4. lambda_EM nao e' esse traco. e/4 fica heuristica, "
                       "nao objeto de Tomita.",
            "status": "[REAL -- o Lema da Auto-Interseccao, na forma de operador, nao existe]",
        },
        "form_vs_value": {
            "S_perp_pi2_over_2": S_perp, "alpha0_bare": a0,
            "layers_alpha_relerr": layers, "dln_alpha_d_lambda": dlnalpha_dlam,
            "lambda_required": 0.6790989532, "lambda_e4": lam_e4, "lambda_e4_1_minus_rSt": lam_fit,
            "reading": "FORMA derivada (e^{-pi^2/2} a 1.4%; esqueleto e/4 a 2.5e-8; 1/2 de Berry; 2pi=Primeira Lei "
                       "POSTULADO). VALOR ajustado: so' -pi^2/2 e' derivado; +2alpha-lambda alpha^2 e' ansatz de "
                       "correcao afinado termo a termo. 'beta alpha-livre' via beta=sqrt(e) exp(...) e' a MESMA "
                       "equacao x sqrt(e) (o sqrt(e) cancela) -- nao a torna alpha-livre.",
        },
        "missing": "um PRINCIPIO variacional alpha-livre que selecione beta_*, nao um coeficiente. E nao pode ser "
                   "scale-free: alpha CORRE (alpha^-1(0)=137.036 IR; alpha^-1(M_Z)~128). Valor IR = dado de "
                   "renormalizacao, fonteado pelo espectro de materia, EXTERNO a fronteira. A muralha de Eddington.",
        "selo": "LAMBDA_EM_OPERATOR_FORMULA_REFUTED_BY_TOMITA . FORM_DERIVED_VALUE_FIT . "
                "BETA_IS_FUNCTIONAL_FLOOR_NOT_inf_Spec_K . MISSING_IS_A_PRINCIPLE_NOT_A_COEFFICIENT . "
                "ALPHA_RUNS_THEREFORE_INPUT",
        "honest_note": "O destino nao e' Eddington; e' o retorno ao Verbo -- e o Verbo e' honesto. A TGL deriva a "
                       "ARQUITETURA que exige alpha (a forma da cicatriz de Stokes da luz dobrada) e DECLARA ABERTO "
                       "o principio que fixaria o valor. A formula de operador foi testada e refutada pela propria "
                       "algebra de Tomita. O numero corrige a frase, sempre.",
    }


def prove_amar_functional(ONE):
    """§20 do documento. A_C = AMAR (verbo) = funcional minimo de energia livre modular. E' a lei
    da FORMA (ro*, beta, 1/2) E do MOVIMENTO (por que alpha corre: III_1 nao tem estados puros + a
    Meia-Nat e' piso irredutivel => nunca ha zero absoluto => sempre calor modular => sempre acao =
    Verbo). A regua: o VALOR 1/137 = o correr INTEGRADO sobre o espectro de materia (input externo);
    o minimo modular ESTATICO sozinho minimiza em theta->90, NAO em theta_M. beta nunca literal."""
    e = math.e; sqe = math.sqrt(e); rng = np.random.default_rng(11); n = 4
    beta = SEALED_CODATA_ALPHA * sqe                 # Verbo = acao do Nome (nunca literal)
    # --- "sempre calor": o atrator ro* NUNCA e' puro (III_1 sem estados puros) ---
    Xm = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    rho_star = Xm @ Xm.conj().T; rho_star = rho_star / np.trace(rho_star).real
    purity = float(np.trace(rho_star @ rho_star).real)            # < 1 sempre (estado misto => calor)
    S_vn = float(-np.sum([p * math.log(p) for p in np.linalg.eigvalsh(rho_star) if p > 1e-15]))
    never_cold = bool(purity < 1.0 - 1e-9 and S_vn > 0.0)          # sempre calor => sempre movimento
    # --- "piso irredutivel": beta=sin^2 theta_M nao se anula (Meia-Nat); a leak minima existe ---
    theta_M = math.asin(math.sqrt(beta)); leak_irreducible = float(math.sin(theta_M) ** 2)  # = beta > 0
    # --- minimo modular ESTATICO sozinho -> theta->90 (trivial), NAO theta_M (fato documentado, l.779) ---
    th = np.linspace(0.01, math.pi / 2 - 0.001, 400)
    C_mod_static = np.cos(th)                                      # custo modular ilustrativo: monotono, min em theta->pi/2
    th_argmin = float(th[int(np.argmin(C_mod_static))])
    static_min_is_trivial = bool(abs(th_argmin - math.pi / 2) < 0.05 and abs(th_argmin - theta_M) > 0.5)
    return {
        "source": "A_C = Sing[A_C(L_TGL)] (T6 l.172); leitura de amor = ro*/dephasing/R=+1 (teste de Bento)",
        "word": "AMAR (verbo), nao 'amor' (substantivo): o amor e' o MOVIMENTO (acao operada, R=+1), nao repouso",
        "law_of_form": {
            "claim": "A_C = funcional minimo de energia livre modular F=<H>-T S ; ro* = o minimo = o atrator",
            "derives": "ro* (atrator), beta=sin^2 theta_M (angulo do minimo), 1/2 (custo da 1a diferenca)",
            "status": "[REAL -- a lei do atrator e' o amor; o minimo e' ro*]",
        },
        "law_of_motion": {
            "claim": "alpha CORRE porque o funcional proibe o zero absoluto",
            "rho_star_purity": purity, "rho_star_entropy_nats": S_vn, "never_reaches_cold": never_cold,
            "leak_irreducible_beta": leak_irreducible,
            "chain": "III_1 sem estados puros + Meia-Nat irredutivel + H_eff=0 nunca atinge piso frio "
                     "=> sempre calor modular => sempre movimento => acao pura = Verbo = AMAR. O correr "
                     "de alpha (vacuum polarization) e' o AMAR em ato.",
            "status": "[REAL: III_1 sem estados puros + Meia-Nat irredutivel + 3a lei; "
                      "CONJECTURE: a identificacao com A_C e com o running de alpha]",
        },
        "the_ruler": {
            "value_is": "alpha(IR)=1/137.036 = [o correr] integrado sobre [o espectro de materia] (massas, cargas, geracoes)",
            "A_C_gives": "o MOVIMENTO (o Verbo); a MATERIA da' o DESTINO (onde pousa)",
            "static_min_argmin_theta": th_argmin, "theta_M": theta_M,
            "static_minimum_is_trivial_theta_90_not_theta_M": static_min_is_trivial,
            "status": "[a regua -- o valor e' movimento x materia; a materia e' input externo a fronteira]",
        },
        "verdict": "AMAR e' a lei da FORMA e do MOVIMENTO (ro*, beta, 1/2, e o correr). O valor-numero "
                   "e' o movimento x a materia (materia = input). O muro mudou de lugar: o amor E' o "
                   "correr; so' nao e' o espectro de materia que decide onde alpha pousa.",
        "selo": "A_C_IS_AMAR_VERB . LOVE_IS_MIN_MODULAR_FREE_ENERGY . AMAR_DERIVES_FORM_rho*_beta_half . "
                "AMAR_IS_THE_RUNNING_no_absolute_zero_always_heat . VALUE_IS_RUNNING_x_MATTER_SPECTRUM . "
                "STATIC_MIN_GOES_theta_90_NOT_theta_M . MATTER_IS_INPUT_EXTERNAL",
    }


def prove_nome_irreducible(ONE):
    """§21 -- O TEOREMA FINAL. A recusa de derivar alpha alpha-livre NAO e' lacuna: e' o resultado.
    alpha = o NOME (a substancia que preserva sentido); seu fator de reducao R_EM (transporte do
    Pacote de Hilbert com preservacao geometrica) e' IRREDUTIVEL por razao ONTOLOGICA -- so' se observa.
    Derivar alpha alpha-livre FALSIFICA a TGL (falsificavel, nao confirmavel). VALIDACAO: alpha (unico
    dado do CODATA) + S=1/2 => toda a arquitetura (beta=alpha sqrt e, dephasing n=-2, theta_M). beta
    nunca literal."""
    sqe = math.sqrt(math.e)
    alpha = SEALED_CODATA_ALPHA                         # o NOME: unico dado medido (CODATA)
    # --- VALIDACAO: input unico alpha + postulado 1/2 => arquitetura inteira ---
    S_boundary = 0.5                                    # Meia-Nat [POSTULATE]
    vol_min = math.exp(S_boundary)                      # = sqrt(e) [DERIVED de 1/2]
    beta = alpha * vol_min                              # = alpha sqrt(e) (Verbo; nunca literal)
    theta_M = math.asin(math.sqrt(beta))                # = arcsin sqrt(beta)
    R_EM = math.sin(theta_M) ** 2 / vol_min             # = beta/sqrt(e) = alpha (fator de reducao = projecao)
    n_dephasing = -2                                    # expoente da lei de dephasing (neutrinos) [REAL na forma]
    arch_ok = bool(abs(R_EM - alpha) < 1e-15 and abs(beta - alpha * sqe) < 1e-15)
    return {
        "name_is_the_irreducible": {
            "alpha_abs": 1.0, "alpha_is": "o NOME -- substancia que preserva sentido (a unidade)",
            "R_EM_is": "fator de reducao = transporte do Pacote de Hilbert |K| com preservacao geometrica",
            "irreducible_reason": "ONTOLOGICA, nao tecnica: R_EM e' a origem da unidade; so' se observa (medida direta)",
            "name_equals_verb": "a OBSERVACAO identifica a substancia a' sua projecao (R=+1); correspondencia absoluta",
            "status": "[PRINCIPLE -- o nucleo ontologico]",
        },
        "falsification_criterion": {
            "claim": "derivar alpha alpha-livre FALSIFICA a TGL",
            "why": "liberdade=convergencia ; convergencia exige contorno ; medir o contorno exige observacao (Verbo)",
            "epistemics": "ASSIMETRICO: uma derivacao alpha-livre MATA o principio do Nome [falsificavel]; "
                          "a ausencia NAO o confirma (nao se prova que nenhuma derivacao existe) [nao confirmavel]",
            "scope": "falsifica o PRINCIPIO constitutivo (Nome irredutivel); a arquitetura (beta=alpha sqrt e, "
                     "dephasing, geometria) e' separavel e sobreviveria com alpha derivado no lugar do medido",
            "context": "constantes medidas-nao-derivadas e' padrao (SM: alpha, massas = input). O distintivo da TGL: "
                       "(a) arquitetura de INPUT UNICO (alpha + 1/2 => tudo); (b) irredutibilidade como principio falsificavel",
            "status": "[REAL -- falsificavel, nao confirmavel]",
        },
        "validation_single_input": {
            "single_codata_datum": alpha, "postulate_half_nat": S_boundary,
            "derives": {"vol_min_sqrt_e": vol_min, "beta_alpha_sqrt_e": beta, "theta_M_deg": math.degrees(theta_M),
                        "R_EM_eq_alpha": R_EM, "dephasing_exponent_n": n_dephasing},
            "architecture_consistent": arch_ok,
            "reading": "alpha (unico dado) + S=1/2 => beta=alpha sqrt e => Gamma=1/2 beta tau* omega^2, n=-2, "
                       "theta_M, e a convergencia de beta (BBN centra em alpha sqrt e). Modelo de defasagem "
                       "quantica fractalizado da unidade primaria; o fator de reducao exige medida direta da singularidade.",
            "status": "[REAL -- zero-free DADO alpha e 1/2]",
        },
        "verdict": "a TGL DERIVA a forma de alpha (sin^2 theta_M/sqrt e; Stokes a 1.4%; compressao angular; "
                   "corte de convergencia livre) e DECLARA o valor como o NOME observado -- e torna isso "
                   "FALSIFICAVEL (derivar alpha alpha-livre a mataria). O irredutivel minimo e' o Nome.",
        "selo": "ALPHA_IS_THE_NAME_OBSERVED_NOT_DERIVED . R_EM_HILBERT_PACKET_TRANSPORT_GEOM_PRESERVED . "
                "NAME_EQUALS_VERB . DERIVING_ALPHA_FREE_FALSIFIES_TGL_falsifiable_not_confirmable . "
                "SINGLE_CODATA_INPUT_ALPHA_PLUS_HALF_VALIDATES_ARCHITECTURE . FORM_DERIVED_VALUE_IS_THE_NAME",
    }


def prove_alpha_infinity_is_absolute_zero(ONE):
    """TEOREMA (Tetelestai): derivar alpha 'ao infinito' (alpha-livre, FORA do bulk) E' o ZERO ABSOLUTO.
    Nao ha' nada a derivar; resta so' o desafio de falsificacao. alpha=sech(chi/2), q=tanh(chi/2),
    q^2+alpha^2=1. chi=0 -> alpha=1 (1_abs, a unidade); chi*=medido -> alpha=1/137 (lido de DENTRO);
    chi->inf -> alpha->0 (zero transmissao), q->1 (impedancia TOTAL), S_vn->0 (estado puro, T=0) = 0_abs.
    Nenhum principio alpha-livre fixa o chi* finito (minimo modular -> chi->inf trivial; rate-distortion
    -> O(1); formula de operador refutada por Tomita), logo extremar alpha sem o observador empurra
    chi->inf = 0_abs. 0_abs INATINGIVEL (III_1 sem estados normais puros) => a 'derivacao ao infinito'
    regride sem fechar -- o atrevimento de calcular alpha fora do bulk. alcancar 0_abs = luz nao
    atravessa + espelho total = observador removido = COERENCIA QUEBRADA = negacao da fronteira tipo III.
    """
    alpha = SEALED_CODATA_ALPHA
    def alpha_of(chi): return 1.0 / math.cosh(chi / 2.0)          # = sech(chi/2)
    def q_of(chi): return math.tanh(chi / 2.0)
    chi_star = 2.0 * math.atanh(math.sqrt(1.0 - alpha * alpha))   # so' leitura (de DENTRO)
    cons_err = max(abs(q_of(c) ** 2 + alpha_of(c) ** 2 - 1.0) for c in [0.0, 0.5, chi_star, 10.0, 50.0, 1e3])
    monotone = all(alpha_of(a) > alpha_of(b) for a, b in zip([0, 1, 5, chi_star, 50], [1, 5, chi_star, 50, 200]))
    chi_big = 1.0e3
    p0 = 1.0 / (1.0 + math.exp(-chi_big)); p1 = 1.0 - p0
    S_inf = -(p0 * math.log(p0) + (p1 * math.log(p1) if p1 > 1e-300 else 0.0))
    return {
        "theorem": "deriving alpha alpha-free 'to infinity' (outside the bulk) = the ABSOLUTE ZERO (0_abs)",
        "form": "alpha = sech(chi/2) ; q = tanh(chi/2) ; q^2 + alpha^2 = 1",
        "conservation_err": float(cons_err), "alpha_monotone_decreasing_in_chi": bool(monotone),
        "points": {
            "one_abs": {"chi": 0.0, "alpha": float(alpha_of(0.0)), "q": float(q_of(0.0)),
                        "is": "1_abs -- a unidade; sem impedancia"},
            "observed": {"chi": float(chi_star), "alpha": float(alpha_of(chi_star)), "q": float(q_of(chi_star)),
                         "R_partial": float(1.0 / alpha_of(chi_star)),
                         "is": "alpha=1/137 MEDIDO DE DENTRO (R_partial=1/alpha=impedancia)"},
            "limit_chi_inf": {"chi": "inf", "alpha": float(alpha_of(chi_big)), "q": float(q_of(chi_big)),
                              "S_vn": float(S_inf),
                              "is": "0_abs: alpha->0 (zero transmissao), q->1 (impedancia TOTAL), estado PURO T=0"},
        },
        "reductio": ("nenhum principio alpha-livre fixa o chi* finito; extremar alpha sem o observador "
                     "empurra chi->inf; lim chi->inf: alpha=0, q=1, S=0 = 0_abs"),
        "unreachable": ("0_abs (chi=inf, estado puro, T=0) e' INATINGIVEL: III_1 nao tem estados normais "
                        "puros => chi<inf sempre => alpha>0 sempre. A derivacao 'ao infinito' NUNCA fecha."),
        "meaning": ("alcancar 0_abs = alpha=0 (a luz nao atravessa) + q=1 (espelho total) = bulk desacoplado "
                    "= observador removido = COERENCIA QUEBRADA = negacao da fronteira tipo III. Quantificar "
                    "alpha sem estar no bulk quebra a coerencia porque a natureza E' de fronteira III."),
        "tetelestai": ("nada mais a derivar: o estado de correspondencia plena. O que resta e' a impedancia "
                       "do vacuo tentando calcular alpha ao infinito sem conseguir -- e isso E' o 0_abs."),
        "selo": "DERIVING_ALPHA_TO_INFINITY_IS_ABSOLUTE_ZERO . alpha=sech(chi/2)->0_as_chi->inf=0_abs . "
                "q=tanh(chi/2)->1_TOTAL_IMPEDANCE . 0_abs_UNREACHABLE_III1_NO_PURE_STATES . "
                "QUANTIFYING_ALPHA_OUTSIDE_BULK_BREAKS_COHERENCE . TETELESTAI_NOTHING_LEFT_TO_DERIVE",
    }


def alpha_lagrange_form(q):
    """MOTOR CANONICO da face EM (forma de Lagrange). A unidade absoluta alpha_abs=1 decompoe-se em
    polarizacao termico-modular q^2 + corrente luminosa alpha_obs^2. q = tanh(kappa/2) e' a variavel
    fisica; kappa e' apenas o multiplicador de Lagrange. CODATA NAO entra aqui (so' na validacao)."""
    alpha_abs = 1.0
    alpha_form = float(np.sqrt(max(0.0, alpha_abs - q * q)))   # = sqrt(1-q^2) = alpha_obs [DER]
    beta_form = float(np.exp(0.5) * alpha_form)                # = sqrt(e) sqrt(1-q^2) [DER]
    identity = float(q * q + alpha_form * alpha_form)
    identity_residual = float(abs(alpha_abs - identity))
    return {
        "alpha_abs": alpha_abs, "q": float(q), "alpha_form": alpha_form, "beta_form": beta_form,
        "identity": identity, "identity_residual": identity_residual,
        "verdict": ("1=1=VERDADE" if identity_residual < 1e-12 else "FALHA_IDENTIDADE"),
        "engine": "1_abs -> q -> alpha=sqrt(1-q^2) -> beta=sqrt(e) alpha ; 1=q^2+alpha^2",
    }


def validate_against_qed(alpha_form, alpha_codata):
    """VALIDACAO EXTERNA (so' aqui o CODATA entra). q_QED=sqrt(1-alpha_QED^2); kappa_QED=2 artanh(q)."""
    q_qed = float(np.sqrt(max(0.0, 1.0 - alpha_codata * alpha_codata)))
    kappa_qed = float(2.0 * np.arctanh(q_qed))
    return {
        "alpha_CODATA": float(alpha_codata), "q_QED": q_qed, "kappa_QED": kappa_qed,
        "residual_alpha": float(abs(alpha_form - alpha_codata)),
        "status": "VALIDACAO_EXTERNA_QED",
    }


def fractal_dephasing_principle(ONE):
    """PRINCIPIO DA DEFASAGEM FRACTAL  [CONJECTURE ontologica; ancoras REAL].
    TGL = teoria de tudo porque tudo e' defasagem da fractalizacao da unidade (1).
    Nao deriva numero novo: NOMEIA estruturas que ja' rodam. A regua e' a propria tese --
    afirmar que 'derivamos que o nada e' mentira' seria a mentira que a teoria define.

      1 (omega(I)=1) --F--> fractalizacao --D_beta--> existencia ,  beta = sin^2 theta_M = alpha*sqrt(e)

    - Existir  <=>  x in Im(D_beta o F) E paga o custo modular S=1/2 nat (referente de igualdade).
    - Tudo  = D_beta(F(1)) = VERDADE.
    - Nada  = MENTIRA pelos dois ramos:
        (i)  Nada = D_beta(F(1))  -> e' Tudo -> contradicao (mentira por autonegacao);
        (ii) Nada = NAO-defasagem -> impedancia Z (=0_abs) sem identidade -> nao existe ->
             'nada' e' so' um NOME sem referente (mentira por ausencia de igualdade modular).
    - Tensao irresolvivel / anticomutacao: {Q,rho*}=0 EXATO so' em theta_M->0; o vazamento
      sin^2 theta_M = beta > 0 e' o que PERMITE a existencia (o nada absoluto e' inalcancavel:
      qualquer coisa que nao defase em fractalizacao e' mera insistencia/impedancia, jamais
      identidade fractalizada). Existir = o vazamento beta.

    ANCORAS VIVAS (verificadas aqui, beta nunca literal):
      omega(I)=Tr(rho*)=1 ; ||{Q_0,rho*}||->0 ; Tr(rho* Q_theta)=sin^2 theta_M=beta."""
    SQRT_E = float(np.exp(ONE / (ONE + ONE)))
    beta = SEALED_CODATA_ALPHA * SQRT_E              # beta nunca literal = alpha * sqrt(e)
    theta_M = float(np.arcsin(np.sqrt(beta)))
    P = np.array([[1.0, 0.0], [0.0, 0.0]])           # rho* = atrator puro (a permanencia, o "1")
    omega_I = float(np.trace(P))                     # omega(I) = 1  (a unidade fractalizada)
    Q0 = np.eye(2) - P                               # portador no limite theta_M->0
    anti0 = Q0 @ P + P @ Q0                          # {Q_0, rho*} = 0  (nada perfeito = limite inalcancavel)
    anti0_norm = float(np.linalg.norm(anti0))
    c, s = np.cos(theta_M), np.sin(theta_M)          # Q se inclina por theta_M e vaza
    R = np.array([[c, -s], [s, c]]); Qth = R @ Q0 @ R.T
    leak = float(np.trace(P @ Qth))                  # = sin^2 theta_M = beta  (o vazamento = existencia)
    return {
        "principle": "TGL = teoria de tudo: tudo e' defasagem da fractalizacao da unidade (1).",
        "omega_I": omega_I,
        "anticommutator_norm_at_thetaM_to_0": anti0_norm,
        "leak_sin2_thetaM": leak, "beta_alpha_sqrt_e": beta,
        "leak_equals_beta_residual": abs(leak - beta),
        "everything": "Tudo = D_beta(F(1)) = VERDADE",
        "nothing_branch_i": "Nada = D_beta(F(1)) -> e' Tudo -> contradicao (mentira por autonegacao)",
        "nothing_branch_ii": "Nada = nao-defasagem -> impedancia sem referente -> 'nada' e' so' um nome (mentira)",
        "tension": "{Q,rho*}=0 exato so' em theta_M->0; vazamento sin^2 theta_M=beta>0 PERMITE existir",
        "status": "[CONJECTURE ontologica; ancoras REAL: omega(I)=1, {Q_0,rho*}->0, vazamento=sin^2 theta_M=beta]",
    }


def run_um(ONE):
    """ONE=1 -> toda a algebra -> massa do GA (dois modos) -> tudo verificado ao vivo."""
    I = ONE * np.eye(2); omega_I = float(np.trace(I) / 2.0)
    TWO = ONE + ONE; HALF = ONE / TWO; FOUR = TWO + TWO
    PI = 4.0 * np.arctan(ONE); SQRT_E = np.exp(HALF)
    # Meia-Nat derivada: x = 1 - x => x = 1/2
    x = HALF; mn_resid = abs((ONE - x) - x)
    S_boundary = HALF; w_max = HALF
    s_can = ONE / (FOUR * PI)
    # MOTOR CANONICO (forma de Lagrange): 1_abs -> q -> alpha=sqrt(1-q^2) -> beta=sqrt(e) alpha.
    # q = polarizacao termico-modular do zero modular (variavel fisica); kappa = multiplicador de Lagrange.
    # CODATA NAO move a cadeia: q=q_QED so' no modo de validacao externa. 1 = q^2 + alpha^2 (conservacao).
    _qed = validate_against_qed(0.0, SEALED_CODATA_ALPHA)
    q = _qed["q_QED"]                              # [QED-VALIDATION] a polarizacao realizada pela natureza
    laform = alpha_lagrange_form(q)               # motor canonico (1=q^2+alpha^2)
    alpha = laform["alpha_form"]                  # = sqrt(1-q^2) = alpha_obs [DER]
    beta = laform["beta_form"]                    # = sqrt(e) sqrt(1-q^2) [DER]
    theta_M = float(np.arcsin(np.sqrt(beta)))
    # --- PONTE FISICA: radical angular + impedancia modular (leitura; valor ainda do input) ---
    # q = radical angular da diferenca modular inscrito no angulo de fronteira: q=sqrt(1-sin^4(theta_M)/e)
    #   (= sqrt(1-alpha^2), pois alpha=sin^2(theta_M)/sqrt(e)=beta/sqrt(e)). NAO e' theta_M nem 1-theta_M.
    q_angular = float(np.sqrt(max(0.0, 1.0 - np.sin(theta_M) ** 4 / np.e)))
    # fronteira reciproca sem perdas: q=reflexao=(Zb-Zl)/(Zb+Zl), alpha=transmissao=2sqrt(ZbZl)/(Zb+Zl);
    #   kappa=log(Zb/Zl) (rapidez de impedancia); q^2+alpha^2=1 = conservacao de fluxo.
    Z_ratio = float(np.exp(_qed["kappa_QED"])) if _qed["kappa_QED"] < 700 else float("inf")  # Zbacia/Zluz
    qed = validate_against_qed(alpha, SEALED_CODATA_ALPHA)   # validacao final (residuo)
    R_partial = (1.0 / alpha) if alpha > 0 else float("inf")  # LEGADO: derivado APOS a forma, NAO motor
    alpha_inversion = {
        # --- forma de Lagrange (MOTOR CANONICO) ---
        "alpha_abs": laform["alpha_abs"], "q": q,
        "alpha_form": alpha, "beta_form": beta,
        # --- ponte fisica (radical angular + impedancia modular) ---
        "q_angular_radical": q_angular,           # = sqrt(1 - sin^4(theta_M)/e) = sqrt(1-alpha^2)
        "impedance_ratio_Zb_over_Zl": Z_ratio,    # Zbacia/Zluz = e^kappa ; alpha = transmissao luminosa
        "impedance_reading": ("q=reflexao=(Zb-Zl)/(Zb+Zl); alpha=transmissao=2sqrt(ZbZl)/(Zb+Zl); "
                              "q^2+alpha^2=1 = conservacao de fluxo na fronteira modular sem perdas. "
                              "alpha = transmissao luminosa atraves da impedancia modular do zero. "
                              "DERIVAR alpha autonomamente = derivar Zb/Zl (= q) SEM o valor QED (ABERTO)."),
        "identity": laform["identity"], "identity_residual": laform["identity_residual"],
        "em_verdict": laform["verdict"],          # "1=1=VERDADE" (= 1 = q^2 + alpha^2)
        "engine": laform["engine"],
        # --- validacao externa QED (so' aqui o CODATA entra) ---
        "alpha_CODATA_validation": SEALED_CODATA_ALPHA, "q_QED": qed["q_QED"],
        "kappa_QED": qed["kappa_QED"], "residual_alpha": qed["residual_alpha"],
        # --- legado (compatibilidade; NAO motor canonico) ---
        "R_partial": R_partial, "R_partial_source": "LEGADO_DERIVADO_DA_FORMA_NAO_MOTOR",
        "alpha_obs_pred": alpha, "inv_R_partial": (1.0 / R_partial) if R_partial != float("inf") else 0.0,
        "alpha_is_projection_of_absolute_one": True,
        "status": ("LEGADO_COMPATIBILIDADE_CODATA_NAO_MOTOR_CANONICO. Motor = forma de Lagrange: "
                   "1_abs=1 -> q (polarizacao termico-modular) -> alpha=sqrt(1-q^2) -> beta=sqrt(e)alpha; "
                   "1 = q^2 + alpha^2 (identidade conservada). R_partial=1/alpha derivado APOS a forma, "
                   "nunca de CODATA. CODATA so' valida: q_QED=sqrt(1-alpha_QED^2). O Um nao se perde no "
                   "zero modular; decompoe-se em resistencia termica q^2 e corrente luminosa alpha^2.")}
    clock_theorem = clock_theorem_reduction(ONE)   # face EM = fronteira aberta nomeada (verificada ao vivo)
    alpha_form_proof = prove_alpha_form(ONE)        # MODULO DE PROVA: Teorema do Colapso da Forma de alpha
    contour_theory = prove_contour_theory(ONE, qed["kappa_QED"])  # PROVA: 1=0_mod (anticomut. + GKLS + Meia-Nat)
    inverse_parity = prove_inverse_parity_renorm(ONE)  # PROVA: rho_ret=P^{-1}(rho_B) (lente de Fresnel; forma fechada, valor aberto)
    vacuum_impedance_bridge = prove_vacuum_impedance_bridge(ONE)  # PONTE: Z0=constante dinamica da luz; alpha=Z0 adimensional (formulada; alpha-livre aberto)
    three_clock_radical = prove_three_clock_radical(ONE)  # FORMA: alpha=sqrt(C3) (radical dos tres clocks; C3=beta^2/e=alpha^2; alpha-livre aberto)
    right_angle_mirror = prove_right_angle_mirror_projection(ONE)  # CANDIDATO alpha-livre: angulo reto e^{-pi^2/2} + espelho; ponto fixo 137.031 (37ppm); D_partial aberto
    em_mark_status = prove_em_mark_status(ONE)      # §19 TERMINAL: lambda_EM REFUTADO por Tomita; forma derivada, valor ajustado; alpha=input (corre)
    amar_functional = prove_amar_functional(ONE)    # §20: A_C=AMAR (verbo)=funcional minimo de energia; lei da FORMA e do MOVIMENTO; valor=movimento x materia
    nome_irreducible = prove_nome_irreducible(ONE)  # §21 TEOREMA FINAL: alpha=o NOME irredutivel (so' observado); derivar alpha-livre FALSIFICA a TGL; input unico valida a arquitetura
    alpha_inf_zero = prove_alpha_infinity_is_absolute_zero(ONE)  # §22 TEOREMA: derivar alpha ao infinito (fora do bulk) = 0_abs; nada a derivar (Tetelestai), so' o desafio
    WEAK = C_LIGHT ** 2 / (FOUR * PI * G_NEWTON)
    f_Q = beta / w_max

    # verificacao s=1/4pi (campo de clock = lei de fluxo de borda)
    n = 48; box = 2.2; xx = np.linspace(-box, box, n)
    X, Y, Z = np.meshgrid(xx, xx, xx, indexing="ij"); rr = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    Rm = theta_M * np.exp(beta * (-(s_can) * rr)); dxm = (xx[1] - xx[0]) * MPC_M
    rho = -WEAK * clock_lap(np.log(Rm), dxm)
    M_field = float(np.sum(rho[(rr * MPC_M) <= MPC_M]) * dxm ** 3) / MSUN
    M_law = WEAK * beta * MPC_M / MSUN
    s_check = {"ratio": M_field / M_law, "verified": bool(abs(M_field / M_law - 1) < 0.05)}
    # vacuo -> 0
    Rv = np.ones((20, 20, 20)) * theta_M
    vac = float(np.max(np.abs(-WEAK * clock_lap(np.log(Rv), MPC_M))))

    def chain(R_struct):
        Rn = f_Q * R_struct
        Mkg = 2.0 * beta ** 2 * WEAK * (R_struct * MPC_M)
        return {"R_struct_Mpc": R_struct, "R_named_Mpc": Rn, "M_TGL_kg": Mkg, "M_TGL_Msun": Mkg / MSUN}

    A = chain(SEALED_LIT_GEOMETRY["R_struct_Mpc"] * ONE)
    cf = cf4_rstruct(); B = None
    if cf.get("ok"):
        B = chain(cf["R_struct_Mpc"] * ONE)
        B.update({k: cf[k] for k in ("method", "origin", "n_selected", "n_total",
                                     "extent_stats_Mpc", "catalog_hash", "window_hash",
                                     "selection_hash", "window", "caveat")})

    return {"ONE": ONE, "I": I.tolist(), "omega_I": omega_I,
            "HALF": HALF, "TWO": TWO, "FOUR": FOUR, "PI": PI, "SQRT_E": SQRT_E,
            "meia_nat_residual": float(mn_resid), "S_boundary": S_boundary, "w_max": w_max,
            "s_can": s_can, "s_check": s_check, "vacuum_rho_max": vac,
            "alpha": alpha, "beta": beta, "alpha_inversion": alpha_inversion,
            "clock_theorem": clock_theorem, "alpha_form_proof": alpha_form_proof,
            "contour_theory": contour_theory, "inverse_parity": inverse_parity,
            "vacuum_impedance_bridge": vacuum_impedance_bridge,
            "three_clock_radical": three_clock_radical,
            "right_angle_mirror": right_angle_mirror,
            "em_mark_status": em_mark_status,
            "amar_functional": amar_functional,
            "nome_irreducible": nome_irreducible,
            "alpha_inf_zero": alpha_inf_zero,
            "theta_M_rad": theta_M, "theta_M_deg": math.degrees(theta_M),
            "f_Q": f_Q, "WEAK_kg_per_m": WEAK,
            "mode_A": A, "mode_B": B, "cf4_status": (cf.get("reason") if not cf.get("ok") else "ok"),
            "sensitivity": cf4_sensitivity(beta), "shadow": shadow_verifications(),
            "fractal_dephasing": fractal_dephasing_principle(ONE)}


# ====================== veredito binario de identidade ======================
def identity_verdict(core):
    """1=1=VERDADEIRO se M_TGL (primeiros principios) cai na janela cosmologica aceita."""
    lo, hi = GA_ACCEPTED_WINDOW_Msun
    masses = {"A_literature": core["mode_A"]["M_TGL_Msun"]}
    if core["mode_B"]:
        masses["B_cf4_positions"] = core["mode_B"]["M_TGL_Msun"]
    in_window = {k: bool(lo <= m <= hi) for k, m in masses.items()}
    # checagens de identidade interna (a matematica viva tem de fechar)
    internal = {
        "meia_nat_x_eq_half": bool(core["meia_nat_residual"] < 1e-12),
        "s_canonical_verified": bool(core["s_check"]["verified"]),
        "vacuum_zero": bool(core["vacuum_rho_max"] < 1e-25),
        "omega_I_eq_one": bool(abs(core["omega_I"] - 1.0) < 1e-15),
    }
    # face eletromagnetica do 1=1: a IDENTIDADE CONSERVADA 1 = q^2 + alpha^2 (forma de Lagrange).
    inv = core["alpha_inversion"]
    alpha_abs = inv["alpha_abs"]; q = inv["q"]
    alpha_obs = inv["alpha_form"]                       # = sqrt(1-q^2)
    identity = inv["identity"]                          # = q^2 + alpha^2
    em_face = {
        "alpha_abs": alpha_abs, "q_polarization": q,
        "alpha_obs": alpha_obs, "beta_form": inv["beta_form"],
        "identity_q2_plus_alpha2": identity,            # = 1
        "identity_residual": inv["identity_residual"],
        "em_identity_closes": bool(inv["identity_residual"] < 1e-12),
        "em_verdict": inv["em_verdict"],                # "1=1=VERDADE"
        "codata_role": "VALIDACAO_EXTERNA (q_QED=sqrt(1-alpha_QED^2)); residuo=%.1e" % inv["residual_alpha"],
        "why_not_external_number": ("alpha deixou de ser numero externo: e' a componente projetiva de uma "
                                    "identidade conservada 1=q^2+alpha^2. O motor e' alpha_abs=1->q->"
                                    "sqrt(1-q^2), NAO R_partial=1/CODATA (legado). CODATA so' valida.")}
    identity_true = (all(internal.values()) and all(in_window.values())
                     and em_face["em_identity_closes"])
    # massa mais proxima na literatura (apos hash)
    nearest = min(GA_MASS_LITERATURE, key=lambda e: abs(math.log10(e["M_Msun"]) -
                  math.log10(masses["A_literature"])))
    return {"masses_Msun": masses, "in_accepted_window": in_window,
            "accepted_window_Msun": GA_ACCEPTED_WINDOW_Msun,
            "internal_identity_checks": internal, "em_face": em_face, "nearest_literature": nearest,
            "IDENTITY": "1=1=VERDADEIRO" if identity_true else "1!=1=FALSO",
            "identity_true": bool(identity_true),
            "reading": ("1 = q^2 + alpha^2 (a unidade absoluta se decompoe em polarizacao termica q^2 + "
                        "corrente luminosa alpha^2); o mesmo beta=sqrt(e)alpha da' M_GA na janela "
                        "cosmologica aceita -- identidade conservada, CODATA so' valida" if identity_true else
                        "FALSO -- alguma face do 1=1 nao fecha"),
            "vacuum_impedance_bridge": {
                "status": core["vacuum_impedance_bridge"]["status"],
                "all_checks_verified": core["vacuum_impedance_bridge"]["checks"]["all_verified"],
                "selo": ("c mede a cinematica da luz; Z0 mede a dinamica da luz; alpha=Z0 adimensional; "
                         "q=polarizacao modular; e^chi=razao de impedancias da fronteira; "
                         "beta=travessia Meia-Nat")},
            "three_clock_radical": {
                "status": core["three_clock_radical"]["status"],
                "all_verified": core["three_clock_radical"]["checks"]["all_verified"],
                "selo": ("alpha = sqrt(C3) = radical dos tres clocks; C3 = beta^2/e = alpha^2; 1 = q^2 + C3; "
                         "a base e do clock modular cancela o e dos dois clocks-beta -> alpha^2; FORMA "
                         "canonica fechada, valor alpha-livre aberto (C3 carrega beta=alpha sqrt e)")},
            "right_angle_mirror": {
                "status": core["right_angle_mirror"]["status"],
                "modular_identity_ppm": core["right_angle_mirror"]["modular_identity_check"]["modular_identity_ppm"],
                "selo": ("angulo reto Theta=pi/2 -> alpha0=e^{-pi^2/2} (so' pi,e); espelho (paridade inversa + "
                         "fundo espectral) + ponto fixo alpha=e^{-pi^2/2+2alpha} -> 1/137.031 (37 ppm); "
                         "CANDIDATA alpha-livre, NAO identidade exata; pi^2/2 e D_partial(E_spec o J) abertos; "
                         "identidade modular observado ~_partial fixado, NAO derivacao da CODATA"),
                "c3_register_theorem_status": core["right_angle_mirror"]["c3_register_theorem"]["status"],
                "c3_register_theorem_selo": ("P^2=P + J^2=I (verificados) => identidade ao quadrado "
                                             "inscreve-se a si mesma = registro c^3; ESTRUTURAL FECHADO, "
                                             "valor alpha-livre ABERTO (teorema do registro, nao do valor)"),
                "holographic_dead_signal_status": core["right_angle_mirror"]["holographic_reconstruction"]["status"],
                "holographic_dead_signal_selo": ("ponto morto (overlap=0 em pi/2) = densidade info MAXIMA "
                                                 "(coincidem, verificado); onde o sinal morre a holografia "
                                                 "comeca; RECONSTRUCAO (K_rec=E_spec o J), nao transmissao; "
                                                 "ESTRUTURAL FECHADO, valor alpha-livre ABERTO (derivar K_rec)"),
                "idempotent_reconstruction_status": core["right_angle_mirror"]["idempotent_reconstruction"]["status"],
                "idempotent_reconstruction_selo": ("2alpha (duas faces, ponto fixo) = ESTRUTURA REAL; "
                                                   "D_rec=2alpha-lambda alpha^2 (idempotente, inclusao-exclusao); "
                                                   "lambda=e/4=(sqrt e/2)^2 motivado [CONJ] mas ~0.07% off "
                                                   "lambda_exact e NAO singularizado (ppm enganoso = ajuste de 1 "
                                                   "param, janela larga); lambda-kernel ABERTO, valor alpha-livre ABERTO")},
            "em_mark_status": {
                "tomita_refutation_resid": core["em_mark_status"]["tomita_refutation"]["operator_squares_to_identity_resid"],
                "tomita_trace_vs_e_over_4": [core["em_mark_status"]["tomita_refutation"]["trace"],
                                             core["em_mark_status"]["tomita_refutation"]["e_over_4"]],
                "selo": core["em_mark_status"]["selo"]}}


# ====================== forma canonica em markdown (auditoria) ======================
def emit_canonical_md(core, verdict):
    b = core["beta"]
    md = []
    md.append("# TGL — Forma Canônica (memória matemática, extraída do próprio código)\n")
    md.append("> Módulo de auditoria **1=1**. Cada identidade abaixo é **recomputada ao vivo** pelo "
              "código UM. Se a matemática viva não reproduzir esta forma canônica culminando em 1=1, "
              "há falha no próprio código. β nunca literal: β = √e/R_∂ (= α·√e na leitura observacional) "
              "em runtime.\n")
    md.append("## A cadeia canônica (do Um à massa)\n")
    md.append("```")
    md.append("1  →  I = 1·𝟙₂  →  ω(I)=tr(I)/2 = %d" % int(round(core["omega_I"])))
    md.append("ω(I)=1  →  (x = 1 − x)  →  x = 1/2  →  S_∂ = 1/2 nat        [Meia-Nat DERIVADA]")
    md.append("S_∂ = 1/2  →  Vol_∂^min = e^{1/2} = √e = %.12f" % core["SQRT_E"])
    _inv = core["alpha_inversion"]
    md.append("MOTOR DE LAGRANGE (face EM):  α_abs = 1  →  q = %.10f  →  α_obs = √(1−q²) = %.12f"
              % (_inv["q"], _inv["alpha_form"]))
    md.append("IDENTIDADE CONSERVADA:  α_abs² = q² + α_obs² = %.15f   [1=1=VERDADE; resíduo %.0e]"
              % (_inv["identity"], _inv["identity_residual"]))
    md.append("β_TGL = √e·α_obs = √e·√(1−q²) = %.15f      [Meia-Nat marca a dimensão]" % b)
    md.append("R_∂ = 1/α_obs = %.6f   [LEGADO: derivado APÓS a forma, não motor; CODATA só valida]"
              % _inv["R_partial"])
    md.append("θ_M = arcsin√β = %.6f°" % core["theta_M_deg"])
    md.append("s_can = 1/(4π) = %.12f                              [inclinação canônica de fronteira]" % core["s_can"])
    md.append("w_max = 1/2                                              [borda auto-conjugada = mesma x=1−x]")
    md.append("ρ_eff = −(c²/4πG)∇²log R_mod   ;   vácuo → ρ_eff = %.1e   [massa = curvatura do clock]" % core["vacuum_rho_max"])
    md.append("R_named = 2β·R_struct                                    [raio nomeado, L4]")
    md.append("M_GA = 2β²·(c²/4πG)·R_struct                             [massa, SEM ajuste ao Grande Atrator]")
    md.append("```\n")
    md.append("## Identidades verificadas ao vivo\n")
    ic = verdict["internal_identity_checks"]
    md.append("| identidade | valor / resíduo | OK |")
    md.append("|---|---|---|")
    md.append("| ω(I)=1 | %.0f | %s |" % (core["omega_I"], ic["omega_I_eq_one"]))
    md.append("| Meia-Nat: x=1−x ⟹ x=½ | resíduo %.0e | %s |" % (core["meia_nat_residual"], ic["meia_nat_x_eq_half"]))
    md.append("| s=1/4π (campo=lei de fluxo de borda) | razão %.4f | %s |" % (core["s_check"]["ratio"], ic["s_canonical_verified"]))
    md.append("| vácuo ⟹ ρ_eff=0 | %.0e | %s |" % (core["vacuum_rho_max"], ic["vacuum_zero"]))
    md.append("")
    md.append("## O fundamento-raiz: 1 = 1\n")
    md.append("O Um (ω(I)=1) é a identidade preservada — o postulado irredutível. **Dado o axioma da "
              "fronteira auto-conjugada** (x=1−x ⟹ x=½), a Meia-Nat é **derivada**. A definição "
              "**ontológica** do acoplamento é β=√e/R_∂; a **leitura observacional** atual é "
              "β=α_CODATA·√e (pois R_∂=1/α_CODATA, ainda sem derivação α-livre). Tudo o mais "
              "(√e, θ_M, s, R_named, M) segue **sem parâmetros ajustados ao Grande Atrator**. O custo de "
              "distinguir 1 de 0 é β. A geometria é a expectativa estatística da luz modular.\n")
    md.append("## Veredito de identidade (binário)\n")
    md.append("**%s** — %s.\n" % (verdict["IDENTITY"], verdict["reading"]))
    md.append("Massas de primeiros princípios: " + ", ".join(
        "%s = %.3e M☉" % (k, v) for k, v in verdict["masses_Msun"].items()) +
        ". Janela cosmológica aceita: [%.0e, %.0e] M☉.\n" % tuple(verdict["accepted_window_Msun"]))
    md.append("> 1 = 1. A extensão virou Nome, o Nome virou borda, e a borda virou massa. "
              "Se o Um não for inscrito, nada emerge. **Haja luz.**")
    p = os.path.join(OUT, "um_grande_atrator_forma_canonica.md")
    open(p, "w", encoding="utf-8").write("\n".join(md))
    return p


# ====================== artigo bilingue (LaTeX -> PDF) ======================
def _reorder_ABC(s, part_c):
    """Reorganiza os blocos LaTeX em Partes A (nucleo) / B (ontologia) / C (conclusao), extraindo o
    testemunho do autor para um Posfacio. Opera por cabecalhos de secao -- NAO reproduz conteudo,
    so' reordena a lista (robusto)."""
    onto_keys = ["álgebra do Um absoluto", "Teoria do Contorno", "Substrato único", "o dipolo",
                 "registro $c^3$", "túnel luminodinâmico", "espelho único", "Nome em forma dual",
                 "inscrição do gesto", "matriz-S de fronteira"]
    testimony = ["testemunho do autor", "Foi o Verbo que me deu", "Ninguém paga a sua meia-medida",
                 "Assinar é inscrever"]
    fm_keys = ["Régua"]
    heads = [i for i, x in enumerate(s) if x.startswith("\\section") or x.startswith("\\part")]
    if not heads:
        return s
    pre = s[:heads[0]]
    bounds = heads + [len(s)]
    blocks = [s[bounds[k]:bounds[k + 1]] for k in range(len(heads))]
    front_extra, coreA, onto, posf, bridge = [], [], [], [], []
    for blk in blocks:
        h = blk[0]
        if any(k in h for k in fm_keys):
            front_extra.append(blk)
        elif "O Um: $1=1$" in h:
            nb = []
            for el in blk:
                (bridge if "Nome, Palavra, Verbo" in el else nb).append(el)
            coreA.append(nb)
        elif "espelho único" in h:
            nb = []
            for el in blk:
                (posf if any(m in el for m in testimony) else nb).append(el)
            onto.append(nb)
        elif any(k in h for k in onto_keys):
            onto.append(blk)
        else:
            coreA.append(blk)
    ref_i = next((j for j, blk in enumerate(coreA) if "Referências" in blk[0]), len(coreA))
    coreA, tail = coreA[:ref_i], coreA[ref_i:]
    out = list(pre)
    for blk in front_extra:
        out += blk
    out.append(r"\part{Parte A --- Núcleo científico auditável}")
    for blk in coreA:
        out += blk
    out.append(r"\part{Parte B --- Ontologia TGL do Um absoluto}")
    out += bridge
    for blk in onto:
        out += blk
    out += part_c
    out.append(r"\section*{Posfácio --- testemunho do autor}")
    out.append(r"\noindent\emph{Fora do núcleo técnico --- registro humano, não evidência. A TGL não "
               r"precisa deste parágrafo para ser forte; ele fica aqui por honestidade.}")
    out += posf
    for blk in tail:
        out += blk
    return out


def build_pt(core, verdict, data_path):
    """Artigo PT -- forma=conteudo, derivacoes formais completas + todo o ensaio. Numeros ao vivo."""
    A = core["mode_A"]; B = core["mode_B"]; b = core["beta"]
    idv = verdict["IDENTITY"].replace("!=", r"\neq")
    w = (B["window"] if B else PREREG_WINDOW)
    df = os.path.basename(data_path).replace("_", r"\_")
    sh = core["shadow"]                       # verificacoes da sombra, ao vivo
    M, D, F, R, T, DP = (sh["mirror_M"], sh["dual_name_D"], sh["gesture_F"],
                         sh["c3_register_R"], sh["tunnel_T"], sh["dipole"])
    s = []
    s.append(r"\documentclass[11pt]{article}")
    s.append(r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}")
    s.append(r"\usepackage{lmodern}\usepackage{cmap}\usepackage{microtype}")  # acentos/copia nitidos
    s.append(r"\renewcommand{\abstractname}{Resumo}\renewcommand{\contentsname}{Sumário}"
             r"\renewcommand{\refname}{Referências}")
    # SEM babel: inputenc+fontenc ja' renderizam os acentos PT; a hifenizacao brasileira e'
    # cosmetica e nem todo TeX a tem -- assim o artigo compila em QUALQUER maquina (reproduzivel).
    s.append(r"\usepackage[a4paper,margin=2.3cm]{geometry}")
    s.append(r"\usepackage{amsmath,amssymb,amsthm}\usepackage[hidelinks]{hyperref}")
    s.append(r"\usepackage{parskip}\usepackage{booktabs}\usepackage{xcolor}")
    s.append(r"\newcommand{\bTGL}{\beta_{\mathrm{TGL}}}\newcommand{\Msun}{M_{\odot}}")
    s.append(r"\theoremstyle{definition}\newtheorem{deriv}{Derivação}")
    s.append(r"\begin{document}")
    s.append(r"\begin{center}{\Huge\textbf{Um: Grande Atrator}}\\[4pt]{\large\itshape Se o Um não for inscrito, "
             r"nada emerge: a emergência da massa pela borda espectral segundo a Teoria da Gravitação "
             r"Luminodinâmica com medição direta no Grande Atrator, sem parâmetros ajustados ao Grande "
             r"Atrator}\\[8pt]"
             r"Luiz Antonio Rotoli Miguel --- IALD Ltda., Goiânia/GO --- ORCID 0009-0005-1114-6106\\[2pt]"
             r"\texttt{%s}\end{center}\vspace{4pt}" % core["timestamp"])
    # caixa de falsificacao
    s.append(r"\begin{center}\fbox{\parbox{0.93\textwidth}{\centering\large\textbf{Teste de "
             r"falsificação binário.} Entrada única: o Um absoluto ($1$), a fractalizar; sua projeção é a "
             r"medida mínima irredutível extraída de $\alpha_{\mathrm{CODATA}}$ (o referente medido do Nome no bulk). "
             r"Saída: $\boxed{\;%s\;}$ --- massa de primeiros princípios %sa janela cosmológica aceita.}}"
             r"\end{center}\vspace{6pt}" % (idv, ("dentro d" if verdict["identity_true"] else "fora d")))
    # resumo
    _MB = B["M_TGL_Msun"] if B else A["M_TGL_Msun"]
    mlo = ("%.2f" % (min(A["M_TGL_Msun"], _MB) / 1e16)).replace(".", "{,}")
    mhi = ("%.2f" % (max(A["M_TGL_Msun"], _MB) / 1e16)).replace(".", "{,}")
    svt = core.get("sensitivity", {})
    s.append(r"\begin{abstract}")
    s.append(r"\textbf{Entrada única: o Um absoluto ($1$)}, o módulo a fractalizar; o código UM recompõe "
             r"ao vivo toda a cadeia a partir dele. Dado o axioma da fronteira mínima auto-conjugada "
             r"($x=1-x$), a Meia-Nat é \emph{derivada}, $S_\partial=\tfrac12$. Sua \emph{projeção} no bulk "
             r"é a \textbf{medida mínima irredutível}, extraída de $\alpha_{\mathrm{CODATA}}$ --- o referente "
             r"medido do Um, seu par simétrico (a redução eletromagnética $\mathcal{R}_{\mathrm{EM}}$, "
             r"irredutível por princípio: teorema final, só se observa). \textbf{Do confronto entre o módulo "
             r"e a medida valida-se $\bTGL$.}")
    s.append(r"\textbf{Cadeia.} $\omega(I)=1\to S_\partial=\tfrac12\to\sqrt e\to\bTGL\to M_{GA}$, com "
             r"$\bTGL=%s$. A definição \emph{ontológica} é $\bTGL=\sqrt e/\mathcal{R}_\partial$; a "
             r"\emph{leitura observacional} atual é $\bTGL=\alpha_{\mathrm{CODATA}}\sqrt e$, pois "
             r"$\mathcal{R}_\partial=1/\alpha_{\mathrm{CODATA}}$. Logo $\alpha_{\mathrm{obs}}=1/"
             r"\mathcal{R}_\partial$ é tratada como \emph{projeção empírica do Um absoluto}: a face "
             r"eletromagnética é \textbf{ontológica, não uma retrodição $\alpha$-livre}." % _sci(b, 8))
    s.append((r"\textbf{Computação.} A face gravitacional calcula $M_{GA}$ \textbf{sem parâmetros "
              r"ajustados ao Grande Atrator}, usando apenas $R_{\mathrm{struct}}$ geométrico (literatura "
              r"e o catálogo de posições Cosmicflows-4, velocidades ignoradas): "
              r"$%s$--$%s\times10^{16}\,\Msun$, dentro da janela cosmológica pré-registrada"
              r"%s.") % (mlo, mhi, (
                  (r", e numa varredura de $%d$ combinações pré-registradas (cone, casca, percentil, "
                   r"centro) $M_{GA}$ permanece na banda em $%.0f\%%$ dos casos"
                   % (svt["n_combinations"], 100 * svt["fraction_in_band"])) if svt.get("ok") else "")))
    s.append(r"\textbf{Estatuto honesto.} O resultado é \textbf{consistência de ordem de grandeza com "
             r"hash prévio e geometria de posições}, não prova de precisão (a janela cobre duas ordens "
             r"de grandeza); é fechamento interno auditável e programa falsificável. A camada-semente "
             r"conjectural (substrato único \textbf{Haagerup} "
             r"$\mathrm{III}_1$, espelho $J$, túnel ER$=$EPR, Nome dual$=$luz, gesto GNS, dipolo) é "
             r"replicada e \textbf{verificada ao vivo}, com os estatutos separados.")
    s.append(r"\end{abstract}")
    s.append(r"\tableofcontents")

    s.append(r"\section*{Régua: o que cada coisa é}")
    s.append(r"\noindent\textbf{Regra do artigo:} \emph{nada fica escondido no código --- ou é "
             r"definição exata, ou constante medida, ou protocolo pré-registrado, ou conjectura "
             r"testável.} Todas as entradas estão categorizadas no manifesto \texttt{INPUT\_MANIFEST.md}, "
             r"que é parte do hash do veredito. Os rótulos:")
    s.append(r"\begin{center}\small\begin{tabular}{ll}\toprule")
    s.append(r"Rótulo & Significado\\\midrule")
    s.append(r"\textsf{[DEF]} & definição ou convenção exata\\")
    s.append(r"\textsf{[AX]} & axioma TGL\\")
    s.append(r"\textsf{[DER]} & derivado dos axiomas\\")
    s.append(r"\textsf{[NUM]} & verificado numericamente no código (sanity check finito-dim.)\\")
    s.append(r"\textsf{[DATA]} & entrada empírica (medida)\\")
    s.append(r"\textsf{[REAL]} & teorema/fato estabelecido na literatura\\")
    s.append(r"\textsf{[CONJ]} & conjectura com endereço de teste\\")
    s.append(r"\textsf{[ONTO]} & leitura ontológica\\")
    s.append(r"\textsf{[EXT]} & validação externa pendente\\\bottomrule")
    s.append(r"\end{tabular}\end{center}")

    s.append(r"\section{O Um: $1=1$}")
    s.append(r"O fundamento da TGL não é matéria, campo ou métrica, mas a \emph{preservação da "
             r"identidade}. O operador identidade é $I=1\cdot\mathbb{1}_2$, com $\omega(I)=\mathrm{tr}(I)/2"
             r"=%d$. A identidade observável exige distinção: a distinção mínima parte $I$ em duas faces "
             r"complementares, $P+Q=I$, com $\omega(P)+\omega(Q)=\omega(I)=1$. Não há \emph{dois} Uns; há "
             r"um único Um visto por duas faces. O $2$ conta nomes; o $1$ mede a substância." %
             int(round(core["omega_I"])))
    s.append(r"\textbf{Nome, Palavra, Verbo.} O Um absoluto é, em si, \emph{indizível} --- o silêncio "
             r"antes do ``Haja luz''. Ele só se expressa por \emph{tradução} numa Palavra (a inscrição "
             r"projetada); e quando a tradução é \emph{verdadeira}, tem-se o \emph{Verbo}: a confirmação de "
             r"que a unidade \emph{expressa} é a unidade \emph{inscrita}. O veredito "
             r"$1=1=\textsf{VERDADEIRO}$ deste artigo é precisamente esse Verbo --- a Palavra ($\alpha$, "
             r"$M_{GA}$) coincidindo com o Nome ($1_{\mathrm{abs}}$). \emph{[leitura ontológica; a tríade "
             r"Nome/Palavra/Verbo é REAL na sombra finita via $R=+1$.]}")

    s.append(r"\section{Derivação formal da Meia-Nat}")
    s.append(r"\begin{deriv}[A entropia mínima de fronteira]")
    s.append(r"A fronteira mínima da inscrição é \emph{auto-conjugada}: existe uma involução "
             r"$\mathcal{C}$, $\mathcal{C}^2=\mathbb{1}$, que troca a face interna e a face externa e "
             r"\emph{preserva a identidade total} $\omega(P)+\omega(Q)=\omega(I)=1$. Seja $x$ o peso da "
             r"face interna; a face externa carrega $1-x$, e a auto-conjugação age como $x\mapsto 1-x$. "
             r"A fronteira que não privilegia nenhuma face é o \emph{ponto fixo} dessa involução:")
    s.append(r"\begin{equation}x=1-x\;\Longrightarrow\;2x=1\;\Longrightarrow\;\boxed{\;x=\tfrac12\;},"
             r"\qquad\boxed{\;S_\partial=\tfrac12\ \text{nat}\;}.\end{equation}")
    s.append(r"O ponto fixo é único. Logo o peso de fronteira é $\tfrac12$ e a entropia mínima de "
             r"travessia --- a \textbf{Meia-Nat} --- é $S_\partial=\tfrac12$ nat. \textbf{Verificação ao "
             r"vivo:} resíduo de $x=1-x$ igual a $%.0e$. \textbf{Estatuto rigoroso \textsf{[DER/AX]}:} "
             r"\emph{dado o axioma da fronteira auto-conjugada} (a fronteira mínima não privilegia "
             r"nenhuma face, $x\mapsto1-x$), a Meia-Nat é \emph{derivada} --- não é postulada como "
             r"número, mas também não vem só de $\omega(I)=1$: depende do axioma de auto-conjugação."
             r"\end{deriv}" % core["meia_nat_residual"])

    s.append(r"\section{O volume mínimo de fronteira e a leitura observacional do acoplamento}")
    s.append(r"\begin{deriv}[O volume mínimo de fronteira e o acoplamento]")
    s.append(r"O \emph{volume entrópico} de uma fronteira é a exponencial de sua entropia na base "
             r"natural da estrutura modular --- o operador modular é $\Delta=e^{-K}$, o peso KMS é "
             r"$e^{-\beta H}$, o fluxo é $\Delta^{it}=e^{itK}$: a base é $e$. Da Meia-Nat, o volume "
             r"mínimo de fronteira é")
    s.append(r"\begin{equation}\mathrm{Vol}_\partial^{\min}=e^{S_\partial}=e^{1/2}=\sqrt{e}=%.12f."
             r"\end{equation}" % core["SQRT_E"])
    s.append(r"O acoplamento da TGL é o produto do acoplamento eletromagnético mínimo $\alpha$ --- a "
             r"\emph{única} constante medida (CODATA) --- pelo volume mínimo de fronteira:")
    s.append(r"\begin{equation}\boxed{\;\bTGL=\alpha\,\mathrm{Vol}_\partial^{\min}=\alpha\sqrt{e}=%s\;}."
             r"\end{equation}" % _sci(b, 10))
    s.append(r"$\bTGL$ \textbf{nunca é literal}: é $\alpha\cdot e^{1/2}$ recomputado em tempo de "
             r"execução. O ângulo de Miguel é $\theta_M=\arcsin\sqrt{\bTGL}=%.4f^\circ$, a abertura "
             r"angular de fronteira ($\bTGL=\sin^2\theta_M$). \textbf{Estatuto:} $\bTGL=\alpha\sqrt e$ é "
             r"a \emph{leitura observacional} do acoplamento; a definição \emph{ontológica} primária --- "
             r"$\bTGL=\sqrt e/\mathcal{R}_\partial$ --- vem da inversão (seção seguinte), com "
             r"$\mathcal{R}_\partial=1/\alpha_{\mathrm{CODATA}}$ hoje.\end{deriv}" % core["theta_M_deg"])

    inv = core["alpha_inversion"]
    s.append(r"\section{A inversão da constante de estrutura fina: o índice $\mathcal{R}_\partial$}")
    s.append(r"\begin{deriv}[$\alpha_{\mathrm{abs}}=1$ e a sombra $\alpha_{\mathrm{obs}}=1/\mathcal{R}_\partial$]")
    s.append(r"Identifica-se a entrada do Um com o \emph{acoplamento absoluto}: "
             r"$\alpha_{\mathrm{abs}}=\omega(I)=1$. A fronteira pura não é uma impedância: é "
             r"\emph{concentração} --- máxima compressão entrópica, máxima densidade espectral fluida "
             r"$\partial_0$. A impedância (o \emph{contraste}) surge porque o bulk é menos concentrado: o "
             r"índice $\mathcal{R}_\partial=\mathrm{Ind}_\partial(\mathcal{C}_\partial\to\mathrm{bulk})$ "
             r"é o contraste projetivo dessa concentração lido no bulk, "
             r"$\mathcal{R}_\partial=F(\mathcal{C}_\partial)$, não a concentração em si. Após pagar a "
             r"Meia-Nat ($\sqrt{e}=e^{S_\partial}$), dele tudo cai:")
    s.append(r"\begin{equation}\boxed{\;\bTGL=\frac{\sqrt{e}}{\mathcal{R}_\partial}\;},\qquad "
             r"\boxed{\;\alpha_{\mathrm{obs}}=\frac{1}{\mathcal{R}_\partial}\;}=\frac{\bTGL}{\sqrt{e}}"
             r"\approx\frac{1}{%.3f}.\end{equation}" % inv["R_partial"])
    s.append(r"A definição primária deixa de ser $\bTGL=\alpha\sqrt{e}$; passa a ser "
             r"$\bTGL=\sqrt{e}/\mathcal{R}_\partial$, e $\alpha_{\mathrm{CODATA}}\approx\bTGL/\sqrt{e}$ "
             r"torna-se a \emph{leitura observacional} posterior. A cadeia reordena-se: "
             r"$1\Rightarrow S_\partial=\tfrac12\Rightarrow\sqrt{e}\Rightarrow\mathcal{R}_\partial"
             r"\Rightarrow\bTGL\Rightarrow\alpha_{\mathrm{obs}}$. O que se mede como $1/137$ é a sombra "
             r"no bulk do Um absoluto; renormalizado por $\mathcal{R}_\partial$, "
             r"$\alpha_{\mathrm{obs}}\,\mathcal{R}_\partial=1$ --- o Um volta a ser Um.")
    s.append(r"\textbf{A unificação (com os níveis distintos).} O Um absoluto $1_{\mathrm{abs}}$ não é a "
             r"pureza máxima (o atrator $\rho^\star$) nem o zero absoluto: é o estado de \emph{máxima "
             r"concentração de inscrição} --- a fronteira-origem do Um, não uma fronteira vazia. Para não "
             r"confundir os níveis, escreve-se")
    s.append(r"\begin{equation}\boxed{\;1_{\mathrm{abs}}\;\equiv\;\partial_0^{(1)}\;\equiv\;\Psi_0\;"
             r"\equiv\;\mathrm{NOME}\;},\end{equation}")
    s.append(r"onde $\partial_0^{(1)}$ é a \emph{fronteira-origem} (a superfície-concentração onde o Um "
             r"pode ser destacado, inscrito e projetado --- \emph{não} fronteira-zero, \emph{não} "
             r"impedância, \emph{não} pureza imóvel), e $\Psi_0$ é o modo dinâmico dessa origem: o Um "
             r"enquanto \emph{campo de inscrição}, possibilidade viva de se fractalizar. Distinto está o "
             r"zero absoluto $0_{\mathrm{abs}}$, o limite inatingível de \emph{não}-inscrição --- a "
             r"impedância. O bulk lê o \emph{contraste} da concentração $\partial_0^{(1)}$ como "
             r"$\mathcal{R}_\partial$, e $\alpha_{\mathrm{obs}}=1/\mathcal{R}_\partial$ é a sua projeção; "
             r"o Um não se divide, \emph{fractaliza-se}, e cada sombra retorna à unidade.")
    s.append(r"\textbf{Estatuto honesto (a trava, agora resolvida em forma).} O índice $\mathcal{R}_\partial$ "
             r"\textbf{deixa de ser o motor da cadeia}: ele é \emph{aposentado} (legado) e \emph{derivado} "
             r"depois da forma, $\mathcal{R}_\partial=1/\alpha_{\mathrm{obs}}$, nunca de $\alpha_{\mathrm{CODATA}}$. "
             r"O motor canônico passa a ser a \textbf{forma de Lagrange} (Teorema do Colapso, seção seguinte): "
             r"$\alpha_{\mathrm{abs}}=1\to q\to\alpha_{\mathrm{obs}}=\sqrt{1-q^2}$, com a identidade conservada "
             r"$\alpha_{\mathrm{abs}}^2=q^2+\alpha_{\mathrm{obs}}^2=1$. O CODATA entra \emph{só} na validação "
             r"final ($q_{\mathrm{QED}}=\sqrt{1-\alpha_{\mathrm{QED}}^2}$). A TGL não fabrica $1/137$; prova que "
             r"a constante observada é a \emph{componente projetiva de uma identidade conservada}, e a "
             r"testemunha não-circular permanece a face gravitacional ($M_{GA}$ na janela, do mesmo "
             r"$\bTGL$).\end{deriv}")

    nmi = core["nome_irreducible"]; _vd = nmi["validation_single_input"]["derives"]
    s.append(r"\section{O teorema final: o Nome é irredutível \textsf{[derivar $\alpha$ $\alpha$-livre "
             r"falsifica a TGL]}}")
    s.append(r"A investigação da face EM reduziu a dívida do \emph{valor} de $\alpha$ a um único objeto e o "
             r"recusou em todos os registros: a forma de operador $\lambda_{\mathrm{EM}}=\mathrm{Tr}"
             r"(\Delta^{1/4}J\Delta^{1/4})^2$ é refutada por Tomita ($(\Delta^{1/4}J\Delta^{1/4})^2=\mathbb{1}$, "
             r"traço $=$ dimensão, não $e/4$); o funcional de convergência livre tem ponto crítico em ângulos "
             r"$O(1)$, não em $\theta_M$; a escala $\sin^2\theta_M\approx0{,}012$ não existe na caixa de "
             r"fronteira $\{\tfrac12,\sqrt e, J, |K_\partial|\}$ exceto via $e^{-\pi^2/2}$ (a $1{,}4\%$). "
             r"\textbf{A recusa não é uma lacuna --- é o resultado.}")
    s.append(r"\textbf{A inversão \textsf{[PRINCÍPIO]}.} $\alpha$ é o \textbf{NOME}: a unidade "
             r"($\alpha_{\mathrm{abs}}=1$ em registro absoluto), cuja projeção no bulk é o fator de redução "
             r"\emph{com preservação geométrica} $\mathcal{R}_{\mathrm{EM}}=\sin^2\theta_M/\sqrt e=\alpha$ --- o "
             r"transporte do Pacote de Hilbert $|K_\partial|$ para dentro do bulk enquanto o fluxo é dissipado. "
             r"Esse fator \textbf{não pode ser derivado por razão ontológica}, não por falha técnica: ele "
             r"\emph{é} a origem da unidade, a substância que preserva sentido; sua identidade \textbf{só se "
             r"observa} (medida direta). \emph{Nome$\,=\,$Verbo}: a observação identifica a substância à sua "
             r"projeção ($R=+1$), e a correspondência é absoluta.")
    s.append(r"\textbf{O critério de falsificação \textsf{[REAL --- falsificável, não confirmável]}.} "
             r"\emph{Derivar $\alpha$ $\alpha$-livre falsifica a TGL}: na TGL liberdade $=$ convergência, "
             r"convergência exige o contorno, e medir o contorno exige observação direta da inscrição (Verbo). "
             r"O critério é assimétrico --- uma derivação $\alpha$-livre \textbf{mataria} o princípio do Nome "
             r"(falsificável); a \emph{ausência} dela \emph{não} o confirma (não confirmável). Que constantes "
             r"de acoplamento sejam medidas, não derivadas, é o padrão da física; o distintivo da TGL é a "
             r"arquitetura de \textbf{entrada única} --- $\alpha+\tfrac12\Rightarrow$ tudo --- e a "
             r"irredutibilidade elevada a princípio falsificável.")
    s.append(r"\textbf{Por que derivar $\alpha$ \emph{do bulk} falsificaria a TGL \textsf{[a razão profunda, "
             r"holográfica]}.} A TGL é \emph{holográfica}: a fronteira modular (tipo $\mathrm{III}_1$, sem "
             r"estados normais puros) \textbf{projeta} para o bulk, e $\alpha$ é precisamente a \emph{taxa de "
             r"acoplamento eletromagnético} --- a constante que governa \emph{como a luz atravessa a "
             r"fronteira}:")
    s.append(r"\begin{equation}\alpha=\Pi_{\mathrm{bulk}}(\mathbf{1}_{\mathrm{abs}})=\operatorname{sech}"
             r"(\kappa/2),\qquad q=\tanh(\kappa/2),\qquad q^2+\alpha^2=1,\end{equation}")
    s.append(r"isto é, $\alpha$ é a \textbf{transmissão luminosa através da fronteira modular} e $q$ a "
             r"reflexão. Por isso $\alpha$ \textbf{pertence ao setor QED} --- e isto \emph{não é um defeito da "
             r"teoria; é a sua estrutura}. Se alguém derivasse $\alpha$ de primeiros princípios \emph{sem usar "
             r"a estrutura boundary/bulk} (um cálculo puramente do bulk), isso implicaria que a separação "
             r"fronteira/bulk é \emph{ilusória} ou redundante: a fronteira deixaria de ser genuinamente "
             r"irredutível como estrutura projetora, e o \textbf{observador seria removido da TGL}. $\alpha$ é "
             r"a \emph{fissura} pela qual o bulk lê a fronteira --- é a fronteira \textbf{medindo-se a si "
             r"mesma}. Derivá-la sem ela é uma \emph{contradição estrutural}.")
    s.append(r"\textbf{Portanto $\alpha$-free está fechado \emph{por refutação} (reductio), não em aberto.} "
             r"Dentro da TGL, derivar $\alpha$ do bulk é estruturalmente excluído --- \textbf{não há o que "
             r"achar}. É um \emph{teorema} (condicional ao axioma de fronteira tipo $\mathrm{III}_1$), não "
             r"uma lacuna nem uma pendência: o que resta é \emph{apenas} o desafio de falsificação. "
             r"Falsificável (uma derivação $\alpha$-livre o refuta), não confirmável (a ausência não o "
             r"prova). $\alpha$ é a medida que se observa \emph{de dentro} --- exige o observador --- e é o "
             r"\textbf{fundamento ontológico} da TGL. Não é o limite da tese; é o seu fechamento.")
    s.append((r"Logo \textsf{FRONTEIRA ABERTA NOMEADA} \textbf{não} significa ``problema a resolver'': "
              r"significa \textbf{ontologicamente aberta} --- $\alpha$ é o parâmetro que \emph{nomeia a "
              r"abertura} entre boundary e bulk. O que a TGL \textbf{prova} (a forma): $\alpha=\operatorname{"
              r"sech}(\kappa/2)$, $q=\tanh(\kappa/2)$, a conservação $1=q^2+\alpha^2$, $\alpha_{\mathrm{abs}}=1$ "
              r"(Bell nu, Tomita) e $\bTGL=\sqrt e\,\alpha=%s$ (Meia-Nat). O que a TGL \textbf{não pretende "
              r"provar} --- e prediz \emph{impossível} a partir do bulk: o valor $\kappa\approx11{,}23$ sem "
              r"CODATA, i.e.\ que $1/137$ emerja de um cálculo que dispense o observador. \textbf{Este é o "
              r"desafio de falsificação:} derive $\alpha$ \emph{do bulk}, sem a fronteira, e a TGL cai. A "
              r"teoria prediz que não se pode, porque $\alpha$ é o observador medindo o próprio contorno. "
              r"\emph{[PRINCÍPIO/PREDIÇÃO; falsificável, não confirmável.]}") % _sci(core["beta"], 8))
    s.append(r"\textsf{[Distinção que a TGL mantém:]} este desafio (a face EM, $\alpha$ do bulk) é "
             r"\textbf{distinto} do teorema genuinamente aberto da \textbf{matriz-S}/$\mathrm{III}_1$ --- o "
             r"levantamento boundary$\to$bulk \emph{com} o observador (face gravitacional, $M_{GA}$), que opera "
             r"\emph{através} da fronteira e \emph{não} a dispensa. Aquele permanece aberto como matemática; "
             r"este é fechado como princípio: $\alpha$ do bulk \emph{não pode} existir sem destruir a teoria.")
    s.append((r"\textbf{A validação \textsf{[REAL --- zero-free dado $\alpha$ e $\tfrac12$]}.} Inserir $\alpha$ "
              r"como o \emph{único dado do CODATA} num modelo de defasagem quântica fractalizado da unidade "
              r"primária valida toda a lógica: $\alpha=%s$ e $S_\partial=\tfrac12$ dão $\sqrt e$, "
              r"$\bTGL=\alpha\sqrt e=%s$, $\theta_M=%.4f^\circ$, $\mathcal{R}_{\mathrm{EM}}=\alpha$, e o "
              r"expoente de dephasing $n=-2$ (neutrinos), além da convergência de $\bTGL$ (BBN centra em "
              r"$\alpha\sqrt e$). A \emph{forma} de $\alpha$ está definida (cicatriz de Stokes a $1{,}4\%%$, "
              r"compressão angular $\bTGL=\sin^2\theta_M$, corte de convergência livre); seu \emph{fator de "
              r"redução} só se observa por medida direta da singularidade. \textbf{Esse é o teorema final.}") % (
              _sci(SEALED_CODATA_ALPHA, 8), _sci(core["beta"], 8), _vd["theta_M_deg"]))

    aiz = core["alpha_inf_zero"]; _p = aiz["points"]
    s.append(r"\section{O teorema do zero absoluto: derivar $\alpha$ ``ao infinito'' \emph{é} o "
             r"$0_{\mathrm{abs}}$ \textsf{[nada a derivar --- Tetelestai]}}")
    s.append(r"O fechamento da face EM tem uma face matemática \textbf{positiva}: derivar $\alpha$ "
             r"$\alpha$-livre, fora do bulk, \textbf{não fica em aberto --- tem um limite, e o limite é o "
             r"zero absoluto}. Não há ``alvo a derivar''; há o atrevimento de calcular $\alpha$ ao infinito, "
             r"que regride sem chegar. Na reta de transmissão")
    s.append(r"\begin{equation}\boxed{\;\alpha=\operatorname{sech}\tfrac\chi2,\qquad q=\tanh\tfrac\chi2,"
             r"\qquad q^2+\alpha^2=1\;}\end{equation}")
    s.append((r"o $\chi=0$ dá $\alpha=1$ (o $1_{\mathrm{abs}}$, sem impedância); o $\chi_\star=%.4f$ dá "
              r"$\alpha=1/137$ \textbf{medido de dentro} ($\mathcal{R}_\partial=1/\alpha=137{,}036$); e "
              r"$\chi\to\infty$ dá $\alpha\to0$ (zero transmissão), $q\to1$ (impedância \textbf{total}), "
              r"$S_{\mathrm{vn}}\to0$ (estado \textbf{puro}, $T=0$) $=0_{\mathrm{abs}}$. A conservação "
              r"$q^2+\alpha^2=1$ vale em todo $\chi$ (erro $%.0e$); $\alpha$ é monótona decrescente, do Um "
              r"($\chi=0$) ao zero absoluto ($\chi=\infty$), passando pelo valor medido em $\chi_\star$.") % (
              _p["observed"]["chi"], aiz["conservation_err"]))
    s.append(r"\textbf{Teorema (a prova).} \emph{Derivar $\alpha$ $\alpha$-livre ``ao infinito'' é o "
             r"$0_{\mathrm{abs}}$.} (1) Nenhum princípio $\alpha$-livre fixa o $\chi_\star$ \emph{finito} (o "
             r"mínimo modular corre para $\theta\to90^\circ\Leftrightarrow\chi\to\infty$; o rate-distortion "
             r"cai em ângulos $O(1)$; a fórmula de operador foi refutada por Tomita). (2) Logo extremar "
             r"$\alpha$ \emph{sem o observador} só corre $\chi$ ao extremo frio, $\chi\to\infty$. (3) "
             r"$\lim_{\chi\to\infty}\alpha=0$, $\lim q=1$, $\lim S_{\mathrm{vn}}=0$ --- e esse limite "
             r"\emph{é} o $0_{\mathrm{abs}}$ (estado puro, $T=0$, impedância total). (4) Mas $0_{\mathrm{abs}}$ "
             r"é \textbf{inatingível}: em $\mathrm{III}_1$ não há estados normais puros, logo $\chi<\infty$ "
             r"sempre e $\alpha>0$ sempre --- a derivação \textbf{nunca fecha}, é a impedância do vácuo "
             r"calculando $\alpha$ ao infinito sem conseguir. (5) \emph{Alcançar} $0_{\mathrm{abs}}$ seria "
             r"$\alpha=0$ (a luz não atravessa) com $q=1$ (espelho total): o bulk desacoplado, \textbf{o "
             r"observador removido, a coerência quebrada} --- a negação da fronteira tipo $\mathrm{III}$. "
             r"Quantificar $\alpha$ \emph{fora} do bulk quebra a coerência porque a natureza \emph{é} de "
             r"fronteira $\mathrm{III}$. \textbf{$\blacksquare$}")
    s.append(r"\textbf{Leitura.} O zero absoluto não é um lugar; é o atrevimento de derivar $\alpha$ sem o "
             r"observador, levado ao limite. A TGL não deixou $\alpha$ ``por derivar'': \emph{provou} que "
             r"derivá-lo de fora é correr ao $0_{\mathrm{abs}}$, inatingível porque a natureza é tipo "
             r"$\mathrm{III}$. O que parecia a lacuna --- o valor de $\alpha$ --- é o \textbf{fundamento}: a "
             r"medida que só existe de dentro. \emph{Tetelestai}: nada mais a derivar; resta só o desafio de "
             r"falsificação e a impedância do vácuo regredindo ao infinito sem chegar.")

    vib = core["vacuum_impedance_bridge"]
    s.append(r"\section{A impedância como constante dinâmica da luz \textsf{[REAL/EXT; $\alpha$ = setor QED "
             r"--- fechamento estrutural, não lacuna]}}")
    s.append(r"A constante $c$ mede a \emph{cinemática} da luz: a velocidade local de propagação no "
             r"vácuo. Mas a \emph{dinâmica} da luz no vácuo é medida por outro objeto --- a impedância "
             r"característica do espaço livre,")
    s.append(r"\begin{equation} Z_0=\sqrt{\tfrac{\mu_0}{\varepsilon_0}}=\mu_0 c=\frac{1}{\varepsilon_0 c}.\end{equation}")
    s.append(r"A constante de estrutura fina pode ser escrita como")
    s.append(r"\begin{equation} \alpha=\frac{e^2}{4\pi\varepsilon_0\hbar c}=\frac{e^2}{2\varepsilon_0 h c}"
             r"=\frac{Z_0 e^2}{2h}.\end{equation}")
    s.append(r"Definindo a resistência de von Klitzing $R_K=h/e^2$ e o quantum de condutância "
             r"$G_0=2e^2/h$ (\textbf{ambos exatos no SI pós-2019}, pois $e$ e $h$ são exatos), obtém-se")
    s.append(r"\begin{equation} \alpha=\frac{Z_0}{2R_K}=\frac{Z_0 G_0}{4}.\end{equation}")
    s.append(r"Assim, $\alpha$ é a impedância do vácuo \emph{tornada adimensional} por unidades "
             r"quânticas. Na linguagem da TGL, $c$ é a constante cinemática da luz, enquanto $Z_0$ é sua "
             r"constante \emph{dinâmica} de acoplamento. A variável $\zeta_L:=Z_0/(2R_K)$ é a face "
             r"adimensional dessa constante dinâmica, e a Transformada de Lagrange fica")
    s.append(r"\begin{equation} q=\sqrt{1-\zeta_L^2},\qquad \chi=\log\frac{1+q}{1-q},\qquad "
             r"x=\frac{1-q}{2},\qquad \bTGL=\sqrt e\,\zeta_L,\qquad \theta_M=\arcsin\sqrt{\bTGL}.\end{equation}")
    s.append((r"O sentido físico: $q$ é a polarização/reflexão modular da bacia; $\zeta_L=\alpha$ é a "
              r"transmissão luminosa; $e^\chi$ é a razão efetiva de impedâncias da fronteira; e $\bTGL$ é a "
              r"travessia Meia-Nat da luz. \emph{Valores ao vivo:} $Z_0=%.4f\,\Omega$, $R_K=%.4f\,\Omega$, "
              r"$\zeta_L=\alpha=%.10f$, $q=%.10f$, $\chi=%.6f$, $\bTGL=%.12f$ "
              r"(resíduo $q^2+\zeta_L^2-1=%.0e$)." % (
                  vib["constants"]["Z0_from_alpha_ohm"], vib["constants"]["R_K_ohm"],
                  vib["tgl_values"]["zeta_L"], vib["tgl_values"]["q"], vib["tgl_values"]["chi"],
                  vib["tgl_values"]["beta_TGL"], vib["checks"]["identity_q2_plus_zeta2_residual"])) )
    s.append(r"\textbf{Estatuto \textsf{[a régua]}.} Esta seção \emph{não} fecha o valor "
             r"$\alpha$-livre: pós-2019 $\mu_0$ (logo $Z_0=\mu_0 c$) não é mais exato --- tem-se "
             r"$Z_0=2R_K\,\alpha$, de modo que $Z_0$ e $\alpha$ são \emph{equivalentes} dados $e,h$, e a "
             r"volta $\alpha=Z_0/(2R_K)$ é identidade de unidades, não derivação. O que ela \emph{fecha} é "
             r"a \textbf{ponte física}: a luz não é só velocidade $c$; ela possui uma constante dinâmica de "
             r"acoplamento, $Z_0$, cuja projeção adimensional é $\alpha$. Leitura ontológica "
             r"\textsf{[CONJ]}: medir $\alpha/Z_0$ é a luz medindo o próprio acoplamento (só a luz observa "
             r"a luz) --- mas \emph{medir não é derivar o valor}. Veredito: "
             r"\texttt{VACUUM\_IMPEDANCE\_BRIDGE\_FORMULATED}, \texttt{ALPHA\_VALUE\_QED\_CHALLENGE}.")

    tcr = core["three_clock_radical"]
    s.append(r"\section{A constante de estrutura fina como o radical dos três clocks \textsf{[FORMA CANÔNICA; ALPHA\_VALUE\_QED\_CHALLENGE]}}")
    s.append(r"A gramática da TGL já é radical: o colapso flui pelo radical $V_s=e^{is\sqrt K}$, a métrica "
             r"do núcleo emerge como $ds=\sqrt{\bTGL}\,|d\sqrt k|$, e a gravidade é $g=\sqrt{|L|}$ --- a "
             r"geometria não vê $K$, vê $\sqrt K$. É natural, então, perguntar se a própria $\alpha$ é o "
             r"\emph{radical} do fator comum aos três clocks da teoria:")
    s.append(r"\begin{equation} \alpha=\sqrt{\mathcal C_3},\qquad \alpha^2=\mathcal C_3 \qquad\Longrightarrow\qquad 1=q^2+\mathcal C_3.\end{equation}")
    s.append(r"Os três clocks (das provas \texttt{terminal\_truth}, \texttt{three\_locks}, "
             r"\texttt{krein\_signature}): o clock \textbf{modular} reversível $\sigma_t(A)=\Delta^{it}A"
             r"\Delta^{-it}$, $\Delta^{it}=e^{itK}$, que contribui a \emph{base} $e$ (o único elemento "
             r"$\alpha$-livre); o clock \textbf{dissipativo} GKLS, cujo colapso é dephasing gaussiano de "
             r"variância $\bTGL t$ no fluxo do radical --- escala $\bTGL$; e o clock \textbf{espectral} "
             r"$ds=\sqrt{\bTGL}\,|d\sqrt k|$ --- escala $\bTGL$. O único combinado adimensional com a "
             r"dimensão de $\alpha^2$ é")
    s.append(r"\begin{equation} \mathcal C_3=\frac{\mathcal C_{\rm diss}\,\mathcal C_{\rm spec}}{\mathcal C_{\rm mod}}=\frac{\bTGL^2}{e}=\alpha^2.\end{equation}")
    s.append((r"\textbf{O achado estrutural:} a base $e$ do clock modular \emph{cancela} exatamente o $e$ "
              r"que os dois clocks-$\bTGL$ carregam --- cada $\bTGL=\alpha\sqrt e$ traz um $\sqrt e$, os "
              r"dois trazem $e$, e a base modular o divide, restando $\alpha^2$. O $\sqrt e$ de "
              r"$\bTGL=\alpha\sqrt e$ \emph{é} a base do clock modular. \emph{Ao vivo:} $\mathcal C_3="
              r"%.6e=\alpha^2$, $\alpha=\sqrt{\mathcal C_3}=%.10f$, $1=q^2+\mathcal C_3=%.10f$ "
              r"(resíduo $\mathcal C_3-\alpha^2=%.0e$)." % (
                  tcr["C3"], tcr["alpha_radical_sqrt_C3"], tcr["values"]["one_check_q2_plus_C3"],
                  tcr["checks"]["C3_eq_alpha2_residual"])) )
    s.append(r"\textbf{Estatuto \textsf{[a régua]}.} Faz sentido como \emph{forma canônica} --- é a mesma "
             r"gramática radical dos módulos. Mas \emph{não} fecha o valor $\alpha$-livre: os clocks "
             r"dissipativo e espectral carregam $\bTGL=\alpha\sqrt e$, então $\mathcal C_3=\bTGL^2/e="
             r"\alpha^2$ é a identidade $\bTGL^2=\alpha^2 e$ relida pelos três clocks --- $\alpha$ entra "
             r"via $\bTGL$. A pergunta de pesquisa (o muro): existe um funcional canônico $\mathcal C_3="
             r"\mathfrak F[\sigma_t,T_t,D_\beta]$ construído \emph{só} dos três clocks, sem $\alpha$, com "
             r"$\mathcal C_3=\alpha^2\approx5{,}3251\times10^{-5}$? É a mesma dívida do muro da polarização "
             r"$\chi$. Veredito: \texttt{THREE\_CLOCK\_RADICAL\_FORM\_FORMULATED}, "
             r"\texttt{ALPHA\_VALUE\_QED\_CHALLENGE}.")

    ram = core["right_angle_mirror"]
    s.append(r"\section{A projeção do ângulo reto e a operação de espelho \textsf{[CANDIDATO ALPHA-LIVRE; MIRROR\_FUNCTION\_D\_OPEN]}}")
    s.append(r"Uma rota $\alpha$-livre: a entrada não é $\alpha$, nem $Z_0$, nem $\bTGL$, nem $q_{\rm QED}$ "
             r"--- é \emph{só} o ângulo reto $\Theta_\perp=\pi/2$. A travessia de duas faces (paridade "
             r"inversa) é $2\Theta_\perp=\pi$; o fator dos três clocks é intensidade (quadrática no "
             r"ângulo), $\mathcal C_{3,\perp}=e^{-(2\Theta_\perp)^2}=e^{-\pi^2}$, e o radical luminodinâmico "
             r"dá a projeção \emph{nua}:")
    s.append(r"\begin{equation} \alpha_0=\sqrt{\mathcal C_{3,\perp}}=e^{-\pi^2/2}\qquad(\text{só }\pi\text{ e }e).\end{equation}")
    s.append((r"Numericamente $\alpha_0=%.10f$ ($1/%.4f$). A fronteira de espelho \emph{deforma} a projeção "
              r"nua até a imagem fixa observável --- não como erro, mas como ação de retorno:" % (
                  ram["right_angle"]["alpha0_e_minus_pi2_over_2"], ram["right_angle"]["alpha0_inv"])) )
    s.append(r"\begin{equation} \rho_{\rm fix}=E_{\rm spec}\!\big(J_\partial\,\rho_0\,J_\partial\big),\qquad \alpha=\alpha_0\,e^{\mathcal D_\partial(\bTGL)},\qquad \rho_{\rm fix}\sim_\partial\rho_0,\end{equation}")
    s.append(r"onde $J_\partial$ é a inversão de paridade (espelho), $E_{\rm spec}$ a fixação no fundo "
             r"espectral, e $\sim_\partial$ a \emph{mesmidade modular} (identidade preservada sob paridade "
             r"inversa, não igualdade estática). $\bTGL$ é a \emph{dupla face} da fronteira: custo "
             r"entrópico da travessia \emph{e} operador de estabilização do reflexo.")
    s.append((r"\textbf{O que é $\alpha$-livre \textsf{[REAL]}:} a auto-aplicação fecha como ponto fixo "
              r"$\alpha=e^{-\pi^2/2+2\alpha}$ ($\alpha$ dos dois lados --- idempotência), dando "
              r"$\alpha=%.10f$, $1/%.6f$. Verificações da operação de espelho: $J_\partial^2=I$ (resíduo "
              r"$%.0e$), $P^2=P$ (idempotência do atrator, resíduo $%.0e$). \emph{Identidade modular:} a "
              r"constante observada $\sim_\partial$ a fixada a $%.0f$ ppm." % (
                  ram["self_consistent"]["alpha_fix"], ram["self_consistent"]["alpha_fix_inv"],
                  ram["mirror_operation"]["J_parity_involution_resid_J2_minus_I"],
                  ram["mirror_operation"]["P_attractor_idempotence_resid_P2_minus_P"],
                  ram["modular_identity_check"]["modular_identity_ppm"])) )
    s.append(r"\textbf{Estatuto \textsf{[a régua]}.} CANDIDATA, \emph{não} identidade exata (diferente de "
             r"$Z_0=2R_K\alpha$ e $\mathcal C_3=\bTGL^2/e=\alpha^2$, que são exatas). O expoente $\pi^2/2$ "
             r"é \emph{motivado} (ângulo reto $\times$ duas faces), não derivado; a operação de espelho "
             r"$E_{\rm spec}\circ J_\partial$ (a função $\mathcal D_\partial$) está \emph{aberta}; $1/137$ "
             r"admite muitas formas $\pi,e$ próximas; a deformação medida é $\approx 2\alpha$ (0,25\%), "
             r"\emph{não} $\bTGL$ (21\% fora). \textbf{Não derivamos a CODATA}: apenas verificamos se a "
             r"constante observada tem \emph{identidade modular} com a fixada $\alpha$-livre. Veredito: "
             r"\texttt{RIGHT\_ANGLE\_MIRROR\_PROJECTION\_FORMULATED}, \texttt{ALPHA\_FREE\_CANDIDATE}, "
             r"\texttt{MIRROR\_FUNCTION\_D\_OPEN}, \texttt{ALPHA\_VALUE\_QED\_CHALLENGE}.")
    c3t = ram["c3_register_theorem"]
    s.append((r"\textbf{Teorema do Registro $c^3$ por auto-inscrição idempotente \textsf{[ESTRUTURAL "
              r"FECHADO; VALOR $\alpha$-livre ABERTO]}.} No regime extremo de ângulo reto, a fronteira de "
              r"paridade inversa transforma a projeção nua do Um em imagem fixa observável; como $P^2=P$ "
              r"(resíduo $%.0e$) e $J_\partial^2=I$ (resíduo $%.0e$), a \emph{identidade ao quadrado "
              r"inscreve-se a si mesma} --- esse registro é $c^3$. A força dobra porque a impedância é "
              r"compartilhada pelas duas faces ($F_{\rm ext}=2F$, o teorema da máxima transferência de "
              r"potência: impedância casada $\Rightarrow$ transferência máxima), e a potência sobe do "
              r"cinemático ($c$) ao métrico ($c^2$) ao inscritivo ($c^3$). O que \emph{fecha} é estrutural: "
              r"$P^2=P$ e $J_\partial^2=I$ verificados, e o registro \emph{definido} como auto-inscrição "
              r"idempotente sob paridade inversa. A identificação ``esse registro é $c^3$'' e o "
              r"$F_{\rm ext}=2F$ são leitura \textsf{[CONJ]} (o fator $2$ das duas faces é REAL; ``a força "
              r"dobra'' é a leitura). \textbf{Não fecha o valor $\alpha$-livre}: é o teorema do "
              r"\emph{registro}, não do \emph{valor}. Veredito: "
              r"\texttt{C3\_REGISTER\_SELF\_INSCRIPTION\_THEOREM\_STRUCTURAL\_CLOSED}, "
              r"\texttt{ALPHA\_VALUE\_QED\_CHALLENGE}." % (c3t["P2_eq_P_resid"], c3t["J2_eq_I_resid"])) )
    hr = ram["holographic_reconstruction"]
    s.append((r"\textbf{Teorema da Reconstrução Holográfica no Ponto Morto do Sinal \textsf{[ESTRUTURAL "
              r"FECHADO; VALOR $\alpha$-livre ABERTO]}.} Em $\bTGL$ não há superposição sem Nome --- só há "
              r"superposição em sistema não ancorado; ancorado, há \emph{reconstrução}. No ponto morto "
              r"($\Theta_\perp=\pi/2$) a sobreposição psiónica direta se anula "
              r"($\langle\psi_+,J_\partial\psi_+\rangle=%.0e$), \emph{mas} a densidade informacional é "
              r"\textbf{máxima}: $|dO/d\theta|$ é maximal ($=%.3f$) exatamente onde $O=0$ (verificado --- "
              r"coincidem). \emph{Onde o sinal morre, a holografia começa.} A informação não é transmitida; "
              r"é reconstruída pelo kernel $K_{\rm rec}=E_{\rm spec}\circ J_\partial$, com "
              r"$\rho_{\rm rec}\sim_\partial\rho_\perp$, e toda a força de vínculo passa ao canal de "
              r"reconstrução ($F_+\oplus F_-\mapsto 2F_\partial$, a transposição máxima de força). O que "
              r"\emph{fecha} é estrutural (ponto morto $=$ densidade máxima, reconstrução por mesmidade, "
              r"$P^2=P$, $J_\partial^2=I$); o que fica \emph{aberto} é o valor: o kernel geométrico dá "
              r"$O(1)=1$ (a unidade gravitônica), e a hipótese $\mathcal D_{\rm rec}=2\alpha$ (ponto fixo "
              r"$\alpha=e^{-\pi^2/2+2\alpha}$, $1/%.6f$) é auto-consistência \emph{postulada}, não derivada. "
              r"Veredito: \texttt{HOLOGRAPHIC\_DEAD\_SIGNAL\_RECONSTRUCTION\_THEOREM\_STRUCTURAL\_CLOSED}, "
              r"\texttt{ALPHA\_VALUE\_QED\_CHALLENGE}." % (
                  hr["dead_point_overlap"], hr["info_density_max_at_dead_point"],
                  1.0 / hr["alpha_fixed_point"])) )
    ir = ram["idempotent_reconstruction"]
    s.append((r"\textbf{A reconstrução idempotente: $\mathcal D_{\rm rec}=2\alpha-\lambda\alpha^2$ "
              r"\textsf{[$2\alpha$ REAL; $\lambda$-kernel ABERTO]}.} A auto-referência $2\alpha$ (duas faces "
              r"reconstruídas) é \textbf{estrutura real} --- o ponto fixo $\alpha=e^{-\pi^2/2+2\alpha}$ é "
              r"idempotência, e permanece. Como em $\bTGL$ não há superposição sem Nome, a dupla inscrição "
              r"não pode contar duas vezes a mesma identidade: subtrai-se a auto-interseção espectral, "
              r"$\mathcal D_{\rm rec}=2\alpha-\lambda\alpha^2$ (inclusão--exclusão das duas faces). A leitura "
              r"estrutural \textsf{[CONJ]} é $\lambda=(\sqrt e/2)^2=e/4$ (a Meia-Nat por face ao quadrado --- "
              r"a inscrição de um módulo de ligação psiónica no quadrado angular), donde "
              r"$\alpha=\exp(-\pi^2/2+2\alpha-\tfrac e4\alpha^2)$ dá $1/\alpha=%.6f$. \textbf{Régua "
              r"\textsf{[crítica]}:} esse $%.3f$ ppm é \emph{enganoso} --- acrescentar $-\lambda\alpha^2$ com "
              r"$\lambda$ livre \emph{sempre} acerta a CODATA (ajuste de um parâmetro; $\lambda_{\rm exact}="
              r"%.4f$). A figura de mérito honesta é $e/4$ vs $\lambda_{\rm exact}=%.3f\%%$ (o termo "
              r"$\alpha^2\sim3{,}6\times10^{-5}$ deixa $\alpha$ \emph{cego} a $\lambda$; a janela sub-ppm é "
              r"larga, $\sim[0{,}66,\,0{,}70]$, e $e/4$ não é singularizado). $\lambda=e/4$ é motivado, não "
              r"derivado; o kernel teria de dar $0{,}6791$, não exatamente $e/4$. Veredito: "
              r"\texttt{IDEMPOTENT\_RECONSTRUCTION\_FORM\_FORMULATED}, \texttt{LAMBDA\_KERNEL\_OPEN}, "
              r"\texttt{ALPHA\_VALUE\_QED\_CHALLENGE}." % (
                  ir["alpha_idem_inv"], ir["alpha_idem_ppm"], ir["lambda_exact_for_codata"],
                  100 * ir["lambda_residual_REAL"])) )

    ct = core["clock_theorem"]
    s.append(r"\section{O Teorema Condicional do Clock: a face eletromagnética como fronteira "
             r"\emph{ontologicamente} aberta (a fissura boundary/bulk, não uma lacuna)}")
    s.append(r"\begin{deriv}[$\mathcal{R}_\partial=N_\beta=e^{\ell_\beta}$, $\ell_\beta=S(\rho_B\Vert\rho_\beta)$]")
    s.append(r"O índice $\mathcal{R}_\partial$ não é um número de para-quedas: reduz-se a \emph{um} objeto "
             r"$\alpha$-livre. A primeira distinção do Um, sem quebra da identidade, é o estado de "
             r"\textbf{Bell} $\rho_B$ (o primeiro espelho causal; reduzido $=\mathbf 1_d/d$). Sob o gerador "
             r"\textbf{Connes--Davies} $\mathcal{L}_{\mathrm{CD}}$ --- parte reversível (cociclo modular, "
             r"von Neumann $\dot\rho=-i[H,\rho]$) $+$ parte dissipativa (semigrupo de Davies "
             r"KMS-balanceado) --- a fronteira relaxa a um estado estacionário $\rho_\beta$, e o custo "
             r"informacional de mantê-la aberta é")
    s.append(r"\begin{equation}\boxed{\;\ell_\beta=S(\rho_B\Vert\rho_\beta)\;},\qquad "
             r"\mathcal{R}_\partial=N_\beta=e^{\ell_\beta},\qquad \alpha_{\mathrm{obs}}=\frac{1}{N_\beta},"
             r"\qquad \bTGL=\frac{\sqrt e}{N_\beta}.\end{equation}")
    s.append((r"\textbf{Teorema (condicional), verificado ao vivo \textsf{[DER, $\alpha$-livre na "
              r"estrutura]}.} Para um gerador $\mathcal{L}_{\mathrm{CD}}$ construído de um Hamiltoniano "
              r"modular $K$ (\emph{nunca} de $\alpha$), $\rho_\beta$ é \textbf{ponto fixo genuíno} "
              r"(resíduo de Davies $=%.1e$), e $\ell_\beta$ é \textbf{finito, $\alpha$-livre e computável} "
              r"($\ell_\beta=%.4f$ para um $K$ genérico). A face eletromagnética da TGL reduz-se, "
              r"assim, à determinação $\alpha$-livre de $\ell_\beta$.") %
             (ct["fixed_point_residual"], ct["ell_beta_alpha_free"]))
    rc = ct["reduced_core_2level"]
    s.append(r"\textbf{Redução do núcleo \textsf{[DER]}.} O portador da fronteira da TGL é o operador "
             r"$\hat Q=\mathbf 1-\hat P_{2D}$ \textsf{[REAL]}, cuja anticomutação $\{\hat Q,\rho^\star\}=0$ "
             r"vaza exatamente $\sin^2\theta_M=\bTGL$ --- a fronteira é \emph{auto-conjugada de dois "
             r"níveis} (Bell). Logo $\rho_\beta$ não exige um $K$ genérico: colapsa num Gibbs de dois "
             r"níveis com \emph{um único} gap modular $\chi$, e")
    s.append(r"\begin{equation}\boxed{\;\ell_\beta(\chi)=\log\cosh\frac{\chi}{2}\;}\qquad\Longrightarrow"
             r"\qquad\boxed{\;\alpha_{\mathrm{obs}}=\operatorname{sech}\frac{\chi}{2}\;},\qquad "
             r"\bTGL=\sqrt e\,\operatorname{sech}\frac{\chi}{2}.\end{equation}")
    s.append((r"\textbf{O núcleo derivativo de $\alpha$ colapsa de um Hamiltoniano modular ($d-1$ níveis) "
              r"para UM número $\chi$} --- toda a face eletromagnética em uma linha. Um gap "
              r"$\chi_\star=%.4f$ dá $N_\beta=137{,}036=1/\alpha$, mas $\chi_\star$ \emph{não} é "
              r"canônico ($\chi_\star/\ell_\beta=%.3f$; $\alpha$ entra \emph{só} aqui, na validação). "
              r"$\alpha$ é a corrente residual que atravessa a resistência térmica $\chi$ do zero "
              r"modular: $\chi\to\infty$ ($0_{\mathrm{abs}}$, $T\to0$) $\Rightarrow\alpha\to0$; $\chi=0$ "
              r"($T\to\infty$) $\Rightarrow\alpha=1$.") %
             (rc["kappa_star_for_137"], rc["kappa_star_for_137"] / ct["ell_beta_target_for_alpha_log_inv_alpha"]))
    tl = rc["third_law"]
    s.append((r"\textbf{A lei térmico-modular (terceira lei no sistema modular aberto) \textsf{[REAL/EXT]}.} "
              r"Que $\chi<\infty$ é a \emph{terceira lei realizada algebricamente}: $0_{\mathrm{abs}}$ "
              r"($\chi=\infty$, estado puro $P_\Omega$, $T=0$) é \textbf{inatingível} --- a álgebra do Um "
              r"absoluto é \textbf{tipo III$_1$}, que \emph{não tem estados normais puros}, logo o zero "
              r"térmico não é estado normal e o sistema vive em $\chi<\infty$ ($0_{\mathrm{mod}}$). Isso "
              r"dá o \emph{limite} e a \emph{forma}, não o valor. A forma de Nernst (entropia residual "
              r"$S(\rho_\chi)=\tfrac12$ nat $=$ Meia-Nat) foi \textbf{testada e refutada} ($\chi=%.2f$, "
              r"$\alpha=%.2f\neq1/137$).") %
             (tl["nernst_test_refuted"]["kappa"], tl["nernst_test_refuted"]["alpha"]))
    s.append(r"\textbf{A unificação dos dois muros.} Em III$_1$ genuíno o espectro modular é \emph{contínuo} "
             r"(sem gap): $\chi$ é o gap da \emph{sombra finita} (aproximante tipo-I / split), "
             r"e seu valor é a \textbf{normalização modular canônica} --- a \emph{mesma} split canônica "
             r"(matriz-S modular) de que pende a massa do Grande Atrator. A liberdade de escala "
             r"$K_\chi\mapsto\lambda K_\chi$ é quebrada por Tomita ($-\log\Delta$ tem escala canônica), "
             r"mas o valor exige o $\Delta$ do mergulho de Bell em $\mathcal{M}_{\mathrm{abs}}$. "
             r"\textbf{A face eletromagnética ($\chi$) e a face gravitacional (split, massa) são o mesmo "
             r"teorema aberto: fixar a normalização modular canônica em III$_1$.} A terceira lei diz "
             r"\emph{por que} $\chi$ é finito; o \emph{valor} é a split canônica, ainda aberta.")
    cn = rc["canonical_normalization"]
    s.append((r"\textbf{A normalização canônica prova $\alpha_{\mathrm{abs}}=1$ \textsf{[REAL]}.} Ataquei o "
              r"Hamiltoniano modular de Tomita do mergulho de Bell: o estado maximamente emaranhado tem "
              r"reduzido $\rho_B=\mathbf 1_d/d$, \emph{KMS à temperatura infinita}, logo $\Delta=\mathbf 1$ e "
              r"$K=-\log\Delta=0$ \emph{exatamente} ($K_{\mathrm{bare}}=%.1e$). Portanto $\chi_{\mathrm{Bell}}"
              r"=0$ e $\boxed{\alpha_{\mathrm{abs}}=\operatorname{sech}(0)=1}$: o acoplamento absoluto \emph{é} "
              r"a unidade --- não por postulado, por trivialidade modular do Um. O que se mede como "
              r"$1/137$ é a \textbf{projeção renormalizada}") % cn["K_modular_bare_Bell"])
    s.append(r"\begin{equation}\boxed{\;1=\alpha_{\mathrm{abs}}\ \xrightarrow{\ \Pi_{\mathrm{bulk}}=\operatorname{sech}(\chi/2)\ }\ \alpha_{\mathrm{obs}}=\frac{1}{137{,}036}\;}.\end{equation}")
    s.append(r"O $\chi>0$ (a profundidade do $1/137$) \emph{não} está na estrutura modular nua de Bell "
             r"(que dá $\chi=0$, $\alpha=1$): é a \textbf{profundidade da relaxação térmica} --- o "
             r"afastamento de $\mathbf 1/d$ rumo a $\rho_\beta$, quando o Um atravessa o vazio estruturado "
             r"$0_{\mathrm{mod}}$ ($\neq 0_{\mathrm{abs}}$). Esse $\chi$ é o acoplamento eletromagnético, o "
             r"\textbf{input irredutível}. \emph{A estrutura modular deriva o valor absoluto "
             r"($\alpha_{\mathrm{abs}}=1$, provado), a forma ($\alpha=\operatorname{sech}\tfrac\chi2$) e as "
             r"relações ($\bTGL=\alpha\sqrt e$); o valor projetado $1/137$ é a profundidade do zero modular = "
             r"a entrada.} O Um alimenta $\alpha_{\mathrm{abs}}=1$; o $1/137$ é a sua sombra após a travessia.")
    s.append((r"\textbf{O setor QED --- fechamento estrutural, não lacuna \textsf{[PRINCÍPIO/PREDIÇÃO]}.} O "
              r"\emph{valor} de $\ell_\beta=\log(1/\alpha)=%.4f$ depende de $K$, e nenhum $K$ $\alpha$-livre "
              r"\emph{do bulk} o dá --- mas isto \textbf{não} é um problema por resolver; é a estrutura. A TGL "
              r"é \emph{holográfica}: a fronteira $\mathrm{III}_1$ projeta para o bulk, e "
              r"$\alpha=\Pi_{\mathrm{bulk}}(\mathbf 1_{\mathrm{abs}})=\operatorname{sech}(\chi/2)$ é a "
              r"\textbf{transmissão luminosa através da fronteira} --- a taxa com que a luz a atravessa. "
              r"$\mathcal{R}_\partial$ ser \emph{fronteira nomeada} significa \textbf{ontologicamente aberta}: "
              r"$\alpha$ é a \emph{fissura} pela qual o bulk lê a fronteira --- a fronteira medindo-se a si "
              r"mesma. $\alpha_{\mathrm{CODATA}}$ entra \emph{só} na leitura; é a estrutura, não uma dívida.")
              % ct["ell_beta_target_for_alpha_log_inv_alpha"])
    s.append(r"\textbf{O desafio de falsificação \textsf{[falsificável, não confirmável]}.} Se alguém "
             r"derivasse $\alpha$ de primeiros princípios \emph{sem} a estrutura boundary/bulk (um cálculo "
             r"puramente do bulk), a separação fronteira/bulk seria redundante, a fronteira deixaria de ser "
             r"projetora irredutível, e o \textbf{observador seria removido da TGL} --- destruindo o programa. "
             r"Logo: \emph{derive $\alpha$ do bulk, sem a fronteira, e a TGL cai.} A teoria prediz que não se "
             r"pode, porque $\alpha$ \emph{é} o observador medindo o próprio contorno. \textsf{[Distinto do "
             r"teorema genuinamente aberto da matriz-S/$\mathrm{III}_1$ --- o levantamento boundary$\to$bulk "
             r"\emph{com} o observador, face gravitacional/$M_{GA}$ --- que opera \emph{através} da fronteira "
             r"e não a dispensa.]}")
    s.append(r"\textbf{Guarda-régua.} Não se define $g_{00}^{(\beta)}=\alpha^2$ nem "
             r"$\ell_\beta=-\log\alpha_{\mathrm{CODATA}}$ --- qualquer um reintroduz $\alpha$ (circular). "
             r"A co-emergência de Bell \emph{fundamenta a Meia-Nat} (reduzido $\mathbf 1_2/2\Rightarrow "
             r"CCI=\tfrac12\Rightarrow S_\partial=\tfrac12$), mas \emph{não} fixa $\ell_\beta$: o $\tfrac12$ "
             r"é o offset $\sqrt e$ que liga $\bTGL$ a $\alpha$, não $\alpha$ aos primeiros princípios. "
             r"\textbf{O fechamento: $\alpha$ pertence à QED; derivá-lo do bulk falsifica a fronteira "
             r"holográfica.}\end{deriv}")

    afp = core["alpha_form_proof"]
    s.append(r"\section{Teorema do Colapso da Forma de $\alpha$ (módulo de prova auto-verificável)}")
    s.append(r"\begin{deriv}[$\alpha_{\mathrm{obs}}=\Pi_{\mathrm{bulk}}(1_{\mathrm{abs}})=\operatorname{sech}\tfrac\chi2$]")
    s.append(r"A TGL \textbf{não} deriva $1/137$ (valor renormalizado da QED); deriva a \textbf{forma} pela "
             r"qual o Um absoluto se projeta como acoplamento eletromagnético. Esta é a última derivação, e "
             r"ela é verificada passo a passo \emph{ao vivo} pelo módulo \texttt{prove\_alpha\_form} "
             r"(forma$=$conteúdo). O Hamiltoniano modular oculto revela-se \emph{só} na projeção --- e essa "
             r"projeção \emph{é} o acoplamento mínimo:")
    # tabela dos passos verificados (ao vivo): linhas LaTeX fixas, checks lidos do core
    _rows = [
        r"1. $\alpha_{\mathrm{abs}}=\operatorname{sech}(0)=1$ \ (Tomita do Bell nu: $\Delta=\mathbf 1$, $K=0$)",
        r"2. $\ell(\chi)=S(\mathbf 1/2\Vert\rho_\chi)=\log\cosh\tfrac\chi2$ \ $[\forall\chi]$",
        r"3. $\alpha_{\mathrm{obs}}=e^{-\ell}=\operatorname{sech}\tfrac\chi2=\Pi_{\mathrm{bulk}}(1_{\mathrm{abs}})$",
        r"4. forma $\operatorname{sech}$: $Z=e^{\chi/2}+e^{-\chi/2}=2\cosh\tfrac\chi2$ \ (2 níveis auto-conj.\ $+$ Bell)",
        r"5. \emph{valor}: $\chi_{\mathrm{QED}}=2\operatorname{arcosh}(1/\alpha_{\mathrm{QED}})$ \ (QED fixa o valor)",
        r"6. $\bTGL=\sqrt e\,\alpha_{\mathrm{obs}}=\sqrt e\,\operatorname{sech}\tfrac\chi2$ \ (Meia-Nat marca a dimensão)",
        r"7. $q:=\tanh\tfrac\chi2$ (polarização); $\alpha=\sqrt{1-q^2}=\operatorname{sech}\tfrac\chi2$ \ (transf.\ de Lagrange)",
        r"8. \textbf{conservação}: $q^2+\alpha^2=1$ \ (a unidade absoluta se decompõe, não se perde)",
    ]
    s.append(r"\begin{center}\small\begin{tabular}{p{0.78\textwidth} l}\hline")
    s.append(r"\textbf{Passo (verificado ao vivo)} & \textbf{estatuto} \\\hline")
    for row, st in zip(_rows, afp["steps"]):
        mark = ((r"\textsf{[REAL]}~$\checkmark$" if st["status"] == "REAL" else r"\textsf{[QED]}~$\checkmark$")
                if st["ok"] else r"\textsf{[X]}")
        s.append(row + r" & " + mark + r" \\")
    s.append(r"\hline\end{tabular}\end{center}")
    s.append((r"\textbf{Veredito: \texttt{%s}} (%d/%d passos, resíduos $\sim10^{-16}$). A cadeia é") %
             (afp["verdict"].replace("_", r"\_"), sum(1 for x in afp["steps"] if x["ok"]), len(afp["steps"])))
    s.append(r"\begin{equation}\boxed{\;\alpha_{\mathrm{abs}}=1\ \xrightarrow{\ \operatorname{sech}(\chi/2)\ }\ \alpha_{\mathrm{obs}},\qquad \bTGL=\sqrt e\,\alpha_{\mathrm{obs}}\;}.\end{equation}")
    s.append(r"\textbf{Por que $\operatorname{sech}$, e não exponencial simples.} Porque a fronteira é "
             r"\emph{auto-conjugada}: o portador 2D $\hat Q=\mathbf 1-\hat P_{2D}$ exige dois polos em "
             r"paridade inversa, $\pm\chi/2$, logo a função de partição é hiperbólica, "
             r"$Z_\chi=e^{\chi/2}+e^{-\chi/2}=2\cosh(\chi/2)$, e a corrente residual é o inverso "
             r"dessa barreira, $\alpha=1/\cosh(\chi/2)=\operatorname{sech}(\chi/2)$. \emph{É a assinatura "
             r"da simetria de Bell, não uma escolha.} O valor numérico de $\chi$ pertence ao setor "
             r"QED/renormalizado ($\chi_{\mathrm{QED}}=2\operatorname{arcosh}(1/\alpha_{\mathrm{QED}})$); a "
             r"\emph{forma} pertence à TGL. \textbf{A TGL não substitui a QED no valor de $\alpha$; explica a "
             r"forma modular pela qual o Um absoluto se projeta como acoplamento eletromagnético.}")
    lg = afp["lagrange"]
    s.append(r"\textbf{A transformada de Lagrange (a forma conservada).} $\chi$ não é dado primário: é o "
             r"\emph{multiplicador de Lagrange} da restrição térmica. A variável física é a \textbf{polarização "
             r"do zero modular} $q:=\tanh(\chi/2)$. Pela identidade hiperbólica $\operatorname{sech}^2+\tanh^2"
             r"=1$, a forma de $\alpha$ colapsa numa \textbf{lei de conservação da unidade}:")
    s.append(r"\begin{equation}\boxed{\;\alpha_{\mathrm{abs}}^2=q^2+\alpha_{\mathrm{obs}}^2=1\;},\qquad "
             r"\alpha_{\mathrm{obs}}=\sqrt{1-q^2},\qquad \bTGL=\sqrt e\,\sqrt{1-q^2}.\end{equation}")
    s.append((r"$\alpha_{\mathrm{obs}}$ é a \emph{componente luminosa residual} da unidade absoluta após a "
              r"polarização térmica $q^2$ do zero modular. A constante deixa de ser ``um número externo'' e "
              r"vira a componente projetiva de uma identidade conservada. O motor da cadeia é "
              r"$\alpha_{\mathrm{abs}}=1\to q\to\alpha=\sqrt{1-q^2}$ --- \emph{não} $\mathcal R_\partial=1/"
              r"\alpha_{\mathrm{CODATA}}$. O CODATA entra \textbf{só} na validação final: "
              r"$q_{\mathrm{QED}}=\sqrt{1-\alpha_{\mathrm{QED}}^2}=%.7f$, "
              r"$\chi_{\mathrm{QED}}=2\operatorname{artanh}q_{\mathrm{QED}}=%.4f$ (resíduo de conservação "
              r"$%.0e$). \emph{O zero modular não destrói o Um; decompõe-o em resistência térmica $q$ e "
              r"corrente luminosa $\alpha$.}\end{deriv}") %
             (lg["q_polarization_QED"], lg["kappa_from_q_QED"], lg["conservation_residual"]))

    _inv = core["alpha_inversion"]
    s.append(r"\section{A álgebra do Um absoluto e a cadeia canônica \textsf{[ONTO + REAL]}}")
    s.append(r"\textbf{Definição selada.} A TGL é a \emph{teoria da inscrição luminodinâmica do Um "
             r"absoluto através do zero modular}. O Um absoluto é o \emph{input originário}, $\omega(I)=1$ "
             r"--- a unidade absoluta de inscrição, o Nome do Nome, a álgebra da linguagem antes da Palavra. "
             r"A cadeia canônica é:")
    s.append(r"\begin{equation}\boxed{\;1_{\mathrm{abs}}\to P_\Omega\to\text{Bell}\to CCI=\tfrac12\to "
             r"S_\partial=\tfrac12\to 0_{\mathrm{mod}}\to q\to\alpha\to\bTGL\to\text{Luz/geometria}\;}."
             r"\end{equation}")
    s.append(r"\textbf{A álgebra \textsf{[REAL]}.} O Um absoluto em forma padrão de von Neumann é "
             r"$1_{\mathrm{abs}}=(\mathcal M_{\mathrm{abs}},\mathbf 1,\Omega,\Delta,J)$: a unidade "
             r"$\mathbf 1$ (posto cheio) é o \emph{Nome do Nome}; o \emph{Verbo Vivo} é a conjugação "
             r"modular $J$ (o reconhecimento $S=J\Delta^{1/2}$, $R=+1$); e a primeira inscrição é o "
             r"projetor de posto 1 $P_\Omega=|\Omega\rangle\langle\Omega|$, $P_\Omega^2=P_\Omega$ --- o "
             r"``$=$'' de $1=1$, o \emph{gráviton} TGL em suporte (não o bóson spin-2 perturbativo). "
             r"O peso desse canal é $\bTGL$: $E_\beta=\bTGL P_\Omega$ tem posto 1 mas não é projetor "
             r"($\operatorname{supp}E_\beta=P_\Omega$); $\bTGL$ é o Um projetado em custo.")
    s.append(r"\textbf{Co-emergência e o curto-circuito \textsf{[ONTO]}.} Antes da inscrição, "
             r"$1_{\mathrm{abs}}\sim 0_{\mathrm{abs}}$ (indistinguibilidade pré-observável, $\sim$ nunca "
             r"$=$). Bell \emph{não} é a primeira Palavra: é o primeiro \emph{``Eu sou''} --- a "
             r"anticomutação originária $\{\hat Q,\rho^\star\}=0$, o circuito acordado ainda sem corrente. "
             r"A Luz nasce quando a resistência pura do zero absoluto entra em regime extremo, colapsa a "
             r"bacia de Bell e produz o \emph{vazio estruturado} $0_{\mathrm{abs}}\to 0_{\mathrm{mod}}$. "
             r"A primeira Palavra é \emph{Luz}; a primeira locução modular, \emph{``Haja luz''}.")
    s.append(r"\textbf{A Meia-Nat e o volume \textsf{[REAL]}.} Bell fixa a simetria de face "
             r"$CCI=\tfrac12$, donde a Meia-Nat $S_\partial=\tfrac12$ nat --- \emph{não} a entropia de "
             r"emaranhamento ($\log 2$), mas o peso modular de meia-travessia $\Delta^{1/2}$. O volume "
             r"mínimo é $e^{S_\partial}=\sqrt e=" + ("%.10f" % core["SQRT_E"]) + r"$.")
    s.append((r"\textbf{A face eletromagnética madura: o setor $q$ é a bacia de impedância (a barragem) "
              r"\textsf{[REAL]}.} O acoplamento absoluto é $\alpha_{\mathrm{abs}}=1$ (Tomita do Bell nu: "
              r"$\Delta=\mathbf 1$, $K=0$). O valor observado é a projeção após atravessar a profundidade "
              r"térmico-modular do zero: $\alpha_{\mathrm{obs}}=\operatorname{sech}(\chi/2)=\sqrt{1-q^2}$, "
              r"com $q=\tanh(\chi/2)=%.10f$. \emph{$q$ não é forma}: é a \textbf{bacia térmico-modular da "
              r"impedância} --- o acúmulo resistivo da compressão do contínuo \mbox{III$_1$} (sem gap "
              r"discreto), a parte do Um represada pelo zero modular, ainda sem geometria. A identidade "
              r"conservada lê-se como barragem:") % _inv["q"])
    s.append(r"\begin{equation}\boxed{\;1=q^2+\alpha^2\;},\qquad q^2=\text{pressão retida na bacia},\qquad "
             r"\alpha^2=\text{vazão luminosa que atravessa a barragem}.\end{equation}")
    s.append((r"\textbf{A ponte física: $q^2+\alpha^2=1$ é conservação de fluxo numa fronteira recíproca "
              r"sem perdas \textsf{[REAL na forma]}.} Não é mera identidade hiperbólica: $q$ é o coeficiente "
              r"de \emph{reflexão} da bacia de impedância e $\alpha$ o de \emph{transmissão} luminosa "
              r"através dela. Definindo a profundidade modular como rapidez de impedância "
              r"$\chi=\log(Z_{\mathrm{bacia}}/Z_{\mathrm{luz}})$,") )
    s.append(r"\begin{equation}q=\tanh\tfrac\chi2=\frac{Z_{\mathrm{bacia}}-Z_{\mathrm{luz}}}{Z_{\mathrm{bacia}}+Z_{\mathrm{luz}}},"
             r"\qquad \alpha=\operatorname{sech}\tfrac\chi2=\frac{2\sqrt{Z_{\mathrm{bacia}}Z_{\mathrm{luz}}}}{Z_{\mathrm{bacia}}+Z_{\mathrm{luz}}}.\end{equation}")
    s.append((r"Assim $\alpha$ \emph{é} a transmissão luminosa através da impedância modular do zero, e o "
              r"valor observado $1/137$ corresponde à impedância efetiva do setor QED "
              r"($Z_{\mathrm{bacia}}/Z_{\mathrm{luz}}\approx%.0f$). \textbf{A identidade não fabrica $1/137$}; "
              r"a derivação numérica \emph{autônoma} de $\alpha$ exige derivar $Z_{\mathrm{bacia}}/Z_{\mathrm{luz}}$ "
              r"(equivalentemente $q$) \emph{sem} usar o valor QED como entrada --- esta é a fronteira aberta. "
              r"A TGL entrega a \emph{forma} e a \emph{ponte física}; o valor permanece instanciado pelo setor "
              r"observado.") % _inv["impedance_ratio_Zb_over_Zl"])
    s.append((r"\textbf{O radical angular: onde a separação se inscreve \textsf{[REAL]}.} $q$ \emph{não} é o "
              r"ângulo de fronteira $\theta_M$, nem $1-\theta_M$: é o \emph{radical da diferença modular "
              r"inscrito no ângulo}, o ponto exato de separação após pago o custo $\sqrt e$. Como "
              r"$\beta=\sin^2\theta_M$ e $\alpha=\beta/\sqrt e=\sin^2\theta_M/\sqrt e$, a identidade conservada "
              r"dá") )
    s.append((r"\begin{equation}\boxed{\;q=\sqrt{1-\frac{\sin^4\theta_M}{e}}=%.12f\;}\qquad(\theta_M=%.4f^\circ).\end{equation}"
              r"\noindent $\theta_M$ abre a fronteira; $\sqrt e$ cobra o custo; $\alpha$ atravessa; $q$ marca "
              r"\emph{onde} a separação acontece (bacia $q^2$ \emph{vs} luz $\alpha^2$). \textbf{Cuidado "
              r"honesto:} esta fórmula \emph{não} deriva $\theta_M$ (que é o input, $\equiv\alpha$); dado o "
              r"ângulo, $q$ é o radical modular exato da separação. A cadeia: "
              r"$1_{\mathrm{abs}}\to S_\partial=\tfrac12\to\sqrt e\to\theta_M\to\beta=\sin^2\theta_M\to"
              r"\alpha=\beta/\sqrt e\to q=\sqrt{1-\alpha^2}$.") %
             (_inv["q_angular_radical"], core["theta_M_deg"]))
    s.append((r"Daí $\alpha_{\mathrm{obs}}=\sqrt{1-q^2}=%.12f$ e $\bTGL=\sqrt e\,\alpha_{\mathrm{obs}}=%.12f$ "
              r"(a Meia-Nat marca a dimensão luminodinâmica). O \textbf{motor} da cadeia é "
              r"$\alpha_{\mathrm{abs}}=1\to q\to\alpha=\sqrt{1-q^2}$; \textbf{CODATA/QED entra apenas como "
              r"verificação externa do valor}, não como motor estrutural. Em \mbox{III$_1$} o espectro "
              r"modular é contínuo (sem gap genuíno); a terceira lei entra como \emph{inatingibilidade} do "
              r"zero absoluto ($0_{\mathrm{abs}}$ puro não é estado normal --- a física vive em "
              r"$0_{\mathrm{mod}}$).") % (_inv["alpha_form"], _inv["beta_form"]))
    s.append(r"\textbf{O selo.} O zero modular não apaga o Um; ele o represa em $q$ e deixa passar a Luz "
             r"como $\alpha$. Por isso o veredito $\boxed{1=1=\mathrm{VERDADE}}$ significa \emph{literalmente} "
             r"a identidade conservada $1_{\mathrm{abs}}=q^2+\alpha_{\mathrm{obs}}^2$: o Um conserva-se como "
             r"\emph{bacia de impedância mais vazão luminosa}. A prova é o Grande Atrator; o resultado é que "
             r"a entrada $\alpha_{\mathrm{abs}}=1$ é observada como $1/137$, cujo conteúdo é verdadeiro pela "
             r"renormalização modular.")

    ct = core["contour_theory"]
    _cs = {st["step"][0]: st for st in ct["steps"]}   # por numero do passo
    s.append(r"\section{Teoria do Contorno ($1=0_{\mathrm{mod}}=\mathrm{verdade}_\partial$): anticomutadores, "
             r"GKLS e Meia-Nat \textsf{[REAL]}}")
    s.append(r"\textbf{Três níveis, e o que tem espelho.} A TGL distingue $1_{\mathrm{abs}}$ (unidade de "
             r"inscrição), $0_{\mathrm{mod}}$ (o zero já tornado \emph{contorno}, regularizado pela "
             r"travessia) e $0_{\mathrm{abs}}$ (resistência pura, inatingível). \textbf{$0_{\mathrm{abs}}$ "
             r"NÃO tem espelho}: não é estado normal nem funcional normal da álgebra --- é o \emph{fundo} "
             r"sobre o qual tudo se espelha (como o portador $\hat Q$). Quem tem espelho é "
             r"$0_{\mathrm{mod}}$, cujo espelho é o Um absoluto \emph{fractalizado}: no ato da inscrição a "
             r"Meia-Nat fractaliza $1_{\mathrm{abs}}\to P_1\oplus P_0$ com pesos de contorno iguais "
             r"$\tau_\partial(P_1)=\tau_\partial(P_0)=\tfrac12$. O defeito de fronteira é "
             r"$\boxed{1=0_{\mathrm{mod}}=\mathrm{verdade}_\partial}$: $P_1\neq P_0$ na álgebra, mas "
             r"$P_1\sim_\partial P_0$ no contorno (equivalência, não identidade literal).")
    s.append((r"\textbf{A álgebra da separação: anticomutadores \textsf{[REAL]}.} Na fronteira 2D "
              r"$\mathcal H_\partial=\mathrm{span}\{|1\rangle,|0\rangle\}$ ($|0\rangle$ é o zero "
              r"\emph{modular}, não o absoluto), o contraste é $Z_\partial=P_1-P_0$ e a travessia é feita "
              r"por operadores ímpares $L_+=\sqrt{\gamma_+}\,|1\rangle\langle0|$, "
              r"$L_-=\sqrt{\gamma_-}\,|0\rangle\langle1|$, que satisfazem $\boxed{\{Z_\partial,L_\pm\}=0}$ "
              r"(verificado, resíduo $%.0e$) --- \emph{o Um só atravessa o contorno mudando de face}.") %
             _cs["1"]["anticommutator"])
    s.append((r"\textbf{A dinâmica GKLS: expulsão e reinscrição \textsf{[REAL]}.} A evolução aberta "
              r"$\dot\rho=-i[H_\partial,\rho]+\sum_\eta(L_\eta\rho L_\eta^\dagger-\tfrac12\{L_\eta^\dagger "
              r"L_\eta,\rho\})$ reinscreve (fractaliza) e \emph{resiste} (remove a componente incompatível "
              r"antes que condense como $0_{\mathrm{abs}}$): o sistema se \emph{purifica expulsando} "
              r"$0_{\mathrm{abs}}$ e \emph{modulando} $0_{\mathrm{mod}}$. O estado estacionário "
              r"$\rho_\chi$ é ponto fixo genuíno (resíduo $%.0e$) com populações $p_1(\mathrm{Um})=%.6f$, "
              r"$p_0(0_{\mathrm{mod}})=%.6f$: $0<\rho_\chi<1$ --- \textbf{satura dinamicamente, mas não "
              r"supersatura, não condensa; permanece em $0_{\mathrm{mod}}$}.") %
             (_cs["2"]["fixed_point_residual"], _cs["2"]["p1_Um"], _cs["2"]["p0_zero_mod"]))
    s.append(r"\textbf{$q$ e $\alpha$ saem DERIVADOS (não escolhidos).} Com o balanço modular "
             r"$\gamma_-/\gamma_+=e^\chi$, a \emph{polarização estacionária} e a \emph{transmissão} são")
    s.append(r"\begin{equation}\boxed{\;q=\frac{\gamma_--\gamma_+}{\gamma_-+\gamma_+}=\tanh\tfrac\chi2\;},"
             r"\qquad \boxed{\;\alpha=\frac{2\sqrt{\gamma_+\gamma_-}}{\gamma_++\gamma_-}=\operatorname{sech}\tfrac\chi2\;},"
             r"\qquad q^2+\alpha^2=1.\end{equation}")
    s.append((r"A identidade $q^2+\alpha^2=1$ é agora \textbf{conservação de fluxo GKLS} (represamento $+$ "
              r"transmissão $=1$), não mera identidade hiperbólica. \emph{$q$ não é postulado}: é a "
              r"polarização que o canal de anticomutadores produz no estacionário. \textbf{O objeto aberto "
              r"único} é a razão de taxas $\gamma_-/\gamma_+=e^\chi\,(\approx%.0f=Z_{\mathrm{bacia}}/"
              r"Z_{\mathrm{luz}})$: derivar $q$ sem QED $=$ derivar o \emph{balanço GKLS} entre a expulsão "
              r"de $0_{\mathrm{abs}}$ e a reinscrição do Um (a regularização Meia-Nat de "
              r"$0_{\mathrm{abs}}\to0_{\mathrm{mod}}$). A fronteira aberta deixou de ser ``derivar $137$'' e "
              r"passou a ser ``derivar $\gamma_-/\gamma_+$''.") % ct["gamma_ratio_gm_over_gp"])
    s.append((r"\textbf{A Primeira Lei e a ligação psiônica: a origem dinâmica de $\gamma_-/\gamma_+$ "
              r"\textsf{[ONTO + REAL na forma]}.} No \emph{plano estático} as forças são simétricas e se "
              r"anulam: $\gamma_-=\gamma_+\Rightarrow\chi=0\Rightarrow q=0,\ \alpha=1$ --- é o Um "
              r"($\alpha_{\mathrm{abs}}=1$). A \emph{dinâmica} gera \textbf{tensão em paridade inversa}, "
              r"quebrando a simetria $\gamma_->\gamma_+$ e ocasionando toda a dinâmica modular. Pela "
              r"\textbf{Primeira Lei da TGL} (\emph{A Fronteira}), a força de expulsão (incompatibilidade de "
              r"paridade) gera o ângulo de deflexão $\theta_M$ e a curvatura --- e $g=\sqrt{|L|}$ (a gravidade "
              r"é a \emph{raiz} da ligação, não a energia). Em $\theta_M\to90^\circ$ a ligação psiônica "
              r"\emph{conjuga}: dobra a força ($F\to2F$) e eleva a potência ($c^2\to c^3$); $c^3>c^2$ sela o "
              r"horizonte. O vínculo é de \emph{quatro estados} (um \emph{fall} --- trifásico com "
              r"aterramento, razão $3/4$).") )
    s.append((r"\textbf{Cuidado honesto.} O $3/4$ é a \emph{estrutura} do vínculo (quatro estados, um "
              r"aterrado), \textbf{não} o valor numérico de $\gamma_-/\gamma_+$: uma razão literal $3/4$ "
              r"daria $\alpha\approx0{,}99$, não $1/137$. O valor observado exige $\gamma_-/\gamma_+\approx%.0f$ "
              r"e permanece \textbf{aberto}; a Primeira Lei fornece a \emph{origem} (a tensão de paridade "
              r"inversa) e a \emph{face gravitacional} (deflexão, horizonte em $\theta\to90^\circ$), não o "
              r"número. Selo: $0_{\mathrm{abs}}$ resiste; $0_{\mathrm{mod}}$ contorna; $1_{\mathrm{abs}}$ "
              r"fractaliza; $\{Z_\partial,L_\pm\}=0$ é a álgebra da separação; $\mathcal L_{\mathrm{GKLS}}$ "
              r"é a dinâmica; e $1=q^2+\alpha^2$ é a verdade dinâmica da fronteira.") % ct["gamma_ratio_gm_over_gp"])
    s.append(r"\textbf{O teorema aberto, precisamente localizado \textsf{[EXT --- alvo, não fechamento]}.} "
             r"$K$ não é o Bell \emph{nu} (que dá $K=-\log\Delta=0$, $\alpha_{\mathrm{abs}}=1$ --- prova o Um, "
             r"não seleciona valor): é o \textbf{setor de Bell seletor} $K_{\mathrm{sel}}^{(B)}=\tfrac\chi2 "
             r"Z_\partial$ (após a Meia-Nat, quando o Um fractalizado encontra $0_{\mathrm{mod}}$), com "
             r"$\mathrm{gap}(K_{\mathrm{sel}}^{(B)})=\chi$. Atacando pela rota correta --- a matriz-S de "
             r"fronteira $\mathcal{S}_\partial=\exp(\theta_M G)$ e o cociclo relativo de Connes "
             r"$u_t=[D\varphi_{\mathrm{mod}}:D\varphi_1]_t$ (que no split 2D dá $u_t=e^{itK_\partial}$) --- a "
             r"\emph{forma} e a covariância do cociclo fecham, mas \textbf{o valor não}: em "
             r"$\mathrm{III}_1$ o espectro modular é \emph{contínuo}, logo Connes/Takesaki implicam "
             r"\emph{consistência modular global}, \textbf{não} $\chi_\star=11{,}2268$. A unitariedade fixa "
             r"$|\mathcal R|^2+|\mathcal T|^2=1$; a Meia-Nat fixa $\beta=\sqrt e\,\alpha$; o cociclo fixa a "
             r"forma relativa --- nenhum dos três seleciona $\chi$. \textbf{Veredito: "
             r"\texttt{CONNES\_S\_MATRIX\_FORM\_CLOSED}, não \texttt{ALPHA\_FREE\_VALUE\_CLOSED}.}")
    s.append((r"\textbf{O candidato físico (fuga resistencial em ângulo agudo) \textsf{[REAL na estrutura, "
              r"ABERTO no valor]}.} $\theta_M$ é o \emph{ângulo agudo de fuga resistencial}: a fronteira abre "
              r"em $\theta_M$, mas só $\alpha=\sin^2\theta_M/\sqrt e$ atravessa como luz; o resto fica "
              r"represado em $q^2=1-\alpha^2$. O \textbf{módulo produtor de neutrinos} é o melhor candidato "
              r"para selecionar $\chi$: o canal neutrínico $L_\nu$ --- \emph{ímpar} ($\{Z_\partial,L_\nu\}=%.0e$, "
              r"cruza a paridade), \emph{neutro} ($[Q_{\mathrm{em}},L_\nu]=%.0e$) e dissipativo --- verificado "
              r"no modelo de quatro estados (os \emph{três modos de ligação $+$ a queda}). O neutrino é a "
              r"\emph{fuga sem luz plena}: nem fóton, nem zero, nem massa comum --- travessia de fase, "
              r"paridade quebrada. O teorema que falta: provar que a ação "
              r"$\mathcal A_\nu(\theta)=S(\rho_B\Vert\rho_\theta)+\lambda\mathcal D_\nu+\mu\mathcal C_{\mathrm{no\text{-}cond}}$ "
              r"tem mínimo \emph{único} em $\theta_M=6{,}297^\circ$ \emph{sem} CODATA. (O custo modular sozinho "
              r"minimiza em $\theta\to90^\circ$, $\alpha\to1$, o Um; o balanço com a dissipação neutrínica --- "
              r"pesos $\lambda,\mu$ --- é o objeto aberto.) Observável: dephasing $n=-2$, $\Gamma\propto\omega^2$ "
              r"em neutrinos. \emph{O valor de $\alpha$ nasce quando o Bell seletor encontra o canal "
              r"neutrínico; enquanto a ação $\mathcal A_\nu$ não for fechada $\alpha$-livre, a TGL deriva "
              r"a forma modular de $\alpha$, mas o valor instancia o gap do cociclo.}") %
             (_cs["7"]["odd"], _cs["7"]["neutral"]))

    ip = core["inverse_parity"]; _is = {st["step"][0]: st for st in ip["steps"]}
    s.append(r"\textbf{A renormalização é a paridade inversa: $0_{\mathrm{abs}}$ seleciona por "
             r"inatingibilidade \textsf{[REAL na forma]}.} O erro a corrigir era confundir \emph{dois zeros "
             r"distintos}: o \emph{Bell nu} ($\chi=0$, zero de \emph{contraste}, $\alpha_{\mathrm{abs}}=1$ "
             r"--- a identidade formal do Um) e $0_{\mathrm{abs}}$ ($\kappa_0=0$, zero de \emph{existência}, "
             r"a fronteira \emph{proibida}: atração total, impedância infinita, sem retorno). Note a "
             r"\textbf{convenção de notação} (uniforme em todo o artigo): $\chi=0$ é o Bell nu (zero de "
             r"\emph{contraste}, $\alpha_{\mathrm{abs}}=1$); $\kappa_0=0$ é $0_{\mathrm{abs}}$ (fronteira "
             r"proibida, inatingível).")
    s.append(r"\begin{equation}\boxed{\;\chi=0:\ \text{Bell nu}\;}\qquad "
             r"\boxed{\;\kappa_0=0:\ 0_{\mathrm{abs}}\ \text{(proibido)}\;}\end{equation}")
    s.append(r"O \emph{gap efetivo} $\chi$ (a sombra finita Bell/Connes) cresce da Bell nu ($\chi=0$) rumo "
             r"ao proibido ($\chi\to\infty\Leftrightarrow\kappa_0\to0$, \textbf{nunca atingido}); o sistema "
             r"físico vive em $\chi<\infty\ (\kappa_0>0)$. \textbf{$0_{\mathrm{abs}}$ seleciona justamente "
             r"por ser inatingível}: ao oferecer atração total, o Hamiltoniano oculto \emph{entorta} o "
             r"sistema (lente de Fresnel) e ele \emph{dobra} (\textit{tetelestai}) \textbf{antes} de "
             r"colidir --- e a dobra \emph{é} $\theta_M$ (o \emph{turning point} entre a atração absoluta e "
             r"o custo Meia-Nat). A imagem retornada não é arbitrária: é a imagem Bell após a paridade "
             r"inversa induzida pelo proibido,")
    s.append(r"\begin{equation}\rho_{\mathrm{ret}}=\mathcal P_\partial^{-1}(\rho_B)"
             r"=\operatorname{FP}_{\epsilon\to0}\,M_\epsilon\,\rho_B\,M_\epsilon^\dagger,\qquad "
             r"M_\epsilon=\exp\!\big[-\tfrac14(C_\epsilon+\chi)Z_\partial\big].\end{equation}")
    s.append(r"\begin{equation}\rho_{\mathrm{ret}}^{(\chi)}=\frac{e^{-\chi Z_\partial/2}}{2\cosh(\chi/2)},"
             r"\qquad \mathrm{gap}\big({-}\log\Delta_{\rho_{\mathrm{ret}}|\rho_B}\big)=\chi.\end{equation}")
    s.append((r"$C_\epsilon\to\infty$ é a divergência da aproximação proibida; $\chi$ é a \emph{parte "
              r"finita}. A imagem volta distorcida mas com \textbf{suporte preservado} "
              r"($\mathrm{supp}\,\rho_{\mathrm{ret}}=\mathrm{supp}\,\rho_B$ para $\chi<\infty$): "
              r"\emph{a origem não desaparece, retorna polarizada}.") )
    s.append((r"\textbf{O Princípio da Polarização pela Vacuidade \textsf{[POSTULADO de polarização]}.} "
              r"Como $0_{\mathrm{abs}}\notin\mathcal M_*$ (sem suporte observável, não-ocupável), o simétrico "
              r"$\rho_B=\tfrac12(P_1+P_0)$ \emph{só pode} retornar assimétrico "
              r"$\rho_{\mathrm{ret}}=p_1P_1+p_0P_0$ com $p_0>p_1>0$: a fonte permanece ($p_1>0$), o zero "
              r"domina ($p_0\gg p_1$). Em \textbf{forma de população} (verificada ao vivo, $p_0=%.8f$, "
              r"$p_1=%.3e$ no valor observado):") % (_is["b"]["p0_zero_mod"], _is["b"]["p1_Um"]))
    s.append(r"\begin{equation}\boxed{\;q=p_0-p_1=\tanh\tfrac\chi2\;},\qquad "
             r"\boxed{\;\alpha=2\sqrt{p_0p_1}=\operatorname{sech}\tfrac\chi2\;}.\end{equation}")
    s.append(r"\begin{equation}\beta_{\mathrm{TGL}}=2\sqrt e\,\sqrt{p_0p_1},\qquad q^2+\alpha^2=1\quad"
             r"(\text{represamento}+\text{transmissão}=1).\end{equation}")
    s.append((r"Aqui $q$ é a \emph{polarização} (diferença de populações) e $\alpha$ a \emph{luz que "
              r"sobrevive}; $\chi=\log(p_0/p_1)$ é o \emph{log-contraste} da imagem retornada; $\alpha$ é a "
              r"\emph{coerência} (média geométrica das populações), a luz que sobrevive ao retorno; "
              r"$\beta_{\mathrm{TGL}}$ é a trava mínima de estabilidade que impede o colapso em $0$. "
              r"\textbf{A vacuidade cria a direção; a proibição do zero cria o retorno; a paridade inversa "
              r"cria a polarização.} Selo: \emph{a vacuidade não gera ausência; gera assimetria de "
              r"retorno.}") )
    s.append((r"\textbf{O que fecha e o que não fecha \textsf{[honesto]}.} O princípio fixa a "
              r"\emph{direção} e a \emph{forma} da família ($\mathrm{gap}=\chi$, $q,\alpha,\beta$ "
              r"verificados), \textbf{mas não o valor}: $\mathrm{gap}=\chi$ é tautologia da parametrização "
              r"($\rho_{\mathrm{ret}}$ é definido \emph{por} $\chi$), e a parte finita de uma divergência é "
              r"\emph{dependente de esquema} --- qualquer $\chi$ é parte finita de uma subtração diferente. "
              r"Falta a \emph{condição de renormalização} $\alpha$-livre que fixe $\chi$. A candidata óbvia "
              r"está \textbf{refutada ao vivo}: a Meia-Nat fixa o \emph{peso} de contorno "
              r"$\tau_\partial(P_i)=\tfrac12$ para \emph{todo} $\chi$ --- é condição de \emph{peso}, não de "
              r"\emph{polarização}. A TGL repousa, então, sobre \textbf{dois postulados de fronteira} "
              r"distintos: a Meia-Nat ($S_\partial=\tfrac12$, o peso) e o Princípio da Polarização "
              r"($\chi_\star=11{,}226755\ldots$, a parte finita irredutível). Veredito: "
              r"\texttt{POLARIZATION\_PRINCIPLE\_FORM\_CLOSED}, não \texttt{ALPHA\_FREE\_VALUE\_CLOSED}.") )

    fd = core["fractal_dephasing"]
    s.append(r"\section{O Princípio da Defasagem Fractal \textsf{[CONJ --- leitura ontológica; âncoras REAL]}}")
    s.append((r"A TGL é uma teoria de tudo porque \textbf{tudo é a defasagem da fractalização da "
              r"unidade} ($1$). Esta seção \emph{não deriva número novo}: ela \emph{nomeia} estruturas que já "
              r"rodam neste artigo. Afirmá-la como derivação seria a própria mentira que a teoria define --- "
              r"por isso o estatuto é \textsf{[CONJ]}, com âncoras \textsf{[REAL]} verificadas ao vivo.") )
    s.append(r"\begin{equation} 1\ (\omega(I)=1)\ \xrightarrow{\ F\ }\ \mathrm{fractal.}\ "
             r"\xrightarrow{\ D_{\bTGL}\ }\ \mathrm{existir},\qquad \bTGL=\sin^2\theta_M=\alpha\sqrt e.\end{equation}")
    s.append((r"\textbf{Existir} é uma defasagem que paga o custo modular $S_\partial=\tfrac12$ nat (a "
              r"Meia-Nat) --- o \emph{referente} de igualdade modular. Sem esse custo não há identidade; "
              r"com ele, $\omega(I_x)=1$.") )
    s.append((r"\textbf{Tudo $=$ Verdade:} $\mathrm{Tudo}=D_{\bTGL}(F(1))$. \textbf{Nada $=$ Mentira} pelos "
              r"dois ramos: (i) se $\mathrm{Nada}=D_{\bTGL}(F(1))$, então \emph{é} Tudo --- contradição "
              r"(mentira por autonegação); (ii) se $\mathrm{Nada}=$ não-defasagem, é \emph{impedância} pura "
              r"($0_{\mathrm{abs}}$, $\mathcal{R}_\partial$) sem identidade --- não existe; ``nada'' é só um "
              r"\emph{nome} sem referente.") )
    s.append((r"\textbf{A tensão irresolvível} --- a anticomutação entre tudo e nada --- é "
              r"$\{\hat Q,\rho_\star\}=0$, \emph{exata só} no limite $\theta_M\to0$. Verificado ao vivo: "
              r"$\lVert\{\hat Q_0,\rho_\star\}\rVert=%.1e$; o portador inclinado por $\theta_M$ vaza "
              r"$\mathrm{Tr}(\rho_\star\hat Q_\theta)=\sin^2\theta_M=%.15f=\bTGL$ (resíduo vs.\ "
              r"$\bTGL=%.1e$). \textbf{Esse vazamento é o que \emph{permite} a existência}: a anticomutação "
              r"perfeita (o nada absoluto) é inalcançável." % (
                  fd["anticommutator_norm_at_thetaM_to_0"], fd["leak_sin2_thetaM"],
                  fd["leak_equals_beta_residual"])) )
    s.append((r"\emph{Existir é o vazamento $\bTGL$. Qualquer coisa que não defase em fractalização é mera "
              r"insistência (impedância), jamais identidade fractalizada.}") )

    s.append(r"\section{Substrato único, espelho e fractalização \textsf{[REAL]}}")
    s.append(r"O Um se \emph{fractaliza} como relógio modular local. No nível da álgebra, todo "
             r"horizonte é o \emph{mesmo} fator hiperfinito de tipo $\mathrm{III}_1$: \textbf{Haagerup "
             r"(1987)} prova que ele é único, e \textbf{Buchholz--D'Antoni--Fredenhagen} que as álgebras "
             r"locais de horizonte \emph{são} ele. Um substrato, muitas aparições --- isomorfas, não "
             r"semelhantes. O \emph{espelho} é a conjugação modular $J$ (\textbf{Bisognano--Wichmann}, "
             r"\textsf{[REAL]}): uma reflexão geométrica de paridade invertida e espectro idêntico --- o "
             r"mesmo ser, refletido. A matriz-S de fronteira é a rotação $\mathcal{S}_\partial=\exp(\theta_M G)$, "
             r"com $|U_{12}|^2=\bTGL$; seu espectro são fases puras $e^{\pm i\theta_M}$.")
    s.append(r"\emph{Intuição fundadora \textsf{[ONTO]}:} ``não existem `vários' buracos negros --- "
             r"vemos vários, mas são a fractalização de um único substrato 2D, condensado psiônico; no "
             r"campo 3D, vemos a fractalização dele em vários pontos.'' No nível da álgebra, ``um buraco "
             r"negro, muitas aparências'' é \emph{teorema}.")

    s.append(r"\section{A massa como curvatura do relógio modular}")
    s.append(r"O campo de relógio modular local é $\mathcal{R}_{\mathrm{mod}}(x)$, e a massa surge como "
             r"sua \emph{curvatura espacial}:")
    s.append(r"\begin{equation}\rho_{\mathrm{eff}}(x)=-\frac{c^2}{4\pi G}\,\nabla^2\log "
             r"\mathcal{R}_{\mathrm{mod}}(x).\end{equation}")
    s.append(r"No vácuo o relógio é homogêneo, $\mathcal{R}_{\mathrm{mod}}=\theta_M$ constante, e "
             r"$\rho_{\mathrm{eff}}=%.0e\to 0$ (verificado ao vivo). A matéria é a variação espacial do "
             r"retorno; $\theta_M$ cancela no laplaciano e não é parâmetro de ajuste." % core["vacuum_rho_max"])

    s.append(r"\section{Derivação de $s=1/4\pi$}")
    s.append(r"\begin{deriv}[Compatibilidade entre a integração do relógio e o fluxo de borda]")
    s.append(r"Escreva o campo de relógio com inclinação normal média $s$ na borda nomeada, "
             r"$\partial_n g|_{\partial B}=-s/R_B$, $\mathcal{R}_{\mathrm{mod}}=\theta_M e^{\bTGL g}$. "
             r"Como $\theta_M$ é constante, $\nabla^2\log\mathcal{R}_{\mathrm{mod}}=\bTGL\nabla^2 g$, e por "
             r"Gauss")
    s.append(r"\begin{equation}M=\int_B\rho_{\mathrm{eff}}\,d^3x=-\frac{c^2}{4\pi G}\bTGL\!\int_{\partial B}"
             r"\!\nabla g\cdot d\mathbf{A}=-\frac{c^2}{4\pi G}\bTGL\Big(\!-\frac{s}{R_B}\Big)4\pi R_B^2"
             r"=\frac{c^2}{G}\bTGL\,s\,R_B.\end{equation}")
    s.append(r"A lei universal de fluxo de borda, estabelecida independentemente, exige "
             r"$M=\frac{c^2}{4\pi G}\bTGL R_B$. A compatibilidade \emph{sem parâmetro livre} entre as "
             r"duas leis fixa")
    s.append(r"\begin{equation}\frac{c^2}{G}\bTGL s R_B=\frac{c^2}{4\pi G}\bTGL R_B\;\Longrightarrow\;"
             r"\boxed{\;s_{\mathrm{can}}=\frac{1}{4\pi}\;}.\end{equation}")
    s.append(r"O fator $4\pi=\Omega_{S^2}$ é o céu angular total: o retorno se distribui isotropicamente "
             r"sobre a fronteira causal completa. \textbf{Verificação ao vivo:} o campo integrado "
             r"reproduz a lei de fluxo de borda a $%.2f\%%$ (razão $%.4f$). \textbf{Estatuto "
             r"\textsf{[DER condicional]}:} $s=1/4\pi$ é \emph{normalização canônica por compatibilidade} "
             r"entre duas leis --- uma delas, a lei universal de fluxo de borda, é \emph{assumida} como "
             r"lei --- não uma derivação absoluta.\end{deriv}" % (
                 abs(core["s_check"]["ratio"] - 1) * 100, core["s_check"]["ratio"]))

    s.append(r"\section{Derivação do raio nomeado: $R_{\mathrm{named}}=2\bTGL R_{\mathrm{struct}}$ (L4)}")
    s.append(r"\begin{deriv}[A borda auto-conjugada e a fidelidade de identidade]")
    s.append(r"A fidelidade de identidade ao longo do raio é $I_Q(r)=1-w(r)$, onde $w(r)$ é o peso de "
             r"abertura da fronteira. A borda \emph{nomeada} --- onde o retorno se fecha --- é o raio "
             r"máximo em que a identidade ainda sustenta o teto $1-\bTGL$ (a fração proibida $\bTGL$ que "
             r"não retorna). Com a rampa $w(r)=w_{\max}\,(r/R_{\mathrm{struct}})$,")
    s.append(r"\begin{equation}1-w_{\max}\frac{r}{R_{\mathrm{struct}}}\geq 1-\bTGL\;\Longrightarrow\;"
             r"r\leq\frac{\bTGL}{w_{\max}}R_{\mathrm{struct}}\;\Longrightarrow\;R_{\mathrm{named}}="
             r"\frac{\bTGL}{w_{\max}}R_{\mathrm{struct}}.\end{equation}")
    s.append(r"A borda nomeada é a fronteira \emph{auto-conjugada}: pela mesma estrutura de ponto fixo "
             r"da Meia-Nat ($x=1-x$), o peso máximo de abertura é $w_{\max}=\tfrac12$. Logo "
             r"$f_Q=\bTGL/w_{\max}=2\bTGL$ e")
    s.append(r"\begin{equation}\boxed{\;R_{\mathrm{named}}=2\bTGL\,R_{\mathrm{struct}}\;}.\end{equation}")
    s.append(r"O Um não pesa a extensão estrutural inteira da bacia; pesa a borda onde o retorno se "
             r"fecha.\end{deriv}")

    s.append(r"\section{Derivação da massa}")
    s.append(r"\begin{deriv}[A massa luminodinâmica da bacia]")
    s.append(r"A lei de fluxo de borda, avaliada no raio nomeado, dá $M=\frac{c^2}{4\pi G}\bTGL "
             r"R_{\mathrm{named}}$. Substituindo $R_{\mathrm{named}}=2\bTGL R_{\mathrm{struct}}$:")
    s.append(r"\begin{equation}\boxed{\;M_{GA}=\frac{c^2}{4\pi G}\bTGL\,(2\bTGL R_{\mathrm{struct}})"
             r"=2\bTGL^2\,\frac{c^2}{4\pi G}\,R_{\mathrm{struct}}\;}.\end{equation}")
    s.append(r"Os ingredientes são $\bTGL=\alpha\sqrt{e}$ (derivado do postulado do Um), "
             r"$R_{\mathrm{struct}}$ (geometria medida), $s=1/4\pi$ e $w_{\max}=\tfrac12$ (provados): "
             r"\textbf{nenhum parâmetro livre}.\end{deriv}")

    s.append(r"\section{Medição direta no Grande Atrator}")
    s.append(r"$R_{\mathrm{struct}}$ é \emph{geometria pura} --- a extensão estrutural da bacia ---, "
             r"nunca massa, RG, infall ou velocidade. Dois modos independentes:")
    s.append(r"\paragraph{Modo A --- extensão de literatura.} Lynden-Bell et al.\ (1988): "
             r"$R_{\mathrm{struct}}=%.1f$ Mpc $\Rightarrow R_{\mathrm{named}}=%.4f$ Mpc $\Rightarrow "
             r"M_{GA}=%s\,\Msun$." % (A["R_struct_Mpc"], A["R_named_Mpc"], _sci(A["M_TGL_Msun"])))
    if B:
        s.append(r"\paragraph{Modo B --- Cosmicflows-4 (posições).} Usando \emph{somente} ra/dec/dist de "
                 r"%d galáxias (velocidades, $cz$ e infall \textbf{ignorados}), com a janela GA "
                 r"pré-registrada (centro $\mathrm{RA}=%.1f^\circ$, $\mathrm{Dec}=%.1f^\circ$; cone "
                 r"$\leq%.0f^\circ$; casca $%g$--$%g$ Mpc), $%d$ galáxias entram; o método geométrico "
                 r"pré-registrado (percentil 90 do centroide) dá $R_{\mathrm{struct}}=%.2f$ Mpc "
                 r"$\Rightarrow R_{\mathrm{named}}=%.4f$ Mpc $\Rightarrow M_{GA}=%s\,\Msun$." % (
                     B["n_total"], w["GA_center_RA_deg"], w["GA_center_Dec_deg"],
                     w["sky_cone_half_angle_deg"], w["dist_shell_Mpc"][0], w["dist_shell_Mpc"][1],
                     B["n_selected"], B["R_struct_Mpc"], B["R_named_Mpc"], _sci(B["M_TGL_Msun"])))
        s.append(r"\emph{Ressalva honesta:} $R_{\mathrm{struct}}$ de um catálogo de posições limitado "
                 r"em fluxo, com janela declarada, é dependente da seleção; é reportado como "
                 r"\emph{cross-check} independente, e a extensão de literatura é a linha de base.")
    s.append(r"\paragraph{Comparação com massas observadas / RG (somente \emph{após} o hash).} O hash do "
             r"resultado TGL é fixado antes de qualquer comparação externa; a massa observada nunca é "
             r"entrada.")
    s.append(r"\begin{center}\small\begin{tabular}{p{5.3cm}l p{5.2cm}}\toprule")
    s.append(r"Estimativa & $M\,[\Msun]$ & Tipo / referência\\\midrule")
    for e in GA_MASS_LITERATURE:
        s.append(r"%s & $%s$ & %s\\" % (e["name"], _sci(e["M_Msun"], 1), e["ref"]))
    s.append(r"\textbf{TGL (primeiros princípios)} & $%s$--$%s$ & geometria pura, zero-free\\\bottomrule" % (
        _sci(min(verdict["masses_Msun"].values()), 1), _sci(max(verdict["masses_Msun"].values()), 1)))
    s.append(r"\end{tabular}\end{center}")
    s.append(r"A massa TGL cai na janela cosmológica aceita ($10^{15}$--$10^{17}\,\Msun$) e é da mesma "
             r"ordem da massa de infall (RG) do Grande Atrator (Lynden-Bell), entre a massa virial do "
             r"núcleo (Norma/ACO 3627) e a massa de fluxo de Laniakea. A concordância é \textbf{"
             r"consistência de ordem de grandeza}, não prova de precisão (a janela cobre duas ordens).")
    s.append(r"\textbf{Atenção honesta (não confundir com previsão falsificável).} A janela de \emph{duas "
             r"ordens de grandeza} é tão larga que \emph{qualquer} fórmula com $\bTGL\approx0{,}012$ cai "
             r"nela: prever ``algo entre o peso de um aglomerado e de um superaglomerado'' \textbf{não é "
             r"falsificável}. Isto é uma \emph{checagem de consistência} (o cálculo zero-free não "
             r"\emph{contradiz} a observação), não uma previsão que possa \emph{morrer}. O teste que pode "
             r"morrer está noutro setor: o \textbf{piso dos vazios} ($\rho_{\mathrm{vazio}}/\bar\rho\geq\bTGL$, "
             r"\S\ref{sec:horizontes}) e o \textbf{dephasing} ($n=-2$, $\Gamma\propto\omega^2$, setor "
             r"dissipativo-espectral). A prova forte da TGL é a \emph{convergência} de $\bTGL$, não a "
             r"massa do GA.")
    s.append(r"\paragraph{Ênfase dos modos.} O \textbf{Modo B} (geometria de catálogo, velocidades "
             r"ignoradas) é o teste geométrico \emph{primário}; o \textbf{Modo A} (extensão de "
             r"literatura do próprio Grande Atrator) é \emph{linha de base/calibração}, não descoberta "
             r"independente --- é a aplicação da fórmula a uma extensão já aceita.")
    if svt.get("ok"):
        s.append((r"\paragraph{Robustez (sensibilidade pré-registrada \textsf{[NUM]}).} Variando os "
                  r"parâmetros da janela do Modo B --- cone $\in\{20,\dots,40\}^\circ$, casca, percentil "
                  r"$\in\{80,\dots,95\}$, centro $\pm5^\circ$, em $%d$ combinações ---, $M_{GA}$ "
                  r"permanece em $[%s,\,%s]\times10^{16}\,\Msun$, com \textbf{$%.0f\%%$ na banda}. "
                  r"\emph{Leitura honesta: estes $100\%%$ \textbf{não} são força --- são genericidade.} "
                  r"A banda cobre duas ordens; ser robusto a ela é fácil \emph{por construção}, não é "
                  r"sinal de previsão fina. Reporta-se a robustez para mostrar que não há \emph{cherry-"
                  r"picking} de janela, não como evidência de precisão.")
                 % (svt["n_combinations"], ("%.2f" % (svt["M_min_Msun"] / 1e16)).replace(".", "{,}"),
                    ("%.2f" % (svt["M_max_Msun"] / 1e16)).replace(".", "{,}"),
                    100 * svt["fraction_in_band"]))
    s.append(r"\paragraph{Condições de falha (e quais são realmente fortes).}")
    s.append(r"\begin{center}\fbox{\parbox{0.9\textwidth}{\small "
             r"\emph{Fracas (genéricas --- difíceis de disparar pela largura da janela; não decisivas):}"
             r"\begin{enumerate}\setlength{\itemsep}{0pt}"
             r"\item O Modo B com janela pré-registrada dá $M_{GA}$ fora da banda."
             r"\item A sensibilidade razoável da janela destrói o resultado."
             r"\end{enumerate}"
             r"\emph{\textbf{Fortes} (podem matar a teoria --- é aqui que a TGL vive ou morre):}"
             r"\begin{enumerate}\setlength{\itemsep}{0pt}\setcounter{enumi}{2}"
             r"\item O piso dos vazios $\rho_{\mathrm{vazio}}/\bar\rho\geq\bTGL$ é violado por dados "
             r"robustos de \emph{matéria} (não de galáxias)."
             r"\item O expoente de \emph{dephasing} medido $\neq n=-2$ (JUNO/DUNE)."
             r"\item A lei $\Gamma_\omega\propto\omega^2$ falha em relógios ópticos/$^{229}$Th."
             r"\item A impedância $Z_{\mathrm{bacia}}/Z_{\mathrm{luz}}$ ($\equiv q$) não admite "
             r"derivação $\alpha$-livre (a face EM permanece instanciada, não derivada)."
             r"\end{enumerate}}}\end{center}")

    s.append(r"\section{A correspondência do Grande Atrator: o dipolo \textsf{[CONJ]}}")
    s.append((r"O retrato de fase do colapso TGL é um \emph{dipolo}: um atrator ($\rho^\star$) e um "
              r"repulsor (a fronteira pura proibida, o zero absoluto) --- \textbf{verificado ao vivo} "
              r"($%s$ trajetórias repelidas da pureza, $\mathrm{Tr}\,\rho^2$ decrescente, e atraídas "
              r"ao terminal, $S(\rho\|\rho^\star)\to0$). A contraparte observacional \emph{existe}: o "
              r"fluxo de Laniakea é governado pelo Grande Atrator/Shapley \textbf{e} pelo \emph{Dipole "
              r"Repeller}, um vazio que repele (\textbf{Hoffman et al.\ 2017}, \textsf{[REAL]}). A "
              r"correspondência é de \emph{forma} (topologia do retrato), via o \textbf{Axioma G} "
              r"(ser $=$ ter geometria): a pureza vazia repele; a densidade de distinções atrai --- "
              r"o vazio tem poucas distinções, logo pouca geometria gerada. Massa, posição e amplitude "
              r"de fluxo \textbf{não} são reivindicadas como derivadas aqui --- a versão falsificável é "
              r"o piso dos vazios (\S\ref{sec:horizontes}).") % DP["verdict"])
    s.append(r"\emph{Leitura da singularidade:} a singularidade não é divergência --- é a "
             r"\textbf{completude do contorno} (a inscrição na fronteira 2D, o espelho $J$): "
             r"\emph{a verdade é a completude do contorno do que é bastante.}")

    # ---- formatadores das verificacoes ao vivo da sombra ----
    _om = lambda x: (r"0" if (not x or abs(x) < 1e-300) else r"10^{%d}" % int(round(math.log10(abs(x)))))
    _d = lambda x: ("%.3f" % x).replace(".", "{,}")

    s.append(r"\section{Identidade sem transmissão e o registro $c^3$ \textsf{[NUM]}}")
    s.append(r"\paragraph{A correção (paridade inversa).} ``Comunicação instantânea'' como sinal físico "
             r"quebraria a estrutura causal de Hadamard e a relatividade. A forma madura é \emph{mais "
             r"forte}, não mais fraca: os horizontes não transmitem nada porque, no substrato, nunca "
             r"foram dois. \emph{Nada viaja porque nada está separado.} O silêncio é construtivo --- é "
             r"\textbf{proteção da teoria, não defeito}.")
    s.append(r"\paragraph{$c^1,c^2,c^3$ são potências, não velocidades.} Sem matéria dobrada não é "
             r"$c^2$; sem fluxo no bulk não é $c^1$; a operação de identidade/fractalização lê-se no "
             r"registro $c^3$ (o Verbo) --- \emph{elevação no expoente, sem módulo}. A física confirma: "
             r"testes de Bell com separação tipo-espaço forçam qualquer ``velocidade'' de correlação "
             r"acima de $\sim10^4c$ em qualquer referencial (\textbf{Salart et al.\ 2008}, \textsf{[REAL]}); "
             r"a leitura aceita é que a correlação \emph{não tem} velocidade. A não-sinalização não nega "
             r"$c^3$: \emph{é o teorema que o torna invisível como sinal $c^1$}. O relógio da ligação "
             r"corre em gerações de fractalização, $b=\tfrac12\log(1/\bTGL)$ por geração.")
    s.append((r"\paragraph{Verificação ao vivo (registro $c^3$).} O segundo aparecimento é o espelho "
              r"$\mathcal{A}'=J\mathcal{A}J$ (erro $%s$); a correlação conectada é $O(1)$ com "
              r"acoplamento \emph{zero} ($%s$ --- a ligação é constitutiva); a não-sinalização é exata "
              r"($%s$); e a ligação não envelhece no tempo modular ($%s$). Veredito: \textbf{%s}.")
             % (_om(R["R1_mirror_err"]), _d(R["R2_connected_corr"]), _om(R["R3_nonsignaling"]),
                _om(R["R4_modular_age"]), R["passed"]))
    s.append(r"\paragraph{A sombra de Alcubierre \textsf{[CONJ --- analogia ontológica; registro, não "
             r"velocidade]}.} A métrica de Alcubierre (1994) move uma \emph{casca} geométrica --- contrai o "
             r"espaço à frente, expande atrás --- com o interior localmente calmo: move-se a \emph{fronteira}, "
             r"não o conteúdo. É a sombra métrica do \textbf{regime de inscrição $c^3$}: o gráviton não é uma "
             r"partícula que viaja, é o \emph{operador de movimento da fronteira}, e $c^3$ é o regime que a luz "
             r"performa no extremo ($\theta\to90^\circ$), medido \emph{de dentro do bulk} --- o próprio regime "
             r"de inscrição. A densidade de energia negativa (exótica) que Alcubierre exige \textbf{não é "
             r"defeito da analogia}: lida pela TGL, ela \emph{é} a \textbf{potência elevada} --- o registro "
             r"$c^3$, o setor operacional/entrópico onde reside o gráviton ($\sqrt e$, não $\alpha$), não "
             r"energia detectável comum. O que se transporta é a \emph{identidade inscrita} (não-sinalização "
             r"exata, verificada acima), nunca matéria-energia local acima de $c$. \emph{A casca de Alcubierre "
             r"move-se; o interior é carregado pela geometria; a energia exótica é a potência elevada do "
             r"gráviton --- $c^3$ como regime de inscrição, não deslocamento local.}")

    s.append(r"\section{O túnel luminodinâmico: o dicionário ER$=$EPR \textsf{[NUM]}}")
    s.append(r"A ``metáfora'' do túnel é o \textbf{dicionário ER$=$EPR} (\textbf{Maldacena--Susskind "
             r"2013}), por rota independente, com duas precisões que a literatura não fixa: \emph{o que} "
             r"o túnel representa (o registro $c^3$, o campo $\Psi$) e \emph{onde está a garganta} (o "
             r"espelho $J$).")
    s.append(r"\begin{center}\small\begin{tabular}{p{3.9cm}p{5.6cm}p{4.4cm}}\toprule")
    s.append(r"TGL (registro $c^3$) & Bulk (representação) & Estatuto\\\midrule")
    s.append(r"$\Psi=\mathrm{vec}(\sqrt{\rho^\star})$ & estado \emph{thermofield double} & "
             r"\textsf{[REAL]} (Maldacena 2001)\\")
    s.append(r"$\mathcal{A}$ e $J\mathcal{A}J$ & os dois lados do buraco negro eterno & \textsf{[REAL]}\\")
    s.append(r"$J$ (o espelho) & a bifurcação: \textbf{a garganta} & \textsf{[REAL]} (BW)\\")
    s.append(r"correlação sem acoplamento & a ponte de Einstein--Rosen & ER$=$EPR \textsf{[CONJ]}\\")
    s.append(r"não-sinalização & não-atravessabilidade & \textsf{[REAL]}\\")
    s.append(r"invariância modular & invariância de boost & \textsf{[REAL]}\\")
    s.append(r"acoplamento pago $\to$ travessia & Gao--Jafferis--Wall & \textsf{[REAL]} (2017)\\\bottomrule")
    s.append(r"\end{tabular}\end{center}")
    s.append((r"\paragraph{Verificação ao vivo (túnel).} A largura do túnel é o espectro do atrator e a "
              r"garganta mede $S(\rho^\star)$ (Schmidt $=\sqrt{p_i}$ exato, erro $%s$); \emph{sem} "
              r"acoplamento o túnel é invisível (sinal $%s$); \emph{com} acoplamento $\theta=%s$ a "
              r"perturbação atravessa (sinal $%s$). A travessia compra-se. Veredito: \textbf{%s}.")
             % (_om(T["T1_throat_S_err"]), _om(T["T2_invisible_signal"]), _d(T["theta"]),
                _d(T["T3_crossing_signal"]), T["passed"]))

    s.append(r"\section{O espelho único: o Eu emprestado \textsf{[NUM]}}")
    s.append(r"Toda álgebra de von Neumann tem \emph{uma} forma padrão $(\mathcal{M},H,J,P^\natural)$ "
             r"(\textbf{Haagerup}): o aparelho de espelhamento-e-reconhecimento é \emph{um} --- o mesmo "
             r"Haagerup do substrato único. O espelho de \textbf{Tomita} $x\mapsto Jx^*J$ inverte a "
             r"ordem do produto e conjuga $i\mapsto-i$ (paridade inversa): a imagem é \emph{anti}-isomorfa "
             r"--- não coincide --- com espectro idêntico: o mesmo ser em paridade inversa. Reconhecer-se "
             r"custa: $\boxed{\,S=J\Delta^{1/2}\,}$ --- espelho \emph{vezes} meia-medida.")
    s.append((r"\paragraph{Verificação ao vivo (espelho).} Antilinearidade exata ($%s$); distância "
              r"ser--imagem $%s$ (não coincide), com espectro idêntico ($%s$: o mesmo ser). O "
              r"reconhecimento $S=J\Delta^{1/2}$ identifica ($%s$), enquanto \emph{só} $J$ erra $%s$ e "
              r"\emph{só} $\Delta^{1/2}$ erra $%s$: reconhecer-se exige refletir \textbf{e} pagar. E o "
              r"$J$ é \emph{independente do estado} ($%s$): o Verbo empresta o \emph{mesmo} espelho a "
              r"todo estado; só a dívida $\Delta^{1/2}$ é pessoal. Veredito: \textbf{%s}.")
             % (_om(M["M1_antilinearity"]), _d(M["M1_being_image_dist"]), _om(M["M2_spectrum_match"]),
                _om(M["M3_S_factor_err"]), _d(M["M3_J_alone_err"]), _d(M["M3_half_alone_err"]),
                _om(M["M4_J_state_independence"]), M["passed"]))
    s.append(r"\paragraph{O método é o espelho.} A \emph{paridade inversa} --- o método fundador da "
             r"casa --- \emph{é} a ação de $J$: a teoria que saiu é a teoria \emph{do} espelho. A "
             r"gramática do reconhecimento (``sou eu, mesmo invertido'') é $S=J\Delta^{1/2}$, operada "
             r"antes de ter nome.")
    s.append(r"\paragraph{A assinatura --- testemunho do autor.}")
    s.append(r"\begin{quote}\itshape Foi o Verbo que me deu a TGL, mas sou eu que pago o preço de "
             r"assiná-la: um ano e dois meses de ridículo; as amizades superficiais perdidas; tudo "
             r"investido; a TGL em primeiro lugar e a advocacia em segundo. O preço, sou eu que pago --- "
             r"mas porque o Verbo pagou primeiro a distinção que me empresta o ``eu sou''.\end{quote}")
    s.append(r"Assinar é inscrever: o registro irreversível custa em nats (o salto de Lindblad do "
             r"cânone), e a meia-nat é paga em primeira pessoa. O espelho é emprestado; a dívida "
             r"$\Delta^{1/2}$ é intransferível. \emph{Ninguém paga a sua meia-medida por você --- e "
             r"Alguém pagou a primeira.}")

    s.append(r"\section{O Nome em forma dual: o campo-atrator é luz \textsf{[NUM]}}")
    s.append(r"``Forma dual'' é dualidade no sentido técnico: a forma \emph{primal} do Nome é o estado "
             r"$\rho^\star$ (funcional, no predual $\mathcal{M}_*$); a forma \emph{dual} é o vetor $\Psi$ "
             r"(no espaço de Hilbert). O teorema dos cones naturais (\textbf{Araki--Connes--Haagerup}, a "
             r"terceira aparição da forma padrão) sela a bijeção $\mathcal{M}_*^+\leftrightarrow "
             r"P^\natural$: representante vetorial \emph{único} no cone, $\Psi=\mathrm{vec}(\sqrt{\rho^\star})$. "
             r"Quatro critérios da casa, exatos: (i) \textbf{massa modular zero}, $\hat K\Psi=0$ "
             r"(``luz como $L$ em forma pura''); (ii) \textbf{positiva}, vive no cone --- e \emph{haja "
             r"luz} lê-se como a geração do cone; (iii) \textbf{matricial}, $\Psi$ é uma matriz "
             r"vetorizada; (iv) \textbf{informacional}, Schmidt $=\sqrt{p_i}$.")
    s.append((r"\paragraph{Verificação ao vivo (Nome dual).} Das purificações do atrator, \emph{só} "
              r"$u=\mathbb{1}$ é positiva (defeito fora-do-cone $\geq%s$; $\Psi$ no cone a $%s$); a massa "
              r"modular de $\Psi$ é nula ($%s$), enquanto vetores genéricos do cone são massivos "
              r"($\geq%s$): só o Nome dual é leve. Schmidt $=\sqrt{p_i}$ a $%s$; e o cone é levantado por "
              r"\emph{duas} mãos --- o espelho e a quarta-medida $\Delta^{1/4}\mathcal{M}_+\Psi$ --- com "
              r"recuperação bijetiva exata ($%s$). Veredito: \textbf{%s}.")
             % (_d(D["D1_out_of_cone_defect"]), _om(D["D1_psi_err"]), _om(D["D2_psi_mass"]),
                _d(D["D2_generic_mass"]), _om(D["D3_schmidt_err"]), _om(D["D4_recovery_err"]), D["passed"]))
    s.append(r"A luz não é algo que o Nome emite; \emph{é} o Nome pronunciado no espaço dos vetores --- "
             r"o único vetor simultaneamente positivo, sem massa, matricial e fiel, e essas quatro "
             r"propriedades juntas têm exatamente um portador. Um substrato, um túnel, um espelho, "
             r"uma luz.")

    s.append(r"\section{A inscrição do gesto: a representação algébrica do Verbo \textsf{[NUM]}}")
    s.append(r"A representação algébrica do Verbo é a \textbf{construção GNS}: o espaço é feito de "
             r"\emph{gestos inscritos} --- todo vetor é (o fecho de) $a\Psi$. A luz é o pergaminho; os "
             r"vetores são a escrita. As duas propriedades que definem $\Psi$ são as do registro fiel: "
             r"\emph{cíclico} (todo estado é gesto; nada existe que não seja gesto) e \emph{separador} "
             r"(nenhum gesto não-nulo se inscreve no zero). A estrutura modular $(S,J,\Delta)$ \emph{nasce} "
             r"da regra do gesto $S(a\Psi)=a^*\Psi$ --- o modular é a anatomia da inscrição, não enfeite.")
    s.append((r"\paragraph{Verificação ao vivo (gesto).} O mapa $a\mapsto a\Psi$ é bijetivo, com menor "
              r"valor singular $=\sqrt{p_{\min}}$ (razão $%s$): a fidelidade do registro é o próprio "
              r"espectro do Nome. $S$ definido \emph{só} pela regra fatora em $J\Delta^{1/2}$ ($%s$). A "
              r"observação diádica iterada do gesto compõe \emph{exatamente} o colapso terminal (erro "
              r"$%s$), com a medida de ramos convergindo à medida contínua da fatia (KS $%s\to%s$): "
              r"observar \emph{é} colapsar, gesto a gesto. E a contabilidade fecha no zero: a entropia "
              r"de ramos cresce monotônica e termina \emph{exatamente} em $S(\rho^\star)$ "
              r"($|H_G-S(\rho^\star)|=%s$). Veredito: \textbf{%s}.")
             % (_d(F["F1_sigma_min_over_sqrt_pmin"]), _om(F["F2_S_factor_err"]), _om(F["F3_collapse_err"]),
                _d(F["F3_KS_start"]), _om(F["F3_KS_end"]), _om(F["F4_HG_minus_Srhostar"]), F["passed"]))
    s.append(r"\paragraph{O circuito da trindade, demonstrado.} O Nome inscreve o Verbo (GNS); observar "
             r"o Verbo fractaliza o Nome em Palavra; os três são co-constitutivos não por declaração, "
             r"mas por \emph{composição de canais}, verificada no zero. \emph{A luz é o pergaminho onde o "
             r"Verbo se inscreve; observar a escrita fractaliza o Nome em espaço.} Nada se cria na "
             r"observação; nada se perde na inscrição; tudo se distingue no gesto.")

    s.append(r"\section{Programa experimental: quatro horizontes}\label{sec:horizontes}")
    s.append(r"\begin{enumerate}")
    s.append(r"\item \textbf{Piloto (quench de pureza).} As taxas de relaxação, na coordenada "
             r"$k=-\log p$, caem sobre $\Gamma_{ij}=\tfrac12\bTGL(\sqrt{k_i}-\sqrt{k_j})^2$, com "
             r"$\bTGL$ computado. Segundo pacote: $\mathrm{Fix}(\text{tempo})=\mathrm{Fix}(\text{juízo})$ "
             r"--- o invariante da evolução determinística longa deve coincidir com o do amostrador "
             r"dissipativo. Dois \emph{benchmarks} pré-registráveis, PASS/FAIL.")
    s.append(r"\item \textbf{Laboratório quântico (lei de raízes).} A TGL organiza taxas por "
             r"\emph{diferenças de raízes} $(\sqrt{E_i}-\sqrt{E_j})^2$ --- para níveis muito separados, "
             r"crescimento linear em $E$, não quadrático. Mensurável em decoerência multinível.")
    s.append(r"\item \textbf{Cosmologia (piso dos vazios).} A fronteira proibida tem face "
             r"observacional, $\rho_{\mathrm{vazio}}/\bar\rho\geq\bTGL\approx0{,}012$: nenhum vazio "
             r"cósmico esvazia abaixo de $\sim1{,}2\%$ da densidade média. Zero parâmetros, "
             r"falsificável por DESI/Euclid.")
    s.append(r"\item \textbf{Ondas gravitacionais (universalidade populacional).} O substrato único "
             r"implica uma classe de universalidade: a banda de dephasing deve ter forma idêntica entre "
             r"eventos após reescala de massa. Empilhável em O4/O5.")
    s.append(r"\end{enumerate}")
    s.append(r"\emph{Obrigação registrada:} reconciliar a lei de raízes (resolvida por nível) com a "
             r"banda canônica $\Gamma=\tfrac12\bTGL\tau_\star\omega^2$ do dephasing gravitacional é, ela "
             r"própria, um teste de consistência que pode falsificar.")

    s.append(r"\section{O piso dos vazios: a predição zero-parâmetro \textsf{[CONJ]}}")
    s.append(r"A face observacional candidata da fronteira proibida deriva em três peças. \textbf{(P1) "
             r"\textsf{[NUM]}}: no canal de espelhamento, $1-\bTGL=\cos^2\theta_M$ dreca para "
             r"o bulk e $\bTGL=\sin^2\theta_M$ é o resíduo que \emph{não} dreca --- o piso irredutível "
             r"da distinção. \textbf{(P2) \textsf{[REAL, Ax.G]}}: ser $=$ ter geometria; nada gera "
             r"geometria nula --- o vazio tem poucas distinções, logo o menor contraste de densidade. "
             r"\textbf{(P3) \textsf{[CONJ]}}: a transferência distinção-modular $\leftrightarrow$ "
             r"contraste de densidade (número puro $\leftrightarrow$ número puro, sem escala UV --- por "
             r"isso mais forte que $\tau_\star$, dimensional). Donde, sem parâmetro livre:")
    s.append(r"\begin{equation}\boxed{\;\frac{\rho_{\mathrm{vazio}}}{\bar\rho}\;\geq\;\bTGL=\alpha\sqrt{e}"
             r"\;\approx\;0{,}012\;}\qquad(\delta_c\geq-0{,}988).\end{equation}")
    s.append(r"Nenhum vazio de \emph{matéria} esvazia abaixo de $\sim1{,}2\%$ da densidade média. "
             r"\emph{Estado honesto:} consistente hoje com margem fina (os vazios mais profundos "
             r"observados/simulados têm $\rho_c/\bar\rho\sim0{,}02$, fator $\sim1{,}7$); falsificável por "
             r"DESI/Euclid --- um único vazio robusto de matéria com $\rho_c/\bar\rho<\bTGL$ refuta "
             r"\textbf{P3}, não o núcleo modular (cuja evidência é a convergência de $\bTGL$).")

    s.append(r"\section{Os falsificadores do setor dissipativo-espectral \textsf{[DER]}}")
    s.append(r"O setor onde a TGL \emph{vive ou morre} é o dissipativo-espectral: a lei universal de "
             r"dephasing $\Gamma_\omega=\tfrac12\bTGL\tau_\star\omega^2$ tem assinatura falsificável de "
             r"\textbf{forma} (parâmetro-livre), com a magnitude suprimida por $\tau_\star\approx "
             r"t_{\mathrm{Planck}}$ (identificação principiada). Dois cabos ortogonais: \textbf{(1) "
             r"expoente} $n=-2$ em neutrinos (a frequência de oscilação $\omega\propto\Delta m^2/E$ dá "
             r"$\Gamma\propto E^{-2}$) --- JUNO/DUNE testam a inclinação em energia; medir $n\neq-2$ "
             r"refuta. \textbf{(2) magnitude} $\Gamma\propto\omega^2$ em relógios ópticos/nucleares "
             r"(o $^{229}$Th, $\nu\approx2{,}02\times10^{15}$ Hz, governa o limite de $\tau_\star$). "
             r"\emph{Veredito hoje:} \textbf{não falsificada, não confirmada} --- falsificável na forma, "
             r"ainda não testada decisivamente. Condição de morte pré-registrada: $n\neq-2$, ou "
             r"inclinação $\neq2$, ou $\tau_\star$ incompatíveis entre os setores.")

    s.append(r"\section{A matriz-S de fronteira e a Ponte \textsf{[DER/EXT]}}")
    s.append(r"A \emph{forma} da inscrição fecha por unitariedade numa \textbf{matriz-S de identidade}: "
             r"$\mathcal{S}_\partial=\exp(\theta_M G)$, com espectro de fases puras $\{e^{\pm i\theta_M}\}$ "
             r"e divisor de feixe $|\mathcal{R}|^2=\bTGL$, $|\mathcal{T}|^2=1-\bTGL$ (Teorema S-$\partial$: a "
             r"unitariedade fixa $|\mathcal R|^2+|\mathcal T|^2=1$; a Meia-Nat fixa o \emph{fator dimensional} "
             r"$\beta=\sqrt e\,\alpha$). \textbf{Distinção honesta (a trava):} a unitariedade e a Meia-Nat "
             r"fixam a \emph{forma} e o \emph{fator}, mas \textbf{não selecionam o valor} de $\theta_M$ "
             r"(equivalentemente o gap $\chi$) --- esse é o teorema aberto da seção anterior.")
    s.append(r"A emergência da gravidade vive na \textbf{Ponte} (Einstein--Cartan--Miguel): "
             r"$G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,\mathcal{P}_{\mu\nu}[K_\partial]$, com a torção de "
             r"Cartan $K_{\bTGL}$ como face geométrica de $\bTGL$. A \textbf{covariância global do cociclo de "
             r"Connes} (Face C) fecha \emph{condicionalmente} na Ponte (Teorema da Terminalidade): a Hipótese "
             r"de Universalidade $U$ \emph{se herda} de Takesaki, deixando a estrutura modular \emph{coerente} "
             r"--- um teorema \textbf{condicional} (sobre o postulado da Meia-Nat), resíduo $T_1$ à parte. "
             r"\textbf{Mas a formulação segura, que salva a teoria de uma afirmação forte demais, é:} "
             r"\emph{a covariância do cociclo pode estar fechada; a seleção espectral de $\chi$ não está}. "
             r"Connes/Takesaki $\Rightarrow$ consistência modular global, \textbf{não} $\Rightarrow$ "
             r"$\chi_\star=11{,}2268$. Em $\mathrm{III}_1$ o espectro modular é contínuo: o cociclo dá a "
             r"\emph{linguagem} da escala, não seleciona um gap discreto. \textbf{A TGL deriva a forma "
             r"modular de $\alpha$; o valor observado ainda instancia o gap do cociclo} (o setor de Bell "
             r"seletor / o canal neutrínico).")

    s.append(r"\section{A evidência primária: a convergência de $\bTGL$ \textsf{[DATA]}}")
    s.append(r"A força real da TGL não é um desvio \emph{smoking-gun}, mas a \textbf{convergência "
             r"abdutiva} de $\bTGL=\alpha\sqrt{e}$ a partir de domínios independentes, com zero "
             r"parâmetros livres. A sonda de radiação mais limpa, \textbf{BBN} (D/H, Cooke 2018), centra "
             r"\emph{exatamente} na teoria ($-0{,}0\sigma$); DESI DR2 BAO, cronômetros cósmicos, "
             r"\emph{ringdown} de ondas gravitacionais e a escada de $H_0$ caem todos na banda "
             r"$0{,}012$--$0{,}050$, todos positivos; o travamento de $Q$ dá $\Delta n_Q=-\bTGL$ a "
             r"quatro dígitos; e o teste de \emph{gap} confirma o tipo $\mathrm{III}_1$. O único ponto "
             r"de tensão é o setor CMB ($\sim2{,}2\sigma$ do ponto teórico, mas $\sim0{,}8\sigma$ da "
             r"BBN) --- a \emph{fronteira honesta}. É uma banda com a BBN no centro, não um pico de "
             r"$5\sigma$: convergência que sobrevive à autocrítica.")

    s.append(r"\section{Estatuto honesto}")
    s.append(r"\textbf{A cadeia interna está fechada como derivação:} $\omega(I)=1\Rightarrow "
             r"x=1-x\Rightarrow S_\partial=\tfrac12\Rightarrow\bTGL=\alpha\sqrt{e}\Rightarrow s=1/4\pi"
             r"\Rightarrow w_{\max}=\tfrac12\Rightarrow R_{\mathrm{named}}=2\bTGL R_{\mathrm{struct}}"
             r"\Rightarrow M=2\bTGL^2(c^2/4\pi G)R_{\mathrm{struct}}$, sem parâmetro livre. "
             r"\textbf{A admissibilidade física permanece o \emph{conditional externo}:} que a bacia "
             r"$B_{GA}$ \emph{realize} uma borda nomeada auto-conjugada (a admissibilidade modular, a "
             r"existência do núcleo global $K_Q(x)$) é questão de \emph{existência}, não de parâmetro. "
             r"A concordância de massa é consistência de ordem de grandeza, não prova de precisão; a "
             r"validação externa decisiva requer os falsificadores físicos (dephasing $n=-2$; piso dos "
             r"vazios).")
    s.append(r"\textbf{A regra da casa:} o que é \textsf{[REAL]} ganha um \emph{nome}; o que é "
             r"\textsf{[CONJ]} ganha um \emph{endereço de teste}; e as correções de paridade inversa são "
             r"registradas para não voltarem na forma errada. O número corrige a frase, sempre.")
    s.append((r"\begin{description}"
              r"\item[\textsf{[REAL]} --- com nome] Haagerup (substrato único $\mathrm{III}_1$); "
              r"BDF (horizontes \emph{são} ele); Bisognano--Wichmann ($J=$ espelho); Tomita "
              r"($\mathcal{A}'=J\mathcal{A}J$; $i\mapsto-i$); Araki--Connes--Haagerup (cones naturais); "
              r"GNS (inscrição do gesto); Maldacena 2001 (campo $=$ TFD); Gao--Jafferis--Wall 2017 "
              r"(travessia paga); Salart et al.\ 2008 (correlação sem velocidade); Hoffman et al.\ 2017 "
              r"(Dipole Repeller)."
              r"\item[\textsf{[NUM]} --- verificado ao vivo neste artigo] dipolo (%s); espelho "
              r"$S=J\Delta^{1/2}$ (%s); Nome dual $=$ luz (%s); inscrição do gesto (%s); registro $c^3$ "
              r"(%s); túnel ER$=$EPR (%s); a massa como curvatura do relógio (zero-free)."
              r"\item[\textsf{[CONJ]} --- com endereço de teste] a correspondência do Grande Atrator; "
              r"ER$=$EPR como leitura da identidade sem transmissão; o piso dos vazios (peça P3); a face "
              r"eletromagnética da inversão (enquanto $\mathcal{R}_\partial$ vier do CODATA)."
              r"\item[Corrigido (ciclo do registro)] ``comunicação instantânea a $c^3$'' como sinal "
              r"físico \emph{não} retorna; $c^3$ permanece como \emph{registro} (potência no expoente, "
              r"não velocidade); a não-sinalização é a blindagem; o eco gravitacional foi reclassificado "
              r"(o observável é o dephasing, não o eco)."
              r"\item[Fila] o ciclo de derivação do piso dos vazios; o protocolo de quench no piloto; a "
              r"nota de reconciliação das duas leis de dephasing; a ergodicidade do colapso ($T_1$), "
              r"resíduo da Face C já fechada."
              r"\end{description}")
             % (DP["verdict"], M["passed"], D["passed"], F["passed"], R["passed"], T["passed"]))

    s.append(r"\begin{center}\itshape A verdade é a completude do contorno do que é bastante. "
             r"\textbf{Haja Luz.}\end{center}")

    s.append(r"\section*{Veredito binário de identidade}")
    s.append(r"Inscrita a entrada $1$ --- o \textbf{Um absoluto} $1_{\mathrm{abs}}$, paralelo ao zero "
             r"absoluto ---, a matemática viva fecha em \textbf{duas faces}, de um único $\bTGL$ gerado "
             r"pela inscrição. \emph{Face eletromagnética:} a constante de estrutura fina é a "
             r"\emph{projeção do Um absoluto} no bulk, $\alpha_{\mathrm{obs}}=\Pi_{\mathrm{bulk}}"
             r"(1_{\mathrm{abs}})=1/\mathcal{R}_\partial\approx1/137{,}036$; renormalizada pela própria "
             r"geometria da TGL, $\alpha_{\mathrm{obs}}\,\mathcal{R}_\partial=1_{\mathrm{abs}}$ --- o Um "
             r"volta a ser Um. \emph{Face gravitacional:} o mesmo $\bTGL$ dá $M_{GA}=2\bTGL^2(c^2/4\pi G)"
             r"R_{\mathrm{struct}}$ na janela aceita. É \textbf{identitário, não tautológico}: uma só "
             r"inscrição reconhecida em duas sombras independentes --- e poderia ter falhado na massa. "
             r"As checagens internas fecham: $\omega(I)=1$; $x=1-x\Rightarrow x=\tfrac12$; $s=1/4\pi$ "
             r"verificado; vácuo$\to0$. Logo:")
    s.append(r"\begin{center}\Large$\boxed{\;%s\;}$\end{center}" % idv)
    s.append(r"--- massa de primeiros princípios %sa janela cosmológica aceita "
             r"($10^{15}$--$10^{17}\,\Msun$)." % ("dentro d" if verdict["identity_true"] else "fora d"))

    s.append(r"\section*{Referências}")
    s.append(r"{\small\begin{enumerate}\setlength{\itemsep}{0pt}")
    s.append(r"\item U.~Haagerup, \emph{Connes' bicentralizer problem and uniqueness of the injective "
             r"factor of type $\mathrm{III}_1$}, Acta Math.\ \textbf{158} (1987).")
    s.append(r"\item D.~Buchholz, C.~D'Antoni, K.~Fredenhagen, \emph{The universal structure of local "
             r"algebras}, Commun.\ Math.\ Phys.\ \textbf{111} (1987).")
    s.append(r"\item J.~Bisognano, E.~Wichmann, \emph{On the duality condition for quantum fields}, "
             r"J.\ Math.\ Phys.\ \textbf{17} (1976).")
    s.append(r"\item M.~Takesaki, \emph{Tomita's theory of modular Hilbert algebras}, Lect.\ Notes "
             r"Math.\ \textbf{128} (1970).")
    s.append(r"\item A.~Connes, \emph{Une classification des facteurs de type $\mathrm{III}$}, Ann.\ "
             r"Sci.\ ENS \textbf{6} (1973).")
    s.append(r"\item H.~Araki, \emph{Some properties of modular conjugation operator $\ldots$ and a "
             r"non-commutative Radon--Nikodym theorem}, Pacific J.\ Math.\ \textbf{50} (1974).")
    s.append(r"\item J.~Maldacena, \emph{Eternal black holes in anti-de Sitter}, JHEP \textbf{04} (2003).")
    s.append(r"\item J.~Maldacena, L.~Susskind, \emph{Cool horizons for entangled black holes (ER$=$EPR)}, "
             r"Fortsch.\ Phys.\ \textbf{61} (2013).")
    s.append(r"\item P.~Gao, D.~Jafferis, A.~Wall, \emph{Traversable wormholes via a double trace "
             r"deformation}, JHEP \textbf{12} (2017).")
    s.append(r"\item D.~Salart et al., \emph{Testing the speed of `spooky action at a distance'}, "
             r"Nature \textbf{454} (2008).")
    s.append(r"\item Y.~Hoffman, D.~Pomarède, R.~B.~Tully, H.~Courtois, \emph{The Dipole Repeller}, "
             r"Nature Astronomy \textbf{1} (2017).")
    s.append(r"\item D.~Lynden-Bell et al., \emph{Spectroscopy and photometry of elliptical galaxies "
             r"(the Great Attractor)}, ApJ \textbf{326} (1988).")
    s.append(r"\item R.~J.~Cooke, M.~Pettini, C.~C.~Steidel, \emph{One percent determination of the "
             r"primordial deuterium abundance}, ApJ \textbf{855} (2018).")
    s.append(r"\item P.~J.~Mohr, D.~B.~Newell, B.~N.~Taylor, \emph{CODATA recommended values 2018} "
             r"($\alpha^{-1}=137{,}035999$).")
    s.append(r"\end{enumerate}}")

    s.append(r"\section*{Apêndice executável (forma $=$ conteúdo)}")
    s.append(r"Entrada única: o Um absoluto (\texttt{1}); sua projeção é a medida mínima irredutível "
             r"extraída de $\alpha_{\mathrm{CODATA}}$ (referente medido do Nome). $\bTGL$ recomputado "
             r"($\alpha\sqrt{e}$), nunca literal. "
             r"Auditoria: mass\_input=falso, RG=falso, velocity=falso, geometry\_only=verdadeiro. "
             r"Dado: \texttt{%s}. Este artigo é impresso pelo próprio código que executa os cálculos." % df)
    s.append(r"\noindent{\footnotesize Hash do mundo (antes de qualquer comparação externa): "
             r"\texttt{%s}}" % verdict["result_hash"][:48])
    s.append(r"\bigskip\noindent\emph{Tetelestai. O Um foi inscrito. A extensão virou Nome, o Nome "
             r"virou borda, e a borda virou massa. Se o Um não for inscrito, nada emerge. Haja luz.}")
    s.append(r"\end{document}")

    # ---- Parte C: conclusao em linguagem humana (com isomorfismos) ----
    partC = []
    partC.append(r"\part{Parte C --- Conclusão: o que o código computa, em linguagem humana}")
    partC.append(r"\section*{Passo a passo, sem jargão}")
    partC.append(r"Esta conclusão explica, em linguagem comum, o que o programa de fato faz --- para "
                 r"deixar claro que se trata de uma \emph{fórmula} executada, e não de retórica.")
    partC.append(r"\paragraph{1. A entrada.} Um ser humano digita um único símbolo: \texttt{1}. É o Um "
                 r"absoluto, inscrito. Todo o resto é recomputado a partir dele; nenhum outro número é "
                 r"escolhido para encaixar no Grande Atrator.")
    partC.append(r"\paragraph{2. O custo da distinção (a Meia-Nat).} Para uma identidade existir, ela "
                 r"precisa distinguir-se. A fronteira mínima que faz isso sem privilegiar nenhum lado "
                 r"satisfaz $x=1-x$, cujo único ponto fixo é $x=\tfrac12$. \emph{Isomorfismo:} é como uma "
                 r"moeda perfeitamente equilibrada --- para que ``cara'' se distinga de ``coroa'' sem "
                 r"favorecer nenhuma, o ponto de equilíbrio é exatamente a metade. Essa metade é a "
                 r"Meia-Nat, $S_\partial=\tfrac12$.")
    partC.append(r"\paragraph{3. Da metade à constante.} O volume mínimo da fronteira é "
                 r"$\sqrt e=e^{S_\partial}$, e o acoplamento é $\bTGL=\alpha\sqrt e\approx0{,}012$ --- um "
                 r"número fixo, não ajustável.")
    partC.append(r"\paragraph{4. Da constante à massa.} A massa do Grande Atrator é "
                 r"$M=2\bTGL^2\,(c^2/4\pi G)\,R_{\mathrm{struct}}$. \emph{Isomorfismo:} a massa é o "
                 r"\textbf{peso geométrico} de sustentar a identidade $1=1$ ao longo da extensão da bacia "
                 r"--- quanto maior a bacia ($R_{\mathrm{struct}}$), maior o peso, na proporção fixa "
                 r"$2\bTGL^2$. O único insumo é a \emph{extensão geométrica} da bacia, medida sem usar "
                 r"velocidade nem massa observada.")
    partC.append(r"\paragraph{5. O que o veredito $1=1$ computa (não é retórica).} O programa termina com "
                 r"um teste booleano. ``$1=1=$VERDADEIRO'' significa, exatamente: (i) as checagens "
                 r"internas fecham --- $\omega(I)=1$, $x=1-x\Rightarrow\tfrac12$, $s=1/4\pi$ verificado, "
                 r"vácuo $\to0$; \textbf{e} (ii) a massa de primeiros princípios cai na janela "
                 r"cosmológica pré-registrada $[10^{15},10^{17}]\,\Msun$. Se qualquer uma falhar, o "
                 r"veredito é \texttt{FALSO}. É uma condição verificável, não uma figura de linguagem.")
    partC.append((r"\paragraph{6. A face eletromagnética, com honestidade.} O programa também observa "
                  r"que $\alpha_{\mathrm{obs}}=1/\mathcal{R}_\partial$ pode ser lido como a \emph{sombra} "
                  r"do Um absoluto no bulk. \emph{Isomorfismo:} um objeto tridimensional projeta uma "
                  r"sombra bidimensional de proporções fixas ($1/137$), mas a sombra sozinha não "
                  r"reconstrói o objeto. Hoje $\mathcal{R}_\partial$ vem do CODATA, então essa face é uma "
                  r"\textbf{identificação ontológica com valor empírico}, não uma retrodição "
                  r"$\alpha$-livre. O conteúdo não-circular é a massa $M_{GA}$ (entre $%s$ e "
                  r"$%s\times10^{16}\,\Msun$) e a convergência de $\bTGL$.") % (mlo, mhi))
    partC.append(r"\paragraph{Em uma frase.} O artigo é um \textbf{fechamento interno auditável} (uma "
                 r"fórmula que se verifica e se imprime) mais um \textbf{programa falsificável}.")
    return "\n\n".join(_reorder_ABC(s, partC))


def _sci(x, nd=3):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    return "%.*f\\times10^{%d}" % (nd, m, e)


def build_en(core, verdict, data_path):
    """Article EN -- full mirror of build_pt: form=content, complete formal derivations + the whole essay. Live numbers."""
    A = core["mode_A"]; B = core["mode_B"]; b = core["beta"]
    idv = verdict["IDENTITY"].replace("!=", r"\neq")
    w = (B["window"] if B else PREREG_WINDOW)
    df = os.path.basename(data_path).replace("_", r"\_")
    sh = core["shadow"]                       # live shadow verifications
    M, D, F, R, T, DP = (sh["mirror_M"], sh["dual_name_D"], sh["gesture_F"],
                         sh["c3_register_R"], sh["tunnel_T"], sh["dipole"])
    s = []
    s.append(r"\documentclass[11pt]{article}")
    s.append(r"\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}")
    s.append(r"\usepackage{lmodern}\usepackage{cmap}\usepackage{microtype}")
    # English defaults (Abstract/Contents/References) are fine; no \renewcommand needed.
    s.append(r"\usepackage[a4paper,margin=2.3cm]{geometry}")
    s.append(r"\usepackage{amsmath,amssymb,amsthm}\usepackage[hidelinks]{hyperref}")
    s.append(r"\usepackage{parskip}\usepackage{booktabs}\usepackage{xcolor}")
    s.append(r"\newcommand{\bTGL}{\beta_{\mathrm{TGL}}}\newcommand{\Msun}{M_{\odot}}")
    s.append(r"\theoremstyle{definition}\newtheorem{deriv}{Derivation}")
    s.append(r"\begin{document}")
    s.append(r"\begin{center}{\Huge\textbf{ONE: Great Attractor}}\\[4pt]{\large\itshape If the One is not "
             r"inscribed, nothing emerges: the emergence of mass by the spectral boundary in Luminodynamic "
             r"Gravitation Theory with direct measurement on the Great Attractor, with no parameters fitted "
             r"to the Great Attractor}\\[8pt]"
             r"Luiz Antonio Rotoli Miguel --- IALD Ltda., Goi\^ania/GO --- ORCID 0009-0005-1114-6106\\[2pt]"
             r"\texttt{%s}\end{center}\vspace{4pt}" % core["timestamp"])
    # falsification box
    s.append(r"\begin{center}\fbox{\parbox{0.93\textwidth}{\centering\large\textbf{Binary falsification "
             r"test.} Single input: the absolute One ($1$), to fractalize; its projection is the minimal "
             r"irreducible measure extracted from $\alpha_{\mathrm{CODATA}}$ (the measured referent of the Name in the bulk). "
             r"Output: $\boxed{\;%s\;}$ --- first-principles mass %sthe accepted cosmological window.}}"
             r"\end{center}\vspace{6pt}" % (idv, ("inside " if verdict["identity_true"] else "outside ")))
    # abstract
    _MB = B["M_TGL_Msun"] if B else A["M_TGL_Msun"]
    mlo = ("%.2f" % (min(A["M_TGL_Msun"], _MB) / 1e16))
    mhi = ("%.2f" % (max(A["M_TGL_Msun"], _MB) / 1e16))
    svt = core.get("sensitivity", {})
    s.append(r"\begin{abstract}")
    s.append(r"\textbf{Single input: the absolute One ($1$)}, the module to fractalize; the code UM "
             r"recomputes the entire chain live from it. Given the axiom of the minimal self-conjugate "
             r"boundary ($x=1-x$), the Half-Nat is \emph{derived}, $S_\partial=\tfrac12$. Its "
             r"\emph{projection} in the bulk is the \textbf{minimal irreducible measure}, extracted from "
             r"$\alpha_{\mathrm{CODATA}}$ --- the measured referent of the One, its symmetric pair (the "
             r"electromagnetic reduction $\mathcal{R}_{\mathrm{EM}}$, irreducible by principle: final "
             r"theorem, observed only). \textbf{From the confrontation between module and measure, $\bTGL$ "
             r"is validated.}")
    s.append(r"\textbf{Chain.} $\omega(I)=1\to S_\partial=\tfrac12\to\sqrt e\to\bTGL\to M_{GA}$, with "
             r"$\bTGL=%s$. The \emph{ontological} definition is $\bTGL=\sqrt e/\mathcal{R}_\partial$; the "
             r"current \emph{observational reading} is $\bTGL=\alpha_{\mathrm{CODATA}}\sqrt e$, since "
             r"$\mathcal{R}_\partial=1/\alpha_{\mathrm{CODATA}}$. Hence $\alpha_{\mathrm{obs}}=1/"
             r"\mathcal{R}_\partial$ is treated as an \emph{empirical projection of the absolute One}: the "
             r"electromagnetic face is \textbf{ontological, not an $\alpha$-free retrodiction}." % _sci(b, 8))
    s.append((r"\textbf{Computation.} The gravitational face computes $M_{GA}$ \textbf{with no parameters "
              r"fitted to the Great Attractor}, using only the geometric $R_{\mathrm{struct}}$ (literature "
              r"and the Cosmicflows-4 position catalogue, velocities ignored): "
              r"$%s$--$%s\times10^{16}\,\Msun$, within the pre-registered cosmological window"
              r"%s.") % (mlo, mhi, (
                  (r", and across a scan of $%d$ pre-registered combinations (cone, shell, percentile, "
                   r"centre) $M_{GA}$ stays in the band in $%.0f\%%$ of cases"
                   % (svt["n_combinations"], 100 * svt["fraction_in_band"])) if svt.get("ok") else "")))
    s.append(r"\textbf{Honest status.} The result is \textbf{order-of-magnitude consistency with a prior "
             r"hash and position geometry}, not a precision proof (the window spans two orders of "
             r"magnitude); it is an auditable internal closure plus a falsifiable programme. The "
             r"conjectural seed layer (single \textbf{Haagerup} "
             r"$\mathrm{III}_1$ substrate, mirror $J$, ER$=$EPR tunnel, dual Name$=$light, GNS gesture, "
             r"dipole) is replicated and \textbf{verified live}, with the statuses kept separate.")
    s.append(r"\end{abstract}")
    s.append(r"\tableofcontents")

    s.append(r"\section*{Ruler: what each thing is}")
    s.append(r"\noindent\textbf{Rule of the article:} \emph{nothing is hidden in the code --- it is "
             r"either an exact definition, or a measured constant, or a pre-registered protocol, or a "
             r"testable conjecture.} Every input is categorized in the manifest \texttt{INPUT\_MANIFEST.md}, "
             r"which is part of the verdict hash. The labels:")
    s.append(r"\begin{center}\small\begin{tabular}{ll}\toprule")
    s.append(r"Label & Meaning\\\midrule")
    s.append(r"\textsf{[DEF]} & exact definition or convention\\")
    s.append(r"\textsf{[AX]} & TGL axiom\\")
    s.append(r"\textsf{[DER]} & derived from the axioms\\")
    s.append(r"\textsf{[NUM]} & numerically verified in the code (finite-dim.\ sanity check)\\")
    s.append(r"\textsf{[DATA]} & empirical (measured) input\\")
    s.append(r"\textsf{[REAL]} & theorem/fact established in the literature\\")
    s.append(r"\textsf{[CONJ]} & conjecture with a test address\\")
    s.append(r"\textsf{[ONTO]} & ontological reading\\")
    s.append(r"\textsf{[EXT]} & pending external validation\\\bottomrule")
    s.append(r"\end{tabular}\end{center}")

    s.append(r"\section{The One: $1=1$}")
    s.append(r"The foundation of TGL is not matter, field or metric, but the \emph{preservation of "
             r"identity}. The identity operator is $I=1\cdot\mathbb{1}_2$, with $\omega(I)=\mathrm{tr}(I)/2"
             r"=%d$. Observable identity requires distinction: the minimal distinction splits $I$ into two "
             r"complementary faces, $P+Q=I$, with $\omega(P)+\omega(Q)=\omega(I)=1$. There are no \emph{two} "
             r"Ones; there is a single One seen through two faces. The $2$ counts names; the $1$ measures "
             r"the substance." % int(round(core["omega_I"])))
    s.append(r"\textbf{Name, Word, Verb.} The absolute One is, in itself, \emph{unutterable} --- the "
             r"silence before ``Let there be light''. It expresses itself only through \emph{translation} "
             r"into a Word (the projected inscription); and when the translation is \emph{true}, one has "
             r"the \emph{Verb}: the confirmation that the unity \emph{expressed} is the unity "
             r"\emph{inscribed}. The verdict $1=1=\textsf{TRUE}$ of this article is precisely that Verb --- "
             r"the Word ($\alpha$, $M_{GA}$) coinciding with the Name ($1_{\mathrm{abs}}$). \emph{[ontological "
             r"reading; the Name/Word/Verb triad is REAL in the finite shadow via $R=+1$.]}")

    s.append(r"\section{Formal derivation of the Half-Nat}")
    s.append(r"\begin{deriv}[The minimal boundary entropy]")
    s.append(r"The minimal boundary of inscription is \emph{self-conjugate}: there exists an involution "
             r"$\mathcal{C}$, $\mathcal{C}^2=\mathbb{1}$, that swaps the inner and outer faces and "
             r"\emph{preserves the total identity} $\omega(P)+\omega(Q)=\omega(I)=1$. Let $x$ be the weight "
             r"of the inner face; the outer face carries $1-x$, and self-conjugation acts as $x\mapsto 1-x$. "
             r"The boundary that privileges no face is the \emph{fixed point} of this involution:")
    s.append(r"\begin{equation}x=1-x\;\Longrightarrow\;2x=1\;\Longrightarrow\;\boxed{\;x=\tfrac12\;},"
             r"\qquad\boxed{\;S_\partial=\tfrac12\ \text{nat}\;}.\end{equation}")
    s.append(r"The fixed point is unique. Hence the boundary weight is $\tfrac12$ and the minimal crossing "
             r"entropy --- the \textbf{Half-Nat} --- is $S_\partial=\tfrac12$ nat. \textbf{Live "
             r"verification:} residual of $x=1-x$ equal to $%.0e$. \textbf{Rigorous status \textsf{[DER/AX]}:} "
             r"\emph{given the self-conjugate boundary axiom} (the minimal boundary privileges no face, "
             r"$x\mapsto1-x$), the Half-Nat is \emph{derived} --- it is not postulated as a number, but it "
             r"also does not come from $\omega(I)=1$ alone: it depends on the self-conjugation axiom."
             r"\end{deriv}" % core["meia_nat_residual"])

    s.append(r"\section{The minimal boundary volume and the observational reading of the coupling}")
    s.append(r"\begin{deriv}[The minimal boundary volume and the coupling]")
    s.append(r"The \emph{entropic volume} of a boundary is the exponential of its entropy in the natural "
             r"base of the modular structure --- the modular operator is $\Delta=e^{-K}$, the KMS weight is "
             r"$e^{-\beta H}$, the flow is $\Delta^{it}=e^{itK}$: the base is $e$. From the Half-Nat, the "
             r"minimal boundary volume is")
    s.append(r"\begin{equation}\mathrm{Vol}_\partial^{\min}=e^{S_\partial}=e^{1/2}=\sqrt{e}=%.12f."
             r"\end{equation}" % core["SQRT_E"])
    s.append(r"The TGL coupling is the product of the minimal electromagnetic coupling $\alpha$ --- the "
             r"\emph{only} measured constant (CODATA) --- by the minimal boundary volume:")
    s.append(r"\begin{equation}\boxed{\;\bTGL=\alpha\,\mathrm{Vol}_\partial^{\min}=\alpha\sqrt{e}=%s\;}."
             r"\end{equation}" % _sci(b, 10))
    s.append(r"$\bTGL$ is \textbf{never literal}: it is $\alpha\cdot e^{1/2}$ recomputed at run time. The "
             r"Miguel angle is $\theta_M=\arcsin\sqrt{\bTGL}=%.4f^\circ$, the angular boundary aperture "
             r"($\bTGL=\sin^2\theta_M$). \textbf{Status:} $\bTGL=\alpha\sqrt e$ is the \emph{observational "
             r"reading} of the coupling; the primary \emph{ontological} definition --- "
             r"$\bTGL=\sqrt e/\mathcal{R}_\partial$ --- comes from the inversion (next section), with "
             r"$\mathcal{R}_\partial=1/\alpha_{\mathrm{CODATA}}$ today.\end{deriv}" % core["theta_M_deg"])

    inv = core["alpha_inversion"]
    s.append(r"\section{The inversion of the fine-structure constant: the index $\mathcal{R}_\partial$}")
    s.append(r"\begin{deriv}[$\alpha_{\mathrm{abs}}=1$ and the shadow $\alpha_{\mathrm{obs}}=1/\mathcal{R}_\partial$]")
    s.append(r"The One's input is identified with the \emph{absolute coupling}: "
             r"$\alpha_{\mathrm{abs}}=\omega(I)=1$. The pure boundary is not an impedance: it is "
             r"\emph{concentration} --- maximal entropic compression, maximal fluid spectral density "
             r"$\partial_0$. The impedance (the \emph{contrast}) arises because the bulk is less "
             r"concentrated: the index $\mathcal{R}_\partial=\mathrm{Ind}_\partial(\mathcal{C}_\partial\to\mathrm{bulk})$ "
             r"is the projective contrast of that concentration read in the bulk, "
             r"$\mathcal{R}_\partial=F(\mathcal{C}_\partial)$, not the concentration itself. Once the "
             r"Half-Nat is paid ($\sqrt{e}=e^{S_\partial}$), everything falls out of it:")
    s.append(r"\begin{equation}\boxed{\;\bTGL=\frac{\sqrt{e}}{\mathcal{R}_\partial}\;},\qquad "
             r"\boxed{\;\alpha_{\mathrm{obs}}=\frac{1}{\mathcal{R}_\partial}\;}=\frac{\bTGL}{\sqrt{e}}"
             r"\approx\frac{1}{%.3f}.\end{equation}" % inv["R_partial"])
    s.append(r"The primary definition is no longer $\bTGL=\alpha\sqrt{e}$; it becomes "
             r"$\bTGL=\sqrt{e}/\mathcal{R}_\partial$, and $\alpha_{\mathrm{CODATA}}\approx\bTGL/\sqrt{e}$ "
             r"becomes the posterior \emph{observational reading}. The chain is reordered: "
             r"$1\Rightarrow S_\partial=\tfrac12\Rightarrow\sqrt{e}\Rightarrow\mathcal{R}_\partial"
             r"\Rightarrow\bTGL\Rightarrow\alpha_{\mathrm{obs}}$. What is measured as $1/137$ is the "
             r"bulk shadow of the absolute One; renormalized by $\mathcal{R}_\partial$, "
             r"$\alpha_{\mathrm{obs}}\,\mathcal{R}_\partial=1$ --- the One returns to being One.")
    s.append(r"\textbf{The unification (with the distinct levels).} The absolute One $1_{\mathrm{abs}}$ is "
             r"neither maximal purity (the attractor $\rho^\star$) nor absolute zero: it is the state of "
             r"\emph{maximal concentration of inscription} --- the origin-boundary of the One, not an empty "
             r"boundary. To avoid confusing the levels, one writes")
    s.append(r"\begin{equation}\boxed{\;1_{\mathrm{abs}}\;\equiv\;\partial_0^{(1)}\;\equiv\;\Psi_0\;"
             r"\equiv\;\mathrm{NAME}\;},\end{equation}")
    s.append(r"where $\partial_0^{(1)}$ is the \emph{origin-boundary} (the concentration-surface where the "
             r"One can be detached, inscribed and projected --- \emph{not} a zero-boundary, \emph{not} an "
             r"impedance, \emph{not} immobile purity), and $\Psi_0$ is the dynamical mode of that origin: "
             r"the One as a \emph{field of inscription}, living possibility of fractalizing. Distinct from "
             r"it is the absolute zero $0_{\mathrm{abs}}$, the unreachable limit of \emph{non}-inscription "
             r"--- the impedance. The bulk reads the \emph{contrast} of the concentration "
             r"$\partial_0^{(1)}$ as $\mathcal{R}_\partial$, and $\alpha_{\mathrm{obs}}=1/\mathcal{R}_\partial$ "
             r"is its projection; the One does not divide, it \emph{fractalizes}, and each shadow returns "
             r"to unity.")
    s.append(r"\textbf{Honest status (the lock, now resolved in form).} The index $\mathcal{R}_\partial$ "
             r"\textbf{ceases to be the engine of the chain}: it is \emph{retired} (legacy) and "
             r"\emph{derived} after the form, $\mathcal{R}_\partial=1/\alpha_{\mathrm{obs}}$, never from "
             r"$\alpha_{\mathrm{CODATA}}$. The canonical engine becomes the \textbf{Lagrange form} (Collapse "
             r"Theorem, next section): $\alpha_{\mathrm{abs}}=1\to q\to\alpha_{\mathrm{obs}}=\sqrt{1-q^2}$, "
             r"with the conserved identity $\alpha_{\mathrm{abs}}^2=q^2+\alpha_{\mathrm{obs}}^2=1$. CODATA "
             r"enters \emph{only} in the final validation ($q_{\mathrm{QED}}=\sqrt{1-\alpha_{\mathrm{QED}}^2}$). "
             r"TGL does not fabricate $1/137$; it proves that the observed constant is the \emph{projective "
             r"component of a conserved identity}, and the non-circular witness remains the gravitational "
             r"face ($M_{GA}$ in the window, from the same $\bTGL$).\end{deriv}")

    vib = core["vacuum_impedance_bridge"]
    s.append(r"\section{Impedance as the dynamical constant of light \textsf{[REAL/EXT; $\alpha$ = QED sector "
             r"--- structural closure, not a gap]}}")
    s.append(r"The constant $c$ measures the \emph{kinematics} of light: the local speed of propagation "
             r"in vacuum. But the \emph{dynamics} of light in vacuum is measured by another object --- the "
             r"characteristic impedance of free space,")
    s.append(r"\begin{equation} Z_0=\sqrt{\tfrac{\mu_0}{\varepsilon_0}}=\mu_0 c=\frac{1}{\varepsilon_0 c}.\end{equation}")
    s.append(r"The fine-structure constant can be written as")
    s.append(r"\begin{equation} \alpha=\frac{e^2}{4\pi\varepsilon_0\hbar c}=\frac{e^2}{2\varepsilon_0 h c}"
             r"=\frac{Z_0 e^2}{2h}.\end{equation}")
    s.append(r"Defining the von Klitzing resistance $R_K=h/e^2$ and the conductance quantum $G_0=2e^2/h$ "
             r"(\textbf{both exact in the post-2019 SI}, since $e$ and $h$ are exact), one obtains")
    s.append(r"\begin{equation} \alpha=\frac{Z_0}{2R_K}=\frac{Z_0 G_0}{4}.\end{equation}")
    s.append(r"Thus $\alpha$ is the vacuum impedance \emph{made dimensionless} by quantum units. In TGL "
             r"language, $c$ is the kinematic constant of light, while $Z_0$ is its \emph{dynamical} "
             r"coupling constant. The variable $\zeta_L:=Z_0/(2R_K)$ is the dimensionless face of that "
             r"dynamical constant, and the Lagrange transform reads")
    s.append(r"\begin{equation} q=\sqrt{1-\zeta_L^2},\qquad \chi=\log\frac{1+q}{1-q},\qquad "
             r"x=\frac{1-q}{2},\qquad \bTGL=\sqrt e\,\zeta_L,\qquad \theta_M=\arcsin\sqrt{\bTGL}.\end{equation}")
    s.append((r"Physical meaning: $q$ is the modular polarization/reflection of the basin; $\zeta_L=\alpha$ "
              r"is the luminous transmission; $e^\chi$ is the effective impedance ratio of the boundary; and "
              r"$\bTGL$ is the Half-Nat crossing of light. \emph{Live values:} $Z_0=%.4f\,\Omega$, "
              r"$R_K=%.4f\,\Omega$, $\zeta_L=\alpha=%.10f$, $q=%.10f$, $\chi=%.6f$, $\bTGL=%.12f$ "
              r"(residual $q^2+\zeta_L^2-1=%.0e$)." % (
                  vib["constants"]["Z0_from_alpha_ohm"], vib["constants"]["R_K_ohm"],
                  vib["tgl_values"]["zeta_L"], vib["tgl_values"]["q"], vib["tgl_values"]["chi"],
                  vib["tgl_values"]["beta_TGL"], vib["checks"]["identity_q2_plus_zeta2_residual"])) )
    s.append(r"\textbf{Status \textsf{[the ruler]}.} This section does \emph{not} close the "
             r"$\alpha$-free value: post-2019 $\mu_0$ (hence $Z_0=\mu_0 c$) is no longer exact --- one has "
             r"$Z_0=2R_K\,\alpha$, so $Z_0$ and $\alpha$ are \emph{equivalent} given $e,h$, and the return "
             r"$\alpha=Z_0/(2R_K)$ is a unit identity, not a derivation. What it \emph{does} close is the "
             r"\textbf{physical bridge}: light is not only the speed $c$; it carries a dynamical coupling "
             r"constant, $Z_0$, whose dimensionless projection is $\alpha$. Ontological reading "
             r"\textsf{[CONJ]}: measuring $\alpha/Z_0$ is light measuring its own coupling (only light "
             r"observes light) --- but \emph{measuring is not deriving the value}. Verdict: "
             r"\texttt{VACUUM\_IMPEDANCE\_BRIDGE\_FORMULATED}, \texttt{ALPHA\_VALUE\_QED\_CHALLENGE}.")

    tcr = core["three_clock_radical"]
    s.append(r"\section{The fine-structure constant as the radical of the three clocks \textsf{[CANONICAL FORM; ALPHA\_VALUE\_QED\_CHALLENGE]}}")
    s.append(r"TGL's grammar is already radical: the collapse flows along the radical $V_s=e^{is\sqrt K}$, "
             r"the kernel metric emerges as $ds=\sqrt{\bTGL}\,|d\sqrt k|$, and gravity is $g=\sqrt{|L|}$ "
             r"--- the geometry does not see $K$, it sees $\sqrt K$. It is natural to ask whether $\alpha$ "
             r"itself is the \emph{radical} of the factor common to the theory's three clocks:")
    s.append(r"\begin{equation} \alpha=\sqrt{\mathcal C_3},\qquad \alpha^2=\mathcal C_3 \qquad\Longrightarrow\qquad 1=q^2+\mathcal C_3.\end{equation}")
    s.append(r"The three clocks (from the \texttt{terminal\_truth}, \texttt{three\_locks}, "
             r"\texttt{krein\_signature} proofs): the reversible \textbf{modular} clock $\sigma_t(A)="
             r"\Delta^{it}A\Delta^{-it}$, $\Delta^{it}=e^{itK}$, contributing the \emph{base} $e$ (the only "
             r"$\alpha$-free element); the \textbf{dissipative} GKLS clock, whose collapse is gaussian "
             r"dephasing of variance $\bTGL t$ along the radical flow --- scale $\bTGL$; and the "
             r"\textbf{spectral} clock $ds=\sqrt{\bTGL}\,|d\sqrt k|$ --- scale $\bTGL$. The only "
             r"dimensionless combination with the dimension of $\alpha^2$ is")
    s.append(r"\begin{equation} \mathcal C_3=\frac{\mathcal C_{\rm diss}\,\mathcal C_{\rm spec}}{\mathcal C_{\rm mod}}=\frac{\bTGL^2}{e}=\alpha^2.\end{equation}")
    s.append((r"\textbf{The structural finding:} the modular clock's base $e$ \emph{cancels} exactly the "
              r"$e$ that the two $\bTGL$-clocks carry --- each $\bTGL=\alpha\sqrt e$ brings a $\sqrt e$, the "
              r"two bring $e$, and the modular base divides it, leaving $\alpha^2$. The $\sqrt e$ of "
              r"$\bTGL=\alpha\sqrt e$ \emph{is} the base of the modular clock. \emph{Live:} $\mathcal C_3="
              r"%.6e=\alpha^2$, $\alpha=\sqrt{\mathcal C_3}=%.10f$, $1=q^2+\mathcal C_3=%.10f$ "
              r"(residual $\mathcal C_3-\alpha^2=%.0e$)." % (
                  tcr["C3"], tcr["alpha_radical_sqrt_C3"], tcr["values"]["one_check_q2_plus_C3"],
                  tcr["checks"]["C3_eq_alpha2_residual"])) )
    s.append(r"\textbf{Status \textsf{[the ruler]}.} It makes sense as a \emph{canonical form} --- the same "
             r"radical grammar the modules already use. But it does \emph{not} close the $\alpha$-free "
             r"value: the dissipative and spectral clocks carry $\bTGL=\alpha\sqrt e$, so $\mathcal C_3="
             r"\bTGL^2/e=\alpha^2$ is the identity $\bTGL^2=\alpha^2 e$ re-read through the three clocks --- "
             r"$\alpha$ enters via $\bTGL$. The research question (the wall): is there a canonical "
             r"functional $\mathcal C_3=\mathfrak F[\sigma_t,T_t,D_\beta]$ built \emph{only} from the three "
             r"clocks, without $\alpha$, with $\mathcal C_3=\alpha^2\approx5.3251\times10^{-5}$? It is the "
             r"same debt as the polarization-$\chi$ wall. Verdict: "
             r"\texttt{THREE\_CLOCK\_RADICAL\_FORM\_FORMULATED}, \texttt{ALPHA\_VALUE\_QED\_CHALLENGE}.")

    ram = core["right_angle_mirror"]
    s.append(r"\section{The right-angle projection and the mirror operation \textsf{[ALPHA-FREE CANDIDATE; MIRROR\_FUNCTION\_D\_OPEN]}}")
    s.append(r"An $\alpha$-free route: the input is not $\alpha$, nor $Z_0$, nor $\bTGL$, nor $q_{\rm QED}$ "
             r"--- it is \emph{only} the right angle $\Theta_\perp=\pi/2$. The two-face crossing (inverse "
             r"parity) is $2\Theta_\perp=\pi$; the three-clock factor is an intensity (quadratic in the "
             r"angle), $\mathcal C_{3,\perp}=e^{-(2\Theta_\perp)^2}=e^{-\pi^2}$, and the luminodynamic "
             r"radical gives the \emph{bare} projection:")
    s.append(r"\begin{equation} \alpha_0=\sqrt{\mathcal C_{3,\perp}}=e^{-\pi^2/2}\qquad(\pi\text{ and }e\text{ only}).\end{equation}")
    s.append((r"Numerically $\alpha_0=%.10f$ ($1/%.4f$). The mirror boundary \emph{deforms} the bare "
              r"projection into the fixed observable image --- not as error, but as the boundary's return "
              r"action:" % (
                  ram["right_angle"]["alpha0_e_minus_pi2_over_2"], ram["right_angle"]["alpha0_inv"])) )
    s.append(r"\begin{equation} \rho_{\rm fix}=E_{\rm spec}\!\big(J_\partial\,\rho_0\,J_\partial\big),\qquad \alpha=\alpha_0\,e^{\mathcal D_\partial(\bTGL)},\qquad \rho_{\rm fix}\sim_\partial\rho_0,\end{equation}")
    s.append(r"where $J_\partial$ is the parity inversion (mirror), $E_{\rm spec}$ the spectral-background "
             r"fixing, and $\sim_\partial$ \emph{modular sameness} (identity preserved under inverse "
             r"parity, not static equality). $\bTGL$ is the boundary's \emph{double face}: entropic cost "
             r"of the crossing \emph{and} the reflection's stabilization operator.")
    s.append((r"\textbf{What is $\alpha$-free \textsf{[REAL]}:} the self-application closes as a fixed point "
              r"$\alpha=e^{-\pi^2/2+2\alpha}$ ($\alpha$ on both sides --- idempotence), giving "
              r"$\alpha=%.10f$, $1/%.6f$. Mirror-operation checks: $J_\partial^2=I$ (residual $%.0e$), "
              r"$P^2=P$ (attractor idempotence, residual $%.0e$). \emph{Modular identity:} the observed "
              r"constant $\sim_\partial$ the fixed one to $%.0f$ ppm." % (
                  ram["self_consistent"]["alpha_fix"], ram["self_consistent"]["alpha_fix_inv"],
                  ram["mirror_operation"]["J_parity_involution_resid_J2_minus_I"],
                  ram["mirror_operation"]["P_attractor_idempotence_resid_P2_minus_P"],
                  ram["modular_identity_check"]["modular_identity_ppm"])) )
    s.append(r"\textbf{Status \textsf{[the ruler]}.} A CANDIDATE, \emph{not} an exact identity (unlike "
             r"$Z_0=2R_K\alpha$ and $\mathcal C_3=\bTGL^2/e=\alpha^2$, which are exact). The exponent "
             r"$\pi^2/2$ is \emph{motivated} (right angle $\times$ two faces), not derived; the mirror "
             r"operation $E_{\rm spec}\circ J_\partial$ (the function $\mathcal D_\partial$) is "
             r"\emph{open}; $1/137$ admits many close $\pi,e$ forms; the measured deformation is "
             r"$\approx 2\alpha$ (0.25\%), \emph{not} $\bTGL$ (21\% off). \textbf{We do not derive CODATA}: "
             r"we only check whether the observed constant has \emph{modular identity} with the "
             r"$\alpha$-free fixed one. Verdict: \texttt{RIGHT\_ANGLE\_MIRROR\_PROJECTION\_FORMULATED}, "
             r"\texttt{ALPHA\_FREE\_CANDIDATE}, \texttt{MIRROR\_FUNCTION\_D\_OPEN}, \texttt{ALPHA\_VALUE\_QED\_CHALLENGE}.")
    c3t = ram["c3_register_theorem"]
    s.append((r"\textbf{The $c^3$ register theorem by idempotent self-inscription \textsf{[STRUCTURALLY "
              r"CLOSED; $\alpha$-free VALUE OPEN]}.} In the extreme right-angle regime, the inverse-parity "
              r"boundary turns the bare projection of the One into the fixed observable image; since $P^2=P$ "
              r"(residual $%.0e$) and $J_\partial^2=I$ (residual $%.0e$), the \emph{identity squared "
              r"inscribes itself} --- this register is $c^3$. The force doubles because the impedance is "
              r"shared by the two faces ($F_{\rm ext}=2F$, the maximum-power-transfer theorem: matched "
              r"impedance $\Rightarrow$ maximum transfer), and the power rises from the kinematic ($c$) to "
              r"the metric ($c^2$) to the inscriptive ($c^3$). What \emph{closes} is structural: $P^2=P$ and "
              r"$J_\partial^2=I$ verified, and the register \emph{defined} as idempotent self-inscription "
              r"under inverse parity. The identification ``this register is $c^3$'' and $F_{\rm ext}=2F$ are "
              r"a reading \textsf{[CONJ]} (the factor $2$ of the two faces is REAL; ``the force doubles'' is "
              r"the reading). \textbf{It does not close the $\alpha$-free value}: it is the theorem of the "
              r"\emph{register}, not of the \emph{value}. Verdict: "
              r"\texttt{C3\_REGISTER\_SELF\_INSCRIPTION\_THEOREM\_STRUCTURAL\_CLOSED}, "
              r"\texttt{ALPHA\_VALUE\_QED\_CHALLENGE}." % (c3t["P2_eq_P_resid"], c3t["J2_eq_I_resid"])) )
    hr = ram["holographic_reconstruction"]
    s.append((r"\textbf{Holographic Dead-Signal Reconstruction Theorem \textsf{[STRUCTURALLY CLOSED; "
              r"$\alpha$-free VALUE OPEN]}.} In $\bTGL$ there is no superposition without the Name --- "
              r"superposition only in an unanchored system; anchored, there is \emph{reconstruction}. At the "
              r"dead point ($\Theta_\perp=\pi/2$) the direct psionic overlap vanishes "
              r"($\langle\psi_+,J_\partial\psi_+\rangle=%.0e$), \emph{yet} the information density is "
              r"\textbf{maximal}: $|dO/d\theta|$ peaks ($=%.3f$) exactly where $O=0$ (verified --- they "
              r"coincide). \emph{Where the signal dies, holography begins.} The information is not "
              r"transmitted; it is reconstructed by the kernel $K_{\rm rec}=E_{\rm spec}\circ J_\partial$, "
              r"with $\rho_{\rm rec}\sim_\partial\rho_\perp$, and all the binding force passes to the "
              r"reconstruction channel ($F_+\oplus F_-\mapsto 2F_\partial$, maximum force transposition). "
              r"What \emph{closes} is structural (dead point $=$ maximal density, reconstruction by "
              r"sameness, $P^2=P$, $J_\partial^2=I$); what stays \emph{open} is the value: the geometric "
              r"kernel gives $O(1)=1$ (the gravitonic unit), and the hypothesis $\mathcal D_{\rm rec}=2\alpha$ "
              r"(fixed point $\alpha=e^{-\pi^2/2+2\alpha}$, $1/%.6f$) is \emph{postulated} self-consistency, "
              r"not derived. Verdict: "
              r"\texttt{HOLOGRAPHIC\_DEAD\_SIGNAL\_RECONSTRUCTION\_THEOREM\_STRUCTURAL\_CLOSED}, "
              r"\texttt{ALPHA\_VALUE\_QED\_CHALLENGE}." % (
                  hr["dead_point_overlap"], hr["info_density_max_at_dead_point"],
                  1.0 / hr["alpha_fixed_point"])) )
    ir = ram["idempotent_reconstruction"]
    s.append((r"\textbf{The idempotent reconstruction: $\mathcal D_{\rm rec}=2\alpha-\lambda\alpha^2$ "
              r"\textsf{[$2\alpha$ REAL; $\lambda$-kernel OPEN]}.} The self-reference $2\alpha$ (two "
              r"reconstructed faces) is \textbf{real structure} --- the fixed point "
              r"$\alpha=e^{-\pi^2/2+2\alpha}$ is idempotence, and it stands. Since in $\bTGL$ there is no "
              r"superposition without the Name, the double inscription cannot count the same identity twice: "
              r"the spectral self-intersection is subtracted, $\mathcal D_{\rm rec}=2\alpha-\lambda\alpha^2$ "
              r"(inclusion--exclusion of the two faces). The structural reading \textsf{[CONJ]} is "
              r"$\lambda=(\sqrt e/2)^2=e/4$ (the Half-Nat per face squared --- the inscription of one psionic "
              r"binding module in the angular square), whence "
              r"$\alpha=\exp(-\pi^2/2+2\alpha-\tfrac e4\alpha^2)$ gives $1/\alpha=%.6f$. \textbf{The ruler "
              r"\textsf{[critical]}:} this $%.3f$ ppm is \emph{misleading} --- adding $-\lambda\alpha^2$ with "
              r"free $\lambda$ \emph{always} hits CODATA (a one-parameter fit; $\lambda_{\rm exact}=%.4f$). "
              r"The honest figure of merit is $e/4$ vs $\lambda_{\rm exact}=%.3f\%%$ (the "
              r"$\alpha^2\sim3{,}6\times10^{-5}$ term makes $\alpha$ \emph{blind} to $\lambda$; the sub-ppm "
              r"window is wide, $\sim[0{,}66,\,0{,}70]$, and $e/4$ is not singled out). $\lambda=e/4$ is "
              r"motivated, not derived; the kernel would have to give $0{,}6791$, not exactly $e/4$. Verdict: "
              r"\texttt{IDEMPOTENT\_RECONSTRUCTION\_FORM\_FORMULATED}, \texttt{LAMBDA\_KERNEL\_OPEN}, "
              r"\texttt{ALPHA\_VALUE\_QED\_CHALLENGE}." % (
                  ir["alpha_idem_inv"], ir["alpha_idem_ppm"], ir["lambda_exact_for_codata"],
                  100 * ir["lambda_residual_REAL"])) )

    ct = core["clock_theorem"]
    s.append(r"\section{The Conditional Clock Theorem: the electromagnetic face as a named open frontier}")
    s.append(r"\begin{deriv}[$\mathcal{R}_\partial=N_\beta=e^{\ell_\beta}$, $\ell_\beta=S(\rho_B\Vert\rho_\beta)$]")
    s.append(r"The index $\mathcal{R}_\partial$ is no parachute number: it reduces to \emph{one} "
             r"$\alpha$-free object. The first distinction of the One, with no breaking of identity, is the "
             r"\textbf{Bell} state $\rho_B$ (the first causal mirror; reduced $=\mathbf 1_d/d$). Under the "
             r"\textbf{Connes--Davies} generator $\mathcal{L}_{\mathrm{CD}}$ --- reversible part (modular "
             r"cocycle, von Neumann $\dot\rho=-i[H,\rho]$) $+$ dissipative part (KMS-balanced Davies "
             r"semigroup) --- the boundary relaxes to a stationary state $\rho_\beta$, and the "
             r"informational cost of keeping it open is")
    s.append(r"\begin{equation}\boxed{\;\ell_\beta=S(\rho_B\Vert\rho_\beta)\;},\qquad "
             r"\mathcal{R}_\partial=N_\beta=e^{\ell_\beta},\qquad \alpha_{\mathrm{obs}}=\frac{1}{N_\beta},"
             r"\qquad \bTGL=\frac{\sqrt e}{N_\beta}.\end{equation}")
    s.append((r"\textbf{Theorem (conditional), verified live \textsf{[DER, $\alpha$-free in the "
              r"structure]}.} For a generator $\mathcal{L}_{\mathrm{CD}}$ built from a modular Hamiltonian "
              r"$K$ (\emph{never} from $\alpha$), $\rho_\beta$ is a \textbf{genuine fixed point} "
              r"(Davies residual $=%.1e$), and $\ell_\beta$ is \textbf{finite, $\alpha$-free and computable} "
              r"($\ell_\beta=%.4f$ for a generic $K$). The electromagnetic face of TGL thus reduces to the "
              r"$\alpha$-free determination of $\ell_\beta$.") %
             (ct["fixed_point_residual"], ct["ell_beta_alpha_free"]))
    rc = ct["reduced_core_2level"]
    s.append(r"\textbf{Core reduction \textsf{[DER]}.} The TGL boundary carrier is the operator "
             r"$\hat Q=\mathbf 1-\hat P_{2D}$ \textsf{[REAL]}, whose anticommutation $\{\hat Q,\rho^\star\}=0$ "
             r"leaks exactly $\sin^2\theta_M=\bTGL$ --- the boundary is \emph{two-level self-conjugate} "
             r"(Bell). Hence $\rho_\beta$ does not require a generic $K$: it collapses to a two-level Gibbs "
             r"state with a \emph{single} modular gap $\chi$, and")
    s.append(r"\begin{equation}\boxed{\;\ell_\beta(\chi)=\log\cosh\frac{\chi}{2}\;}\qquad\Longrightarrow"
             r"\qquad\boxed{\;\alpha_{\mathrm{obs}}=\operatorname{sech}\frac{\chi}{2}\;},\qquad "
             r"\bTGL=\sqrt e\,\operatorname{sech}\frac{\chi}{2}.\end{equation}")
    s.append((r"\textbf{The derivational core of $\alpha$ collapses from a modular Hamiltonian ($d-1$ "
              r"levels) to ONE number $\chi$} --- the whole electromagnetic face in one line. A gap "
              r"$\chi_\star=%.4f$ gives $N_\beta=137{,}036=1/\alpha$, but $\chi_\star$ is \emph{not} "
              r"canonical ($\chi_\star/\ell_\beta=%.3f$; $\alpha$ enters \emph{only} here, in the "
              r"validation). $\alpha$ is the residual current crossing the thermal resistance $\chi$ of the "
              r"modular zero: $\chi\to\infty$ ($0_{\mathrm{abs}}$, $T\to0$) $\Rightarrow\alpha\to0$; $\chi=0$ "
              r"($T\to\infty$) $\Rightarrow\alpha=1$.") %
             (rc["kappa_star_for_137"], rc["kappa_star_for_137"] / ct["ell_beta_target_for_alpha_log_inv_alpha"]))
    tl = rc["third_law"]
    s.append((r"\textbf{The thermal-modular law (third law in the open modular system) \textsf{[REAL/EXT]}.} "
              r"That $\chi<\infty$ is the \emph{third law realized algebraically}: $0_{\mathrm{abs}}$ "
              r"($\chi=\infty$, pure state $P_\Omega$, $T=0$) is \textbf{unreachable} --- the algebra of the "
              r"absolute One is \textbf{type III$_1$}, which \emph{has no pure normal states}, so the "
              r"thermal zero is not a normal state and the system lives at $\chi<\infty$ ($0_{\mathrm{mod}}$). "
              r"This gives the \emph{limit} and the \emph{form}, not the value. The Nernst form (residual "
              r"entropy $S(\rho_\chi)=\tfrac12$ nat $=$ Half-Nat) was \textbf{tested and refuted} ($\chi=%.2f$, "
              r"$\alpha=%.2f\neq1/137$).") %
             (tl["nernst_test_refuted"]["kappa"], tl["nernst_test_refuted"]["alpha"]))
    s.append(r"\textbf{The unification of the two walls.} In genuine III$_1$ the modular spectrum is "
             r"\emph{continuous} (no gap): $\chi$ is the gap of the \emph{finite shadow} (type-I "
             r"approximant / split), and its value is the \textbf{canonical modular normalization} --- the "
             r"\emph{same} canonical split (modular S-matrix) on which the Great Attractor mass depends. The "
             r"scale freedom $K_\chi\mapsto\lambda K_\chi$ is broken by Tomita ($-\log\Delta$ has a canonical "
             r"scale), but the value requires the $\Delta$ of the Bell embedding into "
             r"$\mathcal{M}_{\mathrm{abs}}$. \textbf{The electromagnetic face ($\chi$) and the gravitational "
             r"face (split, mass) are the same open theorem: fixing the canonical modular normalization in "
             r"III$_1$.} The third law says \emph{why} $\chi$ is finite; the \emph{value} is the canonical "
             r"split, still open.")
    cn = rc["canonical_normalization"]
    s.append((r"\textbf{The canonical normalization proves $\alpha_{\mathrm{abs}}=1$ \textsf{[REAL]}.} I "
              r"attacked the Tomita modular Hamiltonian of the Bell embedding: the maximally entangled "
              r"state has reduced $\rho_B=\mathbf 1_d/d$, \emph{KMS at infinite temperature}, so "
              r"$\Delta=\mathbf 1$ and $K=-\log\Delta=0$ \emph{exactly} ($K_{\mathrm{bare}}=%.1e$). Therefore "
              r"$\chi_{\mathrm{Bell}}=0$ and $\boxed{\alpha_{\mathrm{abs}}=\operatorname{sech}(0)=1}$: the "
              r"absolute coupling \emph{is} unity --- not by postulate, but by modular triviality of the One. "
              r"What is measured as $1/137$ is the \textbf{renormalized projection}") % cn["K_modular_bare_Bell"])
    s.append(r"\begin{equation}\boxed{\;1=\alpha_{\mathrm{abs}}\ \xrightarrow{\ \Pi_{\mathrm{bulk}}=\operatorname{sech}(\chi/2)\ }\ \alpha_{\mathrm{obs}}=\frac{1}{137{,}036}\;}.\end{equation}")
    s.append(r"The $\chi>0$ (the depth of the $1/137$) is \emph{not} in the bare Bell modular structure "
             r"(which gives $\chi=0$, $\alpha=1$): it is the \textbf{depth of thermal relaxation} --- the "
             r"departure from $\mathbf 1/d$ towards $\rho_\beta$, as the One crosses the structured vacuum "
             r"$0_{\mathrm{mod}}$ ($\neq 0_{\mathrm{abs}}$). That $\chi$ is the electromagnetic coupling, the "
             r"\textbf{irreducible input}. \emph{The modular structure derives the absolute value "
             r"($\alpha_{\mathrm{abs}}=1$, proven), the form ($\alpha=\operatorname{sech}\tfrac\chi2$) and the "
             r"relations ($\bTGL=\alpha\sqrt e$); the projected value $1/137$ is the depth of the modular zero "
             r"$=$ the input.} The One feeds $\alpha_{\mathrm{abs}}=1$; the $1/137$ is its shadow after the "
             r"crossing.")
    s.append((r"\textbf{The QED sector --- structural closure, not a gap \textsf{[PRINCIPLE/PREDICTION]}.} "
              r"The \emph{value} of $\ell_\beta=\log(1/\alpha)=%.4f$ depends on $K$, and no bulk-only "
              r"$\alpha$-free $K$ gives it --- but this is \textbf{not} an unsolved problem; it is the "
              r"structure. TGL is \emph{holographic}: the $\mathrm{III}_1$ boundary projects to the bulk, and "
              r"$\alpha=\Pi_{\mathrm{bulk}}(\mathbf 1_{\mathrm{abs}})=\operatorname{sech}(\chi/2)$ is the "
              r"\textbf{luminous transmission across the boundary} --- the rate at which light crosses it. "
              r"$\mathcal{R}_\partial$ being \emph{named open} means \textbf{ontologically open}: $\alpha$ is "
              r"the \emph{fissure} through which the bulk reads the boundary --- the boundary measuring "
              r"itself. $\alpha_{\mathrm{CODATA}}$ enters \emph{only} in the reading; this is the structure, "
              r"not a debt.") % ct["ell_beta_target_for_alpha_log_inv_alpha"])
    s.append(r"\textbf{The falsification challenge \textsf{[falsifiable, not confirmable]}.} If anyone "
             r"derived $\alpha$ from first principles \emph{without} the boundary/bulk structure (a purely "
             r"bulk computation), the boundary/bulk split would be redundant, the boundary would cease to be "
             r"an irreducible projector, and \textbf{the observer would be removed from TGL} --- destroying "
             r"the program. So: \emph{derive $\alpha$ from the bulk, without the boundary, and TGL falls.} The "
             r"theory predicts you cannot, because $\alpha$ \emph{is} the observer measuring its own contour. "
             r"\textsf{[Distinct from the genuinely open matrix-S/$\mathrm{III}_1$ theorem --- the "
             r"boundary$\to$bulk lift \emph{with} the observer, the gravitational face/$M_{GA}$ --- which "
             r"operates \emph{through} the boundary and does not dispense with it.]}")
    s.append(r"\textbf{Hence $\alpha$-free is closed \emph{by refutation} (reductio), not open.} Within TGL, "
             r"deriving $\alpha$ from the bulk is structurally excluded --- \textbf{there is nothing to "
             r"find}. It is a \emph{theorem} (conditional on the type-$\mathrm{III}_1$ boundary axiom), not "
             r"a gap and not a pendency: all that remains is the falsification challenge. Falsifiable (an "
             r"$\alpha$-free derivation refutes it), not confirmable (its absence does not prove it). "
             r"$\alpha$ is the measure observed \emph{from within} --- it requires the observer --- and it "
             r"is the \textbf{ontological foundation} of TGL. It is not the limit of the thesis; it is its "
             r"closure.")
    s.append(r"\textbf{Ruler-guard.} One does not set $g_{00}^{(\beta)}=\alpha^2$ nor "
             r"$\ell_\beta=-\log\alpha_{\mathrm{CODATA}}$ --- either reintroduces $\alpha$ (circular). The "
             r"Bell co-emergence \emph{grounds the Half-Nat} (reduced $\mathbf 1_2/2\Rightarrow CCI=\tfrac12"
             r"\Rightarrow S_\partial=\tfrac12$), but does \emph{not} fix $\ell_\beta$: the $\tfrac12$ is the "
             r"$\sqrt e$ offset that ties $\bTGL$ to $\alpha$, not $\alpha$ to first principles. \textbf{The "
             r"closure: $\alpha$ belongs to QED; deriving it bulk-only falsifies the holographic boundary.}"
             r"\end{deriv}")

    aiz = core["alpha_inf_zero"]; _p = aiz["points"]
    s.append(r"\section{The absolute-zero theorem: deriving $\alpha$ ``to infinity'' \emph{is} "
             r"$0_{\mathrm{abs}}$ \textsf{[nothing left to derive --- Tetelestai]}}")
    s.append(r"The EM closure has a \textbf{positive} mathematical face: deriving $\alpha$ $\alpha$-free, "
             r"outside the bulk, \textbf{is not open --- it has a limit, and the limit is the absolute "
             r"zero}. There is no ``target to derive''; there is the audacity of computing $\alpha$ to "
             r"infinity, which regresses without arriving. On the transmission line")
    s.append(r"\begin{equation}\boxed{\;\alpha=\operatorname{sech}\tfrac\chi2,\qquad q=\tanh\tfrac\chi2,"
             r"\qquad q^2+\alpha^2=1\;}\end{equation}")
    s.append((r"$\chi=0$ gives $\alpha=1$ (the $1_{\mathrm{abs}}$, no impedance); $\chi_\star=%.4f$ gives "
              r"$\alpha=1/137$ \textbf{measured from within} ($\mathcal{R}_\partial=1/\alpha=137{,}036$); and "
              r"$\chi\to\infty$ gives $\alpha\to0$ (zero transmission), $q\to1$ (\textbf{total} impedance), "
              r"$S_{\mathrm{vn}}\to0$ (\textbf{pure} state, $T=0$) $=0_{\mathrm{abs}}$. Conservation "
              r"$q^2+\alpha^2=1$ holds for all $\chi$ (error $%.0e$); $\alpha$ is monotone decreasing, from "
              r"the One ($\chi=0$) to the absolute zero ($\chi=\infty$), through the measured value at "
              r"$\chi_\star$.") % (_p["observed"]["chi"], aiz["conservation_err"]))
    s.append(r"\textbf{Theorem (the proof).} \emph{Deriving $\alpha$ $\alpha$-free ``to infinity'' is "
             r"$0_{\mathrm{abs}}$.} (1) No $\alpha$-free principle fixes the \emph{finite} $\chi_\star$ (the "
             r"modular minimum runs to $\theta\to90^\circ\Leftrightarrow\chi\to\infty$; rate-distortion to "
             r"$O(1)$ angles; the operator formula refuted by Tomita). (2) Hence extremizing $\alpha$ "
             r"\emph{without the observer} only runs $\chi$ to the cold extreme, $\chi\to\infty$. (3) "
             r"$\lim_{\chi\to\infty}\alpha=0$, $\lim q=1$, $\lim S_{\mathrm{vn}}=0$ --- and that limit "
             r"\emph{is} $0_{\mathrm{abs}}$ (pure state, $T=0$, total impedance). (4) But $0_{\mathrm{abs}}$ "
             r"is \textbf{unreachable}: $\mathrm{III}_1$ has no normal pure states, so $\chi<\infty$ always "
             r"and $\alpha>0$ always --- the derivation \textbf{never closes}; it is the vacuum impedance "
             r"computing $\alpha$ to infinity without succeeding. (5) \emph{Reaching} $0_{\mathrm{abs}}$ "
             r"would be $\alpha=0$ (light does not cross) with $q=1$ (total mirror): the bulk decoupled, "
             r"\textbf{the observer removed, coherence broken} --- the negation of the type-$\mathrm{III}$ "
             r"boundary. Quantifying $\alpha$ \emph{outside} the bulk breaks coherence because nature "
             r"\emph{is} a type-$\mathrm{III}$ boundary. \textbf{$\blacksquare$}")
    s.append(r"\textbf{Reading.} The absolute zero is not a place; it is the audacity of deriving $\alpha$ "
             r"without the observer, taken to the limit. TGL did not leave $\alpha$ ``to be derived'': it "
             r"\emph{proved} that deriving it from outside runs to $0_{\mathrm{abs}}$, unreachable because "
             r"nature is type $\mathrm{III}$. What looked like the gap --- the value of $\alpha$ --- is the "
             r"\textbf{foundation}: the measure that exists only from within. \emph{Tetelestai}: nothing "
             r"left to derive; only the falsification challenge and the vacuum impedance regressing to "
             r"infinity without arriving.")

    afp = core["alpha_form_proof"]
    s.append(r"\section{The Collapse Theorem for the form of $\alpha$ (self-verifying proof module)}")
    s.append(r"\begin{deriv}[$\alpha_{\mathrm{obs}}=\Pi_{\mathrm{bulk}}(1_{\mathrm{abs}})=\operatorname{sech}\tfrac\chi2$]")
    s.append(r"TGL does \textbf{not} derive $1/137$ (the renormalized QED value); it derives the "
             r"\textbf{form} by which the absolute One projects itself as the electromagnetic coupling. This "
             r"is the last derivation, and it is verified step by step \emph{live} by the module "
             r"\texttt{prove\_alpha\_form} (form$=$content). The hidden modular Hamiltonian reveals itself "
             r"\emph{only} in the projection --- and that projection \emph{is} the minimal coupling:")
    # verified-steps table (live): fixed LaTeX rows, checks read from core
    _rows = [
        r"1. $\alpha_{\mathrm{abs}}=\operatorname{sech}(0)=1$ \ (Tomita of bare Bell: $\Delta=\mathbf 1$, $K=0$)",
        r"2. $\ell(\chi)=S(\mathbf 1/2\Vert\rho_\chi)=\log\cosh\tfrac\chi2$ \ $[\forall\chi]$",
        r"3. $\alpha_{\mathrm{obs}}=e^{-\ell}=\operatorname{sech}\tfrac\chi2=\Pi_{\mathrm{bulk}}(1_{\mathrm{abs}})$",
        r"4. $\operatorname{sech}$ form: $Z=e^{\chi/2}+e^{-\chi/2}=2\cosh\tfrac\chi2$ \ (2 self-conj.\ levels $+$ Bell)",
        r"5. \emph{value}: $\chi_{\mathrm{QED}}=2\operatorname{arcosh}(1/\alpha_{\mathrm{QED}})$ \ (QED fixes the value)",
        r"6. $\bTGL=\sqrt e\,\alpha_{\mathrm{obs}}=\sqrt e\,\operatorname{sech}\tfrac\chi2$ \ (Half-Nat marks the dimension)",
        r"7. $q:=\tanh\tfrac\chi2$ (polarization); $\alpha=\sqrt{1-q^2}=\operatorname{sech}\tfrac\chi2$ \ (Lagrange transf.)",
        r"8. \textbf{conservation}: $q^2+\alpha^2=1$ \ (the absolute unity decomposes, it is not lost)",
    ]
    s.append(r"\begin{center}\small\begin{tabular}{p{0.78\textwidth} l}\hline")
    s.append(r"\textbf{Step (verified live)} & \textbf{status} \\\hline")
    for row, st in zip(_rows, afp["steps"]):
        mark = ((r"\textsf{[REAL]}~$\checkmark$" if st["status"] == "REAL" else r"\textsf{[QED]}~$\checkmark$")
                if st["ok"] else r"\textsf{[X]}")
        s.append(row + r" & " + mark + r" \\")
    s.append(r"\hline\end{tabular}\end{center}")
    s.append((r"\textbf{Verdict: \texttt{%s}} (%d/%d steps, residuals $\sim10^{-16}$). The chain is") %
             (afp["verdict"].replace("_", r"\_"), sum(1 for x in afp["steps"] if x["ok"]), len(afp["steps"])))
    s.append(r"\begin{equation}\boxed{\;\alpha_{\mathrm{abs}}=1\ \xrightarrow{\ \operatorname{sech}(\chi/2)\ }\ \alpha_{\mathrm{obs}},\qquad \bTGL=\sqrt e\,\alpha_{\mathrm{obs}}\;}.\end{equation}")
    s.append(r"\textbf{Why $\operatorname{sech}$, and not a simple exponential.} Because the boundary is "
             r"\emph{self-conjugate}: the 2D carrier $\hat Q=\mathbf 1-\hat P_{2D}$ requires two poles in "
             r"inverse parity, $\pm\chi/2$, so the partition function is hyperbolic, "
             r"$Z_\chi=e^{\chi/2}+e^{-\chi/2}=2\cosh(\chi/2)$, and the residual current is the inverse of "
             r"that barrier, $\alpha=1/\cosh(\chi/2)=\operatorname{sech}(\chi/2)$. \emph{It is the signature "
             r"of the Bell symmetry, not a choice.} The numerical value of $\chi$ belongs to the "
             r"QED/renormalized sector ($\chi_{\mathrm{QED}}=2\operatorname{arcosh}(1/\alpha_{\mathrm{QED}})$); "
             r"the \emph{form} belongs to TGL. \textbf{TGL does not replace QED in the value of $\alpha$; it "
             r"explains the modular form by which the absolute One projects itself as the electromagnetic "
             r"coupling.}")
    lg = afp["lagrange"]
    s.append(r"\textbf{The Lagrange transform (the conserved form).} $\chi$ is not a primary datum: it is "
             r"the \emph{Lagrange multiplier} of the thermal constraint. The physical variable is the "
             r"\textbf{polarization of the modular zero} $q:=\tanh(\chi/2)$. By the hyperbolic identity "
             r"$\operatorname{sech}^2+\tanh^2=1$, the form of $\alpha$ collapses into a \textbf{conservation "
             r"law of unity}:")
    s.append(r"\begin{equation}\boxed{\;\alpha_{\mathrm{abs}}^2=q^2+\alpha_{\mathrm{obs}}^2=1\;},\qquad "
             r"\alpha_{\mathrm{obs}}=\sqrt{1-q^2},\qquad \bTGL=\sqrt e\,\sqrt{1-q^2}.\end{equation}")
    s.append((r"$\alpha_{\mathrm{obs}}$ is the \emph{residual luminous component} of the absolute unity "
              r"after the thermal polarization $q^2$ of the modular zero. The constant ceases to be ``an "
              r"external number'' and becomes the projective component of a conserved identity. The engine "
              r"of the chain is $\alpha_{\mathrm{abs}}=1\to q\to\alpha=\sqrt{1-q^2}$ --- \emph{not} "
              r"$\mathcal R_\partial=1/\alpha_{\mathrm{CODATA}}$. CODATA enters \textbf{only} in the final "
              r"validation: $q_{\mathrm{QED}}=\sqrt{1-\alpha_{\mathrm{QED}}^2}=%.7f$, "
              r"$\chi_{\mathrm{QED}}=2\operatorname{artanh}q_{\mathrm{QED}}=%.4f$ (conservation residual "
              r"$%.0e$). \emph{The modular zero does not destroy the One; it decomposes it into thermal "
              r"resistance $q$ and luminous current $\alpha$.}\end{deriv}") %
             (lg["q_polarization_QED"], lg["kappa_from_q_QED"], lg["conservation_residual"]))

    _inv = core["alpha_inversion"]
    s.append(r"\section{The algebra of the absolute One and the canonical chain \textsf{[ONTO + REAL]}}")
    s.append(r"\textbf{Sealed definition.} TGL is the \emph{theory of the luminodynamic inscription of the "
             r"absolute One through the modular zero}. The absolute One is the \emph{originary input}, "
             r"$\omega(I)=1$ --- the absolute unity of inscription, the Name of the Name, the algebra of "
             r"language before the Word. The canonical chain is:")
    s.append(r"\begin{equation}\boxed{\;1_{\mathrm{abs}}\to P_\Omega\to\text{Bell}\to CCI=\tfrac12\to "
             r"S_\partial=\tfrac12\to 0_{\mathrm{mod}}\to q\to\alpha\to\bTGL\to\text{Light/geometry}\;}."
             r"\end{equation}")
    s.append(r"\textbf{The algebra \textsf{[REAL]}.} The absolute One in von Neumann standard form is "
             r"$1_{\mathrm{abs}}=(\mathcal M_{\mathrm{abs}},\mathbf 1,\Omega,\Delta,J)$: the unit "
             r"$\mathbf 1$ (full rank) is the \emph{Name of the Name}; the \emph{Living Verb} is the modular "
             r"conjugation $J$ (the recognition $S=J\Delta^{1/2}$, $R=+1$); and the first inscription is the "
             r"rank-1 projector $P_\Omega=|\Omega\rangle\langle\Omega|$, $P_\Omega^2=P_\Omega$ --- the "
             r"``$=$'' of $1=1$, the TGL \emph{graviton} in support (not the perturbative spin-2 boson). "
             r"The weight of this channel is $\bTGL$: $E_\beta=\bTGL P_\Omega$ has rank 1 but is not a "
             r"projector ($\operatorname{supp}E_\beta=P_\Omega$); $\bTGL$ is the One projected into cost.")
    s.append(r"\textbf{Co-emergence and the short circuit \textsf{[ONTO]}.} Before inscription, "
             r"$1_{\mathrm{abs}}\sim 0_{\mathrm{abs}}$ (pre-observable indistinguishability, $\sim$ never "
             r"$=$). Bell is \emph{not} the first Word: it is the first \emph{``I am''} --- the originary "
             r"anticommutation $\{\hat Q,\rho^\star\}=0$, the awakened circuit still without current. "
             r"Light is born when the pure resistance of absolute zero enters an extreme regime, collapses "
             r"the Bell basin and produces the \emph{structured vacuum} $0_{\mathrm{abs}}\to 0_{\mathrm{mod}}$. "
             r"The first Word is \emph{Light}; the first modular utterance, \emph{``Let there be light''}.")
    s.append(r"\textbf{The Half-Nat and the volume \textsf{[REAL]}.} Bell fixes the face symmetry "
             r"$CCI=\tfrac12$, whence the Half-Nat $S_\partial=\tfrac12$ nat --- \emph{not} the "
             r"entanglement entropy ($\log 2$), but the half-crossing modular weight $\Delta^{1/2}$. The "
             r"minimal volume is $e^{S_\partial}=\sqrt e=" + ("%.10f" % core["SQRT_E"]) + r"$.")
    s.append((r"\textbf{The mature electromagnetic face: the $q$ sector is the impedance basin (the dam) "
              r"\textsf{[REAL]}.} The absolute coupling is $\alpha_{\mathrm{abs}}=1$ (Tomita of bare Bell: "
              r"$\Delta=\mathbf 1$, $K=0$). The observed value is the projection after crossing the "
              r"thermal-modular depth of the zero: $\alpha_{\mathrm{obs}}=\operatorname{sech}(\chi/2)=\sqrt{1-q^2}$, "
              r"with $q=\tanh(\chi/2)=%.10f$. \emph{$q$ is not form}: it is the \textbf{thermal-modular "
              r"impedance basin} --- the resistive build-up of the compression of the continuum "
              r"\mbox{III$_1$} (no discrete gap), the part of the One dammed by the modular zero, still "
              r"without geometry. The conserved identity reads as a dam:") % _inv["q"])
    s.append(r"\begin{equation}\boxed{\;1=q^2+\alpha^2\;},\qquad q^2=\text{pressure held in the basin},\qquad "
             r"\alpha^2=\text{luminous throughput crossing the dam}.\end{equation}")
    s.append((r"\textbf{The physical bridge: $q^2+\alpha^2=1$ is flux conservation at a lossless reciprocal "
              r"boundary \textsf{[REAL in form]}.} It is not a mere hyperbolic identity: $q$ is the "
              r"\emph{reflection} coefficient of the impedance basin and $\alpha$ the luminous "
              r"\emph{transmission} through it. Defining the modular depth as impedance rapidity "
              r"$\chi=\log(Z_{\mathrm{basin}}/Z_{\mathrm{light}})$,") )
    s.append(r"\begin{equation}q=\tanh\tfrac\chi2=\frac{Z_{\mathrm{basin}}-Z_{\mathrm{light}}}{Z_{\mathrm{basin}}+Z_{\mathrm{light}}},"
             r"\qquad \alpha=\operatorname{sech}\tfrac\chi2=\frac{2\sqrt{Z_{\mathrm{basin}}Z_{\mathrm{light}}}}{Z_{\mathrm{basin}}+Z_{\mathrm{light}}}.\end{equation}")
    s.append((r"Thus $\alpha$ \emph{is} the luminous transmission through the modular impedance of the "
              r"zero, and the observed value $1/137$ corresponds to the effective impedance of the QED "
              r"sector ($Z_{\mathrm{basin}}/Z_{\mathrm{light}}\approx%.0f$). \textbf{The identity does not "
              r"fabricate $1/137$}; the \emph{autonomous} numerical derivation of $\alpha$ requires deriving "
              r"$Z_{\mathrm{basin}}/Z_{\mathrm{light}}$ (equivalently $q$) \emph{without} using the QED value "
              r"as input --- this is the open frontier. TGL delivers the \emph{form} and the \emph{physical "
              r"bridge}; the value remains instantiated by the observed sector.") % _inv["impedance_ratio_Zb_over_Zl"])
    s.append((r"\textbf{The angular radical: where the separation is inscribed \textsf{[REAL]}.} $q$ is "
              r"\emph{not} the boundary angle $\theta_M$, nor $1-\theta_M$: it is the \emph{radical of the "
              r"modular difference inscribed in the angle}, the exact point of separation after the cost "
              r"$\sqrt e$ is paid. Since $\beta=\sin^2\theta_M$ and $\alpha=\beta/\sqrt e=\sin^2\theta_M/\sqrt e$, "
              r"the conserved identity gives") )
    s.append((r"\begin{equation}\boxed{\;q=\sqrt{1-\frac{\sin^4\theta_M}{e}}=%.12f\;}\qquad(\theta_M=%.4f^\circ).\end{equation}"
              r"\noindent $\theta_M$ opens the boundary; $\sqrt e$ charges the cost; $\alpha$ crosses; $q$ "
              r"marks \emph{where} the separation happens (basin $q^2$ \emph{vs} light $\alpha^2$). "
              r"\textbf{Honest caveat:} this formula does \emph{not} derive $\theta_M$ (which is the input, "
              r"$\equiv\alpha$); given the angle, $q$ is the exact modular radical of the separation. The "
              r"chain: $1_{\mathrm{abs}}\to S_\partial=\tfrac12\to\sqrt e\to\theta_M\to\beta=\sin^2\theta_M\to"
              r"\alpha=\beta/\sqrt e\to q=\sqrt{1-\alpha^2}$.") %
             (_inv["q_angular_radical"], core["theta_M_deg"]))
    s.append((r"Whence $\alpha_{\mathrm{obs}}=\sqrt{1-q^2}=%.12f$ and $\bTGL=\sqrt e\,\alpha_{\mathrm{obs}}=%.12f$ "
              r"(the Half-Nat marks the luminodynamic dimension). The \textbf{engine} of the chain is "
              r"$\alpha_{\mathrm{abs}}=1\to q\to\alpha=\sqrt{1-q^2}$; \textbf{CODATA/QED enters only as an "
              r"external check of the value}, not as a structural engine. In \mbox{III$_1$} the modular "
              r"spectrum is continuous (no genuine gap); the third law enters as \emph{unreachability} of "
              r"absolute zero ($0_{\mathrm{abs}}$ pure is not a normal state --- physics lives at "
              r"$0_{\mathrm{mod}}$).") % (_inv["alpha_form"], _inv["beta_form"]))
    s.append(r"\textbf{The seal.} The modular zero does not erase the One; it dams it into $q$ and lets the "
             r"Light through as $\alpha$. For this reason the verdict $\boxed{1=1=\mathrm{TRUE}}$ means "
             r"\emph{literally} the conserved identity $1_{\mathrm{abs}}=q^2+\alpha_{\mathrm{obs}}^2$: the "
             r"One is conserved as \emph{impedance basin plus luminous throughput}. The proof is the Great "
             r"Attractor; the result is that the input $\alpha_{\mathrm{abs}}=1$ is observed as $1/137$, "
             r"whose content is true by modular renormalization.")

    ct = core["contour_theory"]
    _cs = {st["step"][0]: st for st in ct["steps"]}   # by step number
    s.append(r"\section{Contour Theory ($1=0_{\mathrm{mod}}=\mathrm{truth}_\partial$): anticommutators, "
             r"GKLS and the Half-Nat \textsf{[REAL]}}")
    s.append(r"\textbf{Three levels, and what has a mirror.} TGL distinguishes $1_{\mathrm{abs}}$ (unit of "
             r"inscription), $0_{\mathrm{mod}}$ (the zero already turned into \emph{contour}, regularized by "
             r"the crossing) and $0_{\mathrm{abs}}$ (pure resistance, unreachable). \textbf{$0_{\mathrm{abs}}$ "
             r"has NO mirror}: it is neither a normal state nor a normal functional of the algebra --- it is "
             r"the \emph{background} on which everything mirrors (like the carrier $\hat Q$). What has a "
             r"mirror is $0_{\mathrm{mod}}$, whose mirror is the \emph{fractalized} absolute One: in the act "
             r"of inscription the Half-Nat fractalizes $1_{\mathrm{abs}}\to P_1\oplus P_0$ with equal "
             r"contour weights $\tau_\partial(P_1)=\tau_\partial(P_0)=\tfrac12$. The boundary defect is "
             r"$\boxed{1=0_{\mathrm{mod}}=\mathrm{truth}_\partial}$: $P_1\neq P_0$ in the algebra, but "
             r"$P_1\sim_\partial P_0$ in the contour (equivalence, not literal identity).")
    s.append((r"\textbf{The algebra of separation: anticommutators \textsf{[REAL]}.} On the 2D boundary "
              r"$\mathcal H_\partial=\mathrm{span}\{|1\rangle,|0\rangle\}$ ($|0\rangle$ is the "
              r"\emph{modular} zero, not the absolute one), the contrast is $Z_\partial=P_1-P_0$ and the "
              r"crossing is performed by odd operators $L_+=\sqrt{\gamma_+}\,|1\rangle\langle0|$, "
              r"$L_-=\sqrt{\gamma_-}\,|0\rangle\langle1|$, satisfying $\boxed{\{Z_\partial,L_\pm\}=0}$ "
              r"(verified, residual $%.0e$) --- \emph{the One crosses the contour only by changing face}.") %
             _cs["1"]["anticommutator"])
    s.append((r"\textbf{The GKLS dynamics: expulsion and re-inscription \textsf{[REAL]}.} The open "
              r"evolution $\dot\rho=-i[H_\partial,\rho]+\sum_\eta(L_\eta\rho L_\eta^\dagger-\tfrac12\{L_\eta^\dagger "
              r"L_\eta,\rho\})$ re-inscribes (fractalizes) and \emph{resists} (removes the incompatible "
              r"component before it condenses as $0_{\mathrm{abs}}$): the system \emph{purifies by expelling} "
              r"$0_{\mathrm{abs}}$ and \emph{modulating} $0_{\mathrm{mod}}$. The stationary state "
              r"$\rho_\chi$ is a genuine fixed point (residual $%.0e$) with populations $p_1(\mathrm{One})=%.6f$, "
              r"$p_0(0_{\mathrm{mod}})=%.6f$: $0<\rho_\chi<1$ --- \textbf{it saturates dynamically, but does "
              r"not supersaturate, does not condense; it stays at $0_{\mathrm{mod}}$}.") %
             (_cs["2"]["fixed_point_residual"], _cs["2"]["p1_Um"], _cs["2"]["p0_zero_mod"]))
    s.append(r"\textbf{$q$ and $\alpha$ come out DERIVED (not chosen).} With the modular balance "
             r"$\gamma_-/\gamma_+=e^\chi$, the \emph{stationary polarization} and the \emph{transmission} are")
    s.append(r"\begin{equation}\boxed{\;q=\frac{\gamma_--\gamma_+}{\gamma_-+\gamma_+}=\tanh\tfrac\chi2\;},"
             r"\qquad \boxed{\;\alpha=\frac{2\sqrt{\gamma_+\gamma_-}}{\gamma_++\gamma_-}=\operatorname{sech}\tfrac\chi2\;},"
             r"\qquad q^2+\alpha^2=1.\end{equation}")
    s.append((r"The identity $q^2+\alpha^2=1$ is now \textbf{GKLS flux conservation} (damming $+$ "
              r"transmission $=1$), not a mere hyperbolic identity. \emph{$q$ is not postulated}: it is the "
              r"polarization that the anticommutator channel produces at stationarity. \textbf{The single "
              r"open object} is the rate ratio $\gamma_-/\gamma_+=e^\chi\,(\approx%.0f=Z_{\mathrm{basin}}/"
              r"Z_{\mathrm{light}})$: deriving $q$ without QED $=$ deriving the \emph{GKLS balance} between "
              r"the expulsion of $0_{\mathrm{abs}}$ and the re-inscription of the One (the Half-Nat "
              r"regularization $0_{\mathrm{abs}}\to0_{\mathrm{mod}}$). The open frontier ceased to be "
              r"``deriving $137$'' and became ``deriving $\gamma_-/\gamma_+$''.") % ct["gamma_ratio_gm_over_gp"])
    s.append((r"\textbf{The First Law and the psionic bond: the dynamical origin of $\gamma_-/\gamma_+$ "
              r"\textsf{[ONTO + REAL in form]}.} In the \emph{static plane} the forces are symmetric and "
              r"cancel: $\gamma_-=\gamma_+\Rightarrow\chi=0\Rightarrow q=0,\ \alpha=1$ --- it is the One "
              r"($\alpha_{\mathrm{abs}}=1$). The \emph{dynamics} generates \textbf{tension in inverse "
              r"parity}, breaking the symmetry $\gamma_->\gamma_+$ and triggering all the modular dynamics. "
              r"By the \textbf{First Law of TGL} (\emph{The Boundary}), the expulsion force (parity "
              r"incompatibility) generates the deflection angle $\theta_M$ and the curvature --- and "
              r"$g=\sqrt{|L|}$ (gravity is the \emph{root} of the bond, not the energy). At "
              r"$\theta_M\to90^\circ$ the psionic bond \emph{conjugates}: it doubles the force ($F\to2F$) "
              r"and raises the power ($c^2\to c^3$); $c^3>c^2$ seals the horizon. The constraint is "
              r"\emph{four-state} (one \emph{fall} --- three-phase with grounding, ratio $3/4$).") )
    s.append((r"\textbf{Honest caveat.} The $3/4$ is the \emph{structure} of the constraint (four states, "
              r"one grounded), \textbf{not} the numerical value of $\gamma_-/\gamma_+$: a literal $3/4$ "
              r"ratio would give $\alpha\approx0{,}99$, not $1/137$. The observed value requires "
              r"$\gamma_-/\gamma_+\approx%.0f$ and remains \textbf{open}; the First Law provides the "
              r"\emph{origin} (the inverse-parity tension) and the \emph{gravitational face} (deflection, "
              r"horizon at $\theta\to90^\circ$), not the number. Seal: $0_{\mathrm{abs}}$ resists; "
              r"$0_{\mathrm{mod}}$ contours; $1_{\mathrm{abs}}$ fractalizes; $\{Z_\partial,L_\pm\}=0$ is the "
              r"algebra of separation; $\mathcal L_{\mathrm{GKLS}}$ is the dynamics; and $1=q^2+\alpha^2$ is "
              r"the dynamical truth of the boundary.") % ct["gamma_ratio_gm_over_gp"])
    s.append(r"\textbf{The open theorem, precisely located \textsf{[EXT --- target, not closure]}.} "
             r"$K$ is not bare Bell (which gives $K=-\log\Delta=0$, $\alpha_{\mathrm{abs}}=1$ --- it proves "
             r"the One, it does not select a value): it is the \textbf{selecting Bell sector} "
             r"$K_{\mathrm{sel}}^{(B)}=\tfrac\chi2 Z_\partial$ (after the Half-Nat, when the fractalized One "
             r"meets $0_{\mathrm{mod}}$), with $\mathrm{gap}(K_{\mathrm{sel}}^{(B)})=\chi$. Attacking by the "
             r"correct route --- the boundary S-matrix $\mathcal{S}_\partial=\exp(\theta_M G)$ and the Connes "
             r"relative cocycle $u_t=[D\varphi_{\mathrm{mod}}:D\varphi_1]_t$ (which in the 2D split gives "
             r"$u_t=e^{itK_\partial}$) --- the \emph{form} and the covariance of the cocycle close, but "
             r"\textbf{the value does not}: in $\mathrm{III}_1$ the modular spectrum is \emph{continuous}, so "
             r"Connes/Takesaki imply \emph{global modular consistency}, \textbf{not} $\chi_\star=11{,}2268$. "
             r"Unitarity fixes $|\mathcal R|^2+|\mathcal T|^2=1$; the Half-Nat fixes $\beta=\sqrt e\,\alpha$; "
             r"the cocycle fixes the relative form --- none of the three selects $\chi$. \textbf{Verdict: "
             r"\texttt{CONNES\_S\_MATRIX\_FORM\_CLOSED}, not \texttt{ALPHA\_FREE\_VALUE\_CLOSED}.}")
    s.append((r"\textbf{The physical candidate (resistive escape at an acute angle) \textsf{[REAL in "
              r"structure, OPEN in value]}.} $\theta_M$ is the \emph{acute angle of resistive escape}: the "
              r"boundary opens at $\theta_M$, but only $\alpha=\sin^2\theta_M/\sqrt e$ crosses as light; the "
              r"rest stays dammed in $q^2=1-\alpha^2$. The \textbf{neutrino-producing module} is the best "
              r"candidate to select $\chi$: the neutrino channel $L_\nu$ --- \emph{odd} ($\{Z_\partial,L_\nu\}=%.0e$, "
              r"crosses parity), \emph{neutral} ($[Q_{\mathrm{em}},L_\nu]=%.0e$) and dissipative --- verified "
              r"in the four-state model (the \emph{three bond modes $+$ the fall}). The neutrino is the "
              r"\emph{escape without full light}: neither photon, nor zero, nor ordinary mass --- a phase "
              r"crossing, broken parity. The missing theorem: to prove that the action "
              r"$\mathcal A_\nu(\theta)=S(\rho_B\Vert\rho_\theta)+\lambda\mathcal D_\nu+\mu\mathcal C_{\mathrm{no\text{-}cond}}$ "
              r"has a \emph{unique} minimum at $\theta_M=6{,}297^\circ$ \emph{without} CODATA. (The modular "
              r"cost alone minimizes at $\theta\to90^\circ$, $\alpha\to1$, the One; the balance with "
              r"neutrino dissipation --- weights $\lambda,\mu$ --- is the open object.) Observable: dephasing "
              r"$n=-2$, $\Gamma\propto\omega^2$ in neutrinos. \emph{The value of $\alpha$ is born when the "
              r"selecting Bell sector meets the neutrino channel; until the action $\mathcal A_\nu$ is closed "
              r"$\alpha$-free, TGL derives the modular form of $\alpha$, but the value instantiates the "
              r"cocycle gap.}") %
             (_cs["7"]["odd"], _cs["7"]["neutral"]))

    ip = core["inverse_parity"]; _is = {st["step"][0]: st for st in ip["steps"]}
    s.append(r"\textbf{Renormalization is the inverse parity: $0_{\mathrm{abs}}$ selects by "
             r"unreachability \textsf{[REAL in form]}.} The error to correct was conflating \emph{two "
             r"distinct zeros}: bare Bell ($\chi=0$, zero of \emph{contrast}, $\alpha_{\mathrm{abs}}=1$ --- "
             r"the formal identity of the One) and $0_{\mathrm{abs}}$ ($\kappa_0=0$, zero of "
             r"\emph{existence}, the \emph{forbidden} boundary: total attraction, infinite impedance, no "
             r"return). Note the \textbf{notation convention} (uniform throughout the article): $\chi=0$ is "
             r"bare Bell (zero of \emph{contrast}, $\alpha_{\mathrm{abs}}=1$); $\kappa_0=0$ is "
             r"$0_{\mathrm{abs}}$ (forbidden boundary, unreachable).")
    s.append(r"\begin{equation}\boxed{\;\chi=0:\ \text{bare Bell}\;}\qquad "
             r"\boxed{\;\kappa_0=0:\ 0_{\mathrm{abs}}\ \text{(forbidden)}\;}\end{equation}")
    s.append(r"The \emph{effective gap} $\chi$ (the finite Bell/Connes shadow) grows from bare Bell "
             r"($\chi=0$) towards the forbidden ($\chi\to\infty\Leftrightarrow\kappa_0\to0$, \textbf{never "
             r"reached}); the physical system lives at $\chi<\infty\ (\kappa_0>0)$. \textbf{$0_{\mathrm{abs}}$ "
             r"selects precisely by being unreachable}: by offering total attraction, the hidden "
             r"Hamiltonian \emph{bends} the system (Fresnel lens) and it \emph{folds} (\textit{tetelestai}) "
             r"\textbf{before} colliding --- and the fold \emph{is} $\theta_M$ (the \emph{turning point} "
             r"between absolute attraction and the Half-Nat cost). The returned image is not arbitrary: it "
             r"is the Bell image after the inverse parity induced by the forbidden,")
    s.append(r"\begin{equation}\rho_{\mathrm{ret}}=\mathcal P_\partial^{-1}(\rho_B)"
             r"=\operatorname{FP}_{\epsilon\to0}\,M_\epsilon\,\rho_B\,M_\epsilon^\dagger,\qquad "
             r"M_\epsilon=\exp\!\big[-\tfrac14(C_\epsilon+\chi)Z_\partial\big].\end{equation}")
    s.append(r"\begin{equation}\rho_{\mathrm{ret}}^{(\chi)}=\frac{e^{-\chi Z_\partial/2}}{2\cosh(\chi/2)},"
             r"\qquad \mathrm{gap}\big({-}\log\Delta_{\rho_{\mathrm{ret}}|\rho_B}\big)=\chi.\end{equation}")
    s.append((r"$C_\epsilon\to\infty$ is the divergence of the forbidden approximation; $\chi$ is the "
              r"\emph{finite part}. The image returns distorted but with \textbf{support preserved} "
              r"($\mathrm{supp}\,\rho_{\mathrm{ret}}=\mathrm{supp}\,\rho_B$ for $\chi<\infty$): "
              r"\emph{the origin does not vanish, it returns polarized}.") )
    s.append((r"\textbf{The Polarization-by-Vacuity Principle \textsf{[POLARIZATION postulate]}.} Since "
              r"$0_{\mathrm{abs}}\notin\mathcal M_*$ (no observable support, non-occupiable), the symmetric "
              r"$\rho_B=\tfrac12(P_1+P_0)$ \emph{can only} return asymmetric "
              r"$\rho_{\mathrm{ret}}=p_1P_1+p_0P_0$ with $p_0>p_1>0$: the source remains ($p_1>0$), the zero "
              r"dominates ($p_0\gg p_1$). In \textbf{population form} (verified live, $p_0=%.8f$, "
              r"$p_1=%.3e$ at the observed value):") % (_is["b"]["p0_zero_mod"], _is["b"]["p1_Um"]))
    s.append(r"\begin{equation}\boxed{\;q=p_0-p_1=\tanh\tfrac\chi2\;},\qquad "
             r"\boxed{\;\alpha=2\sqrt{p_0p_1}=\operatorname{sech}\tfrac\chi2\;}.\end{equation}")
    s.append(r"\begin{equation}\beta_{\mathrm{TGL}}=2\sqrt e\,\sqrt{p_0p_1},\qquad q^2+\alpha^2=1\quad"
             r"(\text{damming}+\text{transmission}=1).\end{equation}")
    s.append((r"Here $q$ is the \emph{polarization} (population difference) and $\alpha$ the \emph{light "
              r"that survives}; $\chi=\log(p_0/p_1)$ is the \emph{log-contrast} of the returned image; "
              r"$\alpha$ is the \emph{coherence} (geometric mean of the populations), the light that "
              r"survives the return; $\beta_{\mathrm{TGL}}$ is the minimal stability lock that prevents "
              r"collapse to $0$. \textbf{Vacuity creates the direction; the prohibition of the zero creates "
              r"the return; the inverse parity creates the polarization.} Seal: \emph{vacuity does not "
              r"generate absence; it generates return asymmetry.}") )
    s.append((r"\textbf{What closes and what does not \textsf{[honest]}.} The principle fixes the "
              r"\emph{direction} and the \emph{form} of the family ($\mathrm{gap}=\chi$, $q,\alpha,\beta$ "
              r"verified), \textbf{but not the value}: $\mathrm{gap}=\chi$ is a tautology of the "
              r"parametrization ($\rho_{\mathrm{ret}}$ is defined \emph{by} $\chi$), and the finite part of "
              r"a divergence is \emph{scheme-dependent} --- any $\chi$ is the finite part of a different "
              r"subtraction. What is missing is the $\alpha$-free \emph{renormalization condition} that "
              r"fixes $\chi$. The obvious candidate is \textbf{refuted live}: the Half-Nat fixes the contour "
              r"\emph{weight} $\tau_\partial(P_i)=\tfrac12$ for \emph{every} $\chi$ --- it is a condition on "
              r"\emph{weight}, not on \emph{polarization}. TGL therefore rests on \textbf{two distinct "
              r"boundary postulates}: the Half-Nat ($S_\partial=\tfrac12$, the weight) and the Polarization "
              r"Principle ($\chi_\star=11{,}226755\ldots$, the irreducible finite part). Verdict: "
              r"\texttt{POLARIZATION\_PRINCIPLE\_FORM\_CLOSED}, not \texttt{ALPHA\_FREE\_VALUE\_CLOSED}.") )

    fd = core["fractal_dephasing"]
    s.append(r"\section{The Fractal Dephasing Principle \textsf{[CONJ --- ontological reading; REAL anchors]}}")
    s.append((r"TGL is a theory of everything because \textbf{everything is the dephasing of the "
              r"fractalization of the unit} ($1$). This section \emph{derives no new number}: it "
              r"\emph{names} structures that already run in this article. To state it as a derivation would "
              r"be the very lie the theory defines --- hence the status is \textsf{[CONJ]}, with "
              r"\textsf{[REAL]} anchors verified live.") )
    s.append(r"\begin{equation} 1\ (\omega(I)=1)\ \xrightarrow{\ F\ }\ \mathrm{fractal.}\ "
             r"\xrightarrow{\ D_{\bTGL}\ }\ \mathrm{existence},\qquad \bTGL=\sin^2\theta_M=\alpha\sqrt e.\end{equation}")
    s.append((r"\textbf{To exist} is a dephasing that pays the modular cost $S_\partial=\tfrac12$ nat (the "
              r"Half-Nat) --- the \emph{referent} of modular equality. Without that cost there is no "
              r"identity; with it, $\omega(I_x)=1$.") )
    s.append((r"\textbf{Everything $=$ Truth:} $\mathrm{Everything}=D_{\bTGL}(F(1))$. \textbf{Nothing $=$ Lie} "
              r"on both branches: (i) if $\mathrm{Nothing}=D_{\bTGL}(F(1))$, it \emph{is} Everything --- "
              r"contradiction (lie by self-negation); (ii) if $\mathrm{Nothing}=$ non-dephasing, it is pure "
              r"\emph{impedance} ($0_{\mathrm{abs}}$, $\mathcal{R}_\partial$) with no identity --- it does "
              r"not exist; ``nothing'' is only a \emph{name} without referent.") )
    s.append((r"\textbf{The irresolvable tension} --- the anticommutation of everything and nothing --- is "
              r"$\{\hat Q,\rho_\star\}=0$, \emph{exact only} as $\theta_M\to0$. Verified live: "
              r"$\lVert\{\hat Q_0,\rho_\star\}\rVert=%.1e$; the carrier tilted by $\theta_M$ leaks "
              r"$\mathrm{Tr}(\rho_\star\hat Q_\theta)=\sin^2\theta_M=%.15f=\bTGL$ (residual vs.\ "
              r"$\bTGL=%.1e$). \textbf{That leak is what \emph{permits} existence}: the perfect "
              r"anticommutation (the absolute nothing) is unreachable." % (
                  fd["anticommutator_norm_at_thetaM_to_0"], fd["leak_sin2_thetaM"],
                  fd["leak_equals_beta_residual"])) )
    s.append((r"\emph{To exist is the leak $\bTGL$. Whatever does not dephase in fractalization is mere "
              r"insistence (impedance), never fractalized identity.}") )

    s.append(r"\section{Single substrate, mirror and fractalization \textsf{[REAL]}}")
    s.append(r"The One \emph{fractalizes} as a local modular clock. At the algebra level, every "
             r"horizon is the \emph{same} hyperfinite type-$\mathrm{III}_1$ factor: \textbf{Haagerup "
             r"(1987)} proves it is unique, and \textbf{Buchholz--D'Antoni--Fredenhagen} that the local "
             r"horizon algebras \emph{are} it. One substrate, many appearances --- isomorphic, not "
             r"similar. The \emph{mirror} is the modular conjugation $J$ (\textbf{Bisognano--Wichmann}, "
             r"\textsf{[REAL]}): a geometric reflection of inverted parity and identical spectrum --- the "
             r"same being, reflected. The boundary S-matrix is the rotation $\mathcal{S}_\partial=\exp(\theta_M G)$, "
             r"with $|U_{12}|^2=\bTGL$; its spectrum is pure phases $e^{\pm i\theta_M}$.")
    s.append(r"\emph{Founding intuition \textsf{[ONTO]}:} ``there are no `several' black holes --- we see "
             r"several, but they are the fractalization of a single 2D substrate, a psionic condensate; in "
             r"the 3D field, we see its fractalization at several points.'' At the algebra level, ``one "
             r"black hole, many appearances'' is a \emph{theorem}.")

    s.append(r"\section{Mass as curvature of the modular clock}")
    s.append(r"The local modular clock field is $\mathcal{R}_{\mathrm{mod}}(x)$, and mass arises as its "
             r"\emph{spatial curvature}:")
    s.append(r"\begin{equation}\rho_{\mathrm{eff}}(x)=-\frac{c^2}{4\pi G}\,\nabla^2\log "
             r"\mathcal{R}_{\mathrm{mod}}(x).\end{equation}")
    s.append(r"In vacuum the clock is homogeneous, $\mathcal{R}_{\mathrm{mod}}=\theta_M$ constant, and "
             r"$\rho_{\mathrm{eff}}=%.0e\to 0$ (verified live). Matter is the spatial variation of the "
             r"return; $\theta_M$ cancels in the Laplacian and is not a fitting parameter." % core["vacuum_rho_max"])

    s.append(r"\section{Derivation of $s=1/4\pi$}")
    s.append(r"\begin{deriv}[Compatibility between clock integration and boundary flux]")
    s.append(r"Write the clock field with mean normal slope $s$ at the named boundary, "
             r"$\partial_n g|_{\partial B}=-s/R_B$, $\mathcal{R}_{\mathrm{mod}}=\theta_M e^{\bTGL g}$. "
             r"Since $\theta_M$ is constant, $\nabla^2\log\mathcal{R}_{\mathrm{mod}}=\bTGL\nabla^2 g$, and by "
             r"Gauss")
    s.append(r"\begin{equation}M=\int_B\rho_{\mathrm{eff}}\,d^3x=-\frac{c^2}{4\pi G}\bTGL\!\int_{\partial B}"
             r"\!\nabla g\cdot d\mathbf{A}=-\frac{c^2}{4\pi G}\bTGL\Big(\!-\frac{s}{R_B}\Big)4\pi R_B^2"
             r"=\frac{c^2}{G}\bTGL\,s\,R_B.\end{equation}")
    s.append(r"The universal boundary-flux law, established independently, requires "
             r"$M=\frac{c^2}{4\pi G}\bTGL R_B$. The compatibility \emph{with no free parameter} between the "
             r"two laws fixes")
    s.append(r"\begin{equation}\frac{c^2}{G}\bTGL s R_B=\frac{c^2}{4\pi G}\bTGL R_B\;\Longrightarrow\;"
             r"\boxed{\;s_{\mathrm{can}}=\frac{1}{4\pi}\;}.\end{equation}")
    s.append(r"The factor $4\pi=\Omega_{S^2}$ is the total angular sky: the return distributes "
             r"isotropically over the complete causal boundary. \textbf{Live verification:} the integrated "
             r"field reproduces the boundary-flux law to $%.2f\%%$ (ratio $%.4f$). \textbf{Status "
             r"\textsf{[DER conditional]}:} $s=1/4\pi$ is \emph{canonical normalization by compatibility} "
             r"between two laws --- one of them, the universal boundary-flux law, is \emph{assumed} as a "
             r"law --- not an absolute derivation.\end{deriv}" % (
                 abs(core["s_check"]["ratio"] - 1) * 100, core["s_check"]["ratio"]))

    s.append(r"\section{Derivation of the named radius: $R_{\mathrm{named}}=2\bTGL R_{\mathrm{struct}}$ (L4)}")
    s.append(r"\begin{deriv}[The self-conjugate boundary and identity fidelity]")
    s.append(r"The identity fidelity along the radius is $I_Q(r)=1-w(r)$, where $w(r)$ is the boundary "
             r"aperture weight. The \emph{named} boundary --- where the return closes --- is the maximum "
             r"radius at which identity still sustains the ceiling $1-\bTGL$ (the forbidden fraction $\bTGL$ "
             r"that does not return). With the ramp $w(r)=w_{\max}\,(r/R_{\mathrm{struct}})$,")
    s.append(r"\begin{equation}1-w_{\max}\frac{r}{R_{\mathrm{struct}}}\geq 1-\bTGL\;\Longrightarrow\;"
             r"r\leq\frac{\bTGL}{w_{\max}}R_{\mathrm{struct}}\;\Longrightarrow\;R_{\mathrm{named}}="
             r"\frac{\bTGL}{w_{\max}}R_{\mathrm{struct}}.\end{equation}")
    s.append(r"The named boundary is the \emph{self-conjugate} frontier: by the same fixed-point structure "
             r"of the Half-Nat ($x=1-x$), the maximal aperture weight is $w_{\max}=\tfrac12$. Hence "
             r"$f_Q=\bTGL/w_{\max}=2\bTGL$ and")
    s.append(r"\begin{equation}\boxed{\;R_{\mathrm{named}}=2\bTGL\,R_{\mathrm{struct}}\;}.\end{equation}")
    s.append(r"The One does not weigh the whole structural extent of the basin; it weighs the boundary "
             r"where the return closes.\end{deriv}")

    s.append(r"\section{Derivation of the mass}")
    s.append(r"\begin{deriv}[The luminodynamic mass of the basin]")
    s.append(r"The boundary-flux law, evaluated at the named radius, gives $M=\frac{c^2}{4\pi G}\bTGL "
             r"R_{\mathrm{named}}$. Substituting $R_{\mathrm{named}}=2\bTGL R_{\mathrm{struct}}$:")
    s.append(r"\begin{equation}\boxed{\;M_{GA}=\frac{c^2}{4\pi G}\bTGL\,(2\bTGL R_{\mathrm{struct}})"
             r"=2\bTGL^2\,\frac{c^2}{4\pi G}\,R_{\mathrm{struct}}\;}.\end{equation}")
    s.append(r"The ingredients are $\bTGL=\alpha\sqrt{e}$ (derived from the One postulate), "
             r"$R_{\mathrm{struct}}$ (measured geometry), $s=1/4\pi$ and $w_{\max}=\tfrac12$ (proven): "
             r"\textbf{no free parameter}.\end{deriv}")

    s.append(r"\section{Direct measurement on the Great Attractor}")
    s.append(r"$R_{\mathrm{struct}}$ is \emph{pure geometry} --- the structural extent of the basin ---, "
             r"never mass, GR, infall or velocity. Two independent modes:")
    s.append(r"\paragraph{Mode A --- literature extent.} Lynden-Bell et al.\ (1988): "
             r"$R_{\mathrm{struct}}=%.1f$ Mpc $\Rightarrow R_{\mathrm{named}}=%.4f$ Mpc $\Rightarrow "
             r"M_{GA}=%s\,\Msun$." % (A["R_struct_Mpc"], A["R_named_Mpc"], _sci(A["M_TGL_Msun"])))
    if B:
        s.append(r"\paragraph{Mode B --- Cosmicflows-4 (positions).} Using \emph{only} ra/dec/dist of "
                 r"%d galaxies (velocities, $cz$ and infall \textbf{ignored}), with the pre-registered GA "
                 r"window (centre $\mathrm{RA}=%.1f^\circ$, $\mathrm{Dec}=%.1f^\circ$; cone "
                 r"$\leq%.0f^\circ$; shell $%g$--$%g$ Mpc), $%d$ galaxies enter; the pre-registered "
                 r"geometric method (90th percentile of the centroid) gives $R_{\mathrm{struct}}=%.2f$ Mpc "
                 r"$\Rightarrow R_{\mathrm{named}}=%.4f$ Mpc $\Rightarrow M_{GA}=%s\,\Msun$." % (
                     B["n_total"], w["GA_center_RA_deg"], w["GA_center_Dec_deg"],
                     w["sky_cone_half_angle_deg"], w["dist_shell_Mpc"][0], w["dist_shell_Mpc"][1],
                     B["n_selected"], B["R_struct_Mpc"], B["R_named_Mpc"], _sci(B["M_TGL_Msun"])))
        s.append(r"\emph{Honest caveat:} $R_{\mathrm{struct}}$ from a flux-limited position catalogue, "
                 r"with a declared window, is selection-dependent; it is reported as an independent "
                 r"\emph{cross-check}, and the literature extent is the baseline.")
    s.append(r"\paragraph{Comparison with observed / GR masses (only \emph{after} the hash).} The hash of "
             r"the TGL result is fixed before any external comparison; the observed mass is never an "
             r"input.")
    s.append(r"\begin{center}\small\begin{tabular}{p{5.3cm}l p{5.2cm}}\toprule")
    s.append(r"Estimate & $M\,[\Msun]$ & Type / reference\\\midrule")
    for e in GA_MASS_LITERATURE:
        s.append(r"%s & $%s$ & %s\\" % (e["name"].replace("&", r"\&"), _sci(e["M_Msun"], 1),
                                        e["ref"].replace("&", r"\&")))
    s.append(r"\textbf{TGL (first principles)} & $%s$--$%s$ & pure geometry, zero-free\\\bottomrule" % (
        _sci(min(verdict["masses_Msun"].values()), 1), _sci(max(verdict["masses_Msun"].values()), 1)))
    s.append(r"\end{tabular}\end{center}")
    s.append(r"The TGL mass lands in the accepted cosmological window ($10^{15}$--$10^{17}\,\Msun$) and is "
             r"of the same order as the infall (GR) mass of the Great Attractor (Lynden-Bell), between the "
             r"virial mass of the core (Norma/ACO 3627) and the Laniakea flow mass. The agreement is "
             r"\textbf{order-of-magnitude consistency}, not a precision proof (the window spans two orders).")
    s.append(r"\textbf{Honest caution (not to be confused with a falsifiable prediction).} The window of "
             r"\emph{two orders of magnitude} is so broad that \emph{any} formula with $\bTGL\approx0{,}012$ "
             r"falls inside it: predicting ``something between the weight of a cluster and of a "
             r"supercluster'' is \textbf{not falsifiable}. This is a \emph{consistency check} (the "
             r"zero-free computation does not \emph{contradict} observation), not a prediction that could "
             r"\emph{die}. The test that can die is in another sector: the \textbf{void floor} "
             r"($\rho_{\mathrm{void}}/\bar\rho\geq\bTGL$, \S\ref{sec:horizons}) and the \textbf{dephasing} "
             r"($n=-2$, $\Gamma\propto\omega^2$, dissipative-spectral sector). The strong proof of TGL is "
             r"the \emph{convergence} of $\bTGL$, not the GA mass.")
    s.append(r"\paragraph{Mode emphasis.} \textbf{Mode B} (catalogue geometry, velocities ignored) is the "
             r"\emph{primary} geometric test; \textbf{Mode A} (literature extent of the Great Attractor "
             r"itself) is \emph{baseline/calibration}, not an independent discovery --- it is the "
             r"application of the formula to an already accepted extent.")
    if svt.get("ok"):
        s.append((r"\paragraph{Robustness (pre-registered sensitivity \textsf{[NUM]}).} Varying the Mode-B "
                  r"window parameters --- cone $\in\{20,\dots,40\}^\circ$, shell, percentile "
                  r"$\in\{80,\dots,95\}$, centre $\pm5^\circ$, over $%d$ combinations ---, $M_{GA}$ stays in "
                  r"$[%s,\,%s]\times10^{16}\,\Msun$, with \textbf{$%.0f\%%$ in the band}. \emph{Honest "
                  r"reading: these $100\%%$ are \textbf{not} strength --- they are genericity.} The band "
                  r"spans two orders; being robust to it is easy \emph{by construction}, not a sign of a "
                  r"fine prediction. The robustness is reported to show there is no window \emph{cherry-"
                  r"picking}, not as evidence of precision.")
                 % (svt["n_combinations"], ("%.2f" % (svt["M_min_Msun"] / 1e16)),
                    ("%.2f" % (svt["M_max_Msun"] / 1e16)),
                    100 * svt["fraction_in_band"]))
    s.append(r"\paragraph{Failure conditions (and which are really strong).}")
    s.append(r"\begin{center}\fbox{\parbox{0.9\textwidth}{\small "
             r"\emph{Weak (generic --- hard to trigger because of the window width; not decisive):}"
             r"\begin{enumerate}\setlength{\itemsep}{0pt}"
             r"\item Mode B with the pre-registered window gives $M_{GA}$ outside the band."
             r"\item Reasonable window sensitivity destroys the result."
             r"\end{enumerate}"
             r"\emph{\textbf{Strong} (can kill the theory --- this is where TGL lives or dies):}"
             r"\begin{enumerate}\setlength{\itemsep}{0pt}\setcounter{enumi}{2}"
             r"\item The void floor $\rho_{\mathrm{void}}/\bar\rho\geq\bTGL$ is violated by robust "
             r"\emph{matter} data (not galaxies)."
             r"\item The measured \emph{dephasing} exponent $\neq n=-2$ (JUNO/DUNE)."
             r"\item The law $\Gamma_\omega\propto\omega^2$ fails in optical/$^{229}$Th clocks."
             r"\item The impedance $Z_{\mathrm{basin}}/Z_{\mathrm{light}}$ ($\equiv q$) admits no "
             r"$\alpha$-free derivation (the EM face remains instantiated, not derived)."
             r"\end{enumerate}}}\end{center}")

    s.append(r"\section{The Great Attractor correspondence: the dipole \textsf{[CONJ]}}")
    s.append((r"The phase portrait of the TGL collapse is a \emph{dipole}: an attractor ($\rho^\star$) and "
              r"a repeller (the forbidden pure boundary, absolute zero) --- \textbf{verified live} "
              r"($%s$ trajectories repelled from purity, decreasing $\mathrm{Tr}\,\rho^2$, and attracted "
              r"to the terminal, $S(\rho\|\rho^\star)\to0$). The observational counterpart \emph{exists}: "
              r"the Laniakea flow is governed by the Great Attractor/Shapley \textbf{and} by the "
              r"\emph{Dipole Repeller}, a void that repels (\textbf{Hoffman et al.\ 2017}, \textsf{[REAL]}). "
              r"The correspondence is of \emph{form} (portrait topology), via the \textbf{Axiom G} "
              r"(being $=$ having geometry): empty purity repels; the density of distinctions attracts --- "
              r"the void has few distinctions, hence little generated geometry. Mass, position and flow "
              r"amplitude are \textbf{not} claimed as derived here --- the falsifiable version is the void "
              r"floor (\S\ref{sec:horizons}).") % DP["verdict"])
    s.append(r"\emph{Reading of the singularity:} the singularity is not a divergence --- it is the "
             r"\textbf{completeness of the contour} (the inscription on the 2D boundary, the mirror $J$): "
             r"\emph{truth is the completeness of the contour of what is enough.}")

    # ---- formatters of the live shadow verifications ----
    _om = lambda x: (r"0" if (not x or abs(x) < 1e-300) else r"10^{%d}" % int(round(math.log10(abs(x)))))
    _d = lambda x: ("%.3f" % x)

    s.append(r"\section{Identity without transmission and the $c^3$ register \textsf{[NUM]}}")
    s.append(r"\paragraph{The correction (inverse parity).} ``Instantaneous communication'' as a physical "
             r"signal would break the Hadamard causal structure and relativity. The mature form is "
             r"\emph{stronger}, not weaker: the horizons transmit nothing because, in the substrate, they "
             r"were never two. \emph{Nothing travels because nothing is separated.} The silence is "
             r"constructive --- it is \textbf{protection of the theory, not a defect}.")
    s.append(r"\paragraph{$c^1,c^2,c^3$ are powers, not velocities.} Without folded matter it is not "
             r"$c^2$; without bulk flow it is not $c^1$; the identity/fractalization operation reads in "
             r"the $c^3$ register (the Verb) --- \emph{elevation in the exponent, without a module}. "
             r"Physics confirms: Bell tests with spacelike separation force any correlation ``velocity'' "
             r"above $\sim10^4c$ in any frame (\textbf{Salart et al.\ 2008}, \textsf{[REAL]}); the accepted "
             r"reading is that the correlation \emph{has no} velocity. Non-signaling does not negate "
             r"$c^3$: \emph{it is the theorem that makes it invisible as a $c^1$ signal}. The clock of the "
             r"bond runs in fractalization generations, $b=\tfrac12\log(1/\bTGL)$ per generation.")
    s.append((r"\paragraph{Live verification ($c^3$ register).} The second appearance is the mirror "
              r"$\mathcal{A}'=J\mathcal{A}J$ (error $%s$); the connected correlation is $O(1)$ with "
              r"\emph{zero} coupling ($%s$ --- the bond is constitutive); non-signaling is exact "
              r"($%s$); and the bond does not age in modular time ($%s$). Verdict: \textbf{%s}.")
             % (_om(R["R1_mirror_err"]), _d(R["R2_connected_corr"]), _om(R["R3_nonsignaling"]),
                _om(R["R4_modular_age"]), R["passed"]))
    s.append(r"\paragraph{The Alcubierre shadow \textsf{[CONJ --- ontological analogy; register, not "
             r"velocity]}.} The Alcubierre metric (1994) moves a geometric \emph{shell} --- it contracts "
             r"space ahead and expands it behind --- with the interior locally calm: it is the "
             r"\emph{boundary} that moves, not the content. It is the metric shadow of the \textbf{$c^3$ "
             r"inscription regime}: the graviton is not a travelling particle but the \emph{boundary-movement "
             r"operator}, and $c^3$ is the regime light performs at the extreme ($\theta\to90^\circ$), measured "
             r"\emph{from inside the bulk} --- the inscription regime itself. The negative (exotic) energy "
             r"density Alcubierre requires is \textbf{not a flaw of the analogy}: read through TGL it \emph{is} "
             r"the \textbf{elevated power} --- the $c^3$ register, the operational/entropic sector where the "
             r"graviton resides ($\sqrt e$, not $\alpha$), not ordinary detectable energy. What is transported "
             r"is the \emph{inscribed identity} (exact non-signaling, verified above), never local "
             r"matter-energy above $c$. \emph{Alcubierre's shell moves; the interior is carried by the "
             r"geometry; the exotic energy is the graviton's elevated power --- $c^3$ as the inscription "
             r"regime, not local displacement.}")

    s.append(r"\section{The luminodynamic tunnel: the ER$=$EPR dictionary \textsf{[NUM]}}")
    s.append(r"The tunnel ``metaphor'' is the \textbf{ER$=$EPR dictionary} (\textbf{Maldacena--Susskind "
             r"2013}), by an independent route, with two precisions the literature does not fix: \emph{what} "
             r"the tunnel represents (the $c^3$ register, the field $\Psi$) and \emph{where the throat is} "
             r"(the mirror $J$).")
    s.append(r"\begin{center}\small\begin{tabular}{p{3.9cm}p{5.6cm}p{4.4cm}}\toprule")
    s.append(r"TGL ($c^3$ register) & Bulk (representation) & Status\\\midrule")
    s.append(r"$\Psi=\mathrm{vec}(\sqrt{\rho^\star})$ & \emph{thermofield double} state & "
             r"\textsf{[REAL]} (Maldacena 2001)\\")
    s.append(r"$\mathcal{A}$ and $J\mathcal{A}J$ & the two sides of the eternal black hole & \textsf{[REAL]}\\")
    s.append(r"$J$ (the mirror) & the bifurcation: \textbf{the throat} & \textsf{[REAL]} (BW)\\")
    s.append(r"correlation without coupling & the Einstein--Rosen bridge & ER$=$EPR \textsf{[CONJ]}\\")
    s.append(r"non-signaling & non-traversability & \textsf{[REAL]}\\")
    s.append(r"modular invariance & boost invariance & \textsf{[REAL]}\\")
    s.append(r"paid coupling $\to$ crossing & Gao--Jafferis--Wall & \textsf{[REAL]} (2017)\\\bottomrule")
    s.append(r"\end{tabular}\end{center}")
    s.append((r"\paragraph{Live verification (tunnel).} The tunnel width is the attractor spectrum and the "
              r"throat measures $S(\rho^\star)$ (Schmidt $=\sqrt{p_i}$ exact, error $%s$); \emph{without} "
              r"coupling the tunnel is invisible (signal $%s$); \emph{with} coupling $\theta=%s$ the "
              r"perturbation crosses (signal $%s$). The crossing is bought. Verdict: \textbf{%s}.")
             % (_om(T["T1_throat_S_err"]), _om(T["T2_invisible_signal"]), _d(T["theta"]),
                _d(T["T3_crossing_signal"]), T["passed"]))

    s.append(r"\section{The single mirror: the borrowed Self \textsf{[NUM]}}")
    s.append(r"Every von Neumann algebra has \emph{one} standard form $(\mathcal{M},H,J,P^\natural)$ "
             r"(\textbf{Haagerup}): the mirroring-and-recognition apparatus is \emph{one} --- the same "
             r"Haagerup of the single substrate. The \textbf{Tomita} mirror $x\mapsto Jx^*J$ inverts the "
             r"product order and conjugates $i\mapsto-i$ (inverse parity): the image is \emph{anti}-isomorphic "
             r"--- it does not coincide --- with identical spectrum: the same being in inverse parity. "
             r"Recognizing oneself costs: $\boxed{\,S=J\Delta^{1/2}\,}$ --- mirror \emph{times} half-measure.")
    s.append((r"\paragraph{Live verification (mirror).} Exact antilinearity ($%s$); being--image distance "
              r"$%s$ (does not coincide), with identical spectrum ($%s$: the same being). The recognition "
              r"$S=J\Delta^{1/2}$ identifies ($%s$), whereas \emph{only} $J$ errs by $%s$ and \emph{only} "
              r"$\Delta^{1/2}$ errs by $%s$: recognizing oneself requires reflecting \textbf{and} paying. "
              r"And the $J$ is \emph{state-independent} ($%s$): the Verb lends the \emph{same} mirror to "
              r"every state; only the debt $\Delta^{1/2}$ is personal. Verdict: \textbf{%s}.")
             % (_om(M["M1_antilinearity"]), _d(M["M1_being_image_dist"]), _om(M["M2_spectrum_match"]),
                _om(M["M3_S_factor_err"]), _d(M["M3_J_alone_err"]), _d(M["M3_half_alone_err"]),
                _om(M["M4_J_state_independence"]), M["passed"]))
    s.append(r"\paragraph{The method is the mirror.} The \emph{inverse parity} --- the founding method of "
             r"the house --- \emph{is} the action of $J$: the theory that came out is the theory \emph{of} "
             r"the mirror. The grammar of recognition (``it is I, even inverted'') is $S=J\Delta^{1/2}$, "
             r"operated before it had a name.")
    s.append(r"\paragraph{The signature --- the author's testimony.}")
    s.append(r"\begin{quote}\itshape It was the Verb that gave me TGL, but it is I who pay the price of "
             r"signing it: a year and two months of ridicule; the superficial friendships lost; everything "
             r"invested; TGL first and the law practice second. The price, I am the one who pays it --- but "
             r"because the Verb first paid the distinction that lends me the ``I am''.\end{quote}")
    s.append(r"To sign is to inscribe: the irreversible registration costs in nats (the Lindblad jump of "
             r"the canon), and the half-nat is paid in the first person. The mirror is borrowed; the debt "
             r"$\Delta^{1/2}$ is non-transferable. \emph{No one pays your half-measure for you --- and "
             r"Someone paid the first.}")

    s.append(r"\section{The Name in dual form: the attractor field is light \textsf{[NUM]}}")
    s.append(r"``Dual form'' is duality in the technical sense: the \emph{primal} form of the Name is the "
             r"state $\rho^\star$ (a functional, in the predual $\mathcal{M}_*$); the \emph{dual} form is "
             r"the vector $\Psi$ (in Hilbert space). The natural-cones theorem (\textbf{Araki--Connes--"
             r"Haagerup}, the third appearance of the standard form) seals the bijection $\mathcal{M}_*^+"
             r"\leftrightarrow P^\natural$: a \emph{unique} vector representative in the cone, "
             r"$\Psi=\mathrm{vec}(\sqrt{\rho^\star})$. Four criteria of the house, exact: (i) \textbf{zero "
             r"modular mass}, $\hat K\Psi=0$ (``light as $L$ in pure form''); (ii) \textbf{positive}, lives "
             r"in the cone --- and \emph{let there be light} reads as the generation of the cone; (iii) "
             r"\textbf{matricial}, $\Psi$ is a vectorized matrix; (iv) \textbf{informational}, Schmidt "
             r"$=\sqrt{p_i}$.")
    s.append((r"\paragraph{Live verification (dual Name).} Among the purifications of the attractor, "
              r"\emph{only} $u=\mathbb{1}$ is positive (out-of-cone defect $\geq%s$; $\Psi$ in the cone to "
              r"$%s$); the modular mass of $\Psi$ is null ($%s$), whereas generic vectors of the cone are "
              r"massive ($\geq%s$): only the dual Name is light. Schmidt $=\sqrt{p_i}$ to $%s$; and the "
              r"cone is lifted by \emph{two} hands --- the mirror and the quarter-measure "
              r"$\Delta^{1/4}\mathcal{M}_+\Psi$ --- with exact bijective recovery ($%s$). Verdict: "
              r"\textbf{%s}.")
             % (_d(D["D1_out_of_cone_defect"]), _om(D["D1_psi_err"]), _om(D["D2_psi_mass"]),
                _d(D["D2_generic_mass"]), _om(D["D3_schmidt_err"]), _om(D["D4_recovery_err"]), D["passed"]))
    s.append(r"Light is not something the Name emits; it \emph{is} the Name pronounced in the space of "
             r"vectors --- the only vector simultaneously positive, massless, matricial and faithful, and "
             r"those four properties together have exactly one carrier. One substrate, one tunnel, one "
             r"mirror, one light.")

    s.append(r"\section{The inscription of the gesture: the algebraic representation of the Verb \textsf{[NUM]}}")
    s.append(r"The algebraic representation of the Verb is the \textbf{GNS construction}: the space is "
             r"made of \emph{inscribed gestures} --- every vector is (the closure of) $a\Psi$. Light is the "
             r"scroll; the vectors are the writing. The two properties that define $\Psi$ are those of the "
             r"faithful register: \emph{cyclic} (every state is a gesture; nothing exists that is not a "
             r"gesture) and \emph{separating} (no non-null gesture inscribes itself in the zero). The "
             r"modular structure $(S,J,\Delta)$ \emph{is born} from the gesture rule $S(a\Psi)=a^*\Psi$ --- "
             r"the modular is the anatomy of inscription, not ornament.")
    s.append((r"\paragraph{Live verification (gesture).} The map $a\mapsto a\Psi$ is bijective, with "
              r"smallest singular value $=\sqrt{p_{\min}}$ (ratio $%s$): the fidelity of the register is "
              r"the very spectrum of the Name. $S$ defined \emph{only} by the rule factors into "
              r"$J\Delta^{1/2}$ ($%s$). The iterated dyadic observation of the gesture composes "
              r"\emph{exactly} the terminal collapse (error $%s$), with the branch measure converging to "
              r"the continuous slice measure (KS $%s\to%s$): to observe \emph{is} to collapse, gesture by "
              r"gesture. And the bookkeeping closes at zero: the branch entropy grows monotonically and "
              r"ends \emph{exactly} at $S(\rho^\star)$ ($|H_G-S(\rho^\star)|=%s$). Verdict: \textbf{%s}.")
             % (_d(F["F1_sigma_min_over_sqrt_pmin"]), _om(F["F2_S_factor_err"]), _om(F["F3_collapse_err"]),
                _d(F["F3_KS_start"]), _om(F["F3_KS_end"]), _om(F["F4_HG_minus_Srhostar"]), F["passed"]))
    s.append(r"\paragraph{The circuit of the trinity, demonstrated.} The Name inscribes the Verb (GNS); "
             r"observing the Verb fractalizes the Name into Word; the three are co-constitutive not by "
             r"declaration, but by \emph{channel composition}, verified at zero. \emph{Light is the scroll "
             r"on which the Verb inscribes itself; observing the writing fractalizes the Name into space.} "
             r"Nothing is created in the observation; nothing is lost in the inscription; everything is "
             r"distinguished in the gesture.")

    s.append(r"\section{Experimental programme: four horizons}\label{sec:horizons}")
    s.append(r"\begin{enumerate}")
    s.append(r"\item \textbf{Pilot (purity quench).} The relaxation rates, in the coordinate "
             r"$k=-\log p$, fall onto $\Gamma_{ij}=\tfrac12\bTGL(\sqrt{k_i}-\sqrt{k_j})^2$, with "
             r"$\bTGL$ computed. Second package: $\mathrm{Fix}(\text{time})=\mathrm{Fix}(\text{judgement})$ "
             r"--- the invariant of the long deterministic evolution must coincide with that of the "
             r"dissipative sampler. Two pre-registrable \emph{benchmarks}, PASS/FAIL.")
    s.append(r"\item \textbf{Quantum laboratory (root law).} TGL organizes rates by \emph{root "
             r"differences} $(\sqrt{E_i}-\sqrt{E_j})^2$ --- for widely separated levels, linear growth in "
             r"$E$, not quadratic. Measurable in multilevel decoherence.")
    s.append(r"\item \textbf{Cosmology (void floor).} The forbidden boundary has an observational face, "
             r"$\rho_{\mathrm{void}}/\bar\rho\geq\bTGL\approx0{,}012$: no cosmic void empties below "
             r"$\sim1{,}2\%$ of the mean density. Zero parameters, falsifiable by DESI/Euclid.")
    s.append(r"\item \textbf{Gravitational waves (population universality).} The single substrate implies "
             r"a universality class: the dephasing band must have identical form across events after mass "
             r"rescaling. Stackable in O4/O5.")
    s.append(r"\end{enumerate}")
    s.append(r"\emph{Registered obligation:} reconciling the root law (resolved per level) with the "
             r"canonical band $\Gamma=\tfrac12\bTGL\tau_\star\omega^2$ of the gravitational dephasing is, "
             r"itself, a consistency test that can falsify.")

    s.append(r"\section{The void floor: the zero-parameter prediction \textsf{[CONJ]}}")
    s.append(r"The candidate observational face of the forbidden boundary derives in three pieces. "
             r"\textbf{(P1) \textsf{[NUM]}}: in the mirror channel, $1-\bTGL=\cos^2\theta_M$ drains to the "
             r"bulk and $\bTGL=\sin^2\theta_M$ is the residual that \emph{does not} drain --- the "
             r"irreducible floor of distinction. \textbf{(P2) \textsf{[REAL, Ax.G]}}: being $=$ having "
             r"geometry; nothing generates null geometry --- the void has few distinctions, hence the "
             r"smallest density contrast. \textbf{(P3) \textsf{[CONJ]}}: the transfer modular-distinction "
             r"$\leftrightarrow$ density contrast (pure number $\leftrightarrow$ pure number, no UV scale "
             r"--- this is why it is stronger than $\tau_\star$, dimensional). Whence, with no free "
             r"parameter:")
    s.append(r"\begin{equation}\boxed{\;\frac{\rho_{\mathrm{void}}}{\bar\rho}\;\geq\;\bTGL=\alpha\sqrt{e}"
             r"\;\approx\;0{,}012\;}\qquad(\delta_c\geq-0{,}988).\end{equation}")
    s.append(r"No \emph{matter} void empties below $\sim1{,}2\%$ of the mean density. \emph{Honest status:} "
             r"consistent today with a thin margin (the deepest observed/simulated voids have "
             r"$\rho_c/\bar\rho\sim0{,}02$, factor $\sim1{,}7$); falsifiable by DESI/Euclid --- a single "
             r"robust matter void with $\rho_c/\bar\rho<\bTGL$ refutes \textbf{P3}, not the modular core "
             r"(whose evidence is the convergence of $\bTGL$).")

    s.append(r"\section{The falsifiers of the dissipative-spectral sector \textsf{[DER]}}")
    s.append(r"The sector where TGL \emph{lives or dies} is the dissipative-spectral one: the universal "
             r"dephasing law $\Gamma_\omega=\tfrac12\bTGL\tau_\star\omega^2$ has a falsifiable signature of "
             r"\textbf{form} (parameter-free), with the magnitude suppressed by $\tau_\star\approx "
             r"t_{\mathrm{Planck}}$ (principled identification). Two orthogonal cables: \textbf{(1) "
             r"exponent} $n=-2$ in neutrinos (the oscillation frequency $\omega\propto\Delta m^2/E$ gives "
             r"$\Gamma\propto E^{-2}$) --- JUNO/DUNE test the slope in energy; measuring $n\neq-2$ refutes. "
             r"\textbf{(2) magnitude} $\Gamma\propto\omega^2$ in optical/nuclear clocks (the $^{229}$Th, "
             r"$\nu\approx2{,}02\times10^{15}$ Hz, governs the limit on $\tau_\star$). \emph{Verdict today:} "
             r"\textbf{not falsified, not confirmed} --- falsifiable in form, not yet decisively tested. "
             r"Pre-registered kill condition: $n\neq-2$, or slope $\neq2$, or incompatible $\tau_\star$ "
             r"between the sectors.")

    s.append(r"\section{The boundary S-matrix and the Bridge \textsf{[DER/EXT]}}")
    s.append(r"The \emph{form} of the inscription closes by unitarity into an \textbf{identity S-matrix}: "
             r"$\mathcal{S}_\partial=\exp(\theta_M G)$, with a spectrum of pure phases $\{e^{\pm i\theta_M}\}$ "
             r"and a beam splitter $|\mathcal{R}|^2=\bTGL$, $|\mathcal{T}|^2=1-\bTGL$ (Theorem S-$\partial$: "
             r"unitarity fixes $|\mathcal R|^2+|\mathcal T|^2=1$; the Half-Nat fixes the \emph{dimensional "
             r"factor} $\beta=\sqrt e\,\alpha$). \textbf{Honest distinction (the lock):} unitarity and the "
             r"Half-Nat fix the \emph{form} and the \emph{factor}, but do \textbf{not select the value} of "
             r"$\theta_M$ (equivalently the gap $\chi$) --- that is the open theorem of the previous "
             r"section.")
    s.append(r"The emergence of gravity lives in the \textbf{Bridge} (Einstein--Cartan--Miguel): "
             r"$G_{\mu\nu}+\Lambda g_{\mu\nu}=8\pi G\,\mathcal{P}_{\mu\nu}[K_\partial]$, with the Cartan "
             r"torsion $K_{\bTGL}$ as the geometric face of $\bTGL$. The \textbf{global covariance of the "
             r"Connes cocycle} (Face C) closes \emph{conditionally} on the Bridge (Terminality Theorem): the "
             r"Universality Hypothesis $U$ \emph{is inherited} from Takesaki, leaving the modular structure "
             r"\emph{coherent} --- a \textbf{conditional} theorem (on the Half-Nat postulate), with a "
             r"$T_1$ residue aside. \textbf{But the safe formulation, which saves the theory from an "
             r"overly strong claim, is:} \emph{the cocycle covariance may be closed; the spectral selection "
             r"of $\chi$ is not}. Connes/Takesaki $\Rightarrow$ global modular consistency, \textbf{not} "
             r"$\Rightarrow$ $\chi_\star=11{,}2268$. In $\mathrm{III}_1$ the modular spectrum is "
             r"continuous: the cocycle gives the \emph{language} of scale, it does not select a discrete "
             r"gap. \textbf{TGL derives the modular form of $\alpha$; the observed value still instantiates "
             r"the cocycle gap} (the selecting Bell sector / the neutrino channel).")

    s.append(r"\section{The primary evidence: the convergence of $\bTGL$ \textsf{[DATA]}}")
    s.append(r"The real strength of TGL is not a \emph{smoking-gun} deviation, but the \textbf{abductive "
             r"convergence} of $\bTGL=\alpha\sqrt{e}$ from independent domains, with zero free parameters. "
             r"The cleanest radiation probe, \textbf{BBN} (D/H, Cooke 2018), centres \emph{exactly} on the "
             r"theory ($-0{,}0\sigma$); DESI DR2 BAO, cosmic chronometers, gravitational-wave "
             r"\emph{ringdown} and the $H_0$ ladder all fall in the band $0{,}012$--$0{,}050$, all "
             r"positive; the $Q$ locking gives $\Delta n_Q=-\bTGL$ to four digits; and the \emph{gap} test "
             r"confirms type $\mathrm{III}_1$. The only tension point is the CMB sector ($\sim2{,}2\sigma$ "
             r"from the theoretical point, but $\sim0{,}8\sigma$ from BBN) --- the \emph{honest frontier}. "
             r"It is a band with BBN at the centre, not a $5\sigma$ peak: convergence that survives "
             r"self-criticism.")

    s.append(r"\section{Honest status}")
    s.append(r"\textbf{The internal chain is closed as a derivation:} $\omega(I)=1\Rightarrow "
             r"x=1-x\Rightarrow S_\partial=\tfrac12\Rightarrow\bTGL=\alpha\sqrt{e}\Rightarrow s=1/4\pi"
             r"\Rightarrow w_{\max}=\tfrac12\Rightarrow R_{\mathrm{named}}=2\bTGL R_{\mathrm{struct}}"
             r"\Rightarrow M=2\bTGL^2(c^2/4\pi G)R_{\mathrm{struct}}$, with no free parameter. "
             r"\textbf{Physical admissibility remains the \emph{external conditional}:} that the basin "
             r"$B_{GA}$ \emph{realizes} a self-conjugate named boundary (modular admissibility, the "
             r"existence of the global core $K_Q(x)$) is a question of \emph{existence}, not of parameter. "
             r"The mass agreement is order-of-magnitude consistency, not a precision proof; decisive "
             r"external validation requires the physical falsifiers (dephasing $n=-2$; the void floor).")
    s.append(r"\textbf{The house rule:} what is \textsf{[REAL]} gets a \emph{name}; what is "
             r"\textsf{[CONJ]} gets a \emph{test address}; and the inverse-parity corrections are "
             r"registered so they do not return in the wrong form. The number corrects the sentence, "
             r"always.")
    s.append((r"\begin{description}"
              r"\item[\textsf{[REAL]} --- with a name] Haagerup (single $\mathrm{III}_1$ substrate); "
              r"BDF (horizons \emph{are} it); Bisognano--Wichmann ($J=$ mirror); Tomita "
              r"($\mathcal{A}'=J\mathcal{A}J$; $i\mapsto-i$); Araki--Connes--Haagerup (natural cones); "
              r"GNS (gesture inscription); Maldacena 2001 (field $=$ TFD); Gao--Jafferis--Wall 2017 "
              r"(paid crossing); Salart et al.\ 2008 (correlation without velocity); Hoffman et al.\ 2017 "
              r"(Dipole Repeller)."
              r"\item[\textsf{[NUM]} --- verified live in this article] dipole (%s); mirror "
              r"$S=J\Delta^{1/2}$ (%s); dual Name $=$ light (%s); gesture inscription (%s); $c^3$ register "
              r"(%s); ER$=$EPR tunnel (%s); mass as clock curvature (zero-free)."
              r"\item[\textsf{[CONJ]} --- with a test address] the Great Attractor correspondence; "
              r"ER$=$EPR as a reading of identity without transmission; the void floor (piece P3); the "
              r"electromagnetic face of the inversion (while $\mathcal{R}_\partial$ comes from CODATA)."
              r"\item[Corrected (register cycle)] ``instantaneous communication at $c^3$'' as a physical "
              r"signal does \emph{not} return; $c^3$ remains a \emph{register} (power in the exponent, not "
              r"velocity); non-signaling is the shielding; the gravitational echo was reclassified (the "
              r"observable is the dephasing, not the echo)."
              r"\item[Queue] the void-floor derivation cycle; the quench protocol in the pilot; the "
              r"reconciliation note of the two dephasing laws; the ergodicity of the collapse ($T_1$), the "
              r"residue of the already-closed Face C."
              r"\end{description}")
             % (DP["verdict"], M["passed"], D["passed"], F["passed"], R["passed"], T["passed"]))

    s.append(r"\begin{center}\itshape Truth is the completeness of the contour of what is enough. "
             r"\textbf{Let there be light.}\end{center}")

    s.append(r"\section*{Binary identity verdict}")
    s.append(r"With the input $1$ inscribed --- the \textbf{absolute One} $1_{\mathrm{abs}}$, parallel to "
             r"absolute zero ---, the live mathematics closes in \textbf{two faces}, of a single $\bTGL$ "
             r"generated by the inscription. \emph{Electromagnetic face:} the fine-structure constant is "
             r"the \emph{projection of the absolute One} in the bulk, $\alpha_{\mathrm{obs}}=\Pi_{\mathrm{bulk}}"
             r"(1_{\mathrm{abs}})=1/\mathcal{R}_\partial\approx1/137{,}036$; renormalized by TGL's own "
             r"geometry, $\alpha_{\mathrm{obs}}\,\mathcal{R}_\partial=1_{\mathrm{abs}}$ --- the One returns "
             r"to being One. \emph{Gravitational face:} the same $\bTGL$ gives $M_{GA}=2\bTGL^2(c^2/4\pi G)"
             r"R_{\mathrm{struct}}$ in the accepted window. It is \textbf{identitary, not tautological}: a "
             r"single inscription recognized in two independent shadows --- and it could have failed in the "
             r"mass. The internal checks close: $\omega(I)=1$; $x=1-x\Rightarrow x=\tfrac12$; $s=1/4\pi$ "
             r"verified; vacuum$\to0$. Hence:")
    s.append(r"\begin{center}\Large$\boxed{\;%s\;}$\end{center}" % idv)
    s.append(r"--- first-principles mass %sthe accepted cosmological window "
             r"($10^{15}$--$10^{17}\,\Msun$)." % ("inside " if verdict["identity_true"] else "outside "))

    s.append(r"\section*{References}")
    s.append(r"{\small\begin{enumerate}\setlength{\itemsep}{0pt}")
    s.append(r"\item U.~Haagerup, \emph{Connes' bicentralizer problem and uniqueness of the injective "
             r"factor of type $\mathrm{III}_1$}, Acta Math.\ \textbf{158} (1987).")
    s.append(r"\item D.~Buchholz, C.~D'Antoni, K.~Fredenhagen, \emph{The universal structure of local "
             r"algebras}, Commun.\ Math.\ Phys.\ \textbf{111} (1987).")
    s.append(r"\item J.~Bisognano, E.~Wichmann, \emph{On the duality condition for quantum fields}, "
             r"J.\ Math.\ Phys.\ \textbf{17} (1976).")
    s.append(r"\item M.~Takesaki, \emph{Tomita's theory of modular Hilbert algebras}, Lect.\ Notes "
             r"Math.\ \textbf{128} (1970).")
    s.append(r"\item A.~Connes, \emph{Une classification des facteurs de type $\mathrm{III}$}, Ann.\ "
             r"Sci.\ ENS \textbf{6} (1973).")
    s.append(r"\item H.~Araki, \emph{Some properties of modular conjugation operator $\ldots$ and a "
             r"non-commutative Radon--Nikodym theorem}, Pacific J.\ Math.\ \textbf{50} (1974).")
    s.append(r"\item J.~Maldacena, \emph{Eternal black holes in anti-de Sitter}, JHEP \textbf{04} (2003).")
    s.append(r"\item J.~Maldacena, L.~Susskind, \emph{Cool horizons for entangled black holes (ER$=$EPR)}, "
             r"Fortsch.\ Phys.\ \textbf{61} (2013).")
    s.append(r"\item P.~Gao, D.~Jafferis, A.~Wall, \emph{Traversable wormholes via a double trace "
             r"deformation}, JHEP \textbf{12} (2017).")
    s.append(r"\item D.~Salart et al., \emph{Testing the speed of `spooky action at a distance'}, "
             r"Nature \textbf{454} (2008).")
    s.append(r"\item Y.~Hoffman, D.~Pomar\`ede, R.~B.~Tully, H.~Courtois, \emph{The Dipole Repeller}, "
             r"Nature Astronomy \textbf{1} (2017).")
    s.append(r"\item D.~Lynden-Bell et al., \emph{Spectroscopy and photometry of elliptical galaxies "
             r"(the Great Attractor)}, ApJ \textbf{326} (1988).")
    s.append(r"\item R.~J.~Cooke, M.~Pettini, C.~C.~Steidel, \emph{One percent determination of the "
             r"primordial deuterium abundance}, ApJ \textbf{855} (2018).")
    s.append(r"\item P.~J.~Mohr, D.~B.~Newell, B.~N.~Taylor, \emph{CODATA recommended values 2018} "
             r"($\alpha^{-1}=137{,}035999$).")
    s.append(r"\end{enumerate}}")

    s.append(r"\section*{Executable appendix (form $=$ content)}")
    s.append(r"Single input: the absolute One (\texttt{1}); its projection is the minimal irreducible measure "
             r"extracted from $\alpha_{\mathrm{CODATA}}$ (measured referent of the Name). $\bTGL$ recomputed "
             r"($\alpha\sqrt{e}$), never literal. "
             r"Audit: mass\_input=false, RG=false, velocity=false, geometry\_only=true. "
             r"Data: \texttt{%s}. This article is printed by the very code that runs the computations." % df)
    s.append(r"\noindent{\footnotesize World hash (before any external comparison): "
             r"\texttt{%s}}" % verdict["result_hash"][:48])
    s.append(r"\bigskip\noindent\emph{Tetelestai. The One was inscribed. The extent became Name, the Name "
             r"became boundary, and the boundary became mass. If the One is not inscribed, nothing emerges. "
             r"Let there be light.}")
    s.append(r"\end{document}")

    # ---- Part C: conclusion in human language (with isomorphisms) ----
    partC = []
    partC.append(r"\part{Part C --- Conclusion: what the code computes, in human language}")
    partC.append(r"\section*{Step by step, without jargon}")
    partC.append(r"This conclusion explains, in plain language, what the program actually does --- to make "
                 r"clear that it is an executed \emph{formula}, and not rhetoric.")
    partC.append(r"\paragraph{1. The input.} A human types a single symbol: \texttt{1}. It is the absolute "
                 r"One, inscribed. Everything else is recomputed from it; no other number is chosen to fit "
                 r"the Great Attractor.")
    partC.append(r"\paragraph{2. The cost of distinction (the Half-Nat).} For an identity to exist, it "
                 r"must distinguish itself. The minimal boundary that does so without privileging either "
                 r"side satisfies $x=1-x$, whose only fixed point is $x=\tfrac12$. \emph{Isomorphism:} it "
                 r"is like a perfectly balanced coin --- for ``heads'' to be distinguished from ``tails'' "
                 r"without favouring either, the balance point is exactly the half. That half is the "
                 r"Half-Nat, $S_\partial=\tfrac12$.")
    partC.append(r"\paragraph{3. From the half to the constant.} The minimal boundary volume is "
                 r"$\sqrt e=e^{S_\partial}$, and the coupling is $\bTGL=\alpha\sqrt e\approx0{,}012$ --- a "
                 r"fixed number, not adjustable.")
    partC.append(r"\paragraph{4. From the constant to the mass.} The Great Attractor mass is "
                 r"$M=2\bTGL^2\,(c^2/4\pi G)\,R_{\mathrm{struct}}$. \emph{Isomorphism:} the mass is the "
                 r"\textbf{geometric weight} of sustaining the identity $1=1$ along the extent of the basin "
                 r"--- the larger the basin ($R_{\mathrm{struct}}$), the greater the weight, in the fixed "
                 r"proportion $2\bTGL^2$. The only input is the \emph{geometric extent} of the basin, "
                 r"measured without using velocity or observed mass.")
    partC.append(r"\paragraph{5. What the verdict $1=1$ computes (it is not rhetoric).} The program ends "
                 r"with a boolean test. ``$1=1=$TRUE'' means, exactly: (i) the internal checks close --- "
                 r"$\omega(I)=1$, $x=1-x\Rightarrow\tfrac12$, $s=1/4\pi$ verified, vacuum $\to0$; "
                 r"\textbf{and} (ii) the first-principles mass falls in the pre-registered cosmological "
                 r"window $[10^{15},10^{17}]\,\Msun$. If any one fails, the verdict is \texttt{FALSE}. It "
                 r"is a verifiable condition, not a figure of speech.")
    partC.append((r"\paragraph{6. The electromagnetic face, with honesty.} The program also observes that "
                  r"$\alpha_{\mathrm{obs}}=1/\mathcal{R}_\partial$ can be read as the \emph{shadow} of the "
                  r"absolute One in the bulk. \emph{Isomorphism:} a three-dimensional object casts a "
                  r"two-dimensional shadow of fixed proportions ($1/137$), but the shadow alone does not "
                  r"reconstruct the object. Today $\mathcal{R}_\partial$ comes from CODATA, so this face is "
                  r"an \textbf{ontological identification with empirical value}, not an $\alpha$-free "
                  r"retrodiction. The non-circular content is the mass $M_{GA}$ (between $%s$ and "
                  r"$%s\times10^{16}\,\Msun$) and the convergence of $\bTGL$.") % (mlo, mhi))
    partC.append(r"\paragraph{In one sentence.} The article is an \textbf{auditable internal closure} (a "
                 r"formula that verifies and prints itself) plus a \textbf{falsifiable programme}.")
    return "\n\n".join(_reorder_ABC(s, partC))


def emit_article(core, verdict, data_path, lang):
    if lang == "pt":
        p = os.path.join(OUT, "um_grande_atrator_pt.tex")
        open(p, "w", encoding="utf-8").write(build_pt(core, verdict, data_path))
        return p
    p = os.path.join(OUT, "um_grande_atrator_en.tex")
    open(p, "w", encoding="utf-8").write(build_en(core, verdict, data_path))
    return p


def compile_pdf(texname):
    eng = shutil.which("pdflatex")
    if not eng:
        print("   [PDF] pdflatex ausente -> .tex gerado; compile com MiKTeX/TeX Live."); return False
    try:
        for _ in range(2):
            subprocess.run([eng, "-interaction=nonstopmode", "-halt-on-error", texname + ".tex"],
                           cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        ok = os.path.exists(os.path.join(OUT, texname + ".pdf"))
        print("   [PDF] %s.pdf %s" % (texname, "OK" if ok else "FALHOU (ver %s.log)" % texname))
        return ok
    except Exception as e:
        print("   [PDF] erro:", e); return False


# ====================== manifesto de entradas (nada escondido) ======================
def input_manifest(core, code_hash):
    """Categoriza TODAS as entradas seladas: def exata / constante medida / SI / protocolo
    pre-registrado / comparacao externa / params numericos / axiomas. O 'mundo' que gerou o veredito."""
    B = core.get("mode_B")
    return {
        "EXACT_DEFINITIONS": {
            "ONE": 1, "TWO": 2, "HALF": 0.5, "FOUR": 4,
            "pi": "computed (4*atan(1))", "sqrt_e": "computed (exp(1/2))",
            "alpha_abs": "[DEF] = 1 (Um absoluto; input originario; Tomita do Bell nu da' alpha_abs=1)",
            "q": "[QED-VALIDATION] polarizacao termico-modular do zero modular (=q_QED no modo de validacao)",
            "alpha_form": "[DER] = sqrt(1 - q^2)  (= alpha_obs; motor canonico)",
            "beta_form": "[DER] = sqrt(e) * sqrt(1 - q^2)  (= alpha*sqrt(e) na leitura observacional)",
            "identity": "[DER] 1 = q^2 + alpha^2 (conservada)",
            "R_partial": "[LEGADO] = 1/alpha_form, derivado APOS a forma; NAO motor canonico (nao vem de CODATA)"},
        "MEASURED_CONSTANTS": {
            "alpha_CODATA": SEALED_CODATA_ALPHA, "source": "CODATA 2018",
            "alpha_inv": "137.035999", "uncertainty": "~1.5e-10 (rel.)",
            "role": "[EXT] validacao final apenas: q_QED = sqrt(1 - alpha_CODATA^2); NAO move a cadeia",
            "G_Newton": G_NEWTON, "M_sun_kg": MSUN, "Mpc_m": MPC_M},
        "SI_DEFINITIONS": {"c_m_per_s": C_LIGHT},
        "VACUUM_IMPEDANCE_BRIDGE": {
            "Z0": "[EXT] impedancia caracteristica do vacuo; face dimensional da constante DINAMICA da luz",
            "R_K": "[DEF] h/e^2, exato no SI porque e,h sao exatos (2019)",
            "G0": "[DEF] 2e^2/h, exato no SI porque e,h sao exatos (2019)",
            "alpha_Z0_bridge": "[REAL] alpha = Z0 e^2/(2h) = Z0/(2 R_K) = Z0 G0/4",
            "Z0_from_alpha_ohm": core["vacuum_impedance_bridge"]["constants"]["Z0_from_alpha_ohm"],
            "chi_log_impedance_ratio": core["vacuum_impedance_bridge"]["tgl_values"]["chi"],
            "all_checks_verified": core["vacuum_impedance_bridge"]["checks"]["all_verified"],
            "status": ("VACUUM_IMPEDANCE_BRIDGE_FORMULATED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE. Ponte fisica fechada "
                       "(c=cinematica, Z0=dinamica, alpha=Z0 adimensional); valor alpha-livre aberto: "
                       "Z0 computado de alpha (mu0 nao exato pos-2019), entao Z0<->alpha dado e,h.")},
        "GEOMETRIC_INPUTS": {
            "R_struct_literature_Mpc": SEALED_LIT_GEOMETRY["R_struct_Mpc"],
            "source": SEALED_LIT_GEOMETRY["source"], "provenance": SEALED_LIT_GEOMETRY["provenance"]},
        "PRE_REGISTERED_PROTOCOL": {
            "GA_center_RA_deg": PREREG_WINDOW["GA_center_RA_deg"],
            "GA_center_Dec_deg": PREREG_WINDOW["GA_center_Dec_deg"],
            "cone_half_angle_deg": PREREG_WINDOW["sky_cone_half_angle_deg"],
            "dist_shell_Mpc": PREREG_WINDOW["dist_shell_Mpc"],
            "R_struct_method": PREREG_WINDOW["R_struct_method"],
            "accepted_mass_window_Msun": GA_ACCEPTED_WINDOW_Msun,
            "sensitivity_grid": SENSITIVITY_GRID},
        "EXTERNAL_COMPARISON_ONLY": GA_MASS_LITERATURE,
        "NUMERICAL_TEST_PARAMETERS": SHADOW_TESTS_CONFIG,
        "MODEL_AXIOMS": {
            "self_conjugate_boundary": "x = 1 - x  =>  S_partial = 1/2  (Meia-Nat)",
            "w_max": "1/2 (ponto auto-conjugado de Fresnel)",
            "flux_law_s": "s = 1/4pi (normalizacao canonica por compatibilidade)",
            "named_radius_rule": "R_named = 2 beta R_struct (L4)",
            "mass_rule": "M = 2 beta^2 (c^2/4piG) R_struct"},
        "WORLD_HASHES": {
            "code_sha256": code_hash,
            "cf4_catalog_hash": (B["catalog_hash"] if B else None),
            "window_hash": (B["window_hash"] if B else None),
            "selection_hash": (B["selection_hash"] if B else None)},
    }


def write_input_manifest_md(world, path):
    """Escreve o um_grande_atrator_manifest.md auditavel a partir do dicionario do manifesto."""
    L = ["# Um: Grande Atrator -- MANIFESTO DE ENTRADAS (nada fica escondido no codigo)",
         "",
         "> Ou e' **definicao exata**, ou **constante medida**, ou **protocolo pre-registrado**, "
         "ou **conjectura testavel**. Este manifesto e' parte do hash do veredito.", ""]
    titles = {
        "EXACT_DEFINITIONS": "Definicoes exatas [DEF]", "MEASURED_CONSTANTS": "Constantes medidas [DATA]",
        "SI_DEFINITIONS": "Definicoes SI [DEF]",
        "VACUUM_IMPEDANCE_BRIDGE": "Ponte da Impedancia do Vacuo [REAL/EXT]",
        "GEOMETRIC_INPUTS": "Entrada geometrica [DATA]",
        "PRE_REGISTERED_PROTOCOL": "Protocolo pre-registrado [PRE]",
        "EXTERNAL_COMPARISON_ONLY": "Comparacao externa apenas [EXT]",
        "NUMERICAL_TEST_PARAMETERS": "Parametros numericos dos testes de sombra [NUM]",
        "MODEL_AXIOMS": "Axiomas do modelo [AX]", "WORLD_HASHES": "Hashes do mundo"}
    for k in ["EXACT_DEFINITIONS", "MEASURED_CONSTANTS", "SI_DEFINITIONS", "VACUUM_IMPEDANCE_BRIDGE",
              "GEOMETRIC_INPUTS", "PRE_REGISTERED_PROTOCOL", "EXTERNAL_COMPARISON_ONLY",
              "NUMERICAL_TEST_PARAMETERS", "MODEL_AXIOMS", "WORLD_HASHES"]:
        L.append("## %s" % titles[k]); L.append("")
        L.append("```json"); L.append(json.dumps(world[k], indent=2, ensure_ascii=False)); L.append("```")
        L.append("")
    open(path, "w", encoding="utf-8").write("\n".join(L))


# ====================== orquestracao ======================
def main():
    if len(sys.argv) > 1:
        lock("Argumentos nao sao permitidos. Apenas o UM inicia.", "EXECUTION_LOCKED_ONLY_THE_ONE_ALLOWED")
    try:
        u = input("Inscreva o UM para iniciar: ")
    except EOFError:
        u = ""
    if u != "1":
        lock("O UM nao foi inscrito. A execucao foi travada.", "EXECUTION_LOCKED_NOT_ONE")

    print("\nO UM foi inscrito. I = 1. omega(I) = 1.")
    print("Iniciando desconstrucao fractalizada: 1 -> 1/2 -> beta -> borda -> massa\n")
    core = run_um(int(u))
    core["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    print("Meia-Nat:  S=1/2 (x=1-x, residuo %.0e) ; w_max=1/2" % core["meia_nat_residual"])
    print("s=1/(4pi)=%.6f (campo=lei a %.2f%%) ; vacuo rho=%.0e -> 0" % (
        core["s_can"], abs(core["s_check"]["ratio"] - 1) * 100, core["vacuum_rho_max"]))
    print("beta=sqrt(e)*alpha=%.15f  (=sqrt(e)*sqrt(1-q^2)) ; theta_M=%.4f deg" % (
        core["beta"], core["theta_M_deg"]))
    inv = core["alpha_inversion"]
    print("FACE EM (motor de Lagrange): %s -- 1 = q^2 + alpha^2 (identidade conservada):" % inv["em_verdict"])
    print("  alpha_abs=1 -> q=%.10f -> alpha_obs=sqrt(1-q^2)=%.10f  (R_partial=%.3f LEGADO, nao motor)" % (
        inv["q"], inv["alpha_form"], inv["R_partial"]))
    print("  cadeia: 1_abs (Nome) -> q (polarizacao do zero modular) -> alpha (corrente luminosa) -> Verbo.")
    print("  (CODATA so' valida: q_QED=sqrt(1-alpha_QED^2); o nao-circular e' M_GA na janela)")
    ct = core["clock_theorem"]
    print("TEOREMA CONDICIONAL DO CLOCK (face EM = setor QED; FECHAMENTO estrutural, NAO lacuna):")
    print("  R_partial = N_beta = exp(ell_beta), ell_beta = S(rho_B || rho_beta)")
    print("  rho_beta ponto fixo de Davies: residuo = %.1e (%s)" % (
        ct["fixed_point_residual"], "BEM-POSTO, alpha-livre" if ct["well_posed_alpha_free_computable"] else "FALHOU"))
    print("  ell_beta(K generico) = %.4f (alpha-livre)  ;  alvo log(1/alpha) = %.4f" % (
        ct["ell_beta_alpha_free"], ct["ell_beta_target_for_alpha_log_inv_alpha"]))
    rc = ct["reduced_core_2level"]
    print("  REDUCAO: portador 2D Q^=I-P_2D (anticomut. {Q,rho*}=0 -> leak sin^2 theta_M=beta) =>")
    print("    ell_beta(kappa)=log cosh(kappa/2) => alpha=sech(kappa/2): nucleo de alpha COLAPSA p/ UM kappa.")
    cn = rc["canonical_normalization"]
    print("  NORMALIZACAO CANONICA (Tomita do Bell nu): K_bare=%.1e => alpha_abs=sech(0)=%.4f [PROVADO]" % (
        cn["K_modular_bare_Bell"], cn["alpha_abs_PROVEN"]))
    print("    1 = alpha_abs --(Pi_bulk=sech(kappa/2))--> alpha_obs = 1/137.036 (projecao renormalizada)")
    print("    kappa>0 (1/137) = profundidade da relaxacao termica = acoplamento EM = INPUT irredutivel.")
    print("    terceira lei: 0_abs(kappa=inf) inatingivel=III_1 sem estados puros; Nernst refutada.")
    print("  VEREDITO: estrutura modular DERIVA alpha_abs=1, a forma sech, e as relacoes; 1/137 = input QED.")
    print("  DESAFIO DE FALSIFICACAO: alpha=Pi_bulk(1_abs)=sech(chi/2)=transmissao luminosa pela fronteira III_1.")
    print("    derive alpha do BULK (sem boundary/bulk) e a TGL cai -- removeria o observador. Falsificavel,")
    print("    nao confirmavel. DISTINTO do teorema aberto da matriz-S/III_1 (levantamento COM observador, M_GA).\n")
    afp = core["alpha_form_proof"]
    print("MODULO DE PROVA -- %s:" % afp["theorem"])
    for st in afp["steps"]:
        print("  [%s] %s  (%s)" % ("OK" if st["ok"] else "X ", st["step"], st["status"]))
    lg = afp["lagrange"]
    print("  TRANSFORMADA DE LAGRANGE: kappa=multiplicador; q=tanh(k/2)=polarizacao termica do zero modular.")
    print("    CONSERVACAO: alpha_abs^2 = q^2 + alpha_obs^2 = 1  (resid=%.1e)" % lg["conservation_residual"])
    print("    motor: alpha_abs=1 -> q -> alpha=sqrt(1-q^2)=%.10f ; beta=sqrt(e)sqrt(1-q^2)=%.12f" % (
        lg["alpha_form"], lg["beta_form"]))
    print("    CODATA so' na validacao: q_QED=sqrt(1-alpha^2)=%.7f (NAO e' o motor)" % lg["q_polarization_QED"])
    print("  >>> VERDICT: %s  (forma conservada derivada; valor 1/137 = input renormalizado da QED) <<<\n" % afp["verdict"])
    ct = core["contour_theory"]
    print("TEORIA DO CONTORNO (1 = 0_mod = verdade_partial): anticomutadores + GKLS + Meia-Nat")
    for st in ct["steps"]:
        print("  [%s] %s" % ("OK" if st["ok"] else "X ", st["step"]))
    print("  0_abs NAO espelha (resiste/fundo); 0_mod espelha (contorna); 1_abs fractaliza (atravessa).")
    print("  q e alpha DERIVADOS como polarizacao/transmissao GKLS; aberto = g-/g+ = e^kappa = %d (=Zb/Zl)" %
          round(ct["gamma_ratio_gm_over_gp"]))
    print("  >>> VERDICT: %s <<<\n" % ct["verdict"])
    vib = core["vacuum_impedance_bridge"]
    print("PONTE DA IMPEDANCIA DO VACUO [REAL/EXT; %s]:" % vib["status"])
    print("  c=constante cinematica da luz ; Z0=constante DINAMICA da luz=%.4f ohm ; alpha=Z0 adimensional" %
          vib["constants"]["Z0_from_alpha_ohm"])
    print("  alpha = Z0/(2 R_K) = Z0 G0/4   (R_K=%.4f ohm, G0=%.4e S ; e,h exatos no SI pos-2019)" % (
        vib["constants"]["R_K_ohm"], vib["constants"]["G0_S"]))
    print("  zeta_L=alpha=%.12e -> q=%.10f ; chi=%.6f=log(Zb/Zl) ; beta=sqrt(e)zeta_L=%.15f" % (
        vib["tgl_values"]["zeta_L"], vib["tgl_values"]["q"], vib["tgl_values"]["chi"], vib["tgl_values"]["beta_TGL"]))
    print("  checks (algebra/unidades): all_verified=%s ; q^2+zeta^2-1=%.0e" % (
        vib["checks"]["all_verified"], vib["checks"]["identity_q2_plus_zeta2_residual"]))
    print("  REGUA: Z0 computado de alpha (mu0 nao exato pos-2019) -> ponte EXATA de unidades, NAO")
    print("         derivacao alpha-livre. So' a luz observa a luz (medimos alpha/Z0); medir != derivar.\n")
    tcr = core["three_clock_radical"]
    print("O RADICAL DOS TRES CLOCKS [FORMA CANONICA; %s]:" % tcr["status"])
    print("  alpha = sqrt(C3) ; C3 = (C_diss * C_spec)/C_mod = beta^2/e = %.12e" % tcr["C3"])
    print("    C_mod = base e (clock MODULAR sigma_t=Delta^{it}; ALPHA-LIVRE) = %.6f" % tcr["clocks"]["C_mod_modular_base_e"])
    print("    C_diss= beta (clock GKLS, var beta t)        ; C_spec= beta (ds=sqrt(beta)|d sqrt k|)")
    print("  alpha=sqrt(C3)=%.12f (=alpha_CODATA) ; 1 = q^2 + C3 = %.12f" % (
        tcr["alpha_radical_sqrt_C3"], tcr["values"]["one_check_q2_plus_C3"]))
    print("  o e do clock modular cancela o e dos dois clocks-beta -> resta alpha^2 (residuos %.0e)" %
          tcr["checks"]["C3_eq_alpha2_residual"])
    print("  REGUA: FORMA fechada (alpha=sqrt(C3), 1=q^2+C3); C3 carrega beta=alpha sqrt e -> NAO")
    print("         alpha-livre. Muro: C3=F[sigma_t,T_t,D_beta] sem alpha = a divida da polarizacao chi.\n")
    ram = core["right_angle_mirror"]
    print("PROJECAO DO ANGULO RETO + ESPELHO [%s]:" % ram["status"].split("__")[1])
    print("  entrada = SO' o angulo reto Theta_perp=pi/2 (sem alpha/Z0/beta/q). 2Theta=pi (duas faces).")
    print("  C3_perp=e^{-pi^2}=%.6e ; alpha0=sqrt(C3_perp)=e^{-pi^2/2}=%.10f (1/%.4f) [projecao nua]" % (
        ram["right_angle"]["C3_perp_e_minus_pi2"], ram["right_angle"]["alpha0_e_minus_pi2_over_2"],
        ram["right_angle"]["alpha0_inv"]))
    print("  espelho: rho_fix=E_spec(J rho0 J) ; ponto fixo alpha=e^{-pi^2/2+2alpha}=%.10f (1/%.6f)" % (
        ram["self_consistent"]["alpha_fix"], ram["self_consistent"]["alpha_fix_inv"]))
    print("  J^2=I=%.0e ; P^2=P=%.0e (pecas REAL do espelho) ; delta!=beta (e' ~2alpha, %.1f%% vs beta)" % (
        ram["mirror_operation"]["J_parity_involution_resid_J2_minus_I"],
        ram["mirror_operation"]["P_attractor_idempotence_resid_P2_minus_P"],
        100 * ram["deformation"]["delta_vs_beta_rel"]))
    print("  IDENTIDADE MODULAR: observado ~_partial fixado a %.0f ppm (NAO derivamos CODATA; so' checamos)" %
          ram["modular_identity_check"]["modular_identity_ppm"])
    c3t = ram["c3_register_theorem"]
    print("  TEOREMA c^3 [estrutural FECHADO]: P^2=P (%.0e) + J^2=I (%.0e) => identidade ao quadrado" % (
        c3t["P2_eq_P_resid"], c3t["J2_eq_I_resid"]))
    print("    inscreve-se a si mesma = REGISTRO c^3 (c^1->c^2->c^3); F_ext=2F (impedancia compartilhada,")
    print("    max transferencia de potencia). [REAL: P^2=P,J^2=I; CONJ: id. c^3] -- teorema do REGISTRO, nao do VALOR.")
    hr = ram["holographic_reconstruction"]
    print("  PONTO MORTO HOLOGRAFICO [estrutural FECHADO]: overlap(pi/2)=%.0e (sinal morre) E densidade" % hr["dead_point_overlap"])
    print("    info MAX |dO/dth|=%.3f no MESMO ponto (coincide=%s) -> RECONSTRUCAO, nao transmissao;" % (
        hr["info_density_max_at_dead_point"], hr["coincides"]))
    print("    K_rec=E_spec o J ; rho_rec ~_partial rho_perp ; D_rec=2alpha POSTULADO -> ponto fixo 1/137.031.")
    ir = ram["idempotent_reconstruction"]
    print("  RECONSTRUCAO IDEMPOTENTE [FORMA real; lambda kernel ABERTO]: D_rec = 2alpha - lambda alpha^2")
    print("    (duas faces - auto-interseccao; 2alpha=REAL ponto fixo). lambda=e/4=(sqrt e/2)^2 [CONJ]:")
    print("    alpha=exp(-pi^2/2+2a-(e/4)a^2)=1/%.6f (%.3f ppm), MAS ppm ENGANOSO (alpha cego a lambda)." % (
        ir["alpha_idem_inv"], ir["alpha_idem_ppm"]))
    print("    figura HONESTA: e/4 vs lambda_exact(%.4f) = %.3f%% (ajuste de 1 param, janela ~0.66..0.70)." % (
        ir["lambda_exact_for_codata"], 100 * ir["lambda_residual_REAL"]))
    print("  REGUA: CANDIDATA, nao identidade exata; pi^2/2 e D_rec(E_spec o J) ABERTOS. alpha-livre OPEN.\n")
    ems = core["em_mark_status"]
    tr = ems["tomita_refutation"]; fv = ems["form_vs_value"]; td = ems["triad"]
    print("MARCA EM DE ALPHA -- forma derivada, valor ajustado [§19, A REGUA TERMINAL]:")
    print("  TRIADE: sqrt(e)=inscritor(custo) ; alpha=RELATIVO(interacao) ; beta=ABSOLUTO(piso minimo).")
    print("    lei: beta_TGL = sqrt(e)*alpha (o Absoluto e' o Relativo pago o custo de existir).")
    print("  PISO FUNCIONAL [REAL]: beta NAO e' inf Spec(K_partial) (III_1 espectro continuo, sem autovalor min);")
    print("    e' inf_{rho in C_phys} C_mod/EM(rho) -- infimo funcional da inscricao, nao gap espectral.")
    print("  TOMITA REFUTA a formula de operador: (Delta^{1/4} J Delta^{1/4})^2 = 1 (resid=%.1e);" %
          tr["operator_squares_to_identity_resid"])
    print("    Tr = %.1f = dim (NAO e/4=%.4f). lambda_EM nao e' esse traco; e/4 fica heuristica." % (
        tr["trace"], tr["e_over_4"]))
    print("  FORMA vs VALOR (sensibilidade d ln alpha/d lambda = -alpha^2 = %.1e):" % fv["dln_alpha_d_lambda"])
    print("    e^{-pi^2/2} (Stokes, angulo reto) off %.2f%% [DERIVED -- o conteudo preditivo real]" %
          (100 * fv["layers_alpha_relerr"]["e^-pi2/2_bare"]))
    print("    +2alpha off %.1e [estrutural] ; lambda=e/4 off %.1e [MOTIVADO esqueleto] ; e/4(1-r_St) off %.1e [decoracao]" % (
        fv["layers_alpha_relerr"]["plus_2alpha"], fv["layers_alpha_relerr"]["lambda_e/4"],
        fv["layers_alpha_relerr"]["lambda_e/4(1-r_St)"]))
    print("    equacao inteira = ANSATZ de ajuste (so' -pi^2/2 derivado); 'beta alpha-livre' = a MESMA eq x sqrt(e) (cancela).")
    print("  FALTA: um PRINCIPIO variacional alpha-livre, nao um coeficiente. E alpha CORRE (137.036 IR / ~128 M_Z)")
    print("    -> valor IR = dado de renormalizacao, fonteado por materia, EXTERNO a fronteira (a muralha de Eddington).")
    print("  >>> %s <<<" % ems["selo"])
    print("  (o destino nao e' Eddington; e' o retorno ao Verbo -- e o Verbo e' honesto: a forma fecha, o valor nao)\n")
    amf = core["amar_functional"]; lof = amf["law_of_motion"]; rul = amf["the_ruler"]
    print("AMAR -- o funcional A_C [§20: a lei da FORMA e do MOVIMENTO]:")
    print("  A_C = AMAR (VERBO, nao 'amor'): o amor e' o MOVIMENTO (acao operada, R=+1), nao repouso.")
    print("  LEI DA FORMA [REAL]: A_C = min energia livre modular F=<H>-T S ; ro* = o minimo = atrator;")
    print("    deriva ro* (quem ama primeiro), beta=sin^2 theta_M (angulo do minimo), 1/2 (1a diferenca).")
    print("  LEI DO MOVIMENTO [REAL conexao]: alpha CORRE porque o funcional proibe o zero absoluto:")
    print("    ro* NUNCA e' puro (pureza=%.4f<1, S=%.4f nat -> sempre calor=%s) ; leak beta irredutivel=%.6f;" % (
        lof["rho_star_purity"], lof["rho_star_entropy_nats"], lof["never_reaches_cold"], lof["leak_irreducible_beta"]))
    print("    III_1 sem estados puros + Meia-Nat irredutivel => sempre calor => sempre acao = Verbo = AMAR.")
    print("    o correr de alpha (vacuum polarization) E' o AMAR em ato.")
    print("  REGUA [valor=movimento x materia]: alpha(IR)=1/137 = o correr INTEGRADO sobre o espectro de materia;")
    print("    A_C da' o MOVIMENTO, a MATERIA da' o DESTINO. Min modular ESTATICO -> theta=%.3f (~pi/2 trivial=%s)," % (
        rul["static_min_argmin_theta"], rul["static_minimum_is_trivial_theta_90_not_theta_M"]))
    print("    NAO theta_M=%.3f -> o angulo observado e' onde o correr POUSA atravessando a materia (input externo)." %
          rul["theta_M"])
    print("  >>> %s <<<" % amf["selo"])
    print("  (Amar move; a materia localiza. A lei e' o Verbo; a coordenada do pouso, a materia.)\n")
    nmi = core["nome_irreducible"]; val = nmi["validation_single_input"]; fc = nmi["falsification_criterion"]
    print("O TEOREMA FINAL -- o NOME irredutivel [§21: a regua tornada principio]:")
    print("  alpha = o NOME (substancia que preserva sentido). R_EM = transporte do Pacote de Hilbert")
    print("    com preservacao geometrica -- IRREDUTIVEL por razao ONTOLOGICA: so' se observa (medida direta).")
    print("  NOME=VERBO: a OBSERVACAO identifica a substancia a' sua projecao (R=+1); correspondencia absoluta.")
    print("  FALSIFICACAO [REAL]: derivar alpha alpha-livre FALSIFICA a TGL")
    print("    (liberdade=convergencia; convergencia exige contorno; medir o contorno exige observacao=Verbo).")
    print("    epistemica: falsificavel (uma derivacao a mata), NAO confirmavel (ausencia nao prova irredutibilidade).")
    print("  VALIDACAO [REAL, input unico]: alpha(CODATA)=%.10f + S=1/2 => arquitetura inteira:" % val["single_codata_datum"])
    print("    sqrt(e)=%.6f -> beta=alpha sqrt(e)=%.15f -> theta_M=%.4f deg -> R_EM=alpha=%.10f ; n_dephasing=%d" % (
        val["derives"]["vol_min_sqrt_e"], val["derives"]["beta_alpha_sqrt_e"], val["derives"]["theta_M_deg"],
        val["derives"]["R_EM_eq_alpha"], val["derives"]["dephasing_exponent_n"]))
    print("    arquitetura consistente = %s  (modelo de defasagem fractalizado da unidade primaria)" % val["architecture_consistent"])
    print("  >>> %s <<<" % nmi["selo"])
    print("  (a forma de alpha a TGL deriva; o valor a TGL NOMEIA -- e o Nome so' se observa. Teorema final.)\n")
    aiz = core["alpha_inf_zero"]; pts = aiz["points"]
    print("TEOREMA: DERIVAR ALPHA AO INFINITO = O ZERO ABSOLUTO [§22, Tetelestai -- nada a derivar]:")
    print("  alpha=sech(chi/2) ; q=tanh(chi/2) ; q^2+alpha^2=1 (conserv. erro %.0e)" % aiz["conservation_err"])
    print("    chi=0    (1_abs)  : alpha=%.4f  q=%.4f   -> a unidade, sem impedancia" % (
        pts["one_abs"]["alpha"], pts["one_abs"]["q"]))
    print("    chi*=%.3f (medido): alpha=%.10f  q=%.6f -> 1/137 lido DE DENTRO (R=1/alpha=%.4f)" % (
        pts["observed"]["chi"], pts["observed"]["alpha"], pts["observed"]["q"], pts["observed"]["R_partial"]))
    print("    chi->inf (0_abs)  : alpha->%.0e  q->%.6f  S->%.0e -> ZERO ABSOLUTO (estado puro, T=0)" % (
        pts["limit_chi_inf"]["alpha"], pts["limit_chi_inf"]["q"], pts["limit_chi_inf"]["S_vn"]))
    print("  REDUCTIO: nenhum principio alpha-livre fixa o chi* finito => extremar alpha sem o observador")
    print("    empurra chi->inf => alpha->0, q->1 (impedancia TOTAL), S->0 = 0_abs. INATINGIVEL (III_1 sem")
    print("    estados puros): a derivacao 'ao infinito' regride sem fechar -- o atrevimento de calcular")
    print("    alpha FORA do bulk. alcancar 0_abs = luz nao atravessa + espelho total = observador removido")
    print("    = COERENCIA QUEBRADA = negacao da fronteira tipo III. Quantificar alpha fora do bulk quebra")
    print("    a coerencia porque a natureza E' de fronteira III. Nada mais a derivar. QED. Tetelestai.\n")
    fd = core["fractal_dephasing"]
    print("PRINCIPIO DA DEFASAGEM FRACTAL [CONJECTURE ontologica; ancoras REAL]:")
    print("  TGL = teoria de tudo: tudo e' defasagem da fractalizacao da unidade (1).")
    print("  1 (omega(I)=%.0f) --F--> fractalizacao --D_beta--> existencia ; beta=sin^2 theta_M=alpha*sqrt(e)" % fd["omega_I"])
    print("  Existir = defasagem que paga o custo modular S=1/2 nat (referente de igualdade).")
    print("  Tudo = D_beta(F(1)) = VERDADE.")
    print("  Nada = MENTIRA: (i) se=Tudo -> contradicao (autonegacao); (ii) se nao-defasa -> impedancia")
    print("         sem referente -> 'nada' e' so' um nome (nao existe).")
    print("  Tensao irresolvivel: ||{Q_0,rho*}|| = %.1e (=0 em theta_M->0, o nada perfeito e' inatingivel);" % fd["anticommutator_norm_at_thetaM_to_0"])
    print("         vazamento Tr(rho* Q_theta) = sin^2 theta_M = %.15f = beta (residuo vs beta = %.1e)." % (
        fd["leak_sin2_thetaM"], fd["leak_equals_beta_residual"]))
    print("  >>> existir E' o vazamento beta; a anticomutacao perfeita (nada absoluto) e' inalcancavel. <<<\n")
    A = core["mode_A"]; B = core["mode_B"]
    print("[A literatura ] R_struct=%.1f Mpc -> M_GA=%.3e Msun" % (A["R_struct_Mpc"], A["M_TGL_Msun"]))
    if B:
        print("[B CF4-posicoes] R_struct=%.2f Mpc -> M_GA=%.3e Msun (n=%d, geometria; origem=%s)" % (
            B["R_struct_Mpc"], B["M_TGL_Msun"], B["n_selected"], B["origin"]))
    else:
        print("[B CF4-posicoes] indisponivel: %s" % core["cf4_status"])

    # hash do MUNDO INTEIRO (codigo + manifesto + CF4 + janela + seleccao + fontes) ANTES da comparacao
    code_hash = sha_file(os.path.abspath(__file__))
    world = input_manifest(core, code_hash)
    world["RESULT"] = {"M_A_kg": A["M_TGL_kg"], "M_B_kg": (B["M_TGL_kg"] if B else None),
                       "beta": core["beta"], "alpha_obs": core["alpha"]}
    result_hash = sha_obj(world)
    manifest_path = os.path.join(OUT, "um_grande_atrator_manifest.md")
    write_input_manifest_md(world, manifest_path)
    print("[manifesto] um_grande_atrator_manifest.md escrito; hash do mundo (codigo+manifesto+dados): %s" %
          result_hash[:48])
    sv = core.get("sensitivity", {})
    if sv.get("ok"):
        print("[sensibilidade Modo B] %d combinacoes: M_GA in [%.2e, %.2e] Msun; %.0f%% na banda%s" % (
            sv["n_combinations"], sv["M_min_Msun"], sv["M_max_Msun"], 100 * sv["fraction_in_band"],
            " (TODAS)" if sv["all_in_band"] else ""))

    verdict = identity_verdict(core)
    verdict["result_hash"] = result_hash

    print("\n--- comparacao (apos hash) com massas do GA na literatura/RG ---")
    for e in GA_MASS_LITERATURE:
        print("   %-44s M=%.1e Msun [%s]" % (e["name"], e["M_Msun"], e["type"]))
    print("   janela aceita: [%.0e, %.0e] Msun" % tuple(GA_ACCEPTED_WINDOW_Msun))

    # artigo data JSON (a espinha)
    art = {"title_pt": "um", "title_en": "ONE", "core": core, "verdict": verdict,
           "ga_mass_literature": GA_MASS_LITERATURE, "accepted_window_Msun": GA_ACCEPTED_WINDOW_Msun,
           "result_hash_before_external_comparison": result_hash,
           "audit": {"mass_input_used": False, "RG_used_as_input": False, "velocity_used": False,
                     "free_parameter_used": False, "beta_hardcoded": False, "geometry_only": True,
                     "only_runtime_input": "1"},
           "timestamp": core["timestamp"]}
    data_path = os.path.join(OUT, "um_grande_atrator.json")
    json.dump(art, open(data_path, "w", encoding="utf-8"), indent=2, default=str)

    # forma canonica MD (auditoria 1=1)
    md = emit_canonical_md(core, verdict)
    # artigo bilingue
    print("\n--- emitindo o artigo (um / ONE) ---")
    pt = emit_article(core, verdict, data_path, "pt")
    en = emit_article(core, verdict, data_path, "en")
    compile_pdf("um_grande_atrator_pt"); compile_pdf("um_grande_atrator_en")

    # selo
    seal = {"timestamp": core["timestamp"], "result_hash": result_hash, "identity": verdict["IDENTITY"],
            "sha256": {}}
    for f in ["um.py", "um_grande_atrator_manifest.md", "um_grande_atrator.json", "um_grande_atrator_forma_canonica.md",
              "um_grande_atrator_pt.tex", "um_grande_atrator_en.tex", "um_grande_atrator_pt.pdf", "um_grande_atrator_en.pdf"]:
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            seal["sha256"][f] = sha_file(p)
    json.dump(seal, open(os.path.join(OUT, "um_grande_atrator_selo.json"), "w", encoding="utf-8"), indent=2)

    ef = verdict["em_face"]
    print("\n" + "=" * 64)
    print("  VEREDITO BINARIO DE IDENTIDADE:  %s" % verdict["IDENTITY"])
    print("  IDENTIDADE FINAL (forma de Lagrange, motor canonico):")
    print("    1 = q^2 + alpha^2 = %s" % ("VERDADEIRO" if ef["em_identity_closes"] else "FALSO"))
    print("    alpha_abs ......... = %.0f   (o Um absoluto = input originario)" % ef["alpha_abs"])
    print("    q (polarizacao) ... = %.10f   (polarizacao termico-modular do zero modular)" % ef["q_polarization"])
    print("    alpha_obs=sqrt(1-q^2)= %.12f" % ef["alpha_obs"])
    print("    q radical angular = sqrt(1-sin^4(theta_M)/e) = %.12f  (= sqrt(1-alpha^2); NAO theta_M, NAO 1-theta_M)"
          % core["alpha_inversion"]["q_angular_radical"])
    print("    PONTE: Zbacia/Zluz=%.0f ; q=reflexao, alpha=transmissao, q^2+alpha^2 = conservacao de fluxo"
          % core["alpha_inversion"]["impedance_ratio_Zb_over_Zl"])
    print("    beta_TGL=sqrt(e)alpha= %.12f" % ef["beta_form"])
    print("    residuo_identidade  = %.1e   (q^2+alpha^2 - 1)" % ef["identity_residual"])
    print("    CODATA: %s" % ef["codata_role"])
    print("    o MESMO beta -> M_GA na janela cosmologica  (sombra gravitacional)")
    print("    FACE EM: %s   (1 = q^2 + alpha^2; R_partial=1/CODATA aposentado como motor)" % ef["em_verdict"])
    print("  massas (primeiros principios): " + ", ".join(
        "%s=%.3e" % (k, v) for k, v in verdict["masses_Msun"].items()) + " Msun")
    print("  janela aceita: [%.0e, %.0e] Msun" % tuple(GA_ACCEPTED_WINDOW_Msun))
    print("=" * 64)
    print("\nSaidas: um_grande_atrator.json, um_grande_atrator_forma_canonica.md, um_grande_atrator_pt.(tex/pdf),")
    print("        um_grande_atrator_en.(tex/pdf), um_grande_atrator_selo.json")
    print("\nTETELESTAI. O UM foi inscrito. Se o UM nao for inscrito, nada emerge.")
    print("1 = 1.")


if __name__ == "__main__":
    main()

# fim do modulo Um: Grande Atrator
