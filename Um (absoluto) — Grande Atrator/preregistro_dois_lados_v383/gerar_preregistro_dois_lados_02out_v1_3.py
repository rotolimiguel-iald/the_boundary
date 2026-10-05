# -*- coding: utf-8 -*-
"""PRÉ-REGISTRO DOS DOIS LADOS, V1.3 (02/10/2026) — SUPERSEDE o V1.2 (mapa seq 279), o V1.1 (seq 276) e o V1 (seq 275), ANTES de qualquer estimador ou dado
NOVO, pelos achados CONFERIDOS no core da 3ª aferição independente da v382 (nenhum muda desfecho nem contagem; todos são de transcrição e de redação):
  M-b (R4: o conjunto CONGELADO do V11 não tem FALSIFIED, e a MESMA fonte alimenta o degrau experimental do gate); M-c (R5g: o rótulo emitido é INCONCLUSIVE pela
  cláusula 2; a cláusula NÃO aplicada é a da emenda V5 — «poder de R3 < 5σ → NOT_FALSIFIED_UNDERPOWERED» —, não a 3; e as cláusulas das emendas V4/V5 e a do
  «principal = todos» passam a ser transcritas, lidas dos arquivos); M-d (F1: a lei é a D1b, não (1+z)^β; o poder prévio 2,33σ é dito nas duas réguas); M-e (a lei
  do lado transmitido nos cronômetros, anunciada e não escrita, fica PENDENTE); BAIXOS (R1 sem «com os controles»; F3a com o poder ao lado do déficit; «FALSIFIED
  possível» só onde a lei o admite, no campo `canal_futuro_com_falsified`; o marco «desde o V1»; «subamostra cega de O4, não independente»; os rótulos V3/V4/V5;
  o registro do F2 sem GW250114; o F4 com a projeção 2031 e a escala [NORM]; o R5e com o veredito V3 do mesmo par; os números lidos do core e do um.py).
O V1, o V1.1 e o V1.2 NÃO são reescritos: ficam como registro; a correção vai AO LADO. Grava JSON + MD com sha256, publica o MD na Bancada e inscreve a ERRATA no
mapa (corrige 279). β nunca literal (lido do um.py canônico via bancada.fontes)."""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"C:\IALD\Bancada_Um")
sys.path.insert(0, r"C:\IALD\MAPA_DE_ROTAS")
from bancada import fontes  # noqa: E402
import rotas  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
BU = r"C:\IALD\Bancada_Um"
NOS = r"C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós"
F6 = r"C:\IALD\Bancada_Um\investigacao\fase6_eco_radical_01out"
ID = "PREREG_DOIS_LADOS_DELTA_K_20261002_V1_3"
SAIDA_JSON = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_3.json")
SAIDA_MD = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_3.md")
ANTERIORES = {"v1": "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002.json", "v1_1": "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_1.json", "v1_2": "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_2.json"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def get(o, path):
    for p in path:
        if isinstance(o, dict):
            o = o.get(p)
        elif isinstance(o, list) and isinstance(p, int) and p < len(o):
            o = o[p]
        else:
            return None
    return o


alpha, prov_alpha = fontes.alpha_codata_2018()
beta = alpha * math.sqrt(math.e)
theta_M = math.asin(math.sqrt(beta))
selo = json.load(open(os.path.join(NOS, "um_absoluto_selo.json"), encoding="utf-8"))
core = json.load(open(os.path.join(NOS, "um_absoluto.json"), encoding="utf-8"))["core"]
um_txt = open(os.path.join(NOS, "um.py"), encoding="utf-8").read()
um_sha = sha(os.path.join(NOS, "um.py"))
assert selo.get("um_version") == "v381", "o canônico não é a v381 selada: %s" % selo.get("um_version")
V = lambda k: str((core.get(k) or {}).get("verdict") or "")
ant_sha = {k: sha(os.path.join(AQUI, f)) for k, f in ANTERIORES.items()}
codigo = {p: sha(os.path.join(BU, "bancada", p)) for p in ("motor.py", "fundo_gpu.py", "camb_tabela.py", "evidencia.py", "amostrador.py", "regua.py", "fontes.py", "gw.py", "gw_stack.py")
          if os.path.exists(os.path.join(BU, "bancada", p))}
try:
    _, prov_cc = fontes.cronometros_moresco2022()
    _, prov_cov = fontes.cronometros_covariancia_moresco2020()
    _, prov_dr2 = fontes.desi_dr2()
except Exception as e:
    raise SystemExit("fonte recusada: %r" % (e,))
# as constantes do um.py canônico (lidas, nunca digitadas) para o poder de R2
_c = lambda nome: float(re.search(r"^%s = ([0-9.eE+-]+)" % nome, um_txt, re.M).group(1))
C_LIGHT, G_NEWTON, H_PLANCK = _c("C_LIGHT"), _c("G_NEWTON"), _c("H_PLANCK_EXACT")
T_PLANCK = math.sqrt(H_PLANCK / (2.0 * math.pi) * G_NEWTON / C_LIGHT ** 5)
GAMMA_100 = 0.5 * beta * T_PLANCK * (2.0 * math.pi * 100.0) ** 2
# as regras das emendas V4/V5 da Fase 6, LIDAS dos arquivos (com sha256)
EM = {}
for nome in ("PREREGISTRO_FASE6_ECO_RADICAL_20261001.json", "EMENDA_V4_FASE6_20261001.json", "EMENDA_V5_FASE6_20261001.json"):
    p = os.path.join(F6, nome); EM[nome] = dict(sha256=sha(p), regra_de_veredito=json.load(open(p, encoding="utf-8")).get("regra_de_veredito"))
R4V = EM["EMENDA_V4_FASE6_20261001.json"]["regra_de_veredito"]; R5V = EM["EMENDA_V5_FASE6_20261001.json"]["regra_de_veredito"]
assert any(str(x).startswith("V4:") for x in R4V) and any(str(x).startswith("V5:") for x in R5V)
CL_V4 = next(str(x) for x in R4V if str(x).startswith("V4:")); CL_V5 = next(str(x) for x in R5V if str(x).startswith("V5:"))
CL_PRINCIPAL = next(str(x) for x in R4V if "PRINCIPAL" in str(x))

VERB = {
    "chave_20261002": "O último elo incide no hamiltoniano oculto. O custo está no reflexo, o pagamento está na face no rosto, no nome. E com isso você deveria ser capaz de responder tudo que falta. Mas se não conseguir eu vou um a um",
    "otica_20261002": "Concordo com tudo, exceto com a sua visão semiótica, a TGL não é semi ela é ótica, ela lê tanto a face como o custo, a TGL permite ler os dois lados",
    "prossiga_20261002": "isso mesmo, prossiga",
    "lambda_20260915": "lambda é a cauda, é o rastro do dragão, de satanás, onde a realidade emergente se contorna; ele é o espectro de gradiente negativo da presença que não age, não é permanência, é resistência",
}
B = "bancada_fases_5_6_7_v380"
RD = "ringdown_dephasing_result_v2"
LADOS = {
    "reflexo": "o lado do CUSTO: a leitura que passa pelo reflexo (por distância, pela propagação, pela lente, pela cópia atrasada); o que ela lê é o custo β = |R|² ou o seu rastro",
    "face": ("o lado do PAGAMENTO: a leitura que NÃO passa pelo reflexo (sem distância, ou direta, ou o Nome entregue); o que ela lê pode ser o peso 1 − β = |T|², o custo local Γ ∝ β, "
             "ou a RG; R6 é FACE POR CONVENÇÃO (a convenção não canônica que põe o custo no fundo), não pela leitura (BAO e SH0ES são distância) — o critério é dito cobrança a cobrança"),
}
# os números lidos do core (nada digitado)
H0 = get(core, [B, "fase5", "validacao", "H0"]); SH0 = get(core, [B, "fase5", "validacao", "sigma_H0"])
Z_AE = get(core, ["d1_camb_v3_real_v366", "z_alpha_sqrt_e"]); DCHI2 = get(core, ["d1_camb_v3_real_v366", "delta_chi2"]); BELOW = get(core, ["d1_camb_v3_real_v366", "autocorr", "params_below_50tau"])
NPAR = len(get(core, ["d1_camb_v3_real_v366", "autocorr", "tau"]) or [])
NEXC, NSER = get(core, ["clock_test_result_v369", "P2", "n_excluded"]), get(core, ["clock_test_result_v369", "P2", "n_series"]); BAND = get(core, ["clock_test_result_v369", "P2", "band_Hz"])
M2P = get(core, ["neutrino_m2", "values", "m2_pred_meV"]); JUNO = get(core, ["neutrino_m2", "values", "escada_datada", 1]); KILL = get(core, ["neutrino_m2", "frozen", "kill_rule"])
PROJ31 = get(core, ["neutrino_m2", "values", "projecao_2031_sigma"]); ESCALA = get(core, ["neutrino_m2", "frozen", "escala_dimensional"])
RD_REASON = (get(core, [RD, "reasons"]) or [""])[0]; RD_BIAS = get(core, [RD, "stacks", "3.0", "mean_bias_rel"]); RD_DA = get(core, [RD, "stacks", "3.0", "delta_pred_A"])
RD_PB = get(core, [RD, "stacks", "3.0", "power_B"]); RD_DT = get(core, [RD, "stacks", "3.0", "delta_tau"]); RD_SG = get(core, [RD, "stacks", "3.0", "sigma"]); RD_N = get(core, [RD, "stacks", "3.0", "n_used"])
C6 = get(core, ["ringdown_scope_v376", "c6_status"])
m = re.search(r"emenda \(([0-9a-f]{16})\)", str(get(core, [RD, "statuses", "leitura"]) or "")); RD_EMENDA = m.group(1) if m else None
COMA_REF, COMA_SIG, COMA_DATA = get(core, ["coma_reveal_state_v376", "reference_Mpc"]), get(core, ["coma_reveal_state_v376", "reference_sigma_Mpc"]), get(core, ["coma_reveal_state_v376", "reveal_date"])
GW_OK, GW_JOBS = get(core, [B, "gw_bank", "n_ok"]), get(core, [B, "gw_bank", "n_jobs"])
F1_POD = get(core, [B, "fase7", "poder_previo", "moresco2020"]); P1_Z = get(core, ["clock_test_result_v369", "P1", "platforms", 0, "Z_expected"]); P1_ID = get(core, ["clock_test_result_v369", "P1", "platforms", 0, "id"])
V3_MAY_R2 = get(core, [B, "fase6", "veredito_principal", "MAY|R2_peso2_2theta"])
assert all(isinstance(x, (int, float)) for x in (H0, SH0, Z_AE, DCHI2, BELOW, NEXC, NSER, M2P, RD_BIAS, RD_DA, RD_PB, RD_DT, RD_SG, RD_N, COMA_REF, COMA_SIG, GW_OK, GW_JOBS, F1_POD, P1_Z, PROJ31)) and NPAR > 0
assert RD_EMENDA and RD_REASON and str(JUNO.get("fonte", "")).startswith("JUNO") and "DUAS" in str(KILL) and isinstance(BAND, list) and "INCONCLUSIVE" in str(V3_MAY_R2)
assert "H0_SECTORS_D1A_VS_D1V3_INCOMPATIBLE_OPEN" in V("coma_reveal_state_v376") and "FLAG_NOT_WRITTEN_BY_PIPELINE_STATED" in V(B) and "V5_CLAUSE_UNDERPOWERED_NOT_APPLIED_STATED" in V(B)
V11_SET = get(core, ["void_floor_v11", "frozen", "verdicts"]) or []
assert V11_SET and not any(("FALSIFIED" in v and "NOT_FALSIFIED" not in v) for v in V11_SET), "o conjunto congelado do V11 mudou"

C = []


def cob(id_, lado, criterio, registro, lei, leitura, chave, token, como, numero, desfecho, na_regra, ja_lido, qualificador, cego_estimador, cego_dado, futuro, canal_futuro, caminho_veredito=None):
    C.append(dict(id=id_, lado=lado, criterio_de_lado=criterio, registro=registro, lei=lei, leitura=leitura, chave_core=chave, token=token, como=como, numero=numero,
                  caminho_veredito=caminho_veredito, desfecho_lido=desfecho, na_regra=na_regra, ja_lido=ja_lido, qualificador=qualificador,
                  cego={"estimador": cego_estimador, "dado": cego_dado}, futuro=futuro, canal_futuro_com_falsified=canal_futuro))


cob("R1", "reflexo", "leitura por distância (a escada)", "escada de distâncias (SN Ia + cefeidas; SH0ES completo)", "K = E(z*)^{2β/3} sobre o fundo nu; nó fixo em z* (convenção do artigo; R1 ratificada)",
    "δ̂ = β̂ − β_TGL no expoente da leitura por distância", B, "SH0ES_IN_THE_BENCH_H0_LADDER_73P53_PM_1P02", "NUMBER_READ",
    dict(caminho=[B, "fase5", "P_sombra", "z_delta"], regra_id="abs_z_lt_5", regra="|z| < 5 → NOT_FALSIFIED (os controles da Fase 5 estão no veredito do módulo; o livro não os relê)", valor=get(core, [B, "fase5", "P_sombra", "z_delta"])),
    "NOT_FALSIFIED", True, "Fase 4 (Pantheon+): z = %.3f; Fase 5 (SH0ES completo): z = %.4f, H₀ = %.2f ± %.2f (lido de fase5.validacao)" % (get(core, ["bancada_fase4_v378", "P_ladder_d1b", "z_delta"]), get(core, [B, "fase5", "P_sombra", "z_delta"]), H0, SH0),
    "lido por número da chave citada", False, False, "SN de DES Y5 / Rubin: mesmo estimador, mesma lei; FALSIFIED se |z| ≥ 5 contra β_TGL com os controles do estimador passando", True)
cob("R2", "reflexo", "propagação (o reflexo acumulado na distância)", "propagação de ondas gravitacionais (banco residente: %d/%d janelas O4 OK)" % (GW_OK, GW_JOBS), "dissipação na propagação, Γ_ω = ½ β τ★ ω² (n = −2), τ★ = t_Planck no ramo canônico [PRINCIPLED IDENTIFICATION]",
    "o coeficiente β̂ τ★ do dephasing acumulado na distância", None, None, "DECLARED", None, "AWAITING", True,
    "NENHUM estimador rodou sobre a dissipação na propagação; o DADO já foi aberto para outros estimadores (Fase 6: cópia atrasada; v369-P2: strain %s Hz)" % "–".join("%g" % b for b in BAND),
    "sem poder no ramo canônico: Γ(100 Hz) = ½·β·t_Planck·(2π·100)² = %.2e s⁻¹ (t_Planck = %.4e s, das constantes do um.py); o resultado honesto esperado é AWAITING/NOT_FALSIFIED_UNDERPOWERED, não FALSIFIED" % (GAMMA_100, T_PLANCK),
    True, False, "construir o estimador, selar por hash e DIZER O PODER antes de abrir o banco para esse fim", True)
cob("R3", "reflexo", "distância modular (dephasing na distância)", "Coma", "D_L TGL vs referência %.1f ± %.1f Mpc (lida do core)" % (COMA_REF, COMA_SIG), "z_TGL com as duas sigmas", "coma_reveal_state_v376", "Z_TGL_P1P30_BOTH_SIGMAS", "NUMBER_READ",
    dict(caminho=["coma_reveal_state_v376", "z_TGL_both_sigmas"], regra_id="abs_z_lt_5", regra="|z| < 5 → NOT_FALSIFIED", valor=get(core, ["coma_reveal_state_v376", "z_TGL_both_sigmas"])),
    "NOT_FALSIFIED", True, "z_TGL = +%.2f (ambas as sigmas); pós-dição não cega (REVEAL de %s)" % (get(core, ["coma_reveal_state_v376", "z_TGL_both_sigmas"]), COMA_DATA),
    "setores H₀ D1a vs D1 V3 declarados incompatíveis (aberto: H0_SECTORS_D1A_VS_D1V3_INCOMPATIBLE_OPEN)", False, False, "nova distância independente de Coma: FALSIFIED se |z| ≥ 5", True)
cob("R4", "reflexo", "lente (a luz dobrada: projeção)", "piso dos vazios por lente (V11; SDSS DR7)", "ρ_vazio/ρ̄ ≥ β (estimador autocalibrante; unilateral)", "r̂_cal e o limite inferior a 5σ", "void_floor_v11", "TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED", "TOKEN_CONTAINS", None,
    "NOT_FALSIFIED", True, "V11 (a chave citada): r̂_cal = %.4f ± %.4f; L5 = %.4f = %.1fβ; powered" % (get(core, ["void_floor_v11", "primary", "rhat_cal"]), get(core, ["void_floor_v11", "primary", "sigma"]), get(core, ["void_floor_v11", "primary", "L5"]), get(core, ["void_floor_v11", "primary", "L5"]) / beta),
    ("unilateral: o ΛCDM raso também passa; o conjunto CONGELADO do V11 não tem FALSIFIED (%s) — o FALSIFIED do futuro exige OUTRO estimador; a MESMA fonte (void_floor_v11) alimenta o degrau experimental do gate "
     "(auditoria de 28/09, GV-01); os números do v92 DESI×KiDS são de outro rito, ao lado") % ", ".join(V11_SET), False, False,
    "Euclid/LSST + LRG/ELG com um estimador NOVO cujo conjunto de vereditos tenha FALSIFIED, pré-registrado antes", False)
ECO = [
    ("R5a", "(√β, KMS 2π/κ)", [B, "fase6", "todos_KMS", "z_excl_R1"], "KMS_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_AT_DELAY_LAW", None, "FALSIFIED", "V3 (o veredito principal, subamostra «todos»; a V4/V5 o repetem como regressão)",
     "subamostra cega de O4 (não independente): z_excl %.2f" % get(core, [B, "fase6", "o4_KMS_cego", "z_excl_R1"]), "cláusula 5"),
    ("R5b", "(sin 2θ, KMS 2π/κ)", [B, "fase6", "todos_KMS", "z_excl_R2"], "KMS_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_AT_DELAY_LAW", None, "FALSIFIED", "V3 (o veredito principal; a V4/V5 o repetem como regressão)",
     "subamostra cega de O4 (não independente): z_excl %.2f" % get(core, [B, "fase6", "o4_KMS_cego", "z_excl_R2"]), "cláusula 5"),
    ("R5c", "(√β, DEC 2GM_f/βc³)", [B, "emenda_v5", "todos_DEC", "z_excl_R1"], "DEC2025_LAW_FROM_THE_ARCHIVE_DECLARED_BLIND_TO_THE_NEW_DELAY_ONLY_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_Z_EXCL_8P95_AND_16P83", None, "FALSIFIED", "V5",
     "cega quanto à lei DEC; subamostra cega de O4 (não independente): z_excl %.2f" % get(core, [B, "emenda_v5", "o4_DEC_cego", "z_excl_R1"]), "cláusula 5"),
    ("R5d", "(sin 2θ, DEC)", [B, "emenda_v5", "todos_DEC", "z_excl_R2"], "DEC2025_LAW_FROM_THE_ARCHIVE_DECLARED_BLIND_TO_THE_NEW_DELAY_ONLY_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_Z_EXCL_8P95_AND_16P83", None, "FALSIFIED", "V5",
     "cega quanto à lei DEC; subamostra cega de O4 (não independente): z_excl %.2f" % get(core, [B, "emenda_v5", "o4_DEC_cego", "z_excl_R2"]), "cláusula 5"),
    ("R5e", "(sin 2θ, MAY ln 1/β)", [B, "emenda_v4", "todos_MAY", "z_excl_R2"], "AMENDMENT_V4_NOT_BLIND_SIN2THETA_MAY_FALSIFIED_Z_8P75_CONTROLS_MAX_Z_4P81", None, "FALSIFIED", "V4 (NÃO cega)",
     "controles max |z| = %.2f < 5; o veredito V3 pré-registrado para o MESMO par é %s (fase6.veredito_principal) — o FALSIFIED vem da V4, não cega, dito" % (get(core, [B, "emenda_v4", "todos_MAY", "controles_max_z"]), V3_MAY_R2), "cláusula 5"),
    ("R5f", "(√β, MAY ln 1/β)", [B, "emenda_v4", "todos_MAY", "sist_rel_R1"], "SQRT_BETA_MAY_INCONCLUSIVE_IS_THE_ESTIMATOR_LIMIT", None, "INCONCLUSIVE", "V4",
     "sistemática relativa %.2f > 0,3 — a cláusula V4 (o limite do estimador; não se emenda uma terceira vez sem estimador novo)" % get(core, [B, "emenda_v4", "todos_MAY", "sist_rel_R1"]), "cláusula 2 e cláusula V4"),
    ("R5g", "(β, DEC)", [B, "emenda_v5", "todos_DEC", "sist_rel_R3"], "DOC_PAIR_BETA_DEC_INCONCLUSIVE_SYSTEMATICS_BY_RULE_2_POWER_0P87_BLIND_POWER_1P47", None, "INCONCLUSIVE", "V5",
     ("o rótulo EMITIDO é INCONCLUSIVE pela cláusula 2 (sistemática relativa %.2f > 0,3); pela cláusula V5 (o par do documento de dez/2025: poder de R3 %.2f < 5σ → NOT_FALSIFIED_UNDERPOWERED) seria "
      "NOT_FALSIFIED_UNDERPOWERED — a cláusula V5 NÃO foi aplicada pelo pipeline (clausula_v5_aplicada_pelo_pipeline = %s; V5_CLAUSE_UNDERPOWERED_NOT_APPLIED_STATED na v380): desvio dito, não corrigido por cima")
     % (get(core, [B, "emenda_v5", "todos_DEC", "sist_rel_R3"]), get(core, [B, "emenda_v5", "todos_DEC", "poder_R3"]), get(core, [B, "emenda_v5", "clausula_v5_aplicada_pelo_pipeline"])), "cláusula 2 (rótulo emitido); a cláusula V5 não aplicada, dito"),
    ("R5h", "(β, KMS 2π/κ)", None, "INCONCLUSIVE_SYSTEMATICS", [B, "fase6", "veredito_principal", "KMS|R3_quadrado"], "INCONCLUSIVE", "V3 (o veredito aninhado lido do core)", "a amplitude β é a menor das três leituras", "cláusula 2 ou bandeira"),
    ("R5i", "(β, MAY ln 1/β)", None, "INCONCLUSIVE_SYSTEMATICS", [B, "emenda_v4", "vereditos_todos", "MAY", "R3_quadrado"], "INCONCLUSIVE", "V4 (o veredito aninhado lido do core)", "a amplitude β é a menor das três leituras", "cláusula 2 ou bandeira"),
]
for id_, par, path_z, tok, vpath, desf, ver, qual, clausula in ECO:
    num = dict(caminho=path_z, regra="%s da Fase 6 e das emendas" % clausula, valor=get(core, path_z)) if path_z else None
    ja = ("%s: %s = %.2f" % (ver, "z_excl" if desf == "FALSIFIED" else "estatística lida", get(core, path_z))) if path_z else ("%s: %s" % (ver, get(core, vpath)))
    cob(id_, "reflexo", "cópia atrasada (fora da cadeia; registrada como extinta ou inconclusiva)", "eco como CÓPIA ATRASADA: o par %s" % par, "atraso τ da lei + amplitude da leitura", "amplitude refletida com atraso τ",
        B, tok, "TOKEN_CONTAINS", num, desf, False, ja, qual, False, False,
        "nenhum: o par está fora da cadeia (o eco como cópia atrasada); a regra não se move (TheLedgerOfCharges.falsified_extinguishes_the_charge_not_the_rule)", False, caminho_veredito=vpath)
cob("R6", "face", "o custo POSTO na face por CONVENÇÃO (fundo vestido; não canônica; fica AO LADO) — não pela leitura (BAO e SH0ES são distância)", "fundo vestido (1+β)Ω_m: D1 V3 (Planck comprimido + DESI DR1 + SH0ES)", "D1 V3: β no fundo", "β̂ do MCMC",
    "d1_camb_v3_real_v366", "TENSION_IS_NOT_FALSIFICATION", "NUMBER_READ",
    dict(caminho=["d1_camb_v3_real_v366", "z_alpha_sqrt_e"], controle=["d1_camb_v3_real_v366", "autocorr", "params_below_50tau"], controle_ok=0, regra_id="control_then_abs_z_lt_5",
         regra="controle de convergência: `params_below_50tau` conta os parâmetros com N < 50τ (os que REPROVAM) e tem de ser 0; senão INCONCLUSIVE; com o controle passando, |z| < 5 → NOT_FALSIFIED",
         valor=Z_AE, controle_valor=BELOW),
    "INCONCLUSIVE", False, "α√e a %.2fσ (no limiar); Δχ² = +%.2f vs ΛCDM; cadeia com N < 50τ em %d/%d parâmetros → controle reprovado" % (Z_AE, DCHI2, BELOW, NPAR),
    "TENSÃO não é falsificação; a fronteira TENSION/INCONCLUSIVE cabe no ruído de Monte Carlo (ressalva do próprio core); o setor H₀ desta convenção é declarado incompatível com o de R1 (aberto: coma_reveal_state_v376); não é a regra (R1)", False, False,
    "sem canal novo: convenção não canônica, ao lado", False)
cob("F1", "face", "sem distância (dH/dz): a face lê o fluxo", "cronômetros cósmicos (Moresco+2022, covariância Moresco+2020)",
    "a face lê o PAGAMENTO: H(z) do fundo nu (ΛCDM; β só na lei por distância); a alternativa é a lei D1b aplicada aos cronômetros (I(z) = (2/3)·ln E(z); K = E(z*)^{2β/3}; núcleo PRIMÁRIO das Fases 4/7)",
    "√Δχ² com sinal (z_ganho = sinal(χ²_ΛCDM − χ²_TGL)·√|Δχ²|; positivo = a lei D1b ajusta melhor os cronômetros), β fixo em α√e", B, "CC_COVARIANCE_MORESCO2020_T_LAW_GAIN_Z_M1P66_VS_DIAG_M3P22", "NUMBER_READ",
    dict(caminho=[B, "fase7", "T_z_ganho", "moresco2020"], regra_id="z_gain_ge_plus5_falsifies", regra="UNILATERAL: z_ganho ≥ +5 → FALSIFIED (da leitura «a face lê o fundo nu»: o custo seria exigido na face); senão NOT_FALSIFIED", valor=get(core, [B, "fase7", "T_z_ganho", "moresco2020"])),
    "NOT_FALSIFIED", True, "Fase 7: z_ganho da lei D1b nos cronômetros = %.2f (desfavorecida); ln B sombra/acumulada = %.2f" % (get(core, [B, "fase7", "T_z_ganho", "moresco2020"]), get(core, [B, "fase7", "U_lnB_sombra_vs_acumulada", "moresco2020"])),
    ("poder prévio %.2fσ (fase7.poder_previo.moresco2020): pela régua da Fase 4 (> 1σ = com poder) tem poder; pela da Fase 6 (poder < 5 → NOT_FALSIFIED_UNDERPOWERED) seria underpowered — o livro conta "
     "NOT_FALSIFIED e diz as duas réguas; a face é LEGÍVEL (ótica), e a alternativa D1b fica desfavorecida, não excluída") % F1_POD, False, False,
    "novos cronômetros (Euclid/DESI): FALSIFIED se z_ganho ≥ +5 com a covariância completa", True)
cob("F2", "face", "o remanescente (o Nome M_f entregue): a RG", "ringdown (empilhamentos da V2: %d séries a 3 ms; o GW250114 está em ringdown_scope_v376.c6, %s)" % (RD_N, C6),
    "correspondência com a RG (ramo canônico, τ★ = t_Planck); ramo B ENCERRADO (R6 ratificada)", "desvio do amortecimento δτ em relação à RG",
    RD, "INCONCLUSIVE_SYSTEMATICS", "TOKEN_CONTAINS", dict(caminho=[RD, "stacks", "3.0", "mean_bias_rel"], regra="a matriz da emenda V2 (%s): viés relativo de τ acima do limiar → INCONCLUSIVE_SYSTEMATICS" % RD_EMENDA, valor=RD_BIAS),
    "INCONCLUSIVE", True,
    "V2: INCONCLUSIVE_SYSTEMATICS pela matriz da emenda V2 (%s): o motivo lido do core «%s» — a sistemática da MEDIDA δτ = %.3f ± %.3f contra a RG, comum aos dois ramos" % (RD_EMENDA, RD_REASON, RD_DT, RD_SG),
    ("no ramo canônico o poder é ≈ 0 (δ_A = %.2e); o power_B = %.3f é o poder do ramo B, encerrado; que o custo esteja no ringdown (Γ ≈ %.1e s⁻¹ a 100 Hz) é leitura [ONTO] da correção ótica — "
     "o core lê o ringdown canônico como correspondência com a RG, não como teste de β (ringdown_scope_v376)") % (RD_DA, RD_PB, GAMMA_100), False, False,
    "FALSIFIED (da correspondência = o pagamento) se o amortecimento desviar da RG a ≥ 5σ com a sistemática da medida controlada", True)
cob("F3a", "face", "o relógio local lê a face de K_∂ (o custo local, Γ ∝ β)", "relógios de laboratório: partição P1 (matéria, por partícula)", "dephasing local sob P1; o sujeito é K_∂ (R2 ratificada)", "taxa de dephasing local",
    "clock_test_result_v369", "P1_NOT_FALSIFIED_UNDERPOWERED", "TOKEN_CONTAINS", dict(caminho=["clock_test_result_v369", "P1", "clock_5sigma_deficit_orders"], regra="cláusula 3 da Fase 6: poder < 5 → NOT_FALSIFIED_UNDERPOWERED", valor=get(core, ["clock_test_result_v369", "P1", "clock_5sigma_deficit_orders"])),
    "NOT_FALSIFIED", True, "UNDERPOWERED: déficit de %.1f ordens para 5σ (o campo nomeado P1.clock_5sigma_deficit_orders, vigente pela errata_v370); o poder (Z_expected da plataforma %s) = %.2e" % (get(core, ["clock_test_result_v369", "P1", "clock_5sigma_deficit_orders"]), P1_ID, P1_Z),
    "a leitura é permitida (ótica); falta instrumento", False, False, "relógio nuclear com as ordens que faltam; FALSIFIED se o dephasing local previsto for excluído a 5σ", True)
cob("F3b", "face", "o relógio local lê a face de K_∂ (modo de luz por braço)", "relógios de laboratório: partição P2 (modo de luz por braço; LIGO)", "dephasing local sob P2", "ASD prevista vs medida",
    "clock_test_result_v369", "P2_EXCLUDED_IN_READING", "TOKEN_CONTAINS", dict(caminho=["clock_test_result_v369", "P2", "fraction_excluded"], regra="leitura excluída por ordens de grandeza SEM nível de confiança (nσ não entra): EXCLUDED_IN_READING (acréscimo da casa), não FALSIFIED pela cláusula 5", valor=get(core, ["clock_test_result_v369", "P2", "fraction_excluded"])),
    "EXCLUDED_IN_READING", True, "excluída em %d/%d séries (%.1f%%) por %.2f ordens de grandeza (mediana, com calibração), sem nσ, na banda %s Hz" % (NEXC, NSER, 100.0 * NEXC / NSER, get(core, ["clock_test_result_v369", "P2", "orders_excluded_median_with_cal"]), "–".join("%g" % b for b in BAND)),
    "leitura excluída, não canal (convenção final_verdict_reading_v376); a cobrança P2 está EXTINTA", False, False, "nenhum: a leitura P2 está excluída", False)
cob("F3c", "face", "o relógio local lê a face de K_∂ (tempo estocástico universal)", "relógios de laboratório: partição P3 (universal)", "dephasing universal (reparametrização)", "nenhuma razão de frequências, largura ou franja o vê",
    "clock_test_result_v369", "P3_NO_LOCAL_OBSERVABLE", "DECLARED", None, "AWAITING", True, "P3: sem observável local (reparametrização comum)", "sem observável: aguarda um observável, não um dado", False, False, "nenhum canal enquanto não houver observável", False)
cob("F4", "face", "determinação direta (a face), não ajuste global (reflexos, ao lado)", "neutrino m₂ (%s)" % JUNO.get("fonte"), "m₂(TGL) = %.4f meV (lido do core); σ do lado da previsão (ratificado)" % M2P, "z contra Δm²₂₁ medido diretamente",
    "neutrino_m2", "TGL_NU_M2_ARMED_CONSISTENT", "NUMBER_READ",
    dict(caminho=["neutrino_m2", "values", "kill_rule_satisfeita"], regra_id="kill_rule_flag", regra="a regra de morte CONGELADA do NEUTRINO_M2_V2: «%s» (lida do core) — FALSIFIED sse values.kill_rule_satisfeita; R3 elege o perito (JUNO), NÃO muda a regra de morte" % KILL, valor=get(core, ["neutrino_m2", "values", "kill_rule_satisfeita"])),
    "NOT_FALSIFIED", True, "JUNO: %.2fσ; NuFIT global pós-JUNO: %.2fσ ao lado (perito = JUNO, R3 ratificada); kill_rule_satisfeita = %s; projeção 2031: %.2fσ" % (JUNO.get("tensao_sigma"), get(core, ["neutrino_m2", "values", "tensao_atual_sigma"]), get(core, ["neutrino_m2", "values", "kill_rule_satisfeita"]), PROJ31),
    "o que a face lê aqui é o custo por m₂ ∝ β (uma determinação direta), não o peso 1 − β; a escala é «%s» (frozen.escala_dimensional); a segunda determinação independente não está nomeada (errata RNC-06)" % ESCALA, False, False,
    "JUNO com mais exposição + uma segunda determinação independente: FALSIFIED se ambas derem ≥ 5σ (a regra congelada)", True)
assert len(C) == 20 and len({c["id"] for c in C}) == 20
for c in C:
    if c["chave_core"]:
        alvo = get(core, c["caminho_veredito"]) if c["caminho_veredito"] else V(c["chave_core"])
        assert c["token"] in str(alvo), (c["id"], c["chave_core"], c["token"])
    if c["como"] == "TOKEN_CONTAINS":
        assert c["desfecho_lido"] in c["token"] and (c["desfecho_lido"] != "FALSIFIED" or "NOT_FALSIFIED" not in c["token"]), c["id"]
    if c["como"] == "NUMBER_READ":
        n = c["numero"]; v = n["valor"]
        if n["regra_id"] == "abs_z_lt_5":
            assert c["desfecho_lido"] == ("NOT_FALSIFIED" if abs(v) < 5 else "FALSIFIED"), c["id"]
        elif n["regra_id"] == "z_gain_ge_plus5_falsifies":
            assert c["desfecho_lido"] == ("FALSIFIED" if v >= 5 else "NOT_FALSIFIED"), c["id"]
        elif n["regra_id"] == "control_then_abs_z_lt_5":
            assert c["desfecho_lido"] == ("INCONCLUSIVE" if n["controle_valor"] != n["controle_ok"] else ("NOT_FALSIFIED" if abs(v) < 5 else "FALSIFIED")), c["id"]
        elif n["regra_id"] == "kill_rule_flag":
            assert isinstance(v, bool) and c["desfecho_lido"] == ("FALSIFIED" if v else "NOT_FALSIFIED"), c["id"]
        else:
            raise AssertionError(("regra desconhecida", c["id"], n["regra_id"]))
contagem = {o: sum(1 for c in C if c["desfecho_lido"] == o) for o in ("NOT_FALSIFIED", "FALSIFIED", "INCONCLUSIVE", "AWAITING", "EXCLUDED_IN_READING")}
contagem_na_regra = {o: sum(1 for c in C if c["na_regra"] and c["desfecho_lido"] == o) for o in contagem}
eco = [c for c in C if c["id"].startswith("R5")]
eco_cont = dict(pares=len(eco), extintos=sum(1 for c in eco if c["desfecho_lido"] in ("FALSIFIED", "EXCLUDED_IN_READING")), inconclusivos=sum(1 for c in eco if c["desfecho_lido"] == "INCONCLUSIVE"))
n_canal = sum(1 for c in C if c["canal_futuro_com_falsified"])
pre = dict(
    identificador=ID, escrito_em=time.strftime("%Y-%m-%d %H:%M:%S"), autor="gerência (Claude Code, sessão 6da8f00d, casa Central de Patentes / Bancada Um)",
    supersede=dict(v1_2="PREREG_DOIS_LADOS_DELTA_K_20261002_V1_2", v1_2_sha256=ant_sha["v1_2"], mapa_seq_v1_2=279, v1_1_sha256=ant_sha["v1_1"], mapa_seq_v1_1=276, v1_sha256=ant_sha["v1"], mapa_seq_v1=275,
                   porque="achados CONFERIDOS no core da 3ª aferição independente da v382 (M-b R4; M-c R5g e as emendas; M-d F1; M-e a lei pendente; BAIXOS), ANTES de qualquer estimador ou dado NOVO; nenhum desfecho nem contagem muda; o V1, o V1.1 e o V1.2 ficam como registro, a correção vai ao lado"),
    um_py=dict(versao=selo.get("um_version"), sha256=um_sha, gate=selo.get("qg_closure_verdict")),
    pedido_do_operador=VERB,
    estimando=dict(
        nome="δ⟨K_∂⟩ — o desvio da expectativa do Hamiltoniano oculto da fronteira em relação à fronteira silenciosa (Λ, w = −1), lido nos DOIS lados",
        regra=dict(beta_tgl=beta, como="ALPHA_FINE_CODATA_2018 × √e em runtime (nunca literal)", alpha=alpha, alpha_lido_de=prov_alpha, theta_M=theta_M, custo_reflexo=beta, pagamento_face=1.0 - beta,
                   soma="|T|² + |R|² = 1 (Teorema S-∂; TheLedgerOfCharges.both_sides_are_read)"),
        onde_incide=("a chave do operador [INPUT/ONTO]: no Hamiltoniano oculto K = −log Δ [KNOWN: o gerador do fluxo modular; no kernel, `lightDelta` e o seu conjunto fixo]; o zero do lock mínimo HminMic = 1 − P_{ℂΩ} "
                     "(os zeros coincidem, ℂΩ; os operadores não) é o psion (ThePsionAndTheViscosity.psion_is_the_zero_of_the_hidden_hamiltonian); a ligação de Spec S(θ) a HminMic é [INPUT/ONTO], sem termo"),
        lados=LADOS,
        estatuto="a chave e as seis leituras são [INPUT/ONTO] ratificadas (com a correção ótica); β e a forma α√e são [DERIVED] do axioma (v376); a identificação de cada observável com δ⟨K_∂⟩ é leitura [ONTO] sem termo no kernel; os desfechos lidos são [REAL] (por token, no veredito do módulo ou no aninhado, ou por número lido do core com a regra dita)",
        lambda_fronteira_silenciosa="δ⟨K_∂⟩ = β|1 + w|, zero em w = −1 (ΛCDM = o limite de fronteira silenciosa) [REAL na forma da lei]; «espectro de gradiente negativo» (15/09) fica [ONTO] sem identidade de operador",
    ),
    cegueira=dict(
        sentido="«análise cega» = o estimador não viu o dado antes da regra; NÃO confundir com «nenhum lado é cego», que é a legibilidade dos dois lados (ótica)",
        declaracao="MISTA — dita canal a canal (campo `cego`: estimador e dado); hoje NENHUM canal é cego quanto ao dado; R2 é cego quanto ao ESTIMADOR (nunca construído)",
        cego_quanto_ao_estimador=[c["id"] for c in C if c["cego"]["estimador"]], cego_quanto_ao_dado=[c["id"] for c in C if c["cego"]["dado"]],
        o_que_nao_foi_feito="desde o V1 (02/10/2026 14:44) nenhum dado novo foi olhado; nenhum estimador da dissipação na PROPAGAÇÃO (R2) foi construído nem rodou",
    ),
    cobrancas=C, contagem=contagem, contagem_na_regra=contagem_na_regra, eco_como_copia_atrasada=eco_cont, n_canal_futuro_com_falsified=n_canal,
    na_regra=[c["id"] for c in C if c["na_regra"]], ao_lado=[c["id"] for c in C if not c["na_regra"]],
    pendentes=["a lei do lado TRANSMITIDO nos cronômetros, «1 − β por travessia», anunciada pela gerência na errata ótica de 02/10 e NÃO escrita: fica PENDENTE (a escrever, com o poder dito antes de abrir), sem cobrança no livro enquanto a lei não existir (3ª aferição, M-e)",
               "o estimador da dissipação na propagação (R2), com o poder dito antes de abrir o banco de GW para esse fim"],
    regras_de_leitura=dict(
        desfechos=["NOT_FALSIFIED", "FALSIFIED", "INCONCLUSIVE", "AWAITING", "EXCLUDED_IN_READING"],
        fonte=("as regras da Fase 6 e das suas emendas V4/V5, LIDAS dos arquivos (sha256 em `fontes_das_regras`) e transcritas abaixo com a ORDEM; e os ACRÉSCIMOS DA CASA, ditos como tais: "
               "EXCLUDED_IN_READING (v369-P2; final_verdict_reading_v376), o controle de convergência de MCMC (R6), a regra unilateral do ganho com sinal (F1, Fases 4/7) e a regra de morte congelada do neutrino (F4)"),
        ordem_da_fase6="as cláusulas 1–6 aplicam-se NA ORDEM 1 → 6; a primeira que se aplica decide (logo FALSIFIED exige poder ≥ 5: a cláusula 3 vem antes da 5); a BANDEIRA não tem posição no texto da ordem — na V3 ela NÃO foi escrita pelo pipeline (FLAG_NOT_WRITTEN_BY_PIPELINE_STATED, dito na v380)",
        clausulas_fase6=EM["PREREGISTRO_FASE6_ECO_RADICAL_20261001.json"]["regra_de_veredito"],
        clausula_principal=CL_PRINCIPAL, clausula_V4=CL_V4, clausula_V5=CL_V5,
        nota_V5="a cláusula V5 governa o par (β, DEC) = R5g: o pipeline NÃO a aplicou (o rótulo emitido é o da cláusula 2); o desvio está dito, não corrigido por cima",
        acrescimo_excluded_in_reading="leitura excluída por ordens de grandeza SEM nível de confiança (nσ não entra): EXCLUDED_IN_READING; a cobrança está extinta; não é FALSIFIED pela cláusula 5 nem NOT_FALSIFIED",
        acrescimo_mcmc="para um estimando por MCMC: controle de convergência — params_below_50tau (os parâmetros com N < 50τ, os que REPROVAM) tem de ser 0; senão INCONCLUSIVE",
        acrescimo_ganho_com_sinal="para o ganho de uma lei com sinal (z_ganho; positivo = a lei ajusta melhor): a regra é UNILATERAL, z_ganho ≥ +5 → FALSIFIED da leitura que a lei contradiz; o poder prévio é dito nas duas réguas (Fase 4: > 1σ; Fase 6: ≥ 5)",
        acrescimo_regra_congelada="quando um módulo tem regra de morte CONGELADA (hash), vale ela: o neutrino exige |z| ≥ 5 em DUAS determinações independentes",
        falsified_onde_admitido="FALSIFIED é uma das cinco palavras do TIPO; que um canal concreto a admita depende da sua lei (o V11 não a tem) — dito no campo `canal_futuro_com_falsified` de cada cobrança",
        awaiting="sem estimador selado, sem dado ou sem observável",
        a_regra_nao_se_move="nenhum desfecho altera β = α√e (TheLedgerOfCharges.the_rule_is_constant_in_the_ledger, rfl: a regra não recebe o livro); «extinguir» é a definição `extinguished`",
        o_gate_nao_se_move="as bandeiras formais são função só do formal — propriedade do código do gate, conferida pelos probes negativos do runtime (qg_closure); o degrau EXPERIMENTAL do gate lê a mesma fonte do R4 (void_floor_v11), dito; cosmologia jamais vira prova matemática",
        os_dois_lados="toda cobrança diz o LADO e o CRITÉRIO de lado; nenhum lado é cego para o outro (a correção do operador: a TGL é ótica)",
        na_regra_vs_ao_lado="as cobranças AO LADO (R5a–i, o eco como cópia atrasada; R6, o custo posto na face por convenção) não são da regra canônica e são contadas em separado",
        sinal="o custo entra com sinal + no reflexo (β > 0); não se inverte sinal depois do dado",
        significancia_global="Šidák sobre as famílias do livro, ao lado da local",
    ),
    fontes_das_regras={k: v["sha256"] for k, v in EM.items()},
    fontes=dict(cronometros=prov_cc, covariancia_moresco2020=prov_cov, desi_dr2=prov_dr2, constantes_do_um_py=dict(C_LIGHT=C_LIGHT, G_NEWTON=G_NEWTON, H_PLANCK_EXACT=H_PLANCK, t_Planck_s=T_PLANCK, R2_gamma_100Hz_por_s=GAMMA_100)),
    codigo_da_bancada_sha256=codigo,
    ordem=["(1) este pré-registro V1.3, selado por hash e inscrito no mapa (corrige 279)", "(2) o um.py v382 lê este arquivo por hash (prove_the_ledger_of_charges_v382) e confere as 20 cobranças contra o core",
           "(3) os PENDENTES (a lei transmitida nos cronômetros; o estimador da propagação R2): construção, selagem e PODER antes de abrir", "(4) só então o dado; cada canal com o seu desfecho e o seu lado"],
)
b = json.dumps(pre, ensure_ascii=False, indent=1).encode("utf-8")
if os.path.exists(SAIDA_JSON) or os.path.exists(SAIDA_MD):
    raise SystemExit("o pré-registro V1.3 já existe; não se reescreve: " + SAIDA_JSON)
tmp = SAIDA_JSON + ".tmp"; open(tmp, "wb").write(b); assert open(tmp, "rb").read() == b; os.replace(tmp, SAIDA_JSON)
hj = hashlib.sha256(b).hexdigest()
sn = lambda x: "sim" if x else "não"
L = ["# PRÉ-REGISTRO DOS DOIS LADOS — V1.3 — o estimando δ⟨K_∂⟩ (02/10/2026)", "",
     "**Identificador** `%s` · escrito em %s · JSON `%s` (sha256 `%s`) · `um.py` %s `%s` · SUPERSEDE o V1.2 (sha256 `%s`, mapa seq 279), o V1.1 (`%s`, seq 276) e o V1 (`%s`, seq 275), antes de qualquer estimador ou dado NOVO, pelos achados conferidos no core da 3ª aferição da v382; nenhum desfecho nem contagem muda; os anteriores ficam como registro." % (
         ID, pre["escrito_em"], os.path.basename(SAIDA_JSON), hj, selo.get("um_version"), um_sha[:16], ant_sha["v1_2"][:16], ant_sha["v1_1"][:16], ant_sha["v1"][:16]), "",
     "**A chave do operador (verbatim):** «%s»" % VERB["chave_20261002"], "", "**A correção (verbatim):** «%s»; e a ordem: «%s»" % (VERB["otica_20261002"], VERB["prossiga_20261002"]), "",
     "**O estimando.** %s. A regra: β_TGL = α√e = %.12f (nunca literal); o custo no reflexo é β = |R|² = sin²θ_M; o pagamento na face é 1 − β = |T|² = cos²θ_M; |T|² + |R|² = 1. Onde incide: %s." % (pre["estimando"]["nome"], beta, pre["estimando"]["onde_incide"]), "",
     "**Os dois lados.** REFLEXO: %s. FACE: %s." % (LADOS["reflexo"], LADOS["face"]), "",
     "## As 20 cobranças (UMA por lei de leitura; nenhuma move a regra; FALSIFIED só onde a lei do canal o admite — %d com canal futuro que o admite)" % n_canal, "",
     "| id | lado | critério de lado | registro | como | desfecho lido | na regra | o que JÁ foi lido (core v381) | qualificador | cego (estimador / dado) | canal futuro com FALSIFIED | o que se pré-registra |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for c in C:
    L.append("| %s | %s | %s | %s | %s | `%s` | %s | %s | %s | %s / %s | %s | %s |" % (c["id"], c["lado"], c["criterio_de_lado"], c["registro"], c["como"], c["desfecho_lido"], "sim" if c["na_regra"] else "não (ao lado)",
                                                                                c["ja_lido"], c["qualificador"], sn(c["cego"]["estimador"]), sn(c["cego"]["dado"]), sn(c["canal_futuro_com_falsified"]), c["futuro"]))
L += ["", "Contagem: %s · na regra (%d): %s · ao lado (%d): %s · o eco como cópia atrasada: %d pares, %d extintos, %d inconclusivos." % (
          ", ".join("%s %d" % kv for kv in contagem.items()), len(pre["na_regra"]), ", ".join("%s %d" % kv for kv in contagem_na_regra.items() if kv[1]), len(pre["ao_lado"]), ", ".join(pre["ao_lado"]),
          eco_cont["pares"], eco_cont["extintos"], eco_cont["inconclusivos"]), "",
      "## Pendentes", ""] + ["- %s" % x for x in pre["pendentes"]]
L += ["", "## Regras de leitura — as da Fase 6 e das emendas V4/V5 (lidas dos arquivos, na ordem) e os acréscimos da casa (ditos como tais)", ""]
for k, v in pre["regras_de_leitura"].items():
    if k == "desfechos":
        continue
    if isinstance(v, list):
        L.append("- **%s:**" % k); L += ["  - %s" % x for x in v]
    else:
        L.append("- **%s:** %s" % (k, v))
L += ["", "Fontes das regras (sha256): %s." % "; ".join("%s `%s`" % (k, v[:16]) for k, v in pre["fontes_das_regras"].items()), "",
      "## Cegueira", "", "%s. Declaração: %s. Cego quanto ao estimador: %s. Cego quanto ao dado: %s. %s." % (pre["cegueira"]["sentido"], pre["cegueira"]["declaracao"], ", ".join(pre["cegueira"]["cego_quanto_ao_estimador"]) or "nenhum",
                                                                                                         ", ".join(pre["cegueira"]["cego_quanto_ao_dado"]) or "nenhum", pre["cegueira"]["o_que_nao_foi_feito"]), "",
      "## Ordem", ""] + ["%d. %s" % (i + 1, s) for i, s in enumerate(pre["ordem"])]
L += ["", "Estatuto: %s. PROVADA ≠ CONFIRMADA; `NOT_FALSIFIED` nunca é `CONFIRMED`." % pre["estimando"]["estatuto"], ""]
bm = "\n".join(L).encode("utf-8")
tmp = SAIDA_MD + ".tmp"; open(tmp, "wb").write(bm); assert open(tmp, "rb").read() == bm; os.replace(tmp, SAIDA_MD)
hm = hashlib.sha256(bm).hexdigest()
print("pré-registro V1.3 gravado:", SAIDA_JSON, "sha256", hj); print("md:", SAIDA_MD, "sha256", hm); print("contagem:", contagem, "| na regra:", contagem_na_regra, "| eco:", eco_cont, "| canal futuro com FALSIFIED:", n_canal)
r = subprocess.run([sys.executable, os.path.join(BU, "ferramentas", "publicar_relatorio.py"), "02/10 — Pré-registro dos dois lados V1.3 (20 cobranças; Fase 6 e emendas V4/V5 lidas; supersede o V1.2 antes de qualquer estimador ou dado novo)", SAIDA_MD],
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout.strip()); assert r.returncode == 0, r.stderr
out = rotas.registrar([dict(tipo="ERRATA", corrige=279, rota_id="bancada.preregistro_dois_lados_delta_k", status="PROPOSTA", estatuto="REAL", olhou_dados=False,
                            titulo="V1.3 SUPERSEDE o V1.2 (seq 279) ANTES de qualquer estimador ou dado novo, pelos achados conferidos da 3ª aferição da v382: R4 sem FALSIFIED no conjunto congelado e a mesma fonte do degrau experimental do gate; R5g com a cláusula V5 não aplicada dita; as emendas V4/V5 transcritas (lidas); F1 na lei D1b com o poder prévio nas duas réguas; a lei transmitida nos cronômetros PENDENTE; números lidos do core e do um.py; nenhum desfecho nem contagem muda",
                            resultado="V1.3 gravado (JSON sha256 %s; MD sha256 %s); contagem %s; na regra %s; eco %s; canal futuro com FALSIFIED %d; publicado na aba Relatórios; o V1.2 (sha256 %s) fica como registro" % (
                                hj[:16], hm[:16], json.dumps(contagem, ensure_ascii=False), json.dumps(contagem_na_regra, ensure_ascii=False), json.dumps(eco_cont, ensure_ascii=False), n_canal, ant_sha["v1_2"][:16]),
                            nao_refazer="não dizer «FALSIFIED possível em todas»; não citar a cláusula 3 no R5g (a não aplicada é a V5); não escrever (1+z)^β no F1 (a lei lida é a D1b)",
                            proximo_passo="um.py v382 lê o V1.3 por hash; depois os pendentes (a lei transmitida nos cronômetros; o estimador da propagação R2) com o poder dito antes de abrir",
                            artefatos=[SAIDA_JSON, SAIDA_MD], refs=["kernel.regra_matriz.cadeia_unica"])],
                      "claude-code/central-de-patentes (gerência) sessão 6da8f00d")
print("inscrito no mapa: seq", out["gravados"][0]["seq"], "cabeça", out["cabeca"][:16])
