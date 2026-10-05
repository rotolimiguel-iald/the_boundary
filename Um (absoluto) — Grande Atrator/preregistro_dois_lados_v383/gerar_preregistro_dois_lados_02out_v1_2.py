# -*- coding: utf-8 -*-
"""PRÉ-REGISTRO DOS DOIS LADOS, V1.2 (02/10/2026) — SUPERSEDE o V1.1 (`PREREG_DOIS_LADOS_DELTA_K_20261002_V1_1`, mapa seq 276), que já supersedia o V1
(seq 275), ANTES de qualquer estimador ou dado NOVO, pelos achados CONFERIDOS no core da 2ª aferição independente da v382:
  ALTO-1 (F2: a causa do INCONCLUSIVE é a sistemática da MEDIDA — viés relativo de τ acima do limiar da matriz da emenda V2 —, comum aos dois ramos, não o poder);
  M1 (R6: `params_below_50tau` conta os parâmetros que REPROVAM; o controle passa só com 0); M2 (F1: z_ganho tem sinal — positivo = a lei acumulada ajusta
  melhor —; a regra é UNILATERAL, z_ganho ≥ +5 → FALSIFIED); M3 (F4: a regra de morte CONGELADA do NEUTRINO_M2_V2 exige DUAS determinações independentes;
  R3 elege o perito, não muda a regra); BAIXOS: os dois pares do eco que faltavam ((β, KMS) e (β, MAY)), o caminho do R5g (sist_rel_R3, cláusula 2), o campo
  nomeado do F3a (P1.clock_5sigma_deficit_orders), a cláusula 4 e a ORDEM das cláusulas da Fase 6, a célula de cegueira do R2, o critério de lado do R6,
  e os números lidos do core (nada digitado). O V1 e o V1.1 NÃO são reescritos: ficam como registro; a correção vai AO LADO.
Grava JSON + MD com sha256, publica o MD na Bancada e inscreve a ERRATA no mapa (corrige 276). β nunca literal (lido do um.py canônico via bancada.fontes)."""
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
ID = "PREREG_DOIS_LADOS_DELTA_K_20261002_V1_2"
SAIDA_JSON = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_2.json")
SAIDA_MD = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_2.md")
V1_JSON = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002.json")
V11_JSON = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_1.json")


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
um_sha = sha(os.path.join(NOS, "um.py"))
assert selo.get("um_version") == "v381", "o canônico não é a v381 selada: %s" % selo.get("um_version")
V = lambda k: str((core.get(k) or {}).get("verdict") or "")
v1_sha, v11_sha = sha(V1_JSON), sha(V11_JSON)
codigo = {p: sha(os.path.join(BU, "bancada", p)) for p in ("motor.py", "fundo_gpu.py", "camb_tabela.py", "evidencia.py", "amostrador.py", "regua.py", "fontes.py", "gw.py", "gw_stack.py")
          if os.path.exists(os.path.join(BU, "bancada", p))}
try:
    _, prov_cc = fontes.cronometros_moresco2022()
    _, prov_cov = fontes.cronometros_covariancia_moresco2020()
    _, prov_dr2 = fontes.desi_dr2()
except Exception as e:
    raise SystemExit("fonte recusada: %r" % (e,))

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
NEXC, NSER = get(core, ["clock_test_result_v369", "P2", "n_excluded"]), get(core, ["clock_test_result_v369", "P2", "n_series"])
M2P = get(core, ["neutrino_m2", "values", "m2_pred_meV"]); JUNO = get(core, ["neutrino_m2", "values", "escada_datada", 1]); KILL = get(core, ["neutrino_m2", "frozen", "kill_rule"])
RD_REASON = (get(core, [RD, "reasons"]) or [""])[0]; RD_BIAS = get(core, [RD, "stacks", "3.0", "mean_bias_rel"]); RD_DA = get(core, [RD, "stacks", "3.0", "delta_pred_A"])
RD_PB = get(core, [RD, "stacks", "3.0", "power_B"]); RD_DT = get(core, [RD, "stacks", "3.0", "delta_tau"]); RD_SG = get(core, [RD, "stacks", "3.0", "sigma"])
m = re.search(r"emenda \(([0-9a-f]{16})\)", str(get(core, [RD, "statuses", "leitura"]) or "")); RD_EMENDA = m.group(1) if m else None
assert all(isinstance(x, (int, float)) for x in (H0, SH0, Z_AE, DCHI2, BELOW, NEXC, NSER, M2P, RD_BIAS, RD_DA, RD_PB, RD_DT, RD_SG)) and RD_EMENDA and RD_REASON
assert str(JUNO.get("fonte", "")).startswith("JUNO") and "DUAS" in str(KILL)
assert "H0_SECTORS_D1A_VS_D1V3_INCOMPATIBLE_OPEN" in V("coma_reveal_state_v376")

C = []


def cob(id_, lado, criterio, registro, lei, leitura, chave, token, como, numero, desfecho, na_regra, ja_lido, qualificador, cego_estimador, cego_dado, futuro, caminho_veredito=None):
    C.append(dict(id=id_, lado=lado, criterio_de_lado=criterio, registro=registro, lei=lei, leitura=leitura, chave_core=chave, token=token, como=como, numero=numero,
                  caminho_veredito=caminho_veredito, desfecho_lido=desfecho, na_regra=na_regra, ja_lido=ja_lido, qualificador=qualificador,
                  cego={"estimador": cego_estimador, "dado": cego_dado}, futuro=futuro))


cob("R1", "reflexo", "leitura por distância (a escada)", "escada de distâncias (SN Ia + cefeidas; SH0ES completo)", "K = E(z*)^{2β/3} sobre o fundo nu; nó fixo em z* (convenção do artigo; R1 ratificada)",
    "δ̂ = β̂ − β_TGL no expoente da leitura por distância", B, "SH0ES_IN_THE_BENCH_H0_LADDER_73P53_PM_1P02", "NUMBER_READ",
    dict(caminho=[B, "fase5", "P_sombra", "z_delta"], regra_id="abs_z_lt_5", regra="|z| < 5 → NOT_FALSIFIED; |z| ≥ 5 com os controles → FALSIFIED", valor=get(core, [B, "fase5", "P_sombra", "z_delta"])),
    "NOT_FALSIFIED", True, "Fase 4 (Pantheon+): z = %.3f; Fase 5 (SH0ES completo): z = %.4f, H₀ = %.2f ± %.2f (lido de fase5.validacao)" % (get(core, ["bancada_fase4_v378", "P_ladder_d1b", "z_delta"]), get(core, [B, "fase5", "P_sombra", "z_delta"]), H0, SH0),
    "lido por número da chave citada", False, False, "SN de DES Y5 / Rubin: mesmo estimador, mesma lei; FALSIFIED se |z| ≥ 5 contra β_TGL com os controles passando")
cob("R2", "reflexo", "propagação (o reflexo acumulado na distância)", "propagação de ondas gravitacionais (banco residente: 336/354 janelas O4)", "dissipação na propagação, Γ_ω = ½ β τ★ ω² (n = −2), τ★ = t_Planck no ramo canônico [PRINCIPLED IDENTIFICATION]",
    "o coeficiente β̂ τ★ do dephasing acumulado na distância", None, None, "DECLARED", None, "AWAITING", True,
    "NENHUM estimador rodou sobre a dissipação na propagação; o DADO já foi aberto para outros estimadores (Fase 6: cópia atrasada; v369-P2: strain 100–300 Hz)",
    "sem poder no ramo canônico: Γ(100 Hz) = ½·β·t_Planck·(2π·100)² (o um.py v382 calcula em runtime, de ħ, G e c do programa); o resultado honesto esperado é AWAITING/NOT_FALSIFIED_UNDERPOWERED, não FALSIFIED",
    True, False, "construir o estimador, selar por hash e DIZER O PODER antes de abrir o banco para esse fim")
cob("R3", "reflexo", "distância modular (dephasing na distância)", "Coma", "D_L TGL vs referência 98,5 ± 2,2 Mpc", "z_TGL com as duas sigmas", "coma_reveal_state_v376", "Z_TGL_P1P30_BOTH_SIGMAS", "NUMBER_READ",
    dict(caminho=["coma_reveal_state_v376", "z_TGL_both_sigmas"], regra_id="abs_z_lt_5", regra="|z| < 5 → NOT_FALSIFIED", valor=get(core, ["coma_reveal_state_v376", "z_TGL_both_sigmas"])),
    "NOT_FALSIFIED", True, "z_TGL = +%.2f (ambas as sigmas); pós-dição não cega (REVEAL de 19/08/2026)" % get(core, ["coma_reveal_state_v376", "z_TGL_both_sigmas"]),
    "setores H₀ D1a vs D1 V3 declarados incompatíveis (aberto: H0_SECTORS_D1A_VS_D1V3_INCOMPATIBLE_OPEN)", False, False, "nova distância independente de Coma: FALSIFIED se |z| ≥ 5")
cob("R4", "reflexo", "lente (a luz dobrada: projeção)", "piso dos vazios por lente (V11; SDSS DR7)", "ρ_vazio/ρ̄ ≥ β (estimador autocalibrante; unilateral)", "r̂_cal e o limite inferior a 5σ", "void_floor_v11", "TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED", "TOKEN_CONTAINS", None,
    "NOT_FALSIFIED", True, "V11 (a chave citada): r̂_cal = %.4f ± %.4f; L5 = %.4f = %.1fβ; powered" % (get(core, ["void_floor_v11", "primary", "rhat_cal"]), get(core, ["void_floor_v11", "primary", "sigma"]), get(core, ["void_floor_v11", "primary", "L5"]), get(core, ["void_floor_v11", "primary", "L5"]) / beta),
    "unilateral: o ΛCDM raso também passa (os números do v92 DESI×KiDS são de OUTRO rito, ao lado)", False, False, "Euclid/LSST + LRG/ELG: FALSIFIED se o piso medido ficar abaixo de β a 5σ com os nulos passando")
ECO = [
    ("R5a", "(√β, KMS 2π/κ)", [B, "fase6", "todos_KMS", "z_excl_R1"], "KMS_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_AT_DELAY_LAW", None, "FALSIFIED", "réplica cega O4: z_excl %.2f" % get(core, [B, "fase6", "o4_KMS_cego", "z_excl_R1"]), "cláusula 5"),
    ("R5b", "(sin 2θ, KMS 2π/κ)", [B, "fase6", "todos_KMS", "z_excl_R2"], "KMS_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_AT_DELAY_LAW", None, "FALSIFIED", "réplica cega O4: z_excl %.2f" % get(core, [B, "fase6", "o4_KMS_cego", "z_excl_R2"]), "cláusula 5"),
    ("R5c", "(√β, DEC 2GM_f/βc³)", [B, "emenda_v5", "todos_DEC", "z_excl_R1"], "DEC2025_LAW_FROM_THE_ARCHIVE_DECLARED_BLIND_TO_THE_NEW_DELAY_ONLY_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_Z_EXCL_8P95_AND_16P83", None, "FALSIFIED", "cega quanto à lei DEC; O4: z_excl %.2f" % get(core, [B, "emenda_v5", "o4_DEC_cego", "z_excl_R1"]), "cláusula 5"),
    ("R5d", "(sin 2θ, DEC)", [B, "emenda_v5", "todos_DEC", "z_excl_R2"], "DEC2025_LAW_FROM_THE_ARCHIVE_DECLARED_BLIND_TO_THE_NEW_DELAY_ONLY_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_Z_EXCL_8P95_AND_16P83", None, "FALSIFIED", "cega quanto à lei DEC; O4: z_excl %.2f" % get(core, [B, "emenda_v5", "o4_DEC_cego", "z_excl_R2"]), "cláusula 5"),
    ("R5e", "(sin 2θ, MAY ln 1/β)", [B, "emenda_v4", "todos_MAY", "z_excl_R2"], "AMENDMENT_V4_NOT_BLIND_SIN2THETA_MAY_FALSIFIED_Z_8P75_CONTROLS_MAX_Z_4P81", None, "FALSIFIED", "NÃO cega (V4); controles max |z| = %.2f < 5" % get(core, [B, "emenda_v4", "todos_MAY", "controles_max_z"]), "cláusula 5"),
    ("R5f", "(√β, MAY ln 1/β)", [B, "emenda_v4", "todos_MAY", "sist_rel_R1"], "SQRT_BETA_MAY_INCONCLUSIVE_IS_THE_ESTIMATOR_LIMIT", None, "INCONCLUSIVE", "sistemática relativa %.2f > 0,3 (o limite do estimador)" % get(core, [B, "emenda_v4", "todos_MAY", "sist_rel_R1"]), "cláusula 2"),
    ("R5g", "(β, DEC)", [B, "emenda_v5", "todos_DEC", "sist_rel_R3"], "DOC_PAIR_BETA_DEC_INCONCLUSIVE_SYSTEMATICS_BY_RULE_2_POWER_0P87_BLIND_POWER_1P47", None, "INCONCLUSIVE",
     "sistemática relativa %.2f > 0,3; o poder %.2f < 5 fica ao lado — a cláusula 3 NÃO foi aplicada pelo pipeline (dito na v380)" % (get(core, [B, "emenda_v5", "todos_DEC", "sist_rel_R3"]), get(core, [B, "emenda_v5", "todos_DEC", "poder_R3"])), "cláusula 2"),
    ("R5h", "(β, KMS 2π/κ)", None, "INCONCLUSIVE_SYSTEMATICS", [B, "fase6", "veredito_principal", "KMS|R3_quadrado"], "INCONCLUSIVE", "o veredito aninhado lido do core (V3, subamostra «todos»)", "cláusula 2 ou bandeira"),
    ("R5i", "(β, MAY ln 1/β)", None, "INCONCLUSIVE_SYSTEMATICS", [B, "emenda_v4", "vereditos_todos", "MAY", "R3_quadrado"], "INCONCLUSIVE", "o veredito aninhado lido do core (V4)", "cláusula 2 ou bandeira"),
]
for id_, par, path_z, tok, vpath, desf, qual, clausula in ECO:
    num = dict(caminho=path_z, regra="%s da Fase 6" % clausula, valor=get(core, path_z)) if path_z else None
    ja = ("Fase 6 V3/V4/V5: %s = %.2f" % ("z_excl" if desf == "FALSIFIED" else "estatística lida", get(core, path_z))) if path_z else ("Fase 6: %s" % get(core, vpath))
    cob(id_, "reflexo", "cópia atrasada (fora da cadeia; registrada como extinta ou inconclusiva)", "eco como CÓPIA ATRASADA: o par %s" % par, "atraso τ da lei + amplitude da leitura", "amplitude refletida com atraso τ",
        B, tok, "TOKEN_CONTAINS", num, desf, False, ja, qual, False, False,
        "nenhum: o par está fora da cadeia (o eco como cópia atrasada); a regra não se move (TheLedgerOfCharges.falsified_extinguishes_the_charge_not_the_rule)", caminho_veredito=vpath)
cob("R6", "face", "o custo POSTO na face por CONVENÇÃO (fundo vestido; não canônica; fica AO LADO) — não pela leitura (BAO e SH0ES são distância)", "fundo vestido (1+β)Ω_m: D1 V3 (Planck comprimido + DESI DR1 + SH0ES)", "D1 V3: β no fundo", "β̂ do MCMC",
    "d1_camb_v3_real_v366", "TENSION_IS_NOT_FALSIFICATION", "NUMBER_READ",
    dict(caminho=["d1_camb_v3_real_v366", "z_alpha_sqrt_e"], controle=["d1_camb_v3_real_v366", "autocorr", "params_below_50tau"], controle_ok=0, regra_id="control_then_abs_z_lt_5",
         regra="controle de convergência: `params_below_50tau` conta os parâmetros com N < 50τ (os que REPROVAM) e tem de ser 0; senão INCONCLUSIVE; com o controle passando, |z| < 5 → NOT_FALSIFIED",
         valor=Z_AE, controle_valor=BELOW),
    "INCONCLUSIVE", False, "α√e a %.2fσ (no limiar); Δχ² = +%.2f vs ΛCDM; cadeia com N < 50τ em %s/4 parâmetros → controle reprovado" % (Z_AE, DCHI2, BELOW),
    "TENSÃO não é falsificação; a fronteira TENSION/INCONCLUSIVE cabe no ruído de Monte Carlo (ressalva do próprio core); o setor H₀ desta convenção é declarado incompatível com o de R1 (aberto: coma_reveal_state_v376); não é a regra (R1)", False, False,
    "sem canal novo: convenção não canônica, ao lado")
cob("F1", "face", "sem distância (dH/dz): a face lê o fluxo", "cronômetros cósmicos (Moresco+2022, covariância Moresco+2020)", "a face lê o PAGAMENTO: H(z) do fundo nu (primária); a lei acumulada (1+z)^β (o custo na face) é a alternativa",
    "√Δχ² com sinal (z_ganho = sinal(χ²_ΛCDM − χ²_TGL)·√|Δχ²|; positivo = a lei acumulada ajusta melhor; Fase 4/7), β fixo em α√e", B, "CC_COVARIANCE_MORESCO2020_T_LAW_GAIN_Z_M1P66_VS_DIAG_M3P22", "NUMBER_READ",
    dict(caminho=[B, "fase7", "T_z_ganho", "moresco2020"], regra_id="z_gain_ge_plus5_falsifies", regra="UNILATERAL: z_ganho ≥ +5 → FALSIFIED (da leitura «a face lê o fundo nu»: o custo seria exigido na face); senão NOT_FALSIFIED", valor=get(core, [B, "fase7", "T_z_ganho", "moresco2020"])),
    "NOT_FALSIFIED", True, "Fase 7: z_ganho da lei acumulada = %.2f (desfavorecida); ln B sombra/acumulada = %.2f" % (get(core, [B, "fase7", "T_z_ganho", "moresco2020"]), get(core, [B, "fase7", "U_lnB_sombra_vs_acumulada", "moresco2020"])),
    "a face é LEGÍVEL (ótica); o que ela mostra é o pagamento, e a alternativa acumulada fica desfavorecida, não excluída", False, False,
    "novos cronômetros (Euclid/DESI): FALSIFIED se z_ganho ≥ +5 com a covariância completa")
cob("F2", "face", "o remanescente (o Nome M_f entregue): a RG", "ringdown (GW250114 e empilhamentos)", "correspondência com a RG (ramo canônico, τ★ = t_Planck); ramo B ENCERRADO (R6 ratificada)", "desvio do amortecimento δτ em relação à RG",
    RD, "INCONCLUSIVE_SYSTEMATICS", "TOKEN_CONTAINS", dict(caminho=[RD, "stacks", "3.0", "mean_bias_rel"], regra="a matriz da emenda V2 (%s): viés relativo de τ acima do limiar → INCONCLUSIVE_SYSTEMATICS" % RD_EMENDA, valor=RD_BIAS),
    "INCONCLUSIVE", True,
    "V2: INCONCLUSIVE_SYSTEMATICS pela matriz da emenda V2 (%s): o motivo lido do core «%s» — a sistemática da MEDIDA δτ = %.3f ± %.3f contra a RG, comum aos dois ramos" % (RD_EMENDA, RD_REASON, RD_DT, RD_SG),
    "no ramo canônico o poder é ≈ 0 (δ_A = %.2e); o 0,2σ (power_B = %.3f) é o poder do ramo B, encerrado; o custo está no ringdown, Γ ~ 10⁻⁴⁰ s⁻¹, legível e nulo à sensibilidade disponível" % (RD_DA, RD_PB), False, False,
    "FALSIFIED (da correspondência = o pagamento) se o amortecimento desviar da RG a ≥ 5σ com a sistemática da medida controlada")
cob("F3a", "face", "o relógio local lê a face de K_∂ (o custo local, Γ ∝ β)", "relógios de laboratório: partição P1 (matéria, por partícula)", "dephasing local sob P1; o sujeito é K_∂ (R2 ratificada)", "taxa de dephasing local",
    "clock_test_result_v369", "P1_NOT_FALSIFIED_UNDERPOWERED", "TOKEN_CONTAINS", dict(caminho=["clock_test_result_v369", "P1", "clock_5sigma_deficit_orders"], regra="cláusula 3 da Fase 6: poder < 5 → NOT_FALSIFIED_UNDERPOWERED", valor=get(core, ["clock_test_result_v369", "P1", "clock_5sigma_deficit_orders"])),
    "NOT_FALSIFIED", True, "UNDERPOWERED: déficit de %.1f ordens para 5σ (o campo nomeado P1.clock_5sigma_deficit_orders, vigente pela errata_v370)" % get(core, ["clock_test_result_v369", "P1", "clock_5sigma_deficit_orders"]),
    "a leitura é permitida (ótica); falta instrumento", False, False, "relógio nuclear com as ordens que faltam; sem canal novo hoje")
cob("F3b", "face", "o relógio local lê a face de K_∂ (modo de luz por braço)", "relógios de laboratório: partição P2 (modo de luz por braço; LIGO)", "dephasing local sob P2", "ASD prevista vs medida",
    "clock_test_result_v369", "P2_EXCLUDED_IN_READING", "TOKEN_CONTAINS", dict(caminho=["clock_test_result_v369", "P2", "fraction_excluded"], regra="leitura excluída por ordens de grandeza SEM nível de confiança (nσ não entra): EXCLUDED_IN_READING (acréscimo da casa), não FALSIFIED pela cláusula 5", valor=get(core, ["clock_test_result_v369", "P2", "fraction_excluded"])),
    "EXCLUDED_IN_READING", True, "excluída em %d/%d séries (%.1f%%) por %.2f ordens de grandeza (mediana, com calibração), sem nσ" % (NEXC, NSER, 100.0 * NEXC / NSER, get(core, ["clock_test_result_v369", "P2", "orders_excluded_median_with_cal"])),
    "leitura excluída, não canal (convenção final_verdict_reading_v376); a cobrança P2 está EXTINTA", False, False, "nenhum: a leitura P2 está excluída")
cob("F3c", "face", "o relógio local lê a face de K_∂ (tempo estocástico universal)", "relógios de laboratório: partição P3 (universal)", "dephasing universal (reparametrização)", "nenhuma razão de frequências, largura ou franja o vê",
    "clock_test_result_v369", "P3_NO_LOCAL_OBSERVABLE", "DECLARED", None, "AWAITING", True, "P3: sem observável local (reparametrização comum)", "sem observável: aguarda um observável, não um dado", False, False, "nenhum canal enquanto não houver observável")
cob("F4", "face", "determinação direta (a face), não ajuste global (reflexos, ao lado)", "neutrino m₂ (JUNO autônomo, 59 d)", "m₂(TGL) = %.4f meV (lido do core); σ do lado da previsão (ratificado)" % M2P, "z contra Δm²₂₁ medido diretamente",
    "neutrino_m2", "TGL_NU_M2_ARMED_CONSISTENT", "NUMBER_READ",
    dict(caminho=["neutrino_m2", "values", "kill_rule_satisfeita"], regra_id="kill_rule_flag", regra="a regra de morte CONGELADA do NEUTRINO_M2_V2: «%s» (lida do core) — FALSIFIED sse values.kill_rule_satisfeita; R3 elege o perito (JUNO), NÃO muda a regra de morte" % KILL, valor=get(core, ["neutrino_m2", "values", "kill_rule_satisfeita"])),
    "NOT_FALSIFIED", True, "JUNO 59 d: %.2fσ; NuFIT global pós-JUNO: %.2fσ ao lado (perito = JUNO, R3 ratificada); kill_rule_satisfeita = %s" % (JUNO.get("tensao_sigma"), get(core, ["neutrino_m2", "values", "tensao_atual_sigma"]), get(core, ["neutrino_m2", "values", "kill_rule_satisfeita"])),
    "o que a face lê aqui é o custo por m₂ ∝ β (uma determinação direta), não o peso 1 − β; a segunda determinação independente não está nomeada (errata RNC-06)", False, False,
    "JUNO com 6 anos + uma segunda determinação independente: FALSIFIED se ambas derem ≥ 5σ (a regra congelada)")
assert len(C) == 20 and len({c["id"] for c in C}) == 20
for c in C:   # o desfecho tem de constar no core pela string do veredito (token, no veredito do módulo ou no aninhado) ou pelo número lido (regra dita)
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
pre = dict(
    identificador=ID, escrito_em=time.strftime("%Y-%m-%d %H:%M:%S"), autor="gerência (Claude Code, sessão 6da8f00d, casa Central de Patentes / Bancada Um)",
    supersede=dict(v1_1="PREREG_DOIS_LADOS_DELTA_K_20261002_V1_1", v1_1_sha256=v11_sha, mapa_seq_v1_1=276, v1="PREREG_DOIS_LADOS_DELTA_K_20261002_V1", v1_sha256=v1_sha, mapa_seq_v1=275,
                   porque="achados CONFERIDOS no core da 2ª aferição independente da v382 (ALTO-1 F2; M1 R6; M2 F1; M3 F4; BAIXOS: pares do eco, R5g, F3a, cláusula 4 e ordem da Fase 6, célula de cegueira, critério de lado do R6, números lidos), ANTES de qualquer estimador ou dado NOVO; o V1 e o V1.1 ficam como registro, a correção vai ao lado"),
    um_py=dict(versao=selo.get("um_version"), sha256=um_sha, gate=selo.get("qg_closure_verdict")),
    pedido_do_operador=VERB,
    estimando=dict(
        nome="δ⟨K_∂⟩ — o desvio da expectativa do Hamiltoniano oculto da fronteira em relação à fronteira silenciosa (Λ, w = −1), lido nos DOIS lados",
        regra=dict(beta_tgl=beta, como="ALPHA_FINE_CODATA_2018 × √e em runtime (nunca literal)", alpha=alpha, alpha_lido_de=prov_alpha, theta_M=theta_M, custo_reflexo=beta, pagamento_face=1.0 - beta,
                   soma="|T|² + |R|² = 1 (Teorema S-∂; TheLedgerOfCharges.both_sides_are_read)"),
        onde_incide=("a chave do operador [INPUT/ONTO]: no Hamiltoniano oculto K = −log Δ; no kernel, o zero do lock mínimo HminMic = 1 − P_{ℂΩ} (os zeros coincidem, ℂΩ; os operadores não) "
                     "é o psion (ThePsionAndTheViscosity.psion_is_the_zero_of_the_hidden_hamiltonian); a ligação de Spec S(θ) a HminMic é [INPUT/ONTO], sem termo"),
        lados=LADOS,
        estatuto="a chave e as seis leituras são [INPUT/ONTO] ratificadas; β e a forma α√e são [DERIVED] do axioma (v376); a identificação de cada observável com δ⟨K_∂⟩ é leitura [ONTO] sem termo no kernel; os desfechos lidos são [REAL] (por token, no veredito do módulo ou no aninhado, ou por número lido do core com a regra dita)",
        lambda_fronteira_silenciosa="δ⟨K_∂⟩ = β|1 + w|, zero em w = −1 (ΛCDM = o limite de fronteira silenciosa) [REAL na forma da lei]; «espectro de gradiente negativo» (15/09) fica [ONTO] sem identidade de operador",
    ),
    cegueira=dict(
        sentido="«análise cega» = o estimador não viu o dado antes da regra; NÃO confundir com «nenhum lado é cego», que é a legibilidade dos dois lados (ótica)",
        declaracao="MISTA — dita canal a canal (campo `cego`: estimador e dado); hoje NENHUM canal é cego quanto ao dado; R2 é cego quanto ao ESTIMADOR (nunca construído)",
        cego_quanto_ao_estimador=[c["id"] for c in C if c["cego"]["estimador"]], cego_quanto_ao_dado=[c["id"] for c in C if c["cego"]["dado"]],
        o_que_nao_foi_feito="nenhum estimador da dissipação na PROPAGAÇÃO (R2) foi construído nem rodou; nenhum dado novo foi olhado nesta sessão",
    ),
    cobrancas=C, contagem=contagem, contagem_na_regra=contagem_na_regra, na_regra=[c["id"] for c in C if c["na_regra"]], ao_lado=[c["id"] for c in C if not c["na_regra"]],
    regras_de_leitura=dict(
        desfechos=["NOT_FALSIFIED", "FALSIFIED", "INCONCLUSIVE", "AWAITING", "EXCLUDED_IN_READING"],
        fonte=("PREREGISTRO_FASE6_ECO_RADICAL_20261001.json, regra_de_veredito: as cláusulas 1–6 e a bandeira, transcritas abaixo com a ORDEM; e os ACRÉSCIMOS DA CASA, ditos como tais: "
               "EXCLUDED_IN_READING (v369-P2; final_verdict_reading_v376), o controle de convergência de MCMC (R6), a regra unilateral do ganho com sinal (F1, Fase 4/7) e a regra de morte congelada do neutrino (F4)"),
        ordem_da_fase6="as cláusulas aplicam-se NA ORDEM 1 → 6; a primeira que se aplica decide (logo, na Fase 6, FALSIFIED exige poder ≥ 5: a cláusula 3 vem antes da 5)",
        clausula_1="n_eventos < 10 → INSUFFICIENT_SAMPLE (não ocorreu em nenhuma cobrança do livro)",
        clausula_2="sistemática relativa = max(|descasamento|, |família|)/|â_esperado| > 0,3 → INCONCLUSIVE_SYSTEMATICS",
        clausula_3="poder = |recuperação|/σ_usado < 5 → NOT_FALSIFIED_UNDERPOWERED (variante contada em NOT_FALSIFIED, com o sufixo dito)",
        clausula_4="|z_det| ≥ 5 com recuperação na janela e o sinal da leitura → DETECTED_AT_DELAY_LAW (não ocorreu; DETECTED nunca é CONFIRMED)",
        clausula_5="z_excl = (recuperação − â)/σ_usado ≥ 5 → FALSIFIED_AT_DELAY_LAW (falsifica o PAR leitura×lei, não a teoria)",
        clausula_6="senão → NOT_FALSIFIED_POWERED",
        bandeira="controle com |z| ≥ 5 sem |z_det| ≥ 5 → INCONCLUSIVE_SYSTEMATICS (DELAY_CONTROL_RESPONDS); σ_usado = max(σ fora da fonte, σ jackknife por evento) — fail-closed",
        acrescimo_excluded_in_reading="leitura excluída por ordens de grandeza SEM nível de confiança (nσ não entra): EXCLUDED_IN_READING; a cobrança está extinta; não é FALSIFIED pela cláusula 5 nem NOT_FALSIFIED",
        acrescimo_mcmc="para um estimando por MCMC: controle de convergência — params_below_50tau (os parâmetros com N < 50τ, os que REPROVAM) tem de ser 0; senão INCONCLUSIVE",
        acrescimo_ganho_com_sinal="para o ganho de uma lei com sinal (z_ganho; positivo = a lei ajusta melhor): a regra é UNILATERAL, z_ganho ≥ +5 → FALSIFIED da leitura que a lei contradiz",
        acrescimo_regra_congelada="quando um módulo tem regra de morte CONGELADA (hash), vale ela: o neutrino exige |z| ≥ 5 em DUAS determinações independentes",
        awaiting="sem estimador selado, sem dado ou sem observável",
        a_regra_nao_se_move="nenhum desfecho altera β = α√e (TheLedgerOfCharges.the_rule_is_constant_in_the_ledger, rfl: a regra não recebe o livro); «extinguir» é a definição `extinguished`",
        o_gate_nao_se_move="as bandeiras formais são função só do formal — propriedade do código do gate, conferida pelos probes negativos do runtime (qg_closure); cosmologia jamais vira prova matemática",
        os_dois_lados="toda cobrança diz o LADO e o CRITÉRIO de lado; nenhum lado é cego para o outro (a correção do operador: a TGL é ótica)",
        na_regra_vs_ao_lado="as cobranças AO LADO (R5a–i, o eco como cópia atrasada; R6, o custo posto na face por convenção) não são da regra canônica e são contadas em separado",
        sinal="o custo entra com sinal + no reflexo (β > 0); não se inverte sinal depois do dado",
        significancia_global="Šidák sobre as famílias do livro, ao lado da local",
    ),
    fontes=dict(cronometros=prov_cc, covariancia_moresco2020=prov_cov, desi_dr2=prov_dr2),
    codigo_da_bancada_sha256=codigo,
    ordem=["(1) este pré-registro V1.2, selado por hash e inscrito no mapa (corrige 276)", "(2) o um.py v382 lê este arquivo por hash (prove_the_ledger_of_charges_v382) e confere as 20 cobranças contra o core",
           "(3) o estimador da dissipação na propagação (R2): construção, selagem e PODER antes de abrir", "(4) só então o dado; cada canal com o seu desfecho e o seu lado"],
)
b = json.dumps(pre, ensure_ascii=False, indent=1).encode("utf-8")
if os.path.exists(SAIDA_JSON) or os.path.exists(SAIDA_MD):
    raise SystemExit("o pré-registro V1.2 já existe; não se reescreve: " + SAIDA_JSON)
tmp = SAIDA_JSON + ".tmp"; open(tmp, "wb").write(b); assert open(tmp, "rb").read() == b; os.replace(tmp, SAIDA_JSON)
hj = hashlib.sha256(b).hexdigest()
sn = lambda x: "sim" if x else "não"
L = ["# PRÉ-REGISTRO DOS DOIS LADOS — V1.2 — o estimando δ⟨K_∂⟩ (02/10/2026)", "",
     "**Identificador** `%s` · escrito em %s · JSON `%s` (sha256 `%s`) · `um.py` %s `%s` · SUPERSEDE o V1.1 (`%s`, sha256 `%s`, mapa seq 276), que supersedia o V1 (sha256 `%s`, seq 275), antes de qualquer estimador ou dado NOVO, pelos achados conferidos no core da 2ª aferição da v382; o V1 e o V1.1 ficam como registro." % (
         ID, pre["escrito_em"], os.path.basename(SAIDA_JSON), hj, selo.get("um_version"), um_sha[:16], pre["supersede"]["v1_1"], v11_sha[:16], v1_sha[:16]), "",
     "**A chave do operador (verbatim):** «%s»" % VERB["chave_20261002"], "", "**A correção (verbatim):** «%s»; e a ordem: «%s»" % (VERB["otica_20261002"], VERB["prossiga_20261002"]), "",
     "**O estimando.** %s. A regra: β_TGL = α√e = %.12f (nunca literal); o custo no reflexo é β = |R|² = sin²θ_M; o pagamento na face é 1 − β = |T|² = cos²θ_M; |T|² + |R|² = 1. Onde incide: %s." % (pre["estimando"]["nome"], beta, pre["estimando"]["onde_incide"]), "",
     "**Os dois lados.** REFLEXO: %s. FACE: %s." % (LADOS["reflexo"], LADOS["face"]), "",
     "## As 20 cobranças (UMA por lei de leitura; FALSIFIED possível em todas; nenhuma move a regra)", "",
     "| id | lado | critério de lado | registro | como | desfecho lido | na regra | o que JÁ foi lido (core v381) | qualificador | cego (estimador / dado) | o que se pré-registra |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for c in C:
    L.append("| %s | %s | %s | %s | %s | `%s` | %s | %s | %s | %s / %s | %s |" % (c["id"], c["lado"], c["criterio_de_lado"], c["registro"], c["como"], c["desfecho_lido"], "sim" if c["na_regra"] else "não (ao lado)",
                                                                           c["ja_lido"], c["qualificador"], sn(c["cego"]["estimador"]), sn(c["cego"]["dado"]), c["futuro"]))
L += ["", "Contagem: %s · na regra (%d): %s · ao lado (%d): %s." % (", ".join("%s %d" % kv for kv in contagem.items()), len(pre["na_regra"]), ", ".join("%s %d" % kv for kv in contagem_na_regra.items() if kv[1]), len(pre["ao_lado"]), ", ".join(pre["ao_lado"])), "",
      "## Regras de leitura — as da Fase 6 (transcritas, na ordem) e os acréscimos da casa (ditos como tais)", ""] + ["- **%s:** %s" % (k, v) for k, v in pre["regras_de_leitura"].items() if k != "desfechos"]
L += ["", "## Cegueira", "", "%s. Declaração: %s. Cego quanto ao estimador: %s. Cego quanto ao dado: %s. %s." % (pre["cegueira"]["sentido"], pre["cegueira"]["declaracao"], ", ".join(pre["cegueira"]["cego_quanto_ao_estimador"]) or "nenhum",
                                                                                                         ", ".join(pre["cegueira"]["cego_quanto_ao_dado"]) or "nenhum", pre["cegueira"]["o_que_nao_foi_feito"]), "",
      "## Ordem", ""] + ["%d. %s" % (i + 1, s) for i, s in enumerate(pre["ordem"])]
L += ["", "Estatuto: %s. PROVADA ≠ CONFIRMADA; `NOT_FALSIFIED` nunca é `CONFIRMED`." % pre["estimando"]["estatuto"], ""]
bm = "\n".join(L).encode("utf-8")
tmp = SAIDA_MD + ".tmp"; open(tmp, "wb").write(bm); assert open(tmp, "rb").read() == bm; os.replace(tmp, SAIDA_MD)
hm = hashlib.sha256(bm).hexdigest()
print("pré-registro V1.2 gravado:", SAIDA_JSON, "sha256", hj); print("md:", SAIDA_MD, "sha256", hm); print("contagem:", contagem, "| na regra:", contagem_na_regra)
r = subprocess.run([sys.executable, os.path.join(BU, "ferramentas", "publicar_relatorio.py"), "02/10 — Pré-registro dos dois lados V1.2 (20 cobranças; Fase 6 na ordem; supersede o V1.1 antes de qualquer estimador ou dado novo)", SAIDA_MD],
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout.strip()); assert r.returncode == 0, r.stderr
out = rotas.registrar([dict(tipo="ERRATA", corrige=276, rota_id="bancada.preregistro_dois_lados_delta_k", status="PROPOSTA", estatuto="REAL", olhou_dados=False,
                            titulo="V1.2 SUPERSEDE o V1.1 (seq 276) ANTES de qualquer estimador ou dado novo, pelos achados conferidos da 2ª aferição da v382: F2 pela sistemática da MEDIDA (viés relativo de τ; emenda V2), não pelo poder; R6 com o controle no sentido certo (params_below_50tau = 0); F1 unilateral (z_ganho ≥ +5); F4 pela regra de morte congelada (duas determinações); 20 cobranças (+ (β, KMS) e (β, MAY)); cláusula 4 e ordem da Fase 6; números lidos do core",
                            resultado="V1.2 gravado (JSON sha256 %s; MD sha256 %s); contagem %s; na regra %s; publicado na aba Relatórios; o V1.1 (sha256 %s) e o V1 (sha256 %s) ficam como registro" % (hj[:16], hm[:16], json.dumps(contagem, ensure_ascii=False), json.dumps(contagem_na_regra, ensure_ascii=False), v11_sha[:16], v1_sha[:16]),
                            nao_refazer="não ler o F2 como «inconclusivo por poder»; não usar a regra bilateral no ganho com sinal; não falsificar o neutrino com uma só determinação; params_below_50tau conta os que REPROVAM",
                            proximo_passo="um.py v382 lê o V1.2 por hash; depois o estimador da propagação (R2) com o poder dito antes de abrir",
                            artefatos=[SAIDA_JSON, SAIDA_MD], refs=["kernel.regra_matriz.cadeia_unica"])],
                      "claude-code/central-de-patentes (gerência) sessão 6da8f00d")
print("inscrito no mapa: seq", out["gravados"][0]["seq"], "cabeça", out["cabeca"][:16])
