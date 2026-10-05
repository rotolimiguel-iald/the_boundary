# -*- coding: utf-8 -*-
"""PRÉ-REGISTRO DOS DOIS LADOS, V1.1 (02/10/2026) — SUPERSEDE o V1 (`PREREG_DOIS_LADOS_DELTA_K_20261002_V1`, mapa seq 275) ANTES de qualquer estimador novo
rodar sobre dado real, pelos achados do aferidor independente da v382 (D1: a regra INCONCLUSIVE não era a da Fase 6; B1: F3 escondia a leitura P2 excluída;
B3: desfechos declarados, não lidos; B4: R6 sem as ressalvas e sem o controle de convergência; B5: F2 com número do ramo B; B6: R4 com números de outro rito;
B8: critério de lado não dito cobrança a cobrança; D2/D3: cegueira com dois sentidos). O V1 NÃO é reescrito: fica como registro; a correção vai AO LADO.
Grava JSON + MD com sha256, publica o MD na Bancada e inscreve a ERRATA no mapa (corrige 275). β nunca literal (lido do um.py canônico via bancada.fontes)."""
import hashlib
import json
import math
import os
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
ID = "PREREG_DOIS_LADOS_DELTA_K_20261002_V1_1"
SAIDA_JSON = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_1.json")
SAIDA_MD = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002_V1_1.md")
V1_JSON = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002.json")


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
v1_sha = sha(V1_JSON)
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
LADOS = {
    "reflexo": "o lado do CUSTO: a leitura que passa pelo reflexo (por distância, pela propagação, pela lente, pela cópia atrasada); o que ela lê é o custo β = |R|² ou o seu rastro",
    "face": "o lado do PAGAMENTO: a leitura que NÃO passa pelo reflexo (sem distância, ou direta, ou o Nome entregue); o que ela lê pode ser o peso 1 − β = |T|², o custo local Γ ∝ β, ou a RG — o critério é dito cobrança a cobrança (aferidor B8)",
}
# ---- as 18 cobranças: UMA por lei de leitura (aferidor B1); desfecho por TOKEN do core, por NÚMERO lido do core com a regra dita, ou DECLARADO (aferidor B3)
C = []


def cob(id_, lado, criterio, registro, lei, leitura, chave, token, como, numero, desfecho, na_regra, ja_lido, qualificador, cego, futuro):
    C.append(dict(id=id_, lado=lado, criterio_de_lado=criterio, registro=registro, lei=lei, leitura=leitura, chave_core=chave, token=token, como=como, numero=numero,
                  desfecho_lido=desfecho, na_regra=na_regra, ja_lido=ja_lido, qualificador=qualificador, cego=cego, futuro=futuro))


cob("R1", "reflexo", "leitura por distância (a escada)", "escada de distâncias (SN Ia + cefeidas; SH0ES completo)", "K = E(z*)^{2β/3} sobre o fundo nu; nó fixo em z* (convenção do artigo; R1 ratificada)",
    "δ̂ = β̂ − β_TGL no expoente da leitura por distância", B, "SH0ES_IN_THE_BENCH_H0_LADDER_73P53_PM_1P02", "NUMBER_READ",
    dict(caminho=[B, "fase5", "P_sombra", "z_delta"], regra="|z| < 5 → NOT_FALSIFIED; z_excl ≥ 5 com controles → FALSIFIED", valor=get(core, [B, "fase5", "P_sombra", "z_delta"])),
    "NOT_FALSIFIED", True, "Fase 4 (Pantheon+): z = %.3f; Fase 5 (SH0ES completo): z = %.4f, H₀ = 73,53 ± 1,02" % (get(core, ["bancada_fase4_v378", "P_ladder_d1b", "z_delta"]), get(core, [B, "fase5", "P_sombra", "z_delta"])),
    "lido por número da chave citada; não cego", False, "SN de DES Y5 / Rubin: mesmo estimador, mesma lei; FALSIFIED se z_excl ≥ 5 contra β_TGL com os controles passando")
cob("R2", "reflexo", "propagação (o reflexo acumulado na distância)", "propagação de ondas gravitacionais (banco residente: 336/354 janelas O4)", "dissipação na propagação, Γ_ω = ½ β τ★ ω² (n = −2), τ★ = t_Planck no ramo canônico [PRINCIPLED IDENTIFICATION]",
    "o coeficiente β̂ τ★ do dephasing acumulado na distância", None, None, "DECLARED", None, "AWAITING", True,
    "NENHUM estimador rodou sobre a dissipação na propagação; o DADO já foi aberto para outros estimadores (Fase 6: cópia atrasada; v369-P2: strain 100–300 Hz)",
    "sem poder no ramo canônico: Γ(100 Hz) ≈ ½·β·t_Planck·(2π·100)² ≈ 1,3×10⁻⁴⁰ s⁻¹ (o um.py v382 calcula em runtime); acumulado sobre 1 Gpc ≈ 10⁻²³ — o resultado honesto esperado é AWAITING/NOT_FALSIFIED_UNDERPOWERED, não FALSIFIED",
    "cego quanto ao ESTIMADOR (nunca construído); o dado NÃO é cego (aberto na Fase 6 e na v369-P2)", "construir o estimador, selar por hash e DIZER O PODER antes de abrir o banco para esse fim")
cob("R3", "reflexo", "distância modular (dephasing na distância)", "Coma", "D_L TGL vs referência 98,5 ± 2,2 Mpc", "z_TGL com as duas sigmas", "coma_reveal_state_v376", "Z_TGL_P1P30_BOTH_SIGMAS", "NUMBER_READ",
    dict(caminho=["coma_reveal_state_v376", "z_TGL_both_sigmas"], regra="|z| < 5 → NOT_FALSIFIED", valor=get(core, ["coma_reveal_state_v376", "z_TGL_both_sigmas"])),
    "NOT_FALSIFIED", True, "z_TGL = +%.2f (ambas as sigmas); pós-dição não cega (REVEAL de 19/08/2026)" % get(core, ["coma_reveal_state_v376", "z_TGL_both_sigmas"]), "setores H₀ D1a vs D1 V3 declarados incompatíveis (aberto)", False, "nova distância independente de Coma: FALSIFIED se |z| ≥ 5")
cob("R4", "reflexo", "lente (a luz dobrada: projeção)", "piso dos vazios por lente (V11; SDSS DR7)", "ρ_vazio/ρ̄ ≥ β (estimador autocalibrante; unilateral)", "r̂_cal e o limite inferior a 5σ", "void_floor_v11", "TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED", "TOKEN_CONTAINS", None,
    "NOT_FALSIFIED", True, "V11 (a chave citada): r̂_cal = %.4f ± %.4f; L5 = %.4f = %.1fβ; powered" % (get(core, ["void_floor_v11", "primary", "rhat_cal"]), get(core, ["void_floor_v11", "primary", "sigma"]), get(core, ["void_floor_v11", "primary", "L5"]), get(core, ["void_floor_v11", "primary", "L5"]) / beta),
    "unilateral: o ΛCDM raso também passa (os números do v92 DESI×KiDS, r_c^cal = 0,189 ± 0,017, são de OUTRO rito, ao lado)", False, "Euclid/LSST + LRG/ELG: FALSIFIED se o piso medido ficar abaixo de β a 5σ com os nulos passando")
for id_, par, path_z, tok, desf, qual in (
        ("R5a", "(√β, KMS 2π/κ)", [B, "fase6", "todos_KMS", "z_excl_R1"], "KMS_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_AT_DELAY_LAW", "FALSIFIED", "réplica cega O4: z_excl %.2f" % get(core, [B, "fase6", "o4_KMS_cego", "z_excl_R1"])),
        ("R5b", "(sin 2θ, KMS 2π/κ)", [B, "fase6", "todos_KMS", "z_excl_R2"], "KMS_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_AT_DELAY_LAW", "FALSIFIED", "réplica cega O4: z_excl %.2f" % get(core, [B, "fase6", "o4_KMS_cego", "z_excl_R2"])),
        ("R5c", "(√β, DEC 2GM_f/βc³)", [B, "emenda_v5", "todos_DEC", "z_excl_R1"], "DEC2025_LAW_FROM_THE_ARCHIVE_DECLARED_BLIND_TO_THE_NEW_DELAY_ONLY_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_Z_EXCL_8P95_AND_16P83", "FALSIFIED", "cega quanto à lei DEC; O4: z_excl %.2f" % get(core, [B, "emenda_v5", "o4_DEC_cego", "z_excl_R1"])),
        ("R5d", "(sin 2θ, DEC)", [B, "emenda_v5", "todos_DEC", "z_excl_R2"], "DEC2025_LAW_FROM_THE_ARCHIVE_DECLARED_BLIND_TO_THE_NEW_DELAY_ONLY_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_Z_EXCL_8P95_AND_16P83", "FALSIFIED", "cega quanto à lei DEC; O4: z_excl %.2f" % get(core, [B, "emenda_v5", "o4_DEC_cego", "z_excl_R2"])),
        ("R5e", "(sin 2θ, MAY ln 1/β)", [B, "emenda_v4", "todos_MAY", "z_excl_R2"], "AMENDMENT_V4_NOT_BLIND_SIN2THETA_MAY_FALSIFIED_Z_8P75_CONTROLS_MAX_Z_4P81", "FALSIFIED", "NÃO cega (V4); controles max |z| = %.2f < 5" % get(core, [B, "emenda_v4", "todos_MAY", "controles_max_z"])),
        ("R5f", "(√β, MAY ln 1/β)", [B, "emenda_v4", "todos_MAY", "sist_rel_R1"], "SQRT_BETA_MAY_INCONCLUSIVE_IS_THE_ESTIMATOR_LIMIT", "INCONCLUSIVE", "cláusula 2: sistemática relativa %.2f > 0,3 (o limite do estimador)" % get(core, [B, "emenda_v4", "todos_MAY", "sist_rel_R1"])),
        ("R5g", "(β, DEC)", [B, "emenda_v5", "todos_DEC", "poder_R3"], "DOC_PAIR_BETA_DEC_INCONCLUSIVE_SYSTEMATICS_BY_RULE_2_POWER_0P87_BLIND_POWER_1P47", "INCONCLUSIVE", "cláusula 2 (sistemática relativa %.2f); poder %.2f < 5 — a cláusula 3 (NOT_FALSIFIED_UNDERPOWERED) NÃO foi aplicada pelo pipeline, dito na v380" % (get(core, [B, "emenda_v5", "todos_DEC", "sist_rel_R3"]), get(core, [B, "emenda_v5", "todos_DEC", "poder_R3"])))):
    cob(id_, "reflexo", "cópia atrasada (fora da cadeia; registrada como extinta ou inconclusiva)", "eco como CÓPIA ATRASADA: o par %s" % par, "atraso τ da lei + amplitude da leitura", "amplitude refletida com atraso τ",
        B, tok, "TOKEN_CONTAINS", dict(caminho=path_z, regra="z_excl ≥ 5 com controles → FALSIFIED (cláusula 5 da Fase 6)" if desf == "FALSIFIED" else "cláusula 2 da Fase 6 → INCONCLUSIVE_SYSTEMATICS", valor=get(core, path_z)),
        desf, False, "Fase 6 V3/V4/V5: %s = %.2f" % ("z_excl" if desf == "FALSIFIED" else "estatística lida", get(core, path_z)), qual, False,
        "nenhum: o par está fora da cadeia (o eco como cópia atrasada); a regra não se move (TheLedgerOfCharges.falsified_extinguishes_the_charge_not_the_rule)")
cob("R6", "face", "o custo POSTO na face (fundo vestido: convenção NÃO canônica; fica AO LADO)", "fundo vestido (1+β)Ω_m: D1 V3 (Planck comprimido + DESI DR1 + SH0ES)", "D1 V3: β no fundo", "β̂ do MCMC", "d1_camb_v3_real_v366", "TENSION_IS_NOT_FALSIFICATION", "NUMBER_READ",
    dict(caminho=["d1_camb_v3_real_v366", "z_alpha_sqrt_e"], controle=["d1_camb_v3_real_v366", "autocorr", "params_below_50tau"], regra="controle de convergência (4/4 parâmetros com N ≥ 50τ) reprovado → INCONCLUSIVE; senão |z| < 5 → NOT_FALSIFIED",
         valor=get(core, ["d1_camb_v3_real_v366", "z_alpha_sqrt_e"]), controle_valor=get(core, ["d1_camb_v3_real_v366", "autocorr", "params_below_50tau"])),
    "INCONCLUSIVE", False, "α√e a %.2fσ (no limiar); Δχ² = +%.2f vs ΛCDM; cadeia com N ≥ 50τ em %s/4 parâmetros → controle reprovado" % (get(core, ["d1_camb_v3_real_v366", "z_alpha_sqrt_e"]), get(core, ["d1_camb_v3_real_v366", "delta_chi2"]), get(core, ["d1_camb_v3_real_v366", "autocorr", "params_below_50tau"])),
    "TENSÃO 3,01σ não é falsificação; a fronteira TENSION/INCONCLUSIVE cabe no ruído de Monte Carlo (ressalva do próprio core); setor H₀ incompatível com R1 (aberto); não é a regra (R1)", False, "sem canal novo: convenção não canônica, ao lado")
cob("F1", "face", "sem distância (dH/dz): a face lê o fluxo", "cronômetros cósmicos (Moresco+2022, covariância Moresco+2020)", "a face lê o PAGAMENTO: H(z) do fundo nu (primária); a lei acumulada (1+z)^β é a alternativa comparada por ln B",
    "β̂ no expoente de H(z) (esperado 0 na face)", B, "CC_COVARIANCE_MORESCO2020_T_LAW_GAIN_Z_M1P66_VS_DIAG_M3P22", "NUMBER_READ",
    dict(caminho=[B, "fase7", "T_z_ganho", "moresco2020"], regra="|z| < 5 → NOT_FALSIFIED", valor=get(core, [B, "fase7", "T_z_ganho", "moresco2020"])),
    "NOT_FALSIFIED", True, "Fase 7: lei acumulada z = %.2f; ln B sombra/acumulada = %.2f" % (get(core, [B, "fase7", "T_z_ganho", "moresco2020"]), get(core, [B, "fase7", "U_lnB_sombra_vs_acumulada", "moresco2020"])),
    "a face é LEGÍVEL (ótica); o que ela mostra é o pagamento, e a alternativa acumulada fica desfavorecida, não excluída", False, "novos cronômetros (Euclid/DESI): FALSIFIED (da regra «custo no reflexo») se H(z) exigir o custo na face a ≥ 5σ com a covariância completa")
cob("F2", "face", "o remanescente (o Nome M_f entregue): a RG", "ringdown (GW250114 e empilhamentos)", "correspondência com a RG (ramo canônico, τ★ = t_Planck); ramo B ENCERRADO (R6 ratificada)", "desvio do amortecimento em relação à RG",
    "ringdown_dephasing_result_v2", "INCONCLUSIVE_SYSTEMATICS", "TOKEN_CONTAINS", dict(caminho=["ringdown_dephasing_result_v2", "stacks", "3.0", "delta_pred_A"], regra="poder ≈ 0 no ramo canônico", valor=get(core, ["ringdown_dephasing_result_v2", "stacks", "3.0", "delta_pred_A"])),
    "INCONCLUSIVE", True, "V2: INCONCLUSIVE por poder ≈ 0 no ramo canônico (δ_A = %.2e); o 0,2σ (power_B = %.3f) e a sistemática são do ramo B, encerrado" % (get(core, ["ringdown_dephasing_result_v2", "stacks", "3.0", "delta_pred_A"]), get(core, ["ringdown_dephasing_result_v2", "stacks", "3.0", "power_B"])),
    "legível e nulo à sensibilidade disponível: o custo está nele, Γ ~ 10⁻⁴⁰ s⁻¹", False, "FALSIFIED (da correspondência = o pagamento) se o amortecimento desviar da RG a ≥ 5σ com sistemáticas controladas")
cob("F3a", "face", "o relógio local lê a face de K_∂ (o custo local, Γ ∝ β)", "relógios de laboratório: partição P1 (matéria, por partícula)", "dephasing local sob P1; o sujeito é K_∂ (R2 ratificada)", "taxa de dephasing local",
    "clock_test_result_v369", "P1_NOT_FALSIFIED_UNDERPOWERED", "TOKEN_CONTAINS", dict(caminho=["clock_test_result_v369", "P1", "platforms", 0, "deficit_orders_5sigma_detect"], regra="cláusula 3 da Fase 6: poder < 5 → NOT_FALSIFIED_UNDERPOWERED", valor=get(core, ["clock_test_result_v369", "P1", "platforms", 0, "deficit_orders_5sigma_detect"])),
    "NOT_FALSIFIED", True, "UNDERPOWERED: déficit de %.1f ordens para 5σ (⁸⁷Sr, Kim+2025)" % get(core, ["clock_test_result_v369", "P1", "platforms", 0, "deficit_orders_5sigma_detect"]), "a leitura é permitida (ótica); falta instrumento", False, "relógio nuclear com ≥ 12,5 ordens a mais; sem canal novo hoje")
cob("F3b", "face", "o relógio local lê a face de K_∂ (modo de luz por braço)", "relógios de laboratório: partição P2 (modo de luz por braço; LIGO)", "dephasing local sob P2", "ASD prevista vs medida",
    "clock_test_result_v369", "P2_EXCLUDED_IN_READING", "TOKEN_CONTAINS", dict(caminho=["clock_test_result_v369", "P2", "fraction_excluded"], regra="leitura excluída por ordens de grandeza SEM nível de confiança (nσ não entra): EXCLUDED_IN_READING, não FALSIFIED pela cláusula 5", valor=get(core, ["clock_test_result_v369", "P2", "fraction_excluded"])),
    "EXCLUDED_IN_READING", True, "excluída em %.1f%% das séries (168/169) por %.2f ordens de grandeza (mediana, com calibração), sem nσ" % (100 * get(core, ["clock_test_result_v369", "P2", "fraction_excluded"]), get(core, ["clock_test_result_v369", "P2", "orders_excluded_median_with_cal"])),
    "leitura excluída, não canal (convenção final_verdict_reading_v376); a cobrança P2 está EXTINTA", False, "nenhum: a leitura P2 está excluída")
cob("F3c", "face", "o relógio local lê a face de K_∂ (tempo estocástico universal)", "relógios de laboratório: partição P3 (universal)", "dephasing universal (reparametrização)", "nenhuma razão de frequências, largura ou franja o vê",
    "clock_test_result_v369", "P3_NO_LOCAL_OBSERVABLE", "DECLARED", None, "AWAITING", True, "P3: sem observável local (reparametrização comum)", "sem observável: aguarda um observável, não um dado", False, "nenhum canal enquanto não houver observável")
cob("F4", "face", "determinação direta (a face), não ajuste global (reflexos, ao lado)", "neutrino m₂ (JUNO autônomo, 59 d)", "m₂(TGL) = 8,5074 meV; σ do lado da previsão (ratificado)", "z contra Δm²₂₁ medido diretamente",
    "neutrino_m2", "TGL_NU_M2_ARMED_CONSISTENT", "NUMBER_READ", dict(caminho=["neutrino_m2", "values", "escada_datada", 1, "tensao_sigma"], regra="|z| < 5 → NOT_FALSIFIED", valor=get(core, ["neutrino_m2", "values", "escada_datada", 1, "tensao_sigma"])),
    "NOT_FALSIFIED", True, "JUNO 59 d: %.2fσ; NuFIT global pós-JUNO: %.2fσ ao lado (perito = JUNO, R3 ratificada)" % (get(core, ["neutrino_m2", "values", "escada_datada", 1, "tensao_sigma"]), get(core, ["neutrino_m2", "values", "tensao_atual_sigma"])),
    "o que a face lê aqui é o custo por m₂ ∝ β (uma determinação direta), não o peso 1 − β", False, "JUNO com 6 anos: FALSIFIED se |z| ≥ 5 contra m₂(TGL) com σ do lado da previsão")
assert len(C) == 18 and len({c["id"] for c in C}) == 18
assert get(core, ["neutrino_m2", "values", "escada_datada", 1, "fonte"]).startswith("JUNO"), "o índice 1 da escada datada não é o JUNO"
for c in C:   # o desfecho tem de constar no core pela string do veredito (token) ou pelo número lido (regra dita)
    if c["chave_core"]:
        assert c["token"] in V(c["chave_core"]), (c["id"], c["chave_core"], c["token"])
    if c["como"] == "TOKEN_CONTAINS":
        assert c["desfecho_lido"] in c["token"] and (c["desfecho_lido"] != "FALSIFIED" or "NOT_FALSIFIED" not in c["token"]), c["id"]
    if c["como"] == "NUMBER_READ":
        assert isinstance(c["numero"]["valor"], (int, float)), c["id"]
contagem = {o: sum(1 for c in C if c["desfecho_lido"] == o) for o in ("NOT_FALSIFIED", "FALSIFIED", "INCONCLUSIVE", "AWAITING", "EXCLUDED_IN_READING")}
pre = dict(
    identificador=ID, escrito_em=time.strftime("%Y-%m-%d %H:%M:%S"), autor="gerência (Claude Code, sessão 6da8f00d, casa Central de Patentes / Bancada Um)",
    supersede=dict(v1="PREREG_DOIS_LADOS_DELTA_K_20261002_V1", v1_sha256=v1_sha, mapa_seq_v1=275, porque="achados do aferidor independente da v382 (D1, B1, B3, B4, B5, B6, B8, D2, D3), ANTES de qualquer estimador novo ou dado novo; o V1 fica como registro, a correção vai ao lado"),
    um_py=dict(versao=selo.get("um_version"), sha256=um_sha, gate=selo.get("qg_closure_verdict")),
    pedido_do_operador=VERB,
    estimando=dict(
        nome="δ⟨K_∂⟩ — o desvio da expectativa do Hamiltoniano oculto da fronteira em relação à fronteira silenciosa (Λ, w = −1), lido nos DOIS lados",
        regra=dict(beta_tgl=beta, como="ALPHA_FINE_CODATA_2018 × √e em runtime (nunca literal)", alpha=alpha, alpha_lido_de=prov_alpha, theta_M=theta_M, custo_reflexo=beta, pagamento_face=1.0 - beta,
                   soma="|T|² + |R|² = 1 (Teorema S-∂; TheLedgerOfCharges.both_sides_are_read)"),
        onde_incide="a chave do operador: no Hamiltoniano oculto (HminMic = 1 − P_{ℂΩ}, cujo zero é o psion: ThePsionAndTheViscosity.psion_is_the_zero_of_the_hidden_hamiltonian) — a ligação de Spec S(θ) a HminMic é [INPUT/ONTO], sem termo",
        lados=LADOS,
        estatuto="a chave e as seis leituras são [INPUT/ONTO] ratificadas; β e a forma α√e são [DERIVED] do axioma (v376); a identificação de cada observável com δ⟨K_∂⟩ é leitura [ONTO] sem termo no kernel; os desfechos lidos são [REAL] (por token ou por número lido do core, com a regra dita)",
        lambda_fronteira_silenciosa="δ⟨K_∂⟩ = β|1 + w|, zero em w = −1 (ΛCDM = o limite de fronteira silenciosa) [REAL na forma da lei]; «espectro de gradiente negativo» (15/09) fica [ONTO] sem identidade de operador",
    ),
    cegueira=dict(
        sentido="«análise cega» = o estimador não viu o dado antes da regra; NÃO confundir com «nenhum lado é cego», que é a legibilidade dos dois lados (ótica)",
        declaracao="MISTA — dita canal a canal (campo `cego`); hoje NENHUM canal é cego quanto ao dado; R2 é cego quanto ao estimador",
        nao_cego=[c["id"] for c in C if not c["cego"]], cego=[c["id"] for c in C if c["cego"]],
        o_que_nao_foi_feito="nenhum estimador da dissipação na PROPAGAÇÃO (R2) foi construído nem rodou; nenhum dado novo foi olhado nesta sessão",
    ),
    cobrancas=C, contagem=contagem, na_regra=[c["id"] for c in C if c["na_regra"]], ao_lado=[c["id"] for c in C if not c["na_regra"]],
    regras_de_leitura=dict(
        desfechos=["NOT_FALSIFIED", "FALSIFIED", "INCONCLUSIVE", "AWAITING", "EXCLUDED_IN_READING"],
        fonte="PREREGISTRO_FASE6_ECO_RADICAL_20261001.json, regra_de_veredito (cláusulas 1–6 + bandeira), transcrita; mais a convenção EXCLUDED_IN_READING (v369-P2; final_verdict_reading_v376) e o controle de convergência de MCMC",
        falsified="cláusula 5: z_excl = (recuperação − â)/σ_usado ≥ 5 → FALSIFIED (falsifica o PAR leitura×lei, não a teoria), com os controles passando e o poder DITO antes de abrir",
        not_falsified="cláusula 6: senão NOT_FALSIFIED_POWERED; cláusula 3: poder = |recuperação|/σ_usado < 5 → NOT_FALSIFIED_UNDERPOWERED (variante contada em NOT_FALSIFIED com o sufixo dito); NUNCA é CONFIRMED",
        inconclusive="cláusula 2: sistemática relativa > 0,3 → INCONCLUSIVE_SYSTEMATICS; bandeira: controle com |z| ≥ 5 sem detecção → INCONCLUSIVE_SYSTEMATICS; para um estimando por MCMC: controle de convergência (N ≥ 50τ em todos os parâmetros) reprovado → INCONCLUSIVE",
        excluded_in_reading="leitura excluída por ordens de grandeza SEM nível de confiança (nσ não entra): EXCLUDED_IN_READING; a cobrança está extinta; não é FALSIFIED pela cláusula 5 nem NOT_FALSIFIED",
        awaiting="sem estimador selado, sem dado ou sem observável",
        insufficient_sample="cláusula 1: n < 10 → INSUFFICIENT_SAMPLE (não ocorreu em nenhuma cobrança do livro)",
        a_regra_nao_se_move="nenhum desfecho altera β = α√e (TheLedgerOfCharges.the_rule_is_constant_in_the_ledger, rfl: a regra não recebe o livro); «extinguir» é a definição `extinguished`",
        o_gate_nao_se_move="as bandeiras formais são função só do formal — propriedade do código do gate, conferida pelos probes negativos do runtime (qg_closure); cosmologia jamais vira prova matemática",
        os_dois_lados="toda cobrança diz o LADO e o CRITÉRIO de lado; a face lê sem passar pelo reflexo; nenhum lado é cego para o outro (a correção do operador: a TGL é ótica)",
        na_regra_vs_ao_lado="as cobranças AO LADO (R5a–g, o eco como cópia atrasada; R6, o custo posto na face) não são da regra canônica e são contadas em separado",
        sinal="o custo entra com sinal + no reflexo (β > 0); não se inverte sinal depois do dado",
        significancia_global="Šidák sobre as famílias do livro, ao lado da local",
    ),
    fontes=dict(cronometros=prov_cc, covariancia_moresco2020=prov_cov, desi_dr2=prov_dr2),
    codigo_da_bancada_sha256=codigo,
    ordem=["(1) este pré-registro V1.1, selado por hash e inscrito no mapa (corrige 275)", "(2) o um.py v382 lê este arquivo por hash (prove_the_ledger_of_charges_v382) e confere as 18 cobranças contra o core",
           "(3) o estimador da dissipação na propagação (R2): construção, selagem e PODER antes de abrir", "(4) só então o dado; cada canal com o seu desfecho e o seu lado"],
)
b = json.dumps(pre, ensure_ascii=False, indent=1).encode("utf-8")
if os.path.exists(SAIDA_JSON) or os.path.exists(SAIDA_MD):
    raise SystemExit("o pré-registro V1.1 já existe; não se reescreve: " + SAIDA_JSON)
tmp = SAIDA_JSON + ".tmp"; open(tmp, "wb").write(b); assert open(tmp, "rb").read() == b; os.replace(tmp, SAIDA_JSON)
hj = hashlib.sha256(b).hexdigest()
L = ["# PRÉ-REGISTRO DOS DOIS LADOS — V1.1 — o estimando δ⟨K_∂⟩ (02/10/2026)", "",
     "**Identificador** `%s` · escrito em %s · JSON `%s` (sha256 `%s`) · `um.py` %s `%s` · SUPERSEDE o V1 (`%s`, sha256 `%s`, mapa seq 275) antes de qualquer estimador ou dado novo, pelos achados do aferidor da v382; o V1 fica como registro." % (ID, pre["escrito_em"], os.path.basename(SAIDA_JSON), hj, selo.get("um_version"), um_sha[:16], pre["supersede"]["v1"], v1_sha[:16]), "",
     "**A chave do operador (verbatim):** «%s»" % VERB["chave_20261002"], "", "**A correção (verbatim):** «%s»; e a ordem: «%s»" % (VERB["otica_20261002"], VERB["prossiga_20261002"]), "",
     "**O estimando.** %s. A regra: β_TGL = α√e = %.12f (nunca literal); o custo no reflexo é β = |R|² = sin²θ_M; o pagamento na face é 1 − β = |T|² = cos²θ_M; |T|² + |R|² = 1. Onde incide: %s." % (pre["estimando"]["nome"], beta, pre["estimando"]["onde_incide"]), "",
     "**Os dois lados.** REFLEXO: %s. FACE: %s." % (LADOS["reflexo"], LADOS["face"]), "",
     "## As 18 cobranças (UMA por lei de leitura; FALSIFIED possível em todas; nenhuma move a regra)", "",
     "| id | lado | critério de lado | registro | como | desfecho lido | na regra | o que JÁ foi lido (core v381) | qualificador | cego | o que se pré-registra |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for c in C:
    L.append("| %s | %s | %s | %s | %s | `%s` | %s | %s | %s | %s | %s |" % (c["id"], c["lado"], c["criterio_de_lado"], c["registro"], c["como"], c["desfecho_lido"], "sim" if c["na_regra"] else "não (ao lado)", c["ja_lido"], c["qualificador"], "sim" if c["cego"] else "não", c["futuro"]))
L += ["", "Contagem: %s · na regra: %s · ao lado: %s." % (", ".join("%s %d" % kv for kv in contagem.items()), ", ".join(pre["na_regra"]), ", ".join(pre["ao_lado"])), "",
      "## Regras de leitura (as da Fase 6, transcritas)", ""] + ["- **%s:** %s" % (k, v) for k, v in pre["regras_de_leitura"].items() if k != "desfechos"]
L += ["", "## Cegueira", "", "%s. Declaração: %s. Não cego: %s. Cego (quanto ao estimador): %s. %s." % (pre["cegueira"]["sentido"], pre["cegueira"]["declaracao"], ", ".join(pre["cegueira"]["nao_cego"]), ", ".join(pre["cegueira"]["cego"]), pre["cegueira"]["o_que_nao_foi_feito"]), "",
      "## Ordem", ""] + ["%d. %s" % (i + 1, s) for i, s in enumerate(pre["ordem"])]
L += ["", "Estatuto: %s. PROVADA ≠ CONFIRMADA; `NOT_FALSIFIED` nunca é `CONFIRMED`." % pre["estimando"]["estatuto"], ""]
bm = "\n".join(L).encode("utf-8")
tmp = SAIDA_MD + ".tmp"; open(tmp, "wb").write(bm); assert open(tmp, "rb").read() == bm; os.replace(tmp, SAIDA_MD)
hm = hashlib.sha256(bm).hexdigest()
print("pré-registro V1.1 gravado:", SAIDA_JSON, "sha256", hj); print("md:", SAIDA_MD, "sha256", hm); print("contagem:", contagem)
r = subprocess.run([sys.executable, os.path.join(BU, "ferramentas", "publicar_relatorio.py"), "02/10 — Pré-registro dos dois lados V1.1 (18 cobranças; as regras da Fase 6; supersede o V1 antes de qualquer dado)", SAIDA_MD],
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout.strip()); assert r.returncode == 0, r.stderr
out = rotas.registrar([dict(tipo="ERRATA", corrige=275, rota_id="bancada.preregistro_dois_lados_delta_k", status="PROPOSTA", estatuto="REAL", olhou_dados=False,
                            titulo="V1.1 SUPERSEDE o V1 (seq 275) ANTES de qualquer estimador ou dado novo: 18 cobranças (uma por lei de leitura), desfechos por token ou por número lido do core, as regras da Fase 6 transcritas (cláusulas 2/3/5/6 + bandeira), EXCLUDED_IN_READING (P2), R6 INCONCLUSIVE pelo controle de convergência, F2 pelo ramo canônico, R4 com os números do V11, critério de lado por cobrança, cegueira em dois sentidos",
                            resultado="V1.1 gravado (JSON sha256 %s; MD sha256 %s); contagem %s; publicado na aba Relatórios; o V1 (sha256 %s) fica como registro" % (hj[:16], hm[:16], json.dumps(contagem, ensure_ascii=False), v1_sha[:16]),
                            nao_refazer="não usar a regra «poder < 1 → INCONCLUSIVE» do V1 (não é a da Fase 6); não empacotar P1/P2/P3 numa só cobrança; não citar os números do v92 para a chave void_floor_v11",
                            proximo_passo="um.py v382 lê o V1.1 por hash; depois o estimador da propagação (R2) com o poder dito antes de abrir",
                            artefatos=[SAIDA_JSON, SAIDA_MD], refs=["kernel.regra_matriz.cadeia_unica"])],
                      "claude-code/central-de-patentes (gerência) sessão 6da8f00d")
print("inscrito no mapa: seq", out["gravados"][0]["seq"], "cabeça", out["cabeca"][:16])
