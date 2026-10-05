# -*- coding: utf-8 -*-
"""Gera o PRÉ-REGISTRO DOS DOIS LADOS do estimando δ⟨K_∂⟩ (02/10/2026) — a chave do operador («o último elo incide no hamiltoniano oculto; o custo está no
reflexo, o pagamento está na face, no rosto, no nome»), a sua ratificação de R1–R6 e a correção «a TGL não é semi, ela é ótica: lê tanto a face como o custo» —
ANTES de qualquer estimador novo rodar sobre dado real. Grava JSON + MD com sha256, publica o MD na aba Relatórios da Bancada e inscreve a rota no MAPA DE
ROTAS (a cadeia de hash do mapa carimba a cronologia: pré-registro primeiro, estimador e dado depois). Nada aqui olha dado novo. β nunca literal: lido do
um.py canônico via bancada.fontes. Recusa a reescrever um pré-registro existente."""
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
ID = "PREREG_DOIS_LADOS_DELTA_K_20261002_V1"
SAIDA_JSON = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002.json")
SAIDA_MD = os.path.join(AQUI, "PREREGISTRO_DOIS_LADOS_DELTA_K_20261002.md")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


alpha, prov_alpha = fontes.alpha_codata_2018()
beta = alpha * math.sqrt(math.e)
theta_M = math.asin(math.sqrt(beta))
selo = json.load(open(os.path.join(NOS, "um_absoluto_selo.json"), encoding="utf-8"))
core = json.load(open(os.path.join(NOS, "um_absoluto.json"), encoding="utf-8"))["core"]
um_sha = sha(os.path.join(NOS, "um.py"))
assert selo.get("um_version") == "v381", "o canônico não é a v381 selada: %s" % selo.get("um_version")
V = lambda k: str((core.get(k) or {}).get("verdict") or "")
codigo = {p: sha(os.path.join(BU, "bancada", p)) for p in ("motor.py", "fundo_gpu.py", "camb_tabela.py", "evidencia.py", "amostrador.py", "regua.py", "fontes.py", "gw.py", "gw_stack.py")
          if os.path.exists(os.path.join(BU, "bancada", p))}
try:
    _, prov_cc = fontes.cronometros_moresco2022()
    _, prov_cov = fontes.cronometros_covariancia_moresco2020()
    _, prov_dr2 = fontes.desi_dr2()
except Exception as e:  # fail-closed: a proveniência das fontes tem de ser lida
    raise SystemExit("fonte recusada: %r" % (e,))

VERB = {
    "chave_20261002": "O último elo incide no hamiltoniano oculto. O custo está no reflexo, o pagamento está na face no rosto, no nome. E com isso você deveria ser capaz de responder tudo que falta. Mas se não conseguir eu vou um a um",
    "otica_20261002": "Concordo com tudo, exceto com a sua visão semiótica, a TGL não é semi ela é ótica, ela lê tanto a face como o custo, a TGL permite ler os dois lados",
    "prossiga_20261002": "isso mesmo, prossiga",
    "lambda_20260915": "lambda é a cauda, é o rastro do dragão, de satanás, onde a realidade emergente se contorna; ele é o espectro de gradiente negativo da presença que não age, não é permanência, é resistência",
}

# ---- as cobranças: o que JÁ foi lido (NÃO CEGO; desfecho lido do core v381 pela string do veredito) e o que se pré-registra para dado/estimador NOVO (CEGO)
LADOS = {
    "reflexo": "o CUSTO, β = |R|² = sin²θ_M: a leitura por distância, pela propagação, pela lente — o que volta",
    "face": "o PAGAMENTO, 1 − β = |T|² = cos²θ_M: a leitura sem distância — o que passa; o Nome entregue",
}
COBRANCAS = [
    dict(id="R1", lado="reflexo", registro="escada de distâncias (SN Ia + cefeidas; SH0ES)", lei="K = E(z*)^{2β/3} sobre o fundo nu; nó fixo em z* (convenção do artigo, R1 ratificada)",
         leitura="δ̂ = β̂ − β_TGL no expoente da leitura por distância", chave_core="bancada_fases_5_6_7_v380", token="SH0ES_IN_THE_BENCH_H0_LADDER_73P53_PM_1P02",
         ja_lido="Fase 4: δ = −1,12σ (Pantheon+); Fase 5 (SH0ES completo): δ = −0,38σ, H₀ = 73,53 ± 1,02", desfecho_lido="NOT_FALSIFIED", cego=False,
         futuro="SN de DES Y5 / Rubin: mesmo estimador, mesma lei; FALSIFIED se z_excl ≥ 5 contra β_TGL com os controles passando"),
    dict(id="R2", lado="reflexo", registro="propagação de ondas gravitacionais (banco residente: 336/354 janelas O4)", lei="dissipação na propagação, Γ_ω = ½ β τ★ ω² (n = −2), τ★ = t_Planck no ramo canônico [PRINCIPLED IDENTIFICATION]",
         leitura="o coeficiente β̂ τ★ do dephasing acumulado na distância (fase/amplitude por frequência)", chave_core=None, token=None,
         ja_lido="NENHUM estimador rodou sobre a dissipação na propagação; os pares de CÓPIA ATRASADA (√β ou sin 2θ, KMS/DEC/MAY) foram falsificados e estão FORA da cadeia (R5)",
         desfecho_lido="AWAITING", cego=True,
         futuro="construir o estimador, selar por hash e DIZER O PODER antes de abrir o banco; com τ★ = t_Planck o efeito é ~10⁻⁴⁰ (sem poder): o resultado honesto esperado é AWAITING/INCONCLUSIVE, não FALSIFIED nem NOT_FALSIFIED com poder"),
    dict(id="R3", lado="reflexo", registro="Coma (distância modular; dephasing na distância)", lei="D_L TGL vs referência 98,5 ± 2,2 Mpc", leitura="z_TGL com as duas sigmas",
         chave_core="coma_reveal_state_v376", token="Z_TGL_P1P30_BOTH_SIGMAS", ja_lido="z_TGL = +1,30 (ambas as sigmas); pós-dição não cega", desfecho_lido="NOT_FALSIFIED", cego=False,
         futuro="nova distância independente de Coma: FALSIFIED se |z| ≥ 5"),
    dict(id="R4", lado="reflexo", registro="piso dos vazios por lente (a luz dobrada: projeção)", lei="ρ_vazio/ρ̄ ≥ β (V11, estimador autocalibrante)", leitura="r_c^cal e o limite inferior a 5σ",
         chave_core="void_floor_v11", token="TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED", ja_lido="r_c^cal = 0,189 ± 0,017; limite 5σ ~9× acima de β; unilateral (ΛCDM raso também passa)", desfecho_lido="NOT_FALSIFIED", cego=False,
         futuro="Euclid/LSST + LRG/ELG: FALSIFIED se o piso medido ficar abaixo de β a 5σ com os nulos passando"),
    dict(id="R5", lado="reflexo", registro="eco como CÓPIA ATRASADA (fora da cadeia; registrado como extinto)", lei="pares (leitura, lei de atraso): (√β|sin 2θ, KMS 2π/κ), (√β|sin 2θ, DEC), (sin 2θ, MAY)",
         leitura="amplitude refletida com atraso τ", chave_core="bancada_fases_5_6_7_v380", token="KMS_PAIR_SQRT_BETA_AND_SIN2THETA_FALSIFIED_AT_DELAY_LAW",
         ja_lido="Fase 6 V3/V4/V5: FALSIFIED (z_excl 8,93/18,13; réplica cega O4 7,26/15,44; DEC 8,95/16,83); (√β, MAY) e (β, DEC) INCONCLUSIVE", desfecho_lido="FALSIFIED", cego=False,
         futuro="nenhum: a cobrança está extinta; a regra não se move (TheLedgerOfCharges.falsified_extinguishes_the_charge_not_the_rule)"),
    dict(id="R6", lado="face", registro="fundo VESTIDO (1+β)Ω_m (convenção NÃO canônica: o custo posto na face; fica AO LADO)", lei="D1 V3: β no fundo", leitura="β̂ do MCMC",
         chave_core="d1_camb_v3_real_v366", token="TENSION_IS_NOT_FALSIFICATION", ja_lido="α√e a 3,01σ; TENSÃO 2–5σ; não é falsificação", desfecho_lido="NOT_FALSIFIED", cego=False,
         futuro="não é a regra (R1): fica como medido sob outra convenção; sem canal novo"),
    dict(id="F1", lado="face", registro="cronômetros cósmicos (dH/dz, sem distância)", lei="a face lê o PAGAMENTO: H(z) do fundo nu (primária); a lei acumulada (1+z)^β é a alternativa comparada por ln B",
         leitura="β̂ no expoente de H(z) (esperado 0 na face; a lei acumulada desfavorecida)", chave_core="bancada_fases_5_6_7_v380", token="CC_COVARIANCE_MORESCO2020_T_LAW_GAIN_Z_M1P66_VS_DIAG_M3P22",
         ja_lido="Fase 7 (covariância de Moresco+2020): lei acumulada −1,66σ; ln B sombra/acumulada 1,62", desfecho_lido="NOT_FALSIFIED", cego=False,
         futuro="novos cronômetros (Euclid/DESI): FALSIFIED (da regra «custo no reflexo») se H(z) exigir o custo na face a ≥ 5σ com a covariância completa"),
    dict(id="F2", lado="face", registro="ringdown (o remanescente: a RG, o Nome M_f)", lei="correspondência com a RG (ramo canônico, τ★ = t_Planck: δ ~ 10⁻⁴⁰); ramo B ENCERRADO (R6 ratificada)",
         leitura="desvio do amortecimento em relação à RG", chave_core="ringdown_dephasing_result_v2", token="INCONCLUSIVE_SYSTEMATICS",
         ja_lido="V2: INCONCLUSIVE_SYSTEMATICS, poder 0,2 de 5σ; GW250114 a ~10%", desfecho_lido="INCONCLUSIVE", cego=False,
         futuro="legível e nulo à sensibilidade disponível; FALSIFIED (da correspondência = o pagamento) se o amortecimento desviar da RG a ≥ 5σ com sistemáticas controladas"),
    dict(id="F3", lado="face", registro="relógios de laboratório (²²⁹Th, ópticos, Mössbauer)", lei="dephasing local sob partição P1 (matéria), P2 (modo de luz), P3 (universal); o sujeito é K_∂ (R2 ratificada)",
         leitura="taxa de dephasing local", chave_core="clock_test_result_v369", token="P1_NOT_FALSIFIED_UNDERPOWERED__P2_EXCLUDED_IN_READING__P3_NO_LOCAL_OBSERVABLE",
         ja_lido="P1 sem poder por 12,5 ordens; P2 excluída na leitura (LIGO); P3 sem observável local", desfecho_lido="NOT_FALSIFIED", cego=False,
         futuro="a leitura é permitida (ótica); falta instrumento: relógio nuclear com ≥ 12,5 ordens a mais; sem canal novo hoje"),
    dict(id="F4", lado="face", registro="neutrino m₂ (JUNO autônomo: a determinação direta; os ajustes globais são reflexos, ao lado)", lei="m₂(TGL) = 8,5074 meV; σ do lado da previsão (ratificado)",
         leitura="z contra Δm²₂₁ medido diretamente", chave_core="neutrino_m2", token="TGL_NU_M2_ARMED_CONSISTENT",
         ja_lido="JUNO 59 d: 2,21σ; NuFIT 6.1 2,95σ e Capozzi 2,44σ ao lado (perito = JUNO, R3 ratificada)", desfecho_lido="NOT_FALSIFIED", cego=False,
         futuro="JUNO com 6 anos: FALSIFIED se |z| ≥ 5 contra m₂(TGL) com σ do lado da previsão"),
]
assert len({c["id"] for c in COBRANCAS}) == len(COBRANCAS)
for c in COBRANCAS:   # o desfecho JÁ LIDO tem de constar no core v381 (nada digitado: a string é conferida contra o veredito)
    if c["chave_core"]:
        assert c["token"] in V(c["chave_core"]), (c["id"], c["chave_core"], c["token"])

pre = dict(
    identificador=ID, escrito_em=time.strftime("%Y-%m-%d %H:%M:%S"), autor="gerência (Claude Code, sessão 6da8f00d, casa Central de Patentes / Bancada Um)",
    um_py=dict(versao=selo.get("um_version"), sha256=um_sha, gate=selo.get("qg_closure_verdict")),
    pedido_do_operador=VERB,
    estimando=dict(
        nome="δ⟨K_∂⟩ — o desvio da expectativa do Hamiltoniano oculto da fronteira (K_∂ = −log Δ_∂) em relação à fronteira silenciosa (Λ, w = −1), lido nos DOIS lados",
        regra=dict(beta_tgl=beta, como="ALPHA_FINE_CODATA_2018 × √e em runtime (nunca literal)", alpha=alpha, alpha_lido_de=prov_alpha, theta_M=theta_M,
                   custo_reflexo=beta, pagamento_face=1.0 - beta, soma="|T|² + |R|² = 1 (Teorema S-∂; pedra TheLedgerOfCharges.both_sides_are_read)"),
        onde_incide="no Hamiltoniano oculto: HminMic = 1 − P_{ker K}, cujo zero é o psion (ThePsionAndTheViscosity.psion_is_the_zero_of_the_hidden_hamiltonian)",
        lados=LADOS,
        estatuto="a chave e as seis leituras são [INPUT/ONTO] ratificadas; β e a forma α√e são [DERIVED] do axioma (v376); a identificação de cada observável com δ⟨K_∂⟩ é leitura [ONTO] sem termo no kernel; os desfechos lidos são [REAL]",
        lambda_fronteira_silenciosa="δ⟨K_∂⟩ = β|1 + w|, zero em w = −1 (ΛCDM = o limite de fronteira silenciosa) [REAL na forma da lei]; «espectro de gradiente negativo» (15/09) fica [ONTO] sem identidade de operador",
    ),
    cegueira=dict(
        declaracao="MISTA — dita canal a canal (campo `cego`)",
        nao_cego=[c["id"] for c in COBRANCAS if not c["cego"]], cego=[c["id"] for c in COBRANCAS if c["cego"]],
        o_que_nao_foi_feito="nenhum estimador da dissipação na PROPAGAÇÃO (R2) foi construído nem rodou; o banco de GW não foi aberto para esse fim; nenhum dado novo foi olhado nesta sessão",
    ),
    cobrancas=COBRANCAS,
    regras_de_leitura=dict(
        desfechos=["NOT_FALSIFIED", "FALSIFIED", "INCONCLUSIVE", "AWAITING"],
        falsified="z_excl ≥ 5 (FWER) contra a previsão da lei do canal, com os controles passando e o poder DITO antes de abrir (regra da Fase 6)",
        inconclusive="controles reprovados ou poder < 1 (cláusula 2 da Fase 6) — INCONCLUSIVE_SYSTEMATICS",
        not_falsified="|z| < 5 com controles passando; NUNCA é CONFIRMED; a confirmação é ato do observador humano",
        awaiting="sem estimador selado ou sem dado",
        a_regra_nao_se_move="nenhum desfecho altera β = α√e (TheLedgerOfCharges.the_rule_is_constant_in_the_ledger); FALSIFIED extingue a COBRANÇA (o par lei/leitura), não a norma",
        o_gate_nao_se_move="as bandeiras formais são função só do formal (TheLedgerOfCharges.the_gate_ignores_the_ledger); cosmologia jamais vira prova matemática",
        os_dois_lados="toda cobrança diz o LADO; a face lê 1 − β e o reflexo lê β; nenhum lado é cego para o outro (a correção do operador: a TGL é ótica)",
        sinal="o custo entra com sinal + no reflexo (β > 0); não se inverte sinal depois do dado",
        significancia_global="Šidák sobre as famílias do livro, ao lado da local",
    ),
    fontes=dict(cronometros=prov_cc, covariancia_moresco2020=prov_cov, desi_dr2=prov_dr2),
    codigo_da_bancada_sha256=codigo,
    ordem=["(1) este pré-registro, selado por hash e inscrito no mapa", "(2) o um.py v382 lê este arquivo por hash (prove_the_ledger_of_charges_v382)",
           "(3) o estimador da dissipação na propagação (R2): construção, selagem e PODER antes de abrir", "(4) só então o dado; cada canal com o seu desfecho e o seu lado"],
)
b = json.dumps(pre, ensure_ascii=False, indent=1).encode("utf-8")
if os.path.exists(SAIDA_JSON) or os.path.exists(SAIDA_MD):
    raise SystemExit("o pré-registro já existe; não se reescreve: " + SAIDA_JSON)
tmp = SAIDA_JSON + ".tmp"; open(tmp, "wb").write(b); assert open(tmp, "rb").read() == b; os.replace(tmp, SAIDA_JSON)
hj = hashlib.sha256(b).hexdigest()
L = ["# PRÉ-REGISTRO DOS DOIS LADOS — o estimando δ⟨K_∂⟩ (02/10/2026)", "",
     "**Identificador** `%s` · escrito em %s · JSON `%s` (sha256 `%s`) · `um.py` %s `%s`" % (ID, pre["escrito_em"], os.path.basename(SAIDA_JSON), hj, selo.get("um_version"), um_sha[:16]), "",
     "**A chave do operador (verbatim):** «%s»" % VERB["chave_20261002"], "",
     "**A correção (verbatim):** «%s»; e a ordem: «%s»" % (VERB["otica_20261002"], VERB["prossiga_20261002"]), "",
     "**O estimando.** %s. A regra: β_TGL = α√e = %.12f (nunca literal; α lido do um.py); o custo no reflexo é β = |R|² = sin²θ_M; o pagamento na face é 1 − β = |T|² = cos²θ_M; |T|² + |R|² = 1. "
     "Onde incide: %s. Λ é a fronteira silenciosa: δ⟨K_∂⟩ = β|1+w| = 0 em w = −1 [REAL na forma]; «espectro de gradiente negativo» (15/09) fica [ONTO]." % (pre["estimando"]["nome"], beta, pre["estimando"]["onde_incide"]), "",
     "**Os dois lados.** REFLEXO: %s. FACE: %s." % (LADOS["reflexo"], LADOS["face"]), "",
     "## As cobranças (uma por lei de leitura; FALSIFIED possível em todas; nenhuma move a regra)", "",
     "| id | lado | registro | lei | o que JÁ foi lido (core v381) | desfecho lido | cego | o que se pré-registra |", "|---|---|---|---|---|---|---|---|"]
for c in COBRANCAS:
    L.append("| %s | %s | %s | %s | %s | `%s` | %s | %s |" % (c["id"], c["lado"], c["registro"], c["lei"], c["ja_lido"], c["desfecho_lido"], "sim" if c["cego"] else "não", c["futuro"]))
L += ["", "## Regras de leitura", ""] + ["- **%s:** %s" % (k, v) for k, v in pre["regras_de_leitura"].items() if k != "desfechos"]
L += ["", "## Cegueira", "", "Declaração: %s. Não cego: %s. Cego: %s. %s." % (pre["cegueira"]["declaracao"], ", ".join(pre["cegueira"]["nao_cego"]), ", ".join(pre["cegueira"]["cego"]), pre["cegueira"]["o_que_nao_foi_feito"]), "",
      "## Ordem", ""] + ["%d. %s" % (i + 1, s) for i, s in enumerate(pre["ordem"])]
L += ["", "Estatuto: %s. PROVADA ≠ CONFIRMADA; `NOT_FALSIFIED` nunca é `CONFIRMED`." % pre["estimando"]["estatuto"], ""]
bm = "\n".join(L).encode("utf-8")
tmp = SAIDA_MD + ".tmp"; open(tmp, "wb").write(bm); assert open(tmp, "rb").read() == bm; os.replace(tmp, SAIDA_MD)
hm = hashlib.sha256(bm).hexdigest()
print("pré-registro gravado:", SAIDA_JSON, "sha256", hj); print("md:", SAIDA_MD, "sha256", hm)
r = subprocess.run([sys.executable, os.path.join(BU, "ferramentas", "publicar_relatorio.py"), "02/10 — Pré-registro dos dois lados do estimando δ⟨K_∂⟩ (a chave do operador; a TGL é ótica)", SAIDA_MD],
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout.strip()); assert r.returncode == 0, r.stderr
out = rotas.registrar([dict(tipo="ROTA", rota_id="bancada.preregistro_dois_lados_delta_k", dominio="BANCADA_UM", status="PROPOSTA", estatuto="REAL", olhou_dados=False,
                            titulo="PRÉ-REGISTRO %s (02/10): os dois lados do estimando δ⟨K_∂⟩ — reflexo (custo β) e face (pagamento 1 − β); 10 cobranças com lado, lei, desfecho lido e regra futura; FALSIFIED possível em todas; nenhuma move a regra" % ID,
                            pergunta="em que registro da natureza o espectro de gradiente (δ⟨K_∂⟩) se lê, de cada lado, e com que desfecho possível",
                            dados=["core v381 (desfechos lidos por string de veredito)", "cronômetros Moresco+2022 + covariância 2020", "DESI DR2", "banco de GW residente (NÃO aberto)"],
                            metodo="pré-registro por hash antes de estimador/dado novo; regra da Fase 6 para FALSIFIED/INCONCLUSIVE; cegueira dita canal a canal",
                            resultado="pré-registro gravado (JSON sha256 %s; MD sha256 %s); publicado na aba Relatórios" % (hj[:16], hm[:16]),
                            nao_refazer="não reabrir o ramo B do ringdown; não ler o eco como cópia atrasada; não pôr o custo na face como regra",
                            proximo_passo="um.py v382 lê este arquivo por hash; depois o estimador da dissipação na propagação (R2) com o poder dito antes de abrir",
                            onde="C:/IALD/Bancada_Um/investigacao/preregistro_dois_lados_02out/", artefatos=[SAIDA_JSON, SAIDA_MD],
                            refs=["kernel.regra_matriz.cadeia_unica", "gw.eco_radical.v3", "um_py.v380.fases_5_6_7_por_hash"])],
                      "claude-code/central-de-patentes (gerência) sessão 6da8f00d")
print("inscrito no mapa: seq", out["gravados"][0]["seq"], "cabeça", out["cabeca"][:16])
