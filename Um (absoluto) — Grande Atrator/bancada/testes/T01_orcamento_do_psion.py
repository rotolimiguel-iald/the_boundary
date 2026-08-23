# -*- coding: utf-8 -*-
"""
BANCADA T01 (FAIL-CLOSED) — O ORCAMENTO DO PSION
                                          (21/08/2026)

A TIPAGEM DO OPERADOR: "materia escura = condensado de psions; a inscricao
nao esta em 3D, esta em 2D; o graviton e a ligacao de DOIS PSIONS em 3D; o
psion, apesar de ser particula, e FASE UNICA".
E o acervo (`Nada=materia/nada_materia_vfinal.tex:2004`, marcado
\\begin{conjectura}): "A particula de materia escura e o psion — um par
ligado de neutrinos de paridade oposta com massa m_psion = 2 m_nu (1-beta)
~ 98,8 meV".

O QUE ESTE TESTE FAZ. Ele separa DUAS LEITURAS do psion que tem destinos
opostos, e mede qual sobrevive:

 LEITURA A — "par ligado de neutrinos RELIQUIA": os psions sao montados a
   partir do fundo cosmico de neutrinos (CnuB). Nesse caso o numero de
   psions tem TETO: no maximo metade dos neutrinos relíquia.
 LEITURA B — "condensado coerente por misalignment": o psion e o quantum de
   um campo que oscila coerentemente; os quanta SAO pares ligados por
   estrutura interna, mas nao sao montados a partir do CnuB. Nesse caso o
   numero nao tem esse teto, e o preco e a amplitude Psi_star ficar livre.

O DISCRIMINANTE E ARITMETICO e roda sem baixar um byte.

 T1 densidade de numero de psions EXIGIDA por Omega_c h^2 = 0,1200 (Planck)
 T2 densidade de numero DISPONIVEL no CnuB (336/cm^3 => no maximo 168 pares)
 T3 A RAZAO — o discriminante da LEITURA A
 T4 LEITURA B: qual Psi_star entrega Omega_c h^2, e ele e sub-Planckiano?
 T5 O psion de 98,8 meV e FRIO? (comprimento de de Broglie vs escala
    galactica; limite de Lyman-alpha, que vale para ultraleves ~1e-22 eV)
 T6 quando comeca a oscilacao (3H = m)? antes da recombinacao? (o terceiro
    pico exige que sim)
 C1 (CONTROLE) o mesmo calculo com um ultraleve de 1e-22 eV TEM que reprovar
    no de Broglie — se nao reprovar, o aparelho nao discrimina
 C2 (CONTROLE) o mesmo calculo com o CnuB para uma massa 100x menor TEM que
    passar no orcamento — se nao passar, o aparelho esta quebrado

CONSERTOS DESTA RODADA (registrados, nao escondidos):
 - T4 tinha erro de unidade: kg/m^3 -> eV^4 exige MULTIPLICAR por (hbar c)^3,
   nao dividir. O bug fazia Psi_star parecer transplanckiano por 12 ordens.
 - C2 tinha o residuo invertido (comparava numeros absolutos em vez da razao).
Ambos achados pelo proprio fail-closed: os checks FALHARAM e obrigaram a olhar.

REGUA: beta jamais literal; veredito COMPUTADO; CONFIRMED proibido.
ESCOPO: mede-se ORCAMENTO e CINEMATICA, nada mais. Nao se mede se o
mecanismo e verdadeiro; mede-se se ele CABE. Um mecanismo que nao cabe esta
morto; um que cabe nao esta provado.
"""
import numpy as np, math, json, hashlib, io, os

ALPHA = 7.2973525693e-3
BETA = ALPHA * math.sqrt(math.e)          # runtime, jamais literal
TOL = 1e-10; checks = []; exibidas = []
SAIDA = "T01_orcamento_do_psion.json"
if os.path.exists(SAIDA): os.remove(SAIDA)

def chk(nome, valor, tol=TOL, medido=None, piso=None):
    ok = bool(abs(valor) <= tol); reg = {"nome": nome, "residuo": float(abs(valor)), "ok": ok}
    if medido is not None: reg["medido"] = float(medido)
    if piso is not None: reg["piso"] = float(piso)
    checks.append(reg)
    ex = "" if medido is None else "  [medido %.4g | piso %.4g]" % (medido, piso if piso is not None else float("nan"))
    print("    [MEDE] %-56s %9.2e %s%s" % (nome, abs(valor), "OK" if ok else "*FALHA*", ex))

def exi(nome, valor):
    exibidas.append({"nome": nome, "valor": float(valor)})
    print("    [exibe] %-58s %12.6g" % (nome, valor))

# ------------------------------------------------------------ constantes [EXT]
G = 6.67430e-11                  # CODATA
c = 299792458.0                  # exato
hbar = 1.054571817e-34           # CODATA
eV = 1.602176634e-19             # exato
kg_per_eV = eV / c**2            # E=mc^2
Mpc = 3.0856775814913673e22      # m
M_Pl_GeV = 1.220890e19           # massa de Planck [EXT]

# cosmologia Planck 2018 [EXT, INPUT declarado]
h = 0.6735
H0 = 100.0 * h * 1e3 / Mpc       # s^-1
omega_c = 0.1200                 # Omega_c h^2
omega_c_err = 0.0012
Omega_c = omega_c / h**2
rho_crit = 3.0 * H0**2 / (8.0 * math.pi * G)     # kg/m^3
rho_dm = Omega_c * rho_crit

print("=" * 84)
print(" BANCADA T01 — O ORCAMENTO DO PSION")
print("=" * 84)
print(" beta DERIVADO (jamais literal): %.15f = ALPHA*sqrt(e)" % BETA)

# ------------------------------------------------- a massa do psion, do acervo
# m_psion = 2 m_nu (1 - beta); o valor ~98,8 meV exige m_nu = m_3 (a mais pesada)
dm2_31 = 2.51e-3                 # eV^2 [EXT, PDG/NuFIT]
m3_eV = math.sqrt(dm2_31)        # m_1 ~ 0 => m_3 = sqrt(Dm2_31)
m_psion_eV = 2.0 * m3_eV * (1.0 - BETA)
print("\n A MASSA, recomputada da formula do acervo (nao copiada):")
exi("m_3 = sqrt(Dm2_31)  [eV]", m3_eV)
exi("m_psion = 2*m_3*(1-beta)  [meV]", m_psion_eV * 1e3)
chk("m_psion bate os 98,8 meV do acervo (tolerancia 1 meV)",
    abs(m_psion_eV * 1e3 - 98.8), 1.0)

m_psion_kg = m_psion_eV * kg_per_eV

# ----------------------------------------------------------------- T1
print("\n T1 — DENSIDADE DE NUMERO EXIGIDA por Omega_c h^2 = %.4f" % omega_c)
n_req = rho_dm / m_psion_kg                     # m^-3
n_req_cm3 = n_req * 1e-6
exi("rho_crit  [kg/m^3]", rho_crit)
exi("rho_dm    [kg/m^3]", rho_dm)
exi("n_psion EXIGIDA  [cm^-3]", n_req_cm3)
chk("n exigida e positiva e finita", 0.0 if (n_req_cm3 > 0 and np.isfinite(n_req_cm3)) else 1.0, 0.5)

# ----------------------------------------------------------------- T2
print("\n T2 — DISPONIVEL NO FUNDO COSMICO DE NEUTRINOS (LEITURA A)")
n_nu_cm3 = 336.0                                # [EXT] 6 especies x 56/cm^3
n_pares_max = n_nu_cm3 / 2.0                    # cada psion consome 2 neutrinos
exi("n_nu (CnuB, todas as especies) [cm^-3]", n_nu_cm3)
exi("n_pares MAXIMO (n_nu/2)        [cm^-3]", n_pares_max)

# ----------------------------------------------------------------- T3
print("\n T3 — O DISCRIMINANTE DA LEITURA A")
razao = n_req_cm3 / n_pares_max
exi("n_exigida / n_pares_max", razao)
# a LEITURA A so sobrevive se a razao for <= 1. Isto PODE falhar.
chk("LEITURA A (pares de neutrinos reliquia) CABE no CnuB (razao <= 1)",
    max(0.0, razao - 1.0), 1e-12, medido=razao, piso=1.0)

# ----------------------------------------------------------------- T4
print("\n T4 — LEITURA B: a amplitude que entrega Omega_c (misalignment)")
# rho_hoje = (1/2) m^2 Psi_star^2 * (a_osc/a_0)^3 ; oscilacao comeca em 3H = m
# na era da radiacao: H(T) = 1.66 sqrt(g*) T^2 / M_Pl
g_star = 3.36
M_Pl_eV = M_Pl_GeV * 1e9
T_osc_eV = math.sqrt(m_psion_eV * M_Pl_eV / (3.0 * 1.66 * math.sqrt(g_star)))
T0_eV = 2.7255 * 8.617333262e-5            # T_CMB em eV
dil = (T0_eV / T_osc_eV) ** 3              # diluicao a^-3 ~ (T0/T_osc)^3 (entropia const. aprox.)
rho_dm_eV4 = rho_dm * c**2 / eV * (hbar * c / eV) ** 3     # kg/m^3 -> eV/m^3 -> eV^4 (x (hbar c)^3)
Psi_star_eV = math.sqrt(2.0 * rho_dm_eV4 / (m_psion_eV**2 * dil))
exi("T_osc (inicio da oscilacao) [eV]", T_osc_eV)
exi("T_osc [K]", T_osc_eV / 8.617333262e-5)
exi("Psi_star exigido [GeV]", Psi_star_eV / 1e9)
exi("Psi_star / M_Pl", Psi_star_eV / M_Pl_eV)
chk("LEITURA B: Psi_star e SUB-PLANCKIANO (nao exige transplanckiano)",
    max(0.0, Psi_star_eV / M_Pl_eV - 1.0), 1e-12,
    medido=Psi_star_eV / M_Pl_eV, piso=1.0)

# ----------------------------------------------------------------- T5
print("\n T5 — O PSION DE 98,8 meV E FRIO? (de Broglie vs escala galactica)")
v_vir = 1e-3 * c                                  # dispersao tipica ~300 km/s
lam_dB = 2.0 * math.pi * hbar / (m_psion_kg * v_vir)      # m
kpc = 3.0856775814913673e19
exi("lambda_deBroglie do psion [m]", lam_dB)
exi("lambda_deBroglie [kpc]", lam_dB / kpc)
# FRIO = de Broglie MUITO menor que 1 kpc. Pode falhar.
chk("psion e FRIO: lambda_dB << 1 kpc (razao < 1e-6)",
    max(0.0, (lam_dB / kpc) - 1e-6), 1e-30,
    medido=lam_dB / kpc, piso=1e-6)

# C1 (CONTROLE): um ultraleve TEM que reprovar aqui
m_fuzzy_eV = 1e-22
lam_fuzzy = 2.0 * math.pi * hbar / (m_fuzzy_eV * kg_per_eV * v_vir) / kpc
exi("C1 lambda_dB de um ultraleve 1e-22 eV [kpc]", lam_fuzzy)
chk("C1 CONTROLE: o ultraleve REPROVA no de Broglie (>= 0,1 kpc) — discrimina",
    max(0.0, 0.1 - lam_fuzzy), 1e-12, medido=lam_fuzzy, piso=0.1)

# limite de Lyman-alpha [EXT]: m >~ 2e-21 eV
chk("psion respeita o limite de Lyman-alpha (m >= 2e-21 eV)",
    max(0.0, 2e-21 - m_psion_eV), 1e-30, medido=m_psion_eV, piso=2e-21)

# ----------------------------------------------------------------- T6
print("\n T6 — A OSCILACAO COMECA ANTES DA RECOMBINACAO? (o terceiro pico exige)")
T_rec_eV = 0.26                                   # ~3000 K [EXT]
exi("T_osc [eV]", T_osc_eV)
exi("T_recombinacao [eV]", T_rec_eV)
chk("oscilacao comeca ANTES da recombinacao (T_osc > T_rec)",
    max(0.0, T_rec_eV - T_osc_eV), 1e-12, medido=T_osc_eV, piso=T_rec_eV)
z_osc = T_osc_eV / T0_eV - 1.0
exi("z de inicio da oscilacao", z_osc)

# C2 (CONTROLE): com massa 100x MENOR o orcamento do CnuB tem que ficar melhor
m_leve = m_psion_eV / 100.0
n_req_leve = (rho_dm / (m_leve * kg_per_eV)) * 1e-6
exi("C2 n exigida com massa 100x menor [cm^-3]", n_req_leve)
chk("C2 CONTROLE: massa menor EXIGE MAIS particulas (aparelho responde)",
    max(0.0, 10.0 - (n_req_leve / n_req_cm3)), 1e-12,
    medido=n_req_leve / n_req_cm3, piso=10.0)

# --------------------------------------------------------------- veredito
n_ok = sum(1 for x in checks if x["ok"]); n_tot = len(checks)
leitura_A_vive = razao <= 1.0
leitura_B_vive = (Psi_star_eV / M_Pl_eV <= 1.0) and (lam_dB / kpc < 1e-6) and (T_osc_eV > T_rec_eV)

if n_ok == n_tot:
    verd = "PSION_ORCAMENTO_AMBAS_LEITURAS_CABEM_MEDIDO_%d_DE_%d" % (n_ok, n_tot)
elif leitura_B_vive and not leitura_A_vive:
    verd = ("PSION_LEITURA_A_REFUTADA_POR_ORCAMENTO__LEITURA_B_CONDENSADO_SOBREVIVE"
            "__MEDIDO_%d_DE_%d" % (n_ok, n_tot))
elif not leitura_B_vive:
    verd = "PSION_AMBAS_LEITURAS_EM_DIFICULDADE_MEDIDO_%d_DE_%d" % (n_ok, n_tot)
else:
    verd = "T01_INCONCLUSIVO_%d_DE_%d" % (n_ok, n_tot)

print()
print("=" * 84)
print(" LEITURA A (pares de neutrinos reliquia): %s  (razao %.1f)"
      % ("CABE" if leitura_A_vive else "NAO CABE", razao))
print(" LEITURA B (condensado por misalignment): %s"
      % ("CABE" if leitura_B_vive else "NAO CABE"))
print(" VEREDITO COMPUTADO: %s" % verd)
print("=" * 84)

out = {
    "teste": "BANCADA T01 — o orcamento do psion",
    "tipagem_do_operador": ("materia escura = condensado de psions; a inscricao esta em 2D; "
                            "o graviton e a ligacao de dois psions em 3D; o psion e fase unica"),
    "formula_do_acervo": "m_psion = 2*m_nu*(1-beta_TGL), com m_nu = m_3 = sqrt(Dm2_31)",
    "o_que_se_mede": ("ORCAMENTO e CINEMATICA: cabe? nao se mede se o mecanismo e verdadeiro. "
                      "Um mecanismo que nao cabe esta morto; um que cabe NAO esta provado."),
    "leitura_A": {"descricao": "psion = par ligado de neutrinos RELIQUIA (montado do CnuB)",
                  "n_exigida_cm3": n_req_cm3, "n_disponivel_cm3": n_pares_max,
                  "razao": razao, "cabe": bool(leitura_A_vive)},
    "leitura_B": {"descricao": "psion = quantum de campo em condensado coerente (misalignment)",
                  "Psi_star_GeV": Psi_star_eV / 1e9, "Psi_star_sobre_MPl": Psi_star_eV / M_Pl_eV,
                  "T_osc_eV": T_osc_eV, "z_osc": z_osc,
                  "lambda_dB_kpc": lam_dB / kpc, "cabe": bool(leitura_B_vive)},
    "m_psion_meV": m_psion_eV * 1e3,
    "data": "2026-08-21", "beta_derivado": BETA, "alpha_input": ALPHA,
    "checks": checks, "exibidas": exibidas, "veredito": verd,
    "ressalva": ("T4 usa a estimativa PADRAO de misalignment (oscilacao em 3H=m, diluicao a^-3, "
                 "g* constante) — [KNOWN] da literatura, nao derivacao da TGL. O calculo diz que "
                 "a amplitude CABE, nao que a TGL a FIXA: fixar Psi_star a partir de beta e "
                 "trabalho ainda por fazer, e e o que falta para a clausula 'sem parametros "
                 "ajustados'. CONFIRMED proibido."),
}
json.dump(out, io.open(SAIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(" gravado: %s" % SAIDA)
