# -*- coding: utf-8 -*-
"""BANCADA T11 - O TEOREMA DA ESCALA: os numeros da Ponte sao reproduziveis?

Polarizacao de vacuo a 1 laco com massa EXATA. alpha(0) e input declarado (CODATA).
Delta_alpha_had entra como [INPUT] de literatura -- declarado no pre-registro.
"""
import io, os, json, math, hashlib
import numpy as np
from scipy.integrate import quad

ALPHA0 = 7.2973525693e-3            # CODATA 2018 [INPUT declarado]
PI = math.pi

PRE = "PRE_REGISTRO_T11_escala.md"
H11 = hashlib.sha256(io.open(PRE, "rb").read()).hexdigest()
SAIDA = "T11_escala.json"
if os.path.exists(SAIDA):
    os.remove(SAIDA)

# massas em GeV (PDG)
M_E = 0.51099895e-3
M_MU = 105.6583755e-3
M_TAU = 1776.86e-3
M_Z = 91.1876
DA_HAD5 = 0.02761                   # [INPUT] literatura; nao calculavel em perturbacao

print("=" * 92)
print(" BANCADA T11 - O TEOREMA DA ESCALA")
print("=" * 92)
print(" pre-registro T11 : %s" % H11)
print(" alpha(0) = %.13e  [INPUT declarado, CODATA 2018]" % ALPHA0)


def dalpha_exact(Q, m, Nc=1.0, Qf=1.0):
    """Delta_alpha de um fermiao, 1 laco, massa exata (Q, m em GeV)."""
    r = (Q / m) ** 2

    def f(x):
        return x * (1 - x) * math.log1p(x * (1 - x) * r)
    I, _ = quad(f, 0.0, 1.0, limit=200)
    return (2.0 * ALPHA0 / PI) * Nc * Qf * Qf * I


def dalpha_uv(Q, m, Nc=1.0, Qf=1.0):
    return (ALPHA0 / (3 * PI)) * Nc * Qf * Qf * (math.log((Q / m) ** 2) - 5.0 / 3.0)


def dalpha_ir(Q, m, Nc=1.0, Qf=1.0):
    return (ALPHA0 / (15 * PI)) * Nc * Qf * Qf * (Q / m) ** 2


# ---------------------------------------------------------------- CONTROLE
print("\n" + "-" * 92)
print(" CONTROLE OBRIGATORIO - a formula exata bate os DOIS limites conhecidos?")
print("-" * 92)
print("   %-16s %14s %14s %12s" % ("regime", "exata", "assintotica", "razao"))
ok_ctrl = True
for nome, Q, ref in [("UV: Q=1e4*m_e", 1e4 * M_E, dalpha_uv(1e4 * M_E, M_E)),
                     ("UV: Q=1e6*m_e", 1e6 * M_E, dalpha_uv(1e6 * M_E, M_E)),
                     ("IR: Q=1e-2*m_e", 1e-2 * M_E, dalpha_ir(1e-2 * M_E, M_E)),
                     ("IR: Q=1e-3*m_e", 1e-3 * M_E, dalpha_ir(1e-3 * M_E, M_E))]:
    ex = dalpha_exact(Q, M_E)
    r = ex / ref
    if abs(r - 1) > 1e-3:
        ok_ctrl = False
    print("   %-16s %14.6e %14.6e %12.6f" % (nome, ex, ref, r))
print("\n   >>> CONTROLE: %s" % ("PASSOU (<0,1%)" if ok_ctrl else "FALHOU -- teste abortado"))
if not ok_ctrl:
    raise SystemExit(1)

# ---------------------------------------------------------------- A-1 e A-4
print("\n" + "-" * 92)
print(" A-1 e A-4 - o PLATO IR em Q = 1 keV")
print("-" * 92)
Q1 = 1e-6                                    # 1 keV em GeV
da_e = dalpha_exact(Q1, M_E)
da_mu = dalpha_exact(Q1, M_MU)
da_tau = dalpha_exact(Q1, M_TAU)
da_tot = da_e + da_mu + da_tau
ir_approx = dalpha_ir(Q1, M_E)
print("   Delta_alpha(1 keV) por especie:")
print("     eletron : %.6e" % da_e)
print("     muon    : %.6e" % da_mu)
print("     tau     : %.6e" % da_tau)
print("     TOTAL   : %.6e   <- o 'plato IR' do artigo" % da_tot)
print("   aproximacao (alpha/15pi)(Q/m_e)^2 = %.6e   razao exata/aprox = %.6f"
      % (ir_approx, da_e / ir_approx))
A1 = bool(5.0e-10 <= da_tot <= 7.0e-10)
A4 = bool(abs(da_e / ir_approx - 1) < 0.01)
print("   artigo afirma: 5,9e-10")
print("   >>> A-1 (plato em [5,0e-10 , 7,0e-10]) : %s" % ("PASSA" if A1 else "FALHA"))
print("   >>> A-4 (exata/assintotica a <1%%)     : %s" % ("PASSA" if A4 else "FALHA"))

# ---------------------------------------------------------------- A-2 e A-3
print("\n" + "-" * 92)
print(" A-2 e A-3 - alpha(M_Z)")
print("-" * 92)
dl_e = dalpha_exact(M_Z, M_E)
dl_mu = dalpha_exact(M_Z, M_MU)
dl_tau = dalpha_exact(M_Z, M_TAU)
dl = dl_e + dl_mu + dl_tau
print("   Delta_alpha_lep(M_Z):")
print("     eletron : %.6f" % dl_e)
print("     muon    : %.6f" % dl_mu)
print("     tau     : %.6f" % dl_tau)
print("     TOTAL   : %.6f    (literatura: 0,031498)" % dl)
print("   Delta_alpha_had^(5)  : %.6f    [INPUT literatura, declarado]" % DA_HAD5)
dtot = dl + DA_HAD5
aZ = ALPHA0 / (1.0 - dtot)
inv = 1.0 / aZ
corrida = aZ / ALPHA0 - 1.0
print("   Delta_alpha TOTAL    : %.6f" % dtot)
print("   1/alpha(M_Z)         : %.4f     (artigo: 129,0 ; literatura: 128,9)" % inv)
print("   corrida alpha(M_Z)/alpha(0)-1 : %.4f%%   (artigo: 6,2%%)" % (100 * corrida))
A2 = bool(128.5 <= inv <= 129.5)
A3 = bool(0.055 <= corrida <= 0.070)
print("   >>> A-2 (1/alpha em [128,5 , 129,5]) : %s" % ("PASSA" if A2 else "FALHA"))
print("   >>> A-3 (corrida em [5,5%% , 7,0%%])   : %s" % ("PASSA" if A3 else "FALHA"))

# ---------------------------------------------------------------- veredito
print("\n" + "=" * 92)
print(" VEREDITO")
print("=" * 92)
todos = A1 and A2 and A3 and A4
print("   A-1 plato IR      : %s" % ("PASSA" if A1 else "FALHA"))
print("   A-2 1/alpha(M_Z)  : %s" % ("PASSA" if A2 else "FALHA"))
print("   A-3 corrida 6,2%%  : %s" % ("PASSA" if A3 else "FALHA"))
print("   A-4 limite IR     : %s" % ("PASSA" if A4 else "FALHA"))
ver = "T11_NUMEROS_DA_PONTE_REPRODUZIDOS" if todos else "T11_DIVERGENCIA"
print("\n   >>> %s <<<" % ver)
print("\n   E O QUE ISTO NAO DIZ (pre-registrado): nada sobre a TGL. A-2 usa Delta_alpha_had")
print("   como INPUT de literatura -- e' REPRODUCAO de QED, jamais confirmacao da teoria.")
print("   O conteudo proprio do Teorema da Escala e' ESTRUTURAL e nao e' testado aqui.")

json.dump({"pre_registro_sha256": H11, "alpha0": ALPHA0,
           "controle_limites_ok": ok_ctrl,
           "A1_plato_IR_1keV": da_tot, "A1_passa": A1,
           "A4_razao_exata_sobre_assintotica": da_e / ir_approx, "A4_passa": A4,
           "dalpha_lep_MZ": dl, "dalpha_had5_INPUT": DA_HAD5,
           "A2_inv_alpha_MZ": inv, "A2_passa": A2,
           "A3_corrida": corrida, "A3_passa": A3, "veredito": ver},
          io.open(SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n gravado: %s" % SAIDA)
