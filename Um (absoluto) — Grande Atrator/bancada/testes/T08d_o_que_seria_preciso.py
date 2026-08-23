# -*- coding: utf-8 -*-
"""BANCADA T08d - O QUE SERIA PRECISO (a pergunta inversa)

Identidade encontrada:  sech(kappa_ij/2) = 2 sqrt(pi pj)/(pi+pj) = MG(pi,pj)/MA(pi,pj).
Por AM-GM isso e' <= 1 sempre, com igualdade sse pi = pj. Logo
     A_C = tau_F(sech(|K|/2)) = media do DEFEITO AM-GM do espectro de Schmidt.

Pergunta inversa: que espectro seria preciso para que A_C = alpha? Se a resposta for
"nenhum espectro realizavel", a refutacao do T08 e' ESTRUTURAL, nao acidente de escolha.
"""
import math
import numpy as np

ALPHA = 7.2973525693e-3

print("=" * 92)
print(" BANCADA T08d - O QUE SERIA PRECISO PARA A_C = alpha")
print("=" * 92)
print(" identidade: sech(k/2) = 2 sqrt(pi pj)/(pi+pj) = MG/MA   (AM-GM: <= 1, = 1 sse pi=pj)")
print(" alvo A_C = alpha = %.7f\n" % ALPHA)


def A_de_espectro(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    p = p / p.sum()
    sq = np.sqrt(p)
    S = 2.0 * np.outer(sq, sq) / (p[:, None] + p[None, :])
    return float(S.mean())


print(" (1) LEI DE POTENCIA p_k ~ k^-s  (Zipf generalizado), r = 1000")
print("   %6s  %10s  %12s" % ("s", "A_C", "A_C/alpha"))
r = 1000
for s in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0):
    p = np.arange(1, r + 1, dtype=float) ** (-s)
    A = A_de_espectro(p)
    print("   %6.1f  %10.6f  %11.1fx" % (s, A, A / ALPHA))

print("\n (2) ESPECTRO EXPONENCIAL p_k ~ e^{-lam k}, r = 1000")
print("   %6s  %10s  %12s  %12s" % ("lambda", "A_C", "A_C/alpha", "log10 p1/pr"))
for lam in (0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0):
    k = np.arange(r, dtype=float)
    p = np.exp(-lam * k)
    A = A_de_espectro(p)
    print("   %6.3f  %10.6f  %11.1fx  %12.1f" % (lam, A, A / ALPHA, lam * (r - 1) / math.log(10)))

print("\n (3) O ESPECTRO EXTREMO: dois niveis com razao R, pesos iguais em numero")
print("   %12s  %10s  %12s" % ("R = p1/p2", "A_C", "A_C/alpha"))
for R in (1e2, 1e3, 7.5e4, 1e6, 1e9):
    p = np.concatenate([np.full(500, R), np.full(500, 1.0)])
    A = A_de_espectro(p)
    print("   %12.2e  %10.6f  %11.1fx" % (R, A, A / ALPHA))

print("\n (4) O LIMITE TEORICO. A_C inclui SEMPRE a diagonal (i=j), onde sech(0)=1 exato.")
for r_ in (10, 100, 1000, 10000):
    print("     r = %6d  ->  contribuicao so' da diagonal = 1/r = %.6f   (alpha = %.6f)  %s"
          % (r_, 1.0 / r_, ALPHA,
             "PISO ACIMA DE ALPHA" if 1.0 / r_ > ALPHA else "diagonal ja' cabe abaixo"))

print("\n" + "=" * 92)
print(" LEITURA: a diagonal sozinha impoe A_C >= 1/r. Para A_C = alpha e' preciso r > 137")
print(" E, ALEM DISSO, que TODOS os pares fora da diagonal tenham defeito AM-GM extremo.")
print(" Espectros de corpus reais (Zipf s~1) tem A_C ~ 0.5. Para chegar a alpha por lei de")
print(" potencia seria preciso s absurdo; por exponencial, decaimento tal que p1/p_r explode.")
print("=" * 92)
