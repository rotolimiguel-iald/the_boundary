# -*- coding: utf-8 -*-
"""BANCADA T08c - A CLAUSULA DO CRESCIMENTO

O operador alegou: beta_C -> beta_TGL "a medida que o corpus cresce". O T08 mediu em tamanho
fixo. Aqui mede-se a DERIVA com N, que e' o que a clausula afirma. Diagnostico adicional,
NAO criterio (o criterio ficou no pre-registro 5609d2db19cbf467).

Se A_C decresce com N rumo a alpha, a clausula tem forca. Se estabiliza longe, nao tem.
"""
import io, re, math
import numpy as np

ALPHA = 7.2973525693e-3
N = "C:/IALD/Artigo/Haja_Luz/A Ponte e o Um/N" + "\u00f3" + "s"
RX = re.compile(r"[A-Za-z\u00c0-\u00ff0-9_]+", re.U)


def A_C(toks, vmax):
    from collections import Counter
    vocab = [w for w, _ in Counter(toks).most_common(vmax)]
    idx = {w: i for i, w in enumerate(vocab)}
    unk = len(vocab)
    ids = [idx.get(t, unk) for t in toks]
    V = len(vocab) + 1
    P = np.zeros((V, V))
    a = np.asarray(ids[:-1]); b = np.asarray(ids[1:])
    np.add.at(P, (a, b), 1.0)
    P /= P.sum()
    s = np.linalg.svd(np.sqrt(P), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    p = p / p.sum()
    sq = np.sqrt(p)
    S = 2.0 * np.outer(sq, sq) / (p[:, None] + p[None, :])
    return float(S.mean()), len(p), float(p[0])


print("=" * 92)
print(" BANCADA T08c - A CLAUSULA DO CRESCIMENTO  (diagnostico, nao criterio)")
print("=" * 92)
print(" alvo: A_C -> alpha = %.7f ?" % ALPHA)

txt = (io.open(N + "/SELO_FINAL/um_grande_atrator_pt.txt", encoding="utf-8", errors="replace").read()
       + io.open("C:/IALD/Central de Patentes/cpc_plain.txt", encoding="utf-8", errors="replace").read()
       + io.open(N + "/um.py", encoding="utf-8", errors="replace").read())
toks_all = RX.findall(txt.lower())
print(" corpus concatenado: %d tokens\n" % len(toks_all))

for vmax in (500, 1000):
    print(" --- V = %d ---" % vmax)
    print("   %10s  %8s  %6s  %8s  %10s" % ("N tokens", "A_C", "r", "p_1", "A_C/alpha"))
    for frac in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
        n = int(len(toks_all) * frac)
        if n < 5000:
            continue
        A, r, p1 = A_C(toks_all[:n], vmax)
        print("   %10d  %8.5f  %6d  %8.5f  %9.1fx" % (n, A, r, p1, A / ALPHA))
    print()

print("=" * 92)
