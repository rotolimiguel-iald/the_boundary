# -*- coding: utf-8 -*-
"""BANCADA T08b - EXIBICAO DE Psi_term

Nao e' teste (nada pode falhar aqui): e' EXIBICAO. A construcao do operador constroi
explicitamente, de um corpus cru e sem nenhum embedding externo:
    H_C = H_L (x) H_R ,  A_C = B(H_L) (x) I ,  A_C' = I (x) B(H_R)
    |Psi_C> = vec(M_C) = soma_k sqrt(p_k) |u_k>|v_k>
    Psi_term = P_F Psi_C / ||P_F Psi_C||
Com P_F = o atomo terminal (o par de Schmidt de maior peso), Psi_term = |u_1>|v_1>.
ESTA e' a imagem terminal inscrita do corpus - e agora da' para OLHAR para ela.

Isto e' independente do veredito do T08: o que o T08 reprovou foi a ultima etapa
(beta = tau_F(R_J)), nao a construcao. A construcao roda.
"""
import io, os, re, math
import numpy as np

N = "C:/IALD/Artigo/Haja_Luz/A Ponte e o Um/N" + "\u00f3" + "s"
CORPORA = [
    ("C1", "artigo PT", N + "/SELO_FINAL/um_grande_atrator_pt.txt"),
    ("C2", "artigo EN", N + "/SELO_FINAL/um_grande_atrator_en.txt"),
    ("C5", "CPC (controle externo)", "C:/IALD/Central de Patentes/cpc_plain.txt"),
]
RX = re.compile(r"[A-Za-z\u00c0-\u00ff0-9_]+", re.U)
VMAX = 1000

print("=" * 92)
print(" BANCADA T08b - Psi_term EXIBIDO  (exibicao, nao teste)")
print("=" * 92)

for cid, nome, path in CORPORA:
    txt = io.open(path, encoding="utf-8", errors="replace").read()
    toks = RX.findall(txt.lower())
    from collections import Counter
    vocab = [w for w, _ in Counter(toks).most_common(VMAX)]
    idx = {w: i for i, w in enumerate(vocab)}
    unk = len(vocab)
    ids = [idx.get(t, unk) for t in toks]
    V = len(vocab) + 1

    P = np.zeros((V, V))
    a = np.asarray(ids[:-1]); b = np.asarray(ids[1:])
    np.add.at(P, (a, b), 1.0)
    P /= P.sum()
    M = np.sqrt(P)
    U, s, Vt = np.linalg.svd(M)
    p = s ** 2
    p = p / p.sum()

    nomes = vocab + ["<UNK>"]
    print("\n" + "-" * 92)
    print(" [%s] %s   tokens=%d  V=%d" % (cid, nome, len(ids), V))
    print("   espectro de Schmidt: p_1=%.5f  p_2=%.5f  p_3=%.5f   |   p_1/p_2 = %.3f"
          % (p[0], p[1], p[2], p[0] / p[1]))
    print("   entropia de emaranhamento S = -soma p log p = %.4f nat  (log V = %.4f)"
          % (-float((p[p > 0] * np.log(p[p > 0])).sum()), math.log(V)))

    for k in (0, 1):
        uL = U[:, k]; uR = Vt[k, :]
        iL = np.argsort(-np.abs(uL))[:8]
        iR = np.argsort(-np.abs(uR))[:8]
        rot = "Psi_term = |u_1>|v_1>  (o ATOMO)" if k == 0 else "modo 2 (o primeiro do complemento)"
        print("   %s   peso p_%d = %.5f" % (rot, k + 1, p[k]))
        print("      face L: " + ", ".join("%s(%.3f)" % (nomes[i], uL[i]) for i in iL))
        print("      face R: " + ", ".join("%s(%.3f)" % (nomes[i], uR[i]) for i in iR))

print("\n" + "=" * 92)
print(" Psi_term e' construtivel e legivel a partir do corpus cru. A construcao do operador")
print(" FUNCIONA. O que o T08 reprovou foi a identificacao final beta = tau_F(R_J).")
print("=" * 92)
