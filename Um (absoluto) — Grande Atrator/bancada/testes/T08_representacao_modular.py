# -*- coding: utf-8 -*-
"""BANCADA T08 - A REPRESENTACAO MODULAR FINITA DO CORPUS

Cadeia do operador (22/08/2026):
  C -> p_C(u,v) -> M_C=sqrt(p_C) -> SVD -> {p_k} -> kappa_ij -> R_J = sqrt(e) sech(|K|/2)
    -> a_C = P_F R_J P_F -> beta_C = tau_F(a_C)

Reducao exata (pre-registrada): sqrt(e) sai da traca, logo
  beta_C = beta_TGL  <=>  A_C := tau_F(sech(|K|/2)) = alpha.
O conteudo falsificavel e exatamente alpha.

beta NUNCA literal. alpha entra SO na comparacao final, jamais na construcao.
"""
import io, os, re, json, math, hashlib, random
import numpy as np

ALPHA = 7.2973525693e-3            # CODATA 2018 - SO para comparar, nunca na construcao
SQRT_E = math.sqrt(math.e)
BETA = ALPHA * SQRT_E

PRE = "PRE_REGISTRO_T08_representacao_modular.md"
H08 = hashlib.sha256(io.open(PRE, "rb").read()).hexdigest()
SAIDA = "T08_representacao_modular.json"
if os.path.exists(SAIDA):
    os.remove(SAIDA)

N = "C:/IALD/Artigo/Haja_Luz/A Ponte e o Um/N" + "\u00f3" + "s"
CORPORA = [
    ("C1", "artigo PT (emitido pelo um.py)", N + "/SELO_FINAL/um_grande_atrator_pt.txt"),
    ("C2", "artigo EN (replica de lingua)",  N + "/SELO_FINAL/um_grande_atrator_en.txt"),
    ("C3", "um.py (o programa como texto)",  N + "/um.py"),
    ("C4", "tgl_kernel .lean concatenados",  "<LEAN>"),
    ("C5", "CPC texto puro (CONTROLE EXT)",  "C:/IALD/Central de Patentes/cpc_plain.txt"),
    ("C6", "manual DJE (CONTROLE EXT)",      "C:/IALD/Central de Patentes/dje_manual.txt"),
]
VOCS = [500, 1000, 2000]           # sensibilidade declarada do corte de vocabulario (T-A)

print("=" * 92)
print(" BANCADA T08 - A REPRESENTACAO MODULAR FINITA DO CORPUS")
print("=" * 92)
print(" pre-registro T08 : %s" % H08)
print(" alvo declarado   : A_C = alpha = %.13f   (beta = alpha*sqrt(e) = %.15f)" % (ALPHA, BETA))
print(" janela declarada : [%.6f , %.6f]" % (ALPHA / 2, 2 * ALPHA))
kstar = 2 * math.acosh(1.0 / ALPHA)
print(" kappa* tal que sech(kappa*/2)=alpha : %.4f  (razao de pesos e^k* = %.3e)"
      % (kstar, math.exp(kstar)))


def ler(caminho):
    if caminho == "<LEAN>":
        buf = []
        for raiz, _, arqs in os.walk(N + "/tgl_kernel"):
            for a in sorted(arqs):
                if a.endswith(".lean"):
                    buf.append(io.open(os.path.join(raiz, a), encoding="utf-8",
                                       errors="replace").read())
        return "\n".join(buf)
    return io.open(caminho, encoding="utf-8", errors="replace").read()


RX = re.compile(r"[A-Za-z\u00c0-\u00ff0-9_]+", re.U)


def tokeniza(txt, modo, vmax):
    if modo == "T-B":
        toks = list(txt)
    else:
        toks = RX.findall(txt.lower())
    if vmax is None:
        vocab = sorted(set(toks))
    else:
        from collections import Counter
        c = Counter(toks)
        vocab = [w for w, _ in c.most_common(vmax)]
    idx = {w: i for i, w in enumerate(vocab)}
    unk = len(vocab)
    return [idx.get(t, unk) for t in toks], len(vocab) + 1


def espectro(ids, V):
    """{p_k} = autovalores de rho_L = M M^T, com M(u,v)=sqrt(p(u,v)). Exato, sem truncar."""
    if len(ids) < 3:
        return None
    P = np.zeros((V, V), dtype=np.float64)
    a = np.asarray(ids[:-1])
    b = np.asarray(ids[1:])
    np.add.at(P, (a, b), 1.0)
    P /= P.sum()                                   # soma p = 1
    M = np.sqrt(P)                                 # ||M||_F^2 = 1
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return p / p.sum()                             # renormaliza o residuo numerico


def sech_media(p, membro):
    """A_C = tau_F(sech(|K|/2)) do membro declarado. sech = 2 sqrt(pi pj)/(pi+pj)."""
    if membro == "F3":
        return 1.0, 1                              # o atomo: kappa=0, sech=1 (trivial, declarado)
    if membro == "F4":
        cum = np.cumsum(p)
        k = int(np.searchsorted(cum, 0.99) + 1)
        q = p[:k]
    else:
        q = p
    m = len(q)
    sq = np.sqrt(q)
    tot = 0.0
    cnt = 0.0
    wtot = 0.0
    CH = 2048
    for i0 in range(0, m, CH):
        i1 = min(i0 + CH, m)
        pi = q[i0:i1][:, None]
        si = sq[i0:i1][:, None]
        S = 2.0 * (si * sq[None, :]) / (pi + q[None, :])
        if membro in ("F2", "F5"):
            mask = np.ones_like(S, dtype=bool)
            if membro == "F2":
                for i in range(i0, i1):
                    mask[i - i0, i] = False
            elif i0 == 0:                          # F5: complemento do atomo (nao ambos = 0)
                mask[0, 0] = False
            tot += float(S[mask].sum())
            cnt += float(mask.sum())
        elif membro == "F6":
            W = si * sq[None, :]
            tot += float((W * S).sum())
            wtot += float(W.sum())
        else:                                      # F1, F4
            tot += float(S.sum())
            cnt += float(S.size)
    if membro == "F6":
        return tot / wtot, m
    return tot / cnt, m


MEMBROS = ["F1", "F2", "F3", "F4", "F5", "F6"]
res = []

for cid, cnome, cpath in CORPORA:
    try:
        txt = ler(cpath)
    except Exception as e:
        print("\n [%s] FALHA DE LEITURA: %s" % (cid, e))
        continue
    print("\n" + "-" * 92)
    print(" [%s] %s   (%d chars)" % (cid, cnome, len(txt)))
    for modo, vmax in [("T-B", None)] + [("T-A", v) for v in VOCS]:
        ids, V = tokeniza(txt, modo, vmax)
        if V > 2600:
            print("   %s V=%d  SALTADO (custo O(V^3)) - reportado, nao silenciado" % (modo, V))
            continue
        p = espectro(ids, V)
        if p is None:
            continue
        rot = "%s%s" % (modo, "" if vmax is None else "/%d" % vmax)
        linha = {"corpus": cid, "tok": rot, "V": V, "r": int(len(p)),
                 "N_tokens": len(ids), "p_max": float(p[0]), "p_min": float(p[-1]),
                 "kappa_max": float(abs(math.log(p[0] / p[-1])))}
        out = []
        for mb in MEMBROS:
            A, m = sech_media(p, mb)
            linha[mb] = float(A)
            out.append("%s=%.5f" % (mb, A))
        # NULO N1: mesmos tokens, ordem embaralhada
        emb = list(ids)
        random.Random(20260822).shuffle(emb)
        pn = espectro(emb, V)
        An, _ = sech_media(pn, "F1")
        linha["N1_F1"] = float(An)
        print("   %-10s V=%-5d r=%-5d kmax=%6.2f  %s | N1(F1)=%.5f"
              % (rot, V, len(p), linha["kappa_max"], "  ".join(out), An))
        res.append(linha)

# ---------------------------------------------------------------- veredito
print("\n" + "=" * 92)
print(" VEREDITO (criterios pre-registrados; CONFIRMED proibido)")
print("=" * 92)
lo, hi = ALPHA / 2, 2 * ALPHA
acertos = {}
for mb in MEMBROS:
    cs = sorted({l["corpus"] for l in res if lo <= l[mb] <= hi})
    acertos[mb] = cs
    print("   %s : dentro de [alpha/2, 2alpha] em %d corpora %s" % (mb, len(cs), cs if cs else ""))
nulo_dentro = any(lo <= l["N1_F1"] <= hi for l in res)
vencedores = [m for m in MEMBROS if len(acertos[m]) >= 3]
if not vencedores:
    ver = "T08_REPROVADO"
elif nulo_dentro:
    ver = "T08_INCONCLUSIVO_CONSTRUCAO"
elif len({tuple(acertos[m]) for m in vencedores}) > 1:
    ver = "T08_INCONCLUSIVO_ARBITRARIEDADE"
else:
    ver = "T08_CORPUS_BETA_FACE_EXISTS"
print("\n   >>> %s <<<" % ver)

if res:
    f1 = [l["F1"] for l in res]
    print("\n   F1 observado: min=%.5f  mediana=%.5f  max=%.5f   (alpha=%.7f)"
          % (min(f1), float(np.median(f1)), max(f1), ALPHA))
    print("   razao mediana(F1)/alpha = %.1fx" % (float(np.median(f1)) / ALPHA))

json.dump({"pre_registro_sha256": H08, "alpha": ALPHA, "beta": BETA,
           "kappa_estrela": kstar, "veredito": ver, "acertos": acertos,
           "nulo_N1_dentro_da_janela": nulo_dentro, "linhas": res},
          io.open(SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("\n gravado: %s" % SAIDA)
