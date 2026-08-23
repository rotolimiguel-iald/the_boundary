# -*- coding: utf-8 -*-
"""
BANCADA T07 — A AMPLITUDE DA INSCRICAO, MULTI-DRIVER
  pre-registro T07 (hasheado antes de medir os substratos novos)
  T06 PRESERVADO ao lado; o estimador e IDENTICO, so muda o numero de Drivers.

CINCO SUBSTRATOS, cinco maos que nao se coordenaram:
  M Mathlib (263.297)  I Init (26.813)  S Std (21.930)
  T Batteries (1.959)  A TGLExt+TGL (1.662, o operador)

C1 invariancia max/min <= 2 ; C2 todos em [beta/2, 2beta] ; C3 nulo discrimina
em cada um ; C4 >= 3 substratos qualificados (>=3 pares validos).
CONFIRMED proibido.
"""
import numpy as np, math, json, io, os, hashlib
from collections import defaultdict

ALPHA = 7.2973525693e-3
BETA = ALPHA * math.sqrt(math.e)
M_RESAMPLE, N_MIN, N_NULL = 128, 30, 1000        # herdados do T06, nao tocados

TSV = r"C:/IALD/Artigo/Haja_Luz/A Ponte e o Um/N\u00f3s/tgl_kernel/T06_deps_full.tsv".encode().decode("unicode_escape")
SAIDA = "T07_amplitude_multidriver.json"
if os.path.exists(SAIDA): os.remove(SAIDA)

h06 = hashlib.sha256(io.open("PRE_REGISTRO_T06_amplitude.md", "rb").read()).hexdigest()
h07 = hashlib.sha256(io.open("PRE_REGISTRO_T07_amplitude_multidriver.md", "rb").read()).hexdigest()
print("=" * 88)
print(" BANCADA T07 — A AMPLITUDE DA INSCRICAO, MULTI-DRIVER")
print("=" * 88)
print(" pre-registro T06: %s" % h06)
print(" pre-registro T07: %s" % h07)
print(" beta = %.15f | janela C2 = [%.6f , %.6f]" % (BETA, BETA / 2, 2 * BETA))

SUBS = [("M", "Mathlib",        lambda m: m.startswith("Mathlib")),
        ("I", "Init",           lambda m: m.startswith("Init")),
        ("S", "Std",            lambda m: m.startswith("Std")),
        ("T", "Batteries",      lambda m: m.startswith("Batteries")),
        ("A", "TGLExt+TGL",     lambda m: m.startswith("TGLExt") or m.startswith("TGL."))]

print("\n lendo o grafo ...")
nome_id, nomes, modulos, deps_raw = {}, [], [], []
with io.open(TSV, encoding="utf-8", errors="replace") as f:
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        if len(p) < 2: continue
        nome_id[p[0]] = len(nomes); nomes.append(p[0]); modulos.append(p[1])
        deps_raw.append(p[2].split() if len(p) > 2 and p[2] else [])
N = len(nomes)
adj = [[nome_id[d] for d in ds if d in nome_id] for ds in deps_raw]
del deps_raw
print("   %d teoremas, %d arestas efetivas" % (N, sum(len(a) for a in adj)))

print(" profundidade ...")
depth = np.full(N, -1, dtype=np.int32)
for raiz in range(N):
    if depth[raiz] >= 0: continue
    pilha = [(raiz, 0)]
    while pilha:
        u, fase = pilha.pop()
        if fase == 0:
            if depth[u] >= 0: continue
            depth[u] = -2; pilha.append((u, 1))
            for w in adj[u]:
                if depth[w] == -1: pilha.append((w, 0))
        else:
            m = 0
            for w in adj[u]:
                if depth[w] >= 0: m = max(m, depth[w] + 1)
            depth[u] = m
depth[depth < 0] = 0
grau = np.array([len(a) for a in adj], dtype=np.float64)
print("   max = %d" % depth.max())

def angulos(g):
    g = np.sort(np.asarray(g, float)); lo, hi = g.min(), g.max()
    return None if hi <= lo else 2.0 * math.pi * (g - lo) / (hi - lo)

def reamostra(t, M=M_RESAMPLE):
    return np.interp(np.linspace(0, 1, M), np.linspace(0, 1, len(t)), t)

def deltas(cam):
    out, pares = [], []
    for d in sorted(cam):
        if d + 1 not in cam: continue
        a, b = cam[d], cam[d + 1]
        if len(a) < N_MIN or len(b) < N_MIN: continue
        ta, tb = angulos(grau[a]), angulos(grau[b])
        if ta is None or tb is None: continue
        ra, rb = reamostra(ta), reamostra(tb)
        if ra.std() == 0 or rb.std() == 0: continue
        r = float(np.corrcoef(ra, rb)[0, 1])
        out.append(1.0 - r); pares.append((d, len(a), len(b), r))
    return np.array(out), pares

def camadas(mask):
    m = defaultdict(list)
    for i in np.where(mask)[0]: m[int(depth[i])].append(i)
    return {k: np.array(v) for k, v in m.items()}

print("\n" + "=" * 88); print(" A MEDIDA"); print("=" * 88)
rng = np.random.default_rng(20260821)
res, qualificados = {}, []
for cod, rot, pred in SUBS:
    mask = np.array([pred(m) for m in modulos])
    cam = camadas(mask)
    ds, pares = deltas(cam)
    n_teo = int(mask.sum())
    if len(ds) < 3:
        print("\n %s %-12s %7d teoremas -> %d pares validos  ==> NAO QUALIFICADO (excluido, declarado)"
              % (cod, rot, n_teo, len(ds)))
        res[cod] = {"rotulo": rot, "n_teoremas": n_teo, "n_pares": int(len(ds)),
                    "Delta": float(np.median(ds)) if len(ds) else None, "qualificado": False}
        continue
    D = float(np.median(ds)); qualificados.append(cod)
    # nulo
    tam = [(d, len(v)) for d, v in sorted(cam.items())]
    idx = np.where(mask)[0]; nulos = []
    for _ in range(N_NULL):
        perm = rng.permutation(idx); novo, pos = {}, 0
        for d, t in tam:
            novo[d] = perm[pos:pos + t]; pos += t
        dn, _ = deltas(novo)
        if len(dn): nulos.append(float(np.median(dn)))
    nulos = np.array(nulos); lo, hi = np.percentile(nulos, [2.5, 97.5])
    fora = not (lo <= D <= hi)
    print("\n %s %-12s %7d teoremas | %3d pares | Delta = %.8f | Delta/beta = %.4f"
          % (cod, rot, n_teo, len(ds), D, D / BETA))
    print("     nulo: mediana %.6f  IC95 [%.6f , %.6f]  ->  %s"
          % (np.median(nulos), lo, hi, "FORA (discrimina)" if fora else "DENTRO (*nao discrimina*)"))
    res[cod] = {"rotulo": rot, "n_teoremas": n_teo, "n_pares": int(len(ds)), "Delta": D,
                "Delta_sobre_beta": D / BETA, "qualificado": True,
                "nulo": {"mediana": float(np.median(nulos)), "ic95": [float(lo), float(hi)],
                         "fora_do_ic95": bool(fora)}}

print("\n" + "=" * 88); print(" O CRITERIO PRE-REGISTRADO"); print("=" * 88)
Ds = {c: res[c]["Delta"] for c in qualificados}
C4 = len(qualificados) >= 3
if not C4:
    C1 = C2 = C3 = None; verd = "T07_INCONCLUSIVO_POUCOS_SUBSTRATOS"
    print(" C4 PODER: apenas %d substratos qualificados (exige 3)  *FALHA*" % len(qualificados))
else:
    razao = max(Ds.values()) / min(Ds.values())
    C1 = bool(razao <= 2.0)
    C2 = bool(all(BETA / 2 <= v <= 2 * BETA for v in Ds.values()))
    C3 = bool(all(res[c]["nulo"]["fora_do_ic95"] for c in qualificados))
    print(" C4 PODER        %d substratos qualificados: %s   OK" % (len(qualificados), ", ".join(qualificados)))
    print(" C1 INVARIANCIA  max/min = %.4f  (limite 2,0)   %s" % (razao, "OK" if C1 else "*FALHA*"))
    for c in qualificados:
        print("      %s  Delta = %.8f   Delta/beta = %.4f" % (c, Ds[c], Ds[c] / BETA))
    print(" C2 VALOR        todos em [beta/2, 2beta]?   %s" % ("OK" if C2 else "*FALHA*"))
    print(" C3 NULO         todos fora do IC95?   %s" % ("OK" if C3 else "*FALHA*"))
    if C1 and C2 and C3:  verd = "AMPLITUDE_INVARIANTE_ENTRE_DRIVERS_E_BATE_BETA"
    elif not C3:          verd = "NULO_REPRODUZ__SEM_MEDIDA"
    elif not C1:          verd = "AMPLITUDE_DEPENDE_DO_DRIVER"
    else:                 verd = "AMPLITUDE_INVARIANTE_MAS_NAO_E_BETA"

print("\n" + "=" * 88); print(" VEREDITO COMPUTADO: %s" % verd); print("=" * 88)

json.dump({"teste": "T07 — a amplitude da inscricao, multi-driver",
           "pre_registro_T06_sha256": h06, "pre_registro_T07_sha256": h07,
           "estimador": "IDENTICO ao T06 (M=128, N_MIN=30, mediana, Pearson, grau de saida)",
           "beta_derivado": BETA, "universo": {"n_teoremas": int(N)},
           "resultados": res, "qualificados": qualificados,
           "criterio": {"C1_invariancia": C1, "C2_valor": C2, "C3_nulo": C3, "C4_poder": C4},
           "veredito": verd, "data": "2026-08-21",
           "ressalva": ("mede-se amplitude de inscricao em CORPORA DE PROVA FORMAL. Acordo entre "
                        "Drivers e fato sobre como distincoes matematicas se empilham, NAO sobre o "
                        "mundo. Alternativa banal que fica dita: pode refletir como humanos "
                        "organizam demonstracoes — universal de pratica, nao de natureza; o teste "
                        "NAO separa as duas leituras. Nao decide beta = tau_F(R_J) [CONJECTURE]. "
                        "CONFIRMED proibido.")},
          io.open(SAIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(" gravado: %s" % SAIDA)
