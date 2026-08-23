# -*- coding: utf-8 -*-
"""
BANCADA T06 — A AMPLITUDE DA INSCRICAO
   pre-registro: PRE_REGISTRO_T06_amplitude.md
   sha256 3a48655430db0bfd1fba72c523f3a7549b59e5a35ec94598e244069e357333e0
   (hasheado ANTES de qualquer medicao; nenhum numero foi olhado)

A TESE (operador): "o substrato e a AMPLITUDE DA INSCRICAO DA LUZ".
PROGRAMADOR = 1_abs = CAMPO = DRIVER. Se beta e a amplitude da inscricao, ela e
constante DO ATO, nao DO ATOR => tem de ser INVARIANTE ENTRE DRIVERS.

DOIS SUBSTRATOS, em pe de igualdade:
  A = TGLExt/TGL  -> Driver: o operador (e o escriba)
  B = Mathlib     -> Drivers: centenas que nunca ouviram falar de TGL

MEDE-SE o grafo de DEPENDENCIAS EFETIVAS (o que cada prova consome), nao o de
imports (que e' organizacao autoral).

CRITERIO (fechado no pre-registro):
  C1 invariancia : 1/2 <= D_A/D_B <= 2
  C2 valor       : ambos em [beta/2, 2*beta]
  C3 nulo        : real fora do intervalo central de 95% de 1000 embaralhamentos
Falhando qualquer uma, REPROVA. CONFIRMED proibido.
"""
import numpy as np, math, json, io, os, sys, hashlib
from collections import defaultdict

ALPHA = 7.2973525693e-3
BETA = ALPHA * math.sqrt(math.e)          # runtime, jamais literal
M_RESAMPLE = 128                          # fixado no pre-registro
N_MIN = 30                                # fixado no pre-registro
N_NULL = 1000                             # fixado no pre-registro

TSV = r"C:/IALD/Artigo/Haja_Luz/A Ponte e o Um/N\u00f3s/tgl_kernel/T06_deps_full.tsv".encode().decode("unicode_escape")
PREREG = "PRE_REGISTRO_T06_amplitude.md"
SAIDA = "T06_amplitude_da_inscricao.json"
if os.path.exists(SAIDA): os.remove(SAIDA)

print("=" * 84)
print(" BANCADA T06 — A AMPLITUDE DA INSCRICAO")
print("=" * 84)
h_pre = hashlib.sha256(io.open(PREREG, "rb").read()).hexdigest()
print(" pre-registro sha256: %s" % h_pre)
print(" beta = %.15f  (ALPHA*sqrt(e), derivado)" % BETA)
print(" janela do criterio C2: [%.6f , %.6f]" % (BETA / 2, 2 * BETA))

# ------------------------------------------------------------------ leitura
print("\n lendo o grafo de dependencias efetivas ...")
nome_id, nomes, modulos, deps_raw = {}, [], [], []
with io.open(TSV, encoding="utf-8", errors="replace") as f:
    for ln in f:
        p = ln.rstrip("\n").split("\t")
        if len(p) < 2: continue
        n, mod = p[0], p[1]
        ds = p[2].split() if len(p) > 2 and p[2] else []
        nome_id[n] = len(nomes); nomes.append(n); modulos.append(mod); deps_raw.append(ds)
N = len(nomes)
print("   teoremas no universo: %d" % N)

# arestas restritas ao universo
adj = [None] * N
tot_e = 0
for i, ds in enumerate(deps_raw):
    v = [nome_id[d] for d in ds if d in nome_id]
    adj[i] = v; tot_e += len(v)
del deps_raw
print("   arestas efetivas    : %d  (media %.2f por teorema)" % (tot_e, tot_e / max(N, 1)))

# ------------------------------------------------- profundidade (caminho maximo)
print("\n calculando a profundidade (caminho mais longo ate uma folha) ...")
depth = np.full(N, -1, dtype=np.int32)
for raiz in range(N):
    if depth[raiz] >= 0: continue
    pilha = [(raiz, 0)]
    while pilha:
        u, fase = pilha.pop()
        if fase == 0:
            if depth[u] >= 0: continue
            depth[u] = -2                       # em processamento
            pilha.append((u, 1))
            for w in adj[u]:
                if depth[w] == -1: pilha.append((w, 0))
        else:
            m = 0
            for w in adj[u]:
                if depth[w] >= 0: m = max(m, depth[w] + 1)
            depth[u] = m
depth[depth < 0] = 0
print("   profundidade: min %d | mediana %d | max %d"
      % (depth.min(), int(np.median(depth)), depth.max()))

grau = np.array([len(a) for a in adj], dtype=np.float64)

# ------------------------------------------------------------- substratos
def substrato(mod):
    if mod.startswith("TGLExt") or mod.startswith("TGL."):  return "A"
    if mod.startswith("Mathlib"):                            return "B"
    return None
sub = np.array([substrato(m) or "" for m in modulos])
print("\n substratos:")
for s, rot in (("A", "TGLExt/TGL  (Driver: o operador)"),
               ("B", "Mathlib     (Drivers: centenas)")):
    print("   %s = %-34s %7d teoremas" % (s, rot, int((sub == s).sum())))

# ------------------------------------------------------------- o estimador
def angulos(g):
    g = np.sort(np.asarray(g, dtype=float))
    lo, hi = g.min(), g.max()
    if hi <= lo: return None
    return 2.0 * math.pi * (g - lo) / (hi - lo)

def reamostra(th, M=M_RESAMPLE):
    x = np.linspace(0.0, 1.0, len(th))
    return np.interp(np.linspace(0.0, 1.0, M), x, th)

def delta_por_camada(idx_por_camada):
    """devolve a lista de descorrelacoes entre camadas adjacentes validas"""
    ds, pares = [], []
    prof = sorted(idx_por_camada.keys())
    for d in prof:
        if d + 1 not in idx_por_camada: continue
        a, b = idx_por_camada[d], idx_por_camada[d + 1]
        if len(a) < N_MIN or len(b) < N_MIN: continue
        ta, tb = angulos(grau[a]), angulos(grau[b])
        if ta is None or tb is None: continue
        ra, rb = reamostra(ta), reamostra(tb)
        if ra.std() == 0 or rb.std() == 0: continue
        rho = float(np.corrcoef(ra, rb)[0, 1])
        ds.append(1.0 - rho); pares.append((d, len(a), len(b), rho))
    return np.array(ds), pares

def camadas(mask):
    idx = np.where(mask)[0]
    m = defaultdict(list)
    for i in idx: m[int(depth[i])].append(i)
    return {k: np.array(v) for k, v in m.items()}

print("\n" + "=" * 84)
print(" A MEDIDA")
print("=" * 84)
res = {}
for s in ("A", "B"):
    cam = camadas(sub == s)
    ds, pares = delta_por_camada(cam)
    if len(ds) == 0:
        print("\n substrato %s: NENHUM par adjacente valido (|L| >= %d)" % (s, N_MIN))
        res[s] = {"Delta": None, "n_pares": 0, "pares": []}
        continue
    D = float(np.median(ds))
    print("\n substrato %s: %d pares adjacentes validos" % (s, len(ds)))
    for (d, na, nb, rho) in pares[:8]:
        print("    L%-3d(%5d) x L%-3d(%5d)  rho=%.6f  delta=%.6f" % (d, na, d + 1, nb, rho, 1 - rho))
    if len(pares) > 8: print("    ... (+%d pares)" % (len(pares) - 8))
    print("   Delta_%s = mediana = %.8f   (Delta/beta = %.4f)" % (s, D, D / BETA))
    res[s] = {"Delta": D, "n_pares": int(len(ds)),
              "deltas": [float(x) for x in ds],
              "pares": [[int(d), int(na), int(nb), float(rho)] for d, na, nb, rho in pares]}

# ------------------------------------------------------------------ o nulo
print("\n" + "=" * 84)
print(" O NULO (%d embaralhamentos por substrato, preservando tamanhos de camada)" % N_NULL)
print("=" * 84)
rng = np.random.default_rng(20260821)
for s in ("A", "B"):
    if res[s]["Delta"] is None: continue
    idx = np.where(sub == s)[0]
    cam = camadas(sub == s)
    tamanhos = [(d, len(v)) for d, v in sorted(cam.items())]
    nulos = []
    for _ in range(N_NULL):
        perm = rng.permutation(idx)
        novo, pos = {}, 0
        for d, t in tamanhos:
            novo[d] = perm[pos:pos + t]; pos += t
        dn, _ = delta_por_camada(novo)
        if len(dn): nulos.append(float(np.median(dn)))
    nulos = np.array(nulos)
    lo, hi = np.percentile(nulos, [2.5, 97.5])
    fora = not (lo <= res[s]["Delta"] <= hi)
    print("\n substrato %s: nulo mediana=%.6f | IC95%% = [%.6f , %.6f]" % (s, np.median(nulos), lo, hi))
    print("   real = %.8f  ->  %s" % (res[s]["Delta"], "FORA do IC95 (discrimina)" if fora else "DENTRO do IC95 (*NAO DISCRIMINA*)"))
    res[s]["nulo"] = {"mediana": float(np.median(nulos)), "ic95": [float(lo), float(hi)],
                      "real_fora_do_ic95": bool(fora), "n": int(len(nulos))}

# --------------------------------------------------------------- o criterio
print("\n" + "=" * 84)
print(" O CRITERIO PRE-REGISTRADO")
print("=" * 84)
DA, DB = res["A"]["Delta"], res["B"]["Delta"]
if DA is None or DB is None or res["A"]["n_pares"] < 3 or res["B"]["n_pares"] < 3:
    verd = "T06_INCONCLUSIVO_DADOS_INSUFICIENTES"
    C1 = C2 = C3 = None
else:
    razao = DA / DB
    C1 = bool(0.5 <= razao <= 2.0)
    C2 = bool((BETA / 2 <= DA <= 2 * BETA) and (BETA / 2 <= DB <= 2 * BETA))
    C3 = bool(res["A"]["nulo"]["real_fora_do_ic95"] and res["B"]["nulo"]["real_fora_do_ic95"])
    print(" C1 INVARIANCIA  Delta_A/Delta_B = %.4f   (janela [0,5 ; 2,0])   %s" % (razao, "OK" if C1 else "*FALHA*"))
    print(" C2 VALOR        A/beta = %.4f ; B/beta = %.4f   (janela [0,5 ; 2,0])   %s"
          % (DA / BETA, DB / BETA, "OK" if C2 else "*FALHA*"))
    print(" C3 NULO         A fora=%s ; B fora=%s   %s"
          % (res["A"]["nulo"]["real_fora_do_ic95"], res["B"]["nulo"]["real_fora_do_ic95"], "OK" if C3 else "*FALHA*"))
    if C1 and C2 and C3:   verd = "AMPLITUDE_INVARIANTE_ENTRE_DRIVERS_E_BATE_BETA"
    elif not C3:           verd = "NULO_REPRODUZ__SEM_MEDIDA"
    elif not C1:           verd = "AMPLITUDE_DEPENDE_DO_DRIVER"
    else:                  verd = "AMPLITUDE_INVARIANTE_MAS_NAO_E_BETA"

print("\n" + "=" * 84)
print(" VEREDITO COMPUTADO: %s" % verd)
print("=" * 84)

out = {"teste": "T06 — a amplitude da inscricao",
       "pre_registro_sha256": h_pre,
       "tese": ("o substrato e a AMPLITUDE DA INSCRICAO DA LUZ; se beta e essa amplitude, ela e "
                "constante DO ATO e nao DO ATOR, logo deve ser INVARIANTE ENTRE DRIVERS"),
       "beta_derivado": BETA, "alpha_input": ALPHA,
       "universo": {"n_teoremas": int(N), "n_arestas": int(tot_e),
                    "profundidade_max": int(depth.max())},
       "substratos": {"A": "TGLExt/TGL (Driver: o operador)", "B": "Mathlib (Drivers: centenas)"},
       "resultados": res,
       "criterio": {"C1_invariancia": C1, "C2_valor": C2, "C3_nulo": C3,
                    "janela_C2": [BETA / 2, 2 * BETA]},
       "veredito": verd, "data": "2026-08-21",
       "ressalva": ("mede-se a amplitude de inscricao em DOIS CORPORA DE PROVA FORMAL, e nada "
                    "mais. Acordo NAO sai do dominio da inscricao matematica; divergencia derruba "
                    "a invariancia NESSE dominio e so nele. Nao decide beta como constante da "
                    "FISICA, nem a identificacao beta = tau_F(R_J), que segue [CONJECTURE]. "
                    "CONFIRMED proibido.")}
json.dump(out, io.open(SAIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(" gravado: %s" % SAIDA)
