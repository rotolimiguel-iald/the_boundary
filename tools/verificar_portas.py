# -*- coding: utf-8 -*-
"""
VERIFICADOR INDEPENDENTE DAS PORTAS -- nao reusa nada do gerador.
Re-varre os arquivos escritos, extrai TODA URL, e checa:
  1. toda URL raw resolve em arquivo existente e RASTREADO pelo git;
  2. toda URL de arvore resolve em pasta existente;
  3. toda pasta rastreada tem PORTA.md e PORTA.json;
  4. toda PORTA.md tem 'porta acima:' e essa porta acima existe (fecha o ciclo
     ate a raiz);
  5. toda subpasta e' citada pela porta do pai (nenhum beco sem saida);
  6. todo arquivo rastreado aparece em ALGUMA porta;
  7. o selo publicado nas portas confere com o selo em disco;
  8. sha256[:16] de cada arquivo listado no PORTA.json confere com o disco.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import urllib.parse

REPO = r"c:\IALD\Artigo\the_boundary"
RAW = "https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/"
TREE = "https://github.com/rotolimiguel-iald/the_boundary/tree/main/"

out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True)
tracked = set(f for f in out.stdout.decode("utf-8").split("\0") if f)
novos = set(f for f in os.listdir(REPO))  # noqa
# arquivos de porta ainda nao commitados: coleta em disco
portas_md, portas_json = set(), set()
for root, dnames, fnames in os.walk(REPO):
    if ".git" in root.split(os.sep):
        continue
    rel = os.path.relpath(root, REPO).replace(os.sep, "/")
    rel = "" if rel == "." else rel
    for fn in fnames:
        p = (rel + "/" + fn) if rel else fn
        if fn == "PORTA.md":
            portas_md.add(p)
        elif fn == "PORTA.json":
            portas_json.add(p)

universo = tracked | portas_md | portas_json | {"llms.txt"}

erros = []
avisos = []

# --- pastas rastreadas
pastas = set()
for f in tracked:
    parts = f.split("/")
    for i in range(len(parts) - 1):
        pastas.add("/".join(parts[: i + 1]))

# (3) toda pasta tem as duas portas
for d in sorted(pastas):
    if d + "/PORTA.md" not in portas_md:
        erros.append("pasta SEM PORTA.md: %s" % d)
    if d + "/PORTA.json" not in portas_json:
        erros.append("pasta SEM PORTA.json: %s" % d)
for nome in ("PORTA.md", "PORTA.json", "llms.txt"):
    if not os.path.isfile(os.path.join(REPO, nome)):
        erros.append("raiz SEM %s" % nome)

# --- (1)(2) toda URL resolve
URL = re.compile(r"https://(?:raw\.githubusercontent\.com|github\.com)/[^\s\)\]\"'<>|`]+")
arquivos_scan = sorted(portas_md | portas_json | {"llms.txt"})
n_urls, n_raw, n_tree = 0, 0, 0
urls_unicas = set()
alvos_raw = set()
for rel in arquivos_scan:
    txt = io.open(os.path.join(REPO, rel.replace("/", os.sep)), encoding="utf-8").read()
    for u in URL.findall(txt):
        u = u.rstrip(".,;:")
        n_urls += 1
        urls_unicas.add(u)
        if u == RAW or u == RAW.rstrip("/"):
            continue  # a base URL declarada (repo_raw_base), nao um alvo
        if u.startswith(RAW):
            n_raw += 1
            alvo = urllib.parse.unquote(u[len(RAW):])
            alvos_raw.add(alvo)
            if alvo not in universo:
                erros.append("URL raw fora do universo: %s (em %s)" % (u, rel))
            elif not os.path.isfile(os.path.join(REPO, alvo.replace("/", os.sep))):
                erros.append("URL raw sem arquivo em disco: %s (em %s)" % (u, rel))
        elif u.startswith(TREE):
            n_tree += 1
            alvo = urllib.parse.unquote(u[len(TREE):])
            if not os.path.isdir(os.path.join(REPO, alvo.replace("/", os.sep))):
                erros.append("URL de arvore sem pasta: %s (em %s)" % (u, rel))

# --- (4) porta acima existe e a cadeia fecha na raiz
for rel in sorted(portas_md):
    txt = io.open(os.path.join(REPO, rel.replace("/", os.sep)), encoding="utf-8").read()
    m = re.search(r"^porta acima:\s*(\S+)", txt, re.M)
    if not m:
        erros.append("PORTA.md sem 'porta acima:': %s" % rel)
        continue
    u = m.group(1)
    if rel == "PORTA.md":
        if not u.startswith("https://teoriadagravitacaoluminodinamica.com"):
            erros.append("raiz: porta acima nao aponta ao site: %s" % u)
        continue
    alvo = urllib.parse.unquote(u[len(RAW):]) if u.startswith(RAW) else None
    esperado = "/".join(rel.split("/")[:-2])
    esperado = (esperado + "/PORTA.md") if esperado else "PORTA.md"
    if alvo != esperado:
        erros.append("porta acima errada em %s: %s (esperado %s)" % (rel, alvo, esperado))
    # cadeia ate a raiz
    cur, hops = rel, 0
    while cur != "PORTA.md" and hops < 12:
        pai = "/".join(cur.split("/")[:-2])
        cur = (pai + "/PORTA.md") if pai else "PORTA.md"
        hops += 1
        if cur not in portas_md and cur != "PORTA.md":
            erros.append("cadeia quebrada acima de %s em %s" % (rel, cur))
            break
    else:
        if cur != "PORTA.md":
            erros.append("cadeia nao chega a raiz: %s" % rel)

# --- (5) toda subpasta citada pela porta do pai
for d in sorted(pastas):
    pai = "/".join(d.split("/")[:-1])
    ppath = (pai + "/PORTA.md") if pai else "PORTA.md"
    txt = io.open(os.path.join(REPO, ppath.replace("/", os.sep)), encoding="utf-8").read()
    if urllib.parse.quote(d + "/PORTA.md") not in txt:
        erros.append("subpasta nao citada pelo pai: %s (pai %s)" % (d, ppath))

# --- (6) todo arquivo rastreado aparece em alguma porta
faltando = sorted(f for f in tracked if f not in alvos_raw)
for f in faltando:
    erros.append("arquivo rastreado que NAO aparece em nenhuma porta: %s" % f)

# --- (7) selo publicado == selo em disco
selo_p = "Um (absoluto) \u2014 Grande Atrator/um_grande_atrator_selo.json"
selo = json.load(io.open(os.path.join(REPO, selo_p.replace("/", os.sep)), encoding="utf-8"))
rj = json.load(io.open(os.path.join(REPO, "PORTA.json"), encoding="utf-8"))
sc = rj["selo_corrente"]
for k, v in (("pin_um_py", selo["sha256"]["um.py"]),
             ("mundo", selo["sha256"]["um_grande_atrator.json"]),
             ("result_hash", selo["result_hash"]),
             ("data", selo["timestamp"]),
             ("qg_closure_verdict", selo["qg_closure_verdict"]),
             ("formal_source_hash", selo["formal_source_hash"])):
    if sc.get(k) != v:
        erros.append("selo divergente em PORTA.json[%s]: %r != %r" % (k, sc.get(k), v))

# --- (8) sha256[:16] de cada arquivo listado confere com o disco
n_hash = 0
for rel in sorted(portas_json):
    j = json.load(io.open(os.path.join(REPO, rel.replace("/", os.sep)), encoding="utf-8"))
    lista = j.get("arquivos") or j.get("arquivos_raiz") or []
    for a in lista:
        cam = a.get("caminho") or a.get("nome")
        h = a.get("sha256_16")
        if not h or not cam:
            continue
        p = os.path.join(REPO, cam.replace("/", os.sep))
        if not os.path.isfile(p):
            erros.append("PORTA.json %s lista arquivo inexistente: %s" % (rel, cam))
            continue
        d = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                d.update(c)
        n_hash += 1
        if d.hexdigest()[:16] != h:
            erros.append("sha256 divergente em %s: %s" % (rel, cam))

print("pastas rastreadas ............ %d" % len(pastas))
print("PORTA.md em disco ............ %d" % len(portas_md))
print("PORTA.json em disco .......... %d" % len(portas_json))
print("arquivos varridos ............ %d" % len(arquivos_scan))
print("URLs encontradas (total) ..... %d" % n_urls)
print("URLs unicas .................. %d" % len(urls_unicas))
print("  raw (arquivo) .............. %d" % n_raw)
print("  tree (pasta) ............... %d" % n_tree)
print("alvos raw distintos .......... %d" % len(alvos_raw))
print("arquivos rastreados .......... %d" % len(tracked))
print("hashes reconferidos .......... %d" % n_hash)
print("avisos ....................... %d" % len(avisos))
print("ERROS ........................ %d" % len(erros))
for e in erros[:40]:
    print("  ! %s" % e)
print("VEREDITO: %s" % ("TODAS AS PORTAS RESOLVEM" if not erros else "FALHOU"))
