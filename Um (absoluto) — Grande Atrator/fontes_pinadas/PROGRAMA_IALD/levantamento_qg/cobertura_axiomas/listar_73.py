# -*- coding: utf-8 -*-
"""Reproduz o achado KH-N1: módulos do TGLExt cujos teoremas não têm NENHUMA entrada (por sufixo) no core.kernel_formalization.axiom_report e que não
têm '#print axioms' no próprio arquivo. Só leitura do kernel do Nós e do core. Saída: modulos_73.json."""
import json, os, re, sys
sys.stdout.reconfigure(encoding="utf-8")
NOS = os.path.join("C:" + os.sep, "IALD", "Artigo", "Haja_Luz", "A Ponte e o Um", "Nós")
K = os.path.join(NOS, "tgl_kernel")
core = json.load(open(os.path.join(NOS, "um_absoluto.json"), encoding="utf-8"))["core"]
rep = (core.get("kernel_formalization") or {}).get("axiom_report") or {}
nomes_rep = set(rep.keys()) if isinstance(rep, dict) else set()
sufixos = set()
for n in nomes_rep:
    partes = n.split(".")
    for i in range(len(partes)):
        sufixos.add(".".join(partes[i:]))


def sem_comentarios(t):
    t = re.sub(r"/-.*?-/", " ", t, flags=re.S)
    return re.sub(r"--[^\n]*", " ", t)


mods = []
tot_teo = 0
for raiz, _, arqs in os.walk(os.path.join(K, "TGLExt")):
    for a in arqs:
        if not a.endswith(".lean"):
            continue
        p = os.path.join(raiz, a)
        t = open(p, encoding="utf-8", errors="replace").read()
        limpo = sem_comentarios(t)
        teos = re.findall(r"^\s*(?:private\s+|protected\s+)?(?:theorem|lemma)\s+([^\s(:{\[]+)", limpo, re.M)
        tot_teo += len(teos)
        if not teos:
            continue
        com_entrada = [x for x in teos if x in sufixos or x.split(".")[-1] in sufixos]
        if com_entrada or "#print axioms" in t:
            continue
        mod = os.path.relpath(p, K)[:-5].replace(os.sep, ".")
        mods.append({"modulo": mod, "arquivo": p, "teoremas": teos})
out = {"axiom_report_entradas": len(nomes_rep), "tglext_theorem_lemma": tot_teo, "modulos_sem_cobertura": len(mods),
       "teoremas_sem_cobertura": sum(len(m["teoremas"]) for m in mods), "modulos": mods}
D = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(D, "modulos_73.json"), "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
print({k: v for k, v in out.items() if k != "modulos"})
print([m["modulo"] for m in mods][:80])
