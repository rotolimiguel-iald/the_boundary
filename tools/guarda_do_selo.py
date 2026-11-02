# -*- coding: utf-8 -*-
"""GUARDA DO SELO — a licao de 23/08/2026.

O espelho anunciava v198 (um.py, selo, stdout e PDFs todos v198) mas entregava
`fig_escada_qg.pdf` da v170. Passou porque as figuras nao constavam da tabela de
artefatos do handoff, e nada as conferia: a custodia hasheava o vault, as portas
hasheavam o repositorio, mas ninguem comparava o repositorio contra o PROPRIO SELO
que ele publica.

Esta guarda fecha o buraco: toda pasta que contiver `um_grande_atrator_selo.json`
tem TODOS os artefatos do mapa `sha256` do selo re-hasheados contra o disco.
Fail-closed: divergiu, sai com codigo 1.

v2 (07/09/2026) — a segunda licao: o disco conferia 13/13 e o raw do GitHub servia
OUTROS bytes. Com `* text=auto` + core.autocrlf=true o git guardava LF enquanto o
disco e o selo tinham CRLF: o blob de um.py era 9a3becab15f1f011, o selo dizia
e1b74a907c403538. O que o leitor recebe e o BLOB, nao o disco. A guarda agora
confere TAMBEM o blob do indice (`git cat-file -p :caminho`) contra o selo.

v3 (19/10/2026) — a terceira licao (custodia v393+v394): o espelho passou a publicar as FONTES PINADAS que os
registros do um.py releem por caminho absoluto (fontes_pinadas/, com mapa). Uma delas e' uma COPIA do selo da
v391 (insumo da bateria pre-selo da v392). Copia de selo dentro de fontes_pinadas/ nao e' o selo da pasta: a
guarda a confere como insumo pinado — no MAPA_FONTES_PINADAS.json, publicada, disco E blob = o sha256 do pino.

Uso:  python tools/guarda_do_selo.py
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_blob(rel_posix: str):
    """sha256 dos bytes que o git guarda para o caminho (indice) — o que o raw serve.
    None se o caminho nao esta no indice (fail-closed: ausencia e erro, nunca 'ok')."""
    r = subprocess.run(["git", "-C", str(RAIZ), "cat-file", "-p", ":" + rel_posix],
                       capture_output=True)
    if r.returncode != 0:
        return None
    return hashlib.sha256(r.stdout).hexdigest()


def main() -> int:
    erros = 0
    selos = sorted(RAIZ.rglob("um_grande_atrator_selo.json")) + sorted(
        RAIZ.rglob("um_absoluto_selo.json"))  # renome v209: os dois nomes valem
    # v3 (19/10/2026): copia de selo em fontes_pinadas/ e' insumo pinado, nao selo da pasta — conferida pelo mapa
    pinadas = [s_ for s_ in selos if "fontes_pinadas" in s_.relative_to(RAIZ).parts]
    selos = [s_ for s_ in selos if s_ not in pinadas]
    for s_ in pinadas:
        partes = s_.relative_to(RAIZ).parts
        mapa_p = RAIZ.joinpath(*partes[:partes.index("fontes_pinadas") + 1]) / "MAPA_FONTES_PINADAS.json"
        rel_s = s_.relative_to(RAIZ).as_posix()
        try:
            ents = [e for e in json.loads(mapa_p.read_text(encoding="utf-8"))["fontes"]
                    if e.get("publicado") and e.get("caminho_no_espelho") == rel_s]
        except (OSError, KeyError, json.JSONDecodeError) as e:
            print(f"  ! mapa das fontes pinadas ilegivel para {rel_s}: {e}")
            erros += 1
            continue
        pinos = {e.get("sha256") for e in ents}
        if len(pinos) != 1 or None in pinos:
            print(f"  ! COPIA DE SELO EM fontes_pinadas FORA DO MAPA (ou com pinos divergentes): {rel_s}")
            erros += 1
            continue
        pino = pinos.pop()
        if sha256(s_) != pino or sha256_blob(rel_s) != pino:
            print(f"  ! COPIA DE SELO EM fontes_pinadas DIVERGE DO PINO (disco ou blob): {rel_s}")
            erros += 1
            continue
        print(f"[insumo pinado] {rel_s}: copia de selo conferida contra o pino do mapa (disco E blob do git)")
    if not selos:
        print("nenhum selo encontrado — nada a conferir")
        return 0
    for selo_path in selos:
        pasta = selo_path.parent
        rel = pasta.relative_to(RAIZ)
        try:
            selo = json.loads(selo_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! selo ilegivel em {rel}: {e}")
            erros += 1
            continue
        mapa = selo.get("sha256") or {}
        conferidos = 0
        for nome, esperado in mapa.items():
            alvo = pasta / nome
            if not alvo.is_file():
                # o main pode ter sido organizado em subpastas (25/08): resolver por
                # nome na arvore da pasta do selo — fail-closed: exige exatamente os
                # candidatos cujo hash bate; nome achado com hash errado e' divergencia.
                cands = [c for c in pasta.rglob(Path(nome).name) if c.is_file()]
                certo = [c for c in cands if sha256(c) == esperado]
                if len(certo) >= 1:
                    alvo = certo[0]
                elif cands:
                    print(f"  ! DIVERGE DO SELO (achado em subpasta, hash errado): {rel}/{nome}")
                    erros += 1
                    continue
                else:
                    print(f"  ! ARTEFATO AUSENTE: {rel}/{nome} (o selo o cita)")
                    erros += 1
                    continue
            got = sha256(alvo)
            if got != esperado:
                print(f"  ! DIVERGE DO SELO: {rel}/{nome}")
                print(f"      disco {got[:32]}")
                print(f"      selo  {esperado[:32]}")
                erros += 1
                continue
            # v2: o BLOB do git (o que o raw serve) tambem tem de ser o do selo
            blob = sha256_blob(alvo.relative_to(RAIZ).as_posix())
            if blob is None:
                print(f"  ! NAO RASTREADO NO GIT (o raw nao o serve): {rel}/{nome}")
                erros += 1
                continue
            if blob != esperado:
                print(f"  ! O BLOB DO GIT DIVERGE DO SELO (o raw serviria outro byte): {rel}/{nome}")
                print(f"      blob  {blob[:32]}")
                print(f"      selo  {esperado[:32]}")
                print("      causa provavel: normalizacao de fim de linha — exige '* -text' no .gitattributes")
                erros += 1
                continue
            conferidos += 1
        versao = selo.get("timestamp", "?")
        print(f"[selo {versao}] {rel}: {conferidos}/{len(mapa)} artefatos conferem (disco E blob do git)")

    # ------------------------------------------------------------------
    # v4 (02/11/2026) — a quarta licao (custodia v405): PROCEDENCIA DE NUMERO.
    # A guarda conferia hash, blob e pino, e passou 13/13 enquanto o espelho publicava
    # «the rite 3248/3470 clean» — os CENTROS DE VAZIO da pegada do rito V11 — no lugar da
    # escada do kernel (10066/10066), em cinco arquivos e nos dois espelhos. Hash nenhum pega
    # isso: o arquivo errado estava integro. Agora, todo par N/M publicado como a escada do
    # rito e' recomputado contra o SELO, e a escada tem de ser N/N.
    print()
    escada = None
    for s in selos:
        try:
            _s = json.loads(s.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        pilha, achados = [_s], []
        while pilha:
            o = pilha.pop()
            if isinstance(o, dict):
                for k, v in o.items():
                    if k == "n_theorems_clean" and isinstance(v, int):
                        achados.append(v)
                    else:
                        pilha.append(v)
            elif isinstance(o, list):
                pilha.extend(o)
        if achados:
            escada = max(set(achados), key=achados.count)
            break
    PADROES = (r"the rite \| \*\*(\d+)/(\d+) clean", r"the rite (\d+)/(\d+) clean",
               r"(\d+)/(\d+) (?:flags|theorems) of the rite")
    superficies = [RAIZ / n for n in ("ESTADO_ATUAL.md", "README.md", "read-brief.md", "llms.txt")]
    superficies += sorted((RAIZ / "secoes").glob("*.md")) if (RAIZ / "secoes").is_dir() else []
    conferidas, pares = 0, 0
    for p in superficies:
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        achou = False
        for pad in PADROES:
            for m in re.finditer(pad, txt):
                achou, pares = True, pares + 1
                n, d2 = int(m.group(1)), int(m.group(2))
                if n != d2:
                    erros += 1
                    print(f"  ! A ESCADA DO RITO NAO E N/N em {p.name}: {n}/{d2} — isto nao e uma escada")
                elif escada is not None and n != escada:
                    erros += 1
                    print(f"  ! NUMERO SEM PROCEDENCIA em {p.name}: publica {n}/{d2}, o selo diz {escada}")
        if achou:
            conferidas += 1
    print(f"[procedencia do numero] escada do selo {escada} · {pares} pares conferidos em {conferidas} superficies")

    print(f"selos verificados ............ {len(selos)}")
    print(f"ERROS ........................ {erros}")
    print("VEREDITO:", "O QUE SE PUBLICA E O QUE O SELO DESCREVE" if erros == 0
          else "FALHOU — o publicado NAO e o selado")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
