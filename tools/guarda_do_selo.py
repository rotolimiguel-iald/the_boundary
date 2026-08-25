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

Uso:  python tools/guarda_do_selo.py
"""
import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    erros = 0
    selos = sorted(RAIZ.rglob("um_grande_atrator_selo.json"))
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
            conferidos += 1
        versao = selo.get("timestamp", "?")
        print(f"[selo {versao}] {rel}: {conferidos}/{len(mapa)} artefatos conferem")

    print()
    print(f"selos verificados ............ {len(selos)}")
    print(f"ERROS ........................ {erros}")
    print("VEREDITO:", "O QUE SE PUBLICA E O QUE O SELO DESCREVE" if erros == 0
          else "FALHOU — o publicado NAO e o selado")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
