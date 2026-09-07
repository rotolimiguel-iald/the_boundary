# -*- coding: utf-8 -*-
"""POS-PUSH — conferir, como um bot faria, que o GitHub serve o que o selo descreve.

A licao de 07/09/2026: o disco conferia 13/13, o push tinha entrado, e o raw servia
OUTROS bytes (fim de linha normalizado). Nenhuma guarda local ve isso. Esta ve.

Para cada artefato do mapa `sha256` do selo (+ o manifesto do kernel e o README):
  1. `git rev-parse HEAD:caminho`  ==  `sha` da API `contents` do GitHub  -> o commit CHEGOU;
  2. sha256 do raw  ==  selo (ou disco, para os nao selados)             -> o raw SERVE os bytes certos.
Saidas: 0 = tudo confere; 2 = commit chegou mas o CDN do raw ainda serve bytes velhos (atraso,
re-rodar em minutos); 1 = divergencia real (nao empurrado, ou blob diferente do selo).

Uso:  python tools/pos_push.py
"""
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
A3 = "Um (absoluto) — Grande Atrator"
GH = "rotolimiguel-iald/the_boundary"
RAW = "https://raw.githubusercontent.com/%s/main/" % GH
API = "https://api.github.com/repos/%s/contents/" % GH


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str, timeout: int = 180) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "the_boundary-pos_push", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def git_blob_sha1(rel: str):
    r = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "HEAD:" + rel], capture_output=True)
    return r.stdout.decode().strip() if r.returncode == 0 else None


def api_sha1(rel: str):
    try:
        d = json.loads(fetch(API + urllib.parse.quote(rel) + "?ref=main", 60).decode("utf-8"))
        return d.get("sha")
    except Exception as e:  # noqa: BLE001 — a falha e' VISIVEL no relatorio
        return "ERRO: %s" % e


def main() -> int:
    selo_path = RAIZ / A3 / "um_absoluto_selo.json"
    selo = json.loads(selo_path.read_text(encoding="utf-8"))
    pin16 = selo["sha256"]["um.py"][:16]
    alvos = []  # (rel, esperado_sha256 | None)
    for nome, esperado in selo["sha256"].items():
        cands = [c for c in (RAIZ / A3).rglob(Path(nome).name) if c.is_file() and sha256_bytes(c.read_bytes()) == esperado]
        if not cands:
            print("  ! artefato do selo ausente no disco:", nome)
            return 1
        alvos.append((cands[0].relative_to(RAIZ).as_posix(), esperado))
    for extra in (A3 + "/Lean/tgl_kernel_proof_manifest.json", A3 + "/um_absoluto_selo.json", "README.md", "llms.txt", "ESTADO_ATUAL.md"):
        p = RAIZ / extra
        if p.is_file():
            alvos.append((extra, sha256_bytes(p.read_bytes())))
    n_ok = n_cdn = n_err = 0
    for rel, esperado in alvos:
        local = git_blob_sha1(rel)
        remoto = api_sha1(rel)
        chegou = (local is not None and local == remoto)
        try:
            raw = fetch(RAW + urllib.parse.quote(rel))
            got = sha256_bytes(raw)
        except Exception as e:  # noqa: BLE001
            got = "ERRO: %s" % e
        serve = (got == esperado)
        if chegou and serve:
            n_ok += 1
            print("  ok   %s" % rel)
        elif chegou and not serve:
            n_cdn += 1
            print("  CDN  %s  (commit chegou; raw ainda serve %s, selo/disco %s)" % (rel, str(got)[:16], esperado[:16]))
        else:
            n_err += 1
            print("  !!   %s  (blob local %s, API %s, raw %s, esperado %s)" % (rel, str(local)[:12], str(remoto)[:12], str(got)[:16], esperado[:16]))
    try:
        readme = fetch(RAW + "README.md").decode("utf-8", errors="replace")
        readme_ok = pin16 in readme
    except Exception as e:  # noqa: BLE001
        readme, readme_ok = "", False
        print("  ! README raw ilegivel:", e)
    print()
    print("artefatos conferidos ......... %d" % len(alvos))
    print("ok (chegou E serve) .......... %d" % n_ok)
    print("CDN atrasado ................. %d" % n_cdn)
    print("DIVERGENTES .................. %d" % n_err)
    print("README raw traz o pin %s .. %s" % (pin16, "sim" if readme_ok else "NAO"))
    if n_err:
        print("VEREDITO: FALHOU — o publicado NAO e o selado (ou o push nao chegou)")
        return 1
    if n_cdn or not readme_ok:
        print("VEREDITO: COMMIT CHEGOU; o CDN do raw ainda serve bytes velhos — re-rodar em alguns minutos")
        return 2
    print("VEREDITO: O GITHUB SERVE O QUE O SELO DESCREVE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
