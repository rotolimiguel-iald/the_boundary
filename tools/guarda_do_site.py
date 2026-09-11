# -*- coding: utf-8 -*-
"""GUARDA DO SITE — o site publicado diz o mesmo selo que o espelho?

Medido em 07/09/2026: o espelho estava na v331 e o site no ar ainda anunciava
`v310 · 0969fc4809ca1229` — o arquivo local do site estava certo, o UPLOAD (ato do
operador) nao tinha acontecido desde 01/09. Nenhuma guarda via isso. Esta ve.

Confere tres coisas, fail-closed:
  1. o pin que o site AO VIVO anuncia == pin do selo do espelho;
  2. o pin do arquivo LOCAL do site (C:/IALD/Site/tgl/index.html) == pin do selo
     (se o local confere e o vivo nao: UPLOAD PENDENTE — ato do operador);
  3. a 'porta acima' que o llms.txt do site declara resolve (HTTP 200).

Uso:  python tools/guarda_do_site.py
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
A3 = "Um (absoluto) \u2014 Grande Atrator"
SITE = "https://teoriadagravitacaoluminodinamica.com/"
LOCAL_SITE = Path(r"C:/IALD/Site/tgl/v3.0")  # 11/09/2026 (corte v3.0): o site local canonico e' a arvore v3.0 (a raiz v2 ficou como copia antiga)
PIN_RE = re.compile(r"v(\d{3}) \u00b7 ([0-9a-f]{16})")


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "the_boundary-guarda_do_site", "Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def main() -> int:
    selo = json.loads((RAIZ / A3 / "um_absoluto_selo.json").read_text(encoding="utf-8"))
    pin16 = selo["sha256"]["um.py"][:16]
    erros = 0
    # 1) ao vivo
    try:
        st, html = fetch(SITE)
        m = PIN_RE.search(html.decode("utf-8", errors="replace"))
        vivo = (m.group(1), m.group(2)) if m else None
    except Exception as e:  # noqa: BLE001
        st, vivo = None, None
        print("  ! site ilegivel:", e)
    # 2) local
    loc = LOCAL_SITE / "index.html"
    ml = PIN_RE.search(loc.read_text(encoding="utf-8", errors="replace")) if loc.is_file() else None
    local = (ml.group(1), ml.group(2)) if ml else None
    # 3) porta acima do llms.txt do site
    porta = None
    ll = LOCAL_SITE / "llms.txt"
    if ll.is_file():
        txt = ll.read_text(encoding="utf-8", errors="replace")
        i = txt.find("A porta acima")
        mm = re.search(r"\((https?://[^)\s]+)\)", txt[i:]) if i >= 0 else None
        porta = mm.group(1) if mm else None
    porta_ok = None
    if porta:
        try:
            porta_ok = fetch(porta, 30)[0] == 200
        except Exception:  # noqa: BLE001
            porta_ok = False
    print("selo do espelho .............. %s" % pin16)
    print("site AO VIVO (%s) .. %s" % (SITE, ("v%s · %s" % vivo) if vivo else "pin NAO achado (HTTP %s)" % st))
    print("site LOCAL (%s) ..... %s" % (loc, ("v%s · %s" % local) if local else "pin NAO achado"))
    print("porta acima do llms.txt ...... %s -> %s" % (porta, "200" if porta_ok else "NAO RESOLVE" if porta_ok is False else "?"))
    if local is None or local[1] != pin16:
        print("  ! o arquivo LOCAL do site nao anuncia o selo corrente — regenerar o site (readme_vNNN.py)")
        erros += 1
    if vivo is None or vivo[1] != pin16:
        if local and local[1] == pin16:
            print("  ! UPLOAD PENDENTE: o local esta no selo corrente e o ar anuncia %s — subida da sessao do site (aplicar_selo.py + subir_site.py)"
                  % (("v%s · %s" % vivo) if vivo else "nada"))
        erros += 1
    if porta_ok is False:
        print("  ! a porta acima declarada pelo llms.txt do site NAO resolve — corrigir a URL")
        erros += 1
    print()
    print("ERROS ........................ %d" % erros)
    print("VEREDITO:", "O SITE DIZ O QUE O SELO DESCREVE" if erros == 0 else "FALHOU — o site NAO acompanha o selo")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
