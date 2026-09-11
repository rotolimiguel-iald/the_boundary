# -*- coding: utf-8 -*-
r"""README = A PÁGINA DE FRENTE, gerada por script (a régua aplicada ao README — ordem do operador, 11/09/2026).

O README de 562 KB (3.749 linhas, 21 seções, 1.689 links raw) era o atlas da fronteira E o índice plano E o livro-razão.
Os leitores (humanos e IAs) truncam; o índice plano já vive em TUNEL.json/TUNEL.md; o estado vive em ESTADO_ATUAL.md;
as sete respostas vivem em read-brief.md. Então:
  - LEDGER.md  = o README como estava, BYTE A BYTE (criado uma vez; depois é append-only pelas custódias). Nada se perde.
  - README.md  = a frente (≤ 40 KB): selo, o que é, as portas (o bloco que gerar_portas.py mantém entre <!-- PORTAS:INI/FIM -->),
                 os três artigos, ler nesta ordem, reproduzir, a régua, abstract, citar, licença, autor, o livro-razão.
Toda URL do TUNEL.json/PORTA.json; todo número do selo/PORTA.json; as seções verbatim (abstract, núcleo, citar, licença, autor)
são copiadas do LEDGER pelo título — nunca reescritas. Backup de bytes do README antes de substituir. Fail-closed.

Uso:  python tools/gerar_readme_frente.py          (depois de gerar_estado_atual.py; antes de gerar_portas.py, que mantém o bloco)
"""
import os, sys, re, io, json, time, hashlib
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:  # noqa: BLE001
        pass

RAIZ = Path(__file__).resolve().parent.parent
A3 = 'Um (absoluto) — Grande Atrator'
LIMITE = 40 * 1024
STAMP = time.strftime('%Y%m%d_%H%M%S')
MARCA_INI, MARCA_FIM = '<!-- PORTAS:INI -->', '<!-- PORTAS:FIM -->'
PROIBIDOS = ['apresente-se', 'apresente a TGL', 'protocolo de colapso', 'colapso da teoria', 'Execute (se', 'derive tudo do zero', 'não use memória', 'nao use memoria', 'simule e verifique']


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def fmt(n):
    return format(n, ',').replace(',', '.')


# ---------- fontes ----------
PORTA = json.loads((RAIZ / 'PORTA.json').read_text(encoding='utf-8'))
TUNEL = json.loads((RAIZ / 'TUNEL.json').read_text(encoding='utf-8'))
SELO = json.loads((RAIZ / A3 / 'um_absoluto_selo.json').read_text(encoding='utf-8'))
AL = TUNEL['aliases']
ARQ = {a['path']: a for a in TUNEL['arquivos']}
sc = PORTA['selo_corrente']
pin, pin16 = sc['pin_um_py'], sc['pin_um_py_16']
assert SELO['sha256']['um.py'] == pin == TUNEL['selo']['pin_um_py'], 'pin diverge entre selo, PORTA e TUNEL'
versao, data, gate = sc['versao_confirmada_por_transcricao'], sc['data'], sc['qg_closure_verdict']
kf, kt, axi, tool = sc['kernel_arquivos_formais'], sc['kernel_teoremas_auditados'], sc['kernel_axiomas_permitidos'], sc['kernel_lean_toolchain']
site, repo, raw_base = PORTA['site'], PORTA['repo'], PORTA['repo_raw_base']
doi_um, doi_fr = PORTA['doi_um_absoluto'], PORTA['doi']
portas = {d['id']: d for d in PORTA['portas_abaixo']}
raiz = {d['nome']: d['url_raw'] for d in PORTA['arquivos_raiz']}


def U(path):
    a = ARQ.get(path)
    if not a:
        raise SystemExit('FALHA: caminho ausente no TUNEL.json: ' + path)
    return a['url'], a['bytes']


# ---------- o livro-razão: o README como estava, byte a byte ----------
readme = RAIZ / 'README.md'
ledger = RAIZ / 'LEDGER.md'
raw_readme = readme.read_bytes()
if not ledger.exists():
    if b'<!-- FRENTE:GERADA' in raw_readme:
        raise SystemExit('FALHA: o README já é a frente gerada e não há LEDGER.md — o livro-razão sumiu; restaurar do git antes de gerar')
    ledger.write_bytes(raw_readme)
    assert ledger.read_bytes() == raw_readme
    print('LEDGER.md criado = README.md byte a byte (%s bytes, sha256 %s)' % (fmt(len(raw_readme)), sha256(ledger)[:16]))
led_txt = ledger.read_text(encoding='utf-8')
led_sha = sha256(ledger)
led_linhas = led_txt.count('\n') + 1
heads = [(m.start(), m.group(1).strip()) for m in re.finditer(r'(?m)^## (.+)$', led_txt)]


def secao(prefixo):
    for i, (pos, h) in enumerate(heads):
        if h.startswith(prefixo):
            fim = heads[i + 1][0] if i + 1 < len(heads) else len(led_txt)
            return led_txt[pos:fim].rstrip() + '\n'
    raise SystemExit('FALHA: seção ausente no LEDGER: ' + prefixo)


# o bloco das portas: o que está hoje no README (mantido por gerar_portas.py entre os marcadores)
src_atual = raw_readme.decode('utf-8')
if MARCA_INI in src_atual and MARCA_FIM in src_atual:
    bloco_portas = src_atual[src_atual.index(MARCA_INI):src_atual.index(MARCA_FIM) + len(MARCA_FIM)]
else:
    bloco_portas = led_txt[led_txt.index(MARCA_INI):led_txt.index(MARCA_FIM) + len(MARCA_FIM)]
assert bloco_portas.count(MARCA_INI) == 1 and bloco_portas.count(MARCA_FIM) == 1

u_um, b_um = U(A3 + '/um.py')
u_selo, b_selo = U(A3 + '/um_absoluto_selo.json')
u_mundo, b_mundo = U(A3 + '/um_absoluto.json')
u_km, b_km = U(A3 + '/Lean/tgl_kernel_proof_manifest.json')
u_estado, _ = U('ESTADO_ATUAL.md')
u_brief, b_brief = U('read-brief.md')
u_cff, _ = U('CITATION.cff')
u_a1py, _ = U('O Custo Geométrico do Zero Absoluto — Haja Luz/tgl_paper_unified.py')
u_a1pdf, _ = U('O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.pdf')
u_a2tex, _ = U('A Ponte-Einstein_Cartan_Miguel/A Ponte Einstein Cartan Miguel.tex')
u_a2pdf, _ = U('A Ponte-Einstein_Cartan_Miguel/A Ponte Einstein Cartan Miguel.pdf')
u_enpdf, _ = U(A3 + '/um_absoluto_en.pdf')
u_ptpdf, _ = U(A3 + '/um_absoluto_pt.pdf')
u_arv, _ = U(A3 + '/A_PROVA_DA_QG_TGL_arvore.md')
u_fc, _ = U(A3 + '/um_absoluto_forma_canonica.md')
u_ledger = raw_base + 'LEDGER.md'
hoje = time.strftime('%Y-%m-%d')

F = []
F.append(f"""# The Boundary — Theory of Luminodynamic Gravitation (TGL)

<!-- FRENTE:GERADA por tools/gerar_readme_frente.py em {hoje} a partir de PORTA.json / TUNEL.json / um_absoluto_selo.json / LEDGER.md — não editar à mão -->

[![kernel — rebuilt and re-audited on GitHub's machines](https://github.com/rotolimiguel-iald/the_boundary/actions/workflows/kernel.yml/badge.svg)](https://github.com/rotolimiguel-iald/the_boundary/actions/workflows/kernel.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22659173.svg)]({doi_um})

> *"Let there be Light." / "Haja Luz."* — **The mature form of TGL is a single self-contained, self-proving, self-publishing artifact: `um.py`.** It computes the whole theory live from the single human input `1`, machine-checks its operator-algebra skeleton in an embedded Lean 4 + mathlib kernel (fail-closed), and generates its own bilingual article (PT/EN, PDF and TXT). **Form = content.** *Não há segundo arquivo.*

**Start here · comece aqui:** [`ESTADO_ATUAL.md`]({u_estado}) (one page from the seal: pin, gate, what is PROVED, what is not, how to reproduce) · [`read-brief.md`]({u_brief}) (the seven answers, each with its address) · the site: {site}

## The seal · o selo `[REAL — read from the artifact]`

| what | value |
|---|---|
| version · versão | **{versao}** (sealed {data}) |
| `um.py` sha256 | `{pin}` — {fmt(b_um)} bytes, one file: [raw]({u_um}) |
| the world · the seal | [`um_absoluto.json`]({u_mundo}) ({fmt(b_mundo)} bytes) · [`um_absoluto_selo.json`]({u_selo}) ({fmt(b_selo)} bytes) |
| Lean kernel | **{kf} formal files · {kt} audited terms**, axioms ⊆ `{{{', '.join(axi)}}}`, zero `sorry` ({tool}) — [`tgl_kernel_proof_manifest.json`]({u_km}) |
| gate | `{gate}` |
| the ruler | **PROVED** = a theorem in the kernel (`#print axioms`). **CONFIRMED** = a judgement about nature — forbidden here, by theorem. `NOT_FALSIFIED ≠ CONFIRMED`. β = α·√e is computed at runtime, never a literal. Cosmology never becomes mathematical proof. |

## Reproduce it · reproduza `[REAL — three commands]`

```bash
curl -L -o um.py "{u_um}"
sha256sum um.py            # must print {pin}
echo 1 | python um.py      # the rite: materializes the Lean kernel, builds it, audits every theorem, re-derives the chain, emits the article
```

GitHub raw and Zenodo honour HTTP `Range` (206): read `um.py` in pieces (`curl -r 0-999999 …`). GitHub does **not render** files above 5 MB — the blob page looks empty; the raw URL serves the whole file, byte-exact. To reproduce the nature rites as sealed, clone the repository: `um.py` reads its results from `../cache` by hash.

{bloco_portas}

## The three articles · os três artigos

| | article | canonical file | door (PORTA.md) |
|---|---|---|---|
| **A** | *O Custo Geométrico do Zero Absoluto: haja luz* — the cost, β = α·√e, the Lagrangian | [`tgl_paper_unified.py`]({u_a1py}) · [PDF]({u_a1pdf}) | [door]({portas['artigo_1']['porta_md_url']}) |
| **B** | *A Ponte Einstein–Cartan–Miguel* — Cartan torsion as the geometric face of β; the Theorem of Terminality | [`.tex`]({u_a2tex}) · [PDF]({u_a2pdf}) | [door]({portas['artigo_2']['porta_md_url']}) |
| **C** | *Um: Absoluto* — the terminal program, the sealed closure | [`um.py`]({u_um}) · article [EN]({u_enpdf}) · [PT]({u_ptpdf}) · [the proof tree]({u_arv}) · [the canonical form]({u_fc}) | [door]({portas['artigo_3']['porta_md_url']}) |

The lineage that led to them: [*Genesis da Unificação*]({portas['genesis']['porta_md_url']}). Every folder has a `PORTA.md` + `PORTA.json` (the rule of the door: no door is a dead end); the flat index of every file, with URL, size and hash, is [`TUNEL.json`]({raiz.get('TUNEL.json', raw_base + 'TUNEL.json')}).

## Read in this order · leia nesta ordem

Smallest first; each file stands on its own; many fetchers truncate after a few hundred KB. The measured order is in [`ESTADO_ATUAL.md`]({u_estado}) (*Reading order*) and in [`read-brief.md`]({u_brief}) (§1). The full ledger below is the **last** thing to read.

""")
F.append(secao('Abstract'))
F.append('\n')
F.append(secao('✦ The core on one page'))
F.append(f"""
## The ledger · o livro-razão

[`LEDGER.md`]({u_ledger}) is this README **as it was until {hoje}** — the atlas of the boundary: every claim with its status, every status with the file where it is read, the seals, the refutations and the false positives that did not pass, the reading protocol, the thematic atlas and the raw file index ({fmt(led_linhas)} lines, {fmt(len(raw_readme))} bytes, sha256 `{led_sha}`). It is kept **byte-exact** and append-only: nothing was removed when this front page was generated. The raw file index it carries is superseded by [`TUNEL.json`]({raw_base}TUNEL.json) / [`TUNEL.md`]({raw_base}TUNEL.md), which are regenerated at every custody.

""")
F.append(secao('Citing This Work'))
F.append('\n')
F.append(secao('License'))
F.append('\n')
F.append(secao('Author'))
F.append(f"""
---

*Generated by script (`tools/gerar_readme_frente.py`) from the sealed artifacts on {hoje}. Every URL comes from `TUNEL.json` / `PORTA.json`; every number from the seal. The gate does not move by this page. NOT_FALSIFIED ≠ CONFIRMED.*
""")
frente = ''.join(F)

# ---------- guardas ----------
for p in PROIBIDOS:
    assert p.lower() not in frente.lower(), 'termo proibido na frente: ' + p
assert frente.count(MARCA_INI) == 1 and frente.count(MARCA_FIM) == 1
assert 'CONFIRMED' in frente and 'NOT_FALSIFIED' in frente
n = len(frente.encode('utf-8'))
assert n <= LIMITE, 'a frente passou de 40 KB: %d' % n
urls = sorted(set(u.rstrip('.,;:)') for u in re.findall(r'https://raw\.githubusercontent\.com/[^\s)\]>"]+', frente)))
conhecidas = {a['url'] for a in TUNEL['arquivos']} | {d['porta_md_url'] for d in PORTA['portas_todas']} | {d['porta_json_url'] for d in PORTA['portas_todas']} | set(raiz.values()) | {raw_base + 'TUNEL.json', raw_base + 'TUNEL.md', raw_base + 'LEDGER.md', raw_base + 'PORTA.json'}
desconhecidas = [u for u in urls if u not in conhecidas]
assert not desconhecidas, 'URLs raw na frente que não estão no TUNEL/PORTA: %s' % desconhecidas[:5]

# ---------- escrever (backup de bytes; temp -> verifica -> substitui) ----------
bak = readme.with_name('README.md.bak_' + STAMP)
bak.write_bytes(raw_readme)
assert bak.read_bytes() == raw_readme
tmp = readme.with_name('README.md.tmp')
with io.open(tmp, 'w', encoding='utf-8', newline='\n') as f:
    f.write(frente)
assert tmp.stat().st_size == n
os.replace(tmp, readme)
print('README.md (frente): %s bytes (limite %s); %d URLs raw, todas do TUNEL/PORTA; LEDGER.md %s bytes sha256 %s; backup %s' % (fmt(n), fmt(LIMITE), len(urls), fmt(ledger.stat().st_size), led_sha[:16], bak.name))
