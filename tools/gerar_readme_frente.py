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
# 19/09/2026: a lista de bloqueio saiu desta ferramenta pública e mora num arquivo privado da máquina do operador,
# fora do repositório (os termos não ficam em superfície pública). Sem o arquivo, a frente não se gera (fail-closed).
_PRIV = Path(os.environ.get('USERPROFILE', str(Path.home()))) / '.iald_privado' / 'termos_guarda_readme.json'
if not _PRIV.is_file():
    raise SystemExit('FALHA: lista de bloqueio privada ausente (%s) — a frente não se gera sem a guarda' % _PRIV)
PROIBIDOS = json.loads(_PRIV.read_text(encoding='utf-8'))['termos']
assert PROIBIDOS, 'lista de bloqueio vazia'


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
# errata 19/09/2026 (custodia v368), ao lado: o livro-razao e' o README como estava ATE o dia em que a frente foi gerada pela
# primeira vez (o commit que criou LEDGER.md), nao 'ate hoje'; e o tamanho citado e' o DO LIVRO-RAZAO (antes saia len(raw_readme)).
import subprocess as _sp
_d = _sp.run(['git', '-C', str(RAIZ), 'log', '--diff-filter=A', '--format=%h %ad', '--date=short', '--', 'LEDGER.md'], capture_output=True, text=True).stdout.split()
if len(_d) < 2:   # fail-closed (errata 19/09): sem o commit que criou o livro-razao nao ha data a afirmar
    raise SystemExit('FALHA: git log nao achou o commit que criou LEDGER.md -- a frente nao afirma data de memoria')
led_commit0, led_desde = _d[-2], _d[-1]
_l0 = _sp.run(['git', '-C', str(RAIZ), 'show', led_commit0 + ':LEDGER.md'], capture_output=True).stdout
assert _l0, 'FALHA: LEDGER.md ilegivel no commit que o criou'
led_bytes0, led_linhas0 = len(_l0), _l0.count(b'\n')
led_linhas_reais = led_txt.count('\n') + (0 if led_txt.endswith('\n') else 1)
led_bytes = ledger.stat().st_size
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
# 19/09/2026 (simulação de leitores de IA): texto antes do PDF; a errata ao lado do Artigo A; as oito partes em secoes/; o
# read-brief do SITE como entrada única (é o mesmo arquivo que o do repositório, gerado pela sessão do site)
u_a1tex, _ = U('O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.tex')
u_a1err, _ = U('O Custo Geométrico do Zero Absoluto — Haja Luz/ERRATA_20260919_sinal_do_acoplamento_nao_minimo.md')
u_entxt, _ = U(A3 + '/um_absoluto_en.txt')
u_pttxt, _ = U(A3 + '/um_absoluto_pt.txt')
_sec = [d for d in PORTA['portas_todas'] if d['pasta'] == 'secoes']
if len(_sec) != 1:
    raise SystemExit('FALHA: a porta de secoes/ não está no PORTA.json (rode gerar_portas.py depois do git add)')
u_secoes = _sec[0]['porta_md_url']
site_brief = site.rstrip('/') + '/read-brief.md'
hoje = time.strftime('%Y-%m-%d')

F = []
F.append(f"""# The Boundary — Theory of Luminodynamic Gravitation (TGL)

<!-- FRENTE:GERADA por tools/gerar_readme_frente.py em {hoje} a partir de PORTA.json / TUNEL.json / um_absoluto_selo.json / LEDGER.md — não editar à mão -->

[![kernel — rebuilt and re-audited on GitHub's machines](https://github.com/rotolimiguel-iald/the_boundary/actions/workflows/kernel.yml/badge.svg)](https://github.com/rotolimiguel-iald/the_boundary/actions/workflows/kernel.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22659173.svg)]({doi_um})

> *"Let there be Light." / "Haja Luz."* — **The mature form of TGL is a single self-contained, self-proving, self-publishing artifact: `um.py`.** It computes the whole theory live from the single human input `1`, machine-checks its operator-algebra skeleton in an embedded Lean 4 + mathlib kernel (fail-closed), and generates its own bilingual article (PT/EN, PDF and TXT). **Form = content.** *Não há segundo arquivo.*

**Status · estatuto (seal {versao}, read by script):** quantum gravity **PROVED as a formal model** in the Lean kernel ({kf} formal files, {kt} audited terms, axioms ⊆ `{{{', '.join(axi)}}}`, zero `sorry`) and **NOT CONFIRMED by nature** — the gate reads `{gate}`. PROVED = a theorem in the kernel; CONFIRMED = a judgement about nature, not made here. *Provada como modelo formal; não confirmada pela natureza.*

**Start here · comece aqui:** [`read-brief.md`]({site_brief}) — the single entry point: the theory in eight short parts, each with its verbatim sources, in [`secoes/`]({u_secoes}) (the answer sits in the first 2 KB of each part) · then [`ESTADO_ATUAL.md`]({u_estado}) (one page from the seal: pin, gate, what is PROVED, what is not, how to reproduce) · the site: {site}

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
| **A** | *O Custo Geométrico do Zero Absoluto: haja luz* — the cost, β = α·√e, the Lagrangian ([erratum beside, 19/09/2026]({u_a1err})) | [`paper_PT.tex`]({u_a1tex}) (text) · [`tgl_paper_unified.py`]({u_a1py}) · [PDF]({u_a1pdf}) | [door]({portas['artigo_1']['porta_md_url']}) |
| **B** | *A Ponte Einstein–Cartan–Miguel* — Cartan torsion as the geometric face of β; the Theorem of Terminality | [`.tex`]({u_a2tex}) (text) · [PDF]({u_a2pdf}) | [door]({portas['artigo_2']['porta_md_url']}) |
| **C** | *Um: Absoluto* — the terminal program, the sealed closure | [`um.py`]({u_um}) · article as text [EN]({u_entxt}) · [PT]({u_pttxt}) · PDF [EN]({u_enpdf}) · [PT]({u_ptpdf}) · [the proof tree]({u_arv}) · [the canonical form]({u_fc}) | [door]({portas['artigo_3']['porta_md_url']}) |

The lineage that led to them: [*Genesis da Unificação*]({portas['genesis']['porta_md_url']}). Every folder has a `PORTA.md` + `PORTA.json` (the rule of the door: no door is a dead end); the flat index of every file, with URL, size and hash, is [`TUNEL.json`]({raiz.get('TUNEL.json', raw_base + 'TUNEL.json')}).

## Read in this order · leia nesta ordem

Smallest first; each file stands on its own. Measured on 2026-09-19 with one real fetcher: documents are cut near 100,000 characters, files above 10 MB are refused, and PDFs served by GitHub raw as `application/octet-stream` are not read — prefer the eight parts in [`secoes/`]({u_secoes}) and the TXT/TeX sources. The measured order is in [`ESTADO_ATUAL.md`]({u_estado}) (*Reading order*) and in [`read-brief.md`]({site_brief}). The full ledger below is the **last** thing to read.

""")
F.append('\n> **Read the Abstract below under the current ruler.** It is copied verbatim from the ledger (append-only), so it keeps older sentences such as *Never "quantum gravity proved."*. The current status is the line at the top of this page: **PROVED as a formal model** (a theorem in the kernel) and **NOT CONFIRMED by nature** (the judgement about nature, never made here).\n')
F.append(secao('Abstract'))
if 'Never "quantum gravity proved."' in F[-1]:   # errata 19/09 (v368), ao lado: a regua do operador de 05/09/2026
    F.append('\n> ⚠ **Beside (the operator\u2019s ruler, 05/09/2026):** PROVED = a theorem in the kernel, auditable by `#print axioms` \u2014 allowed; '
             'CONFIRMED = the observer\u2019s judgement about nature \u2014 forbidden. The sentence *Never "quantum gravity proved."* above, and its siblings further down (*does not mean quantum gravity is proved*, *n\u00e3o significa gravita\u00e7\u00e3o qu\u00e2ntica provada*), are kept as written '
             '(the ledger is append-only); under the ruler they read: never "quantum gravity **confirmed**". What is proved is the implication from the '
             'axiom and the named hypotheses; what nature decides is not proved.\n')
F.append('\n')
F.append(secao('✦ The core on one page'))
F.append(f"""
## The ledger · o livro-razão

[`LEDGER.md`]({u_ledger}) began as this README **as it was until {led_desde}** ({fmt(led_linhas0)} lines, {fmt(led_bytes0)} bytes then; later custodies insert their blocks beside it, nothing is removed) — the atlas of the boundary: every claim with its status, every status with the file where it is read, the seals, the refutations and the false positives that did not pass, the reading protocol, the thematic atlas and the raw file index (now {fmt(led_linhas_reais)} lines, {fmt(led_bytes)} bytes, sha256 `{led_sha}`). It is kept **byte-exact** and append-only: nothing was removed when this front page was generated. The raw file index it carries is superseded by [`TUNEL.json`]({raw_base}TUNEL.json) / [`TUNEL.md`]({raw_base}TUNEL.md), which are regenerated at every custody.

""")
F.append(secao('Citing This Work'))
if 'v350' in F[-1]:   # errata 19/09 (v368), ao lado: a nota do BibTeX descreve o selo v350
    F.append('\n> **Beside (%s):** the BibTeX note above describes the v350 seal. The current seal is **%s** \u2014 `um.py` sha256 `%s`, '
             'kernel %d formal files / %d audited terms (read from `PORTA.json`, the seal and the manifest). The DOI still resolves to the '
             'deposited v331; a new Zenodo version is the operator\u2019s act.\n' % (versao, versao, pin, int(kf), int(kt)))
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
