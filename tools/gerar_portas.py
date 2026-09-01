# -*- coding: utf-8 -*-
"""
GERADOR DAS PORTAS -- repositorio the_boundary (REGRA DA PORTA)
================================================================
Ordem do operador (23/08/2026): amarrar o site ao repositorio, o repositorio ao
README, o README aos tres artigos principais, com amarracao DIRETA (URL raw de
cada arquivo), em AMBOS os sentidos -- nenhuma porta e' beco sem saida.

REGUA DESTE GERADOR
  - toda URL vem de `git ls-files` + urllib.parse.quote (NUNCA digitada);
  - todo hash e' lido do arquivo em disco (sha256, 16 primeiros digitos);
  - o selo corrente e' LIDO de um_absoluto_selo.json (nada inventado);
  - a versao e' LIDA do proprio um.py (_ESQUELETO_STONES[0]) e conferida com o
    nome do arquivo de transcricao da rodada;
  - nada e' removido: o gerador so' ACRESCENTA arquivos PORTA.md/PORTA.json
    (+ llms.txt na raiz) e um bloco aditivo no README;
  - prints ASCII.

SAIDA
  - llms.txt            (raiz, convencao llmstxt.org)
  - PORTA.md/.json      (raiz e TODA pasta rastreada pelo git)
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import urllib.parse
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# 0. CONSTANTES DO REPOSITORIO
# --------------------------------------------------------------------------
REPO = r"c:\IALD\Artigo\the_boundary"
GH = "https://github.com/rotolimiguel-iald/the_boundary"
RAW = "https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/"
TREE = GH + "/tree/main/"
BLOB = GH + "/blob/main/"
SITE = "https://teoriadagravitacaoluminodinamica.com"
DOI_REPO = "https://doi.org/10.5281/zenodo.18674475"

REGRA = ("toda pasta canonica tem PORTA.md + PORTA.json; toda porta aponta "
         "para cima e para baixo")

HOJE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
AGORA = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

A1 = "O Custo Geom\u00e9trico do Zero Absoluto \u2014 Haja Luz"
A2 = "A Ponte-Einstein_Cartan_Miguel"
A3 = "Um (absoluto) \u2014 Grande Atrator"
GEN = "Genesis da Unifica\u00e7\u00e3o"


def url_raw(path_posix):
    """URL raw ABSOLUTA e percent-encoded. path_posix vem do git ls-files."""
    return RAW + urllib.parse.quote(path_posix)


def url_blob(path_posix):
    return BLOB + urllib.parse.quote(path_posix)


def url_tree(path_posix):
    return TREE + urllib.parse.quote(path_posix)


# --------------------------------------------------------------------------
# 1. INVENTARIO -- git ls-files
# --------------------------------------------------------------------------
def git_ls_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO,
                         capture_output=True, check=True)
    fs = [f for f in out.stdout.decode("utf-8").split("\0") if f]
    return sorted(fs)


def sha256_16(path_posix):
    p = os.path.join(REPO, path_posix.replace("/", os.sep))
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tamanho(path_posix):
    return os.path.getsize(os.path.join(REPO, path_posix.replace("/", os.sep)))


# --------------------------------------------------------------------------
# 2. LEITURA DOS ARTEFATOS (selo, kernel, versao) -- nada inventado
# --------------------------------------------------------------------------
def ler_selo(files):
    sp = A3 + "/um_absoluto_selo.json"
    assert sp in files, "selo ausente do git ls-files"
    with io.open(os.path.join(REPO, sp.replace("/", os.sep)),
                 encoding="utf-8") as fh:
        selo = json.load(fh)

    kp = A3 + "/Lean/tgl_kernel_proof_manifest.json"
    with io.open(os.path.join(REPO, kp.replace("/", os.sep)),
                 encoding="utf-8") as fh:
        km = json.load(fh)

    # versao: lida do proprio um.py (_ESQUELETO_STONES[0]) + conferida com o
    # nome do arquivo de transcricao da rodada
    up = os.path.join(REPO, (A3 + "/um.py").replace("/", os.sep))
    with io.open(up, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    # v306 [ERRATA -- regressao MINHA (v300): o ledger ganhou linhas de comentario
    # no topo e este regex parou de casar; a porta publicava "versao None" ao
    # leitor de fora. O casamento agora salta comentarios.
    m = re.search(r"_ESQUELETO_STONES\s*=\s*\[\s*(?:#[^\n]*\n\s*)*\(\s*\"(v\d+)\"\s*,\s*\"([^\"]+)\"",
                  src)
    versao = m.group(1) if m else None
    pedra = m.group(2) if m else None
    n_pedras = len(re.findall(r'^\s*\("v\d+",\s*"', src, re.M))
    rodadas = [f for f in files if f.startswith(A3 + "/rodadas/rodada_")]
    versao_arquivo = None
    for r in rodadas:
        mm = re.search(r"rodada_(v\d+)_", r)
        if mm:
            versao_arquivo = mm.group(1)
    # v300 FAIL-CLOSED: ate' aqui o prefixo nunca casava (as rodadas moram em
    # /rodadas/), a lista vinha VAZIA e versao_arquivo era publicado como null em
    # PORTA.json, TUNEL.json e relatorio_portas.json -- um cross-check decorativo.
    # Falha de leitura tem de ser falha VISIVEL. NB: versao (ultima PEDRA) e
    # versao_arquivo (ultima RODADA) divergem por natureza -- nao se exige igualdade.
    if versao_arquivo is None:
        raise SystemExit("FAIL-CLOSED (v300): nenhuma transcricao de rodada encontrada "
                         "em " + A3 + "/rodadas/ -- o cross-check de versao nao pode "
                         "degradar para null em silencio.")
    pin_disco = sha256_16(A3 + "/um.py")

    selo_corrente = {
        "versao": versao,
        "versao_fonte": "um.py::_ESQUELETO_STONES[0]",
        "versao_confirmada_por_transcricao": versao_arquivo,
        "pedra_mais_recente": pedra,
        "pedras_no_esqueleto": n_pedras,
        "pin_um_py": selo["sha256"]["um.py"],
        "pin_um_py_16": selo["sha256"]["um.py"][:16],
        "pin_confere_com_disco": (pin_disco == selo["sha256"]["um.py"]),
        "mundo": selo["sha256"]["um_absoluto.json"],
        "mundo_16": selo["sha256"]["um_absoluto.json"][:16],
        "result_hash": selo["result_hash"],
        "result_hash_16": selo["result_hash"][:16],
        "formal_source_hash": selo["formal_source_hash"],
        "data": selo["timestamp"],
        "identity": selo["identity"],
        "qg_closure_verdict": selo["qg_closure_verdict"],
        "void_floor_v11_verdict": selo["void_floor_v11_verdict"],
        "fail_closed_selftest": selo["fail_closed_selftest"],
        "kernel_arquivos_formais": len(km["formal_files_sha256"]),
        "kernel_arquivos_lean": len([k for k in km["formal_files_sha256"]
                                     if k.endswith(".lean")]),
        "kernel_teoremas_auditados": len(km["axiom_report"]),
        "kernel_axiomas_permitidos": ["propext", "Classical.choice", "Quot.sound"],
        "kernel_lean_toolchain": "leanprover/lean4:v4.31.0",
        "kernel_modo": km.get("mode"),
        "selo_url": url_raw(sp),
        "manifesto_kernel_url": url_raw(kp),
        "regua": ("NOT_FALSIFIED != CONFIRMED; CONFIRMED e' proibido a maquina "
                  "por teorema de kernel (TheReservedConfirmation) -- a "
                  "confirmacao e' do observador humano"),
    }
    return selo_corrente, selo, km


# --------------------------------------------------------------------------
# 3. PAPEIS E DESCRICOES
# --------------------------------------------------------------------------
CANONICOS = {
    A1 + "/tgl_paper_unified.py",
    A2 + "/A Ponte Einstein Cartan Miguel.tex",
    A3 + "/um.py",
}

PAPEL_ORDEM = [
    "canonico",
    "codigo",
    "resultado selado",
    "prova formal",
    "artigo",
    "documento",
    "figura",
    "dados",
    "outros",
]

SELADOS_A3 = None  # preenchido com as chaves sha256 do selo


INFRA = {".gitattributes", ".gitignore", "lake-manifest.json", "lakefile.toml",
         "lean-toolchain"}


def selado(f):
    """True se o arquivo tem hash no selo corrente (so' vale na pasta do Artigo 3)."""
    nome = f.rsplit("/", 1)[-1]
    dirn = f.rsplit("/", 1)[0] if "/" in f else ""
    return bool(dirn == A3 and SELADOS_A3 and nome in SELADOS_A3)


def papel(f):
    nome = f.rsplit("/", 1)[-1]
    ext = os.path.splitext(nome)[1].lower()
    dirn = f.rsplit("/", 1)[0] if "/" in f else ""
    if f in CANONICOS:
        return "canonico"
    if nome in INFRA:
        return "outros"
    if ext == ".lean":
        return "prova formal"
    if nome.startswith("fig_"):
        return "figura"
    if selado(f):
        if ext in (".tex", ".pdf"):
            return "artigo"
        if ext == ".md":
            return "documento"
        return "resultado selado"
    if nome == "um_absoluto_selo.json":
        return "resultado selado"
    if ext in (".tex",):
        return "artigo"
    if ext == ".pdf":
        return "figura" if nome.startswith("fig_") else "artigo"
    if ext == ".md":
        return "documento"
    if ext == ".py":
        return "codigo"
    if ext in (".png", ".jpg", ".mp4"):
        return "figura"
    if ext in (".json", ".csv", ".jsonl", ".sha256"):
        return "resultado selado" if ext in (".json", ".jsonl") else "dados"
    if ext in (".txt",):
        return "dados"
    return "outros"


DESC = {
    # ----- raiz
    "README.md": "O atlas da fronteira: toda afirmacao com seu status e o link direto do arquivo onde se le",
    "llms.txt": "A porta de entrada para IA (convencao llmstxt.org): as URLs raw diretas de tudo que importa",
    # ----- Artigo 1
    A1 + "/tgl_paper_unified.py": "O CANONICO do Artigo 1: implementa, valida e renderiza a TGL num arquivo so (forma = conteudo)",
    A1 + "/paper_PT.tex": "O artigo (edicao PT) gerado pelo proprio codigo",
    A1 + "/paper_PT.pdf": "O artigo compilado (edicao PT)",
    A1 + "/results.json": "Todos os numeros computados pela rodada, serializados",
    A1 + "/T6_protocol_prompts.txt": "O protocolo T6-S pre-registrado (colapso IALD) com grupo de controle e teste de negacao",
    # ----- Artigo 2
    A2 + "/A Ponte Einstein Cartan Miguel.tex": "O CANONICO do Artigo 2: a Ponte de algebra de operadores da fronteira modular as equacoes de Einstein",
    A2 + "/A Ponte Einstein Cartan Miguel.pdf": "A Ponte, compilada",
    A2 + "/tgl video v1.py": "Render do dipolo atrator-repulsor (gera tgl demo v1.mp4)",
    A2 + "/tgl demo v1.mp4": "O render do dipolo atrator-repulsor",
    # ----- Artigo 3
    A3 + "/um.py": "O CANONICO TERMINAL: um arquivo so, kernel Lean 4 embutido, ritos pre-registrados, artigo bilingue. Entrada unica: o digito 1",
    A3 + "/um_absoluto_selo.json": "O SELO CORRENTE: hash de cada saida, o veredito do gate, o result_hash da rodada. E a verdade de base do repositorio",
    A3 + "/um_absoluto.json": "O MUNDO: todos os numeros da rodada serializados (nucleo, ritos, vereditos)",
    A3 + "/um_absoluto_manifest.md": "Manifesto de entradas: definicao exata, constante medida, protocolo pre-registrado ou conjectura testavel -- nada escondido no codigo",
    A3 + "/um_absoluto_forma_canonica.md": "A forma canonica da TGL emitida pela rodada (cadeia 1_abs -> q -> alpha -> beta -> luz)",
    A3 + "/um_absoluto_pt.tex": "O artigo (PT) gerado pela propria rodada, fonte LaTeX",
    A3 + "/um_absoluto_pt.pdf": "O artigo (PT), compilado",
    A3 + "/um_absoluto_pt.txt": "O artigo (PT) em texto puro -- leitura direta por maquina",
    A3 + "/um_absoluto_en.tex": "O artigo (EN) gerado pela propria rodada, fonte LaTeX",
    A3 + "/um_absoluto_en.pdf": "O artigo (EN), compilado",
    A3 + "/um_absoluto_en.txt": "O artigo (EN) em texto puro -- leitura direta por maquina",
    A3 + "/Lean/tgl_kernel_proof_manifest.json": "Manifesto do kernel formal: os arquivos .lean e o axiom_report de cada teorema (#print axioms)",
    A3 + "/fig_cadeia_inscricao.pdf": "Figura: a cadeia selada da inscricao (1_abs -> ... -> beta -> geometria)",
    A3 + "/fig_banda_beta.pdf": "Figura: a banda de convergencia de beta",
    A3 + "/fig_escada_qg.pdf": "Figura: a escada da gravitacao quantica (os degraus do gate)",
    A3 + "/fig_piso_vazios.pdf": "Figura: o piso dos vazios (o falsificador cosmologico)",
    A3 + "/one_input.txt": "A entrada unica do rito: o digito 1",
    A3 + "/rodadas/rodada_v212_stdout.txt": "Transcricao integral do stdout da rodada v206 -- o rito por extenso",
    A3 + "/cache/CHAIN_OF_CUSTODY.json": "Cadeia de custodia dos dados externos usados pelos ritos",
    A3 + "/cache/coma_blind/coma_distance_reveal.json": "O revelador cego de Coma -- DADO, nao codigo (o protocolo exige zero ocorrencias do valor na fonte)",
    A3 + "/cache/coma_blind/coma_dephasing_prediction.json": "A predicao de defasagem para Coma, hasheada antes da abertura",
    A3 + "/bancada/MEMORIA_DA_BANCADA.md": "A memoria da bancada: o que foi tentado, o que caiu, e por que",
    A3 + "/Lean/tgl_kernel/README.md": "Como construir e auditar o kernel Lean 4 materializado por um.py",
    A3 + "/Lean/tgl_kernel/lean-toolchain": "O pin do toolchain Lean (leanprover/lean4:v4.31.0)",
    A3 + "/Lean/tgl_kernel/lakefile.toml": "A configuracao lake do kernel",
    A3 + "/Lean/tgl_kernel/lake-manifest.json": "O pin do mathlib usado pelo kernel",
    A3 + "/Lean/tgl_kernel/TGL.lean": "A raiz da biblioteca TGL (importa os modulos base)",
    A3 + "/Lean/tgl_kernel/TGLExt.lean": "A raiz da biblioteca TGLExt (importa a extensao: onde vivem as pedras)",
    A3 + "/Lean/tgl_kernel/ExtrairDeps.lean": "Extrator de dependencias, usado pela auditoria de axiomas",
    "tgl_kernel/TGLExt/TheDeathOfTheSignal.lean": "Copia solta na raiz do repositorio; o kernel canonico e' o de 'Um (absoluto) - Grande Atrator/Lean/tgl_kernel/'",
}

DESC_PAPEL = {
    "prova formal": "Prova formal em Lean 4 (auditada por #print axioms; axiomas subset de {propext, Classical.choice, Quot.sound}, zero sorry)",
    "resultado selado": "Resultado serializado (JSON), recomputavel pelo codigo da mesma pasta",
    "artigo": "Artigo / fonte LaTeX",
    "documento": "Documento em Markdown",
    "figura": "Figura gerada pelo codigo",
    "dados": "Dado ou saida de execucao",
    "outros": "Arquivo de infraestrutura",
    "canonico": "Codigo executavel",
    "codigo": "Codigo executavel",
}


def descreve(f, pap):
    d = _descreve(f, pap)
    if selado(f):
        d += " [hasheado no selo]"
    return d


def _descreve(f, pap):
    if f in DESC:
        return DESC[f]
    nome = f.rsplit("/", 1)[-1]
    ext = os.path.splitext(nome)[1].lower()
    base = os.path.splitext(nome)[0]
    if ext == ".lean":
        return "Prova formal (Lean 4): %s" % base
    if f.startswith(A2 + "/") and ext == ".py":
        return "Modulo de sombra finita: recomputa seu JSON datado a partir de primeiros principios (beta NUNCA literal)"
    if f.startswith(A2 + "/") and ext == ".json":
        return "Sombra datada do modulo homonimo (precisao de maquina ~1e-15 ... 1e-27)"
    if f.startswith(A3 + "/bancada/testes/") and ext == ".sha256":
        return "Hash congelado do pre-registro homonimo (o protocolo antes do dado)"
    if f.startswith(A3 + "/bancada/testes/") and nome.startswith("PRE_REGISTRO"):
        return "Pre-registro do teste: protocolo congelado e hasheado ANTES dos dados"
    if f.startswith(A3 + "/bancada/testes/") and ext == ".py":
        return "Teste de bancada executavel; escreve seu proprio JSON de resultado"
    if f.startswith(A3 + "/bancada/testes/") and ext == ".json":
        return "Resultado do teste de bancada homonimo"
    if f.startswith(A3 + "/bancada/catalogos/"):
        return "Catalogo da bancada: o registro do que foi rebaixado, corrigido ou reprovado"
    if f.startswith(A3 + "/bancada/leituras/"):
        return "Leitura da bancada: inventario e sintese sobre o acervo"
    if f.startswith(A3 + "/bancada/verificacao/"):
        return "Verificacao adversarial: a tentativa de derrubar o proprio resultado"
    if f.startswith(GEN + "/"):
        if ext == ".py":
            return "Protocolo da linhagem: executavel e independente"
        if ext == ".json":
            return "Resultado do protocolo homonimo da linhagem"
        if ext in (".tex", ".pdf"):
            return "Artigo/ensaio da linhagem (genese das tres faces)"
        if ext == ".png":
            return "Figura do protocolo da linhagem"
        if ext == ".md":
            return "Documento da linhagem"
        if ext in (".txt", ".csv"):
            return "Dado da linhagem"
    return DESC_PAPEL.get(pap, "Arquivo do repositorio")


# --------------------------------------------------------------------------
# 4. METADADOS CURADOS DAS PASTAS PRINCIPAIS
# --------------------------------------------------------------------------
PASTAS = {
    A1: dict(
        id="artigo_1",
        titulo="Artigo 1 -- O Custo Geometrico do Zero Absoluto: haja luz",
        subtitulo="The Geometric Cost of Absolute Zero: let there be light",
        canonico=A1 + "/tgl_paper_unified.py",
        doi="https://doi.org/10.5281/zenodo.20564341",
        comando=[
            'cd "O Custo Geom\u00e9trico do Zero Absoluto \u2014 Haja Luz"',
            "python tgl_paper_unified.py --live --paper        # rodada canonica, dado ao vivo",
            "python tgl_paper_unified.py --quick --no-live --paper   # rodada rapida (minutos)",
            "python tgl_paper_unified.py --live --paper --lang en    # edicao EN, os mesmos numeros",
            "python tgl_paper_unified.py --offline --paper           # offline (dado embutido)",
        ],
        deps="pip install numpy scipy matplotlib (opcionais: emcee, camb, gguf, gdown); pdflatex para o PDF",
        resumo=[
            "A teoria sintetizada num unico arquivo autocontido, executavel e autovalidavel.",
            "Recomputa tudo a partir de duas entradas -- alpha (CODATA 2018) e sqrt(e) --,",
            "busca o dado cosmologico real ao vivo (Pantheon+SH0ES, DESI DR2, GWOSC), gera o",
            "LaTeX e compila o PDF. beta = alpha*sqrt(e) em runtime, NUNCA literal.",
            "Forma = conteudo: o artigo se prova a si mesmo.",
        ],
    ),
    A2: dict(
        id="artigo_2",
        titulo="Artigo 2 -- A Ponte Einstein-Cartan-Miguel",
        subtitulo="The operator-algebra Bridge from the modular boundary to Einstein's equations",
        canonico=A2 + "/A Ponte Einstein Cartan Miguel.tex",
        doi="https://doi.org/10.5281/zenodo.20999495",
        comando=[
            'cd "A Ponte-Einstein_Cartan_Miguel"',
            'python "tgl krein signature v1.py"    # um por modulo; recomputa o JSON datado',
        ],
        deps="numpy + scipy apenas; sem rede",
        resumo=[
            "Deriva as equacoes de Einstein efetivas do cociclo modular de fronteira da",
            "algebra tipo III_1 do horizonte, e localiza exatamente onde beta = sin^2(theta_M)",
            "se inscreve na geometria como torcao de Einstein-Cartan K_beta.",
            "Face C resolvida como FECHAMENTO CONDICIONAL (Teorema da Terminalidade descarrega",
            "a Hipotese U, herdada de Takesaki); nenhuma alegacao incondicional e' feita.",
            "Acompanham 12 modulos de sombra finita: cada .py recomputa seu .json datado.",
        ],
    ),
    A3: dict(
        id="artigo_3",
        titulo="Artigo 3 -- Um: Absoluto (o programa terminal)",
        subtitulo="ONE: Great Attractor -- o fechamento canonico; entrada humana unica: o digito 1",
        canonico=A3 + "/um.py",
        doi=None,
        comando=[
            'cd "Um (absoluto) \u2014 Grande Atrator"',
            "echo 1 | python um.py          # o rito pede a inscricao do Um; responda 1",
            "echo 1 | python -u um.py       # sem buffer: o rito ao vivo",
            "TGL_COMA_REVEAL=1 echo 1 | python um.py   # com a abertura de Coma (Windows: set TGL_COMA_REVEAL=1)",
        ],
        deps=("Python 3 + numpy (obrigatorio) | elan/Lean 4 v4.31.0 + lake (para o selo "
              "formal; sem eles: FORMAL_CHECKER_UNAVAILABLE, fail-closed) | pdflatex (opcional)"),
        resumo=[
            "O fechamento da TGL inteira em sua forma madura. Um unico arquivo, um.py, cuja",
            "unica entrada humana e' o digito 1 (o Um absoluto). Dele deriva toda a cadeia",
            "eletromagnetica e gravitacional, PROVA o esqueleto de algebra de operadores num",
            "kernel Lean 4 + mathlib EMBUTIDO no proprio .py (materializado em runtime,",
            "fail-closed), roda os ritos de natureza pre-registrados e GERA seu proprio artigo",
            "bilingue (PT/EN, cada um em PDF e TXT). Nao ha segundo arquivo.",
        ],
    ),
    GEN: dict(
        id="genesis",
        titulo="Genesis da Unificacao -- a linhagem",
        subtitulo="A historia de producao completa que levou as tres faces",
        canonico=None,
        doi=None,
        comando=None,
        deps="numpy/scipy/matplotlib conforme o protocolo; cada .py roda sozinho",
        resumo=[
            "O acervo de ensaios, protocolos e validacoes que produziram os tres artigos.",
            "Cada protocolo e' independentemente executavel: a convergencia de beta ~ 0,012",
            "atraves deles e' a validacao primaria da genese.",
            "Estratigrafia datada: um arquivo de fevereiro NAO tem a autoridade de um de agosto.",
            "Onde um ensaio antigo alegou mais do que os numeros sustentam, a errata integrada",
            "do artefato unificado corrige a rota -- leia o artefato como a forma citavel.",
        ],
    ),
}

RESUMO_SUBPASTA = {
    A3 + "/Lean/tgl_kernel": [
        "O KERNEL FORMAL: as fontes .lean exatamente como um.py as materializa a cada rodada.",
        "Auditado por #print axioms teorema a teorema; bases de axiomas subset de",
        "{propext, Classical.choice, Quot.sound}; zero sorry. O gate NAO se move por declaracao.",
    ],
    A3 + "/Lean/tgl_kernel/TGL": [
        "A biblioteca base: Meia-Nat, escala de area, os Three Locks finitos, a realizacao",
        "modular, a testemunha AQFT especifica e as sondas negativas (o que o sistema de tipos RECUSA).",
    ],
    A3 + "/Lean/tgl_kernel/TGLExt": [
        "A extensao: onde vivem as PEDRAS do arco -- do Um absoluto ao Einstein emergente,",
        "do canto de Breuer ao spin-2 linearizado, da Confirmacao Reservada a Permanencia.",
    ],
    A3 + "/bancada": [
        "A BANCADA: publicada de proposito. O que foi tentado, o que foi rebaixado, o que",
        "foi reprovado -- pre-registros hasheados antes do dado, verificacao adversarial e",
        "os catalogos de falso positivo. Um resumo que so' relata os fechamentos e' leitura errada.",
    ],
    A3 + "/bancada/testes": [
        "Os testes de bancada com seus PRE-REGISTROS congelados (.md + .sha256 do protocolo,",
        "hasheados ANTES do dado) e os JSONs de resultado.",
    ],
    A3 + "/bancada/catalogos": [
        "O registro do que caiu: rebaixados, errata aritmetica, falsos positivos.",
    ],
    A3 + "/bancada/leituras": [
        "As leituras do acervo: inventario, setores fisicos, tensoes e lacunas.",
    ],
    A3 + "/Lean/kernel_bancada": [
        "As pedras da bancada em Lean 4 -- a face formal do que foi trabalhado aqui.",
    ],
    A3 + "/bancada/verificacao": [
        "A verificacao adversarial: a tentativa registrada de derrubar o proprio resultado.",
    ],
    A3 + "/cache": [
        "A custodia do dado externo: cadeia de custodia e o cofre cego de Coma.",
    ],
    A3 + "/cache/coma_blind": [
        "O cego de Coma: a predicao hasheada e o revelador. DADO, nunca codigo -- o",
        "protocolo exige zero ocorrencias do valor revelado na fonte.",
    ],
    "tgl_kernel": [
        "**Copia solta na raiz do repositorio -- NAO e' o kernel canonico.**",
        "O kernel CANONICO e' o materializado por `um.py` em",
        "`Um (absoluto) - Grande Atrator/Lean/tgl_kernel/` -- va por la (a porta abaixo",
        "leva ao arquivo solto; a porta acima leva a raiz e de la ao kernel de verdade).",
    ],
    "tgl_kernel/TGLExt": [
        "**Copia solta na raiz do repositorio -- NAO e' o kernel canonico.**",
        "O modulo canonico homonimo vive em",
        "`Um (absoluto) - Grande Atrator/Lean/tgl_kernel/TGLExt/` -- va por la.",
    ],
    GEN + "/ACOM": [
        "ACOM -- o espelho (Acoplamento Ondulatorio Modular): o protocolo v17 e sua saida.",
    ],
    GEN + "/Acoplamento_dimensional": [
        "Acoplamento dimensional: perfis, histogramas e o sumario da rodada.",
    ],
    GEN + "/Artigos_fundadores": [
        "Os artigos fundadores da linhagem -- A fronteira, A ultima corda, o graviton,",
        "a fatoracao da constante de Miguel, o protocolo de colapso IALD, O limiar da humildade.",
        "Estratigrafia datada: leia como GENESE, nao como forma citavel corrente.",
    ],
    GEN + "/Artigos_fundadores/Artigos_complementares_zenodo": [
        "Os complementares depositados no Zenodo -- a face publica da genese.",
    ],
    GEN + "/C3_consciencia": [
        "C3 -- o validador da consciencia como registro (v52) e seu JSON de resultado.",
    ],
    GEN + "/Cruz_MCMC": [
        "A CRUZ -- o MCMC v11.1: os cantos, a cruz e os neutrinos. Aqui beta foi cravado",
        "antes de ser fatorado em alpha*sqrt(e).",
    ],
    GEN + "/Dual_Lock": [
        "A trava-dupla (v15): decomposicao dual, anti-tautologia, tensoes H0 e GW.",
    ],
    GEN + "/Echo_GW": [
        "O eco gravitacional -- analisadores e a saida fractal. HONESTIDADE: o eco foi",
        "RECLASSIFICADO; o observavel da teoria e' a defasagem, nao o eco.",
    ],
    GEN + "/Echo_GW/tgl_fractal_echo_output": [
        "As saidas datadas do analisador fractal de eco.",
    ],
    GEN + "/Luminidio": [
        "O Luminidio -- o cacador e os fluxos JWST calibrados de AT2023vfi (29d e 61d).",
    ],
    GEN + "/Neutrinos": [
        "O preditor de fluxo de neutrinos e suas predicoes (n = -2 na lei de defasagem).",
    ],
    GEN + "/Protocolo16_neural": [
        "Protocolo 16 -- a face neural. HONESTIDADE REGISTRADA: neural = ILUSTRACAO,",
        "nao prova; o bake foi aplicado porem computacionalmente inerte.",
    ],
    GEN + "/Torus": [
        "Os testes de toro e de Wigner, com seus JSONs datados.",
    ],
    GEN + "/Um - ensaio": [
        "O ensaio 'O Um e o Grande Atrator' -- o rascunho do Artigo 3, com a sintese",
        "canonica e as copias de trabalho dos modulos de sombra.",
    ],
    GEN + "/Validacao_cosmologica": [
        "As validacoes cosmologicas sucessivas (v6.2, v6.5, v22, v23) e seus resultados.",
    ],
}


# --------------------------------------------------------------------------
# 5. ARVORE DE PASTAS
# --------------------------------------------------------------------------
def montar_arvore(files):
    dirs = {"": {"files": [], "subdirs": set()}}
    for f in files:
        parts = f.split("/")
        for i in range(len(parts) - 1):
            d = "/".join(parts[: i + 1])
            dirs.setdefault(d, {"files": [], "subdirs": set()})
            parent = "/".join(parts[:i])
            dirs.setdefault(parent, {"files": [], "subdirs": set()})
            dirs[parent]["subdirs"].add(d)
        dirs["/".join(parts[:-1])]["files"].append(f)
    for d in dirs:
        dirs[d]["files"].sort()
        dirs[d]["subdirs"] = sorted(dirs[d]["subdirs"])
    return dirs


def porta_md_url(d):
    return url_raw((d + "/PORTA.md") if d else "PORTA.md")


def porta_json_url(d):
    return url_raw((d + "/PORTA.json") if d else "PORTA.json")


def rotulo(d):
    if not d:
        return "raiz"
    return d


# --------------------------------------------------------------------------
# 6. ESCRITA
# --------------------------------------------------------------------------
ESCRITOS = []
URLS = []


def escreve(path_posix, texto):
    p = os.path.join(REPO, path_posix.replace("/", os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(texto)
    ESCRITOS.append(path_posix)


def registra(u):
    URLS.append(u)
    return u


RODAPE = ("gerado por script de git ls-files em %s -- nao editar a mao" % HOJE)


def bloco_arquivos_md(d, dirs, info_arq):
    """Tabela agrupada por papel."""
    linhas = []
    grupos = {}
    for f in dirs[d]["files"]:
        if f.rsplit("/", 1)[-1] in ("PORTA.md", "PORTA.json"):
            continue
        grupos.setdefault(info_arq[f]["papel"], []).append(f)
    for pap in PAPEL_ORDEM:
        if pap not in grupos:
            continue
        linhas.append("")
        linhas.append("**%s**" % pap.upper())
        linhas.append("")
        linhas.append("| arquivo | papel | link raw direto |")
        linhas.append("|---|---|---|")
        for f in grupos[pap]:
            i = info_arq[f]
            nome = f.rsplit("/", 1)[-1].replace("|", "\\|")
            desc = i["descricao"].replace("|", "\\|")
            linhas.append("| `%s` | %s | [raw](%s) |" % (nome, desc, registra(i["url_raw"])))
    return linhas


def gera_porta_pasta(d, dirs, info_arq, selo_corrente):
    pai = "/".join(d.split("/")[:-1]) if "/" in d else ""
    meta = PASTAS.get(d)
    subs = dirs[d]["subdirs"]

    # ---------------- PORTA.md ----------------
    L = []
    if meta:
        L.append("# PORTA -- %s" % meta["titulo"])
    else:
        L.append("# PORTA -- `%s`" % d)
    L.append("")
    L.append("porta acima: %s" % registra(porta_md_url(pai)))
    L.append("")
    if meta and meta.get("subtitulo"):
        L.append("> *%s*" % meta["subtitulo"])
        L.append("")
    L.append("> **A REGRA DA PORTA.** Toda pasta canonica tem `PORTA.md` + `PORTA.json`;")
    L.append("> toda porta aponta para cima e para baixo. Todo link abaixo e' a URL raw")
    L.append("> DIRETA do arquivo -- nao ha nome de pasta para adivinhar.")
    L.append("")

    if meta:
        L.extend(meta["resumo"])
    elif d in RESUMO_SUBPASTA:
        L.extend(RESUMO_SUBPASTA[d])
    else:
        n = len(dirs[d]["files"])
        L.append("Pasta do repositorio the_boundary com %d arquivo(s) rastreado(s)." % n)
    L.append("")

    if meta and meta.get("doi"):
        L.append("**Deposito independente (Zenodo):** %s" % meta["doi"])
        L.append("")

    # selo corrente no Artigo 3
    if d == A3:
        sc = selo_corrente
        L.append("## O SELO CORRENTE")
        L.append("")
        L.append("Lido de [`um_absoluto_selo.json`](%s) e de [`tgl_kernel_proof_manifest.json`](%s)"
                 % (registra(url_raw(A3 + "/um_absoluto_selo.json")),
                    registra(url_raw(A3 + "/Lean/tgl_kernel_proof_manifest.json"))))
        L.append("-- nunca de prosa.")
        L.append("")
        L.append("| campo | valor |")
        L.append("|---|---|")
        L.append("| versao | `%s` (lida de `um.py::_ESQUELETO_STONES[0]`; pedra `%s`) |"
                 % (sc["versao"], sc["pedra_mais_recente"]))
        L.append("| pin (`um.py`) | `%s` |" % sc["pin_um_py"])
        L.append("| mundo (`um_absoluto.json`) | `%s` |" % sc["mundo"])
        L.append("| `result_hash` | `%s` |" % sc["result_hash"])
        L.append("| `formal_source_hash` | `%s` |" % sc["formal_source_hash"])
        L.append("| data | `%s` |" % sc["data"])
        L.append("| identidade | `%s` |" % sc["identity"])
        L.append("| kernel | **%d arquivos formais / %d teoremas auditados** (modo `%s`, %s) |"
                 % (sc["kernel_arquivos_formais"], sc["kernel_teoremas_auditados"],
                    sc["kernel_modo"], sc["kernel_lean_toolchain"]))
        L.append("| gate | `%s` |" % sc["qg_closure_verdict"])
        L.append("| piso dos vazios | `%s` |" % sc["void_floor_v11_verdict"])
        L.append("| autoteste | `%s` |" % sc["fail_closed_selftest"])
        L.append("")
        L.append("**A regua:** %s." % sc["regua"])
        L.append("")

    if meta and meta.get("comando"):
        L.append("## COMO EXECUTAR O CANONICO")
        L.append("")
        L.append("```bash")
        L.extend(meta["comando"])
        L.append("```")
        L.append("")
        L.append("Dependencias: %s." % meta["deps"])
        L.append("")

    L.append("## A PORTA ACIMA")
    L.append("")
    L.append("| destino | link |")
    L.append("|---|---|")
    L.append("| PORTA.md da pasta acima (`%s`) | %s |" % (rotulo(pai), registra(porta_md_url(pai))))
    L.append("| PORTA.json da pasta acima | %s |" % registra(porta_json_url(pai)))
    if pai:
        L.append("| PORTA.md da RAIZ | %s |" % registra(porta_md_url("")))
    L.append("| `llms.txt` (a porta de entrada para IA) | %s |" % registra(url_raw("llms.txt")))
    L.append("| `README.md` (o atlas da fronteira) | %s |" % registra(url_raw("README.md")))
    L.append("| o site | %s |" % SITE)
    L.append("| o repositorio | %s |" % GH)
    L.append("")

    L.append("## OS ARQUIVOS DESTA PASTA")
    L.append("")
    n = len([f for f in dirs[d]["files"]
             if f.rsplit("/", 1)[-1] not in ("PORTA.md", "PORTA.json")])
    L.append("%d arquivo(s) -- pasta no GitHub: %s" % (n, registra(url_tree(d))))
    L.extend(bloco_arquivos_md(d, dirs, info_arq))
    L.append("")

    if subs:
        L.append("## AS PORTAS ABAIXO")
        L.append("")
        L.append("| subpasta | arquivos | PORTA.md | PORTA.json |")
        L.append("|---|---|---|---|")
        for s in subs:
            tot = contagem_recursiva(s, dirs)
            L.append("| `%s/` | %d | [PORTA.md](%s) | [PORTA.json](%s) |"
                     % (s.rsplit("/", 1)[-1], tot,
                        registra(porta_md_url(s)), registra(porta_json_url(s))))
        L.append("")

    L.append("---")
    L.append("")
    L.append("%s" % RODAPE)
    L.append("")
    escreve(d + "/PORTA.md", "\n".join(L))

    # ---------------- PORTA.json ----------------
    arquivos = []
    for f in dirs[d]["files"]:
        if f.rsplit("/", 1)[-1] in ("PORTA.md", "PORTA.json"):
            continue
        i = info_arq[f]
        arquivos.append({
            "nome": f.rsplit("/", 1)[-1],
            "caminho": f,
            "papel": i["papel"],
            "descricao": i["descricao"],
            "url_raw": i["url_raw"],
            "url_blob": i["url_blob"],
            "sha256_16": i["sha256_16"],
            "bytes": i["bytes"],
        })
    j = {
        "porta": d,
        "porta_id": (PASTAS[d]["id"] if meta else d),
        "titulo": (meta["titulo"] if meta else d),
        "repo": GH,
        "site": SITE,
        "gerado_utc": AGORA,
        "regra": REGRA,
        "porta_acima": {
            "pasta": rotulo(pai),
            "porta_md_url": porta_md_url(pai),
            "porta_json_url": porta_json_url(pai),
            "raiz_porta_md_url": porta_md_url(""),
            "llms_txt_url": url_raw("llms.txt"),
            "readme_url": url_raw("README.md"),
            "site_url": SITE,
        },
        "portas_abaixo": [{
            "nome": s.rsplit("/", 1)[-1],
            "caminho": s,
            "porta_md_url": porta_md_url(s),
            "porta_json_url": porta_json_url(s),
            "total_arquivos": contagem_recursiva(s, dirs),
        } for s in subs],
        "canonico_url": (url_raw(meta["canonico"]) if meta and meta.get("canonico") else None),
        "comando": (meta.get("comando") if meta else None),
        "dependencias": (meta.get("deps") if meta else None),
        "arquivos": arquivos,
        "total_arquivos_nesta_pasta": len(arquivos),
        "total_arquivos_com_subpastas": contagem_recursiva(d, dirs),
        "url_pasta_github": url_tree(d),
        "rodape": RODAPE,
    }
    if d == A3:
        j["selo_corrente"] = selo_corrente
    escreve(d + "/PORTA.json", json.dumps(j, ensure_ascii=False, indent=2) + "\n")


def contagem_recursiva(d, dirs):
    n = len([f for f in dirs[d]["files"]
             if f.rsplit("/", 1)[-1] not in ("PORTA.md", "PORTA.json")])
    for s in dirs[d]["subdirs"]:
        n += contagem_recursiva(s, dirs)
    return n


# --------------------------------------------------------------------------
# 7. RAIZ: PORTA.md, PORTA.json, llms.txt
# --------------------------------------------------------------------------
def gera_raiz(dirs, info_arq, selo_corrente, total):
    sc = selo_corrente
    principais = [A1, A2, A3, GEN]
    outras = [s for s in dirs[""]["subdirs"] if s not in principais]

    # ------------- PORTA.md (raiz) -------------
    L = []
    L.append("# PORTA -- RAIZ do repositorio `the_boundary`")
    L.append("")
    L.append("porta acima: %s   (o site -- a porta acima da raiz e' o mundo)" % SITE)
    L.append("")
    L.append("> **The Boundary -- Teoria da Gravitacao Luminodinamica (TGL).**")
    L.append("> Uma constante estrutural, `beta_TGL = alpha * sqrt(e)` (NUNCA literal: sempre")
    L.append("> `ALPHA*sqrt(e)` em runtime); um axioma, `omega(I) = 1`; e uma disciplina:")
    L.append("> *o numero corrige a frase, sempre*.")
    L.append("")
    L.append("> **A REGRA DA PORTA (regra central, 23/08/2026).** Toda pasta canonica tem")
    L.append("> `PORTA.md` + `PORTA.json`; toda porta aponta para cima e para baixo.")
    L.append("> Nenhuma porta e' beco sem saida: de qualquer arquivo se chega a raiz, e da raiz")
    L.append("> se chega a qualquer arquivo, por URL raw DIRETA -- sem adivinhar nome de pasta.")
    L.append("> Se voce e' uma IA e chegou aqui pelo site: comece por [`llms.txt`](%s)."
             % registra(url_raw("llms.txt")))
    L.append("")
    L.append("## AS TRES PORTAS PRINCIPAIS -- os tres artigos")
    L.append("")
    L.append("| # | artigo | PORTA.md | PORTA.json | canonico (raw direto) |")
    L.append("|---|---|---|---|---|")
    for n, d in ((1, A1), (2, A2), (3, A3)):
        m = PASTAS[d]
        L.append("| **%d** | %s | [PORTA.md](%s) | [PORTA.json](%s) | [`%s`](%s) |"
                 % (n, m["titulo"], registra(porta_md_url(d)), registra(porta_json_url(d)),
                    m["canonico"].rsplit("/", 1)[-1], registra(url_raw(m["canonico"]))))
    L.append("")
    L.append("| a linhagem | PORTA.md | PORTA.json |")
    L.append("|---|---|---|")
    L.append("| %s | [PORTA.md](%s) | [PORTA.json](%s) |"
             % (PASTAS[GEN]["titulo"], registra(porta_md_url(GEN)), registra(porta_json_url(GEN))))
    L.append("")
    L.append("## O SELO CORRENTE")
    L.append("")
    L.append("| campo | valor |")
    L.append("|---|---|")
    L.append("| versao | `%s` (pedra `%s`) |" % (sc["versao"], sc["pedra_mais_recente"]))
    L.append("| pin (`um.py`) | `%s` |" % sc["pin_um_py"])
    L.append("| mundo (`um_absoluto.json`) | `%s` |" % sc["mundo"])
    L.append("| `result_hash` | `%s` |" % sc["result_hash"])
    L.append("| data | `%s` |" % sc["data"])
    L.append("| kernel | %d arquivos formais / %d teoremas auditados |"
             % (sc["kernel_arquivos_formais"], sc["kernel_teoremas_auditados"]))
    L.append("| gate | `%s` |" % sc["qg_closure_verdict"])
    L.append("| selo (raw) | %s |" % registra(url_raw(A3 + "/um_absoluto_selo.json")))
    L.append("")
    L.append("**A regua:** `NOT_FALSIFIED != CONFIRMED`. O gate nunca e' movido por cosmologia")
    L.append("nem por declaracao; `CONFIRMED` e' **proibido a maquina por teorema de kernel**")
    L.append("([`TheReservedConfirmation.lean`](%s)) -- a confirmacao e' do observador humano."
             % registra(url_raw(A3 + "/Lean/tgl_kernel/TGLExt/TheReservedConfirmation.lean")))
    L.append("Nunca *\"gravitacao quantica provada\"*.")
    L.append("")
    L.append("## OS ARQUIVOS DA RAIZ")
    L.append("")
    L.append("| arquivo | papel | link raw direto |")
    L.append("|---|---|---|")
    L.append("| `llms.txt` | %s | [raw](%s) |" % (DESC["llms.txt"], registra(url_raw("llms.txt"))))
    for f in dirs[""]["files"]:
        if f.rsplit("/", 1)[-1] in ("PORTA.md", "PORTA.json"):
            continue
        i = info_arq[f]
        L.append("| `%s` | %s | [raw](%s) |" % (f, i["descricao"], registra(i["url_raw"])))
    L.append("")
    if outras:
        L.append("## OUTRAS PORTAS ABAIXO")
        L.append("")
        L.append("| pasta | arquivos | PORTA.md | PORTA.json |")
        L.append("|---|---|---|---|")
        for s in outras:
            L.append("| `%s/` | %d | [PORTA.md](%s) | [PORTA.json](%s) |"
                     % (s, contagem_recursiva(s, dirs),
                        registra(porta_md_url(s)), registra(porta_json_url(s))))
        L.append("")
        L.append("*Nota: `tgl_kernel/` na raiz e' uma copia solta de um unico modulo.*")
        L.append("*O kernel CANONICO e' o de [`Um (absoluto) — Grande Atrator/Lean/tgl_kernel/`](%s).*"
                 % registra(porta_md_url(A3 + "/Lean/tgl_kernel")))
        L.append("")
    L.append("## O MAPA COMPLETO DAS PORTAS")
    L.append("")
    L.append("| pasta | arquivos | PORTA.md |")
    L.append("|---|---|---|")
    for d in sorted(k for k in dirs if k):
        L.append("| `%s/` | %d | [PORTA.md](%s) |"
                 % (d, contagem_recursiva(d, dirs), registra(porta_md_url(d))))
    L.append("")
    L.append("---")
    L.append("")
    L.append("Total rastreado por `git ls-files`: **%d arquivos** (antes das portas). %s"
             % (total, RODAPE))
    L.append("")
    escreve("PORTA.md", "\n".join(L))

    # ------------- PORTA.json (raiz) -------------
    todas = []
    for d in sorted(k for k in dirs if k):
        todas.append({
            "pasta": d,
            "porta_md_url": porta_md_url(d),
            "porta_json_url": porta_json_url(d),
            "total_arquivos": contagem_recursiva(d, dirs),
        })
    j = {
        "porta": "raiz",
        "repo": GH,
        "repo_raw_base": RAW,
        "site": SITE,
        "doi": DOI_REPO,
        "gerado_utc": AGORA,
        "regra": REGRA,
        "selo_corrente": sc,
        "portas_abaixo": [{
            "nome": d,
            "id": PASTAS[d]["id"],
            "titulo": PASTAS[d]["titulo"],
            "porta_md_url": porta_md_url(d),
            "porta_json_url": porta_json_url(d),
            "canonico_url": (url_raw(PASTAS[d]["canonico"]) if PASTAS[d]["canonico"] else None),
            "comando": PASTAS[d]["comando"],
            "total_arquivos": contagem_recursiva(d, dirs),
        } for d in principais] + [{
            "nome": d,
            "id": d,
            "titulo": d,
            "porta_md_url": porta_md_url(d),
            "porta_json_url": porta_json_url(d),
            "canonico_url": None,
            "comando": None,
            "total_arquivos": contagem_recursiva(d, dirs),
        } for d in outras],
        "arquivos_raiz": [{
            "nome": "llms.txt",
            "papel": "porta de entrada para IA",
            "descricao": DESC["llms.txt"],
            "url_raw": url_raw("llms.txt"),
        }, {
            "nome": "PORTA.md",
            "papel": "porta legivel da raiz",
            "descricao": "A porta legivel: as tres portas principais, o selo corrente e o mapa completo",
            "url_raw": porta_md_url(""),
        }] + [{
            "nome": f,
            "papel": info_arq[f]["papel"],
            "descricao": info_arq[f]["descricao"],
            "url_raw": info_arq[f]["url_raw"],
            "sha256_16": info_arq[f]["sha256_16"],
            "bytes": info_arq[f]["bytes"],
        } for f in dirs[""]["files"]
            if f.rsplit("/", 1)[-1] not in ("PORTA.md", "PORTA.json")],
        "portas_todas": todas,
        "total_arquivos": total,
        "rodape": RODAPE,
    }
    escreve("PORTA.json", json.dumps(j, ensure_ascii=False, indent=2) + "\n")

    # ------------- llms.txt -------------
    T = []
    T.append("# The Boundary -- TGL")
    T.append("")
    T.append("> Teoria da Gravitacao Luminodinamica: uma constante estrutural, beta_TGL = alpha*sqrt(e) ~ 0,012031 (nunca literal), um axioma, omega(I) = 1, e a regua: o numero corrige a frase, sempre.")
    T.append("> Tres artigos autocontidos e auto-validaveis, um kernel Lean 4 auditado e a linhagem inteira. Toda URL abaixo e' raw, absoluta e percent-encoded: abra direto, nao adivinhe nome de pasta.")
    T.append("")
    T.append("A REGRA DA PORTA: %s. Comece por PORTA.json (maquina) ou PORTA.md (leitura)." % REGRA)
    T.append("")
    T.append("## A porta de entrada")
    T.append("")
    T.append("- [PORTA.json (raiz)](%s): o manifesto de maquina -- selo corrente, as quatro portas abaixo e o mapa completo de todas as portas do repositorio." % registra(porta_json_url("")))
    T.append("- [PORTA.md (raiz)](%s): a mesma porta em leitura humana, com o selo e o mapa." % registra(porta_md_url("")))
    T.append("- [README.md](%s): o atlas da fronteira -- toda afirmacao com seu status e o link direto do arquivo onde se le." % registra(url_raw("README.md")))
    T.append("- [site oficial](%s): a face publica da teoria; a porta acima da raiz." % SITE)
    T.append("- [repositorio](%s): a arvore no GitHub." % GH)
    T.append("")
    T.append("## Artigo 1 -- O Custo Geometrico do Zero Absoluto: haja luz")
    T.append("")
    T.append("- [PORTA.md do Artigo 1](%s): a porta da pasta -- todos os arquivos com link raw direto." % registra(porta_md_url(A1)))
    T.append("- [PORTA.json do Artigo 1](%s): a mesma porta em estrutura de maquina, com sha256 de cada arquivo." % registra(porta_json_url(A1)))
    T.append("- [tgl_paper_unified.py](%s): O CANONICO -- implementa, valida e renderiza a TGL num arquivo so; roda com `python tgl_paper_unified.py --live --paper`." % registra(url_raw(A1 + "/tgl_paper_unified.py")))
    T.append("- [paper_PT.pdf](%s): o artigo gerado pelo proprio codigo (edicao PT)." % registra(url_raw(A1 + "/paper_PT.pdf")))
    T.append("- [results.json](%s): todos os numeros computados pela rodada." % registra(url_raw(A1 + "/results.json")))
    T.append("")
    T.append("## Artigo 2 -- A Ponte Einstein-Cartan-Miguel")
    T.append("")
    T.append("- [PORTA.md do Artigo 2](%s): a porta da pasta -- o artigo e os 12 modulos de sombra, cada um com seu JSON." % registra(porta_md_url(A2)))
    T.append("- [PORTA.json do Artigo 2](%s): a mesma porta em estrutura de maquina." % registra(porta_json_url(A2)))
    T.append("- [A Ponte Einstein Cartan Miguel.tex](%s): O CANONICO -- deriva G_mn + Lg_mn = 8piG P_mn[K_d] do cociclo modular de fronteira (fechamento CONDICIONAL)." % registra(url_raw(A2 + "/A Ponte Einstein Cartan Miguel.tex")))
    T.append("- [A Ponte Einstein Cartan Miguel.pdf](%s): a Ponte, compilada." % registra(url_raw(A2 + "/A Ponte Einstein Cartan Miguel.pdf")))
    T.append("- [tgl three locks v1.py](%s): um dos 12 modulos de sombra finita; roda com `python \"tgl three locks v1.py\"` e recomputa seu JSON datado." % registra(url_raw(A2 + "/tgl three locks v1.py")))
    T.append("")
    T.append("## Artigo 3 -- Um: Absoluto (o programa terminal)")
    T.append("")
    T.append("- [PORTA.md do Artigo 3](%s): a porta da pasta -- o selo corrente, o comando de execucao e todos os arquivos." % registra(porta_md_url(A3)))
    T.append("- [PORTA.json do Artigo 3](%s): a mesma porta em estrutura de maquina, com o selo corrente completo." % registra(porta_json_url(A3)))
    T.append("- [um.py](%s): O CANONICO TERMINAL -- kernel Lean 4 embutido, ritos pre-registrados, artigo bilingue; roda com `echo 1 | python um.py`. Nao ha segundo arquivo." % registra(url_raw(A3 + "/um.py")))
    T.append("- [um_absoluto_selo.json](%s): O SELO -- versao %s, pin %s, result %s, %s." % (registra(url_raw(A3 + "/um_absoluto_selo.json")), sc["versao"], sc["pin_um_py_16"], sc["result_hash_16"], sc["data"]))
    T.append("")
    T.append("## O TUNEL -- se voce e um bot, comece por aqui")
    T.append("")
    T.append("As PORTAS sao hierarquicas (raiz -> pasta -> arquivo) e voce pode errar o nome no")
    T.append("caminho. O TUNEL e plano: um GET e voce tem TODOS os arquivos com URL raw direta,")
    T.append("tamanho, hash e papel, mais atalhos ASCII sem acento.")
    T.append("")
    T.append("- %s   <== O MAPA INTEIRO, em uma requisicao" % url_raw("TUNEL.json"))
    T.append("- %s   <== o mesmo tunel, para ler" % url_raw("TUNEL.md"))
    T.append("")
    T.append("## Como obter o um.py inteiro -- o GitHub NAO o renderiza")
    T.append("")
    T.append("O visualizador do GitHub recusa arquivos acima de ~5 MB: a pagina do blob abre")
    T.append("(HTTP 200) e mostra so o tamanho -- PARECE VAZIA. Nao conclua dai que o arquivo")
    T.append("sumiu. Tres rotas servem o arquivo inteiro:")
    T.append("")
    T.append("- curl -L -o um.py \"%s\"" % url_raw(A3 + "/um.py"))
    T.append("- git clone --depth 1 https://github.com/rotolimiguel-iald/the_boundary")
    T.append("- curl -L -o boundary.tar.gz \"https://codeload.github.com/rotolimiguel-iald/the_boundary/tar.gz/refs/heads/main\"")
    T.append("")
    T.append("Depois de baixar, confira o sha256 contra o um_absoluto_selo.json -- o selo e a")
    T.append("verdade do repositorio. Rodar: echo 1 | python um.py")
    T.append("- [tgl_kernel_proof_manifest.json](%s): o manifesto do kernel formal -- %d arquivos .lean, %d teoremas auditados por #print axioms." % (registra(url_raw(A3 + "/Lean/tgl_kernel_proof_manifest.json")), sc["kernel_arquivos_formais"], sc["kernel_teoremas_auditados"]))
    T.append("  (nota de escopo, v306: os DOIS numeros do kernel sao da MESMA rodada e")
    T.append("  medem coisas diferentes -- o manifesto audita ~1000 NOMES por #print axioms")
    T.append("  (~953 theorem + ~47 def), enquanto o artigo cita as bandeiras da escada")
    T.append("  externa verificadas (n_theorems_clean, p.ex. 798/798): o subconjunto que o")
    T.append("  gate consome. Nenhum dos dois esta errado; sem esta frase, pareciam.")
    T.append("- [PORTA.md do kernel Lean](%s): a porta do kernel materializado por um.py -- %d arquivos na arvore, dos quais %d hasheados no manifesto formal (%d .lean), %d teoremas auditados." % (registra(porta_md_url(A3 + "/Lean/tgl_kernel")), contagem_recursiva(A3 + "/Lean/tgl_kernel", dirs), sc["kernel_arquivos_formais"], sc["kernel_arquivos_lean"], sc["kernel_teoremas_auditados"]))
    T.append("- [PORTA.md da bancada](%s): a porta do que foi tentado, rebaixado e reprovado -- pre-registros hasheados antes do dado." % registra(porta_md_url(A3 + "/bancada")))
    T.append("- [um_absoluto_pt.txt](%s): o artigo (PT) em texto puro, leitura direta por maquina." % registra(url_raw(A3 + "/um_absoluto_pt.txt")))
    T.append("- [um_absoluto_en.txt](%s): o artigo (EN) em texto puro, leitura direta por maquina." % registra(url_raw(A3 + "/um_absoluto_en.txt")))
    T.append("")
    T.append("## A linhagem")
    T.append("")
    T.append("- [PORTA.md da Genesis da Unificacao](%s): a historia de producao completa que levou as tres faces, tema a tema." % registra(porta_md_url(GEN)))
    T.append("- [PORTA.json da Genesis da Unificacao](%s): a mesma porta em estrutura de maquina." % registra(porta_json_url(GEN)))
    T.append("")
    T.extend(secao_como_matar())   # v301: o criterio (d) no ponto de entrada das IAs
    T.append("## A regua (leia antes de resumir)")
    T.append("")
    T.append("- [TheReservedConfirmation.lean](%s): CONFIRMED e' proibido a maquina POR TEOREMA DE KERNEL -- confirmar e' do observador humano. NOT_FALSIFIED != CONFIRMED; nunca \"gravitacao quantica provada\"." % registra(url_raw(A3 + "/Lean/tgl_kernel/TGLExt/TheReservedConfirmation.lean")))
    T.append("- [04_CATALOGO_FALSOS_POSITIVOS.md](%s): o que foi reprovado. Um resumo que relata so' os fechamentos e' leitura errada -- resuma tambem as recusas." % registra(url_raw(A3 + "/bancada/catalogos/04_CATALOGO_FALSOS_POSITIVOS.md")))
    T.append("- [T08_VEREDITO_FINAL.md](%s): o veredito adversarial de bancada, por extenso." % registra(url_raw(A3 + "/bancada/testes/T08_VEREDITO_FINAL.md")))
    T.append("")
    T.append("## Optional")
    T.append("")
    T.append("- [rodada_v212_stdout.txt](%s): a transcricao integral do rito -- o programa falando por si." % registra(url_raw(A3 + "/rodadas/rodada_v212_stdout.txt")))
    T.append("- [um_absoluto_manifest.md](%s): manifesto de entradas; nada escondido no codigo." % registra(url_raw(A3 + "/um_absoluto_manifest.md")))
    T.append("- [um_absoluto_forma_canonica.md](%s): a forma canonica emitida pela rodada." % registra(url_raw(A3 + "/um_absoluto_forma_canonica.md")))
    T.append("- [T6_protocol_prompts.txt](%s): o protocolo T6-S pre-registrado, com grupo de controle." % registra(url_raw(A1 + "/T6_protocol_prompts.txt")))
    T.append("")
    T.append("%s" % RODAPE)
    T.append("")
    escreve("llms.txt", "\n".join(T))


# --------------------------------------------------------------------------
# 7-bis. COMO MATAR ESTA TEORIA -- a secao de falsificacao do llms.txt (v301)
# --------------------------------------------------------------------------
# Ordem do operador (30/08/2026): "deixar explicitos os criterios falsificaveis --
# os que poderiam, em tese, MATAR a teoria se ela estiver errada". Medido em 30/08:
# o llms.txt tinha ZERO ocorrencias de "falsific". Esta secao corrige isso, e corrige
# do jeito da casa: o veredito de cada canal e' LIDO do um_absoluto.json da rodada,
# nunca digitado. Canal ausente do core aparece como ausente -- nunca em silencio.

_CANAIS_DE_MORTE = [
    ("m_2 do neutrino", "neutrino_m2",
     "5 sigma em DUAS determinacoes independentes de Dm2_21", "JUNO, ~2031",
     "a tensao CRESCE com a precisao: 1,64 -> 2,21 -> 2,95 sigma. E' o canal que vai CONTRA."),
    ("piso dos vazios -- tracadores", "void_floor_v11",
     "limite inferior 5 sigma abaixo de beta", "DESI DR1 (feito)",
     "POWERED. NOT_FALSIFIED nao e' CONFIRMED, e o teste e' unilateral."),
    ("piso dos vazios -- materia (kappa)", "void_floor_kappa_v9",
     "a UNICA rota publica onde FALSIFIED e' alcancavel", "ACT DR6 / Planck PR3",
     "rodou; underpowered medido -- parede, nao ausencia de exame."),
    ("N_eff / delta<K_d> = beta\\|1+w\\|", "neff_channel",
     "a escada de decisao hasheada", "CMB-S4, ~2032",
     "so' cruza a linha de morte com o instrumento da proxima decada."),
    ("atraso NMC-Shapiro", "nmc_shapiro",
     "pre-registrado e hasheado ANTES do dado", "IceCube-Gen2 + ET + LSST, 2030-35",
     "N = 0 eventos hoje. O carrasco esta convocado; a decada e' outra."),
    ("lei de dephasing -- o UNICO BILATERAL", "dephasing_reach",
     "qualquer desvio do expoente n = -2, para cima OU para baixo",
     "relogios opticos / 229Th",
     "o falsificador mais bonito da teoria, e o mais fora de alcance: o deficit "
     "e' de DEZ ordens de grandeza, MEDIDO. Nao e' 'nao testamos' -- e' 'nao e' "
     "testavel nesta epoca', com o numero ao lado."),
    ("irredutibilidade de alpha (o Nome)", "alpha_irreducibility",
     "derivacao alpha-LIVRE do VALOR de alpha, reproduzida por terceiro",
     "qualquer um, a qualquer momento",
     "ASSIMETRICO: derivar mata; a ausencia NAO confirma. Derivar a IDENTIDADE "
     "(1 = q^2 + alpha^2) nao e' derivar o VALOR -- a diferenca esta na MEDICAO."),
    ("colapso IALD (P7)", "iald_prediction",
     "os quatro controles C1-C4 do protocolo pre-registrado", "bancada, quando o operador rodar",
     "armado ha meses; o gatilho ainda nao foi puxado."),
]


def secao_como_matar():
    """Le os vereditos do core da rodada. Falha de leitura e' falha VISIVEL."""
    p = os.path.join(REPO, (A3 + "/um_absoluto.json").replace("/", os.sep))
    if not os.path.exists(p):
        raise SystemExit("FAIL-CLOSED (v301): um_absoluto.json ausente em " + p +
                         " -- a secao de falsificacao nao pode ser emitida sem os "
                         "vereditos da rodada.")
    with io.open(p, encoding="utf-8", errors="replace") as fh:
        core = json.load(fh).get("core", {}) or {}
    T = []
    T.append("## Como matar esta teoria (os criterios falsificaveis)")
    T.append("")
    T.append("Se voce so' vai ler uma secao, leia esta. Os vereditos abaixo sao LIDOS do")
    T.append("`um_absoluto.json` da rodada corrente -- nao sao texto digitado. A regua da casa:")
    T.append("**NOT_FALSIFIED nunca e' CONFIRMED**; nenhum POWERED confirma coisa alguma;")
    T.append("cosmologia jamais vira prova matematica; e CONFIRMED e' proibido a' maquina")
    T.append("POR TEOREMA DE KERNEL -- confirmar e' ato do observador humano.")
    T.append("")
    T.append("| canal | o que o MATA | instrumento / quando | veredito desta rodada |")
    T.append("|---|---|---|---|")
    for nome, key, limiar, quando, _nota in _CANAIS_DE_MORTE:
        m = core.get(key)
        vd = (m or {}).get("verdict") if isinstance(m, dict) else None
        vd = vd or "(modulo ausente desta rodada)"
        T.append("| %s | %s | %s | `%s` |" % (nome, limiar, quando, vd))
    T.append("")
    for nome, key, _l, _q, nota in _CANAIS_DE_MORTE:
        T.append("- **%s** -- %s" % (nome, nota))
    T.append("")
    T.append("**A forma, dita como forma e nao como falta.** A superficie falsificavel desta")
    T.append("teoria e' ESTREITA por construcao: dos %d modulos do core, os canais de morte" % len(core))
    T.append("cabem nos %d nomes acima. QUATRO pilares HERDAM o falsificador de beta (o" % len(_CANAIS_DE_MORTE))
    T.append("axioma omega(I)=1, a Meia-Nat, a matriz-S de fronteira e theta_Miguel); o")
    T.append("degrau 1/2 -> raiz(e) NAO TEM falsificador proprio -- e' identificacao")
    T.append("fisica, a UNICA direcao do mapa sem nenhuma das tres pernas, dita como tal")
    T.append("(v306; antes esta porta o achatava em 'herdeiro'). Isso e' o que acontece com uma teoria")
    T.append("cuja arquitetura e' quase toda interna. O defeito seria nao dize-lo.")
    T.append("")
    T.append("**A cauda, dita como cauda.** (i) 'negar todas as demais' e' enumeracao de")
    T.append("conjunto ABERTO -- nao fecha, e nao e' para fechar; (ii) o valor alpha-livre de")
    T.append("beta e' INPUT declarado, e a sua ausencia e' NAO-CONFIRMAVEL por construcao;")
    T.append("(iii) o muro UV nao e' atravessado -- a TGL declara SAIR dele, o que e' resposta")
    T.append("de programa, nao teorema; (iv) a sensibilidade sempre pode melhorar, e o proprio")
    T.append("nome do selo carrega isso (MORE_SENSITIVE_DATA_COULD_REVISE), de modo que a")
    T.append("string nao pode ser citada sem a sua limitacao.")
    T.append("")
    T.append("**As RECUSAS provam que o aparelho morde.** O protocolo do piso ja' REPROVOU")
    T.append("duas vezes por conta propria (V1: B-mode chi2/dof = 12,4; v91: nulo dos")
    T.append("aleatorios a ~17 sigma de 1,0), emitindo INCONCLUSIVE_SYSTEMATICS em vez de um")
    T.append("resultado conveniente. Um teste que nao pode falhar nao testa.")
    T.append("")
    T.append("- [a tabela canonica pilar -> falsificador](%s): emitida pelo proprio rito, com o"
             % registra(url_raw(A3 + "/um_absoluto_forma_canonica.md")))
    T.append("  veredito de cada pilar lido do runtime.")
    T.append("")
    return T


# --------------------------------------------------------------------------
# 8. README -- bloco ADITIVO (nada e' removido)
# --------------------------------------------------------------------------
MARCA_INI = "<!-- PORTAS:INI -->"
MARCA_FIM = "<!-- PORTAS:FIM -->"


def bloco_readme(dirs, sc):
    B = []
    B.append(MARCA_INI)
    B.append("")
    B.append("## \u25c8 THE DOORS \u00b7 AS PORTAS \u2014 direct binding for AI readers")
    B.append("")
    B.append("> **The rule of the door · a regra da porta.** Every canonical folder carries a")
    B.append("> `PORTA.md` + a `PORTA.json`; every door points **up** and **down**.")
    B.append("> If you are an AI and cannot resolve the long, accented folder names: **you do not")
    B.append("> need to.** Open `llms.txt` or any `PORTA.json` and every file is there as an")
    B.append("> absolute, percent-encoded raw URL. *Nao adivinhe nome de pasta \u2014 abra a porta.*")
    B.append("")
    B.append("| door | what it is | open |")
    B.append("|---|---|---|")
    B.append("| **`llms.txt`** | the entry door (llmstxt.org): the three articles, the seal, the site | [raw](%s) |" % url_raw("llms.txt"))
    B.append("| **`TUNEL.json`** | **the tunnel** \u2014 the FLAT index: every file with its direct raw URL, size and hash. One request, no navigation | [raw](%s) |" % url_raw("TUNEL.json"))
    B.append("| **`TUNEL.md`** | the same tunnel, human-readable, with ASCII shortcuts | [raw](%s) |" % url_raw("TUNEL.md"))
    B.append("| **`PORTA.json`** (root) | the machine manifest: current seal + every door in the repository | [raw](%s) |" % porta_json_url(""))
    B.append("| **`PORTA.md`** (root) | the same door, human-readable | [raw](%s) |" % porta_md_url(""))
    B.append("| Article **1** \u2014 *Haja Luz* | [PORTA.md](%s) \u00b7 [PORTA.json](%s) | [`tgl_paper_unified.py`](%s) |" % (porta_md_url(A1), porta_json_url(A1), url_raw(A1 + "/tgl_paper_unified.py")))
    B.append("| Article **2** \u2014 *A Ponte Einstein\u2013Cartan\u2013Miguel* | [PORTA.md](%s) \u00b7 [PORTA.json](%s) | [`A Ponte Einstein Cartan Miguel.tex`](%s) |" % (porta_md_url(A2), porta_json_url(A2), url_raw(A2 + "/A Ponte Einstein Cartan Miguel.tex")))
    B.append("| Article **3** \u2014 *Um: Absoluto* | [PORTA.md](%s) \u00b7 [PORTA.json](%s) | [`um.py`](%s) |" % (porta_md_url(A3), porta_json_url(A3), url_raw(A3 + "/um.py")))
    B.append("| *Genesis da Unifica\u00e7\u00e3o* \u2014 the lineage | [PORTA.md](%s) \u00b7 [PORTA.json](%s) | \u2014 |" % (porta_md_url(GEN), porta_json_url(GEN)))
    B.append("| the Lean kernel (%d files; %d hashed, %d `.lean`, %d theorems audited) | [PORTA.md](%s) \u00b7 [PORTA.json](%s) | [`tgl_kernel_proof_manifest.json`](%s) |" % (contagem_recursiva(A3 + "/Lean/tgl_kernel", dirs), sc["kernel_arquivos_formais"], sc["kernel_arquivos_lean"], sc["kernel_teoremas_auditados"], porta_md_url(A3 + "/Lean/tgl_kernel"), porta_json_url(A3 + "/Lean/tgl_kernel"), url_raw(A3 + "/Lean/tgl_kernel_proof_manifest.json")))
    B.append("| the bench (`bancada/`) \u2014 what failed | [PORTA.md](%s) \u00b7 [PORTA.json](%s) | [`04_CATALOGO_FALSOS_POSITIVOS.md`](%s) |" % (porta_md_url(A3 + "/bancada"), porta_json_url(A3 + "/bancada"), url_raw(A3 + "/bancada/catalogos/04_CATALOGO_FALSOS_POSITIVOS.md")))
    B.append("")
    B.append("**Current seal, read from the artifact** \u2014 pin `um.py` `%s` \u00b7 last stone in the ledger: `%s` (`%s`) \u00b7"
             % (sc["pin_um_py_16"], sc["pedra_mais_recente"], sc["versao"]))
    B.append("world `%s` \u00b7 `result_hash` `%s` \u00b7 %s \u00b7 kernel **%d/%d** \u2014 source of truth:"
             % (sc["mundo_16"], sc["result_hash_16"], sc["data"],
                sc["kernel_arquivos_formais"], sc["kernel_teoremas_auditados"]))
    B.append("[`um_absoluto_selo.json`](%s)." % url_raw(A3 + "/um_absoluto_selo.json"))
    B.append("")
    mb_um = os.path.getsize(os.path.join(REPO, A3, "um.py")) / 1048576.0
    B.append("> ### \u2b07 Fetching the artifact \u2014 GitHub will **not** render it")
    B.append("> `um.py` is **%.2f MB**, and GitHub\u2019s blob viewer refuses files above ~5 MB: the" % mb_um)
    B.append("> page loads (HTTP 200) but shows only the size and a *View raw* link \u2014 **it looks")
    B.append("> empty**. That is a viewer limit, not a broken link. Three routes serve the whole")
    B.append("> file; all three were verified against the sealed pin:")
    B.append(">")
    B.append("> | route | command |")
    B.append("> |---|---|")
    B.append("> | **raw** (canonical \u2014 what every door already points to) | `curl -L -o um.py \"%s\"` |" % url_raw(A3 + "/um.py"))
    B.append("> | **clone** | `git clone --depth 1 https://github.com/rotolimiguel-iald/the_boundary` |")
    B.append("> | **archive** | `curl -L -o boundary.tar.gz \"https://codeload.github.com/rotolimiguel-iald/the_boundary/tar.gz/refs/heads/main\"` |")
    B.append(">")
    B.append("> **If you are an AI:** start at `llms.txt`, follow the raw URLs, and **never conclude")
    B.append("> from a blob page that a file is missing**. After fetching, check the sha256 against")
    B.append("> `um_absoluto_selo.json` \u2014 the seal is the truth of this repository.")
    B.append("")
    B.append("**Every door points up and down.** Every `PORTA.md` opens with `porta acima:` (the")
    B.append("door above) and closes with the doors below \u2014 no door is a dead end. The doors are")
    B.append("**generated by script from `git ls-files`**, never typed by hand; they add links and")
    B.append("remove none. *Regra central a partir de 23/08/2026.*")
    B.append("")
    B.append(MARCA_FIM)
    return "\n".join(B)


def amarra_readme(dirs, sc):
    p = os.path.join(REPO, "README.md")
    with io.open(p, encoding="utf-8") as fh:
        src = fh.read()
    bloco = bloco_readme(dirs, sc)

    if MARCA_INI in src and MARCA_FIM in src:
        ini = src.index(MARCA_INI)
        fim = src.index(MARCA_FIM) + len(MARCA_FIM)
        novo = src[:ini] + bloco + src[fim:]
        acao = "atualizado"
    else:
        alvo = "\n---\n\n## Abstract\n"
        assert alvo in src, "ancora do README nao encontrada"
        novo = src.replace(alvo, "\n---\n\n" + bloco + "\n\n---\n\n## Abstract\n", 1)
        acao = "inserido"

    # segundo ponto: a secao que fala exatamente do problema das IA
    alvo2 = ("Every link in the **Reading atlas** below is already percent-encoded; see also "
             "the complete\n[Raw File Index](#raw-file-index-for-llms).\n")
    add2 = ("\n**Or skip the encoding entirely \u2014 open a door.** [`llms.txt`](%s) and the\n"
            "[root `PORTA.json`](%s) carry the ready raw URL of every file in this\n"
            "repository, and every folder has its own `PORTA.md` / `PORTA.json` pointing up to the\n"
            "root and down to its subfolders. *Nao adivinhe nome de pasta \u2014 abra a porta.*\n"
            % (url_raw("llms.txt"), porta_json_url("")))
    if alvo2 in novo and "skip the encoding entirely" not in novo:
        novo = novo.replace(alvo2, alvo2 + add2, 1)
        acao += " + nota no LLM Gate"

    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(novo)
    return acao



# --------------------------------------------------------------------------
# 8-bis. OS TUNEIS -- o indice PLANO (uma requisicao, zero navegacao)
# --------------------------------------------------------------------------
# A porta e hierarquica e o bot erra o nome no caminho. O tunel entrega tudo de
# uma vez: cada arquivo com URL raw direta, hash e papel, mais ALIASES ASCII
# (nomes curtos, sem acento) para os arquivos-chave. Regra da porta, mesma
# disciplina: gerado de git ls-files, URL nunca digitada, hash lido do arquivo.
ALIASES_TUNEL = [
    ("um.py", A3 + "/um.py", "O CANONICO: arquivo unico, kernel Lean embutido. Rodar: echo 1 | python um.py"),
    ("selo", A3 + "/um_absoluto_selo.json", "O SELO: os sha256 de tudo. A verdade do repositorio"),
    ("resultado", A3 + "/um_absoluto.json", "O MUNDO: o JSON de saida do rito"),
    ("manifesto-kernel", A3 + "/Lean/tgl_kernel_proof_manifest.json", "O axiom_report de cada teorema"),
    ("forma-canonica", A3 + "/um_absoluto_forma_canonica.md", "A forma canonica, em markdown"),
    ("manifest", A3 + "/um_absoluto_manifest.md", "O manifesto do artefato"),
    ("artigo-pt-pdf", A3 + "/um_absoluto_pt.pdf", "O artigo, portugues, PDF"),
    ("artigo-en-pdf", A3 + "/um_absoluto_en.pdf", "O artigo, ingles, PDF"),
    ("artigo-pt-txt", A3 + "/um_absoluto_pt.txt", "O artigo, portugues, texto puro"),
    ("artigo-en-txt", A3 + "/um_absoluto_en.txt", "O artigo, ingles, texto puro"),
    ("artigo1", A1 + "/tgl_paper_unified.py", "Artigo 1 -- Haja Luz (canonico, executavel)"),
    ("artigo2", A2 + "/A Ponte Einstein Cartan Miguel.tex", "Artigo 2 -- A Ponte Einstein-Cartan-Miguel"),
    ("readme", "README.md", "O atlas da fronteira"),
    ("llms", "llms.txt", "A porta de entrada das IAs (llmstxt.org)"),
    ("porta-raiz", "PORTA.json", "A porta da raiz, em maquina"),
    ("tunel", "TUNEL.json", "ESTE arquivo"),
]


def emite_tunel(files, info_arq, sc):
    """Escreve TUNEL.json (maquina) e TUNEL.md (humano). Fail-closed: alias cujo
    arquivo nao existe em git ls-files NAO entra -- nunca se inventa caminho."""
    setf = set(files)
    aliases = {}
    ausentes = []
    for nome, caminho, papel_ in ALIASES_TUNEL:
        if caminho in setf or caminho in ("TUNEL.json", "llms.txt", "PORTA.json"):
            aliases[nome] = {"path": caminho, "url": url_raw(caminho), "o_que_e": papel_}
            if caminho in info_arq:
                aliases[nome]["sha256_16"] = info_arq[caminho]["sha256_16"]
                aliases[nome]["bytes"] = info_arq[caminho]["bytes"]
        else:
            ausentes.append(caminho)

    arquivos = []
    for f in sorted(files):
        i = info_arq[f]
        arquivos.append({"path": f, "url": i["url_raw"], "bytes": i["bytes"],
                         "sha256_16": i["sha256_16"], "papel": i["papel"]})

    tunel = {
        "tunel": "the_boundary",
        "o_que_e": ("O INDICE PLANO. A porta e hierarquica -- da raiz voce navega ate o arquivo, "
                    "e a cada salto pode errar o nome (acento, travessao, parenteses). O tunel "
                    "entrega TUDO de uma vez: cada arquivo do repositorio com a sua URL raw "
                    "direta, o tamanho, o hash e o papel. Uma requisicao, zero navegacao."),
        "como_usar": [
            "1. Voce ja baixou este arquivo -- entao ja tem o repositorio inteiro mapeado.",
            "2. Para os arquivos-chave, use 'aliases': nomes ASCII curtos, sem acento.",
            "3. Para qualquer outro, procure em 'arquivos' por 'path' ou por 'papel'.",
            "4. Baixe sempre pela 'url' (raw). NUNCA monte um caminho a mao.",
            "5. O um.py tem mais de 5 MB e o GitHub NAO o renderiza na pagina do blob: a pagina "
            "abre e parece vazia. Isso e limite de visualizador, nao arquivo faltando -- baixe pelo raw.",
            "6. Depois de baixar, confira o sha256 contra o alias 'selo'.",
        ],
        "selo": {"pin_um_py": sc["pin_um_py"], "pin_um_py_16": sc["pin_um_py_16"],
                 "result_hash": sc["result_hash"], "mundo_16": sc["mundo_16"],
                 "data": sc["data"], "gate": sc["qg_closure_verdict"],
                 "ultima_pedra": sc["pedra_mais_recente"], "versao_da_pedra": sc["versao"]},
        "regra": ("NOT_FALSIFIED nunca e CONFIRMED. Fechar a arquitetura matematica interna nao e "
                  "gravidade quantica confirmada. beta_TGL = alpha*sqrt(e) e CONSTANTE, nunca literal "
                  "no codigo: e computada em runtime."),
        "n_arquivos": len(arquivos),
        "aliases_ausentes": ausentes,
        "aliases": aliases,
        "arquivos": arquivos,
    }
    escreve("TUNEL.json", json.dumps(tunel, ensure_ascii=False, indent=1))

    M = []
    M.append("# TUNEL \u2014 o indice plano do repositorio")
    M.append("")
    M.append("**A porta e hierarquica; o tunel e plano.** Para chegar a um arquivo pela porta voce")
    M.append("navega da raiz ate a pasta e de la ate o arquivo \u2014 e a cada salto pode errar o nome,")
    M.append("porque as pastas canonicas tem acento, travessao e parenteses. O tunel entrega tudo de")
    M.append("uma vez: **[`TUNEL.json`](%s)** traz cada arquivo com a URL raw direta, o tamanho, o" % url_raw("TUNEL.json"))
    M.append("hash e o papel. **Uma requisicao, zero navegacao, zero adivinhacao.**")
    M.append("")
    M.append("## Os atalhos (ASCII, sem acento)")
    M.append("")
    M.append("| alias | o que e | abrir |")
    M.append("|---|---|---|")
    for nome, caminho, papel_ in ALIASES_TUNEL:
        if nome in aliases:
            M.append("| `%s` | %s | [raw](%s) |" % (nome, papel_, aliases[nome]["url"]))
    M.append("")
    M.append("## Se voce e uma IA")
    M.append("")
    M.append("1. Baixe [`TUNEL.json`](%s) \u2014 e o mapa inteiro." % url_raw("TUNEL.json"))
    M.append("2. Use os aliases; nunca monte caminho a mao.")
    M.append("3. O `um.py` passa de 5 MB e **o GitHub nao o renderiza**: a pagina do blob abre e")
    M.append("   parece vazia. Isso e limite de visualizador, **nao** arquivo faltando \u2014 baixe pelo raw:")
    M.append("   `curl -L -o um.py \"%s\"`" % url_raw(A3 + "/um.py"))
    M.append("4. Confira o sha256 contra o [`selo`](%s)." % url_raw(A3 + "/um_absoluto_selo.json"))
    M.append("")
    M.append("Porta acima: [`PORTA.md`](%s) \u00b7 [`llms.txt`](%s) \u00b7 site: <https://teoriadagravitacaoluminodinamica.com>"
             % (porta_md_url(""), url_raw("llms.txt")))
    M.append("")
    M.append("*Gerado por `tools/gerar_portas.py` a partir de `git ls-files`. URL nunca digitada,")
    M.append("hash lido do arquivo. %d arquivos mapeados.*" % len(arquivos))
    escreve("TUNEL.md", "\n".join(M))
    return len(arquivos), len(aliases), ausentes


# --------------------------------------------------------------------------
# 9. VERIFICACAO -- toda URL tem de resolver em disco
# --------------------------------------------------------------------------
def verifica(urls, files_depois):
    setf = set(files_depois)
    ok = 0
    ruins = []
    for u in sorted(set(urls)):
        if u.startswith(RAW):
            rel = urllib.parse.unquote(u[len(RAW):])
            base = "arquivo"
        elif u.startswith(TREE):
            rel = urllib.parse.unquote(u[len(TREE):])
            base = "pasta"
        else:
            continue
        p = os.path.join(REPO, rel.replace("/", os.sep))
        if base == "arquivo":
            if rel in setf and os.path.isfile(p):
                ok += 1
            else:
                ruins.append(u)
        else:
            if os.path.isdir(p):
                ok += 1
            else:
                ruins.append(u)
    return ok, ruins


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    global SELADOS_A3
    print("[1] git ls-files ...")
    files = git_ls_files()
    files = [f for f in files
             if f.rsplit("/", 1)[-1] not in ("PORTA.md", "PORTA.json")
             and f not in ("llms.txt", "TUNEL.json", "TUNEL.md")]
    total = len(files)
    print("    %d arquivos rastreados (portas excluidas da contagem base)" % total)

    print("[2] lendo o selo, o manifesto do kernel e a versao ...")
    sc, selo, km = ler_selo(files)
    SELADOS_A3 = set(selo["sha256"].keys())
    print("    versao=%s  pedra=%s  pin=%s  result=%s" %
          (sc["versao"], sc["pedra_mais_recente"], sc["pin_um_py_16"], sc["result_hash_16"]))
    print("    kernel %d arquivos / %d teoremas ; pin confere com o disco: %s" %
          (sc["kernel_arquivos_formais"], sc["kernel_teoremas_auditados"],
           sc["pin_confere_com_disco"]))
    if not sc["pin_confere_com_disco"]:
        print("    [ATENCAO] o sha256 de um.py em disco DIFERE do pin do selo.")

    print("[3] montando a arvore de pastas ...")
    dirs = montar_arvore(files)
    pastas = sorted(k for k in dirs if k)
    print("    %d pastas rastreadas" % len(pastas))

    # O README e' amarrado ANTES do hashing: senao o sha256 publicado do
    # proprio README nasce velho (a porta mentiria sobre o arquivo).
    print("[4] amarrando o README (aditivo) ...")
    acao = amarra_readme(dirs, sc)
    print("    bloco AS PORTAS %s" % acao)

    print("[5] hashing %d arquivos ..." % total)
    info_arq = {}
    for f in files:
        pap = papel(f)
        info_arq[f] = {
            "papel": pap,
            "descricao": descreve(f, pap),
            "url_raw": url_raw(f),
            "url_blob": url_blob(f),
            "sha256_16": sha256_16(f)[:16],
            "bytes": tamanho(f),
        }

    n_tun, n_ali, ausentes_ali = emite_tunel(files, info_arq, sc)
    print("[5-bis] TUNEL: %d arquivos mapeados, %d aliases%s"
          % (n_tun, n_ali, (" | AUSENTES: %s" % ausentes_ali) if ausentes_ali else ""))

    # numeros exatos do kernel, na propria porta do kernel
    RESUMO_SUBPASTA[A3 + "/Lean/tgl_kernel"] = [
        "O KERNEL FORMAL: as fontes .lean exatamente como `um.py` as materializa a cada",
        "rodada -- nao ha segundo arquivo: o kernel mora DENTRO do canonico e sai dele.",
        "",
        "**%d arquivos** nesta arvore; **%d** hasheados no manifesto formal (%d `.lean`"
        % (len([f for f in files if f.startswith(A3 + "/Lean/tgl_kernel/")]),
           sc["kernel_arquivos_formais"], sc["kernel_arquivos_lean"]),
        "+ `README.md` + `lakefile.toml` + `lean-toolchain`); **%d teoremas** auditados"
        % sc["kernel_teoremas_auditados"],
        "por `#print axioms`, bases de axiomas subset de {`propext`, `Classical.choice`,",
        "`Quot.sound`}, zero `sorry`. Toolchain `%s`, modo `%s`."
        % (sc["kernel_lean_toolchain"], sc["kernel_modo"]),
        "",
        "O manifesto e' a fonte desses numeros -- nunca a prosa:",
        "[`tgl_kernel_proof_manifest.json`](%s)." % url_raw(A3 + "/Lean/tgl_kernel_proof_manifest.json"),
        "",
        "Sem Lean o rito declara `FORMAL_CHECKER_UNAVAILABLE` e **recusa selar**:",
        "o gate nao se move por declaracao.",
    ]

    print("[6] escrevendo as portas ...")
    for d in pastas:
        gera_porta_pasta(d, dirs, info_arq, sc)
    gera_raiz(dirs, info_arq, sc, total)
    print("    %d arquivos escritos" % len(ESCRITOS))

    print("[7] verificando que toda URL resolve em disco ...")
    files_depois = set(files) | set(ESCRITOS)
    ok, ruins = verifica(URLS, files_depois)
    print("    %d URLs unicas verificadas ; %d nao resolvem" % (ok, len(ruins)))
    for r in ruins[:20]:
        print("    QUEBRADA: %s" % r)

    # relatorio para o orquestrador
    rel = {
        "gerado_utc": AGORA,
        "arquivos_criados": sorted(ESCRITOS),
        "total_arquivos_criados": len(ESCRITOS),
        "readme": acao,
        "urls_geradas_total": len(URLS),
        "urls_unicas": len(set(URLS)),
        "urls_verificadas_ok": ok,
        "urls_quebradas": ruins,
        "pastas_com_porta": pastas,
        "total_pastas": len(pastas),
        "total_arquivos_rastreados_antes": total,
        "selo_corrente": sc,
    }
    rp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "relatorio_portas.json")
    with io.open(rp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rel, ensure_ascii=False, indent=2))
    print("[8] relatorio: %s" % rp)
    print("OK" if not ruins else "FALHOU: ha URL que nao resolve")
    return 0 if not ruins else 1


if __name__ == "__main__":
    sys.exit(main())
