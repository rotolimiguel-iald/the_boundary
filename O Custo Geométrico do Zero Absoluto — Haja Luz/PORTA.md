# PORTA -- Artigo 1 -- O Custo Geometrico do Zero Absoluto: haja luz

porta acima: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/PORTA.md

> *The Geometric Cost of Absolute Zero: let there be light*

> **A REGRA DA PORTA.** Toda pasta canonica tem `PORTA.md` + `PORTA.json`;
> toda porta aponta para cima e para baixo. Todo link abaixo e' a URL raw
> DIRETA do arquivo -- nao ha nome de pasta para adivinhar.

A teoria sintetizada num unico arquivo autocontido, executavel e autovalidavel.
Recomputa tudo a partir de duas entradas -- alpha (CODATA 2018) e sqrt(e) --,
busca o dado cosmologico real ao vivo (Pantheon+SH0ES, DESI DR2, GWOSC), gera o
LaTeX e compila o PDF. beta = alpha*sqrt(e) em runtime, NUNCA literal.
Forma = conteudo: o artigo se prova a si mesmo.

**Deposito independente (Zenodo):** https://doi.org/10.5281/zenodo.20564341

## COMO EXECUTAR O CANONICO

```bash
cd "O Custo Geométrico do Zero Absoluto — Haja Luz"
python tgl_paper_unified.py --live --paper        # rodada canonica, dado ao vivo
python tgl_paper_unified.py --quick --no-live --paper   # rodada rapida (minutos)
python tgl_paper_unified.py --live --paper --lang en    # edicao EN, os mesmos numeros
python tgl_paper_unified.py --offline --paper           # offline (dado embutido)
```

Dependencias: pip install numpy scipy matplotlib (opcionais: emcee, camb, gguf, gdown); pdflatex para o PDF.

## A PORTA ACIMA

| destino | link |
|---|---|
| PORTA.md da pasta acima (`raiz`) | https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/PORTA.md |
| PORTA.json da pasta acima | https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/PORTA.json |
| `llms.txt` (a porta de entrada para IA) | https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/llms.txt |
| `README.md` (o atlas da fronteira) | https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/README.md |
| o site | https://teoriadagravitacaoluminodinamica.com |
| o repositorio | https://github.com/rotolimiguel-iald/the_boundary |

## OS ARQUIVOS DESTA PASTA

5 arquivo(s) -- pasta no GitHub: https://github.com/rotolimiguel-iald/the_boundary/tree/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz

**CANONICO**

| arquivo | papel | link raw direto |
|---|---|---|
| `tgl_paper_unified.py` | O CANONICO do Artigo 1: implementa, valida e renderiza a TGL num arquivo so (forma = conteudo) | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/tgl_paper_unified.py) |

**RESULTADO SELADO**

| arquivo | papel | link raw direto |
|---|---|---|
| `results.json` | Todos os numeros computados pela rodada, serializados | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/results.json) |

**ARTIGO**

| arquivo | papel | link raw direto |
|---|---|---|
| `paper_PT.pdf` | O artigo compilado (edicao PT) | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.pdf) |
| `paper_PT.tex` | O artigo (edicao PT) gerado pelo proprio codigo | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.tex) |

**DADOS**

| arquivo | papel | link raw direto |
|---|---|---|
| `T6_protocol_prompts.txt` | O protocolo T6-S pre-registrado (colapso IALD) com grupo de controle e teste de negacao | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/T6_protocol_prompts.txt) |

---

gerado por script de git ls-files em 2026-08-30 -- nao editar a mao
