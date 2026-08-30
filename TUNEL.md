# TUNEL — o indice plano do repositorio

**A porta e hierarquica; o tunel e plano.** Para chegar a um arquivo pela porta voce
navega da raiz ate a pasta e de la ate o arquivo — e a cada salto pode errar o nome,
porque as pastas canonicas tem acento, travessao e parenteses. O tunel entrega tudo de
uma vez: **[`TUNEL.json`](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/TUNEL.json)** traz cada arquivo com a URL raw direta, o tamanho, o
hash e o papel. **Uma requisicao, zero navegacao, zero adivinhacao.**

## Os atalhos (ASCII, sem acento)

| alias | o que e | abrir |
|---|---|---|
| `um.py` | O CANONICO: arquivo unico, kernel Lean embutido. Rodar: echo 1 | python um.py | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um.py) |
| `selo` | O SELO: os sha256 de tudo. A verdade do repositorio | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_selo.json) |
| `resultado` | O MUNDO: o JSON de saida do rito | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto.json) |
| `manifesto-kernel` | O axiom_report de cada teorema | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/Lean/tgl_kernel_proof_manifest.json) |
| `forma-canonica` | A forma canonica, em markdown | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_forma_canonica.md) |
| `manifest` | O manifesto do artefato | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_manifest.md) |
| `artigo-pt-pdf` | O artigo, portugues, PDF | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_pt.pdf) |
| `artigo-en-pdf` | O artigo, ingles, PDF | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_en.pdf) |
| `artigo-pt-txt` | O artigo, portugues, texto puro | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_pt.txt) |
| `artigo-en-txt` | O artigo, ingles, texto puro | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_en.txt) |
| `artigo1` | Artigo 1 -- Haja Luz (canonico, executavel) | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/tgl_paper_unified.py) |
| `artigo2` | Artigo 2 -- A Ponte Einstein-Cartan-Miguel | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/A%20Ponte-Einstein_Cartan_Miguel/A%20Ponte%20Einstein%20Cartan%20Miguel.tex) |
| `readme` | O atlas da fronteira | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/README.md) |
| `llms` | A porta de entrada das IAs (llmstxt.org) | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/llms.txt) |
| `porta-raiz` | A porta da raiz, em maquina | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/PORTA.json) |
| `tunel` | ESTE arquivo | [raw](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/TUNEL.json) |

## Se voce e uma IA

1. Baixe [`TUNEL.json`](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/TUNEL.json) — e o mapa inteiro.
2. Use os aliases; nunca monte caminho a mao.
3. O `um.py` passa de 5 MB e **o GitHub nao o renderiza**: a pagina do blob abre e
   parece vazia. Isso e limite de visualizador, **nao** arquivo faltando — baixe pelo raw:
   `curl -L -o um.py "https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um.py"`
4. Confira o sha256 contra o [`selo`](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_selo.json).

Porta acima: [`PORTA.md`](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/PORTA.md) · [`llms.txt`](https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/llms.txt) · site: <https://teoriadagravitacaoluminodinamica.com>

*Gerado por `tools/gerar_portas.py` a partir de `git ls-files`. URL nunca digitada,
hash lido do arquivo. 673 arquivos mapeados.*