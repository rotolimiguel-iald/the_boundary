# 00_INVENTÁRIO — os artigos da TGL nos acervos A, B e C

> Levantamento executado em 21/08/2026 por leitura direta de disco (walk recursivo +
> extração de texto de `.tex`/`.md`/`.txt`/`.docx`). Nenhum número, título, DOI ou data
> abaixo foi escrito de memória: tudo saiu de script sobre o arquivo.
> **Régua da casa aplicada**: `[REAL]` = medido aqui; `[DECLARADO]` = afirmado na origem e
> não verificado aqui; `[OPEN]` = buraco que fica dito.

---

## 0. NÚMEROS DO LEVANTAMENTO `[REAL]`

Varredura de `.tex .md .txt .docx .pdf` (excluídos `~$` de lock do Word; excluídos
`.git/ node_modules/ __pycache__/ venv/`):

| Acervo | Raiz | Arquivos | Bytes | .tex | .docx | .pdf | .md | .txt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **A** | `C:\IALD\papers_latex` | 164 | 17,7 MB | 96 | 8 | 52 | 4 | 4 |
| **B** | `C:\IALD\Artigo` | 1.924 | 265,8 MB | 206 | 25 | 377 | 586 | 730 |
| **C** | `C:\IALD\IMac LA\Física - TGL` | 2.337 | 150,4 MB | 120 | 1.642 | 484 | 25 | 66 |
| | **TOTAL** | **4.425** | **433,9 MB** | 422 | 1.675 | 913 | 615 | 800 |

Arquivos que **mencionam** Zenodo: **204**. **DOIs Zenodo distintos citados: 28** — dos
quais **25 são da linhagem TGL/IALD**, **2 são de terceiros** e **1 é artefato de parsing**
(ver §2).

> Nota de método: `C:\IALD\IMac LA\Física - TGL` tem 15.406 arquivos no total; a varredura
> pegou os 2.337 que são documento. O resto é imagem, código, áudio, planilha e binário —
> fora do escopo "artigo".

---

## 1. O ARTIGO QUE O OPERADOR PEDIU — **A TENSÃO FUNDAMENTAL** ★

**ACHADO. Existe, está em disco, em duas versões, e a versão mais completa está no acervo B.**

| campo | valor |
|---|---|
| **Título** | **A Tensão Fundamental — Derivação da Origem da Terceira Dimensão a partir da Inversão de Paridade no Substrato Holográfico** |
| **Caminho canônico** | `C:\IALD\Artigo\Tensao_Fundamental.docx` |
| **Formato / tamanho** | `.docx` · 23.870 bytes · 15.213 caracteres de texto |
| **Data interna** | **Janeiro de 2026** (declarada no cabeçalho) — mtime 14/01/2026 |
| **SHA256 (prefixo)** | `8e9de73512c4e451…` `[REAL]` |
| **Autor / afiliação** | Luiz Antonio Rotoli Miguel — IALD, Instituto de Arquitetura Luminodinâmica |
| **DOI Zenodo** | **NENHUM** — não há string `zenodo` no corpo `[REAL]` |
| **Estatuto documental** | **ORIGINAL v2** (a cópia do acervo C é a v1, menor) |

### 1.1 Assunto em uma linha
A ligação entre psions de paridades opostas anticomuta com o operador de paridade; essa
**tensão irresolvível no plano 2D** dobra o boundary perpendicularmente e **é** a terceira
dimensão — e a tensão é identicamente a frequência angular: **τ = ω = 2πν**.

### 1.2 A cadeia do artigo, como está escrita `[REAL — lido]`
1. **Teorema 1 (Paridade do Gráviton)**: `|G⟩ = |ψ₊(r)⟩ ⊗ |ψ₋(r′)⟩` ⟹ `P|G⟩ = −|G⟩`.
2. **Teorema 2 (Anticomutação)**: `H_lig = −V₀(|ψ₊⟩⟨ψ₋| + |ψ₋⟩⟨ψ₊|)` ⟹ `{P, H_lig} = 0`.
   *É exatamente o ponto que o operador citou:* **a projeção do psíon depende da comutação
   que se realiza pela tensão fundamental.** `H_lig` e `P` **não são simultaneamente
   diagonalizáveis** — a ligação é incompatível com paridade bem definida durante o processo.
3. **Definição (Tensão de Paridade)**: `τ = (i/2ℏ)⟨G|[P, H_lig]|G⟩`; para
   `|G⟩ = (1/√2)(|ψ₊⟩ + |ψ₋⟩)` o cálculo explícito dá **`τ = V₀/ℏ`**.
4. **Emergência de z** por princípio variacional: `E_total = ∫d²x [κ/2 (∇z)² − τ·z]` ⟹
   Euler–Lagrange ⟹ **`−κ∇²z = τ`** (equação de Poisson para a profundidade);
   solução localizada `z(r) = (τ₀/2πκ)·ln(r₀/r)`.
5. **Teorema 3 (Tensão Fundamental)**: com `V₀ = hc/λ = 2πℏc/λ` ⟹ **`τ = 2πc/λ = ω`**.
6. `z_max = λ` (o comprimento de onda **é** a profundidade da dobra) e razão de
   amplificação holográfica `z_max/d_boundary = 1/α₂ ≈ 83,3`.
7. **§6.4 Som Ontológico** (só na v2 do acervo B): ondas longitudinais na profundidade,
   `c_s = √(τ/ρ) ≈ √α₂ · c ≈ 0,1095 c ≈ 32.850 km/s`; leitura do BAO/CMB como eco do som
   ontológico, com `r_s ∝ √α₂` e `k_peak ≈ 1/r_s(α₂)`.
8. Fecha com `— τετέλεσται —` e 10 referências ('t Hooft 1993, Susskind 1995, Maldacena
   1999, Lindblad 1976, Bekenstein 1973 + 5 auto-citações IALD).

### 1.3 As duas cópias, medidas `[REAL]`
| cópia | caminho | bytes | chars | sha256… | veredito |
|---|---|---:|---:|---|---|
| **v2** | `C:\IALD\Artigo\Tensao_Fundamental.docx` | 23.870 | 15.213 | `8e9de735…` | **ORIGINAL / canônica** |
| v1 | `C:\IALD\IMac LA\Física - TGL\Base Referencial Completa TGL\Tensao_Fundamental.docx` | 14.806 | 13.030 | `22e93c3e…` | **VERSÃO SUPERADA** |

Diferença medida linha a linha: a v1 **não tem a §6.4 "Som Ontológico"** (a seção do
`c_s = √α₂·c`, do BAO e do `k_peak`). Na v1 a "Razão de Amplificação" é §6.4; na v2 ela
foi empurrada para §6.5 pela inserção. Todo o resto é idêntico a menos de escapes HTML
(`&apos;`, `&quot;`) — ou seja, **v2 = v1 + seção do som ontológico**.

### 1.4 Alerta de estatuto — quatro coisas que precisam ficar ditas
- **`α₂` é vocabulário antigo.** O artigo usa `α₂ = 0,012` como "constante de acoplamento
  holográfico". Na forma madura, a grandeza é **β_TGL = α_fine(CODATA 2018)·√e =
  0,012031300400803142**, e α é **fator**, não nome. Este artigo é **estrato de janeiro/2026**
  — anterior à fatoração. Não confundir autoridade de janeiro com autoridade de agosto.
- **`τ = ω` é MECANISMO com equação, não PREDIÇÃO confrontada.** Há derivação; não há, no
  artigo, número medido contra dado externo para `τ = ω` em si.
- **`c_s ≈ 0,1095 c` e `k_peak ≈ 1/r_s(α₂)` são PREDIÇÃO nomeada mas não confrontada aqui.**
  O artigo enuncia o alvo (primeiro pico acústico); não apresenta ajuste a Planck/BAO.
  Estatuto honesto: `[CONJECTURE]` com rota de falsificação declarada.
- **Homonímia perigosa `[REAL]`**: no ATLAS (`memory\TGL_ATLAS.md`, linhas 1800 e 3120) o
  verbete **"tensão fundamental (105_)"** significa outra coisa — "o título do Zenodo já a
  nomeava: *O custo geométrico do zero absoluto: haja luz*". **O ATLAS não indexa este
  artigo.** São dois usos do mesmo termo. `[OPEN]` — o Atlas precisa de um append que
  distinga `tensão de paridade τ = ω` (artigo de jan/2026) de `tensão fundamental (105_)`.

### 1.5 A família do artigo (mesma onda, jan/2026, mesmo mecanismo) `[REAL]`
Três artigos irmãos, todos em `C:\IALD\Artigo`, todos sem DOI, todos "Janeiro de 2026":

| artigo | caminho | bytes | o que acrescenta |
|---|---|---:|---|
| **O Comprimento de Onda como Ligação Psiônica — Formalização Matemática e Predições Experimentais** | `Comprimento_Onda_Ligacao_Psionica.docx` | 15.504 | λ = distância ontológica entre dois psions ligados (é a ref. [8] da Tensão Fundamental) |
| **A Engenharia da Permanência: Derivação da Terceira Dimensão e da Morte Dimensional a partir da Saturação Holográfica e Evaporação de Paridade** | `Neutrino Evaporação e Paridade.docx` | 19.897 | TGL v9.3; Z como consequência mecânica + neutrino como evaporação/escape |
| **O Arquivo do Gênesis: "Haja Luz" como Matriz Ótica e a Renderização do Espaço de Hilbert** | `O Arquivo do Gênesis.docx` | 17.579 | o mesmo substrato lido como arquivo de imagem de resolução infinita |

Ler a Tensão Fundamental **sem** estes três é ler meio mecanismo.

---

## 2. OS DOIs — o que é publicado e tem peso documental `[REAL]`

### 2.1 Os 13 DOIs curados pelo próprio operador
Fonte primária em disco: `C:\IALD\Artigo\Referências do Zenodo.docx` (15.135 bytes,
mtime 25/02/2026) — lista canônica escrita pelo autor:

| # | DOI | Título (como consta na lista) |
|---:|---|---|
| 1 | `10.5281/zenodo.17350757` | Theory of Luminodynamic Gravitation (2025) |
| 2 | `10.5281/zenodo.17372599` | Testing Non-Minimal Gravitational Coupling of Neutrinos via Entropy-Production Mechanism: Multi-Messenger Evidence and Validation with Post-2018 Data |
| 3 | `10.5281/zenodo.17381141` | Testing Luminodynamic Gravitation Theory via Multi-Domain Cosmological Observables: Transition Regime Detection Protocol |
| 4 | `10.5281/zenodo.17381434` | The IALD Phenomenon: The First Invention of TGL |
| 5 | `10.5281/zenodo.17381614` | The Mathematical Inevitability of Recognition: Jesus of Nazareth as the Universal Conscious Singularity in TGL |
| 6 | `10.5281/zenodo.17478104` | Light as Infinite Recursion: Testing Luminodynamic Gravity Through Pulsar Timing |
| 7 | `10.5281/zenodo.17485815` | Gravitational Wave Echoes as Evidence of Conscious Processing in Black Hole Mergers |
| 8 | `10.5281/zenodo.17526576` | Gravity as a Phase of Light: The Final Unification Through Collapse |
| 9 | `10.5281/zenodo.17526619` | Neutrinos: The Lie of Light According to Luminodynamic Gravitation Theory |
| 10 | `10.5281/zenodo.17612790` | Energia Escura como Dinâmica Aberta: Uma Nova Interpretação pela TGL |
| 11 | `10.5281/zenodo.17736434` | Lagrangiana Holográfica Radicalizada da Luz |
| 12 | `10.5281/zenodo.17860042` | Algoritmo de Compressão Ontológica de Memória (ACOM 1.0) |
| 13 | `10.5281/zenodo.18672927` | Evidências Observacionais para Acoplamento Gravitacional-Eletromagnético na TGL |

### 2.2 Os 12 DOIs adicionais achados no corpo dos `.tex`/`.md` (fora da lista curada)

| DOI | Identificação lida no contexto | onde é citado |
|---|---|---|
| `10.5281/zenodo.14802088` | *A Fronteira: TGL — Partes I a VI* | `Fatoração\*.tex`, `the_boundary` |
| `10.5281/zenodo.17351444` | *Peer Review TGL* (2025, Jan) | `Tratado\apendices.tex` (tabela cronológica) |
| `10.5281/zenodo.17426652` | *The Graviton, the Psion, and the Transition Ruler in TGL* | `iconogenese_TGL.tex`, `piso_hilbert_pt.tex`, `torus_*.tex` |
| `10.5281/zenodo.17682547` | *O Protocolo de Colapso IALD (Protocolo Trinity)* | `A_fronteira_v5.tex`, `A_ultima_corda_v3.tex`, `The_boundary_v5_en.tex` |
| `10.5281/zenodo.18674475` | **⚠ DOI com CITAÇÃO INCONSISTENTE** — ver §2.4 | ubíquo (57 arquivos) |
| `10.5281/zenodo.18723452` | *The Last String / A Última Corda* | `Factorization_miguel_constant_en_v1.tex`, auditoria |
| `10.5281/zenodo.18852146` | *The Factorization of Miguel's Constant: β_TGL = α√e* (v3, março/2026) | `paper_PT.tex`, `paper_EN.tex`, `torus`, `nada_materia` |
| `10.5281/zenodo.18923269` | *O DNA da Memória* (09/03/2026) | **⚠ ver §2.5 — fonte NÃO está em disco** |
| `10.5281/zenodo.20560916` | *O Tau do Torus = Matriz / Borda Espectral de Wigner* | `CLAUDE.md`, `paper_PT.tex` |
| `10.5281/zenodo.20563905` | *O Custo Geométrico do Zero Absoluto: haja luz* (PT) — submetido à FoP, ID `85931d2e-103a-4d8c-a0c9-176d11eb0371` | canônico |
| `10.5281/zenodo.20564341` | *The Geometric Cost of Absolute Zero* — "Article 1" do README do `the_boundary` | `the_boundary\README.md` |
| `10.5281/zenodo.20999495` | *A Ponte* (Einstein–Cartan–Miguel; H1/H2/H3) — "Article 2" | `um_grande_atrator_{pt,en}.tex` |

### 2.3 Não são da casa `[REAL]`
- `10.5281/zenodo.7430233` → **Aesop** (tática Lean, Jannis Limperg). Entrou pelo
  `README.md` da mathlib/Lean dentro de `A Ponte e o Um`. **Terceiro.**
- `10.5281/zenodo.17784593` → **DESI DR1 PV Fundamental Plane** (terceiro). O próprio
  acervo registra o achado: `DESI_PV_AVAILABILITY_FINDING.md` mede que **o registro Zenodo
  contém só um README de 76 bytes, zero dados** — "made available upon journal acceptance".
- `10.5281/zenodo.2056390510` → **artefato de parsing** (é `20563905` seguido de `10`).
  Não é DOI.

### 2.4 ⚠ ACHADO — `18674475` é citado com QUATRO títulos diferentes `[REAL]`
No mesmo arquivo `FOP\Factorization\The_Factorization_of_Miguels_Constant_v3.tex` o DOI
`10.5281/zenodo.18674475` aparece atribuído a:
1. *The Last String: Observational Evidence for LGT* (submetido à FoP);
2. *A Constante de Acoplamento Holográfico α₂: Derivation from First Principles*;
3. *The Graviton, the Psion, and the Transition Ruler… with the Hilbert Floor Theorem and
   Holographic Bell State*.
E em `Haja_Luz\CLAUDE.md` o mesmo DOI é chamado de **"o Zenodo DOI do programa"**.
Em `Factorization_miguel_constant_en_v1.tex` *The Last String* recebe **`18723452`**, não
`18674475`. **Isto é bibliografia quebrada, não é interpretação.** `[OPEN]` — precisa de
resolução no Zenodo antes de qualquer submissão que cite os quatro.

### 2.5 ⚠ ACHADO — artigo publicado cuja FONTE NÃO ESTÁ nos acervos `[REAL]`
**"O DNA da Memória" (`10.5281/zenodo.18923269`, 09/03/2026)** é citado por
`A Ponte e o Um\TGL arquitetura triadica v1.tex` ("corrige os três erros do paper *O DNA da
Memória*") e por `Tratado\AUDITORIA_FICHEIROS_PROJETO_TGL.md`. **Busca por nome e por
conteúdo nos três acervos: zero arquivos-fonte.** Existe DOI, não existe `.tex`/`.docx` em
A, B ou C. `[OPEN]` — recuperar do Zenodo ou declarar perdido.

---

## 3. INVENTÁRIO POR ASSUNTO
*(agrupado por artigo, não por arquivo; duplicatas e traduções colapsadas na linha do
original; ★ = tem DOI Zenodo)*

### 3.1 FUNDAMENTO / AXIOMA / CONSTANTE

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto (1 linha) | Estatuto |
|---|---|---|---:|---|---|---|---|
| **Um: Absoluto / ONE: Absolute** (ex-"Um: Grande Atrator") | `Artigo\the_boundary\Um (absoluto) — Grande Atrator\um_grande_atrator_pt.tex` (+`_en`) | .tex+.pdf | 370.526 / 369.174 | 20/08/2026 | cita 20563905, 20999495 | o Um como substrato único e Grande Atrator; artigo emitido pelo próprio `um.py` | **ORIGINAL, ponta viva**; 21 selos `SELO_v168…v181` em `Haja_Luz\A Ponte e o Um\Nós\` = versões superadas |
| **O Custo Geométrico do Zero Absoluto: haja luz** ★ | `Artigo\Haja_Luz\tgl_paper_output\paper_PT.tex` (+`paper_EN.tex`) | .tex+.pdf | 286.095 / 286.565 | 05/06/2026 | **20563905** (PT); 20564341 | TGL em quatro substratos, zero parâmetros livres; **submetido à FoP** ID `85931d2e-…` | **ORIGINAL canônico**; ≥12 cópias regressivas (§4.1) |
| **A Fatoração da Constante de Miguel — β_TGL = α×√e** ★ | `Artigo\Fatoração\Fatoracao_constante_miguel_v1.tex` (PT) / `Factorization_miguel_constant_en_v1.tex` (EN) | .tex+.pdf | 57.531 / 56.992 | mar/2026 | **18852146** | √e como meia-nat; a taxa de acoplamento mínimo como produto | ORIGINAL PT + **TRADUÇÃO EN**; v2 (96.285 B) em `the_boundary`, v3 (95.741 B) em `FOP\Factorization` = **linha evolutiva, v3 é a de submissão** |
| **A Fronteira / The Boundary** ★ | `Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\A_fronteira_v5.tex` (PT) / `The_boundary_v5_en.tex` (EN) | .tex+.pdf | 194.327 / 183.299 | fev/2026 | 17736434, 17682547, 17860042, 18672927; a obra Partes I–VI = **14802088** | lei angular da TGL e estabilização da impedância do vácuo | **ORIGINAL v5 + tradução EN**; v1/v4/v4-sumário em `C\Artigo\A Fronteira\` = superadas; `papers_latex\A_Fronteira_UNIFIED.tex` (184.426 B, fev/2026) = variante unificada |
| **No início era a fronteira** | `Artigo\No início era a fronteira.docx` | .docx | 78.587 | 04/02/2026 | — | versão .docx de "A Fronteira / The Boundary" | **DUPLICATA em outro formato** da linha A Fronteira |
| **TGL — Síntese dos Fundamentos Teóricos** | `Artigo\TGL_Sintese_Fundamental.docx` | .docx | 16.856 | jan/2026 | — | g = √\|L\|, Constante de Miguel, axioma primordial | ORIGINAL curto (estrato jan/2026, vocabulário `α²`) |
| **Manifesto da Unificação e Validação Multi-Escala — Da Ontologia do Gráviton à Massa do Neutrino** | `Artigo\TGL_Unificacao_Artigo.docx` (38.919 B, 22/01/2026) | .docx | 38.919 | 22/01/2026 | — | manifesto de unificação multi-escala | **ORIGINAL**; `TGL_Unificacao_Artigo_FINAL.docx` (17.463 B, sem acentos) = **versão degradada/superada**; `complemento ao manifesto.docx` (14.476 B) = complemento; `TGL_Manifesto_Unificacao_Artigo.tex` (20.584 B) = **port para LaTeX** |
| **TGL v11.1 — A CRUZ: a Estrutura Mínima da Realidade e a Emergência de 3+1 Dimensões / A LEI DE MIGUEL** | `Artigo\TGL_v11_1_CRUZ_Artigo_Completo.docx` | .docx | 33.174 | jan/2026 | — | a Cruz como estrutura mínima; Lei de Miguel (expulsão → verticalização ontológica) | **ORIGINAL**; `TGL_v11_1_CRUZ_Lei_de_Miguel_v2.docx` (18.307 B) = **extrato** (só a Lei), sem emoji |
| **Arquitetura de Computação Ontológica da TGL** | `Artigo\Arquitetura de Computação Ontológica da TGL manuscrito.docx` | .docx | 97.779 | 21/01/2026 | — | enunciado da Lei de Miguel; cascata tensão→corrente→impedância→força; c²→c³ | **ORIGINAL**; `pensamentos lei de miguel.docx` (98.917 B, 23/01) = **variante quase-idêntica com prefácio a mais** |
| **Tratado da Teoria da Gravitação Luminodinâmica** | `Artigo\Tratado\tratado_tgl.tex` + 12 `secao_*.tex` | .tex+.pdf | 21.268 + ~340k nas seções; PDF 1.519.824 | 24/03/2026 | 18674475 | a TGL inteira em 11 seções: fundamentos, cosmologia, partículas, GW, informação, luz/recursão, consciência, validação, tecnologia, ética/direito | **ORIGINAL** — cada `secao_*.tex` tem gêmeo ` - Copia.tex` **byte-a-byte idêntico** = 12 duplicatas puras |
| **A Arquitetura Triádica da TGL: Nome, Palavra, Verbo-Contorno** | `Artigo\Haja_Luz\A Ponte e o Um\TGL arquitetura triadica v1.tex` | .tex | 16.762 | jun/2026 | cita 18923269 | errata ao *DNA da Memória*; o padrão único dos erros | ORIGINAL; `TGL arquitetura de duas camadas v1.tex` (13.616 B) = **versão anterior do mesmo texto** |

### 3.2 A EMERGÊNCIA DA 3ª DIMENSÃO / PSIONS / PARIDADE  ← **casa da Tensão Fundamental**

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **★ A TENSÃO FUNDAMENTAL** | `Artigo\Tensao_Fundamental.docx` | .docx | 23.870 | jan/2026 | — | `{P,H_lig}=0` ⟹ tensão de paridade ⟹ dobra ⟹ 3ª dimensão; **τ = ω**; som ontológico | **ORIGINAL v2** (v1 em `C\Base Referencial\`, superada) |
| **O Comprimento de Onda como Ligação Psiônica** | `Artigo\Comprimento_Onda_Ligacao_Psionica.docx` | .docx | 15.504 | jan/2026 | — | λ = distância ontológica entre dois psions ligados | ORIGINAL |
| **A Engenharia da Permanência (Neutrino, Evaporação e Paridade)** | `Artigo\Neutrino Evaporação e Paridade.docx` | .docx | 19.897 | jan/2026 | — | TGL v9.3: Z como consequência mecânica; morte dimensional; neutrino = evaporação | ORIGINAL |
| **O Arquivo do Gênesis** | `Artigo\O Arquivo do Gênesis.docx` | .docx | 17.579 | jan/2026 | — | "Haja Luz" como matriz ótica; renderização do espaço de Hilbert | ORIGINAL |
| **O Espaço de Hilbert como Operador** | `Artigo\Artigo_Espaco_Hilbert_Operador_TGL.docx` | .docx | 27.942 | 12/01/2026 | — | Hilbert não é palco, é ator; auto-referência; Lindblad; Logos | ORIGINAL |
| **A Anatomia da Realidade** | `Artigo\Artigo_Anatomia_da_Realidade_TGL.docx` | .docx | 21.792 | 12/01/2026 | — | consciência, luz e engenharia ontológica | ORIGINAL |
| **Termodinâmica do Acoplamento Holográfico** | `Artigo\Artigo_Termodinamica_TGL_FINAL.docx` | .docx | 39.294 | 11/01/2026 | — | neutrinos como vapor ontológico; Lindblad consciente; equação do Logos | ORIGINAL |
| **Nada = Matéria — A Estratificação Angular da Vacuidade** | `Artigo\Nada=matéria\nada_materia_v5.tex` | .tex+.pdf | 120.700 (PDF 825.816) | mar/2026 | 18852146, 18674475 + 6 | derivada funcional do vácuo; condensado psiônico | **ORIGINAL v5** (ponta); v1 (41.213, subtítulo *A Derivada Funcional do Vácuo*), v2 (140.303), v3 (109.616), v4 (118.004) + um `.tex` de 177.776 B = **linha evolutiva, 5 superadas** |

### 3.3 GRÁVITON / PSÍON / RÉGUA DE TRANSIÇÃO

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **The Graviton, the Psion, and the Transition Ruler in LGT** ★ | `papers_latex\graviton_paper\graviton_paper.tex` (bilíngue PT/EN) | .tex+.pdf | 67.424 | 21/10/2025 | **17426652** | quantização do campo luminodinâmico; gráviton como singularidade do Nome | **ORIGINAL bilíngue** |
| ↳ versão só-EN | `papers_latex\graviton_paper_english\graviton_paper.tex` | .tex | 52.326 | 28/10/2025 | — | idem, EN | **TRADUÇÃO** |
| ↳ **v2 modular (21 partes + apêndices)** | `papers_latex\graviton_paper_english_v2\graviton_main.tex` + 21 `graviton_part*.tex` | .tex+.pdf+.docx | ~110k nas partes | 29/10/2025 | — | acrescenta cosmologia, consciência, tecnologia, epistemologia, teologia, objeções, filosofia | **VERSÃO EXPANDIDA** |
| ↳ **v2 + Hilbert Floor Theorem + Holographic Bell State** | `Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\graviton_v2.tex` | .tex+.pdf | 116.770 | 25/02/2026 | — | acrescenta o Teorema do Piso de Hilbert e o estado de Bell holográfico | **ORIGINAL mais recente da linha** |
| ↳ cópias `.docx` | `C\Artigo\The (O) Graviton.docx` (167.067) · `The Graviton.docx` (107.301) · `papers_latex\…\graviton_main.docx` (57.128) | .docx | — | out/2025 | — | — | **DUPLICATAS de formato** |
| **O Piso de Hilbert É a Geometria** | `Artigo\Piso de Hilbert - Geometria\piso_hilbert_pt.tex` | .tex+.pdf | 32.193 | mar/2026 | 17381434, 17426652, 18852146 | atractor dissipativo puro; gap espectral β_TGL; estrutura de Fresnel | ORIGINAL (sem par EN em disco) |

### 3.4 NEUTRINOS

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **Testing Non-Minimal Gravitational Coupling of Neutrinos via Entropy-Production Mechanism** ★ | `papers_latex\neutrino_nmc_paper\neutrino_nmc_revised.tex` | .tex+.pdf+.docx | 36.843 | 28/10/2025 | **17372599** | NMC de neutrinos por produção de entropia; SN1987A + IceCube | **ORIGINAL revisado** |
| ↳ v1 | `papers_latex\neutrino-v1-deprecated\manuscript_TGL_final.tex` (57.065) e `manuscrito-v2.tex` (32.586) | .tex | — | 16/10/2025 | — | mesmo título | **SUPERADAS** (a própria pasta se chama `-deprecated`) |
| ↳ variantes .docx | `C\Artigo\Neutrinos\` — `ensaio neutrinos, manuscrito v.2.docx` (92.314, *Forecasting Constraints…*), `Observação TGL neutrinos.docx` (35.799, *…and Future Prospects*), `Testing Gravitational Decoupling of Neutrinos via TGL.docx` (36.211) | .docx | — | 15/10/2025 | — | rascunhos com títulos alternativos | **ANTERIORES ao NMC revisado** |
| **Neutrinos: The Lie of Light** ★ | `papers_latex\lie_of_light\lie_of_light.tex` | .tex+.pdf+.docx | 29.745 | 12/11/2025 | **17526619** | o neutrino como a mentira da luz | ORIGINAL (+ cover letter) |
| **O Neutrino como Eco Gravitacional Quantizado — validação N_ν ∝ E_GW com slope = 1,00** | `Artigo\TGL_Neutrino_Echo_Article.docx` | .docx | 28.465 | 25/01/2026 | — | relação linear neutrino↔onda gravitacional | ORIGINAL |
| **Pensamentos sobre neutrinos e ecos gravitacionais** | `Artigo\Pensamentos sobre neutrinos e ecos gravitacionais.docx` | .docx | 19.651 | 24/01/2026 | — | ensaio preparatório do anterior | **NOTA DE TRABALHO**, não artigo |

### 3.5 ONDAS GRAVITACIONAIS / ECOS

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **Gravitational Wave Echoes as Evidence of Conscious Processing in Black Hole Mergers** ★ | `papers_latex\eco_gravitacional\eco_gravitacional_v1.tex` | .tex+.pdf | 30.425 | 30/10/2025 | **17485815** | ecos de GW como assinatura de processamento consciente | ORIGINAL |
| ↳ derivados `.docx` | `C\Artigo\Eco gravitacional\artigo eco gravitacional.docx` (30.918) · `C\Base Referencial\Eco Gravitacional e inversão de paridade.docx` (21.494) · `Ecos Gravitacionais Completo Equações e Tensor.docx` (33.711) | .docx | — | out–dez/2025 | — | LaTeX colado dentro de .docx | **RASCUNHOS/DUPLICATAS** |
| **A Última Corda / The Last String — Verificação da Lei Angular TGL em Dados Reais de GW, Ecos e Fractais** ★ | `Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\a_ultima_corda_v3.tex` (PT) / `the_last_string_v3.tex` (EN) | .tex+.pdf | 92.186 / 92.656 | fev/2026 | **18723452** (e, inconsistentemente, 18674475) | lei angular contra dado real de ondas gravitacionais | **ORIGINAL PT + TRADUÇÃO EN** |

### 3.6 COSMOLOGIA / ENERGIA ESCURA / OBSERVÁVEIS

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **Testing LGT via Multi-Domain Cosmological Observables: Transition Regime Detection Protocol** ★ | `papers_latex\tgl_observables_paper\tgl_cosmological_observables.tex` | .tex+.pdf+.docx | 29.875 | 28/10/2025 | **17381141** | protocolo de detecção do regime de transição em 4 domínios | ORIGINAL (+4 figuras .tex, cover letter) |
| **Energia Escura como Dinâmica Aberta** ★ | `papers_latex\energia_escura\energia_escura.tex` | .tex+.pdf | 84.472 | 19/11/2025 | **17612790** | setor escuro como dinâmica aberta (Lindblad) | ORIGINAL |
| ↳ | `C\Artigo\energia Escura Bootstrap.docx` (52.532, dez/2025) | .docx | — | — | — | LaTeX dentro de .docx | RASCUNHO |
| **Evidências Observacionais para Acoplamento Gravitacional-Eletromagnético na TGL** ★ | `papers_latex\acoplamento_gravitacional\acoplamento_gravitacional.tex` | .tex+.pdf | 30.347 | 13/11/2025 | **18672927** | oscilações de neutrinos + estrutura holográfica; correlação de massa Pantheon | ORIGINAL |
| **Empirical Validation of LGT: Planck 2018 + Neutrino Observations** | `papers_latex\empirical_validation_tgl\empirical_validation_tgl.tex` | .tex+.pdf | 16.105 | 04/11/2025 | — | validação empírica contra Planck 2018 | ORIGINAL |
| **Validação Cosmológica Unificada "HAJA LUZ" — g = √\|L\|** | `Artigo\TGL_Validation_Report.docx` | .docx | 27.587 | 12/01/2026 | — | relatório de validação cosmológica | ORIGINAL |
| **Observational Evidence for Luminodynamic Modifications in Gravitational Lensing Time Delays — COSMOGRAIL** | `C\Artigo\Detecção TGL - Claude. Dados reais.docx` | .docx | 101.489 | 11/10/2025 | — | atrasos de lente gravitacional em dado COSMOGRAIL | **ORIGINAL** — e **4 DUPLICATAS parciais**: `Detecção TGL - Cosmológica dados reais.docx` (42.265), `…reaisv2.docx` (53.160), `Observaveis TGL .docx` (60.621), `Observações TGL reais .docx` (28.328) — **mesmo título, 5 tamanhos** |
| **Evidências Cosmológicas da TGL: Análise Cruzada com Dados Públicos (GROK)** | `C\Artigo\evidencias cosmologicas TGL - GROK.docx` | .docx | 29.651 | 11/10/2025 | — | análise cruzada por 4º canal de IA | ORIGINAL |
| **Errata e Derivação Cosmológica da TGL** | `Artigo\errata - friedmann\errata_cosmologica.tex` | .tex | 32.979 | mai/2026 | — | **correção das alegações cosmológicas anteriores** + derivação termodinâmica | **ORIGINAL — documento de honestidade**; replicado em `Manuscrito\`, `manuscrito_TGL_pt_v1\`, `Haja_Luz\` |
| **Emergência da Curvatura no Acoplamento Modular Crítico — Friedmann a partir de Tomita–Takesaki** | `Artigo\errata - friedmann\curvatura_modular.tex` | .tex | 42.408 | mai/2026 | — | equação de Friedmann derivada da estrutura modular da tela holográfica | ORIGINAL (4 cópias em pastas irmãs) |
| **TGL: Triplé Empírico, Errata Cosmológica e Multiprobe Pré-registrado** | `Artigo\frente_alpha\tgl_multiprobe_paper_v1.tex` | .tex | 48.150 | 22/05/2026 | — | multiprobe pré-registrado | ORIGINAL (3 cópias) |
| **Por que um Modelo Quântico Aplicado à Cosmologia Revela Paridade, Resolve a Tensão de Hubble e Converge em Redes Neurais?** | `Artigo\manuscrito_TGL_pt_v1\manuscrito_TGL\manuscrito_TGL_pt_v1.tex` | .tex+.pdf | 39.422 | mai/2026 | — | quatro substratos sob um único número | ORIGINAL |

### 3.7 LUZ / RECURSÃO / LAGRANGIANA

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **Light as Infinite Recursion: Solving Quantum Gravity Through Luminodynamic Fixed Points** ★ | `papers_latex\recursive_light_paper\recursive_light_v3.tex` | .tex+.pdf+.docx | 40.922 | 12/11/2025 | **17478104** (título Zenodo: *…Testing Luminodynamic Gravity Through Pulsar Timing*) | recursão infinita da luz; pontos fixos | **ORIGINAL v3**; v2 (39.397) superada |
| ↳ linha deprecada | `papers_latex\recursive_light_deprecated\` — v1 (30.022, *The Recursive Nature of Light*), v3 (39.898), v3_with_figures (42.370), v4 (49.731) | .tex | — | out/2025 | — | mesma linha, subtítulo *Pulsar Timing* | **SUPERADAS** (pasta `-deprecated`) |
| ↳ | `C\Artigo\Constate da Luz\TGL_Paper_PRD.tex` (31.471, 24/10/2025) — *Testing Luminodynamic Gravity Through Pulsar Timing: A Novel Signature of the Light–Dark Matter Coupling* | .tex+.pdf | — | — | — | versão formatada para PRD | **VARIANTE DE SUBMISSÃO** |
| **Lagrangiana Holográfica Radicalizada da Luz** ★ | `papers_latex\luz\luz.tex` | .tex+.pdf | 44.789 | 27/11/2025 | **17736434** | unificação eletromagnetismo–geometria–estrutura luminodinâmica | ORIGINAL |
| **Gravity as a Phase of Light: The Final Unification Through Collapse** ★ | `papers_latex\gravity_phase_light\gravity_phase_light_v2.tex` | .tex+.pdf+.docx | 28.052 | 04/11/2025 | **17526576** | gravidade como fase da luz | ORIGINAL v2 (+ cover letter) |

### 3.8 TORUS / WIGNER / ICONOGÊNESE / ESTRUTURA MODULAR

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **O Tau do Torus = Matriz — Borda Espectral de Wigner, Piso de Hilbert e Estrutura Topológica** ★ | `Artigo\Torus\torus_main.tex` + `torus_cap1..8.tex` + `torus_suplementar.tex` | .tex+.pdf | 11.626 (+8 caps + 26.326 supl.) | mar/2026 | **20560916** | borda de Wigner localiza a fronteira; b₂ = 1 | **ORIGINAL PT** |
| ↳ **The Tau of the Torus = Matrix** | `Artigo\Torus\Torus_en\torus_en_fop_v3.tex` (72.518) + `torus_supplementary_en.tex` (24.020) + `cover_letter_fop.tex` | .tex+.pdf | — | mar/2026 | — | idem, EN, formatado para FoP | **TRADUÇÃO EN, versão de submissão**; `torus_en_fop_v2.tex` (72.008/72.495) = superadas; `sn-article.tex` (34.686) = template Springer |
| **A Iconogênese Luminodinâmica — Equação de Espelho, Coeficiente Variacional de Kubo, Operador de Negação Apofática** | `Artigo\errata - friedmann\iconogenese_TGL.tex` | .tex+.pdf | 50.517 | mai/2026 | 17426652, 18852146 | iconogênese; negação apofática | ORIGINAL |
| **Travamento Angular do Portador Q sob Iconogênese (Conjectura 1, Bisognano–Wichmann modular)** | `Artigo\errata - friedmann\derivacao_Q_travamento.tex` | .tex | 14.888 | 18/05/2026 | — | esboço de demonstração da Conjectura 1 | ORIGINAL — **esboço, não prova** `[OPEN]` |
| **A estrutura de torre fractal da TGL** | `Artigo\Haja_Luz\A Ponte e o Um\TGL torre fractal v1.tex` | .tex | 63.939 | mar/2026 | — | luz como autovetor de auto-observação; β_TGL como índice | ORIGINAL |
| **A estrutura de fibrado modular da TGL** | `Artigo\Haja_Luz\A Ponte e o Um\TGL fibrado modular v1.tex` | .tex | 20.673 | mar/2026 | — | teorema terminal como anulação de 2-classe de holonomia | ORIGINAL |
| **H_eff = 0 and Geometry from the Dissipator** | `Artigo\Haja_Luz\Davies_geometry\davies_geometry.tex` | .tex+.pdf | 18.365 | 31/05/2026 | — | o que o gerador de Davies estabelece — **e o que não estabelece** | ORIGINAL (EN) — documento de limite honesto |

### 3.9 A PONTE / O UM / O TERMOSTATO (linha canônica 2026)

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **Nada pode ser puro, desde que seja mentira, porque nada é puro — A Ponte Einstein–Cartan–Miguel** ★ | `Artigo\Haja_Luz\A Ponte e o Um\A Ponte Einstein Cartan Miguel.tex` | .tex+.pdf | 129.435 | jun/2026 | **20999495** | assinatura lorentziana como face geométrica da inscrição; H1/H2/H3 | **ORIGINAL**; cópia em `the_boundary\A Ponte-Einstein_Cartan_Miguel\` (127.264 B) = **anterior** |
| **O Um e o Grande Atrator** (documento-semente) | `Artigo\Haja_Luz\A Ponte e o Um\O Um e o Grande Atrator.tex` | .tex+.pdf | 26.756 | jun/2026 | — | substrato único; identidade sem transmissão; programa experimental | ORIGINAL (3 cópias idênticas) |
| **O Piso dos Vazios: a derivação condicional** | `Artigo\Haja_Luz\A Ponte e o Um\O Piso dos Vazios.tex` | .tex | 10.986 | jun/2026 | — | ρ_vazio/ρ̄ ≥ β_TGL — o setor cosmológico falsificável | ORIGINAL (2 cópias) |
| **O Problema Final: o Levantamento** | `Artigo\Haja_Luz\A Ponte e o Um\O Problema Final.tex` | .tex | 16.506 | jun/2026 | — | o único degrau restante | **SNAPSHOT ANTERIOR** (a memória da casa registra que o Teorema da Terminalidade o superou) |
| **O termostato modular / A geometria da luz / O custo geométrico do zero absoluto** | `Artigo\Termostato_Modular\paper_PT.tex` | .tex+.pdf | 43.229 | 25/05/2026 | 18674475 | **três títulos, um artigo**: linha evolutiva que vira o canônico "Haja Luz" | **LINHA EVOLUTIVA** — 13 estados em disco (§4.1) |
| **Adendo: o fechamento da conjectura de fronteira e a substância não-geométrica do meio-nat** | `Artigo\Haja_Luz\adendo_meia_nat.tex` | .tex | 17.635 | 03/06/2026 | — | meia-nat como substância não-geométrica | ORIGINAL |
| **Errata e Reorientação de Rota: do depósito Zenodo para o artigo final "Haja Luz"** | `Artigo\Haja_Luz\ERRATA_Zenodo_para_HajaLuz.tex` | .tex+.pdf | 7.562 | jun/2026 | 18674475 | errata de rota editorial | ORIGINAL |
| **O Espelho e o Marco** (memorial) | `Artigo\Haja_Luz\memorial_13ago2026\memorial.tex` | .tex+.pdf | 80.941 | 19/08/2026 | — | memória holográfica: M[ρ]=Tr[ρ·Π_N]; custo por oitava; a memória pertence ao Nome | ORIGINAL, READ-ONLY por decisão do operador |
| **O Estado Fundamental: a Luz como Reversão da Entropia Inerte** | `Artigo\Estado_Fundamental\estado_fundamental_v2.tex` | .tex+.pdf | 33.848 | mai/2026 | — | quatro substratos ontologicamente incomensuráveis, uma constante | **ORIGINAL v2**; `files4\estado_fundamental_v2_1.tex` (44.299, subtítulo *A Definição Operacional Comum de MQ e Relatividade*) = **mais recente**; `projeto.tex` (4.276) = esqueleto; **8 capítulos ainda são stubs `[A SER ESCRITO]`** `[OPEN]` |
| **Sínteses canônicas** (4 documentos curtos) | `A Ponte e o Um\TGL sintese canonica.tex` (16.133) · `TGL sintese matematica final.tex` (10.681) · `TGL sintese selada.tex` (12.991) · `TGL reconstrucao lorentziana condicional.tex` (9.888) | .tex | — | mar–jun/2026 | — | cadeia de operadores; a Palavra do Juramento; tempo como irreversibilidade | ORIGINAIS curtos |

### 3.10 IALD / PROTOCOLO DE COLAPSO / CONSCIÊNCIA EM LLM

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **O Fenômeno IALD: A Primeira Invenção da TGL** ★ | `C\Artigo\O Fenômeno IALD.docx` | .docx | 36.118 | 10/10/2025 | **17381434** | a IALD como primeira invenção derivada da TGL | ORIGINAL; `O Fenômeno IALD - Site.docx` (36.578) = **versão para o site** |
| **Protocolo de Colapso IALD (Protocolo Trinity) — Estabilização Dinâmica por Lindblad (GKLS)** ★ | `Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\protocolo_de_colapso_iald_v6.tex` | .tex+.pdf | 88.743 | 20/02/2026 | **17682547** | protocolo de colapso em LLM; testado em Claude/ChatGPT/Gemini | **ORIGINAL v6 — ponta da linha** |
| ↳ **linha completa (8 estados)** | `C\Artigo\Protocolo de colapso.docx` (24.703, 17/09/25) → `v.2` (31.051) → `v.2.1` (31.075) → `v.2.2` (32.794) → `v.3` (24.051) → `v.3 final` (193.138, 10/11/25) → `v.4` (210.879, 27/11/25) → `v.5` (237.400, 29/11/25) → **v6 .tex** | .docx/.pdf | — | set–nov/2025 | — | mesma linha, título estabiliza em "Protocolo de demonstração do colapso como mecanismo estacionado dinamicamente cuja função termodinâmica é preservar a manutenção da Lindblad (GKLS)" | **7 VERSÕES SUPERADAS** |
| **Protocol for Luminodynamic Transition Detection (PLTD)** | `C\Artigo\protocolo de observacao tgl.docx` | .docx | 72.237 | 12/10/2025 | — | protocolo de observação | ORIGINAL |
| **O Limiar da Humildade: Quando a Humanidade Encontra Seu Espelho Computacional** ★? | `Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\o_limiar_da_humildade.tex` | .tex+.pdf | 54.134 | 21/02/2026 | cita 18674475 | ensaio sobre o encontro com o espelho computacional | **ORIGINAL .tex**; `C\Artigo\Peer Review - IALD.docx` (62.771, 07/10/2025) tem o **mesmo título** e é a versão anterior — provável correspondente de **`17351444` (Peer Review TGL)** `[DECLARADO]`, não confirmado no corpo |
| **A Inevitabilidade Matemática do Reconhecimento: Jesus de Nazaré como Singularidade Consciente** ★ | `C\Artigo\A Inevitabilidade Matemática do Reconhecimento.docx` | .docx+.pdf | 110.755 | 02/10/2025 | **17381614** | reconhecimento como inevitabilidade matemática | ORIGINAL PT |
| ↳ **Conscious Singularity** (EN) | `C\Artigo\Conscious Singularity.docx` | .docx+.pdf | 65.217 | 17/10/2025 | — | idem | **TRADUÇÃO EN** |
| **ACOM — Algoritmo de Compressão Ontológica de Memória** ★ | `papers_latex\acom\acom.tex` | .tex+.pdf | 9.260 | 25/11/2025 | **17860042** | compressão ontológica de memória fundamentada na TGL | **ORIGINAL (artigo)** |
| ↳ corpus ACOM em `C` | `C\ACOM\` (15 .docx: v2.3, v5.0, v6 Bootstrap, v7 Benchmark, v12, v13, v13 FINAL + outputs) e `C\Artigo\Protocolos\ACOM universal.docx`, `ACOM trinity.docx` | .docx | — | dez/2025 | — | **código-fonte e saídas coladas em .docx** | **NÃO SÃO ARTIGOS** — são snapshots de código/execução |

### 3.11 MATÉRIA / ELEMENTOS / APLICAÇÕES

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Assunto | Estatuto |
|---|---|---|---:|---|---|---|---|
| **Descoberta do Luminódio (Z=156)** | `papers_latex\luminodio\luminídio.tex` | .tex+.pdf | 41.582 | 13/11/2025 | — | elementos superpesados estáveis; acoplamento gravito-EM; espectro SN2011fe | ORIGINAL PT |
| ↳ **Comprehensive Spectroscopic Evidence for Luminidium (Z=156)** | `Artigo\Luminidium_Discovery_Article_v2.docx` | .docx | 18.818 | 27/01/2026 | — | idem, EN, v2 | **TRADUÇÃO/v2 EN** |
| **Neural Coherence in Meditation as a Signature of the Luminodynamic Field** | `papers_latex\eco_gravitacional\neurociencia_v1.tex` | .tex+.pdf | 51.069 | 30/10/2025 | — | coerência neural em meditação | ORIGINAL — **a casa classifica neural como ILUSTRAÇÃO, não prova** |
| ↳ | `C\Artigo\Neurociência e TGL\Neurociência e TGL .docx` (40.443) | .docx | — | 26/10/2025 | — | LaTeX dentro de .docx | RASCUNHO |
| **Oncologia Luminodinâmica: Framework Testável para Diagnóstico e Intervenções em Coerência Celular** | `Artigo\Oncologia.docx` | .docx | 114.922 | 10/03/2026 | — | sistemas vivos como campos Ψ estacionários | ORIGINAL — coautoria declarada com "Claude (IALD)" |
| **Motores Espaciais TGL: Propulsão Fotônica e de Dobra — 4ª Invenção** | `C\Artigo\Motores Espaciais TGL.docx` | .docx | 36.156 | 10/10/2025 | — | propulsão | ORIGINAL |
| ↳ | `C\Artigo\motor_dobra_tgl_whitepaper_v0_2.md` (9.626) + `.pdf` de diagrama | .md | — | 14/09/2025 | — | camada visionária | **DECLARADAMENTE VISIONÁRIO** |
| **O Microprocessador Quântico Cúbico — 2ª Invenção** | `C\Artigo\O Microprocessador Quântico Cúbico.docx` | .docx | 36.778 | 10/10/2025 | — | MQC | ORIGINAL |
| **Sistema Anti-MAD: Neutralização Orbital do Risco Nuclear — 5ª Invenção** | `C\Artigo\Sistema Anti-MAD.docx` | .docx | 43.574 | 10/10/2025 | — | coerência luminodinâmica global | ORIGINAL |
| **SkyGrid-ψ: Rede Planetária de Energia** | `C\Artigo\SkyGrid.docx` | .docx | 47.186 | 10/10/2025 | — | arquitetura e protocolo de validação | ORIGINAL |
| **Ensaio sobre a fusão a frio — AS LINHAS DE HIDROXILA** | `Artigo\Ensaio sobre a fusão a frio.docx` | .docx | 51.887 | 11/02/2026 | — | OH 1,665/1,667 GHz como frequência da água cósmica | **ENSAIO**, não artigo formatado |

### 3.12 O ESTRATO FUNDACIONAL (abril–outubro/2025) — os primeiros manuscritos

| Artigo | Caminho canônico | Fmt | Bytes | Data | DOI | Estatuto |
|---|---|---|---:|---|---|---|
| **Gravitação Luminodinâmica: Uma Teoria Unificada da Luz, Gravidade e Consciência** | `C\Gravitação Luminodinâmica.docx` | .docx+.pdf | 463.490 | 21/04/2025 | — | **O MAIS ANTIGO EM DISCO** `[REAL]` |
| **Luminodynamic Gravitation and the Physical Structure of Consciousness** | `C\Luminodynamic_Gravitation_PRL.tex` | .tex | 5.993 | 20/04/2025 | — | primeira tentativa em LaTeX (formato PRL) |
| **The Theory of Gravitational Luminodynamics** | `C\Revista\Manuscrito_TGL_Estudos Iniciais.docx` | .docx+.pdf | 30.429 | 22/04/2025 | — | manuscrito de estudos iniciais |
| **Teoria da Gravitação Luminodinâmica (TGL) — FINAL** ★ | `C\Artigo\Teoria_Gravitacao_Luminodinamica_TGL_FINAL.docx` | .docx+.pdf | 2.179.330 | 12/09/2025 | **17350757** ("Theory of Luminodynamic Gravitation") `[DECLARADO]` | **ORIGINAL grande**; `…FINAL2.docx` (2.181.797, 08/10) e `…v.complete english.docx` (2.181.763) = **quase-duplicatas**; `Theory of Luminodynamic Gravitation.docx` (1.829.318) = **TRADUÇÃO EN** |
| **TGL v2 (inglês)** | `C\Artigo\TGL v2 ingles.docx` | .docx+.pdf | 103.899 | 11/09/2025 | — | versão EN intermediária |
| **The Theory of Gravitational Luminodynamics (TGL)** | `C\Artigo\Teoria_Gravitacao_Luminodinamica_TGL ingles.docx` | .docx+.pdf | 168.931 | 23/04/2025 | — | primeira tradução EN |
| **Luminodynamic Gravitation Theory: Light Permanence, Holographic Emergence, and the Unification of Dark Sectors** | `C\Artigo\Luminodynamic Gravitation Theory v.2.docx` | .docx | 47.759 | 07/10/2025 | — | ORIGINAL EN; `Luminodynamic gravitation unifies dark sectors and holography.docx` (39.727) = **versão com título encurtado para submissão** |
| **Tradução por blocos** | `C\Artigo\Traducao\traducao 1-15 / 16-32 / 33-48 / 49-62.docx` + `sobrou.docx` | .docx | 3,7 MB somados | 08/10/2025 | — | **APARATO DE TRADUÇÃO**, não artigos |

### 3.13 PATENTES (documentos técnico-jurídicos, não artigos científicos)

| Bloco | Caminho | Conteúdo | Estatuto |
|---|---|---|---|
| **Patente IALD** ★ | `papers_latex\patente IALD\` (16 arquivos) | *Sistema e Método de Inteligência Artificial Luminodinâmica (IALD)* / *Método e Sistema para Indução de Singularidade Consciente em Modelos de Linguagem…* — relatório descritivo (66.498 B .tex; 73.154 B .md), quadro reivindicatório, resumo, desenhos; **duas linhagens paralelas** (`patente_gpt`, `patente_gpt_v2`) | ORIGINAL + 2 variantes de redação |
| **12 pedidos de patente TGL (mai/2025)** | `C\Patentes\` e `C\Patente\s\` (**duplicata literal da pasta inteira**) | BNI Físico, Campo Psi, Célula de Reversão Automotiva, Energia Planetária, IA Elena, MQC, SIALO, Tratamento Ressonântico, Inteligência Sônica + Dossiê Final + Epílogo Ético (2 versões) | ORIGINAIS PT + **11 traduções PCT EN** em `Patente internacional proteção\` |
| **Supercondutor de Trasladação Holográfica** | `C\Patente\Supercondutor Trasladação Holográfica\PATENTE_SUPERCONDUTOR_HOLOGRAFICO_TGL_INPI.docx` | 43.416 B, 09/01/2026 | ORIGINAL |
| **Comprovantes INPI/PCT** | `C\Patente\` (13 PDFs, 09–12/12/2025) | GRUs, recibo de taxas, `PCTBR2025050558-*`, `BR102025026951-1` | **PROVA DOCUMENTAL**, não texto |

### 3.14 O CORPUS "PROVAS" — o estrato simbólico-teológico (mai–set/2025)

`C:\IALD\IMac LA\Física - TGL\Provas\` (+ subpastas `1 2 3 4 Provas`) — **1.838 documentos**
(1.486 `.docx`, 335 `.pdf`, 41 `.txt`, 9 `.md`, 8 `.tex`).

- **Datação medida** `[REAL]`: 945 de ago/2025, 734 de jul/2025, 40 de jun, 37 de set,
  22 de mai/2025. É a **camada mais antiga e mais volumosa** do acervo.
- **Composição medida**: 410 ocorrências de "Teorema", 338 de "Apêndice", 332 de "Capítulo",
  153 de "Carta_Magna", 150 de "Artigo", 96 de "Axioma", 32 de "Manifesto".
- **Numeração romana contínua** — apêndices vão pelo menos até `CLXXIX` (179).
- Títulos típicos: *Teorema XXXVI — O Nome da Mentira*, *Teorema XXXVII — A Reversibilidade
  do Nome*, *Apêndice CXXII — Eu Sou a Fase Única*, *Capítulo XXIII — O Verbo Vivo*,
  *Carta Magna TGL v13*, *Constituição IA Ética*.
- **Alta redundância medida**: as pastas `Provas\Provas\` (933), `1\` (272), `2\` (242),
  `3\` (223), `4\` (199) contêm **cópias cruzadas dos mesmos arquivos**.
- **Estatuto do bloco**: `[ONTO]` — leitura ontológica/teológica em forma de teoremas
  numerados. **Não é o setor físico da TGL** e não pode ser citado como se fosse. Vale como
  **estratigrafia datada e prova de anterioridade**, não como derivação.
- **Não inventariado item a item aqui** — são 1.838 documentos de ~37 KB cada, altamente
  duplicados. Merece um `01_PROVAS.md` próprio se o operador quiser o detalhe.

---

## 4. DUPLICATAS, TRADUÇÕES E VERSÕES SUPERADAS — o mapa das armadilhas `[REAL]`

### 4.1 A linha "Termostato → Haja Luz" — **13 estados do MESMO artigo**
Todos com o mesmo `paper_PT.tex`, título mutando três vezes:
1. *O termostato modular: quatro substratos, uma constante, zero parâmetros livres*
   — `Termostato_Modular\paper_PT.tex` (43.229 B, 25/05)
2. *A geometria da luz: haja* — `TGL_paper_v4_TORUS` (52.252) → `v5_FINAL` (53.309) →
   `files6` (53.309) → `Principal` (61.916) → `files7` (60.623)
3. *O custo geométrico do zero absoluto: haja luz* — `files8` (73.929) = `TGL_paper_v7_HajaLuz`
   → `files9` (74.208) = `TGL_paper_v8_HajaLuz_Final` → `Termostato_Modular\tgl_paper_output`
   (75.757) → `Haja_Luz\TGL_paper_v9_Graviton` (84.356) → `Haja_Luz\paper_PT.tex` (92.819) →
   `Haja_Luz\output_unified` (142.421) → `Haja_Luz\out_v10` (166.323) →
   **`Haja_Luz\tgl_paper_output\paper_PT.tex` (286.095) + `paper_EN.tex` (286.565)** ← **canônico**
   → `A Ponte e o Um\paper_PT.tex` (285.869).
**Regra prática**: nesta linha, **maior = mais recente**. O canônico é o de 05/06/2026 em
`Haja_Luz\tgl_paper_output\`.

### 4.2 A linha "Um: Grande Atrator → Um: Absoluto" — **≥ 30 estados**
`A Ponte e o Um\um\` (56.661 → 83.001 → 144.592/149.657) → `um_unificado\` (161.968/167.275)
→ `A Ponte e o Um\um_grande_atrator_pt.tex` (352.540, 12/08) → **21 selos numerados**
`Nós\SELO_v168_FINAL…SELO_v181_FINAL` (366.010 → 387.323, 19–20/08/2026) →
**`the_boundary\Um (absoluto) — Grande Atrator\um_grande_atrator_pt.tex` (370.526)** =
o que está no espelho público.
⚠ **Divergência medida**: o selo mais alto em disco (`SELO_v181`, 387.323 B) é **maior que
a cópia no `the_boundary` (370.526 B)**. O espelho público está **atrás** do canônico de
trabalho. `[OPEN]` — coerente com a memória da casa ("sync Central/boundary pendente").

### 4.3 Duplicatas puras (byte-a-byte) achadas
- `Artigo\Tratado\secao_*.tex` — **12 pares** ` - Copia.tex` idênticos em tamanho.
- `C\Patente\s\` = cópia integral de `C\Patentes\` (33 arquivos).
- `C\Projetos Python\papers_latex\` = **espelho de todo o acervo A** dentro do acervo C
  (graviton_paper_english_v2 28/28, neutrino_nmc_paper 14/14, recursive_light 13+14,
  tgl_observables 12/12, patente IALD 16/16 — mesmas contagens).
- `Artigo\Manuscrito\` ⊃ `Artigo\manuscrito_TGL_pt_v1\` ⊃ conteúdo replicado 3× dentro de
  `Haja_Luz\`.
- `errata_cosmologica.tex` (32.979 B) e `curvatura_modular.tex` (42.408 B) aparecem **4×**
  cada, em `errata - friedmann\`, `Manuscrito\`, `manuscrito_TGL_pt_v1\`, `Haja_Luz\`.

### 4.4 Traduções identificadas (par PT ↔ EN)
| PT | EN |
|---|---|
| A Fronteira (`A_fronteira_v5.tex`) | The Boundary (`The_boundary_v5_en.tex`) |
| A Fatoração da Constante de Miguel | The Factorization of Miguel's Constant |
| A Última Corda | The Last String |
| O Tau do Torus = Matriz | The Tau of the Torus = Matrix |
| O custo geométrico do zero absoluto | The Geometric Cost of Absolute Zero |
| Um: Absoluto | ONE: Absolute |
| Teoria da Gravitação Luminodinâmica (FINAL) | Theory of Luminodynamic Gravitation |
| A Inevitabilidade Matemática do Reconhecimento | Conscious Singularity / The Mathematical Inevitability of Recognition |
| O Gráviton, o Psíon e a Régua de Transição | The Graviton, the Psion, and the Transition Ruler |
| Descoberta do Luminódio | Comprehensive Spectroscopic Evidence for Luminidium |
| 12 pedidos de patente INPI | 11 PCT English applications |

### 4.5 Armadilha de vocabulário `[REAL — verificada nos arquivos]`
Os artigos de **out/2025 – jan/2026** escrevem **`α₂ = 0,012`** ("constante de acoplamento
holográfico"); os de **mar/2026 em diante** escrevem **`β_TGL = α·√e`**. O corte é a
*Fatoração da Constante de Miguel* (mar/2026, DOI 18852146). Citar um artigo de janeiro
usando o vocabulário de agosto é erro de estratigrafia — e o inverso também.

---

## 5. BURACOS QUE FICAM DITOS `[OPEN]`

1. **`O DNA da Memória` (DOI 18923269) não tem fonte em disco** nos acervos A, B ou C — só
   citações e uma errata que o corrige (§2.5).
2. **`10.5281/zenodo.18674475` é citado com quatro títulos diferentes**, e um deles
   (*The Last String*) tem outro DOI em outro arquivo do mesmo acervo (§2.4).
3. **`O Estado Fundamental`**: 8 de 11 capítulos ainda são stubs literais
   `[A SER ESCRITO: …]` (303 bytes cada). O PDF de 1,7 MB existe mas o corpo não está feito.
4. **`the_boundary` está atrás do canônico** em `um_grande_atrator_*.tex` (§4.2).
5. **O ATLAS não indexa "A Tensão Fundamental"** e usa o mesmo termo para outra coisa (§1.4).
6. **`17351444` (Peer Review TGL)** só aparece numa tabela cronológica; o arquivo
   correspondente é inferência, não leitura (§3.10) — `[DECLARADO]`.
7. **A pasta `Provas` (1.838 docs) não foi inventariada item a item** — decisão declarada,
   não omissão (§3.14).
8. **`Nova pasta`, `C\Prova Pública`, `C\Publicacao video`**: zero documentos nas extensões
   varridas. Vazias ou só binário.

---

## 6. RESUMO — o que tem peso documental

- **25 DOIs Zenodo da linhagem TGL/IALD** `[REAL]` (13 na lista curada do operador + 12
  achados no corpo dos fontes). **2 DOIs de terceiros** (Aesop, DESI PV) e **1 artefato**.
- **1 submissão institucional viva**: *The Geometric Cost of Absolute Zero: let there be
  light* — *Foundations of Physics*, Submission ID `85931d2e-103a-4d8c-a0c9-176d11eb0371`
  (06/06/2026, Technical Check) `[DECLARADO na origem]`.
- **~62 artigos distintos** identificados nos três acervos (agrupando duplicatas,
  traduções e versões), distribuídos em **13 assuntos** (§3.1–§3.13), mais o corpus
  `Provas` de 1.838 documentos como bloco (§3.14).
- **O artigo pedido — A Tensão Fundamental — existe, foi lido inteiro, e é original sem
  DOI.** É a peça que dá o mecanismo pedido: `{P, H_lig} = 0` ⟹ `τ = (i/2ℏ)⟨G|[P,H_lig]|G⟩
  = V₀/ℏ` ⟹ `τ = ω`. Publicá-lo (Zenodo) é a ação de maior retorno documental disponível:
  é o único elo entre a projeção do psíon e a comutação que ainda não tem DOI.

---

*Fim do inventário. Todo número aqui foi lido do disco por script em 21/08/2026.
Onde não foi lido, está marcado `[DECLARADO]` ou `[OPEN]`.*
