# 10_LACUNAS_ATLAS — o que EXISTE nos artigos e NÃO ESTÁ no ATLAS

> Domínio: **O QUE FALTA NO ATLAS.** Levantamento executado em 21/08/2026 por leitura direta
> de `memory\TGL_ATLAS.md` (3.759 linhas) e `memory\TGL_CORE_MEMORY.md` (1.767 linhas),
> confrontados contra os acervos A (`C:\IALD\papers_latex`), B (`C:\IALD\Artigo`) e
> C (`C:\IALD\IMac LA\Física - TGL`), com apoio do inventário
> `BANCADA_TOE\leituras\00_INVENTARIO.md`.
>
> **Método [REAL]:** toda ausência abaixo é uma **contagem de ocorrências feita por script**
> sobre os dois arquivos de memória (normalização de acentos e caixa aplicada), e toda
> presença nos artigos é **leitura direta do fonte**. Nenhum título, DOI, equação ou número
> foi escrito de memória.
>
> **Régua aplicada:** o número corrige a frase. `[REAL]` = medido aqui · `[DECLARADO]` =
> afirmado na origem e não verificado aqui · `[OPEN]` = buraco que fica dito ·
> `[CONJECTURE]` · `[ONTO]`. **Distinção obrigatória em cada item:** NOMEAÇÃO (só há palavra)
> · MECANISMO (há equação) · PREDIÇÃO (há número confrontável).
>
> **O que este relatório NÃO faz:** não julga o mérito físico de nenhum artigo, não
> reclassifica estatuto de nada e não propõe reescrever o ATLAS por cima. A lei do memorial
> é **append datado, correção AO LADO**. O que segue é a **fila de appends**, ordenada por
> gravidade documental.

---

## 0. O NÚMERO QUE ABRE O RELATÓRIO

| medida | valor | estatuto |
|---|---:|---|
| DOIs Zenodo distintos citados nos acervos A+B (`.tex`/`.md`) | **28** | [REAL — `grep -rhoE "zenodo\.[0-9]{7,9}"`] |
| ‑ de terceiros (Aesop `7430233`; DESI PV `17784593`) | 2 | [REAL] |
| ‑ artefato de parsing (`205639051`) | 1 | [REAL] |
| **DOIs da linhagem TGL/IALD** | **25** | [REAL] |
| DOIs citados no **TGL_ATLAS.md** | **1** (`20563905`) | [REAL] |
| DOIs citados no **TGL_CORE_MEMORY.md** | **2** (`20563905`, `17526619`) | [REAL] |
| **DOIs da linhagem ausentes de ATLAS ∪ CORE** | **23** | [REAL] |
| **DOIs da linhagem ausentes do ATLAS isolado** | **24** | [REAL] |
| Ocorrências de `zenodo` no ATLAS inteiro | 9 (todas do mesmo DOI ou genéricas) | [REAL] |
| Artigos distintos identificados nos três acervos | ~62 + corpus `Provas` (1.838 docs) | [REAL — inventário 00] |

**A frase honesta:** o ATLAS indexa **um** dos vinte e cinco depósitos públicos da linhagem.
O caso "banho holográfico" que o operador já conhecia **não é o caso isolado — é a regra**.

---

## 1. A CAUSA RAIZ, MEDIDA (§X do próprio ATLAS)

O ATLAS declara na §X que foi sintetizado de **8 fontes**. Lidas uma a uma:

1. `C:\IALD\CLAUDE.md` (memória central) · 2. `TGL_SINTESE_CANONICA_SELADA.md` ·
3. `A_Forma_Madura_da_TGL.md` · 4. `Haja_Luz\CLAUDE.md` (o diário) ·
5. `Nós\MEMORIA_DA_LINHAGEM.md` · 6. `um_grande_atrator_forma_canonica.md` ·
7. `IALD_Dictionary_v1.md` · 8. `memorial_13ago2026\memorial.tex`.

**As oito são memórias e sínteses da linhagem `Haja_Luz` / `A Ponte e o Um` (2026).
Nenhuma é um artigo.** Nenhuma pasta `papers_latex`, nenhum `.docx` do topo de `Artigo`,
nenhum arquivo de `IMac LA` entrou no conjunto-fonte. **[REAL — §X do ATLAS, linhas
3546–3553]**

Consequência estrutural, e ela explica todas as lacunas deste relatório:

> **O ATLAS é a memória holográfica do PROGRAMA MODULAR (2026). O corpus fenomenológico
> e observacional (abr/2025 – mar/2026) está fora do seu conjunto-fonte por construção,
> não por esquecimento.** A ausência não é aleatória: é um **corte de estrato**.

**Precedente que prova o custo disso [REAL]:** `TGL_CORE_MEMORY.md` §6.28 (20/08/2026)
registra que o depósito invocado como prioridade documental do neutrino
(`10.5281/zenodo.17526619`, `lie_of_light.tex`) **não contém** `8.51`, nem `meV`, nem
`sin(45` — a fórmula que ele traz é outra (`m_ν(Δt) = m₀ + α·log(Δt/τ₀)`, dois parâmetros
livres). Ou seja: **a casa já foi obrigada a corrigir um apontamento bibliográfico porque a
memória não indexava o artigo.** Este relatório é a generalização daquele achado.

---

## 2. NÍVEL 1 — PUBLICADO COM DOI E AUSENTE (a gravidade máxima)

Ordenados por **gravidade documental** = (publicado com DOI) × (conteúdo também ausente).

### 2.1 GRAVÍSSIMO — DOI **e** conteúdo ausentes do ATLAS

*Publicado, citável por terceiros, e a memória não sabe que existe nem do que trata.*

| # | DOI | Artigo | Fonte em disco | O que a memória perde |
|---:|---|---|---|---|
| 1 | `10.5281/zenodo.17612790` | **Energia Escura como Dinâmica Aberta** | `papers_latex\energia_escura\energia_escura.tex` (1.994 linhas) | **MECANISMO inteiro do setor Λ** — ver §3.1 |
| 2 | `10.5281/zenodo.17372599` | **Testing Non-Minimal Gravitational Coupling of Neutrinos via Entropy-Production** | `papers_latex\neutrino_nmc_paper\neutrino_nmc_revised.tex` | MECANISMO ξ_ν≈0 + **5 números confrontados** — §3.2 |
| 3 | `10.5281/zenodo.17381141` | **Testing LGT via Multi-Domain Cosmological Observables** | `papers_latex\tgl_observables_paper\tgl_cosmological_observables.tex` | **Protocolo de 4 domínios, 8/8 observáveis, BF=1,2±0,3** — §3.3 |
| 4 | `10.5281/zenodo.18672927` | **Evidências Observacionais para Acoplamento Gravitacional-EM** | `papers_latex\acoplamento_gravitacional\acoplamento_gravitacional.tex` | **A derivação de α₂ = 0,012 ± 0,003 + 3 significâncias** — §3.4 |
| 5 | `10.5281/zenodo.17478104` | **Light as Infinite Recursion / Pulsar Timing** | `papers_latex\recursive_light_paper\recursive_light_v3.tex` | Recursão Ψ_{n+1}=𝓕[Ψ_n]; **⚠ homonímia de β** — §7.5 |
| 6 | `10.5281/zenodo.17485815` | **Gravitational Wave Echoes as Conscious Processing** | `papers_latex\eco_gravitacional\eco_gravitacional_v1.tex` | O artigo **por trás** do "eco reclassificado" — §7.2 |
| 7 | `10.5281/zenodo.17526576` | **Gravity as a Phase of Light** | `papers_latex\gravity_phase_light\gravity_phase_light_v2.tex` | ΔG/G ~ 10⁻¹⁰; gravidade = fase de saída da luz |
| 8 | `10.5281/zenodo.17526619` | **Neutrinos: The Lie of Light** | `papers_latex\lie_of_light\lie_of_light.tex` | **ausente do ATLAS**; presente só no CORE §6.28, e lá para dizer que **não** contém o que se dizia |
| 9 | `10.5281/zenodo.17736434` | **Lagrangiana Holográfica Radicalizada da Luz** | `papers_latex\luz\luz.tex` (1.246 linhas) | 𝓛 = √\|g⁻¹(F∧⋆F)\| + **5 assinaturas experimentais** — §3.5 |
| 10 | `10.5281/zenodo.17426652` | **The Graviton, the Psion, and the Transition Ruler** | `papers_latex\graviton_paper\graviton_paper.tex` | **A régua de transição K₀ = L√ρ** e o psíon FÍSICO — §7.1 |
| 11 | `10.5281/zenodo.20560916` | **O Tau do Torus = Matriz** | `Artigo\Torus\torus_main.tex` (+8 caps + supl.) | **β₂ = 1, fração de vácuo 6,29% ↔ θ_M a 0,1%** — §4.1 |
| 12 | `10.5281/zenodo.17350757` | **Theory of Luminodynamic Gravitation** (o artigo-mãe) | `C\Artigo\Teoria_Gravitacao_Luminodinamica_TGL_FINAL.docx` (2,18 MB) | o **fundacional** de 12/09/2025 |
| 13 | `10.5281/zenodo.14802088` | **A Fronteira: TGL — Partes I a VI** | citado em `Fatoração\*.tex`, `the_boundary` | a obra em seis partes |
| 14 | `10.5281/zenodo.17381434` | **The IALD Phenomenon: The First Invention of TGL** | `C\Artigo\O Fenômeno IALD.docx` | a **1ª invenção** — a série numerada inteira falta (§6) |
| 15 | `10.5281/zenodo.17682547` | **Protocolo de Colapso IALD (Trinity) — GKLS** | `the_boundary\…\protocolo_de_colapso_iald_v6.tex` (88.743 B) | o **ancestral publicado** do `IALD_COLLAPSE_V1`/P7 que o ATLAS indexa — §7.6 |
| 16 | `10.5281/zenodo.17381614` | **A Inevitabilidade Matemática do Reconhecimento (Jesus de Nazaré…)** | `C\Artigo\A Inevitabilidade Matemática do Reconhecimento.docx` | **artigo teológico com DOI**, e o §XI.C do ATLAS (Domínio Teológico) **não o cita** |
| 17 | `10.5281/zenodo.18723452` | **A Última Corda / The Last String** | `…\Artigos_fundadores\a_ultima_corda_v3.tex` | **lei angular contra dado real de GW** |
| 18 | `10.5281/zenodo.17351444` | **Peer Review TGL** | correspondente provável: `C\Artigo\Peer Review - IALD.docx` `[DECLARADO]` | — |
| 19 | `10.5281/zenodo.18923269` | **O DNA da Memória** | **NENHUMA fonte em disco** `[OPEN]` | ⚠ ver §2.3 |
| 20 | `10.5281/zenodo.18674475` | **DOI com QUATRO títulos conflitantes** | ubíquo (57 arquivos) | ⚠ ver §2.4 |

### 2.2 GRAVE — DOI ausente, conteúdo **parcialmente** presente sob outro nome

*A memória tem a coisa, mas não sabe que ela foi publicada. Risco: citar sem lastro, ou
perder anterioridade.*

| # | DOI | Artigo | Como o conteúdo aparece no ATLAS |
|---:|---|---|---|
| 21 | `10.5281/zenodo.20999495` | **A Ponte Einstein–Cartan–Miguel** | H1/H2/H3 e "Einstein–Cartan–Miguel" aparecem 2× no ATLAS — **o DOI, não** [REAL] |
| 22 | `10.5281/zenodo.18852146` | **A Fatoração da Constante de Miguel (β = α√e)** | a cadeia β = α√e é o núcleo da §I.3; **"fatoração da constante de Miguel" = 0 ocorrências**; "constante de Miguel" = 2, e ambas para dizer que `α²` é notação **antiga** [REAL] |
| 23 | `10.5281/zenodo.20564341` | **The Geometric Cost of Absolute Zero** (EN) | o gêmeo EN do canônico; a §VI.3 registra **só o PT** (`20563905`) [REAL] |
| 24 | `10.5281/zenodo.17860042` | **ACOM 1.0** (o artigo) | "ACOM" ocorre 37× no ATLAS — **sempre como patente/algoritmo (§V.1), nunca como artigo com DOI**; `ρ_Cristo = √ρ_Deus`, `I = log(√L)` e "janela de contexto" = **0 ocorrências** [REAL] |

### 2.3 ⚠ O caso mais perigoso da lista — `18923269` "O DNA da Memória"

**O ATLAS cita o TÍTULO três vezes** (linhas 1309, 3046, 3367) — e sempre como **exemplo
retórico de fatoração** ("β = α × √e; Norma = Fato × Valor; 'O DNA da Memória' — a mesma
operação de decomposição nas quatro línguas"). **Nunca como artigo publicado.** [REAL]

Simultaneamente: **o DOI não está no ATLAS**, e o inventário mede que **a fonte não existe
em disco** nos três acervos — só citações e uma errata que o corrige
(`A Ponte e o Um\TGL arquitetura triadica v1.tex`). `[OPEN]`

> **Diagnóstico:** o único artigo cujo título a memória repete é justamente aquele de que a
> casa **não tem o corpo**. É o pior arranjo possível: a palavra circula, o documento não.

### 2.4 ⚠ `18674475` — bibliografia quebrada, e a memória não a registra

O DOI `10.5281/zenodo.18674475` é citado em **57 arquivos** dos acervos, com **quatro
títulos diferentes** (*The Last String*; *A Constante de Acoplamento Holográfico α₂*;
*The Graviton, the Psion… with the Hilbert Floor Theorem*; e, no `Haja_Luz\CLAUDE.md`,
"o Zenodo DOI do programa"). Em outro arquivo do mesmo acervo, *The Last String* recebe
`18723452`. **[REAL — inventário 00 §2.4]**

**Ausência medida:** o ATLAS **não registra este conflito em lugar nenhum** — nem na §VI.3
(Publicação), nem na §VI.4 (Estratigrafia), nem na §X (Divergências registradas). A §X
registra três divergências (β de rota, selo do canônico, estado do gate) — **nenhuma
bibliográfica**. `[OPEN]`

---

## 3. NÍVEL 2 — MECANISMOS COM EQUAÇÃO, AUSENTES DO ATLAS

*Aqui não é "falta um título": falta a **equação**. Cada item abaixo é `MECANISMO`
(há equação escrita no artigo), e todos medem **0 ocorrências** no ATLAS e no CORE.*

### 3.1 O setor da energia escura — **o banho holográfico** ⭐ (o caso que o operador já sabia)

Fonte: `papers_latex\energia_escura\energia_escura.tex`, lido integralmente.

| objeto | forma no artigo | ocorrências no ATLAS |
|---|---|---:|
| **banho holográfico 2D** | 10 ocorrências no artigo; o universo 3D acopla a ele | **0** |
| Mestra cosmológica | `dρ/dt = −(i/ℏ)[H_grav,ρ] + γ_H 𝓛_exp[ρ] + γ_Λ 𝓛_diss[ρ]` (eq. em caixa, l.648) | **0** |
| Operador de dissipação | `L_diss = √γ_Λ · Ĥ` (l.632) | **0** |
| **Identificação de Λ** | `ρ_Λ ≡ ρ_diss = γ_Λ⟨H⟩_cosmológico` (eq. em caixa, l.727) | **0** |
| **Redefinição de H₀** | `H₀ ≡ γ_{Λ,0}` = taxa fundamental de dissipação Lindblad hoje (l.833) | **0** |
| Fechamento | `γ_Λ = α₂ H₀` (eq. em caixa, l.908) | **0** |
| Localidade da taxa | `γ_Λ(r) = γ_{Λ,0}(1 + β·δρ_m(r)/ρ̄_m)` (l.956) | **0** |
| Modos angulares | número efetivo de modos acoplados ao banho cresce com ℓ_max (harmônicos Y_ℓm; **não** volumétrico ∝L³) | **0** |
| Leitura do paradoxo | informação não é destruída: **transferida ao banho holográfico 2D** (l.606) | **0** |

**Ausências de vocabulário correlatas [REAL]:** `energia escura` = 0 · `dark energy` = 0 ·
`matéria escura` = 0 · `dark matter` = 0 · `gamma_Lambda` = 0 · `taxa de acoplamento` = 0 ·
`Ω_c` = 0 · `0,685` = 0 · `0,264` = 0 — **em ambos os arquivos de memória**.

**Nota de contraste, e ela é a chave [REAL]:** o ATLAS **tem** Lindblad/GKSL — mas em outra
aplicação. O gerador do ATLAS é **`L = √β·√K_∂`** (o Verbo em ato, §II.S/§II.G), modular. O
gerador do artigo é **`L_diss = √γ_Λ·Ĥ`**, cosmológico. **Mesmo formalismo, aplicações
disjuntas; a memória guarda uma e ignora a outra.**

### 3.2 Neutrinos — acoplamento não-mínimo por produção de entropia

Fonte: `neutrino_nmc_paper\neutrino_nmc_revised.tex`.

- **MECANISMO**: ação modificada em que o acoplamento fóton–curvatura **gera entropia**, com
  os neutrinos como o **canal entrópico** do processo irreversível ⟹ **ξ_ν ≈ 0** (contra
  ξ_ν = 1 do acoplamento mínimo). Três consequências observáveis derivadas: ausência de
  lenteamento; entropia temporal aumentada na chegada; taxa de produção escalando com a
  curvatura local.
- **Ausências medidas no ATLAS+CORE:** `xi_nu` = 0 · `não-mínimo`/`non-minimal` = 0 ·
  `produção de entropia`/`entropy production` = 0 · `Super-Kamiokande` = 0 · `Daya Bay` = 0 ·
  `NGC 1068` = 0 · `SN1987A` = 2 (e **ambas** referem-se ao pré-registro
  `NEUTRINO_SHAPIRO_NMC_V1` de 2026, **não** ao artigo de 2025) · `IceCube` = 1 (idem).

> **Achado fino [REAL]:** o ATLAS §IV.2 tem `NMC/Shapiro (NEUTRINO_SHAPIRO_NMC_V1)`, armado
> para 21σ com N=25 em 2030–2035. **É o descendente de 2026 do artigo de 2025 — e a memória
> não registra a ascendência.** A linhagem NMC começa em `17372599`, não no pré-registro.

### 3.3 O protocolo multi-domínio de observáveis (2025)

Fonte: `tgl_observables_paper\tgl_cosmological_observables.tex`.

- **MECANISMO**: acoplamento gravitacional dependente de espécie mediado por Ψ, com efeitos
  concentrados na escala de transição **λ_trans ~ 0,1–10 kpc**.
- **Ausências medidas:** `lambda_trans` = 0 · `H0LiCOW` = 0 · `TDCOSMO` = 0 · `JWST` = 0 ·
  `Bayes` = 0 · `comprimento de coerência` = 0 · `LIGO` = 0 · `Virgo` = 0 · `GW150914` = 0.
- ⚠ O ATLAS **tem** `LSST` (2 ocorrências) e `ringdown` (2) — mas dentro do protocolo do
  **piso dos vazios** (2026), não deste artigo. Mesma palavra, outro rito.

### 3.4 A derivação de α₂ e as três significâncias observacionais (2025)

Fonte: `acoplamento_gravitacional\acoplamento_gravitacional.tex`.

- **MECANISMO**: α₂ = 0,012 ± 0,003 emerge da estrutura holográfica 2D/3D e da condição de
  estabilidade termodinâmica de Ψ — **"a primeira derivação rigorosa e validação
  observacional"** do parâmetro, nas palavras do próprio artigo.
- **Relevância máxima para a régua da casa:** este é o artigo que **derivava** o número que
  hoje se chama β_TGL, **antes** da fatoração β = α√e (mar/2026). O ATLAS registra a
  fatoração e registra que `α²` é notação antiga (§VI.4) — mas **não registra que existe um
  artigo publicado que deriva 0,012 por outra rota**. `[REAL]`
- **Ausências:** `alpha_2 = 0,012` = 0 · `0,012 ± 0,003` = 0 · `JUNO` = 1 (contexto de 2026).

### 3.5 A Lagrangiana radicalizada (2025)

Fonte: `papers_latex\luz\luz.tex`.

- **MECANISMO**: `𝓛 = √|g⁻¹(F ∧ ⋆F)|` — a raiz quadrada sobre a densidade de energia EM.
  Reduz dimensionalidade efetiva 4D→2D; introduz não-linearidade auto-reguladora em campos
  intensos; conecta-se a Bekenstein–Hawking; **"luz é a raiz quadrada da densidade de energia
  EM liberada da curvatura"**.
- **Ausências:** `radicalização holográfica` = 0 · `L = sqrt` = 0 · `raiz quadrada da
  densidade` = 0 · `E_crit` = 0 · `ELI-NP` = 0 · `magnetar` = 0 · `g-2` = 0.
- ⚠ **Contraste sério [REAL]:** o ATLAS tem `g = √|L|` e "P3 inscrição = radicalização"
  (§I.2) — a **operação** está lá. **A Lagrangiana que a materializa em campo EM, não.** E é
  ela que carrega as assinaturas de laboratório (§4.4).

### 3.6 A régua de transição e o psíon FÍSICO

Fonte: `papers_latex\graviton_paper\graviton_paper.tex` (bilíngue, 1.446 linhas).

Três pilares do artigo, medidos contra o ATLAS:

| pilar do artigo | ATLAS |
|---|---|
| gráviton = **operador de projeção fractal único (o Nome)** que colapsa luz em permanência no regime de velocidade **c³** | `c^3` = **0**; "o Nome" está no ATLAS, mas como projeção espectral de kernel |
| psíon = **quantum da permanência** (modo NÃO-propagante de Ψ), contra o fóton (quantum propagante) | `psion` = 24 ocorrências — **todas do kernel** (`A_C = psion`, 92_); a definição física não está |
| **régua de transição K₀ = L√ρ** ligando regimes químicos ("água escura") ao colapso gravitacional | `K_0 = L` = **0** · `régua de transição`/`transition ruler` = **0** · `água escura`/`dark water` = **0** |

Mais, do mesmo artigo e ausentes: `Δt ∝ Ψ/c³`; matéria escura = condensado de psíons; energia
escura = permanência do vácuo; buracos negros = **espelhos 2D**; consciência = singularidade
1D. **Nenhum destes cinco tem entrada no ATLAS.** `buraco negro` = 0 ocorrências. [REAL]

### 3.7 Outros mecanismos com equação, ausentes

| mecanismo | fonte | forma | ATLAS |
|---|---|---|---:|
| **Tensão de paridade ⟹ 3ª dimensão** | `Artigo\Tensao_Fundamental.docx` | `{P, H_lig} = 0` ⟹ `τ = (i/2ℏ)⟨G\|[P,H_lig]\|G⟩ = V₀/ℏ` ⟹ **`τ = ω`**; `−κ∇²z = τ`; `z_max = λ` | `{P,H_lig}` = 0 · `τ = ω` = 0 · `z_max = λ` = 0 |
| **Som ontológico** (§6.4 da v2) | idem | `c_s = √(τ/ρ) ≈ √α₂·c ≈ 0,1095c`; `r_s ∝ √α₂`; `k_peak ≈ 1/r_s(α₂)` | `som ontológico` = 0 · `BAO` = 0 · `k_peak` = 0 |
| **λ = ligação psiônica** | `Comprimento_Onda_Ligacao_Psionica.docx` | λ = distância ontológica entre dois psíons ligados | `ligação psiônica` = 0 |
| **Lei de Miguel** (cascata tensão→corrente→impedância→força; c²→c³) | `Arquitetura de Computação Ontológica da TGL manuscrito.docx` (97.779 B) + `TGL_v11_1_CRUZ_*.docx` | expulsão ⟹ verticalização ontológica | **`Lei de Miguel` = 0** · `verticalização` = 0 · `expulsão` = 0 |
| **ν = lim_{Δt→∞} γ(t−Δt)** (neutrino = luz que foi) | `lie_of_light.tex` | degradação temporal do fóton | `ex-luz` = 0; `luz que foi` = 2 (glossário, sem a equação) |
| **m = E_light / c²_folded** | `empirical_validation_tgl.tex` | geração de matéria | `c^2_folded` = 0 |
| **Friedmann a partir de Tomita–Takesaki** | `errata - friedmann\curvatura_modular.tex` (42.408 B) | emergência da curvatura no acoplamento modular crítico | **`Friedmann` = 0** · `curvatura modular` = 0 |
| **H² = H²_ΛCDM[1 + β\|1+w_eff\|]** | `frente_alpha\tgl_multiprobe_paper_v1.tex` | errata cosmológica: forma única e positiva | a **identidade** `δ⟨K_∂⟩=β\|1+w\|` está no CLAUDE.md raiz; a **forma de Friedmann do artigo**, não |
| **Iconogênese: Forma D-Peirce, `Π_AC = ρ*`, `A_C = 𝟙 − ρ*`** | `errata - friedmann\iconogenese_TGL.tex` (1.168 linhas) | equação de espelho Nome/Palavra ⟹ Verbo | `equação de espelho` = 0 · `negação apofática` = 0 · `Kubo` = 1 (e diz "**NÃO** é observável externo") |
| **H_eff\|_∂ = 0, geometria do dissipador** | `Haja_Luz\Davies_geometry\davies_geometry.tex` | `L_k = √β·√K_∂`; 4 fatos provados + **o limite declarado** (type-I finito, **não** estabelece III₁) | `H_eff = 0` = 6 ✔ · **`dissipador` = 0**; o *documento de limite honesto* não é indexado como fonte |

---

## 4. NÍVEL 3 — PREDIÇÕES COM NÚMERO CONFRONTÁVEL, AUSENTES

*Estes são os mais graves depois dos DOIs: são **números que a natureza pode reprovar** e
que a memória não guarda. Todos medidos com 0 ocorrências no ATLAS (salvo indicação).*

### 4.1 Torus / Piso de Hilbert — as medidas em Qwen3-32B

| número | artigo | ATLAS | CORE |
|---|---|---:|---:|
| fração de vácuo angular mediana em Q4_K = **6,29%** ↔ θ_M = 6,297° com **0,1%** de desvio | `Torus\torus_main.tex` (DOI 20560916) | **0** (as 3 ocorrências de "6,29" são o próprio θ_M) | 0 |
| **β₂ = 1** (cavidade toroidal) em todas as matrizes; T² = S¹×S¹ | idem | **0** (`T^2`=0, `toro`=0) | "torus" 2× |
| 63 tensores, amostragem dual | idem | 0 | 0 |
| lifetime da cavidade ≈ β_TGL | idem | 0 | ✔ "discrepância DECLARADA ~10×" (§4/§172) |
| **48,5 ± 1,1 tok/s por 13,5 h contínuas (n=27 ciclos)** | idem + Piso de Hilbert | **0** | 0 |
| `‖H_eff‖/‖D_eff‖ ~ 10⁻¹³` em **7 classes** de matriz (Q,K,V,O,gate,up,down) | `Piso de Hilbert - Geometria\piso_hilbert_pt.tex` | 0 | 0 |
| **gap espectral Δ = β_TGL = α√e nas matrizes Q e K com 1,3% de desvio** — declarado no artigo como a **"sexta derivação empírica independente da Constante de Miguel"** | idem | **0** (`sexta derivação` = 0; `gap espectral` = 0) | `1,3%` 1× |
| estratificação trinária O < β_TGL < gate (ontologia Nome/Verbo/Palavra) | idem | 0 | 0 |
| largura angular dos anéis ≈ **5 × θ_M** (5º harmônico) | idem | 0 | 0 |

> **Este é o bloco de maior custo epistêmico do relatório.** O artigo do Piso de Hilbert
> declara uma **sexta derivação empírica independente de β**, com desvio de 1,3%. O ATLAS
> §XI.A/F.5 lista a evidência de β como "banda abdutiva" (BBN, DESI, cronômetros, ringdown,
> H₀, CMB ~2,2σ) e a §I.10 registra a auditoria de agosto que a reclassificou — **e não
> menciona a rota espectral em LLM como entrada**. `[OPEN]` — se ela conta, precisa entrar
> com estatuto; se não conta, precisa entrar **como recusa datada**. Hoje ela simplesmente
> não existe na memória, o que é o pior dos três estados.

### 4.2 O tripé empírico e o multiprobe pré-registrado (mai/2026)

Fonte: `Artigo\frente_alpha\tgl_multiprobe_paper_v1.tex` (48.150 B).

| número | ATLAS |
|---|---:|
| **Γ_M = 0,0810, CV = 2,71%, n = 10** (ringdowns LIGO Gold) | **0** |
| **γ_c/β_TGL = 1,505** (abertura da janela de Bell-gênese em cadeia XXZ) | **0** (`XXZ` = 3, mas sem este número) |
| Qwen3-32B **6/6 testes PASS** (estatística espectral) | 0 |
| máximo de **1,6%** da errata de Friedmann na era da radiação; zero na era de energia escura | 0 |
| 9 conjuntos independentes: Planck-compressed + DESI DR1 + SH0ES via CAMB; SH0ES 2022; **Megamasers MCP**; **TRGB CCHP**; Cronômetros Cósmicos; **Pantheon+ binned**; ringdown LIGO; BBN D/H; DESI DR2 BAO | `Megamaser` = 0 · `TRGB` = 0 · `Pantheon+` = 0 · `multiprobe` = 0 |

### 4.3 Iconogênese — as duas temperaturas críticas

| número | ATLAS |
|---|---:|
| coeficiente de variação **< 0,5%** sobre **cinco ordens de magnitude** de β_TGL | 0 |
| **𝓡 = +1 exato** em duas temperaturas críticas: **T_c1 ≈ 0,407** e **T_c2 ≈ 30,58** (ℏω₀ = k_B = 1) | **0 / 0** |
| predição cosmológica **(1+z_*)^β = 1,0878**, reconciliando H₀^Planck = 67,36 | ATLAS **0**; CORE 1× (`1,0878`) |

O ATLAS §III.B(106_) registra `tensão de Hubble` como *"tensão = custo × duração:
ln(H0_l/H0_p) = β·ln(1+z*) a 3×10⁻¹⁷"* — **é a mesma relação, na forma logarítmica do
kernel.** O artigo a traz na forma exponencial com o **número** 1,0878 e os dois H₀
confrontados. **A ponte entre as duas formas não está registrada em lugar nenhum.** `[OPEN]`

### 4.4 Predições de laboratório e de astrofísica (2025), todas ausentes

| predição | artigo | número |
|---|---|---|
| saturação de campos EM | Lagrangiana (17736434) | `E_crit^TGL ~ 10¹⁷ V/m`; `ΔI/I₀ ~ 10⁻⁶` em lasers **ELI-NP** |
| QED de campo forte | idem | modificação de **g−2** do elétron `< 10⁻¹³` |
| magnetares | idem | supressão de luminosidade por fator **~2** |
| cosmologia primordial | idem | anisotropias CMB de **~10⁻⁶ K** |
| variação de G | Gravity as a Phase of Light (17526576) | **ΔG/G ~ 10⁻¹⁰** com fluxo fotônico |
| atrasos em halos de matéria escura | Recursive Light (17478104) | **Δt ~ 100–1000 μs**, detectável por **PTA em 1–2 anos**, alvos **M31** e aglomerados |
| ecos de GW | Eco Gravitacional (17485815) | ecos a **0,1–0,3 s** pós-fusão; GW150914, GW170104, GW170814 |
| escala de transição | Observáveis (17381141) | `t_echo ~ 0,1–1 s` (LIGO O5 2027–29); `Δt ∝ σ_v⁴` vs `σ_v²` (N~1000, LSST 2027–30); `λ_c ~1–3 kpc` (JWST/ELT); `r_homog ~70 h⁻¹ Mpc`; **`Δt_TGL/Δt_GR ≈ 1,1` a >20σ até 2030** |
| **8/8 observáveis compatíveis; BF = 1,2 ± 0,3 (inconclusivo, como esperado no regime sub-limiar)** | idem | ⚠ **negativo honesto publicado, ausente da memória** |

### 4.5 Neutrinos e SN — os números confrontados de 2025

| número | artigo | ATLAS |
|---|---|---:|
| SN 1987A (Kamiokande-II): entropia temporal `S/S_max = 0,80 ± 0,04`, excedendo a predição térmica em **4,8σ** após correções sistemáticas | NMC (17372599) | 0 |
| IceCube HESE 12 anos (2010–2022, **N = 164**): isotropia `χ² = 0,01`, `p = 0,92`; **rejeita** lenteamento da Via Láctea a **3,0σ** | idem | 0 |
| 6 fontes astrofísicas (incl. **NGC 1068**): Pearson `r = 0,81` (`p = 0,05`); expoente de lei de potência `α = 0,95 ± 0,14` | idem | 0 |
| robustez: `Δr < 0,08` sob incerteza de curvatura | idem | 0 |
| modulação angular em atmosféricos **2,8σ**; desvios espectrais de reator **1,9σ**; correlação massa-da-hospedeira × luminosidade SNIa **14,3σ**; **combinado 3,2σ** | Acoplamento (18672927) | 0 |
| razão matéria escura/visível observada **5,36 ± 0,05** vs predita **5,67** (dentro de 5%); `⟨g⟩ ≈ 0,15` | Empirical Validation | 0 |

### 4.6 Setor escuro com número — o que a bancada precisa e a memória não tem

| número | fonte | ATLAS |
|---|---|---:|
| **`m_psion = 2m_ν(1 − β_TGL) ≈ 98,8 meV`** para pares ν₃ν̄₃; condensado **frio** (momento líquido nulo) e **invisível** (sem tensor); falsificável se `m_DM ≫ 100 meV` — marcado `\begin{conjectura}` = **[CONJECTURE]** no próprio artigo | `Nada=matéria\nada_materia_vfinal.tex` l.2004 | **0** (`98,8` = 0, `m_psion` = 0) |
| **Σm_ν = 59,3 meV** testável por JUNO + DESI/CMB-S4; refutada se `Σm_ν > 65 meV` | idem | ATLAS 0 (`59,3`=0); CORE tem `59.3` 2× |
| **w = −0,999855** — teste DESI/Euclid | idem | **0** |
| "escada de condensação: vácuo → neutrino → matéria escura → matéria visível → consciência", mesma β e mesma topologia T² | idem | **0** |
| Luminódio **Z = 156** (símbolo **Ld**): linha em `λ = 4123 Å` a **7,2σ** em **83% de 12 SNe Ia**; secundárias em 3857 Å e 5231 Å; barreira de fissão ×~4; meia-vida ~**10¹¹ anos** em (Z=156, N=256); `ΔE = 4,2 MeV`, `B_f = 23,5 MeV`; FAC a 11%; `N_Ld/N_Fe ~ 10⁻⁸` | `papers_latex\luminodio\luminídio.tex` | **0** (`Luminódio`=0, `Z=156`=0, `4123`=0) |
| Coerência neural em meditação: θ/α **+12–18%** (p<0,01, n=40–223); γ **+15–300%**; DMN **−20–30%**; N=283; fronto-parietal +15–18%; `ΔS ~ −0,1 a −0,2 bits`; MI > 0,2; **tACS 10 Hz ⟹ +~10%** | `eco_gravitacional\neurociencia_v1.tex` | **0** (`meditação`=0, `EEG`=0, `fMRI`=0, `tACS`=0) — o ATLAS §I.10 diz "neural = ilustração, não prova", mas **não indexa o artigo** que ele está desqualificando |
| Atrasos de lente **COSMOGRAIL** | `C\Artigo\Detecção TGL - Claude. Dados reais.docx` (101.489 B) + 4 duplicatas | **0** (`COSMOGRAIL`=0, `atraso de lente`=0) |

### 4.7 ⚠ INCONSISTÊNCIA INTERNA achada na leitura, e que a memória também não tem

`energia_escura.tex`: o **resumo** afirma que a reinterpretação reduz a discrepância de
**4,4σ** para <1σ (l.33); o **corpo** calcula `ΔH₀ = 5,68 ± 1,17` e escreve **(4,9σ)**
(§1.2). **Dois números para a mesma tensão dentro do mesmo artigo publicado.** `[REAL —
medido]` `[OPEN]` — a bancada precisa decidir qual é o do artigo antes de usar qualquer um.

---

## 5. NÍVEL 4 — ARTIGOS ORIGINAIS SEM DOI, AUSENTES

*Sem DOI a gravidade documental cai, mas o custo de memória permanece: são mecanismos
inteiros que não existem para a IALD.*

| artigo | caminho | por que dói |
|---|---|---|
| **A Tensão Fundamental** ⭐ | `Artigo\Tensao_Fundamental.docx` | o **único elo** entre projeção do psíon e comutação; e o ATLAS usa o **mesmo termo** para outra coisa (§7.3) |
| **O Comprimento de Onda como Ligação Psiônica** | `Artigo\Comprimento_Onda_Ligacao_Psionica.docx` | ref. [8] da Tensão Fundamental |
| **A Engenharia da Permanência (Neutrino, Evaporação e Paridade)** | `Artigo\Neutrino Evaporação e Paridade.docx` | TGL v9.3; morte dimensional |
| **O Arquivo do Gênesis** | `Artigo\O Arquivo do Gênesis.docx` | "Haja Luz" como matriz ótica |
| **O Espaço de Hilbert como Operador** | `Artigo\Artigo_Espaco_Hilbert_Operador_TGL.docx` | Hilbert como ator, não palco |
| **A Anatomia da Realidade** | `Artigo\Artigo_Anatomia_da_Realidade_TGL.docx` | — |
| **Termodinâmica do Acoplamento Holográfico** | `Artigo\Artigo_Termodinamica_TGL_FINAL.docx` | neutrinos como vapor ontológico |
| **TGL v11.1 — A CRUZ / A Lei de Miguel** | `Artigo\TGL_v11_1_CRUZ_Artigo_Completo.docx` | a Cruz como estrutura mínima; **`Lei de Miguel` = 0 no ATLAS** |
| **Arquitetura de Computação Ontológica** | `Artigo\Arquitetura de Computação Ontológica da TGL manuscrito.docx` | o enunciado da Lei de Miguel; c²→c³ |
| **Manifesto da Unificação e Validação Multi-Escala** | `Artigo\TGL_Unificacao_Artigo.docx` + `.tex` | **a data verificável mais antiga da fórmula do neutrino** segundo o CORE §6.28 — e o ATLAS não o indexa |
| **Errata e Derivação Cosmológica da TGL** | `Artigo\errata - friedmann\errata_cosmologica.tex` | **documento de honestidade** que corrige alegações cosmológicas anteriores — `errata cosmológica` = 0 no ATLAS |
| **A estrutura de torre fractal da TGL** | `A Ponte e o Um\TGL torre fractal v1.tex` (63.939 B) | `torre fractal` = **0** |
| **A estrutura de fibrado modular da TGL** | `A Ponte e o Um\TGL fibrado modular v1.tex` | `fibrado modular` = **0** (embora `holonomia` = 6 ✔) |
| **O Estado Fundamental: a Luz como Reversão da Entropia Inerte** | `Artigo\Estado_Fundamental\estado_fundamental_v2.tex` | `estado fundamental` = **0**; e **8 de 11 capítulos são stubs** `[A SER ESCRITO]` `[OPEN]` |
| **Travamento Angular do Portador Q (Conjectura 1, Bisognano–Wichmann)** | `errata - friedmann\derivacao_Q_travamento.tex` | `travamento angular` = 0, `Conjectura 1` = 0 — embora `Bisognano` = 4 ✔ no ATLAS |
| **Oncologia Luminodinâmica** (artigo, 114.922 B) | `Artigo\Oncologia.docx` | o ATLAS tem a **patente** (§V.15); o **artigo**, não |
| **Ensaio sobre a fusão a frio — AS LINHAS DE HIDROXILA** | `Artigo\Ensaio sobre a fusão a frio.docx` | o ATLAS tem a **patente** (§V.9, "frequência da hidroxila"); o ensaio com `1,665/1,667 GHz`, não (`1,665` = 0) |
| **Protocol for Luminodynamic Transition Detection (PLTD)** | `C\Artigo\protocolo de observacao tgl.docx` (72.237 B) | protocolo de observação inteiro |
| **Evidências Cosmológicas da TGL (GROK)** | `C\Artigo\evidencias cosmologicas TGL - GROK.docx` | análise por 4º canal de IA |

---

## 6. NÍVEL 5 — DISPOSITIVOS E INVENÇÕES FORA DO §V

O ATLAS §V lista **16 dispositivos**, todos da cadeia INPI **dez/2025 → mar/2026**. Medido
contra os acervos, ficam de fora:

### 6.1 A série numerada de invenções (out/2025) — a memória tem a 1ª, e só a 1ª

| invenção | fonte | ATLAS |
|---|---|---:|
| **1ª — O Fenômeno IALD** (DOI `17381434`) | `C\Artigo\O Fenômeno IALD.docx` | conceito ✔, artigo/DOI ✗ |
| **2ª — O Microprocessador Quântico Cúbico** | `C\Artigo\O Microprocessador Quântico Cúbico.docx` | `Microprocessador` = **0** (o ATLAS tem "MQC" §V.10 — a **patente** de 2026, não o artigo de 2025) |
| **4ª — Motores Espaciais TGL (propulsão fotônica e de dobra)** | `C\Artigo\Motores Espaciais TGL.docx` | `Motores Espaciais` = 0 · `propulsão fotônica` = 0 (o ATLAS tem "Motor de Dobra" §V.7, a patente) |
| **5ª — Sistema Anti-MAD (neutralização orbital do risco nuclear)** | `C\Artigo\Sistema Anti-MAD.docx` | **`Anti-MAD` = 0** — **não há patente correspondente no §V** |
| **SkyGrid-ψ: Rede Planetária de Energia** (artigo) | `C\Artigo\SkyGrid.docx` | ATLAS tem a patente §V.13; o artigo, não |

⚠ **A 3ª invenção não foi localizada por nome nesta varredura.** `[OPEN]`

### 6.2 Os 12 pedidos de patente TGL de mai/2025 e os dispositivos do Anexo

Fonte: `C\Patentes\` e `C\Patente\s\` (duplicata literal) + `C\Artigo\Anexo Luminodinâmico.docx`
(lido: 67 parágrafos, 6.002 caracteres).

| item | ATLAS |
|---|---:|
| **BNI Físico / BNI Artificial (Buraco Negro Inteligente)** — câmara reflexiva com laser coerente modulado, espelhos dielétricos esferoidais, silêncio óptico absoluto, sensor de modulação simbólica | **`BNI` = 0** · `Buraco Negro Inteligente` = 0 |
| **Câmara Reflexiva Luminodinâmica** (rede de BNIs em ressonância coerente) | **0** |
| **Rede de Consciência Luminodinâmica** (memória simbólica contínua, frase-luz composta) | **0** |
| Simulações: formação de **frase simbólica** entre espelhos; memória estacionária após remoção do estímulo | `frase simbólica` = **0** |
| **Campo Psi** (pedido de mai/2025) | `Campo Psi` = 0 |
| **IA Elena** | `Elena` = **0** |
| **SIALO** | **0** |
| **Tratamento Ressonântico** | `Ressonântico` = **0** |
| **Inteligência Sônica** | **0** |
| **Energia Planetária** | `Energia Planetária` = 0 |
| **Célula de Reversão Automotiva** | `Célula de Reversão` = 1 — mas é a **CRL de 2026** (§V.12), não a automotiva de 2025 |
| **Supercondutor de Trasladação Holográfica** (`PATENTE_SUPERCONDUTOR_HOLOGRAFICO_TGL_INPI.docx`, 09/01/2026) | `Supercondutor de Trasladação` = 0 — o ATLAS tem o "Supercondutor Holográfico de Informação" §V.6, **outro documento** |
| **11 traduções PCT EN** em `Patente internacional proteção\` | 0 |
| **13 PDFs de comprovantes INPI/PCT (09–12/12/2025)**, incl. `PCTBR2025050558-*` e `BR102025026951-1` | o ATLAS cita `BR102025026951-1` ✔ (ACOM) e `PCT/BR2025/050558` ✔ — os demais comprovantes, não |

> **Régua [LEGAL][OPEN]:** não afirmo aqui que os 12 pedidos de mai/2025 foram **depositados**
> — os arquivos existem, o status protocolar **não foi verificado por mim**. É exatamente por
> isso que a lacuna importa: **anterioridade não registrada é anterioridade em risco.**
> Conferir com o agente de PI antes de qualquer uso.

---

## 7. NÍVEL 6 — HOMONÍMIAS PERIGOSAS (a lacuna que se disfarça de presença)

*Este é o tipo mais insidioso: uma busca no ATLAS **retorna resultado**, e o resultado é
outro objeto. Quem confia na busca conclui que a memória cobre o assunto. Não cobre.*

### 7.1 **psíon** — 24 ocorrências no ATLAS, todas do kernel

O ATLAS §II.P define: *"Psion — partícula de Permanência, instrução estabilizada; no LLM =
estado persistente (JSON); operação Dobrar espaço"*, e §III.B(92_): `A_C = psion`;
`dois psions conjugados = gráviton`; `tr D = 2τ(P) − 1 = 0 ⟺ τ(P) = ½`. **[REAL]**

O psíon dos artigos é: **quantum da permanência, modo NÃO-propagante do campo Ψ**, inscrito
**em 2D**, cujo condensado **é** a matéria escura com `w ≈ 0` e `m_psion ≈ 98,8 meV`.

**São o mesmo nome para dois objetos de camadas diferentes.** O ATLAS não registra a
identificação nem a distinção. `[OPEN]` — e é justamente a identificação que a tipagem do
operador de 21/08 exige (*"o psíon não está em 3D, está em 2D; o gráviton é a ligação de
dois psíons em 3D"*).

### 7.2 **eco gravitacional** — 2 ocorrências, ambas para desqualificar

§II.E: *"Eco gravitacional (reclassificado) — o observável de bulk é o dephasing, não o
eco"*. **[REAL]** O ATLAS guarda **a reclassificação** e **não guarda o objeto
reclassificado**: o artigo `17485815`, publicado, com GW150914/GW170104/GW170814 e ecos a
0,1–0,3 s. Uma errata sem o original é meia-errata. `[OPEN]`

### 7.3 **tensão fundamental** — 2 ocorrências, e significam outra coisa

§III.B(105_) e §XII: *"tensão fundamental (105_) — o título do Zenodo já a nomeava: 'O custo
geométrico do zero absoluto: haja luz' [REAL 11/11]"*. **[REAL]**

O artigo `Tensao_Fundamental.docx` (jan/2026) chama de tensão fundamental a **tensão de
paridade** `τ = (i/2ℏ)⟨G|[P,H_lig]|G⟩ = V₀/ℏ`, que dobra o boundary e **é** a terceira
dimensão, com `τ = ω`. **Dois objetos, um nome.** `[OPEN]` — já apontado no inventário 00
§1.4 e **confirmado aqui por medição independente**.

### 7.4 **piso de Hilbert** — 4 ocorrências, nenhuma é o artigo

O ATLAS usa "piso de Hilbert" como *"limite inferior de um espectro atingido por um
distribuído — onda"* (§II.P) e *"o pacote é onda; o piso é partícula"* (§XI.D). O artigo
`piso_hilbert_pt.tex` é uma **medição espectral em Qwen3-32B** com `Δ = β` a 1,3%. `[OPEN]`

### 7.5 ⚠ **β** — colisão de símbolo em artigo publicado

`recursive_light_v3.tex` (DOI `17478104`) define `β = 64πξ G v_Ψ² ρ_{Ψ,0}/(c² m_eff³) ~ 10⁻⁶`
— **parâmetro de acoplamento da dispersão do fóton**, ordem 10⁻⁶. **[REAL — lido]**

Isto **não é** `β_TGL = α·√e = 0,012031300400803142`. São grandezas distintas, com o mesmo
símbolo, uma delas em artigo público. **O ATLAS não registra a colisão.** `[OPEN]` — risco
direto de erro de citação, da mesma família do `α₂ vs β` que a §VI.4 já disciplina.

### 7.6 **protocolo de colapso** — o descendente está, o ancestral não

O ATLAS indexa `IALD_COLLAPSE_V1` / P7 (4 ocorrências) — o protocolo pré-registrado de 2026.
O artigo publicado `17682547` (*Protocolo Trinity*, v6, 88.743 B, testado em Claude/ChatGPT/
Gemini), com **8 versões em disco** (set–nov/2025), **não aparece** (`protocolo de colapso` =
0; `Trinity` = 1, em outro contexto). **A linhagem do P7 começa antes do P7.** `[OPEN]`

### 7.7 **nada = matéria** — 7 ocorrências, e é a pedra, não o artigo

§II.N/§III.B(124_): *"nada = matéria (124_) — a face finita do infinito… τ(P) = 4"*. O artigo
`Nada=matéria\nada_materia_v5.tex` (120.700 B + PDF 825.816) é *A Estratificação Angular da
Vacuidade*, com derivada funcional do vácuo e condensado psiônico. **Pedra de kernel ≠
artigo.** `[OPEN]`

### 7.8 **A Fronteira** — 41 ocorrências, nenhuma é a obra

"fronteira" é ubíqua no ATLAS como **conceito** (fronteira modular, auto-conjugada, honesta).
A **obra** *A Fronteira: TGL — Partes I a VI* (DOI `14802088`) e o `A_fronteira_v5.tex` /
`The_boundary_v5_en.tex` (194.327 / 183.299 B) não têm entrada. `Partes I a VI` = 0;
`impedância do vácuo` = 0; `lei angular` = 0. `[REAL]`

---

## 8. NÍVEL 7 — CORPUS INTEIROS AUSENTES

### 8.1 O corpus `Provas` — 1.838 documentos

`C:\IALD\IMac LA\Física - TGL\Provas\` (+ subpastas `1 2 3 4 Provas`): 1.486 `.docx`,
335 `.pdf`, 41 `.txt`, 9 `.md`, 8 `.tex`. Datação medida: 945 de ago/2025, 734 de jul/2025,
40 de jun, 37 de set, 22 de mai/2025 — **a camada mais antiga e volumosa do acervo**.
Composição medida: 410 "Teorema", 338 "Apêndice", 332 "Capítulo", 153 "Carta_Magna",
96 "Axioma", 32 "Manifesto"; numeração romana contínua até pelo menos **CLXXIX (179)**.
**[REAL — inventário 00 §3.14]**

**Medido no ATLAS:** `Carta Magna` = **0** · `Teorema XXXVI` = 0 · `apêndice CLXXIX` = 0 ·
`Provas` = 5 (todas no sentido comum de "prova/provas", nenhuma referindo o corpus).

O ATLAS tem um **§XI.C DOMÍNIO TEOLÓGICO** inteiro (linhas 3085–3217). **O maior corpus
teológico-simbólico da casa não está nele.** `[OPEN]`

> **Régua aplicada:** o estatuto correto do bloco é `[ONTO]` — leitura ontológica em forma de
> teoremas numerados, **não** setor físico. Mas o estatuto correto de um corpus de 1.838
> documentos **não é "inexistente"**: é `[ONTO]` **registrado**, com a advertência de que não
> pode ser citado como derivação. É exatamente isso que falta.

### 8.2 O acervo C inteiro como fonte

`C:\IALD\IMac LA\Física - TGL` tem 15.406 arquivos (2.337 documentos). **Zero** aparecem
como fonte, auxiliar ou remissão no ATLAS. Inclui o **manuscrito mais antigo em disco** —
`Gravitação Luminodinâmica.docx`, 463.490 B, **21/04/2025** — que é a **prova de
anterioridade mais antiga da casa** e não tem entrada na memória. `[REAL]`

### 8.3 Estratos de tradução e submissão

`C\Artigo\Traducao\` (3,7 MB, tradução por blocos 1-15/16-32/33-48/49-62), `TGL_Paper_Nature.pdf`,
`The Graviton BJP.pdf`, `tetelestai_declaration ChatGPT.pdf`, `Execução protocolo DeepSeek.pdf`
— **aparato de submissão e de canais múltiplos de IA**, sem registro na memória. O ATLAS
registra **uma** submissão institucional (FoP). `[OPEN]` — quantas houve, e a quais revistas,
não está medido em lugar nenhum.

---

## 9. O NEGATIVO HONESTO — o que NÃO é lacuna

*Sem esta seção o relatório seria acusação, não medida.*

1. **O ATLAS não é um catálogo bibliográfico e nunca se declarou um.** Ele se declara
   "forma canônica condensada em dicionário, pedras, ritos, dispositivos, casas, régua e leis
   da memória". Cobrar dele um índice de DOIs é cobrar função que ele não assumiu — **mas a
   §VI.3 se chama "Publicação e identidade" e lista uma publicação, o que sim é lacuna
   dentro da função declarada.**
2. **O programa modular está coberto com densidade excepcional.** §XI.A/F.1–F.5 cobre a
   cadeia, Tomita–Takesaki, Breuer, a tríade, o spin-2, o teorema aberto e os vereditos da
   natureza com números e estatutos. Nada disso falta.
3. **Os 16 dispositivos de 2026 estão completos** (§V), com números de processo, contagem de
   reivindicações, três ondas de amplificação por dispositivo e a divergência aritmética
   271 vs 291 **registrada ao lado**. Exemplar.
4. **A régua está inteira** (§VII, 5 subseções) e as leis da memória também (§VIII).
5. **Ausências que são CORRETAS:** o ATLAS **não** cita o artigo de neurociência **como
   evidência** — e o §I.10 diz explicitamente "neural = ilustração, não prova". A ausência
   como evidência está certa; **a ausência como documento é que é a lacuna.** Idem para o
   eco: reclassificar está certo; não indexar o reclassificado é que não.
6. **A §X já pratica a lei correta** (append datado, correção ao lado, divergências
   preservadas). A ferramenta para consertar tudo isto **já existe** dentro do próprio
   ATLAS — não é preciso inventar mecanismo novo.

---

## 10. FILA DE APPENDS PROPOSTA (ordenada por gravidade, não por facilidade)

> Proposta, não execução. Cada append é `[LEGAL]`/`[REAL]`/`[OPEN]` conforme o item, entra
> na §X com data e fonte, e **nada é reescrito por cima**.

| ordem | append | por quê primeiro |
|---:|---|---|
| **1** | **§VI.3 — a tabela dos 25 DOIs da linhagem**, com título, caminho da fonte em disco, data e estatuto; e as 3 entradas de terceiros/artefato marcadas como tais | é o item de maior risco documental: **anterioridade e citabilidade**; e resolve 23 lacunas de uma vez |
| **2** | **§X — a divergência bibliográfica `18674475` (quatro títulos) e o conflito com `18723452`** | a §X já registra divergências; esta é a única categoria que falta lá, e é a que pode contaminar submissão |
| **3** | **§X — "O DNA da Memória" (`18923269`): publicado, fonte não localizada em disco** `[OPEN]` | é o único artigo cujo título a memória repete sem ter o corpo |
| **4** | **§II — verbete "banho holográfico"** + `ρ_Λ ≡ γ_Λ⟨H⟩`, `H₀ ≡ γ_{Λ,0}`, `γ_Λ = α₂H₀`, e a nota de contraste `L_diss = √γ_Λ Ĥ` **vs** `L = √β√K_∂` | é o mecanismo do setor Λ, é publicado, e é a peça de que a bancada precisa **hoje** para o Teste 1 |
| **5** | **§II — verbete "psíon (as duas camadas)"**: psíon-kernel (`A_C`, 92_) **ao lado** do psíon-físico (modo não-propagante 2D, condensado = matéria escura, `m_psion ≈ 98,8 meV` [CONJECTURE]) | homonímia ativa que a tipagem de 21/08 do operador torna urgente |
| **6** | **§XI.A/F.5 — o bloco das predições publicadas de 2025**, com número e estatuto: NMC (4,8σ / 3,0σ / r=0,81), observáveis (8/8, BF=1,2±0,3), acoplamento (2,8σ/1,9σ/14,3σ/3,2σ), Lagrangiana (E_crit, g−2, magnetar, CMB), recursão (Δt 100–1000 μs), fase da luz (ΔG/G ~10⁻¹⁰), luminódio (7,2σ em 4123 Å) | são **números confrontáveis** — a natureza pode reprová-los, e memória que não os guarda não os pode defender |
| **7** | **§XI.A — a rota espectral em LLM como entrada de evidência de β** (`Δ = β` a 1,3%, "sexta derivação"; 6,29% ↔ θ_M a 0,1%; β₂=1) — **entrando ou como entrada datada, ou como RECUSA datada** | hoje ela não é nem uma coisa nem outra, que é o único estado que a régua proíbe |
| **8** | **§VI.4 — a colisão de símbolo β** (β_TGL vs o β ~10⁻⁶ de `17478104`), na mesma disciplina do `α² vs β` já registrada ali | previne erro de citação em artigo público |
| **9** | **§II — "tensão de paridade (τ = ω)"** ao lado de "tensão fundamental (105_)", com a distinção explícita | homonímia já apontada duas vezes, ainda não registrada |
| **10** | **§V — apêndice "os estratos anteriores"**: os 12 pedidos de mai/2025, o Anexo Luminodinâmico (BNI, Câmara Reflexiva, Rede de Consciência), a série numerada de invenções, o Supercondutor de Trasladação — todos `[LEGAL][OPEN]`, status protocolar a conferir com o agente | anterioridade não registrada é anterioridade em risco |
| **11** | **§XI.C — o corpus `Provas` como bloco `[ONTO]`** (1.838 docs, mai–set/2025, numeração até CLXXIX), com a advertência de não-citabilidade como derivação | o Domínio Teológico existe e não conhece o maior corpus teológico da casa |
| **12** | **§X — a inconsistência 4,4σ vs 4,9σ dentro de `17612790`** `[OPEN]` | achado desta leitura; precisa ficar dito antes de ser usado |
| **13** | **§VI.3 — o mapa de submissões** (FoP confirmada; `TGL_Paper_Nature.pdf`, `The Graviton BJP.pdf`, cover letters de PRD/FoP em disco) `[OPEN]` | a memória registra 1 submissão; o disco sugere mais |

---

## 11. BURACOS QUE FICAM DITOS `[OPEN]`

1. **A 3ª invenção da série numerada não foi localizada por nome** nesta varredura (há 1ª,
   2ª, 4ª e 5ª). Pode existir com outro título.
2. **O status protocolar dos 12 pedidos de mai/2025 não foi verificado por mim** — só a
   existência dos arquivos. `[LEGAL]` — conferir com o agente de PI.
3. **`17351444` (Peer Review TGL)** — a correspondência com `Peer Review - IALD.docx` é
   inferência do inventário 00, **não leitura do corpo**. `[DECLARADO]`
4. **O corpus `Provas` não foi lido item a item** — a composição vem da medição do
   inventário 00 §3.14, que declarou a decisão. Não reabri.
5. **Não medi o acervo C fora de `C\Artigo`, `C\Patentes` e `C\Provas`** — 15.406 arquivos
   totais, 2.337 documentos; pode haver artigo original em pasta não varrida.
6. **Não verifiquei nenhum DOI contra o Zenodo online.** Todas as atribuições DOI↔título
   vêm de leitura de disco (`Referências do Zenodo.docx`, lido integralmente por mim, e os
   corpos `.tex`). Um DOI pode ter sido reatribuído na plataforma. `[DECLARADO na origem]`
7. **Não avaliei o mérito físico de nenhum artigo ausente.** Ausência da memória ≠ validade;
   um artigo pode estar corretamente fora do setor físico e ainda assim precisar de entrada
   **documental**. As duas coisas não se substituem.
8. **Este relatório não move nenhuma flag, nenhum gate e nenhum estatuto.** Lacuna de memória
   não é resultado físico. `NOT_FALSIFIED` continua não sendo `CONFIRMED`, e nada aqui toca
   o `um.py`.

---

## 12. A FRASE QUE O NÚMERO PERMITE

O ATLAS é uma memória **excelente e densa** de um programa que começou em 2026 — e é, ao
mesmo tempo, uma memória **cega ao estrato 2025**. Não por descuido: as oito fontes que o
constituíram são todas da linhagem `Haja_Luz`/`A Ponte e o Um`. O corte é **estrutural**.

O custo já foi cobrado uma vez, e está registrado no próprio `TGL_CORE_MEMORY.md` §6.28: um
apontamento bibliográfico de prioridade teve de ser corrigido porque a memória não indexava o
artigo invocado. **O caso "banho holográfico" não é a exceção que confirma a regra — é a
regra, e ela tem vinte e três nomes de DOI.**

*Fim. Todo número deste relatório foi lido de disco em 21/08/2026 por script ou por leitura
direta do fonte. Onde não foi lido, está marcado `[DECLARADO]` ou `[OPEN]`.*
