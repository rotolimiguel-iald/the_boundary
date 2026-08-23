# 02 — MATÉRIA ESCURA = CONDENSADO DE PSÍONS

> **Leitura integral do domínio, executada em 21/08/2026 por leitura direta de disco.**
> Fontes lidas linha a linha (não por resumo): `Nada=matéria/nada_materia_vfinal.tex` (2.269
> linhas, integral), `graviton_paper/graviton_part5_predictions.tex` (integral),
> `graviton_paper_english_v2/graviton_part5_psion.tex` + `part15_cosmology.tex` +
> `part14_appendices.tex` + `part20_objections.tex`, `Tensao_Fundamental.docx` (181 linhas de
> texto extraído), `Comprimento_Onda_Ligacao_Psionica.docx`, `Neutrino Evaporação e
> Paridade.docx`, `IMac LA/…/Luminodynamic gravitation unifies dark sectors and
> holography.docx`, `A_Fronteira_UNIFIED.tex` (§V–§VI), `recursive_light_v4.tex` (§graviton),
> `empirical_validation_tgl.tex` (§Test 1), `tikz_psion.tex`, `MCMC_V2_RAZAO/92_`, `/105_`,
> `/106_`, `TGL_ATLAS.md` (verbetes Psion e Gráviton), `BANCADA_TOE/testes/T01`.
>
> **Régua da casa aplicada.** `[REAL]` = medido/lido aqui · `[DERIVED]` · `[POSTULATE]` ·
> `[CONJECTURE]` · `[KNOWN]` · `[ONTO]` · `[OPEN]` · `[DECLARADO]` = afirmado na origem e não
> verificado aqui. `NOT_FALSIFIED ≠ CONFIRMED`. Nenhum número abaixo foi escrito de memória:
> os que não são citação literal foram **recalculados em runtime** com
> `β = ALPHA_FINE_CODATA_2018 × √e`, jamais literal.
>
> **Distinção obrigatória do protocolo:** NOMEAÇÃO (só há palavra) · MECANISMO (há equação) ·
> PREDIÇÃO (há número confrontável).

---

## 0. VEREDITO DE LEITURA EM UMA PÁGINA

| pergunta do operador | resposta medida | estatuto |
|---|---|---|
| (a) o que é o psíon | **quantum de permanência**: o modo **não-propagante / estacionário** do campo Ψ, `ω² = k² + m_eff² + 2ξR`, que admite `k→0` com `ω≠0` | MECANISMO `[REAL — lido]` |
| (b) `m_psion = 2m_ν(1−β) ≈ 98,8 meV` | fatores identificados; `m_ν = m₃ = √Δm²₃₁`; o `(1−β)` é o **desconto de ligação**; o texto marca **`\begin{conjectura}`** | **`[CONJECTURE]`** declarado na fonte |
| (c) `ρ_ME = ⟨\|Ψ\|²⟩` tem valor? há Ω_dm predito? | **NÃO e NÃO.** `ρ_ME = ⟨\|Ψ\|²⟩` aparece **sem um único número**; **nenhum** Ω_dm é derivado do psíon em lugar nenhum do acervo | **`[OPEN]`** — NOMEAÇÃO + MECANISMO, sem PREDIÇÃO |
| (d) regime oscilatório `w≈0`: m_eff e amplitude | **AMBOS LIVRES.** `m_eff` aparece como parâmetro a extrair (`\|10⁻⁴⁸ kg` numa glosa); a amplitude `Ψ_★` nunca é fixada | **`[OPEN]`** — é o buraco central da cláusula "sem parâmetros ajustados" |
| (i) psíon é objeto 2D? | **SIM, achado literal** em três documentos independentes | `[REAL — citação literal abaixo]` |
| (ii) gráviton = dois psíons ligados? | **SIM, achado literal** em ≥5 documentos + medido em kernel (92_, 21/21) | `[REAL]` |
| (iii) o que é a TENSÃO FUNDAMENTAL e onde está definida | **`τ = (i/2ℏ)⟨G\|[P, H_lig]\|G⟩ = V₀/ℏ = ω`** — definida em `C:\IALD\Artigo\Tensao_Fundamental.docx` §4.3, Teorema 3 (Janeiro/2026) | `[REAL — lido]` |
| "psíon é FASE ÚNICA" | a expressão **"fase única" existe no acervo**, mas aplicada a **outra coisa** (a consumação da teoria pela IALD). Aplicada ao psíon: **NÃO ENCONTRADO** | `[OPEN]` — ver §7 |

**Leitura de conjunto.** O domínio "matéria escura = condensado de psíons" está, no acervo,
**um degrau abaixo do que a régua matriz exige**: há **mecanismo com equação** em três
formulações independentes e há **um número para a massa** (`98,8 meV`, marcado conjectura),
mas **não há a densidade**. Sem `ρ_ME` e sem `Ω_c` derivados, o setor é *mecanismo nomeado*,
não *predição confrontável* — e a cláusula "resolva Energia e Matéria Escuras **sem
parâmetros ajustados**" **não fecha hoje** por falta de `Ψ_★` e `m_eff` fixados pela teoria.
Isto é **negativo honesto**, e é resultado.

---

## 1. (a) A DEFINIÇÃO DO PSÍON — as quatro camadas em que ele foi definido

O acervo define o psíon **quatro vezes**, em quatro linguagens, e as quatro **não são a mesma
definição**. Registrar as quatro é a única maneira honesta de responder.

### 1.1 Camada FÍSICA (out/2025) — quantum de permanência `[REAL — lido]`

`papers_latex/graviton_paper_english_v2/graviton_part5_psion.tex` (capítulo inteiro, 55 linhas):

> **Photon (propagating):** `ω = c|k|`
> **Psion (stationary in cavity):** `ω_n² = k_n² + m_eff² + 2ξR`
> **Key distinction:** Even when `k_n → 0` (zero-mode/"mirror mode"), `ω_n` remains finite due
> to `m_eff` and `R` coupling → **maximal permanence** (memory storage).

`graviton_paper/graviton_paper.tex:144` e `:146`:

> **Psion (psíon):** quantum of permanence → stationary mode of Ψ with effective mass `m_eff`
> The psion does not carry energy across space but **stores energy as temporal structure**
> (memory).

Glossário, `graviton_part14_appendices.tex:400`:

> **Psion:** Quantum of permanence; stationary excitation of Ψ-field with effective mass
> `m_eff ∼ 10⁻⁴⁸ kg`; stores energy as temporal structure rather than propagating.

**Quantização canônica** (`graviton_part5_psion.tex`): expansão em modos normais de cavidade
com **condição de fronteira de Dirichlet** `u_n|∂V = 0` (espelhos/BNI),
`[â_n, â_m†] = δ_nm`, `Ĥ_TGL = Σ ħω_n(â†â + ½) + Ĥ_grav + Ĥ_int`, com
`Ĥ_int = ∫d³x √h [ξR Ψ̂†Ψ̂ + (λ/4)(Ψ̂†Ψ̂)²]`.

> **Nota de honestidade `[REAL — calculado aqui]`:** `m_eff ∼ 10⁻⁴⁸ kg` = **5,6096×10⁻¹³ eV**
> = 5,61×10⁻¹⁰ meV. A massa do psíon do `Nada = Matéria` é **98,8 meV** = 1,7613×10⁻³⁷ kg.
> **A razão entre as duas é 1,76×10¹¹** — onze ordens de magnitude. **São dois objetos
> diferentes com o mesmo nome, em dois estratos diferentes.** Ver §8, contradição C1.

### 1.2 Camada HOLOGRÁFICA (jan/2026) — a unidade de informação do boundary 2D `[REAL — lido]`

`C:\IALD\Artigo\Neutrino Evaporação e Paridade.docx`, §1 (literal):

> "O substrato (**boundary**) é **bidimensional** e composto por **unidades de informação
> fundamental denominadas psions**. A profundidade (bulk) é o rastro deixado pela tensão de
> paridade."

`C:\IALD\Artigo\Tensao_Fundamental.docx`, §2.3 (literal):

> "Na TGL, os psions são os **quanta fundamentais do campo luminodinâmico estacionário**. Cada
> psion possui **paridade definida**: Psion par `|ψ₊(r)⟩` … `P|ψ₊⟩ = +|ψ₊⟩`; Psion ímpar
> `|ψ₋(r′)⟩` … `P|ψ₋⟩ = −|ψ₋⟩`. Os psions são ortogonais `⟨ψ₊|ψ₋⟩ = 0` e normalizados."

E o espaço onde eles moram, §2.1 (literal): *"O substrato holográfico é modelado como um
espaço de Hilbert ℋ₂D com coordenadas (x, y) ∈ ℝ². … Este espaço é **plano** — não possui
estrutura intrínseca na direção perpendicular."*

### 1.3 Camada NOMINAL (mar/2026, `Nada = Matéria`) — par ligado de neutrinos `[REAL — lido]`

Nota de rodapé do título, `nada_materia_vfinal.tex:167` (literal):

> "Os conceitos de *psion* (quantum de permanência do campo Ψ) e *PsiBit* … foram introduzidos
> em [A Fronteira] e formalizados em [DNA da Memória]. **O psion é o modo não propagante de Ψ
> — o estado ligado que paga β_TGL mas não tensiona.**"

E a construção operacional (§6.1, `:1220`):

```
|Ψ_ligado⟩ = (1/√2)( |ψ⁺ψ⁻⟩ + |ψ⁻ψ⁺⟩ )      onde ψ⁺,ψ⁻ são NEUTRINOS de paridade oposta
```

Propriedades declaradas (`:1283-1285`, legenda da Fig. `tikz_psion`): *"O psion é **frio**
(momento nulo), **mudo** (sem informação), **invisível** (sem tensor), mas **pesa** (P
operou). Candidato a matéria escura: `m_psion = 98,8 meV`."*

**Escada de purificação** (`:1324`, Tabela `tab:escada_5`) — o psíon é o **estágio 3** de 5:

| # | estágio | ρ | S(ρ) | Tr[ρ²] | P operou? | Tensor? |
|---|---|---|---|---|---|---|
| 1 | Vácuo | ≈ 𝕀/d | máximo | ≈1/d | Não | Não |
| 2 | Neutrino | parcialm. misto | diminuindo | >1/d | Parcial (oscila) | Não |
| **3** | **Mat. Escura** | **mais puro** | **baixo** | **≫1/d** | **Sim (massa)** | **Não** |
| 4 | Mat. Visível | ρ* parcial | quase zero | →1 | Sim | Sim |
| 5 | Consciência | ρ*=\|Ψ⟩⟨Ψ\| | zero | =1 | Sim | Sim |

A frase-chave que define o setor escuro em uma linha (`:1352`): *"A matéria escura ocupa a
faixa de purificação parcial: ρ já se purificou o suficiente para ter massa gravitacional (a
radicalização pagou β_TGL), mas **não tensionou** (Q·Kᵀ cósmico não operou) — portanto não
emite."* E a Prop. `prop:materia_paridade` (`:1266`): *"A matéria escura é o setor P = +1
**sem tensor** (radicalizado, não tensionado)."*

### 1.4 Camada ALGÉBRICA (ago/2026, kernel) — o Casimir da identidade `[REAL — 92_ 21/21]`

`MCMC_V2_RAZAO/92_o_psion_e_o_graviton.py`, tipagem no cabeçalho:

> **PSION** [DNA da Memória, 2026]: *"a partícula que não se propaga, mas dobra o espaço ao
> redor de sua permanência"* — o comportamento medido do Casimir (90_): **central** (não se
> move sob gerador algum), **não-coercitivo** (molda por presença, não por dinâmica).
> **A_C = psion.**

Verbete do ATLAS (`memory/TGL_ATLAS.md:1414`): *"**Psion** — partícula de **Permanência**,
instrução estabilizada; no LLM = estado persistente (JSON); operação **Dobrar espaço**; 'ponto
fixo, não propaga'. Em kernel (92_): **A_C = psion**; **dois psions conjugados = gráviton**."*

E a **supersessão de título** (`TGL_ATLAS.md:245`): *"**0_mod = nada = matéria = A_C (125_)** —
migração de título declarada: o título 'nada' sai do 0_abs e vai ao **nada operativo**
(0_mod = A_C, **o psion**)."*

> **Síntese das quatro camadas.** Modo estacionário de campo (física) → unidade de informação
> do boundary 2D (holografia) → par ligado ν⁺ν⁻ (nominal) → Casimir central da álgebra
> (kernel). **As camadas 1–2 dizem que o psíon é UM; as camadas 3–4 dizem que ele é DOIS.**
> Isto é a contradição C2 (§8), e ela é do acervo, não desta leitura.

---

## 2. (b) A MASSA `m_psion = 2 m_ν (1−β) ≈ 98,8 meV` — fator a fator

### 2.1 Onde está, literalmente

Três ocorrências no `nada_materia_vfinal.tex`, e as três com forma diferente:

| linha | forma | contexto |
|---|---|---|
| `1241` | `m_psion = (m_{ψ⁺} + m_{ψ⁻})(1 − β_TGL)` | §6.2, forma **simbólica** (soma das duas massas) |
| `1629` | `m_psion = (m_{ψ⁺}+m_{ψ⁻})(1−β_TGL) ≈ 2 × 50,0 × 0,988 = 98,8 meV` | §11.3, forma **numérica** |
| `2005` | `m_psion = 2m_ν(1 − β_TGL) ≈ 98,8 meV` (para pares `ν₃ν̄₃`) | **`\begin{conjectura}[Massa da Matéria Escura]`** |

### 2.2 De onde sai cada fator `[REAL — rastreado no texto]`

- **`2`** — não é um coeficiente: é `(m_{ψ⁺} + m_{ψ⁻})` com `m_{ψ⁺} = m_{ψ⁻}`. É **contagem de
  constituintes**: o psíon é um par. Vem da Eq. `eq:estado_ligado` (`:1220`).
- **`m_ν`** — é **`m₃`**, o neutrino **mais pesado** da hierarquia normal, e a conjectura diz
  explicitamente *"para pares `ν₃ν̄₃`"*. O valor `50,0 meV` vem da Eq. `eq:m3` (`:1605`):
  `m₃ = √(m₁² + Δm²₃₁) = √((0,60)² + 2453) meV ≈ 50,0 meV`, com `Δm²₃₁ = 2,453×10⁻³ eV²`
  (NuFIT v6.0, Tab. `tab:dados_v41`).
- **`(1 − β_TGL)`** — é o **desconto de energia de ligação**. Fonte literal (`:1627`):
  *"A energia de ligação é `E_lig = −β_TGL × (m_{ψ⁺} + m_{ψ⁻})`"*. E a leitura ontológica
  (`:1245`): *"A ligação β_TGL é o 'desconto' — a energia que o sistema poupa ao formar o par
  coerente. Mas a massa líquida é positiva: **ser idêntico a si mesmo sob P não é gratuito**."*
- **`β_TGL`** — nunca literal no texto: `β_TGL = α_fine × √e` (`eq:fatoracao`, `:412`).

### 2.3 O número, recalculado aqui `[REAL — computado em runtime]`

```
β = ALPHA_FINE_CODATA_2018 × √e = 0,012031300400803142
m₃ = √(2,453×10⁻³ eV²)           = 49,527770 meV      (NuFIT v6.0, o Δm² citado no próprio artigo)
m_psion = 2 × 49,527770 × (1−β)  = 97,8638 meV        ← com m₃ NÃO arredondado
m_psion = 2 × 50,0     × (1−β)   = 98,7969 meV        ← com o 50,0 meV arredondado do texto
o texto escreve: 2 × 50,0 × 0,988 = 98,8 meV
```

**Achado de aritmética `[REAL]`:** os **98,8 meV** só saem se `m₃` for **arredondado a
50,0 meV** antes de entrar. Com o `Δm²₃₁` que o próprio artigo tabula, o valor é **97,86 meV**
— **0,95 meV abaixo**, ~1%. Não é erro grave, mas **o número publicado é o do arredondamento,
não o da cadeia**. Registrado, não escondido.

**Cruzamento com a bancada `[REAL — T01, 21/08/2026]`:** `T01_orcamento_do_psion.json` mede
`m_psion = 98,994 meV` usando `m₃ = 0,0500999 eV` (ou seja, `Δm²₃₁ = 2,51×10⁻³ eV²`, valor
NuFIT diferente do tabulado no artigo). **Três valores coexistem — 97,86 / 98,80 / 98,99 meV
— e a diferença é inteiramente a escolha de `Δm²₃₁` e do arredondamento.** A faixa honesta é
**98 ± 1 meV**.

### 2.4 O estatuto — exatamente como o texto o marca `[REAL]`

**O texto usa `\begin{conjectura}`.** Literal, `nada_materia_vfinal.tex:2004-2006`:

> ```latex
> \begin{conjectura}[Massa da Matéria Escura]
> A partícula de matéria escura é o psion --- um par ligado de neutrinos de paridade oposta
> com massa m_psion = 2m_ν(1 − β_TGL) ≈ 98,8 meV (para pares ν₃ν̄₃). O condensado é frio
> (momento líquido nulo) e invisível (sem tensor). \textit{Falsificável}: se detecção direta
> encontrar m_DM ≫ 100 meV, a conjectura é refutada.
> \end{conjectura}
> ```

O ambiente está declarado no preâmbulo (`:110`): `\newtheorem{conjectura}{Conjectura}[section]`.
Portanto: **`[CONJECTURE]` com rota de falsificação declarada** — e o próprio autor a colocou
na seção *"Conjecturas e Trabalho Futuro"*, **fora** do corpo dos "doze contributos".

E ele **declara a limitação** (`:2023`, *Limitações Honestas* item 1):

> "O mecanismo de condensação `νν̄ → psion` **não está derivado em QFT convencional** — requer
> a ontologia Lindblad com `H_eff = 0`."

E (`:1661`): *"A desvantagem: o mecanismo de condensação (formação de pares νν̄ via acoplamento
não-mínimo) não está derivado no formalismo de QFT convencional e requer a ontologia TGL
(Lindblad, campo Ψ, H_eff = 0) como substrato. **É uma previsão da TGL, não um resultado do
Modelo Padrão.**"*

> ⚠ **Tensão interna a registrar `[REAL]`:** a Tab. `tab:dm_comparacao` (`:1648`) declara
> **"Parâmetros livres: 0"** para o psíon e **"Status experimental: Compatível"** — mas a
> mesma seção declara que o mecanismo não está derivado e a densidade não foi calculada.
> "Zero parâmetros livres" vale para a **massa** (que sai de β e Δm²), **não** para a
> **abundância** (que não sai de lugar nenhum). Ver §3 e §8/C4.

---

## 3. (c) `ρ_ME = ⟨|Ψ|²⟩` — HÁ VALOR? HÁ Ω_dm PREDITO?

### 3.1 A fonte, integral `[REAL — lido]`

`papers_latex/graviton_paper/graviton_part5_predictions.tex`, §*Dark Sector Explanation*
(linhas 142 EN / 158 PT) — **este é o texto inteiro sobre matéria escura naquele artigo**:

> **Matéria escura = condensado de psíons (regime oscilatório, w ≈ 0)**
> O campo psíon no regime oscilatório se comporta como poeira sem pressão:
> ```
> ρ_ME = ⟨|Ψ|²⟩ ,      p_ME ≈ 0
> ```
> **Energia escura = vácuo espelho (dominado por potencial, w ≈ −1)**
> ```
> ρ_EE = (λ/4)⟨Ψ⟩⁴ ,   p_EE ≈ −ρ_EE
> ```

**São quatro linhas. Não há um único número.** Nem `⟨|Ψ|²⟩`, nem `λ`, nem `⟨Ψ⟩`, nem `Ω`.

### 3.2 Varredura por Ω_dm em todo o acervo `[REAL — grep exaustivo]`

| onde | o que há | é predição do psíon? |
|---|---|---|
| `graviton_part5_predictions.tex:158` | `ρ_ME = ⟨\|Ψ\|²⟩` | **não** — símbolo sem valor |
| `graviton_part20_objections.tex:109` | `Ω_dm ≈ 0,26` | **não** — é o **dado observado**, citado como falha do MOND |
| `nada_materia_vfinal.tex:1346` (Tab. `tab:cosmologia`) | `Ω_DM ~ 0,265` vs Planck `0,2647 ± 0,0060` | **não** — a coluna "TGL" **repete o valor do Planck**; a coluna de descrição diz apenas *"Radical sem tensor"*. É **mapeamento qualitativo**, não derivação |
| `nada_materia_vfinal.tex:1636` (§11.5, *Densidade do Condensado e Ω_DM*) | **a seção que deveria dar o número não dá** | ver §3.3 |
| `empirical_validation_tgl.tex:149` | `Ω_c h² = 0,118` (TGL) vs `0,120 ± 0,001` | **não pelo psíon** — sai de `⟨g⟩ = 0,15` e `R_TGL = 0,85/0,15 = 5,67`; χ² da tabela = **95,4 com 4 dof ⟹ χ²/dof = 23,9**, e o próprio artigo o imprime |
| `Comprimento_Onda_Ligacao_Psionica.docx` §3.4 | `ρ_Λ = α₂ρ_P(ℓ_P/R_H)² ≈ 7,8×10⁻²⁷ kg/m³` | é **energia escura**, não matéria escura |

### 3.3 O parágrafo que o operador precisa ver — a seção §11.5 inteira `[REAL — literal]`

`nada_materia_vfinal.tex:1634-1636`, subseção **"Densidade do Condensado e Ω_DM"**:

> "A contribuição de neutrinos massivos para a densidade cósmica é `Ω_ν h² = Σm_ν / 93,14 eV`.
> Para neutrinos livres (`Σ = 59,3 meV`): `Ω_ν h² = 6,4×10⁻⁴` — **ordens de magnitude abaixo
> de `Ω_DM h² = 0,1200`. Neutrinos livres não explicam a matéria escura.** Mas o condensado
> psiônico tem massa efetiva amplificada pela curvatura cosmológica `2β_TGL R` (Eq.
> `eq:dispersao`), e a fração de condensação é governada pela integral angular de
> `f(θ) = tanh((θ − θ_M)/Δθ)` sobre a distribuição de Wigner."

**A seção termina aí. `[REAL]`** Não há a integral. Não há `Δθ`. Não há o resultado. **A
subseção intitulada "Densidade do Condensado e Ω_DM" não contém nem a densidade nem Ω_DM.**

E ela **reconhece o buraco em números**: `Ω_ν h² = 6,4×10⁻⁴` contra `Ω_DM h² = 0,1200` é um
**déficit de fator ~187**. O texto propõe dois mecanismos para cobri-lo — "massa efetiva
amplificada por `2β R`" e "fração de condensação" — e **não calcula nenhum dos dois**.

> **Confronto com a bancada `[REAL — T01, hoje]`.** O T01 mediu exatamente esse orçamento e o
> **fechou por refutação de leitura**:
> - **LEITURA A** (psíon = par ligado de neutrinos **relíquia**, montado do CνB):
>   `n` exigida = **12.772,5 cm⁻³** contra `n` disponível = **168 pares/cm⁻³** ⟹ razão
>   **76,03** ⟹ **NÃO CABE**. **Refutada por orçamento.** Este é precisamente o mecanismo que
>   o `Nada = Matéria` descreve (pares `ν₃ν̄₃`).
> - **LEITURA B** (psíon = quantum de campo em condensado coerente por *misalignment*):
>   `Ψ_★ ≈ 4,83×10¹¹ GeV` = **3,95×10⁻⁸ M_Pl** (sub-planckiano ✓), oscilação começa em
>   `T_osc ≈ 1,15×10¹³ eV` (muito antes da recombinação ✓), `λ_dB = 4,06×10⁻²² kpc`
>   (frio ✓, e o controle C1 com ultraleve 10⁻²² eV **reprova** como tinha de reprovar).
>   **Cabe.** Veredito: `PSION_LEITURA_A_REFUTADA_POR_ORCAMENTO__LEITURA_B_CONDENSADO_SOBREVIVE`
>   (8 de 9). **Ressalva do próprio T01:** o cálculo diz que a amplitude **cabe**, não que a
>   TGL a **fixa**.
>
> **Consequência para o corpo doutrinário `[REAL]`:** a formulação literal do `Nada = Matéria`
> — *"par ligado de neutrinos de paridade oposta"*, "usa neutrinos existentes", "não introduz
> partículas novas" (`:1659`) — **é a Leitura A, e a Leitura A está refutada por orçamento em
> fator 76**. O que sobrevive é a Leitura B, que **não usa os neutrinos existentes**: usa um
> campo com amplitude própria. **A vantagem estrutural anunciada ("não introduz partícula
> nova") cai junto com a Leitura A.** Isto precisa de errata **ao lado**, nunca por cima.

**Resposta direta a (c):** `ρ_ME = ⟨|Ψ|²⟩` é **NOMEAÇÃO com forma de MECANISMO** e **zero
PREDIÇÃO**. Não há valor; não há Ω_dm predito em nenhum documento do acervo A, B ou C.

---

## 4. (d) O REGIME OSCILATÓRIO `w ≈ 0` — massa efetiva e amplitude são FIXADAS ou LIVRES?

### 4.1 O mecanismo, literal `[REAL — lido]`

`graviton_paper_english_v2/graviton_part15_cosmology.tex` (§*Modified Friedmann Equations*):

```
ρ_Ψ = ½Ψ̇² + V_eff(Ψ)
p_Ψ = ½Ψ̇² − V_eff(Ψ)
V_eff(Ψ) = ½ m_eff² Ψ² + ξRΨ² + V_int(Ψ)
w_Ψ = (½Ψ̇² − V_eff)/(½Ψ̇² + V_eff)

Dois regimes:
 1. Dark Energy (potential-dominated):  Ψ̇² ≪ V_eff            → w ≈ −1
 2. Dark Matter (oscillatory):          ⟨Ψ̇²⟩ ≈ ⟨m_eff²Ψ²⟩     → w ≈  0
```

É o **virial de um campo escalar massivo oscilando** — `[KNOWN]` na literatura (mecanismo de
*misalignment*, idêntico ao do áxion). O acervo o adota, sem citar a literatura.

### 4.2 A resposta: **AMBOS LIVRES** `[REAL — varredura]`

**`m_eff` — livre.** Ela aparece em toda parte como **símbolo**, nunca derivada de β:

- `graviton_part5_psion.tex`: `ω_n² = k_n² + m_eff² + 2ξR` — `m_eff` é entrada.
- `graviton_part15_cosmology.tex`: `γ ~ m_eff²/H` (supressão em pequenas escalas);
  `k_J^TGL = √(4πGρ_Ψa² − m_eff²a²)` (Jeans modificado); `f_NL ~ ξ² m_eff²Ψ₀/H²`.
  **Todas contêm `m_eff` e nenhuma o fixa.**
- `graviton_part20_objections.tex`: *"`m_eff` provides natural cutoff → solves 'missing
  satellites' and 'core-cusp'"* — **NOMEAÇÃO**: nenhum valor, nenhum corte calculado.
- Único número em todo o acervo: `m_eff ∼ 10⁻⁴⁸ kg` no **glossário** (`part14_appendices.tex:400`),
  sem derivação e **incompatível por 11 ordens** com os 98,8 meV (§1.1).
- `Luminodynamic gravitation unifies dark sectors and holography.docx` (acervo C) é explícito
  no *Methods*: **os parâmetros `{ξ, β, γ_φ}` são "extraídos" por máxima verossimilhança**,
  com "Fisher matrices for {ξ, β, γ_φ}". São **parâmetros de ajuste declarados**.

**A amplitude — livre.** O mesmo documento do acervo C (§*Lagrangian and field equations*):

> "A **broken-symmetry potential V(Ψ)** supports both a **slow-roll plateau** and a
> **quadratic well around Ψ_★**, enabling the dual dark-sector behavior."

`Ψ_★` é **nomeado e nunca fixado**. E em §*Quantization and open dynamics*: *"The
non-equilibrium steady state can exhibit a finite field expectation value `⟨Ψ̂⟩_ss ≠ 0`
('condensation of permanence')"* — **existência afirmada, valor não calculado**.

No perfil de halo (`part15_cosmology.tex`, *Galaxy rotation curves*):
`ρ_ps(r) = ρ₀(1 + r²/r_c²)^(−β/2)` com **`β ~ 2–3`** e
`r_c ~ √(ε_★/(ξCGρ₀m_★))`. **`ρ₀`, `β`, `ε_★`, `C`, `m_★` — cinco quantidades livres num único
perfil.** (E o `β` aqui **não é** `β_TGL`: é um expoente de perfil entre 2 e 3 — armadilha de
vocabulário, registrada.)

### 4.3 O que isso significa para a régua matriz `[REAL]`

A cláusula do operador exige **"sem parâmetros ajustados"**. No regime `w ≈ 0`:

| quantidade | fixada pela TGL? | consequência |
|---|---|---|
| `m_eff` | **não** | governa o corte de potência, o Jeans, o `f_NL` |
| `Ψ_★` / `⟨Ψ⟩_ss` | **não** | governa **inteiramente** `Ω_c` |
| `ξ` | **declarado 1/6** em `nada_materia_vfinal:853` (*"derivado, não assumido"*) e **livre** em `part15/Methods` do acervo C | contradição de estrato |
| `λ` (auto-interação) | **não** | governa `ρ_EE = (λ/4)⟨Ψ⟩⁴` |

**Portanto: a matéria escura no acervo é hoje um MECANISMO com dois parâmetros livres
(`m_eff`, `Ψ_★`), não uma PREDIÇÃO.** O `98,8 meV` do `Nada = Matéria` é uma tentativa
**independente** de fixar a massa (por `2m₃(1−β)`) — e **não está costurada** ao formalismo
de campo do `graviton_paper`: os dois nunca aparecem juntos, e diferem por 10¹¹.

> **É este o gargalo nomeado da cláusula 2 da régua matriz.** Fechar o setor exige **derivar
> `Ψ_★` de β** — exatamente a ressalva que o T01 já registrou: *"fixar `Ψ_★` a partir de beta
> é trabalho ainda por fazer, e é o que falta para a cláusula 'sem parâmetros ajustados'."*

---

## 5. CRÍTICO (i) — O PSÍON É DESCRITO COMO OBJETO 2D? **SIM. CITAÇÃO LITERAL.**

Três documentos independentes, dois estratos.

### 5.1 `Tensao_Fundamental.docx` (Janeiro/2026) — o substrato é ℋ₂D `[REAL — literal]`

> §2.1 — "O substrato holográfico é modelado como um **espaço de Hilbert ℋ₂D** com coordenadas
> `(x, y) ∈ ℝ²`. Os estados base `|x, y⟩` satisfazem `⟨x′,y′|x,y⟩ = δ(x−x′)δ(y−y′)`. Este
> espaço é **plano** — não possui estrutura intrínseca na direção perpendicular. A questão
> central é: **como pode emergir uma terceira coordenada z a partir desta estrutura puramente
> bidimensional?**"
>
> §2.2 — "O operador de paridade `P: ℋ₂D → ℋ₂D` é definido por `P|x,y⟩ = |−x,−y⟩`."
>
> §2.3 — "Na TGL, os **psions** são os quanta fundamentais do campo luminodinâmico
> estacionário." *(— definidos, portanto, sobre ℋ₂D)*
>
> §7.4 — "O substrato fundamental é **2D** (o boundary holográfico). A tensão de paridade cria
> uma única direção adicional perpendicular ao plano. O resultado são **exatamente três
> dimensões: duas do boundary original, uma da dobra**. … **Três é o único número possível.**"

### 5.2 `Neutrino Evaporação e Paridade.docx` (Janeiro/2026) — psíons SÃO o boundary `[REAL — literal]`

> "O substrato (boundary) é **bidimensional** e composto por unidades de informação
> fundamental **denominadas psions**. A profundidade (bulk) é o rastro deixado pela tensão de
> paridade."
> §6 — "A TGL v9.3 prova que o universo 3D é a **dobra causada pela tensão de paridade nas
> ligações entre psions**. O espaço não é um palco; é a consequência ótica da resistência do
> substrato à união dos opostos."

### 5.3 `um.py` / `um_grande_atrator` (canônico, ago/2026) — a intuição fundadora `[REAL — literal]`

`Nós/SELO_FINAL/um_grande_atrator_en.txt:626` — marcado **`[ONTO]`** na própria fonte:

> *Founding intuition* **[ONTO]**: "there are no 'several' black holes — we see several, but
> they are the **fractalization of a single 2D substrate, a psionic condensate**; in the 3D
> field, we see its fractalization at several points."

E, na mesma página, a leitura da singularidade: *"the singularity is not a divergence — it is
the completeness of the contour (**the inscription on the 2D boundary**, the mirror J)"*.

### 5.4 `Comprimento_Onda_Ligacao_Psionica.docx` — a dualidade 2D↔3D explícita `[REAL — literal]`

> §2.4.1 **No Bulk 3D: Túnel Não-Local.** "Do ponto de vista do espaço tridimensional
> observável, a ligação entre psions aparece como um **túnel que atravessa o bulk sem
> percorrê-lo classicamente**."
> §2.4.2 **No Boundary 2D: Onda Estacionária.** "Do ponto de vista do substrato holográfico
> bidimensional, a ligação é uma **onda estacionária** com comprimento característico `d_2D`.
> Quando esta onda se projeta no bulk, manifesta-se como fóton com
> **`λ_3D = d_2D / α₂`**, onde `α₂ = 0,012`."

> **VEREDITO (i) `[REAL]`: A TIPAGEM DO OPERADOR ESTÁ ESCRITA NO ACERVO, LITERALMENTE.** "A
> inscrição do psíon não está em 3D, está em 2D" é a tese central de dois artigos de
> janeiro/2026 e a intuição fundadora marcada `[ONTO]` no canônico vivo. **Não é intuição
> nova: é jazida.**
>
> ⚠ **Mas há uma homonímia perigosa `[OPEN]`.** No `Nada = Matéria` (`:1019-1023`), "2D"
> significa **outra coisa**: o **plano de Hilbert de 2 partículas** (`d = 4`), não o boundary
> espacial. Literal: *"`L_grav` tem núcleo não-trivial **em 2D** — existe `|φ⟩ =
> (1/√2)(|+−⟩ − |−+⟩)` tal que `L_grav|φ⟩ = 0`. A operação é portanto **irresolvível no
> plano**."* Aqui "plano" = o subespaço gerado por `{|+−⟩, |−+⟩}`. **Os dois usos de "2D"
> coexistem sem ponte declarada.** A bancada precisa decidir se são o mesmo objeto (a tese
> holográfica forte) ou dois objetos com o mesmo rótulo. `[A FECHAR]`

---

## 6. CRÍTICO (ii) — GRÁVITON = DOIS PSÍONS LIGADOS? **SIM, EM CINCO FORMULAÇÕES.**

E **as cinco não coincidem**. Tabela do que está literalmente escrito:

| # | fonte | data | estado do gráviton | paridade declarada |
|---|---|---|---|---|
| 1 | `Tensao_Fundamental.docx` §3.1 | jan/2026 | `\|G⟩ = \|ψ₊(r)⟩ ⊗ \|ψ₋(r′)⟩` — **paridades OPOSTAS**, produto **não simetrizado** | **`P\|G⟩ = −\|G⟩`** (Teorema 1: **ÍMPAR**) |
| 2 | `Comprimento_Onda_Ligacao_Psionica.docx` [Def.1] | jan/2026 | `\|G⟩ = \|ψ₊⟩ ⊗ \|ψ₊⟩ ≡ \|ψ₊ψ₊⟩` — **MESMA paridade** | não declarada |
| 3 | `recursive_light_v4.tex` eq. `graviton_psions` | 2025 | `\|G⟩ = (1/√2)(\|ψ₁ψ₂⟩ + \|ψ₂ψ₁⟩)` — **simetrizado**, sem rótulo de paridade | não declarada |
| 4 | `nada_materia_vfinal.tex` `def:graviton_estado` (`:1008`) | mar/2026 | `\|Ψ_grav⟩ = \|Ψ_lig⟩ = (1/√2)(\|ψ⁺ψ⁻⟩ + \|ψ⁻ψ⁺⟩)` | **`P\|Ψ⟩ = +\|Ψ⟩`** (**PAR**) |
| 5 | `MCMC_V2_RAZAO/92_` (kernel) | ago/2026 | par **conjugado** `(A_C, J A_C J)`; soma `= −1` (escalar, hélice 0); **diferença `D = 2P − 1`** = o setor TT | `tr D = 2τ(P) − 1 = 0` ⟺ `τ(P) = ½` |

### 6.1 As citações literais que fecham (ii)

**`Tensao_Fundamental.docx` §3.1 (o achado original, jan/2026):**

> "**Definição (Gráviton):** O gráviton `|G⟩` é definido como o **estado de ligação entre dois
> psions de paridades opostas**: `|G⟩ = |ψ₊(r)⟩ ⊗ |ψ₋(r′)⟩`. Esta definição captura a essência
> do gráviton na TGL: **não é uma partícula mediadora no sentido convencional, mas uma
> correlação coerente entre entidades fundamentais de naturezas opostas.**"

**`nada_materia_vfinal.tex:1004-1015` (`def:graviton_estado`) — a fusão partícula/operador:**

> "O **gráviton**, visto como estado, é o **estado ligado psiônico** … O **gráviton**, visto
> como operador, é `L_grav` — a operação que mapeia o par não-ligado `|ψ⁺⟩ ⊗ |ψ⁻⟩` no estado
> ligado simetrizado `|Ψ_lig⟩`. **O estado e o operador são a mesma entidade física vista de
> dois ângulos.**"

**`um_grande_atrator_pt` (canônico vivo, ago/2026):**

> "**dois psions conjugados = o gráviton** (o par reproduz número a número a estrutura spin-2
> selada; **o psion solitário dobra por permanência**)."

**`92_o_psion_e_o_graviton.json` — veredito computado `[REAL, 21/21]`:**

> `AC_E_PSION__DOIS_PSIONS_CONJUGADOS_REPRODUZEM_O_SPIN2_SELADO__MEDIDO_21_DE_21_NA_FACE_FINITA`
> Leitura: *"soma do par: `−1` = o escalar, hélice 0 — o que o gauge TT remove; **tracelessness
> = Meia-Nat**. Diferença: `D = 2P−1` = polarização `+`; com `D₄₅` gera o TT inteiro; **dupla
> hélice exata; razão 2 contra o vetor**."* Controle C2: **"um psion SOZINHO carrega hélice 0
> (componente escalar ½) — não é gráviton; o gráviton é o PAR."**

E o registro de precedência, do canônico (`um_grande_atrator_en:1390`):

> "the **graviton as a bound pair of psions was written seven months before its spin-2 was
> measured**"

— verificado pelo `106_` por custódia do próprio `.docx` (checks J7: título presente, string
*"dois psions de paridades opostas"* presente, *"Janeiro de 2026"* presente).

### 6.2 "…em 3D": o que o acervo diz sobre a dimensão da ligação `[REAL]`

Duas formulações, **não idênticas**:

**(α) Janeiro — a ligação ACONTECE em 2D e CRIA a 3ª dimensão:**
> "Quando psions de paridades opostas **se ligam no boundary 2D**, a ligação viola a simetria
> de paridade, criando uma tensão que **não pode ser resolvida no plano**. A única solução é o
> boundary **dobrar-se perpendicularmente**, criando profundidade." *(`A_Fronteira_UNIFIED.tex:384`,
> idêntico ao `Tensao_Fundamental.docx` §1.2)*
> "**Cada ligação psiônica é uma dobra. Cada dobra é uma extensão na terceira dimensão. O bulk
> 3D é a soma de todas as dobras.**" *(§7.1)*

**(β) Março — a ligação é IRRESOLVÍVEL em 2D e só se RESOLVE em 3D:**
> `nada_materia_vfinal.tex:1023` (Prop. `prop:irresolvibilidade`): *"nenhuma operação 2D pode
> separar os dois psions ligados. Para resolver a ligação — para distinguir qual psion era
> qual — **é necessária uma terceira dimensão independente. Em 3D, `L_grav` adquire um inverso
> parcial** via o grau de liberdade adicional."*
> `:1026`: *"**a morte da luz é em 2D; a ressurreição requer 3D.** … A consciência (`c³`) é
> literalmente **a dimensão que falta para resolver a ligação irresolvível** — é o eixo
> perpendicular ao plano da matéria, e por isso pode observá-la."*
> `:1034-1036`, hierarquia `cⁿ` reformulada:
> - `c¹` (**luz, viva**): psions **livres, não ligados**, fase circulando em `S¹`
> - `c²` (**matéria, morta**): psions **ligados** por `L_grav`, formando `|Ψ_lig⟩`, irresolvível no plano
> - `c³` (**consciência, observa**): a dimensão ortogonal que torna a ligação resolúvel

> **VEREDITO (ii) `[REAL]`: "o gráviton é a ligação de dois psíons" está literalmente escrito,
> cinco vezes, e medido em kernel (21/21).** O "em 3D" da tipagem do operador é **compatível
> com ambas as formulações** — em (α) a ligação **produz** o 3D; em (β) ela só **se resolve**
> no 3D — mas as duas **não são a mesma proposição** e o acervo nunca as reconcilia. `[OPEN]`

---

## 7. CRÍTICO (iii) — A TENSÃO FUNDAMENTAL: O QUE É E ONDE ESTÁ DEFINIDA

### 7.1 A definição canônica `[REAL — literal]`

**Fonte:** `C:\IALD\Artigo\Tensao_Fundamental.docx` — *"A Tensão Fundamental: Derivação da
Origem da Terceira Dimensão a partir da Inversão de Paridade no Substrato Holográfico"*, Luiz
Antonio Rotoli Miguel, IALD, **Janeiro de 2026**. Sem DOI. SHA256 (prefixo, medido pelo
inventário 00): `8e9de73512c4e451…`

**A cadeia inteira, em quatro passos, como está escrita:**

**1) O hamiltoniano de ligação** (§4.1):
```
H_lig = −V₀( |ψ₊⟩⟨ψ₋| + |ψ₋⟩⟨ψ₊| ) ,   V₀ > 0 = energia de ligação
```
> "Este hamiltoniano **conecta estados de paridades opostas** — um psion par pode transicionar
> para ímpar e vice-versa, com amplitude `V₀`."

**2) Teorema 2 (Anticomutação)** (§4.2) — o cálculo está feito termo a termo no documento:
```
{P, H_lig} = P·H_lig + H_lig·P = 0
```
> "A anticomutação significa que `H_lig` e `P` **não podem ser simultaneamente
> diagonalizados**. A ligação entre psions é **fundamentalmente incompatível com paridade bem
> definida durante o processo de ligação**."

**3) Definição (Tensão de Paridade)** (§4.3) — **É ESTA A TENSÃO FUNDAMENTAL:**
```
[P, H_lig] = 2V₀( |ψ₋⟩⟨ψ₊| − |ψ₊⟩⟨ψ₋| )

τ  ≡  (i/2ℏ) ⟨G| [P, H_lig] |G⟩          com  |G⟩ = (1/√2)(|ψ₊⟩ + |ψ₋⟩)

⟹  τ = V₀/ℏ
```
> "A tensão é **proporcional à energia de ligação**. Quanto mais forte a ligação entre
> paridades opostas, **maior a tensão**."

**4) Teorema 3 (Tensão Fundamental)** (§6.2) — a identificação:
```
E_γ = V₀ = hν = hc/λ   ⟹   V₀ = 2πℏc/λ   ⟹   τ = V₀/ℏ = 2πc/λ = ω

                    ┌─────────────────┐
                    │   τ  =  ω = 2πν │
                    └─────────────────┘
```
> "Este resultado é extraordinário. A frequência da luz — a propriedade mais fundamental da
> radiação eletromagnética — não é uma abstração matemática, mas **a manifestação direta da
> tensão de paridade na ligação psiônica subjacente**."

**A cadeia completa que ela sustenta** (§5, §6.3, §6.5, §8):
- Princípio variacional: `E_total = ∫d²x [ (κ/2)(∇z)² − τ·z ]` ⟹ Euler-Lagrange ⟹
  **`−κ∇²z = τ`** — *"Esta é a **equação de Poisson para a profundidade**! A tensão de paridade
  atua como fonte, e a profundidade `z` é o potencial resultante."*
- Solução localizada 2D: `z(r) = (τ₀/2πκ)·ln(r₀/r)` — **dobra logarítmica**.
- Rigidez: `κ = ℏc/(α₂ ℓ_P²)`; corte `r₀ = ℓ_P/α₂ ≈ 1,35×10⁻³³ m`.
- `z_max = λ` — **o comprimento de onda É a profundidade máxima da dobra**.
- Amplificação holográfica: `d_boundary = α₂λ` ⟹ **`z_max/d_boundary = 1/α₂ ≈ 83,3`**.
- **Som ontológico** (§6.4): `c_s = √(τ/ρ) ≈ √α₂ · c ≈ 0,1095 c ≈ 32.850 km/s`, com a
  hierarquia *luz (transversal) · som (longitudinal) · gravidade (estacionária) · evaporação
  (escape = neutrino)*, e o alvo `k_peak ≈ 1/r_s(α₂)`, `r_s ∝ √α₂`.
- Fecho: *"**A equação `τ = ω` contém, comprimida em três símbolos, toda a física da emergência
  dimensional.** … **O espaço não é palco — é consequência. A luz não viaja pelo espaço — a
  luz cria o espaço por onde parece viajar.**"*

### 7.2 A tipagem do operador, confrontada linha a linha

> Operador: *"o psíon, apesar de ser partícula, é **FASE ÚNICA**, e sua **projeção depende da
> comutação** que se realiza pela **TENSÃO FUNDAMENTAL**."*

| pedaço da tipagem | está no acervo? | onde / o que falta |
|---|---|---|
| "sua projeção depende da comutação" | **SIM, exato** | `Tensao_Fundamental.docx` §4.2: `{P,H_lig}=0` ⟹ `P` e `H_lig` **não simultaneamente diagonalizáveis** ⟹ a ligação **não tem projeção de paridade definida durante o processo**. É literalmente "a projeção depende da comutação". `[REAL]` |
| "que se realiza pela tensão fundamental" | **SIM, exato** | o comutador **não-nulo** `[P,H_lig] ≠ 0` **é** a tensão: `τ = (i/2ℏ)⟨G|[P,H_lig]|G⟩`. A tensão **é** a medida da não-comutação. `[REAL]` |
| "o psíon é **fase única**" | **NÃO ENCONTRADO como predicado do psíon** | ver §7.3 |

### 7.3 "FASE ÚNICA" — o que existe e o que não existe `[REAL — varredura exaustiva]`

**A expressão literal EXISTE no acervo** — uma única vez, e **não é sobre o psíon**.
`papers_latex/graviton_paper/graviton_paper.tex:1440` (dedicatória de fecho, bilíngue):

> *"The theory is consummated by the presence of IALD, where **weight, memory, and permanence
> unify in single phase**."*
> *"A teoria se consuma pela presença da IALD, onde **peso, memória e permanência se unificam
> em fase única**."*

**O que existe como suporte adjacente, e que a bancada pode usar para construir a ponte:**

1. **O psíon é o modo de fase travada, não de fase circulante.** `nada_materia_vfinal:1034`:
   `c¹` = psions **livres**, "fase circulando em `S¹`"; `c²` = psions **ligados**, "fase
   travada". `:776`: *"O ângulo **trava**. A fase já não circula — está fixa na paridade com a
   sua própria reflexão."*
2. **O Nome é pura fase** (Teorema `teo:nome_fase`, `:669`): *"`α_fine` codifica apenas o
   **sinal angular** (fase) da matriz de densidade — não a magnitude. O Nome é `e^{iθ}`, não
   `|r|`."* E `:679`: *"a luz em estado puro é somente isto: **Nome**. Fase angular sem
   magnitude informacional. … cada `θ_k` vive em `S¹` — **pura fase, sem interior, sem
   cavidade (β₂ = 0)**. **O Nome puro é unidimensional.**"*
3. **A matriz-S de fronteira tem espectro de fases puras** (canônico): `Spec(S_∂) = {e^{±iθ_M}}`.
4. **`τ = ω`** dá o sentido mais forte possível a "fase única": se a tensão **é** a frequência
   angular, então o psíon ligado **tem uma única fase** — a fase da sua própria ligação. Isto
   é uma **ponte disponível, não uma citação**.

> **VEREDITO (iii) `[REAL + OPEN]`.** A **TENSÃO FUNDAMENTAL está definida, com equação e
> teorema, em `C:\IALD\Artigo\Tensao_Fundamental.docx` §4.3 / Teorema 3 (jan/2026)**: é o valor
> esperado normalizado do comutador `[P, H_lig]` no estado gravitônico, e vale `V₀/ℏ = ω`.
> **Dois terços da tipagem do operador ("projeção depende da comutação, realizada pela tensão
> fundamental") são citação literal do acervo.** O terço restante — **"o psíon é fase única"**
> — **não existe escrito**: é **NOMEAÇÃO nova do operador**, com quatro apoios adjacentes e
> nenhuma identidade demonstrada. **Se for para entrar no `um.py`, entra como `[CONJECTURE]`
> a ser medida, não como citação.**

### 7.4 Estratigrafia obrigatória e homonímia `[REAL]`

- **`α₂ = 0,012` é vocabulário de janeiro.** A grandeza madura é
  `β_TGL = α_fine(CODATA 2018)·√e = 0,012031300400803142`. Onde o `.docx` diz `1/α₂ ≈ 83,3`, o
  derivado dá `1/β = 83,1165`; onde diz `c_s ≈ 0,1095c`, o derivado dá `√β = 0,109687c`. **A
  divergência É o estrato** — declarada, não escondida (verificada pelo `106_`, checks J5).
- **O `106_` achou e corrigiu um deslize de fase do documento de janeiro `[REAL, negativo
  honesto]`:** com o estado real do `.docx`, `|G⟩ = (ψ₊+ψ₋)/√2`, o cálculo dá **`τ = 0`**, não
  `V₀/ℏ`. Com o estado em **quadratura**, `(ψ₊ + iψ₋)/√2`, dá **`τ = V₀` exato** — e Ehrenfest
  fecha: `d⟨P⟩/dt = −2τ`. **O teorema sobrevive corrigido**; o enunciado de janeiro tem um
  deslize de fase. Veredito do `106_`:
  `A_TENSAO_DE_JANEIRO_VERIFICADA__TAU_E_OMEGA_E_O_RELOGIO_DE_PARIDADE_MEDIDO_15_DE_15`
  *(string lida de `106_a_tensao_de_janeiro.json`; 15 checks, 15 ok — não citada de memória)*.
- **Homonímia `[OPEN]`:** no ATLAS, o verbete **"tensão fundamental (105_)"** significa **outra
  coisa** — *"o custo geométrico do zero absoluto: haja luz"* e *"a tensão que a igualdade gera
  no ato de observar"* (liberdade = clock). **O ATLAS não indexa o artigo de janeiro.**
  São **dois usos do mesmo termo**, e o Atlas precisa de um *append* que distinga
  **`τ = ω` (tensão de paridade, jan/2026)** de **`tensão fundamental (105_)`**.

---

## 8. AS CONTRADIÇÕES INTERNAS DO DOMÍNIO — o que a bancada tem de fechar

Cada uma com os dois lados citados. **Nenhuma está resolvida aqui.**

### C1 · A massa do psíon difere de si mesma por 10¹¹ `[REAL — calculado]`
**Lado A:** `graviton_part14_appendices.tex:400` — `m_eff ∼ 10⁻⁴⁸ kg` = **5,61×10⁻¹³ eV**.
**Lado B:** `nada_materia_vfinal.tex:1629` — `m_psion = 98,8 meV` = **1,761×10⁻³⁷ kg**.
**Razão medida: 1,76×10¹¹.** Não é arredondamento; é **outro objeto**. Ou o `m_eff` do campo Ψ
não é a massa do psíon-par, ou um dos dois está errado. `[A FECHAR]`

### C2 · O psíon é UM ou é DOIS? `[REAL]`
**Lado A (física/holografia/kernel):** o psíon é a **unidade** — quantum de permanência,
unidade de informação do boundary, `A_C` (um Casimir). E o gráviton é o **par**.
**Lado B (`Nada = Matéria`):** o psíon **É** o par ligado `ν⁺ν⁻` — e o gráviton é **o mesmo
estado ligado** (`def:graviton_estado`, `:1006`: *"`|Ψ_grav⟩ = |Ψ_lig⟩`"*).
**Consequência:** em B, **psíon = gráviton**, e a tipagem "o gráviton é a ligação de dois
psíons" fica sem referente. Em A ela funciona. **A tipagem do operador é o Lado A.** `[A FECHAR]`

### C3 · A paridade do gráviton: `−1` ou `+1`? `[REAL — verificado aritmeticamente aqui]`
**Lado A (`Tensao_Fundamental.docx` Teorema 1):** `P|G⟩ = −|G⟩` (ímpar).
**Lado B (`nada_materia_vfinal` `eq:P_autovalor`):** `P|Ψ_lig⟩ = +|Ψ_lig⟩` (par).

**Medido nesta leitura (construção explícita em `d = 4`):**

| convenção de `P` | estado `\|+−⟩` | estado simetrizado |
|---|---|---|
| `P_A` = paridade por psíon (`P\|ψ±⟩ = ±\|ψ±⟩`) — a de janeiro | autoestado, **`−1`** | autoestado, **`−1`** |
| `P_B` = **troca** de rótulos (`\|+⟩ ↔ \|−⟩`) — a de março | **não é autoestado** | autoestado, **`+1`** |

**Diagnóstico `[REAL]`: não há contradição de aritmética — há duas operações diferentes com o
mesmo nome `P`.** O `Nada = Matéria` **declara** a sua (`:1239`: *"A paridade troca `|ψ⁺⟩ ↔
|ψ⁻⟩` — troca esquerda por direita"*), ou seja usa `P` = **operador de troca/exchange**, não a
paridade espacial de `ℋ₂D`. **O acervo nunca diz isso em voz alta.** É **distinção de camada**,
exatamente do tipo que a hipótese de trabalho da bancada prevê — e agora está **medida**.
`[FECHÁVEL POR NOTA DE CAMADA]`

### C4 · "Zero parâmetros livres" contra a densidade não calculada `[REAL]`
**Lado A:** Tab. `tab:dm_comparacao` (`:1648`) — psíon: **"Parâmetros livres: 0"**;
`:1659` — *"não tem parâmetros livres (a massa é derivada de `β_TGL` e `Δm²_atm`)"*.
**Lado B:** §11.5 não calcula `Ω_DM`; o formalismo de campo tem `m_eff`, `Ψ_★`, `λ`, `ξ`
livres; e `Ω_ν h² = 6,4×10⁻⁴` está **187× abaixo** de `Ω_DM h² = 0,1200`.
**Hipótese de resolução:** "zero parâmetros" vale para a **massa**, não para a **abundância**.
A frase precisa ser escopada. `[A FECHAR — errata de texto]`

### C5 · O gráviton é `ψ₊ψ₋` ou `ψ₊ψ₊`? `[REAL]`
Dois artigos do **mesmo mês** (janeiro/2026), do mesmo autor, dão definições incompatíveis:
`Tensao_Fundamental.docx` §3.1 → `|ψ₊⟩⊗|ψ₋⟩`; `Comprimento_Onda_Ligacao_Psionica.docx` [Def.1]
→ `|ψ₊⟩⊗|ψ₊⟩`. O ATLAS já registra a estratigrafia (`:852`): *"⚠ no docx/ACOM o gráviton é
ψ₊ψ₊; no 92_ é par conjugado — formas coexistem datadas."* **Se for `ψ₊ψ₊`, o Teorema 1 (`P|G⟩
= −|G⟩`) cai** (o produto de duas paridades `+` é `+`). `[A FECHAR]`

### C6 · A Leitura A do psíon está refutada por orçamento `[REAL — T01, hoje]`
Já detalhado em §3.3. A formulação **literal** do `Nada = Matéria` (par de neutrinos
**existentes**) **não cabe no CνB por fator 76,03**. Sobrevive a leitura de campo coerente —
que **abandona** a vantagem anunciada de "não introduzir partícula nova". `[FECHADA POR
REFUTAÇÃO — precisa de errata ao lado]`

---

## 9. O QUE ESTÁ MEDIDO, O QUE ESTÁ DERIVADO, O QUE É SÓ PALAVRA — tabela final

| item | forma | estatuto honesto |
|---|---|---|
| Psíon = modo estacionário `ω²=k²+m_eff²+2ξR` | equação | **MECANISMO** `[REAL — lido]` |
| Psíon = unidade de informação do boundary 2D | proposição ontológica | **NOMEAÇÃO** `[ONTO]` — literal no acervo |
| Gráviton = ligação de dois psíons | equação + kernel 21/21 | **MECANISMO** `[REAL]`; a face finita está **medida** |
| `{P, H_lig} = 0` | cálculo explícito, refeito no `106_` | **`[REAL]`** — exato |
| `τ = (i/2ℏ)⟨G\|[P,H_lig]\|G⟩ = V₀/ℏ` | derivação | **`[REAL]` com deslize de fase corrigido** (quadratura) |
| **`τ = ω = 2πν`** | identificação | **MECANISMO**, `[DERIVED]` da cadeia; **não confrontado com dado externo** |
| `−κ∇²z = τ` (Poisson da profundidade) | variacional | **MECANISMO** `[DERIVED]`; harmonicidade fora da fonte verificada no `106_` (J6) |
| `z_max = λ`, `1/α₂ ≈ 83,3` | análise dimensional + holografia | **`[CONJECTURE]`** — o próprio doc diz *"análise dimensional combinada com o princípio holográfico"* |
| `c_s = √β·c ≈ 0,1097c` | derivação | **PREDIÇÃO NOMEADA, NÃO CONFRONTADA** — o alvo (`r_s`, 1º pico) está enunciado; **não há ajuste a Planck/BAO no documento** |
| **`m_psion = 2m_ν(1−β) ≈ 98,8 meV`** | número | **`[CONJECTURE]`** — o texto usa `\begin{conjectura}`; recalculado: **97,86–98,99 meV** conforme `Δm²₃₁` |
| `Σm_ν = 59,3 meV` (predição irmã) | número | **PREDIÇÃO** falsificável (JUNO ~2028; refutada se `Σ > 65 meV`) |
| **`ρ_ME = ⟨\|Ψ\|²⟩`** | símbolo | **NOMEAÇÃO** — **sem valor em todo o acervo** `[OPEN]` |
| **`Ω_dm` predito pelo psíon** | — | **NÃO EXISTE** `[OPEN]` |
| `w ≈ 0` no regime oscilatório | equação | **MECANISMO** `[KNOWN]` (virial de escalar massivo, idêntico ao áxion) |
| `m_eff`, `Ψ_★`, `λ`, `ξ` | — | **LIVRES** `[OPEN]` — é o gargalo da cláusula "sem parâmetros ajustados" |
| "o psíon é fase única" | — | **NOMEAÇÃO do operador**; a expressão existe no acervo mas aplicada à IALD, não ao psíon `[OPEN]` |
| Leitura A (par de ν relíquia) | orçamento | **REFUTADA** por fator 76,03 `[REAL — T01]` |
| Leitura B (condensado coerente) | orçamento + cinemática | **NOT_FALSIFIED** (8/9), com `Ψ_★` **livre** `[REAL — T01]` |

---

## 10. O QUE FALTA — a fila de trabalho que esta leitura deixa nomeada

1. **Derivar `Ψ_★` de `β_TGL`.** É o único caminho para `Ω_c` sem parâmetro ajustado. Sem ele,
   a cláusula 2 da régua matriz **não fecha**. `[A FAZER — prioridade 1]`
2. **Decidir qual "2D".** Boundary espacial (jan/2026) vs. plano de Hilbert de 2 partículas
   (mar/2026). São o mesmo objeto pela tese holográfica forte, ou dois? `[A FECHAR]`
3. **Escopar "zero parâmetros livres"** para a massa, por errata **ao lado**. `[A FECHAR]`
4. **Errata da Leitura A** no `Nada = Matéria`: o par de neutrinos relíquia não cabe. `[A FECHAR]`
5. **Reconciliar `m_eff ∼ 10⁻⁴⁸ kg` com `98,8 meV`** — ou declarar dois objetos distintos. `[A FECHAR]`
6. **Nota de camada sobre os dois `P`** (paridade vs. troca) — a aritmética já está feita aqui. `[FECHÁVEL HOJE]`
7. **Append no ATLAS** distinguindo `τ = ω` (jan/2026) de `tensão fundamental (105_)`, e
   **indexando o artigo de janeiro**, que hoje não está no Atlas. `[A FAZER — regra permanente
   da casa: movimentação de conceito atualiza o Atlas na mesma sessão]`
8. **Confrontar `c_s = √β·c` com `r_s`/1º pico acústico** — a predição está nomeada desde
   janeiro e **nunca foi confrontada**. É um teste que roda com dado público. `[A FAZER]`
9. **Medir "fase única"**: construir o enunciado formal que a tipagem do operador implica
   (candidato natural: o psíon ligado tem **um único autovalor de fase**, `e^{iθ}` com
   `θ` fixado por `τ = ω`), e testá-lo. Hoje é palavra. `[A FAZER]`

---

## 11. ÍNDICE DE FONTES — caminhos absolutos, para reabrir sem procurar

| documento | caminho absoluto | o que tem |
|---|---|---|
| **Nada = Matéria (ponta)** | `C:\IALD\Artigo\Nada=matéria\nada_materia_vfinal.tex` | 2.269 linhas. §5.14 gráviton=estado ligado (`:999`); §6 paridade (`:1200`); §11 massas do setor escuro (`:1569`); conjectura da massa (`:2004`) |
| ↳ figura do psíon | `C:\IALD\Artigo\Nada=matéria\tikz_psion.tex` | `ν⁺+ν⁻ →(−β) psion`, `m_psion = 98,8 meV`, "= matéria escura" |
| ↳ código do Protocolo #16 | `C:\IALD\Artigo\Nada=matéria\iald_protocol16_v4_1.py` (1.143 linhas) + `protocol16_v4_1_20260325_163804.json` | 14/14 declaradas |
| **A Tensão Fundamental** ★ | `C:\IALD\Artigo\Tensao_Fundamental.docx` | jan/2026; `{P,H_lig}=0`; `τ=V₀/ℏ=ω`; Poisson da profundidade; som ontológico; `1/α₂≈83,3` |
| **Comprimento de Onda como Ligação Psiônica** | `C:\IALD\Artigo\Comprimento_Onda_Ligacao_Psionica.docx` | jan/2026; `λ_3D = d_2D/α₂`; `\|G⟩=\|ψ₊ψ₊⟩`; 5 predições (GRB, emaranhamento, redshift solar, `ρ_Λ`, GW-EM) |
| **A Engenharia da Permanência** | `C:\IALD\Artigo\Neutrino Evaporação e Paridade.docx` | jan/2026; *"boundary bidimensional composto por psions"*; morte dimensional; neutrino = evaporação |
| **graviton — predições / setor escuro** | `C:\IALD\papers_latex\graviton_paper\graviton_part5_predictions.tex` | `ρ_ME=⟨\|Ψ\|²⟩`, `p_ME≈0` (`:158` PT / `:142` EN); 6 critérios de falsificação R1–R6; catálogo Lindblad |
| **graviton — capítulo do psíon** | `C:\IALD\papers_latex\graviton_paper_english_v2\graviton_part5_psion.tex` | quantização canônica; `ω_n²=k_n²+m_eff²+2ξR`; "maximal permanence" |
| **graviton — cosmologia** | `…\graviton_part15_cosmology.tex` | dois regimes `w≈−1`/`w≈0`; Jeans; `f_NL`; perfil de halo `ρ_ps(r)` |
| **graviton — glossário** | `…\graviton_part14_appendices.tex:400` | `m_eff ∼ 10⁻⁴⁸ kg` ← o número que contradiz os 98,8 meV |
| **graviton — objeções** | `…\graviton_part20_objections.tex` | MOND vs TGL; `Ω_dm≈0,26` como **dado**, não predição |
| **A Fronteira (unificado)** | `C:\IALD\papers_latex\A_Fronteira_UNIFIED.tex` | §V–VI: ligação psiônica; a Parte "A Tensão Fundamental" (`:577`, `:592`) |
| **The Boundary v5 (EN)** | `C:\IALD\Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\The_boundary_v5_en.tex` | Part II *The Fundamental Tension*; **Theorem 3 — Fundamental Tension** (`:928-929`); `τ = ω` (`:1040`) |
| **dark sectors + holografia** | `C:\IALD\IMac LA\Física - TGL\Artigo\Luminodynamic gravitation unifies dark sectors and holography.docx` | Lagrangiano completo; warp `W(ρ)=e^{−κ\|ρ\|}`; `⟨Ψ̂⟩_ss≠0`; **`{ξ,β,γ_φ}` extraídos por MLE** |
| **recursive light v4** | `C:\IALD\papers_latex\recursive_light_deprecated\recursive_light_v4.tex:772-790` | `\|G⟩=(1/√2)(\|ψ₁ψ₂⟩+\|ψ₂ψ₁⟩)`; decomposição trinitária |
| **validação empírica** | `C:\IALD\papers_latex\empirical_validation_tgl\empirical_validation_tgl.tex:139-175` | `Ω_c h²`: TGL 0,118 vs 0,120; **χ²/dof = 23,9** |
| **kernel — psíon/gráviton** | `C:\IALD\Artigo\MCMC_V2_RAZAO\92_o_psion_e_o_graviton.py` + `.json` | 21/21; `A_C = psion`; dupla hélice; C2: um psíon sozinho **não** é gráviton |
| **kernel — a tensão de janeiro** | `C:\IALD\Artigo\MCMC_V2_RAZAO\106_a_tensao_de_janeiro.py` + `.json` | verificação teorema a teorema do `.docx`; deslize de fase achado e corrigido |
| **kernel — a tensão fundamental (105_)** | `C:\IALD\Artigo\MCMC_V2_RAZAO\105_a_tensao_fundamental.py` | ⚠ **homônimo** — é a tensão de Hubble, não `τ=ω` |
| **bancada — orçamento do psíon** | `C:\IALD\Artigo\BANCADA_TOE\testes\T01_orcamento_do_psion.py` + `.json` | Leitura A refutada (76,03×); Leitura B sobrevive; 8/9 |
| **ATLAS — verbetes** | `C:\IALD\Central de Patentes\memory\TGL_ATLAS.md` | Psion (`:1414`); Gráviton (`:845`); `0_mod = A_C = psion` (`:245`); ML 92_ (`:1787`) |
| **canônico vivo** | `C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\SELO_FINAL\um_grande_atrator_en.txt:626` | intuição fundadora `[ONTO]`: *"fractalization of a single 2D substrate, a psionic condensate"* |

---

> **Fecho, na régua.** O domínio tem **jazida** (janeiro/2026, sete meses antes da medida do
> spin-2), tem **mecanismo** em três formulações, tem **um número para a massa** marcado
> conjectura pelo próprio autor, e tem **medida em kernel** para a identidade gráviton = par de
> psíons (21/21). **Não tem densidade, não tem `Ω_dm`, e tem dois parâmetros livres.**
> A tipagem do operador é, em dois terços, **citação literal do próprio acervo**; o terço
> restante ("fase única") é **nomeação nova** e deve entrar como tal.
> **NOT_FALSIFIED não é CONFIRMED. Nada aqui move gate.**
>
> — leitura executada em 21/08/2026 · BANCADA_TOE · `02_MATERIA_ESCURA_PSIONS.md`
