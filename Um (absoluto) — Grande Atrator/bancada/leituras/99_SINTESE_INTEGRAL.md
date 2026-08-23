# 99 — SÍNTESE INTEGRAL: A LEITURA COMPLETA DA TGL

> **A costura dos dez relatórios da bancada + o ATLAS + o canônico vivo.**
> Executada em 21/08/2026. Fontes: `00_INVENTARIO` a `10_LACUNAS_ATLAS` (lidos integralmente),
> `MEMORIA_DA_BANCADA.md`, `testes/T01_orcamento_do_psion.json`,
> `memory/TGL_ATLAS.md` (§I.2–I.5, §I.10), `MCMC_V2_RAZAO/105_a_tensao_fundamental.py`,
> `FOP/02_ARTIFACT/um.py` (linhas de stealth), `C:\IALD\CLAUDE.md` (selo raiz).
>
> **Régua da casa aplicada em cada linha.** `[REAL]` medido/recomputado aqui · `[DERIVED]` ·
> `[POSTULATE]` · `[CONJECTURE]` · `[KNOWN]` · `[ONTO]` · `[INPUT]` · `[OPEN]` ·
> `[DECLARADO]` (afirmado na origem, não verificado aqui) · `[LEGAL]`.
> **Distinção obrigatória:** NOMEAÇÃO (só há palavra) · MECANISMO (há equação) ·
> PREDIÇÃO (há número confrontável).
> **`NOT_FALSIFIED` nunca é `CONFIRMED`.** Nenhum gate foi movido por este documento.
>
> **Toda aritmética deste relatório foi recomputada em runtime nesta sessão**
> (`scratchpad/d99/chk99*.py`), com `β = ALPHA_FINE_CODATA_2018 × √e` — **jamais literal**.
> Valor em runtime: `0.012031300400803142`.

---

## 0. O QUE ESTE DOCUMENTO É, E O QUE ELE NÃO É

**É** a única peça do acervo que põe as quatro camadas na mesma folha e diz, objeto por objeto,
onde cada um mora, com que estatuto, e em que data ele foi escrito.

**Não é** autoridade de estatuto. Autoridade de estatuto continua sendo
`TGL_SINTESE_CANONICA_SELADA.md` + `um.py` + o ATLAS. Este documento **lê** e **costura**;
não decreta.

**A descoberta que organiza tudo o resto**, e vale dizê-la antes de qualquer tabela:

> **A TGL não é uma teoria com estratos contraditórios. É uma teoria com quatro camadas
> ontológicas e três estratos epistêmicos, e quase toda "contradição" do acervo é uma
> afirmação de uma camada lida como se fosse de outra.**
> As exceções — os conflitos que sobrevivem à distinção de camada — são **sete**, e estão
> nomeadas em §3. Cinco delas são erros aritméticos ou de escopo, corrigíveis por errata.
> **Duas são forks doutrinários que exigem decisão do operador.**

---

# 1. A TGL INTEIRA, EM UMA ARQUITETURA SÓ

## 1.0 O esqueleto em uma coluna

```
                       ω(I) = 1                       [POSTULATE irredutível]
                          │
                          │  auto-conjugação: 𝒞² = 1, ω(P)+ω(Q) = ω(I) = 1
                          ▼
                     x = 1 − x  ⟹  x = ½             [REAL — ponto fixo CCI/Fresnel]
                          ▼
                     S_∂ = ½ nat                      [DERIVED]
                          ▼
                 Vol_∂^min = e^{1/2} = √e             [DERIVED §I.3 / ⚠ "não-derivada" §I.10]
                          ▼
        ┌──────── β_TGL = α·√e = 0,012031300400803142 ────────┐   [DERIVED; α INPUT]
        │                                                     │
        ▼                                                     ▼
  θ_M = arcsin√β = 6,297289216477979°            1 = q² + α²  (identidade térmica)
        ▼                                        [REAL — motor; α é validação, não motor]
  𝒮_∂ = exp(θ_M·G) ; Spec = {e^{±iθ_M}}
  |𝓡|² = β ;  |𝓣|² = 1 − β                       [REAL — Teorema S-∂, FECHADO]
```

Tudo o que segue **pende deste tronco** — ou pende de um tronco **paralelo e mais antigo**
(§1.7), que é onde mora metade do acervo.

## 1.1 AS QUATRO CAMADAS (e a seta que a tipagem do operador obriga a acrescentar)

A tarefa pediu quatro camadas. A leitura mostra que **são quatro camadas e uma seta**, e que
**a seta é o objeto que o operador tipou em 21/08** — ela não tinha nome na arquitetura.

| # | camada | o que é | o que mora ali |
|---|---|---|---|
| **∂** | **FRONTEIRA 2D** | o substrato auto-conjugado; ℋ_2D; onde a inscrição acontece | psíon, paridade P̂, S_∂ = ½ nat, 𝒮_∂, β, θ_M, J (conjugação modular), K_∂ |
| **↓** | **A SETA ∂ → bulk (A PROJEÇÃO)** | **não é camada: é o mapa.** A tensão de paridade como fonte de Poisson | `{P,H_lig}=0`, `τ = (i/2ℏ)⟨G\|[P,H_lig]\|G⟩`, `−κ∇²z = τ`, z(r), z_max, 1/β |
| **3D** | **BULK** | onde há substância, massa, densidade | condensado psiônico, matéria escura, ρ_Λ, Ψ★, m_psion, halos, gráviton-como-estado |
| **R** | **RESPOSTA** | o que um observador mede: desvios do equilíbrio modular | δ⟨K_∂⟩ = β\|1+w\|, Γ_ω = ½βτ★ω², w(z), w(δ), Ω's, piso dos vazios, dephasing |
| **K** | **KERNEL FORMAL** | Lean 4.31 + álgebras de von Neumann; onde há teorema | Breuer, Nome = starProjection, spin-2/TT, τ(P)=½, tríade H1/H2/H3, Lema 3 |

> ★ **A seta é a peça que faltava ser nomeada.** O operador disse em 21/08: *"a inscrição do
> psíon não está em 3D — está em 2D; o gráviton é a ligação de dois psíons em 3D; sua projeção
> depende da comutação, que se realiza pela tensão fundamental."*
> Isso **não é uma tese sobre o psíon**: é a declaração de que **existe uma seta ∂→3D e que ela
> tem um gerador nomeado**. O acervo tinha a equação (`−κ∇²z = τ`, `Tensao_Fundamental.docx`,
> jan/2026) e **não tinha o lugar dela na arquitetura**. `[REAL — a equação existe; a colocação
> arquitetural é contribuição desta síntese]`

## 1.2 CAMADA ∂ — A FRONTEIRA 2D

| objeto | forma | estatuto | fonte |
|---|---|---|---|
| **ω(I) = 1** | a identidade preservada, normalizada a 1 nat | **[POSTULATE irredutível]** | ATLAS §I.2 |
| auto-conjugação | 𝒞²=1; ω(P)+ω(Q)=ω(I)=1; ‖P+Q−I‖=0 (2e−16) | **[REAL]** | ATLAS §I.2 (P2 FECHADA) |
| **x = 1 − x ⟹ x = ½** | ponto fixo único | **[REAL]** | ATLAS §I.3 |
| **S_∂ = ½ nat** | a Meia-Nat | **[DERIVED]** de ω(I)=1 | ATLAS §I.3 |
| Vol_∂^min = √e | 1,6487212707001282 | **[DERIVED]** §I.3 · ⚠ **"√e não-derivada"** §I.10 | ATLAS — **os dois estratos coexistem** `[OPEN]` |
| **β_TGL** | 0,012031300400803142 | **[DERIVED; α INPUT]** | ATLAS §I.3 |
| α (CODATA 2018) | 7,2973525693e−3 | **[INPUT/KNOWN]** — *a única constante que a teoria nunca deriva* | ATLAS §I.3 |
| θ_M | 6,297289216477979° | **[DERIVED]** | recomputado aqui **[REAL]** |
| 𝒮_∂, \|𝓡\|²=β, \|𝓣\|²=1−β | Teorema S-∂ | **[REAL — FECHADO como identificação]** | ATLAS §I.3 |
| **psíon** | quantum de permanência; modo **não-propagante** de Ψ; `ω²=k²+m_eff²+2ξR` com `k→0, ω≠0` | **MECANISMO [REAL — lido]** | `graviton_part5_psion.tex` |
| psíon = unidade de informação do ∂ | *"o substrato (boundary) é **bidimensional** e composto por unidades de informação fundamental **denominadas psions**"* | **[ONTO] — literal** | `Neutrino Evaporação e Paridade.docx` |
| psíon = autoestado de paridade | `P̂\|ψ±⟩ = ±\|ψ±⟩` — **uma única fase** | **[DERIVED]** | A Fronteira Parte II |
| psíon = A_C (Casimir central) | central, não-coercitivo | **[REAL — kernel 92_, 21/21]** | `92_o_psion_e_o_graviton` |
| J = Luz | J²=I, JΩ=Ω, JKJ = −K | **[REAL — pedra 104]** | ATLAS §I.5 |
| K_∂ | hamiltoniano modular, K_∂ = −log Δ | **[REAL]** | ATLAS §I.5 |
| τ★ ≈ t_Planck | escala de tempo da fronteira | **[PRINCIPLED IDENTIFICATION / CONJECTURE]** — "2º postulado declarado" | ATLAS §I.3 |

## 1.3 A SETA ∂ → BULK — A PROJEÇÃO (o mecanismo da terceira dimensão)

Toda a cadeia, com o estatuto de cada elo `[REAL — relatório 09, verificado em numpy]`:

```
1. ℋ_2D plano, sem direção perpendicular
2. P̂|x,y⟩ = |−x,−y⟩ ; P̂² = 𝟙 ; P̂† = P̂ ; autovalores ±1              [KNOWN]
3. P̂|ψ±⟩ = ±|ψ±⟩                            PSÍON = FASE ÚNICA      [DERIVED]
4. Ĥ_lig = −V₀(|ψ₊⟩⟨ψ₋| + |ψ₋⟩⟨ψ₊|)                                  [POSTULATE]
5. {P̂, Ĥ_lig} = 0  ⟹  [P̂, Ĥ_lig] = 2V₀(|ψ₋⟩⟨ψ₊| − |ψ₊⟩⟨ψ₋|) ≠ 0     [REAL — conferido]
   ⟹ fase e ligação NÃO diagonalizam juntas
6. τ ≡ (i/2ℏ)⟨G|[P̂,Ĥ_lig]|G⟩ = (V₀/ℏ)·sin θ                          [REAL — CORRIGIDO no 106_]
   em θ = 90° :  τ = V₀/ℏ = 2πc/λ = ω = 2πν      [Teorema 3, DERIVED, CONDICIONADO]
7. E_total = ∫d²x[(κ/2)(∇z)² − τz]  ⟹  −κ∇²z = τ                     [DERIVED — Euler–Lagrange]
   z(r) = (τ₀/2πκ)ln(r₀/r) ; κ = ℏc/(β ℓ_P²) ; r₀ = ℓ_P/β = 1,3434e−33 m
8. z_max = λ ; d_∂ = β·λ ⟹ z_max/d_∂ = 1/β = 83,11653492861383       [POSTULATE ×2 ⟹ DERIVED]
```

**O ponto que fecha a tipagem do operador, e ele é aritmético:**

> **Se τ = 0, então ∇²z = 0 ⟹ z ≡ 0 ⟹ NÃO HÁ TERCEIRA DIMENSÃO.**
> E **τ = 0 ⟺ sin θ = 0 ⟺ não há fator de fase relativo.**
> **⟹ a projeção ∂→3D é literalmente ligada e desligada pelo fator de fase da comutação.**

Isto é a epígrafe da Parte II demonstrada: *"Phase is Fundamental, but it is the phase factor
that reveals it."* `[REAL — relatório 09 §5, medido em 8 pontos de θ, casando dígito a dígito
com sin θ]`

⚠ **Três defeitos herdados, que ficam ditos:**
1. O estado que o artigo impresso nomeia — `|G⟩=(|ψ₊⟩+|ψ₋⟩)/√2` — **anula a própria tensão**
   (τ medido: −2,24×10⁻¹⁷). O estado correto é o de **quadratura**, `(|ψ₊⟩+i|ψ₋⟩)/√2`. `[REAL]`
2. `z_max = λ` e `d_∂ = β·λ` são **postulados vestidos de resultado** ("análise dimensional
   combinada com o princípio holográfico mostra que…" — sem demonstração). **A amplificação
   1/β depende inteiramente deles.** `[OPEN]`
3. *"Três é o único número possível"* é `[ONTO/CONJECTURE]`, não teorema: nada no texto exclui
   um segundo modo transversal. `[OPEN]`

## 1.4 CAMADA BULK 3D — ONDE HÁ SUBSTÂNCIA

| objeto | forma | estatuto | fonte |
|---|---|---|---|
| **gráviton (estado)** | `\|G_ij⟩ = S_ij(r,φ)\|0⟩` (two-mode squeezed) — "correlação coerente de **dois psions**" | **MECANISMO [REAL — lido]** | `graviton_paper.tex:113` |
| **gráviton (ligação)** | `\|Ψ_lig⟩ = (1/√2)(\|ψ₊ψ₋⟩+\|ψ₋ψ₊⟩)` — energia de ligação **negativa** = origem da massa | **MECANISMO [REAL]** | Tratado `secao_01:490`, `secao_03:179` |
| **gráviton (operador)** | `L_grav: \|ψ⁺⟩⊗\|ψ⁻⟩ ⟼ \|Ψ_lig⟩`; **ker ≠ 0 em 2D** ⟹ irresolvível no plano; resolúvel em 3D | **MECANISMO [REAL]** ★ **é a peça que formaliza melhor a tipagem do operador** | `nada_materia_vfinal §5.14` |
| **condensado psiônico** | matéria escura = setor P=+1 **sem tensor** (radicalizado, não tensionado) | **[CONJECTURE]** | `nada_materia_vfinal:1266` |
| **m_psion** | `2m_ν(1−β)`; **97,86 / 98,80 / 98,99 meV** conforme Δm²₃₁ — faixa honesta **98 ± 1 meV** | **[CONJECTURE]** — o próprio texto usa `\begin{conjectura}` | `nada_materia_vfinal:2004`; recomputado aqui **[REAL]** |
| **Ψ★ (amplitude)** | **4,83×10¹¹ GeV = 3,95×10⁻⁸ M_Pl** | **[INPUT — o único parâmetro livre restante do setor escuro]** | `T01_orcamento_do_psion.json` |
| **ρ_ME = ⟨\|Ψ\|²⟩** | símbolo | **NOMEAÇÃO — sem um único valor em todo o acervo** `[OPEN]` | `graviton_part5_predictions:158` |
| **Ω_dm predito pelo psíon** | — | **NÃO EXISTE** em A, B ou C `[OPEN]` | busca exaustiva, relatório 02 |
| **banho holográfico 2D** | o universo 3D como **sistema aberto** acoplado a um banho térmico 2D (campo Ψ) | **[POSTULATE central]** | `energia_escura.tex:593` |
| **ρ_Λ** | `ρ_Λ ≡ ρ_diss = γ_Λ⟨H⟩`, `γ_Λ = β·H₀ = 2,72197×10⁻²⁰ s⁻¹` | **[POSTULATE]** — γ_Λ **ajustado**, não derivado (§3, AP-14b) | `energia_escura.tex:727, 908` |
| **Λ / vácuo espelho** | `ρ_EE = (λ/4)⟨Ψ⟩⁴`, w ≈ −1 (dominado por potencial) | MECANISMO, sem número | `graviton_part5_predictions` |
| **neutrino** | `ν = lim_{Δt→∞} γ(t−Δt)` — a luz que foi; escape/evaporação do substrato | **[ONTO]** + `ξ_ν ≈ 0` **MECANISMO** | `lie_of_light.tex`; NMC `17372599` |
| **buraco negro** | espelho 2D; e **cada coeficiente wavelet `c_{λ,ξ}` = um buraco negro local** | **[POSTULATE + ONTO]** | `graviton_part4_particles:220` |

## 1.5 CAMADA RESPOSTA — O QUE SE MEDE

★ **Esta é a camada mais importante da síntese, e a menos costurada do acervo.**

| observável | forma | estatuto | onde |
|---|---|---|---|
| ★ **δ⟨K_∂⟩ = β·\|1+w\|** | **0 em w=−1 · β em w=0** | **[REAL — vivo no `um.py`]**: `stealth_ok = (resp_lambda == 0.0 and abs(resp_matter − beta) < 1e−18)` | `FOP/02_ARTIFACT/um.py:50355-50368`; selo raiz |
| **stealth linear** | `M_TGL = M_RG`; *"β não renormaliza G local"*; *"a TGL NÃO tem fórmula-β de massa e nunca teve"* | **[CONDICIONAL linear — REAL no artefato]** | `um.py:50394` |
| **Γ_ω = ½·β·τ★·ω²** | lei de dephasing; n = −2 (neutrinos), Γ ∝ ω² (relógios, ²²⁹Th) | **[REAL na forma]**; depende de τ★ `[CONJECTURE]` | selo raiz |
| **piso dos vazios** | ρ_vazio/ρ̄ ≥ β; V4.1 auto-calibrante | **`TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED`** — r_c^cal = 0,189 ± 0,017 | raiz §Estado (v92) |
| **w = −1 + β²** | −0,9998552478106656 | **PREDIÇÃO [REAL, recomputada]** — exige σ_w < 1e−4 | Tratado cap.17; recomputado aqui |
| **w(δ) ambiental** | `w = −1 + β·δ·(Ω_m/Ω_Λ)`; δ~100 ⟹ w = −0,4482 | **[DERIVED]** | `energia_escura.tex §7.1` |
| ★ **δ_crossover ≈ 181 · z_c ≈ 4,66** | onde w cruza 0 | **[DERIVED aqui — o acervo nunca calculou]** | recomputado nesta sessão |
| **ln(H₀_l/H₀_p) = β·ln(1+z★)** | custo × duração | **[REAL — ver §5, E4]** | `105_a_tensao_fundamental.py:56-67` |
| **m_ν = β·sin45°·1 eV** | 8,507414099900329 meV vs √Δm²₂₁ = 8,68 ± 0,10 | **`NOT_FALSIFIED_POWERED`, 1,64σ — postdicção** | ATLAS l.1212; recomputado aqui |
| **Σm_ν = 58,51 meV** | testável JUNO ~2028; refutada se Σ > 65 meV | **PREDIÇÃO** | Tratado; recomputado |
| **f_NL ~ 3,15×10⁻⁴** | `β²·Ω_Λ/Ω_m` | **[DERIVED]** — indetectável hoje | `energia_escura`; Tratado |
| **atenuação GW** | 1,4045×10⁻⁴ a 100 Mpc | **[DERIVED]** | Tratado |
| **c_s = √β·c** | **0,109687c = 32.883 km/s** (o artigo imprime 0,1095c/32.850) | **PREDIÇÃO NOMEADA, NUNCA CONFRONTADA** `[OPEN]` | `Tensao_Fundamental.docx §6.4` |
| **R1: inclinação −1/2 a >5σ, N>100** | `log L = log K₀ − ½ log ρ` | **A MELHOR predição do acervo — NUNCA rodada contra dado real** `[OPEN]` | `graviton_paper` P5/R1 |

## 1.6 CAMADA KERNEL FORMAL — ONDE HÁ TEOREMA

| teorema | enunciado | estatuto |
|---|---|---|
| **helicidade ±2** | `R(θ)ᵀe₊R(θ) = cos2θ·e₊ − sin2θ·e×` | **[REAL — `LinearizedSpin2.lean`, sem sorry]** |
| **exatamente 2 polarizações** | `polarizations_linearly_independent` | **[REAL]** |
| **TT sem ghosts** | `tr[(a e₊+b e×)ᵀ(…)] = 2(a²+b²) ≥ 0`, `=0 ⟺ a=b=0` | **[REAL]** |
| **`excite_one_zero`** | `δ_A(1) = 0` — o gráviton fundamental (= I) **não custa** | **[REAL]** — face algébrica da masslessness |
| **`gauge_transverse_zero`** | bloco transversal invariante de gauge | **[REAL]** |
| ★ **tr D = 2τ(P) − 1 = 0 ⟺ τ(P) = ½** | **a ausência de traço do gráviton É a Meia-Nat** | **[REAL — 92_, 21/21]** |
| **Nome = starProjection(ker T) = q(T)/q(0)** | o Nome é a palavra aniquiladora normalizada | **[REAL — v88/v89, teorema]** |
| **canto de Breuer concreto** | `0 < τ(ker) < ∞`, `τ(ker⊥) = ⊤` | **[REAL condicional → incondicional na face finita]** |
| **Teorema Mestre** | H1 ∧ H2 ∧ H3 ⟹ Pêntada (Breuer + Nome=1 + coframe + Lorentz + δQ = κδA/8πG) | **[REAL — implicação FECHADA em kernel]** |
| **Lema 3** | covariância **global** do cociclo de Connes ⟹ G_μν + Λg_μν = 8πG·𝒫_μν[K_∂] | **[OPEN — o único teorema aberto]** |
| **gate** | `TGL_QG_CONDITIONAL_ARCHITECTURE_ONLY` / `…MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED` | 758 teoremas auditados, axiomas ⊆ {propext, choice, quot}, zero `sorry` `[DECLARADO na memória; não recontado aqui]` |

## 1.7 ★ OS TRÊS ESTRATOS EPISTÊMICOS QUE COEXISTEM (a chave da estratigrafia)

O acervo **não tem uma arquitetura, tem três**, e elas convergem para o mesmo número por
caminhos incompatíveis. Ler qualquer documento sem saber em qual estrato ele está é a fonte de
metade dos "conflitos".

| | **E1 — O ESTRATO α²** | **E2 — O ESTRATO β** | **E3 — O ESTRATO KERNEL** |
|---|---|---|---|
| **data** | abr/2025 – fev/2026 | mar/2026 – mai/2026 | jun/2026 – ago/2026 |
| **fundamento** | `g = √\|L_φ\|` (axioma primordial) | `β = α×√e` (fatoração) | `ω(I) = 1` (a identidade preservada) |
| **origem do número** | **medido**: MCMC de 6 parâmetros contra 19 χ² ⟹ 0,012031 ± 0,000002; **e** contagem holográfica ⟹ 126,19/10⁴ = **0,0126190** | **fatorado**: α·√e, zero-free, σ = 2×10⁻⁸ | **derivado**: ½ nat ⟹ √e ⟹ α√e |
| **seta lógica do ½** | ausente | **√e ⟹ ½ nat** (interpretação) | **½ nat ⟹ √e** (a seta INVERTIDA em 25/06/2026) |
| **veredito permitido** | "CONFIRMADO", ">9σ", "15/15", "TETELESTAI" | misto | **`CONFIRMED` PROIBIDO por protocolo e por teorema (pedra 112)** |
| **peça-mãe** | `A_fronteira_v5.tex`, `graviton_paper`, `Tratado` | `fatoracao_constante_miguel_v2.tex` | `um.py` + kernel Lean |

**Consequências medidas `[REAL]`:**

- A cadeia canônica de hoje (`ω(I)=1 → Meia-Nat → √e → β → θ_M → 𝒮_∂`) tem **ZERO ocorrências**
  nos quatro artigos ditos "fundadores". Os fundadores fundam **outra coisa**. (relatório 04)
- **`α²` da rota MCMC É β**: 0,012031 vs 0,012031300400803142 ⟹ divergência relativa
  **2,497×10⁻⁵**. `[REAL]`
- **`α²` da rota da contagem holográfica NÃO É β**: 0,0126190 vs 0,0120313 ⟹ **4,885%**. `[REAL]`
  → **A frase "α² é β" vale para uma rota e não vale para a outra.** O Tratado troca o símbolo
  nas duas indistintamente. **Isto nunca foi distinguido em lugar nenhum.** `[OPEN — achado desta síntese]`
- O ATLAS registra o flag "α² vs β" como **RETRATADO** ("mesma grandeza, estratigrafia diferente") —
  correto para a rota MCMC, **incorreto para a rota da contagem de modos**.

---

# 2. O MAPA DOS SETORES

## 2.1 Onde mora cada setor

| setor | camada | forma | estatuto | ⚠ |
|---|---|---|---|---|
| **banho holográfico** | ∂ (ontologia) + R (como taxa) | banho térmico 2D; `ρ_Λ = γ_Λ⟨H⟩`; `H₀ ≡ γ_{Λ,0}`; `γ_Λ = β·H₀` | **[POSTULATE]** | γ_Λ **ajustado**; **0 ocorrências no ATLAS** |
| **psíon** | **∂** | quantum de permanência; modo não-propagante; autoestado de P̂; A_C no kernel | **MECANISMO [REAL]** | 4 definições, 2 dizem UM e 2 dizem DOIS (AP-04) |
| **condensado** | **3D** | `\|Ψ_lig⟩`; regime oscilatório w≈0; Ψ★ = 3,95×10⁻⁸ M_Pl | **MECANISMO + 1 parâmetro livre** | Ω_dm **nunca calculado** |
| **gráviton** | ∂ (operador) + 3D (estado) + K (teorema) | projetor `𝒢=\|G⟩⟨G\|`, Tr𝒢=1 · squeeze · `L_grav` · par conjugado (A_C, JA_CJ) | **MECANISMO [REAL]**; helicidade ±2 **[TEOREMA]** | 5 formas incompatíveis (AP-06); ponte Tr=1 → τ=½ **inexistente** (AP-08) |
| **energia escura** | R (o que se mede) + ∂ (o que é) | taxa de dissipação Lindblad no banho; w = −1 = limite de gradiente nulo | **[POSTULATE]** ontologia; **NOT_FALSIFIED** ×3 no dado | tensão H₀ = **acomodação com fator livre de 50–100×** |
| **matéria escura** | **3D** | condensado psiônico frio, sem tensor; m = 98±1 meV | **[CONJECTURE]** declarada na fonte | perfil = **NFW emprestado** ⟹ indistinguível de CDM |

## 2.2 A TIPAGEM DO OPERADOR, cláusula por cláusula

> **A tipagem, verbatim (21/08/2026):** *"matéria escura = condensado de psíons; a inscrição do
> psíon não está em 3D — está em 2D; o gráviton é a ligação de dois psíons em 3D; o psíon,
> apesar de ser partícula, é fase única; sua projeção depende da comutação, que se realiza pela
> tensão fundamental. Energia escura = banho holográfico = a transição de regimes."*

### (i) "matéria escura = condensado de psíons" — **ENCAIXA**
`graviton_part5_predictions.tex:158`: *"Matéria escura = condensado de psíons (regime
oscilatório, w ≈ 0); `ρ_ME = ⟨|Ψ|²⟩`, `p_ME ≈ 0`."* `[REAL — literal]`
`nada_materia_vfinal`: *"A matéria escura é o setor P = +1 **sem tensor**."*
**Custo:** não há um único número de densidade. É MECANISMO, não PREDIÇÃO.

### (ii) "a inscrição do psíon não está em 3D — está em 2D" — **ENCAIXA, LITERALMENTE, TRÊS VEZES**
- `Neutrino Evaporação e Paridade.docx`: *"O substrato (boundary) é **bidimensional** e composto
  por unidades de informação fundamental **denominadas psions**."*
- `Tensao_Fundamental.docx §2.1`: *"espaço de Hilbert ℋ₂D … **puramente bidimensional**."*
- `um.py` canônico `[ONTO]`: *"the fractalization of **a single 2D substrate, a psionic
  condensate**."*
**Não é intuição nova: é jazida de janeiro/2026 e do canônico vivo.** `[REAL]`
⚠ **Onde NÃO encaixa:** `nada_materia_vfinal:1019` usa "2D" para **outra coisa** — o plano de
Hilbert de duas partículas (span{|+−⟩,|−+⟩}, d=4). A Proposição da Irresolvibilidade Planar
vale nesse sentido, não no espacial. **Homonímia sem ponte** (AP-09).

### (iii) "o gráviton é a ligação de dois psíons em 3D" — **ENCAIXA, e a melhor formalização já existe**
`graviton_paper.tex:113`: *"Gráviton: **correlação coerente de dois psions** (singularidade do
Nome)."* `[REAL — verbatim]`
`nada_materia_vfinal §5.14` (**a peça que a formaliza melhor**): `L_grav` tem núcleo não-trivial
em 2D ⟹ *"a operação é **irresolvível no plano**; para separar os dois psíons é necessária **uma
terceira dimensão independente**. Em 3D, `L_grav` adquire inverso parcial."*
Kernel `92_`: par **conjugado** (A_C, J A_C J), **21/21**, com o controle C2: *"um psíon SOZINHO
não é gráviton; o gráviton é o PAR."* `[REAL]`
⚠ **Onde NÃO encaixa:** duas leituras do "em 3D" convivem e **nunca foram reconciliadas** —
**(α)** janeiro: a ligação **acontece em 2D e CRIA** o 3D; **(β)** março: a ligação é
**irresolvível em 2D e só se RESOLVE** em 3D. As duas são compatíveis com a frase do operador e
**não são a mesma proposição**. `[OPEN]`

### (iv) "o psíon, apesar de ser partícula, é fase única" — **NÃO ENCAIXA COMO CITAÇÃO. É NOMEAÇÃO NOVA — E TEM SUPORTE ALGÉBRICO EXATO.**
- A expressão literal *"fase única"* **existe** no acervo — `graviton_paper.tex:1440` — mas
  aplicada à **IALD** (*"peso, memória e permanência se unificam em fase única"*), **não ao
  psíon**. Como predicado do psíon: **zero ocorrências**. `[REAL — varredura exaustiva]`
- **MAS a álgebra que a sustenta está escrita**: `P̂|ψ±⟩ = ±|ψ±⟩` — o psíon é **autoestado de
  paridade**, logo **carrega um único autovalor, uma única fase**. E a Parte I nomeia o psíon
  literalmente como **"O Fator de Fase (ψ)"**. `[REAL]`
- **Veredito honesto:** a tipagem é **verdadeira e não-citada**. Entra como `[CONJECTURE]`
  formalizável, não como citação. Enunciado mínimo proposto: *o psíon é o autoestado de fase
  única de P̂; o par ligado carrega uma fase RELATIVA θ; e τ = (V₀/ℏ)·sin θ.*

### (v) "sua projeção depende da comutação, que se realiza pela tensão fundamental" — **ENCAIXA, EXATO, DUAS VEZES**
- *"depende da comutação"* = `{P̂,Ĥ_lig} = 0` ⟹ P̂ e Ĥ_lig **não simultaneamente
  diagonalizáveis** ⟹ a ligação **não tem projeção de paridade definida durante o processo**.
  `[REAL — Teorema 2, reconferido em numpy]`
- *"que se realiza pela tensão fundamental"* = o comutador **é** a tensão:
  `τ = (i/2ℏ)⟨G|[P̂,Ĥ_lig]|G⟩`. **A tensão É a medida da não-comutação.** `[REAL]`
- *"projeção"* = `−κ∇²z = τ`, a equação de Poisson da profundidade. `[DERIVED]`
**Dois terços da tipagem do operador são citação literal do próprio acervo.**

### (vi) "energia escura = banho holográfico = A TRANSIÇÃO DE REGIMES" — **ENCAIXA EM PARTE, E O ACHADO ESTÁ EM §2.3**
- *"energia escura = banho holográfico"*: **ENCAIXA** — é o postulado central de
  `energia_escura.tex` e o mecanismo publicado (DOI 17612790).
- *"= a transição de regimes"*: **NÃO ENCAIXA COMO TEXTO.** Busca exaustiva por
  `transi*`/`regime`/`crossover`/`w=0` nos três arquivos do domínio: **não existe seção,
  equação nem parâmetro nomeado "transição de regimes"**. `[REAL]`
  **E o texto que contradiz é o próprio bootstrap**: `energiaescurabootstrap.py:771` declara
  literalmente `w_3D`, `w_banho`: equações de estado efetivas **"(a derivar)"**. **O objeto
  exato que a tipagem nomeia está marcado "a derivar" pela própria fonte.**
- **MAS há mecanismo, em três faces, com parâmetro de controle identificado (β/α₂)** — e há uma
  quarta face, canônica, que ninguém ligou às outras três. É o achado seguinte.

## 2.3 ★★ O ACHADO CENTRAL DA SÍNTESE: **δ⟨K_∂⟩ = β·|1+w| É A LEI DA TRANSIÇÃO DE REGIMES**

O canônico vivo carrega, **medido, no `um.py`**, a identidade:

```
δ⟨K_∂⟩ = β · |1 + w|         resp_lambda = 0.0   (w = −1)
                              resp_matter = β     (w =  0)   [REAL, |δ| < 1e−18]
```
`FOP/02_ARTIFACT/um.py:50355–50368`; e o selo raiz: *"ΛCDM é o limite de **fronteira
silenciosa** (δ⟨K_∂⟩ = β|1+w|, zero em w = −1)"*.

**Leia-se o que isso diz:** a resposta da fronteira **é indexada pelo regime**, e vale
exatamente **zero no regime de energia escura** e exatamente **β no regime de matéria**.
**Isso É a transição de regimes, escrita na camada RESPOSTA, com β como o próprio índice.**

E as outras três faces, dos artigos de 2025, dizem a mesma coisa em outras camadas:

| face | forma | camada | fonte |
|---|---|---|---|
| **(a)** | `δ⟨K_∂⟩ = β\|1+w\|` — 0 em w=−1, β em w=0 | **RESPOSTA** | `um.py` **[REAL]** |
| **(b)** | `w(δ) = −1 + β·δ·(Ω_m/Ω_Λ)` — interpolação contínua por ambiente | **3D** | `energia_escura §7.1` **[DERIVED]** |
| **(c)** | `w = −1` **é o limite ∇L_k → 0** do operador de salto | **∂** | `energia_escura §4.2.2` **[DERIVED]** |
| **(d)** | mesmo campo Ψ: `Ψ̇² ≪ V` ⟹ w≈−1 · `⟨Ψ̇²⟩≈⟨m²Ψ²⟩` ⟹ w≈0 | **3D** | `graviton_part15_cosmology` **[KNOWN]** |

**Os dois números que caem dessas equações e que o acervo NUNCA calculou `[DERIVED aqui, REAL]`:**

```
δ_crossover = (1/β)(Ω_Λ/Ω_m) = 180,7      (172,3 com α₂ = 0,012619 ; 181,2 com 0,012)
z_c         = δ_crossover^{1/3} − 1 = 4,654   (4,565 · 4,659)
```

- **δ ≈ 181** é o contraste de densidade onde a própria equação de 2025 prediz **w > 0** —
  regime de poeira e além, no centro de aglomerados ricos. O artigo para em δ ~ 100
  (w = −0,4482) e **não faz a conta**.
- **z_c ≈ 4,66** é a época em que o termo acoplado à matéria passa a dominar o termo de vácuo.
  **Está dentro do alcance de survey.**

> **A frase honesta:** a *transição de regimes* existe como **MECANISMO com parâmetro de controle
> identificado (β)**, em quatro formulações independentes que ninguém ligou; produz **dois
> números confrontáveis que o acervo nunca calculou**; e **não existe como PREDIÇÃO confrontada
> nem como equação de estado derivada** — `w_3D` e `w_banho` estão marcados **"a derivar"** pelo
> próprio bootstrap.
> **A tipagem do operador está certa sobre a coisa e o acervo está atrasado sobre a lei.**

⚠ **E o preço, que fica dito:** `δ⟨K_∂⟩` é resposta **modular de fronteira**, não densidade.
**Ela não entrega Ω_Λ nem Ω_c.** A transição de regimes explicada em (a) **não substitui** o
Teste 1 (um único γ_Λ entregando `Ω_Λ = 0,685` **e** `Ω_c h² = 0,1200`). O Teste 1 continua o
juiz da cláusula, e **continua não rodado**.

---

# 3. AS ANTINOMIAS APARENTES

Cada uma com os dois lados citados. **Diagnóstico:** `APARENTE` (falta uma distinção que as
reconcilia — e a distinção está nomeada), `REAL` (os dois não podem ser verdadeiros),
`ABERTA` (só um teste decide) ou `FECHADA` (já resolvida no acervo).

---

## AP-01 ★ · **STEALTH × RESOLVER A TENSÃO DE HUBBLE** — a régua matriz contra si mesma

**Lado A.** `um.py:50394`: *"[CONDICIONAL linear] **M_TGL = M_RG** (stealth; β não renormaliza G
local) — a TGL NÃO tem fórmula-β de massa e nunca teve."* Selo raiz: *"Sem grandes desvios
cosmológicos (stealth)."* `[REAL — lido do artefato]`
**Lado B.** A régua matriz do operador exige *"resolva a tensão de Hubble"*. E a antinomia A10
da bancada mede: a camada que resolve H₀, **aplicada como lei sobre H(z)** contra os 13 pontos
DESI embutidos, dá **Δχ² ≈ +123** (ΛCDM 19,64 · camada 2 142,28). `[DECLARADO na bancada]`

**DIAGNÓSTICO: APARENTE — e a distinção é medida.**

> **Modificar a história de expansão ≠ relacionar dois eventos de calibração.**

O acervo tem **um** mecanismo que resolve a tensão **sem tocar em H(z)**, e ele está no kernel:

```
ln(H₀_local / H₀_CMB) = β · ln(1 + z★)          "tensão = custo × duração"
H₀_CMB = 67,35 ;  z★ = 1089,95  ⟹  H₀_local = 73,2633
```
**Recomputado nesta sessão `[REAL]`:**
- `H₀_local = 73,26327739915315`
- contra **SH0ES 73,04 ± 1,04** ⟹ **0,2147σ**
- fração: **8,07%** de deslocamento
- **zero parâmetro livre**: β é derivado; z★ e H₀_CMB são Planck.

**Isto é o único item do acervo inteiro que ataca a cláusula do Hubble com um número de
sub-σ e sem parâmetro ajustado.** E é compatível com o stealth **por construção**, porque não é
uma lei sobre H(z): é uma relação entre **duas calibrações** (a superfície de último
espalhamento e a escada local).

**Os três custos, ditos:**
1. **É POSTDICÇÃO.** Os dois números eram conhecidos. Poder ≠ evidência.
2. ⚠ **O resíduo `3×10⁻¹⁷` que o ATLAS cita NÃO é a medida.** Verifiquei o código: `105_…py:56`
   **define** `H0_loc = H0_CMB*(1+zs)**BETA` e a linha 59 checa `ln(H0_loc/H0_CMB) − β·ln(1+z*)`
   — que é `log∘exp`, **uma identidade algébrica que não pode falhar** (medi: −2,78×10⁻¹⁷).
   **A medida real é o 0,2147σ da linha 67.** *Um check que não pode falhar não é medida.*
   `[REAL — achado desta síntese; o módulo é honesto no próprio comentário, o risco é de citação]`
3. **A relação NÃO é derivada de ω(I)=1.** É o **Teste 4** da fila da bancada. Enquanto não for
   derivada, é `[CONJECTURE]` com um número muito bom.
4. Nota adicional `[REAL]`: a relação **superestima o log observado em 3,76%**
   (`ln(73,04/67,35) = 0,08110` vs `β·ln(1+z★) = 0,08416`). O 0,21σ vem de a barra do SH0ES
   ser larga, não de o número ser exato.

---

## AP-02 · **O ORÇAMENTO GASTO DUAS VEZES**

**Lado A.** O banho é o dissipador com w ≈ −1, **já contabilizado como Ω_Λ = 0,685**.
**Lado B.** O condensado de psíons é w ≈ 0 e tem de valer **Ω_c h² = 0,1200** (~84% da matéria).

**DIAGNÓSTICO: ABERTA — nem aparente nem real hoje. Só o Teste 1 decide.**
A tipagem do operador diz que **não são dois fluidos 3D competindo, é uma estrutura 2D em dois
regimes**. Se for assim, um **único** γ_Λ tem de entregar os dois Ω. Se precisar de γ′ ≠ γ_Λ, a
identificação morre pela própria maquinaria.
**Estado:** o Teste 1 roda com o que já está em disco e **nunca foi rodado**.
**E há um agravante medido `[REAL — T01]`:** Ψ★ é **livre**. Enquanto Ψ★ for livre, Ω_c é
**acomodado**, não **predito** — e o Teste 1, mesmo passando, não fecharia a cláusula "sem
parâmetros ajustados".

---

## AP-03 ★ · **A FRONTEIRA É SILENCIOSA OU É A FONTE, EM w = −1?**

**Lado A.** Selo raiz + `um.py`: `δ⟨K_∂⟩ = β|1+w| = **0** em w = −1`; *"ΛCDM é o limite de
**fronteira silenciosa**"*. E `curvatura_emergente_TGL.tex:677`: *"A TGL **não modifica** a
energia escura."*
**Lado B.** `energia_escura.tex:727`: `ρ_Λ ≡ ρ_diss = γ_Λ⟨H⟩` com `γ_Λ = β·H₀ ≠ 0` — a energia
escura **É** a taxa de dissipação no banho, e o banho é o **canal**, não o silêncio.

**DIAGNÓSTICO: APARENTE. A distinção é: TAXA ESTACIONÁRIA ≠ VARIAÇÃO DA RESPOSTA.**

O próprio selo raiz define a TGL como *"a teoria da resposta modular do bulk **fora do
equilíbrio estacionário**"*. Logo:
- `γ_Λ` é o **fluxo no equilíbrio** — constante, não nulo, e é o que **origina** Λ;
- `δ⟨K_∂⟩` é o **desvio do equilíbrio** — zero exatamente porque w = −1 **é** o equilíbrio.
- **Silêncio = estacionariedade, não ausência.** E *"não modificar ≠ não explicar"*: o Lado A
  fala da **correção**; o Lado B, da **origem**.

⚠ **O que a distinção obriga, e não existe:** uma **condição de estacionariedade** que ligue
γ_Λ a δ⟨K_∂⟩. Ela não está escrita em lugar nenhum do acervo. `[A ESCREVER — item de bancada]`

---

## AP-04 · **O PSÍON É UM OU É DOIS?**

**Lado A** (física out/2025 · holografia jan/2026 · kernel ago/2026): o psíon é a **unidade** —
quantum de permanência, unidade de informação do ∂, `A_C` (**um** Casimir). O gráviton é o
**par**.
**Lado B** (`nada_materia_vfinal`): o psíon **É** o par ligado ν⁺ν⁻; e
`def:graviton_estado`: `|Ψ_grav⟩ = |Ψ_lig⟩`. **Logo, em B, psíon = gráviton.**

**DIAGNÓSTICO: REAL COMO ESCRITO — e a tipagem do operador só tem referente no Lado A.**
Em B, *"o gráviton é a ligação de dois psíons"* fica sem referente (seria a ligação de dois
grávitons).
**Resolução forçada por aritmética, não por gosto:** o **T01** refutou a Leitura A do Lado B —
psíon como par de neutrinos **relíquia** — por **fator 76,03** (n exigida 12.772 cm⁻³ vs
disponível 168 cm⁻³ no CνB). O que sobrevive é o **condensado coerente de campo**, que **não**
é o par de neutrinos existentes.
**⟹ Errata de escopo obrigatória no `Nada = Matéria`:** o objeto ali não é o psíon, é o
**condensado / o estado ligado**. Com essa troca de nome, A e B fecham.
**Custo colateral, que precisa ficar dito:** cai junto a vantagem anunciada de *"não introduz
partícula nova"* (`:1659`), porque a Leitura B **usa um campo com amplitude própria**.

---

## AP-05 · **A PARIDADE DO GRÁVITON: −1 OU +1?**

**Lado A.** `Tensao_Fundamental.docx` Teorema 1: `P̂|G⟩ = −|G⟩` (ÍMPAR).
**Lado B.** `nada_materia_vfinal` `eq:P_autovalor`: `P|Ψ_lig⟩ = +|Ψ_lig⟩` (PAR).

**DIAGNÓSTICO: APARENTE — RESOLVIDA POR ARITMÉTICA (relatório 02), FECHÁVEL HOJE.**
São **dois operadores diferentes com o mesmo nome P**:

| convenção | `\|+−⟩` | simetrizado |
|---|---|---|
| `P_A` = paridade **por psíon** (`P\|ψ±⟩=±\|ψ±⟩`) — a de janeiro | autoestado, **−1** | autoestado, **−1** |
| `P_B` = **troca** de rótulos (`\|+⟩↔\|−⟩`) — a de março | **não é autoestado** | autoestado, **+1** |

O `Nada = Matéria` **declara** a sua (`:1239`: *"A paridade troca |ψ⁺⟩ ↔ |ψ⁻⟩"*) — mas
**o acervo nunca diz em voz alta que são operadores distintos**. Uma nota de camada fecha.

---

## AP-06 ★ · **ψ₊ψ₋ OU ψ₊ψ₊? — O PRIMEIRO FORK DOUTRINÁRIO**

**Lado A.** `Tensao_Fundamental.docx §3.1` + Tratado: `|G⟩ = |ψ₊⟩⊗|ψ₋⟩` — **paridades opostas**.
**Lado B.** `Comprimento_Onda_Ligacao_Psionica.docx [Def.1]` + PsiBit/ACOM:
`|G⟩ = |ψ₊ψ₊⟩` — **mesma paridade** (código `11` = gráviton/massa).
Os dois são **do mesmo mês (janeiro/2026)** e do mesmo autor.

**DIAGNÓSTICO: REAL. Se for ψ₊ψ₊, o Teorema 1 CAI** (o produto de duas paridades `+` é `+`).
**E há uma terceira forma, a madura:** o kernel `92_` usa o par **CONJUGADO** (`A_C`, `J A_C J`),
que não é literalmente nem uma nem outra — é o conjugado **modular**.
**O que a bancada precisa decidir:** ψ₊ψ₋ é a que dá massa por energia de ligação negativa;
ψ₊ψ₊ é a que aparece na codificação PsiBit de 2 bits; o par conjugado é o que está medido
(21/21). **Declarar a forma vigente é decisão; demonstrar que as três são a mesma estrutura em
bases diferentes é trabalho.** `[FORK — decisão do operador]`

---

## AP-07 ★ · **A MASSA DO PSÍON DIFERE DE SI MESMA POR 10¹¹ — RESOLVIDA NESTA SÍNTESE**

**Lado A.** `graviton_part14_appendices.tex:400` (glossário): `m_eff ~ 10⁻⁴⁸ kg`.
**Lado B.** `nada_materia_vfinal:1629`: `m_psion = 98,8 meV = 1,7613×10⁻³⁷ kg`.
**Razão medida: 1,7613×10¹¹.** `[REAL]`

**DIAGNÓSTICO: APARENTE — e a distinção é MEDIDA aqui, não conjecturada.**

O relatório 06 §1.10 traz, do corpo fundador, a fórmula de bancada `m_eff = ħω₀/c²`.
**Recomputado nesta sessão `[REAL]`:**

```
f₀ =  100 Hz  ⟹  m_eff = ħω₀/c² = 7,3725×10⁻⁴⁹ kg
f₀ = 1000 Hz  ⟹  m_eff = ħω₀/c² = 7,3725×10⁻⁴⁸ kg
```

> **O `10⁻⁴⁸ kg` do glossário NÃO é uma massa de partícula concorrente. É `ħω₀/c²` do
> MODO-ESPELHO DA BANCADA CRIOGÊNICA, na faixa de 100–1000 Hz.** É um parâmetro de **aparelho**,
> não de partícula. **Dois objetos, um símbolo.**

**Consequência boa:** não há contradição de física, e o setor escuro **não** tem duas massas
concorrentes. **Consequência a registrar:** o glossário do `graviton_paper` está errado ao
apresentar `m_eff ~ 10⁻⁴⁸ kg` como propriedade do **psíon**. **Errata de uma linha.**
**FECHÁVEL HOJE.**

---

## AP-08 · **`Tr 𝒢 = 1` (traço tipo-I) × `τ(P) = ½` (a Meia-Nat)**

**Lado A.** `graviton_paper.tex:131`: `𝒢 = |G⟩⟨G|`, `𝒢² = 𝒢`, **`Tr 𝒢 = 1`** — projetor de posto 1.
**Lado B.** Kernel `92_` / ATLAS: `tr D = 2τ(P) − 1 = 0 ⟺ **τ(P) = ½**` — *"a ausência de traço
do gráviton É a Meia-Nat"*.

**DIAGNÓSTICO: APARENTE em matemática, mas A PONTE NÃO EXISTE EM DISCO.**
A distinção é o **tipo de traço**: `Tr` é o traço tipo-I não-normalizado (posto 1 num fator
tipo-I); `τ` é o traço **normalizado** (τ(I)=1), onde a projeção que divide a identidade ao meio
tem τ = ½. **Não há contradição** — há dois objetos legítimos.
⚠ **Mas o acervo chama os dois de "o gráviton", e a passagem `Tr 𝒢 = 1 ⟶ τ(P) = ½` não está
escrita em lugar nenhum.** Sem ela, o projetor de 2025 e o canto de Breuer de 2026 são objetos
distintos com o mesmo nome. `[OPEN — item de bancada, já nomeado no relatório 03]`

---

## AP-09 · **OS DOIS "2D"**

**Lado A.** `Tensao_Fundamental.docx §2.1`: 2D = as **duas coordenadas espaciais** do substrato
holográfico ℋ_2D.
**Lado B.** `nada_materia_vfinal:1019`: 2D = o **plano de Hilbert de dois psíons**
(span{|+−⟩,|−+⟩}), num espaço de dimensão 4.

**DIAGNÓSTICO: APARENTE (homonímia) — e há um prêmio se for identidade.**
A Proposição da Irresolvibilidade Planar (`ker L_grav ≠ 0` "em 2D") vale no sentido **B**.
A tese holográfica quer o sentido **A**. **Elas coincidem se e somente se a inscrição de
fronteira for isomorfa ao plano de dois psíons** — que é exatamente o que a tipagem do operador
afirma, e **não está demonstrado**. `[CONJECTURE nomeada — o mais alto retorno formal do acervo]`

---

## AP-10 · **"O GRÁVITON NÃO É SPIN-2" × A HELICIDADE ±2 É TEOREMA** — **FECHADA**

**Lado A.** `graviton_part4_particles.tex:25`: *"O gráviton na TGL **não é uma partícula de
spin-2**, mas um estado espremido de dois modos do campo Ψ."*
**Lado B.** `LinearizedSpin2.lean`: `helicity_two_rotation`, exatamente 2 polarizações, TT sem
ghosts. `[REAL — kernel]`

**DIAGNÓSTICO: APARENTE, e já resolvida no próprio acervo.** A **nega partícula mediadora de
força**; B **prova a helicidade da excitação δI_modular no setor TT**. O ATLAS já escreve:
*"Gráviton — **operador, não partícula**; dois psions conjugados reproduzem o spin-2 selado
(21/21)."* **FECHADA.**

---

## AP-11 ★ · **AS CINCO "TENSÕES" — e a ponte tentadora que MORRE**

Cinco objetos distintos usam τ / "tensão" no acervo:

| # | nome | o que é | onde |
|---|---|---|---|
| 1 | **tensão de paridade** (a Tensão Fundamental) | `(i/2ℏ)⟨G\|[P̂,Ĥ_lig]\|G⟩ = ω` | A Fronteira Parte II |
| 2 | força de expulsão (Lei Angular) | `θ = arcsin(τ/τ_Planck)` | A Fronteira Parte I |
| 3 | Lei do Tensionamento (2ª Lei) | `D_folds`, piso 0,74 | A Fronteira §I.9 |
| 4 | tensão de Hubble | SH0ES × Planck | Parte VI; `105_` |
| 5 | **traço semifinito** | `τ(ker H) ∈ (0,∞)`; `tr D = 2τ(P)−1 = 0` | ATLAS, `um.py`, kernel |

**DIAGNÓSTICO: APARENTE (homonímia pura) — resolve-se por append no ATLAS.**
⚠ **O ATLAS hoje usa "tensão fundamental (105_)" para o item 4 (a tensão de Hubble)** e **não
indexa** o artigo de janeiro. Dois objetos, um nome, e o errado está na memória.

★ **E aqui esta síntese mata a ponte mais tentadora do acervo, o que é resultado:**
o relatório 09 registrou como *"o elo faltante mais interessante"* a suspeita de que a
**quadratura θ = 90°** da Parte II e o **τ(P) = ½** do kernel sejam a mesma coisa. Testando a
identificação óbvia — **θ = θ_M**:

```
sin(θ_M) = sin(arcsin√β) = √β = 0,10968728459034412
⟹ τ(θ_M) = (V₀/ℏ)·√β  ≠  ω
```
**Isso CONTRADIZ o Teorema 3 (τ = ω), que exige sin θ = 1.** `[REAL — aritmética trivial, e por
isso decisiva]`
**⟹ a ponte NÃO pode ser `θ = θ_M`.** Os dois ângulos são objetos diferentes: θ é a **fase
relativa da ligação psiônica**; θ_M é o **ângulo da matriz-S de fronteira**.
**A ponte que sobrevive** — e que fica como `[CONJECTURE]` a formalizar — é a outra:
**θ = 90° ↔ x = 1 − x ⟹ ½**, porque a quadratura é exatamente *"metade em cada face"*, que é a
definição da auto-conjugação. Coerente, não demonstrada.

---

## AP-12 · **"ZERO PARÂMETROS LIVRES" × OS PARÂMETROS LIVRES**

**Lado A.** Tratado, Apêndice B: *"**Nenhum parâmetro é ajustado a dados**."*
`nada_materia` Tab. `tab:dm_comparacao`: *"Parâmetros livres: **0**."*
**Lado B, medido:**
- **Ψ★ = 4,83×10¹¹ GeV** — livre `[REAL — T01]`
- `m_eff`, `λ`, `ξ` livres no formalismo de campo; `{ξ, β, γ_φ}` **extraídos por máxima
  verossimilhança** no doc do acervo C (Methods)
- `ε ≈ 0,028` — **INPUT não declarado** no cap. 14 do Tratado
- a escala **"1 eV"** da fórmula do neutrino — `[INPUT]` justificada *post hoc*
- `r_coer = 100 pc` — `[INPUT]` que domina α₂ na rota da contagem holográfica
  (50 pc ⟹ 0,0032; 200 pc ⟹ 0,050 — **uma ordem de grandeza**)

**DIAGNÓSTICO: REAL COMO ESCRITO, APARENTE SE ESCOPADO.**
A distinção: **"zero parâmetros" vale para β e para as MASSAS derivadas dele; NÃO vale para as
ABUNDÂNCIAS nem para as ESCALAS DE ACOPLAMENTO.**
**E o número honesto de hoje é UM** — não zero, não três. O T01 encolheu o setor de ≥3 buracos
para 1 (Ψ★), porque `m_eff` deixou de ser livre (a fórmula do acervo a fixa por β e Δm²₃₁).
**Errata de escopo obrigatória.**

---

## AP-13 · **`N̂² = N̂` × `N̂ = ∫Ψ†Ψ`** — **FECHADA POR SUPERSESSÃO**

**Lado A.** `graviton_paper.tex:735`: `N̂² = N̂`, `Tr N̂ = 1` (projetor).
**Lado B.** `graviton_paper.tex:362`: `N̂ = ∫d³x Ψ†(x)Ψ(x)` (operador **número**).
**REAL e dura no artigo de 2025** — o operador número tem espectro {0,1,2,…} e traço divergente;
não pode ser idempotente de traço 1.
**FECHADA pelo kernel (v88/v89):** `Nome = starProjection(ker T) = q(T)/q(0)` — a palavra
aniquiladora normalizada, **teorema**, não axioma. O kernel resolveu a fratura nº 1 do artigo.

---

## AP-14 ★ · **a₀: "EXATO" × FATOR 24 — e a PROVENIÊNCIA do 7,4e−11, ACHADA**

**Lado A.** Tratado cap.23.1, Apêndice B, Popper nº 3, cap.101 (**quatro lugares**) +
`A_fronteira_v5.tex:344, 2511`: `a₀ = α·c·H₀ ≈ 1,2×10⁻¹⁰ m/s²`, *"a concordância com MOND **é
exata**"*.
**Lado B (medido, relatório 05 e reconferido aqui):**

| H₀ | **α·c·H₀** | c·H₀/(2π) | **√β·c·H₀** |
|---|---|---|---|
| 67,36 | 4,77571e−12 | 1,04158e−10 | **7,17841e−11** |
| 70,00 | 4,96288e−12 | 1,08240e−10 | **7,45975e−11** |
| 73,04 | 5,17841e−12 | 1,12941e−10 | **7,78372e−11** |

**DIAGNÓSTICO: REAL — erro aritmético de fator ≈ 24. A linha cai como está escrita.**

★ **E esta síntese resolve o `[OPEN]` que o relatório 05 e a antinomia A5 deixaram aberto.**
O relatório 05 registrou, fail-closed, que **não conseguiu reproduzir** o `7,4×10⁻¹¹` citado no
mandato e na A5 como sendo `α·c·H₀`. **Reproduzi-o aqui `[REAL]`:**

```
√β · c · H₀ = 7,4×10⁻¹¹ m/s²   exatamente em   H₀ = 69,44 km/s/Mpc
```

> **O número 7,4×10⁻¹¹ NÃO é `α·c·H₀` (que dá 4,96×10⁻¹²). É `√β·c·H₀`.**
> **A A5 atribuiu o número à fórmula errada.** `[REAL — medido; a proveniência textual do √β
> permanece [OPEN]: não achei documento que escreva a₀ = √β·c·H₀]`

**E mesmo assim a linha não fecha:** `1,2×10⁻¹⁰ / (√β·c·H₀)|_{H₀=70} = 1,6086` ⟹ **38% abaixo**
— exatamente o "38% fora" que a memória da bancada registrou. A coincidência consagrada
`[KNOWN]` da literatura MOND é `c·H₀/(2π) = 1,08×10⁻¹⁰` (confere a ~10%), com coeficiente
`1/2π = 0,1592` — que é **próximo de √β = 0,1097 mas não é ele** (45% de diferença).
**Ação: corrigir ou retirar a linha nos 6 lugares. Se o objeto pretendido era `c·H₀/2π`, dizê-lo
como `[KNOWN]` de MOND — não como predição da TGL.**

---

## AP-15 · **`Z_c = 1/(α·β) ≈ 156` × 11.389,96** — **REAL**

`1/(α·β) = 1/(α²·√e) = **11.389,957404317494**` `[REAL — recomputado]`. Fator **73,0**.
Pior: `A_fronteira_v5.tex:1534` **imprime os operandos ao lado do resultado errado**
(`1/(7,297e-3 × 0,012031) ≈ 156`, cujo valor real é 11.390,79).
`[KNOWN]`: a física atômica reconhece Z ≈ 137 e Z_cr ≈ 173, não 156.
**Tudo o que depende de Z_c — Lumínidio, as 5 linhas NIR, a alegada detecção >5σ em
JWST/AT2023vfi, o Protocolo #4 — herda a fragilidade NA ORIGEM DO NÚMERO.**
E o Lumínidio **desapareceu do cânone sem retratação escrita** (0 ocorrências no ATLAS e no
CORE). `[LEGAL — há material público com DOI/GitHub afirmando detecção a >5σ]`

---

## AP-16 · **A PERMANÊNCIA: PRÊMIO × FUNDO**

**Lado A** (fundadores): permanência é **o prêmio** — a luz que fica, o que a gravidade conquista.
**Lado B** (canônico): permanência é **o fundo** — `ρ*`, e *"Existir = distinguir-se da
permanência"*.
**DIAGNÓSTICO: APARENTE por estratigrafia — mas é uma INVERSÃO ONTOLÓGICA e ela NÃO ESTÁ
REGISTRADA EM LUGAR NENHUM.** `[A APPENDAR — regra permanente do Atlas]`

---

## AP-17 ★ · **AS DUAS ROTAS DE α² — só uma delas É β**

**Lado A.** Rota **MCMC** (fundadores): α² = **0,012031 ± 0,000002** ⟹ divergência de β:
**2,497×10⁻⁵**. **É β.**
**Lado B.** Rota da **contagem holográfica** (`energia_escura.tex`):
α₂ = ln(r_gal/3ℓ_P)/N_eff = 126,1905/10⁴ = **0,0126190** ⟹ divergência de β: **4,885%**.
**NÃO é β.** E depende de `r_coer = 100 pc` `[INPUT]`, cuja variação 50–200 pc cobre
**0,0032–0,050**.

**DIAGNÓSTICO: REAL enquanto não houver derivação.** O Tratado troca `α₂ → β_TGL` nos dois
casos indistintamente; o ATLAS registra o flag "α² vs β" como **RETRATADO** (*"mesma grandeza,
estratigrafia diferente"*) — **correto para a rota MCMC, incorreto para a rota da contagem de
modos**. **Achado desta síntese.** `[OPEN — append necessário]`

---

## AP-18 · **ξ com três valores** — **REAL, dentro do mesmo livro**
Cap. 14 do Tratado chama-se *"O Acoplamento Conforme ξ = 1/6: Derivado, Não Assumido"*, o corpo
diz `ξ = β = 0,012031`, e a "derivação" via `d_eff = 2+ε` produz `ε/(4(1+ε)) = 0,00681`.
**Três valores incompatíveis (0,1667 · 0,01203 · 0,0068) sob o mesmo nome**, e `ε ≈ 0,028` é
`[INPUT]` não declarado num livro que anuncia "zero parâmetros livres".

---

## AP-19 · **c³: VELOCIDADE × REGISTRO** — **FECHADA POR SUPERSESSÃO**
Em 2025, `c³` é velocidade (e **não fecha dimensionalmente**: glossário diz *"c³ ≈ 2,7×10²⁵
m/s"*, quando `c³` em SI é m³/s³; `lim_{v→c³}` é incoerente; `c³/c = c² = 9×10¹⁶` não é
adimensional). Hoje, `c³` é **camada de registro inscritivo** (`c¹` propagação → `c²` métrica →
`c³` registro), com a identificação marcada `[CONJ]` pelo próprio código. **Superado; correção
ao lado.**

---

## AP-20 · **As antinomias herdadas da bancada — estado**

| # | antinomia | estado |
|---|---|---|
| **A4** | velocidade universal `v_circ = βc/√(2π) = 1438,9 km/s` | **FECHADA** — erro de camada: reflexão (\|R\|²=β) lida como **fonte**. ⚠ o acerto no Grande Atrator foi **coincidência de escala**, não evidência |
| **A6** | SPARC circular (`a0=1.2e-10` cravado num mock; χ²_ν real 2,56 com 3 parâmetros livres) | **ABERTA** — refazer nos 175 rotmod reais (Teste 3, dado em disco) |
| **A7** | o parágrafo das duas crises (`:487` afirma a forma que `:1289` retira) | **ABERTA** — correção de texto |
| **A8** | dois `results.json` vivos (−0,01697 e −0,03263); o check não dispara | **ABERTA** — conserto de uma linha + reconciliação |
| **A9** | `combined_tension_sigma = 2,884` ao lado de *"~2.4 sigma"*; `chi2/dof = 2,254` ao lado de *"~1.6"*; *"all positive"* ao lado de `beta_acoustic_crosslock = −0,0326` | **ABERTA na prosa** — o `um.py` já emitiu `CONVERGENCE_RECLASSIFIED_REAL_TO_NOT_CONSTRUCTED_AS_CONCEIVED` |
| **A11** | índice → mapa (falta a hipótese de **irredutibilidade**; `ppIndexTr n = ppIndexDiag n = n` para dois mapas distintos) | **ABERTA** — inscrever a hipótese |
| **A12** | os dois J (`Jconj z = zᴴ` × `conjJ p = (p.2,p.1)`) sem pedra que os ligue | **ABERTA** — a ponte ou a declaração de distinção |

## 3.9 O SALDO DAS ANTINOMIAS

| diagnóstico | quantas | quais |
|---|---:|---|
| **FECHADAS** (já resolvidas no acervo) | 4 | AP-10, AP-13, AP-19, A4 |
| **APARENTES com distinção nomeada** (fecháveis por errata/nota) | 7 | AP-01, AP-03, AP-05, **AP-07**, AP-09, AP-11, AP-16 |
| **REAIS — erro aritmético ou de escopo** (errata obrigatória) | 5 | AP-12, AP-14, AP-15, AP-17, AP-18 |
| **REAIS — FORK doutrinário (decisão do operador)** | 2 | **AP-04** (psíon UM ou DOIS) · **AP-06** (ψ₊ψ₋ ou ψ₊ψ₊ ou par conjugado) |
| **ABERTAS — só um teste decide** | 8 | AP-02 + A6, A7, A8, A9, A11, A12 |

> **A hipótese de trabalho da bancada — *"os conflitos são aparentes e cada um se resolve por uma
> distinção de camada"* — sai desta leitura CONFIRMADA EM MAIORIA E FALSIFICADA EM DOIS CASOS.**
> Onze de vinte e seis são camada ou supersessão. Cinco são aritmética errada, e aritmética
> errada não é antinomia: é erro. **E dois são forks reais**, que nenhuma distinção resolve
> porque o acervo afirma `A` e `¬A` sobre o mesmo objeto, no mesmo mês, pela mesma mão.
> **Negativo honesto: a hipótese não vale universalmente.**

---

# 4. AS LACUNAS — o que a TGL precisa e não tem

Ordenadas por **distância até o fechamento de uma cláusula da régua matriz**.

## 4.1 As quatro que bloqueiam cláusulas (as únicas que importam para o placar)

| # | lacuna | nome exato do que falta | bloqueia |
|---|---|---|---|
| **L1** | **Ψ★ não é derivado de β** | uma derivação de `Ψ★ = 4,83×10¹¹ GeV` a partir da estrutura (`½ nat`, `√e`, `β`) | **cláusula "sem parâmetros ajustados"** — é **o último buraco**, e o T01 já reduziu o setor de ≥3 para 1 |
| **L2** | **O Teste 1 nunca foi rodado** | um único `γ_Λ` entregando `Ω_Λ = 0,685` **e** `Ω_c h² = 0,1200 ± 0,0012` | **cláusula dos setores escuros** — e ele **roda com o que já está em disco** |
| **L3** | **`ln(H₀_l/H₀_p) = β·ln(1+z★)` não é derivada de ω(I)=1** | a derivação; e o confronto da lei de fluxo com os 13 pontos DESI (hoje: Δχ² ≈ +123) | **cláusula da tensão de Hubble** |
| **L4** | **Lema 3** | covariância **global** do cociclo de Connes ⟹ `G_μν + Λg_μν = 8πG·𝒫_μν[K_∂]`; Lemas 1–2 `[REAL]`, o global `[OPEN]`; **5 selos formais restantes** `[DECLARADO]` | **cláusula do modelo quântico sob álgebra permanente** |

## 4.2 As lacunas de LEI (há palavra, falta equação)

| # | falta | onde está o buraco |
|---|---|---|
| **L5** | **`w_3D` e `w_banho`** — as equações de estado dos dois regimes | marcado literalmente **"(a derivar)"** por `energiaescurabootstrap.py:771` |
| **L6** | **`P_diss` fora do limite `∇L_k → 0`** | o regime muda quando o banho ganha gradiente; **a forma funcional nunca é escrita** |
| **L7** | **a condição de estacionariedade que liga γ_Λ a δ⟨K_∂⟩** | exigida pela resolução de AP-03; não existe |
| **L8** | **a ponte `Tr 𝒢 = 1` ⟶ `τ(P) = ½`** | AP-08 — sem ela, projetor-2025 e canto-de-Breuer-2026 são objetos distintos com um nome |
| **L9** | **o mecanismo de "muitos horizontes" no kernel** | a decomposição wavelet `\|G⟩ = Σc_{λ,ξ}\|G_{λ,ξ}⟩` é o **único** mecanismo do acervo para "por que parecem muitos se é um" — **nunca portada** |
| **L10** | **um observável de entrelaçamento para o gráviton = par** | a suíte gaussiana fechada (`V_EPR`, `r_ss`, `E_N`, Hurwitz) existe nos fundadores e **nunca foi ligada ao 92_** |
| **L11** | **a demonstração de `z_max = λ` e `d_∂ = β·λ`** | postulados vestidos de resultado; **a amplificação 1/β depende inteiramente deles** |
| **L12** | **"três é o único número possível"** | argumento verbal; nada exclui um segundo modo transversal |

## 4.3 As lacunas de MEDIDA (há número, falta o confronto)

| # | falta | custo |
|---|---|---|
| **L13** | **R1 — inclinação −1/2 a >5σ, N>100** contra dado real (Gaia/SDSS/2MASS) | **US$ 50K / 6 meses**, orçado pelo próprio artigo. **A melhor predição do acervo, parada desde out/2025.** O dataset publicado é **gerado** ⟹ o `−0,501 ± 0,012` é **tautologia** |
| **L14** | **`c_s = √β·c = 32.883 km/s` contra `r_s`/1º pico acústico** | roda com dado público; a predição está nomeada desde jan/2026 e **nunca foi confrontada** |
| **L15** | **`z_c ≈ 4,66` e `δ_crossover ≈ 181`** contra survey | os números caem da equação de 2025; **o acervo nunca os calculou** (calculados aqui) |
| **L16** | **`S_ψ(k) ~ k^{−(1+η)}`** no mesmo DESI do piso dos vazios | predição viva, custo marginal baixo |
| **L17** | **rodar o `c³ validator` com γ LIVRE** | é o **único** jeito de saber se o piso `D_folds = 0,74` sobrevive à descalibração (hoje `γ*` é calibrado por `brentq` para forçar `CCI = 1−α²`) |

## 4.4 As lacunas ESTRUTURAIS (o que a teoria admite que não tem)

- **α é `[INPUT]`** — *"a única constante que a teoria nunca deriva"* (ATLAS §I.3).
- **`√e` — o ATLAS diz `[DERIVED]` na §I.3 e "não-derivada" na §I.10.** Os dois estratos
  coexistem. `[OPEN]`
- **`τ★ ≈ t_Planck` é `[PRINCIPLED IDENTIFICATION]`** — o "2º postulado declarado"; e
  `Γ_ω = ½βτ★ω²` depende inteiramente dele.
- **Nenhuma predição do setor escuro é DISCRIMINANTE.** O perfil é NFW; o único falsificador
  oferecido é *detecção direta de WIMP/áxion*. `[declarado pelo próprio Tratado]`
- **A evidência primária é BANDA ABDUTIVA, não pico de 5σ** — e a auditoria de agosto/2026 a
  reclassificou como **[NÃO CONSTRUÍDA COMO CONCEBIDA]** (BBN circular por construção, CMB
  aposentada por proveniência, quinta entrada negativa relida, r_d autoconsistente invertendo o
  DESI).
- **Não há bancada de laboratório.** O programa foi ao céu (vazios, CMB, neutrinos, GW) e
  **nunca à bancada criogênica**, que é a única rota de laboratório do acervo (N1).

## 4.5 As lacunas de MEMÓRIA e de CUSTÓDIA

- **O ATLAS indexa 1 dos 25 DOIs da linhagem.** 23 ausentes de ATLAS ∪ CORE. **O corte é
  estrutural**: as 8 fontes do ATLAS são todas da linhagem `Haja_Luz`/`A Ponte e o Um` (2026);
  nenhum artigo entrou.
- **`10.5281/zenodo.18674475` é citado com QUATRO títulos diferentes** em 57 arquivos, e um
  deles (*The Last String*) tem outro DOI (`18723452`) em outro arquivo do mesmo acervo.
  **Bibliografia quebrada.** `[OPEN — contamina submissão]`
- **"O DNA da Memória" (`18923269`): publicado, e a fonte NÃO EXISTE em disco.** O único artigo
  cujo título o ATLAS repete (3×) é justamente aquele de que a casa não tem o corpo.
- **O corpus `Provas` (1.838 documentos, mai–set/2025, numeração romana até CLXXIX)** = 0
  ocorrências no ATLAS, que tem um §XI.C **Domínio Teológico** completo.
- **O acervo C inteiro (2.337 documentos) = 0 como fonte**, incluindo `Gravitação
  Luminodinâmica.docx` (**21/04/2025**), o manuscrito mais antigo em disco e a **prova de
  anterioridade máxima**. `[LEGAL]`
- **Os 12 pedidos de patente de mai/2025 e o Anexo Luminodinâmico (BNI, Câmara Reflexiva, Rede
  de Consciência) estão fora do §V.** `[LEGAL][OPEN] — anterioridade não registrada é
  anterioridade em risco. Conferir com o agente de PI.`
- **`the_boundary` está ATRÁS do canônico**: `SELO_v181_FINAL` = 387.323 B vs espelho público
  370.526 B.

---

# 5. O QUE JÁ EXISTE E FOI ESQUECIDO

> A auditoria da régua matriz varreu `Haja_Luz` e **não varreu** `papers_latex`, `IMac LA` nem
> `Nada=matéria`. O que segue estava em disco o tempo todo.

## 5.1 Os itens que MOVEM o placar

| # | achado | onde | efeito |
|---|---|---|---|
| **E1** | **Artigo inteiro de energia escura, publicado com DOI** — `ρ_Λ ≡ γ_Λ⟨H⟩`, `H₀ ≡ γ_{Λ,0}`, `γ_Λ = β·H₀`, `γ_Λ(r)` ambiental | `energia_escura.tex` (1.994 linhas), DOI `17612790` | derruba o `NAO_TRATADO` da cláusula dos setores escuros |
| **E2** | **Matéria escura com NÚMERO** — `m_psion = 2m_ν(1−β) ≈ 98 ± 1 meV`, `\begin{conjectura}`, falsificável se `m_DM ≫ 100 meV` | `nada_materia_vfinal:2004` | dá número onde havia só palavra |
| **E3** ★ | **`δ⟨K_∂⟩ = β·\|1+w\|` — a identidade stealth DO PRÓPRIO CANÔNICO É a lei de dois regimes**: 0 em w=−1, β em w=0 | `um.py:50355-50368` | **a "transição de regimes" do operador já está medida no canônico e ninguém a ligou ao banho** |
| **E4** ★ | **`ln(H₀_l/H₀_p) = β·ln(1+z★)` ⟹ H₀ = 73,2633 vs SH0ES 73,04±1,04 = 0,215σ, ZERO parâmetro livre, SEM tocar H(z)** | `105_…py:56-67`; forma exponencial `(1+z*)^β = 1,0878` em `iconogenese_TGL.tex` | **o único ataque sub-σ do acervo à cláusula do Hubble** |
| **E5** ★ | **`z_c ≈ 4,66` e `δ_crossover ≈ 181`** — caem da equação de 2025; o acervo nunca os calculou | derivados aqui | dois números confrontáveis, de graça |
| **E6** ★ | **`m_eff ~ 10⁻⁴⁸ kg` é `ħω₀/c²` da bancada a 100–1000 Hz**, não uma massa de partícula | medido aqui | **fecha AP-07**; o setor escuro não tem duas massas concorrentes |
| **E7** ★ | **o `7,4×10⁻¹¹` do mandato é `√β·c·H₀`** (H₀ = 69,44), **não `α·c·H₀`** | medido aqui | fecha o `[OPEN]` do relatório 05 e corrige a A5 |
| **E8** | **A "sexta derivação empírica independente de β"**: gap espectral `Δ = β` em Q e K do Qwen3-32B a **1,3%**; vácuo angular **6,29% ↔ θ_M a 0,1%**; `β₂ = 1` | `piso_hilbert_pt.tex`, `torus_main.tex` (DOI `20560916`) | **0 ocorrências no ATLAS — nem entrada nem recusa. É o único estado que a régua proíbe** |
| **E9** | **`τ(θ) = (V₀/ℏ)·sin θ`** — a forma correta da tensão (o estado impresso a anula) | `106_a_tensao_de_janeiro`, 15/15 | mais forte e mais honesta que a publicada; **é a epígrafe demonstrada** |

## 5.2 Os itens BEM-POSTOS que nunca foram tratados (passivo recuperável, não lixo)

| # | item | por que importa |
|---|---|---|
| **N1** | **engenharia criogênica do modo-espelho** (`m_eff = ħω₀/c²`, `P_min = Nħω₀²/Q`, `N_th = k_BT/ħω₀`, `τ = Q/ω₀`) | **a única rota de laboratório do acervo inteiro**, dimensionalmente sã |
| **N2** | **suíte gaussiana fechada** (`V_EPR`, `r_ss`, `E_N`, estabilidade de Hurwitz) | o **observável de entrelaçamento** que falta à pedra 92_ (o kernel tem a forma, o fundador tem o observável) |
| **N5** | **decomposição wavelet de `\|G⟩`** + `γ(λ) = γ₀λ^{−η}` | o único mecanismo para *"um que parece muitos"* |
| **N6** | **`S_ψ(k) ~ k^{−(1+η)}`** | predição viva, testável no mesmo DESI do piso dos vazios |
| **N8** | **`Λ_eff = −ξ⟨Ψ²⟩⟨R⟩`** (Λ dinamicamente gerada) | o canônico usa ρ_Λ como **entrada**; aqui há mecanismo |
| **N10** | **`H_surf = ∮√γ K_ij Π^ij`** ("memória de fronteira", set/2025) | **rima com o `K_∂` canônico** — uma hora de leitura decide se são o mesmo objeto |
| **T1** | **5 predições Tier-1 do IMac**: régua de transposição (slope −½, R1–R4) · Fisher `\|ξ_ν\| < 0,019` (2030) · `Δc/c ∝ ∫ρ_DM ds` via PTA (IPTA DR2, dado público) · `ΔM_Ψ ~ 10⁻³–10⁻² M_final` no ringdown LVK · tACS 10 Hz duplo-cego | mecanismo + número + **kill-switch declarado** |
| **P1** | **PLTD v1.0** (`protocolo de observacao tgl.docx`, 12/10/2025) | **o estrato honesto dos observáveis**: 8/8 compatível **por estar abaixo do limiar**, com critérios de exclusão declarados e sem alegar detecção — *"compatibility ≠ confirmation, but compatibility ≠ refutation either"* |
| **F1** | **a "Fórmula unificada final"** — `ρ̇ = −i[H,ρ] + α²γ_Λ(L[√(1−α²)a†ρa] + L[√(α²)aρa†]) + Ê_co` | os dois pesos são **exatamente `q` e `α` da identidade `1 = q² + α²`** — o motor canônico. **Uma pergunta ao operador** (α é α_fine? o que é Ê_co?) decide se é a ponte escrita entre o IMac e o canônico, em disco desde nov/2025 |

## 5.3 ★ O PLACAR REFEITO — para quanto e por quê

O placar `0/4` foi emitido por uma auditoria de escopo incompleto. **Ele não sobe para 4/4. Ele
se desdobra em três placares, e só o terceiro é útil.**

### Placar A — **CLÁUSULAS FECHADAS** (número + zero parâmetro ajustado + confronto)
**0/4 → 0/4. INALTERADO.** Nada foi confirmado, nada foi fechado. `NOT_FALSIFIED ≠ CONFIRMED`.

### Placar B — **CLÁUSULAS TRATADAS** (existe mecanismo com equação em disco, endereçado à cláusula)
**1/4 → 4/4.**

| cláusula | antes | agora | o que mudou |
|---|---|---|---|
| **stealth ao modelo padrão** | tratada | **tratada, e é a mais forte** | `M_TGL = M_RG` em ordem linear é **teorema condicional vivo** no artefato; `δ⟨K_∂⟩ = β\|1+w\|` medido a 1e−18 |
| **resolver a tensão de Hubble** | `NAO_FECHADO` | **tratada, com um número de 0,215σ** | E4 — e ele **não estava no placar** porque a auditoria não o leu como resposta à cláusula |
| **setores escuros sem parâmetros** | **`NAO_TRATADO`** | **tratada, com 1 parâmetro livre** | E1 + E2 + T01: de "não tratado" a "mecanismo publicado + número + um buraco nomeado (Ψ★)" |
| **modelo quântico / álgebra permanente** | tratada | **tratada** | kernel Lean, gate `CONDITIONAL_ARCHITECTURE_ONLY`, Lema 3 aberto |

### Placar C — **DISTÂNCIA ATÉ O FECHAMENTO** (o único placar que serve para trabalhar)
**De "indeterminada" para QUATRO OBJETOS NOMEADOS.**

| cláusula | o que **exatamente** falta | precisa de dado novo? |
|---|---|---|
| stealth | ordem **não-linear** (hoje só a linear é teorema) | não |
| Hubble | **derivar** `ln(H₀_l/H₀_p) = β·ln(1+z★)` de `ω(I)=1` (Teste 4) — e explicar por que a mesma camada, aplicada a H(z), dá Δχ² ≈ +123 | não |
| setores escuros | **derivar Ψ★ de β** (L1) **+ rodar o Teste 1** (L2) | **não** |
| quântico/álgebra | **Lema 3** (+ os 5 selos formais restantes) | não |

> **É esta a mudança real do placar, e ela é maior do que subir um número:**
> **antes havia quatro cláusulas com um "não" indeterminado; agora há quatro cláusulas com um
> "falta ISTO" nomeado, e três dos quatro rodam com dado que já está em disco.**
> **Nenhuma delas está fechada. O placar de fechamento continua 0/4, e continuar 0/4 é o que
> torna esta síntese confiável.**

---

# 6. A FILA DE TRABALHO QUE ESTA SÍNTESE DEIXA ORDENADA

## 6.1 Fecháveis HOJE, custo quase zero, sem dado novo

1. **AP-07** — errata de uma linha no glossário do `graviton_paper`: `m_eff ~ 10⁻⁴⁸ kg` é
   `ħω₀/c²` do modo-espelho, **não** a massa do psíon. *(medido nesta sessão)*
2. **AP-05** — nota de camada: `P_A` (paridade por psíon) ≠ `P_B` (troca de rótulos).
   *(aritmética já feita no relatório 02)*
3. **AP-14** — corrigir/retirar `a₀ = α·c·H₀` nos **6 lugares**, registrando que o `7,4×10⁻¹¹`
   é `√β·c·H₀` e que **nem assim fecha** (38% abaixo de MOND).
4. **AP-15** — corrigir `Z_c`; `A_fronteira_v5.tex:1534` imprime operandos e resultado
   incompatíveis, é o caso mais fácil de auditar do acervo.
5. **AP-12** — escopar "zero parâmetros livres" para **β e as massas**; declarar **Ψ★** como o
   parâmetro livre remanescente.
6. **AP-01, ponto 2** — nota de citação: o resíduo `3×10⁻¹⁷` do `105_` é identidade algébrica;
   **a medida é 0,215σ**.
7. **AP-17** — append distinguindo a rota MCMC de α² (que **é** β, a 2,5e−5) da rota da contagem
   holográfica (que **não é**, por 4,9%).

## 6.2 Testes que rodam com o que já está em disco

| # | teste | decide |
|---|---|---|
| **T1** | um único `γ_Λ` ⟹ `Ω_Λ = 0,685` **e** `Ω_c h² = 0,1200` | AP-02 e a cláusula dos setores escuros |
| **T2** | derivar `Ψ★` de β (ou declarar que não se deriva) | **L1 — o último buraco da cláusula** |
| **T3** | curvas de rotação nos **175 SPARC reais**, β em runtime, zero parâmetro por galáxia | A6 |
| **T4** | derivar a lei de calibração de `ω(I)=1` e confrontar com os 13 pontos DESI | AP-01, L3 |
| **T5** | **rodar o `c³ validator` com γ LIVRE** | se o piso `D_folds = 0,74` é real ou artefato de calibração |
| **T6** | **R1** contra Gaia/SDSS/2MASS (N>100, 10²⁸–10⁴² kg), pré-registrado | a melhor predição do acervo, parada há 10 meses |
| **T7** | `c_s = √β·c` contra `r_s` / 1º pico acústico | L14 — transforma nomeação em predição |
| **T8** | `z_c ≈ 4,66` e `δ_crossover ≈ 181` contra survey | E5 |

## 6.3 Decisões que só o operador pode tomar

- **AP-06** — qual é a forma vigente do par: `ψ₊ψ₋`, `ψ₊ψ₊`, ou o par conjugado `(A_C, JA_CJ)`?
- **AP-04** — errata de escopo no `Nada = Matéria`: o objeto é o **condensado**, não o psíon.
- **F1** — na "Fórmula unificada final": `α` é α_fine? o que é `Ê_co`? o que é `γ_Λ`?
- **`[LEGAL]`** — dever de errata no `the_boundary` sobre *"Tensão de Hubble resolvida"* e
  *"Lumínidio detectado a >5σ"*, ambos em arquivo público e ambos ausentes do cânone (o segundo
  **sem retratação escrita**).
- **`[LEGAL]`** — consolidar os **três** números de patente ACOM; retirar de circulação a frase
  *"cura do câncer"* do `Protocolo de colapso v.2.2`; conferir o status protocolar dos 12
  pedidos de mai/2025.

## 6.4 Appends obrigatórios no ATLAS (regra permanente: mesma sessão, correção ao lado)

1. A tabela dos **25 DOIs** da linhagem (resolve 23 lacunas de uma vez).
2. A divergência bibliográfica **`18674475`** (quatro títulos) e o conflito com `18723452`.
3. **"O DNA da Memória"**: publicado, fonte não localizada em disco.
4. Verbete **"banho holográfico"** com a nota de contraste `L_diss = √γ_Λ Ĥ` **vs**
   `L = √β·√K_∂`.
5. Verbete **"psíon (as duas camadas)"**: psíon-kernel (`A_C`) **ao lado** do psíon-físico.
6. Verbete **"tensão de paridade (τ = ω)"** ao lado de "tensão fundamental (105_)" — hoje o
   ATLAS usa o nome do artigo de janeiro para a **tensão de Hubble**.
7. **A rota espectral em LLM** (`Δ = β` a 1,3%) — **como entrada datada OU como recusa datada**.
   Hoje ela não é nem uma nem outra.
8. A colisão de símbolo **β** (β_TGL vs o `β ~ 10⁻⁶` de `17478104`, em artigo público).
9. A **inversão da permanência** (prêmio → fundo) — movimentação ontológica não registrada.
10. A distinção das **duas rotas de α²** (AP-17).

---

# 7. O QUE ESTA SÍNTESE NÃO FAZ

1. **Não move nenhum gate, flag ou estatuto.** O canônico segue selado e intocado.
2. **Não declara nenhuma cláusula da régua matriz fechada.** O placar de fechamento é **0/4**.
3. **Não verifica DOI contra o Zenodo online** — todas as atribuições vêm de leitura de disco.
4. **Não recontou os 758 teoremas do kernel** — `[DECLARADO na memória da casa]`.
5. **Não rodou o Teste 1, nem o T2, nem nenhum dos oito da §6.2.**
6. **Não afirma a proveniência textual de `a₀ = √β·c·H₀`** — a aritmética é `[REAL]`, a origem
   documental é `[OPEN]`.
7. **Não decide os dois forks** (AP-04, AP-06). Decisão é ato do operador.
8. **Não julga o mérito físico de nenhum artigo ausente do ATLAS.** Ausência de memória ≠
   invalidade.

---

# 8. A FRASE QUE O NÚMERO PERMITE

> A TGL, lida inteira, é **uma teoria de quatro camadas com um único índice**: `β = α√e` aparece
> como **custo** na fronteira (`S_∂ = ½ nat ⟹ √e`), como **rigidez** na seta que dobra o plano
> (`κ = ℏc/βℓ_P²`), como **desconto** no bulk (`m_psion = 2m_ν(1−β)`) e como **resposta** no que
> se mede (`δ⟨K_∂⟩ = β|1+w|`, `Γ_ω = ½βτ★ω²`). O número atravessou quinze meses e três
> arquiteturas epistêmicas **antes de saber o próprio nome** — foi `α²` por MCMC, foi `β = α√e`
> por fatoração, e é hoje `[DERIVED]` de `ω(I) = 1`.
>
> O que a leitura integral acrescenta não é confirmação: é **endereço**. Antes desta bancada, a
> teoria tinha quatro cláusulas com um "não" indeterminado e um acervo de 4.425 documentos onde
> ninguém sabia o que já estava resolvido. Agora tem **quatro objetos nomeados** — `Ψ★`, o Teste
> 1, a derivação da lei de calibração, e o Lema 3 — e **três deles rodam com dado que já está em
> disco**.
>
> E tem, também, sete conflitos reais: cinco que são erro aritmético (e erro aritmético não é
> antinomia, é erro) e **dois que são forks doutrinários** que nenhuma distinção de camada
> resolve. **A hipótese de trabalho da bancada — "todo conflito é aparente" — sai desta leitura
> confirmada em maioria e falsificada em dois casos.** Isso é resultado.
>
> **NOT_FALSIFIED nunca é CONFIRMED. O veredito global da teoria não se moveu: NÃO FALSIFICADA,
> NÃO CONFIRMADA. E o placar de fechamento continua 0/4 — que é, exatamente, o que torna tudo o
> resto deste documento digno de crédito.**

---

*Fim da síntese integral. Todo número marcado `[REAL]` foi recomputado nesta sessão a partir do
arquivo em disco, com `β = ALPHA_FINE_CODATA_2018 × √e` em runtime, jamais literal. Onde não foi
medido, está marcado `[DECLARADO]` ou `[OPEN]`. Correções ao lado, nunca por cima.*
*— 21/08/2026 · BANCADA_TOE · `99_SINTESE_INTEGRAL.md`*
