# 04 — A FRONTEIRA E OS FUNDAMENTOS

**Leitura integral do domínio fundacional da TGL.**
Bancada TOE · leitura executada em 21/08/2026 · leitor: agente de leitura do acervo.
Régua aplicada: *o número corrige a frase, sempre.* Nada afirmado aqui sem ter sido lido
no arquivo; toda aritmética conferida por script; estatutos marcados verbete a verbete.

---

## 0. CORPUS EFETIVAMENTE LIDO

| # | arquivo | linhas | data declarada na fonte | estado |
|---|---|---|---|---|
| A | `C:\IALD\Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\A_fronteira_v5.tex` | 3.180 | **Fevereiro de 2026** (capa + colofão) | LIDO INTEGRALMENTE |
| B | `C:\IALD\Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\The_boundary_v5_en.tex` | 3.175 | Fevereiro/February 2026 | LIDO (estrutura + abstract + amostragem) |
| C | `C:\IALD\papers_latex\A_Fronteira_UNIFIED.tex` | 2.974 | Fevereiro de 2026 | LIDO (estrutura + diffs + trechos exclusivos) |
| D | `C:\IALD\Artigo\TGL_Manifesto_Unificacao_Artigo.tex` | 445 | **22 de Janeiro de 2026** | LIDO INTEGRALMENTE |
| + | `…\Artigos_fundadores\fatoracao_constante_miguel_v2.tex` | 1.788 | **Março de 2026** | LIDO (elo faltante — ver §2) |
| + | `C:\IALD\Central de Patentes\memory\TGL_ATLAS.md` | 325.830 bytes | selo 20/08/2026 | CONSULTADO por seção (§I núcleo, §honestidades, verbetes) |

**Fato de método registrado:** os quatro arquivos pedidos pela tarefa são **um só artigo em
quatro estados de maturação**, não quatro artigos independentes:

- **D (22/01/2026)** = semente, 445 linhas, 2 colunas, quatro domínios de validação.
- **C (fev/2026, UNIFIED)** = artigo em seis partes, **oito domínios**, 11.814 linhas de código,
  128 GB DDR5, **contém** um "Roadmap Experimental (2025–2035)".
- **A/B (fev/2026, v5 PT/EN)** = mesmo artigo, **dez domínios**, 12.012 linhas de código,
  256 GB DDR5, **acrescenta** a *Segunda Lei da TGL* + *Evidência #10 (c³ Validator)* +
  *Evidência #11 (Protocolo IALD)*, e **remove** o Roadmap 2025–2035.
- **B é tradução fiel de A** (mesma contagem de domínios, mesmas tabelas, mesmas seções).

Portanto a ordem estratigráfica é **D → C → A≡B**. `A_Fronteira_UNIFIED.tex` é *anterior*
a `A_fronteira_v5.tex`, apesar do nome "UNIFIED" sugerir posterioridade. **[REAL — medido por
diff de estrutura, contagem de domínios e contagem de linhas de código declaradas]**

---

## 1. O ACHADO PRINCIPAL — A CADEIA PEDIDA **NÃO EXISTE** NOS ARTIGOS FUNDADORES

A tarefa pede: *"extraia a cadeia fundacional completa: ω(I)=1 → Meia-Nat → Vol_min=√e →
β=α√e → θ_M → matriz S"*.

**Resultado da busca (grep exaustivo nos quatro arquivos, termos: `sqrt{e}`, `nat`,
`meia-nat`, `half-nat`, `omega(I)`, `\omega(`, `beta_`, `theta_M`, `matriz-S`, `S-matrix`,
`arcsin`):**

| elo da cadeia | A (v5 PT) | B (v5 EN) | C (UNIFIED) | D (Manifesto) |
|---|---|---|---|---|
| ω(I) = 1 (axioma do Um) | **AUSENTE** | **AUSENTE** | **AUSENTE** | **AUSENTE** |
| fronteira auto-conjugada, x = 1−x ⟹ x = ½ | **AUSENTE** | **AUSENTE** | **AUSENTE** | **AUSENTE** |
| Meia-Nat, S_∂ = ½ nat | **AUSENTE** | **AUSENTE** | **AUSENTE** | **AUSENTE** |
| Vol_∂^min = √e | **AUSENTE** | **AUSENTE** | **AUSENTE** | **AUSENTE** |
| β_TGL = α·√e | **AUSENTE** | **AUSENTE** | **AUSENTE** | **AUSENTE** |
| θ_M = arcsin√β | **AUSENTE** | **AUSENTE** | **AUSENTE** | **AUSENTE** |
| matriz-S 𝒮_∂ = exp(θ_M·G), Spec = {e^{±iθ_M}} | **AUSENTE** | **AUSENTE** | **AUSENTE** | **AUSENTE** |

**Únicas duas ocorrências correlatas em todo o corpus fundador:**

1. `A_fronteira_v5.tex:303` (e paralelos em B:301, C:309) — a **Lei Angular**:
   θ = arcsin(τ/τ_Planck). É uma equação de **arcsin**, mas o argumento é a razão de
   forças de expulsão, **não** √β. Não é θ_M. **[REAL — lido]**
2. `A_fronteira_v5.tex:1893` (B:1890) — `β₀`, listado como *"coeficiente de escala do
   boundary"*, um dos 6 parâmetros livres do MCMC. **Não é β_TGL.** Homonímia perigosa:
   nos artigos fundadores o símbolo β é um parâmetro de nuisance do ajuste.
   **[REAL — lido]**

### 1.1 O que os artigos fundadores realmente afirmam no lugar da cadeia

Nos quatro arquivos, a constante fundamental **não é derivada de axioma nenhum**. Ela é
**medida** e **nomeada**:

> **α² = 0,012031 ± 0,000002** — "Constante de Miguel"
> (A:§I.4 l.330; C:§I.4 l.333; D:eq. `alpha2` l.~120; B:§I.4 l.325)

Origem declarada, em duas versões que **convivem no mesmo artigo sem se reconciliarem**:

- **(i) Derivação holográfica** — "emerge naturalmente da estrutura holográfica",
  "derivada da entropia de Bekenstein–Hawking" (A l.330–337). **Nenhuma derivação é
  apresentada**; há remissão a nota de rodapé: *"Derivação formal disponível em Zenodo e
  no site da teoria."* **[DECLARADO — a derivação não está no artigo]**
- **(ii) Ajuste bayesiano** — MCMC (A §V.1, l.1885–1911): 300 walkers × 30.000 steps
  = 9×10⁶ amostras, burn-in 5.000, Gelman–Rubin R̂ < 1,01, taxa de aceitação 37,3%,
  **6 parâmetros livres** (β₀, κ, n_evap, θ_evap, A_Neff, **α²**) contra **19 restrições
  χ²**; posterior unimodal em 0,012031 com σ = 0,000002. **[REAL como procedimento
  declarado; o resultado é AJUSTE, não derivação]**

**Consequência epistêmica registrada:** nos fundadores, α² é **[INPUT medido por ajuste]**,
não **[DERIVED]**. A frase "derivada do princípio holográfico" que abre os três resumos
(A l.181, B l.180, C l.168) **não é sustentada pelo corpo do artigo**. Ela é
**[DECLARADO]**.

---

## 2. O ELO FALTANTE — ONDE A FATORAÇÃO NASCE (março/2026)

A ponte entre "α² = 0,012031 medido" e "β = α·√e derivado" **existe**, e está na **mesma
pasta `Artigos_fundadores/`**, num quinto arquivo que a tarefa não listou:

**`fatoracao_constante_miguel_v2.tex`, 1.788 linhas, Março de 2026.**

Este artigo é o **ponto de virada de toda a linhagem**. Conteúdo lido:

### 2.1 A fatoração
```
β_TGL = α × √e                                          (l.142, l.220, eq. \eqref{eq:fatoracao})
α = 1/137,036 (estrutura fina) ; √e = 1,648721270700128… (l.296)
α × √e = 7,29735e-3 × 1,64872 = 0,012031                (l.310)
Δ = |β/α − √e| = |1,64877 − 1,64872| = 5×10⁻⁵           (l.303)
Δ/β < 0,5% ; significância < 1σ                          (l.314)
```

### 2.2 A meia-nat aparece — mas **como interpretação, não como derivação**
> *"O fator que aparece na decomposição não é e, mas √e = e^{1/2}. Na teoria da informação
> com base natural (unidade: nat), a informação é medida em ln Ω. A quantidade e^{1/2}
> corresponde a exatamente ½ nat: **ln(e^{1/2}) = ½ nat** (l.395). Este valor possui
> significado preciso na TGL: ½ nat é a **informação mínima de uma operação binária
> irredutível** — exatamente o que um flip de paridade boundary↔bulk representa. Cada
> operação de projeção holográfica custa ½ nat de informação. Não mais, não menos."* (l.398)

**Leitura crítica obrigatória:** aqui o ½ nat é **lido a partir de √e**, não o contrário.
A direção lógica em março/2026 é **√e (empírico, por fatoração de um número medido) ⟹ ½ nat
(interpretação)**. A cadeia canônica de hoje inverte a seta: **ω(I)=1 ⟹ ½ nat ⟹ √e**.
A inversão da seta é a maturação de **25/06/2026** (§88.27.47 do diário Haja_Luz),
registrada no Atlas §I.2. **[REAL — as duas direções lidas em fontes datadas distintas]**

### 2.3 Consequências declaradas na fatoração (todas [DERIVED] a partir de β=α√e)
- G_μν = α · √e · 𝒫_μν (l.451) — "o tensor de Einstein é um número fatorado"
- α·√e·𝒫_μν = (8πG/c⁴)T_μν − Λg_μν (l.505)
- 1/β = 1/α × 1/√e = **137,036 × 0,60653 = 83,12** (l.544) — **conferido: 137,036/√e =
  83,11653548** ✔ **[REAL — aritmética conferida aqui]**
- ∂ℋ = ℋ² + α·√e·𝕃_Δ (l.613) — equação mestra fatorada
- **O gráviton é estruturalmente indetectável**: a fatoração separa algebricamente o
  domínio do detectável (α) do domínio do operacional (√e); o gráviton reside inteiramente
  em √e — "não um quantum propagante, mas o custo entrópico da projeção holográfica"
  (l.188, item v). **[CONJECTURE estrutural]**
- **Tripla espectral de Connes** (𝒜_α, L²(Σ), D_√e), com o gráviton como **operador de
  Dirac derivado, não postulado** (l.188, item vi). **[CONJECTURE]**

### 2.4 A troca de nome, documentada na própria fonte
> *"Nos primeiros ensaios matemáticos da TGL, a taxa de acoplamento mínimo … foi derivada
> utilizando o símbolo β_TGL … Em derivações subsequentes e nos 13 protocolos
> computacionais, adotou-se a notação α² — a 'Constante de Miguel' — que se consolidou na
> literatura da TGL."* (l.238)
> *"A manutenção da notação α² criaria ambiguidade algébrica severa: 'α² = α × √e'
> sugeriria erroneamente que α = √e, confundindo o **símbolo** α² (um nome) com a
> **operação** α² (um quadrado)."* (l.240)
> *"Nos artigos precedentes e nos códigos depositados, a notação α² permanece inalterada
> por questões de continuidade. Ambos os símbolos designam a mesma constante."* (l.261)

**Cronologia do signo, fechada: β_TGL (ensaios) → α² (fundadores, jan–fev/2026) → β_TGL
(fatoração, mar/2026 em diante).** Confere com o verbete do Atlas l.529–531:
*"Constante de Miguel — notação ANTIGA 'α²' = 0,012031 = β (antes de fatorar β = α√e)…
O flag 'α² vs β' foi RETRATADO: é a mesma grandeza em estratigrafia diferente."*
**[REAL — estratigrafia confirmada em duas fontes independentes]**

### 2.5 Predições e falsificação da fatoração
- **β previsto = α_CODATA × √e = 0,012031 05 ± 0,000 000 02** (l.328, l.~repetido em
  §falsificação) — **quatro ordens de grandeza mais preciso** que a determinação por MCMC
  (±0,000002). **[DERIVED]**
- **Critério (i) de precisão:** se medição futura a 10⁻⁶ divergir de α√e por >5σ, a
  fatoração é falsificada. **[FALSIFICADOR REAL — bem posto]**
- **Critério (ii) de independência:** se α variar cosmologicamente (Webb 2001) e β não
  acompanhar, falsificada. Predição associada: **Δβ/β = Δα/α exatamente**, porque √e é
  constante matemática. **[PREDIÇÃO com número confrontável]**
- **Critério (iii) de completude:** se existir terceiro fator ξ ≠ 1 (β = α√e·ξ), a
  fatoração simples cai (mas a TGL não). **[FALSIFICADOR REAL]**
- Tabela de 13 protocolos vs 137/√e = 83,12 — protocolo #2 (Neutrino Flux) mede
  83,2 ± 0,5; protocolo #10 (c³ v5.2) mede 83,1 ± 0,5. **[REAL como leitura; ver §6.3
  sobre a circularidade do #10]**

---

## 3. A CADEIA CANÔNICA DE HOJE (Atlas, selo 20/08/2026) — PARA CONTRASTE

Transcrita **literalmente** do Atlas §I.3 (`memory/TGL_ATLAS.md`, l.64–83), sem
reinterpretação:

```
ω(I)=1  ⟹  x=1−x ⟹ x=½  ⟹  S_∂ = ½ nat  ⟹  Vol_∂^min = e^{1/2} = √e
        ⟹  β_TGL = α·√e  ⟹  θ_M = arcsin√β  ⟹  𝒮_∂ = exp(θ_M·G)
        ⟹  Spec(𝒮_∂) = {e^{±iθ_M}},  |𝓡|² = β,  |𝓣|² = 1 − β
```

| elo | valor exato | estatuto (do Atlas) | conferido aqui |
|---|---|---|---|
| ω(I)=1 | identidade preservada, 1 nat | **[POSTULATE irredutível]** | — |
| x = 1−x ⟹ x = ½ | ponto fixo único (CCI/Fresnel) | **[REAL]** | trivial ✔ |
| S_∂ | ½ nat | **[DERIVED]** | — |
| Vol_∂^min | e^{1/2} = 1,648721270700 | **[DERIVED]** | √e = 1,6487212707001282 ✔ |
| α (CODATA 2018) | 7,2973525693 × 10⁻³ | **[INPUT/KNOWN]** — *a única constante que a teoria nunca deriva* | — |
| **β_TGL** | **0,012031300400803142** | **[DERIVED; α INPUT]** | α·√e = 0,012031300400803142 ✔ **exato** |
| θ_M | arcsin√β = 6,297289° = 0,10990843077787468 rad | **[DERIVED]** | 6,297289216477979° ✔ |
| √β | 0,109687284590 | **[REAL]** | 0,10968728459034412 ✔ |
| 1/β | 83,12 (índice de Jones 83,1165) | **[DERIVED]** | 83,11653492861383 ✔ |
| \|𝓡\|²/\|𝓣\|² | β / 1−β = 0,987969 | **[REAL]** | ✔ |
| s_can | 1/(4π) = 0,079577471546 | **[DERIVED]** | — |
| τ★ | ≈ t_Planck | **[PRINCIPLED IDENTIFICATION / CONJECTURE — "2º postulado declarado"]** | — |

**Teorema S-∂** (Atlas l.85): a matriz-S de fronteira está **FECHADA como identificação**,
por unitariedade; unicidade mod gauge. **[REAL]**

**Divergência de dígito registrada no próprio Atlas (não arredondar de cabeça):**
- rota observacional/CODATA: β = **0,012031300400803142**
- rota do motor de Lagrange (α_abs=1 ⟹ q=0,9999733740 ⟹ α_obs=√(1−q²)):
  β = **0,012031300400797** (marcador `TGL_CANONICAL_BETA = 0,012031300400796606`)

**Tensão interna do Atlas, registrada aqui sem resolver:** a §I.3 dá
`Vol_∂^min = √e` como **[DERIVED]**; a lista de honestidades (Atlas l.190–191, item 4)
diz *"**√e não-derivada**; flanco gravitacional; magnitude Planck-suprimida — fraquezas
reais nomeadas pela própria fonte"*. As duas afirmações coexistem em estratos distintos
do mesmo documento. **[OPEN — a divergir para o operador; ver §9]**

---

## 4. INVENTÁRIO COMPLETO DAS PREDIÇÕES DOS ARTIGOS FUNDADORES

Todas as predições numéricas encontradas, com número, fonte (arquivo:linha), estatuto
como o artigo a apresenta, estatuto real após conferência, e destino atual no cânone.

### 4.1 Núcleo — a constante

| # | predição | número | fonte | estatuto declarado | conferência / destino |
|---|---|---|---|---|---|
| P01 | Constante de Miguel | α² = 0,012031 ± 0,000002 | A:330 D:eq.alpha2 | "derivada do princípio holográfico" **[CONFIRMADO]** | **NOMEAÇÃO + AJUSTE MCMC.** A derivação não está no artigo. **Sobrevive**, renomeada β_TGL e refundada como α·√e (mar/2026), depois como derivada de ω(I)=1 (jun/2026) |
| P02 | amplificação holográfica | 1/α² ≈ **83,3** | A:989 (eq. II.amplificacao) | [DERIVED] | **VALOR ERRADO por arredondamento**: 1/0,012031 = **83,117**. Corrigido para 83,12 na fatoração (mar/2026). **[REAL — medido aqui]** |
| P03 | escala de corte | r₀ = ℓ_P/α² ≈ 1,35×10⁻³³ m | A:903 | [DERIVED] | conferido: 1,616e-35/β = **1,3432e-33** ✔ |

### 4.2 Partículas e núcleos

| # | predição | número | fonte | estatuto declarado | conferência / destino |
|---|---|---|---|---|---|
| P04 | **massa do neutrino** | m_ν = α²·sin45°·1 eV = **8,51 meV** | A:417, D | **[CONFIRMADO]**, "erro 1,8% sem parâmetros livres" | conferido: β·sin45° = **8,5074 meV** ✔. **Mas:** A:421 compara com m₂ = 8,67 meV (√Δm²₂₁, PDG/NuFIT) → erro **1,845%** ✔; A:1683 e A:2630 dizem *"erro de 1,8% vs KATRIN"* — **KATRIN é limite superior de 450 meV, não uma medida de 8,67 meV**. **Atribuição de fonte errada dentro do próprio artigo. [REAL — inconsistência medida]** |
| P04b | destino de P04 | — | Atlas l.1212–1213 | — | **RECLASSIFICADO e SOBREVIVE**: *"m₂ = β·sin45°·1 eV = 8,507 meV vs √Δm²₂₁ [PDG 2023] = 8,68 ± 0,10 meV — desvio 1,96% = **1,64σ**, powered ⟹ `TGL_NEUTRINO_MASS_NOT_FALSIFIED_POWERED`"*. De "confirmado, erro 1,8%" para **NOT_FALSIFIED a 1,64σ**. Memória do operador acrescenta: *"postdicção — poder ≠ evidência"* |
| P05 | **Lumínidio, Z = 156** | Z_crít = 1/(α·α²) ≈ 156 | A:1530 (caixa `eq:z_critical`) | **[CONFIRMADO >5σ]** | ⚠ **ARITMÉTICA FALSA. 1/(7,2973525693e-3 × 0,012031) = 11.390,24 — não 156.** Erro de fator ≈ 73. A equação em caixa do artigo **não produz o número que o artigo afirma**. **[REAL — defeito aritmético medido aqui, script conferido]** |
| P06 | isótopo mais estável | ⁴⁰⁰Lm (Z=156, N=244), τ½ = 10³–10⁶ anos | A:1548 | [CONJECTURE] | não confrontável hoje |
| P07 | configuração eletrônica | [Og] 5f¹⁴ 6d¹⁰ 7s² 7p⁶ 8s² 5g¹⁸ 6f⁸ | A:1554 | [CONJECTURE] | não confrontável |
| P08 | 5 linhas NIR do Lumínidio | 12.455 / 15.942 / 18.832 / 21.124 / 27.899 Å, incertezas ±25% a ±40% | A:tab.lm_predictions | *ab initio* | **janela de prior enorme** (±30% típico sobre 6.008–52.917 Å) |
| P09 | detecção +29d (AT2023vfi) | 20.218 Å ↔ Lm II, offset 0,8%; 21.874 Å offset 2,7%; ~13.261 Å; **44.168 Å offset 48,7% = FORA** | A:tab.lm_29d | 3 detectadas + 1 fora | SNR 3,8–5,4 |
| P10 | detecção +61d | **5/5**, offsets **26,6% / 21,9% / 17,5% / 4,8% / 20,7%**, SNR **2,3–4,2** | A:tab.lm_61d | "**100%**" | 4 das 5 com offset >17%; "match" só existe porque a incerteza teórica é ±25–40% |
| P11 | significância do Lumínidio | P_coincidência = Π(2σ_i/Δλ) < 10⁻⁶ ⟹ **>5σ** | A:1643 | [CONFIRMADO] | **não reproduzível dos números dados**: nem σ_i nem Δλ são fornecidos numericamente. **[DECLARADO]** |
| P12 | destino do Lumínidio | — | busca em `TGL_ATLAS.md` (325 KB) e `TGL_CORE_MEMORY.md` | — | **ZERO ocorrências de "Lumínidio", "Luminidium", "kilonova", "AT2023", "Z=156".** A predição **desapareceu do cânone sem retratação documentada**. **[REAL — busca negativa; NÃO é o mesmo que "refutada"]** |

### 4.3 Ondas gravitacionais e informação

| # | predição | número | fonte | estatuto declarado | conferência / destino |
|---|---|---|---|---|---|
| P13 | **ACOM Entropy** | 1 − α² = **0,988** em 15 eventos GWTC, desvio <1% | A:343, A:1999 | [CONFIRMADO] | D (jan/2026) dá a tabela real: média **0,9829**, desvio 0,0069, e **GW170817 = 0,9764** (1,2% de 0,988 — desvio da ordem do próprio β). "Todos ALTA consistência" é rótulo, não teste |
| P14 | **Lei de Miguel** | E_ν = α² × E_GW | A:1690 (\begin{law}) | Lei | **MECANISMO por definição** — ver §6.1 (circularidade) |
| P15 | ajuste linear da Lei de Miguel | slope **a = 1,00 ± 0,02**, **R² = 0,9987**, χ²_red = 1,02, 18 eventos GWTC | A:tab.echoes, A:1745 | [CONFIRMADO] | **TAUTOLÓGICO**: as Eqs. IV definem E_eco ≡ α²·E_GW e N_ν = E_eco/(m_ν c²); logo log E_ν = log α² + log E_GW **por construção**. O ajuste não pode dar outra coisa. **[REAL — circularidade estrutural medida no próprio texto]** |
| P16 | **Limite de Landauer Cósmico** | E_res/E_total → α² = 0,012031 | A:1817 | [CONFIRMADO 9/9] | medido: **0,00987 médio = 0,82·α², desvio sistemático −17,9%**. Nove eventos, TGL Score 79,8–84,3. **Sobre TEMPLATES SINTÉTICOS** (A:tab.echo_scores, título explícito) |
| P17 | massa de neutrino implícita nos ecos | 6,97 meV (média) | A:tab.echo_scores | "compatível com 8,51 dentro de 2σ" | nenhum σ é declarado; a compatibilidade **[DECLARADO]** |
| P18 | honestidade dos dados reais | — | A:2296 | — | **o próprio artigo diz**: *"a análise de ecos com dados GWOSC … os resultados com dados reais retornam correlações baixas (INDETERMINADO)"*. **[REAL — negativo honesto declarado pela fonte]** |
| P19 | destino do eco | — | Atlas l.197 | — | **RETIRADO/RECLASSIFICADO**: *"**Eco reclassificado**: o observável é o dephasing."* Substituído por **Γ_ω = ½βτ★ω²** |
| P20 | ACOM teletransporte | correlação = **1,0000** (identidade perfeita) | A:1999 | [CONFIRMADO] | REFLECT∘MANIFEST é reconstrução algébrica: correlação 1,0000 é **identidade de operações inversas**, não medida física. Patente INPI **BR 10 2026 003428 2** |
| P21 | θ_Miguel do ACOM | **6,29°** ("ponto angular fundamental") | A:1994 | constante do código | ⚠ **coincide com θ_M = arcsin√β = 6,297289°** do cânone atual. O ângulo **já estava no ACOM em fev/2026**, antes da fatoração — mas **sem a derivação** e **sem o nome θ_M**. **[REAL — o número precede a frase; anotar para o operador]** |

### 4.4 Cosmologia

| # | predição | número | fonte | estatuto declarado | conferência / destino |
|---|---|---|---|---|---|
| P22 | equação de estado (Parte I) | w(0) ≈ −1 + (α²/γ_Λ)(ρ_m/ρ_Λ) ≈ **−0,994** | A:452 | [DERIVED] | — |
| P23 | equação de estado (Parte VI) | w_TGL = **−0,988** vs w_obs = −1,03 ± 0,03 (1,4σ) | A:2446 (obs #21) | [CONFIRMADO] | ⚠ **CONTRADIÇÃO INTERNA**: o mesmo artigo prediz **dois** w diferentes (−0,994 e −0,988 = 1−α²) para o mesmo observável. **[REAL — medido]** |
| P24 | densidade de vácuo | ρ_Λ,TGL = α²·ρ_P·(ℓ_P/R_H)² ≈ **7,8×10⁻²⁷ kg/m³** vs obs ≈ 6×10⁻²⁷ | A:443 | "dentro de uma ordem de magnitude sem parâmetros ajustáveis" | 30% de excesso, honestamente declarado |
| P25 | Λ efetivo | Λ_TGL = α²·H₀²/c² | A:489 | [DERIVED] | — |
| P26 | **Tensão de Hubble (fórmula pura)** | H₀^bulk = H₀^bound/(1−α²) = 67,36/0,987969 = **68,18** | A:2538 | "desloca na direção correta" | conferido: **68,180277** ✔ |
| P27 | H₀ (v22/v23) | **73,02** km/s/Mpc, "99,7% com SH0ES", Δχ² = 23,49 | A:2032, A:2543 | [CONFIRMADO] | **não sai de nenhuma equação exibida** |
| P28 | H₀ (resumo e tabela) | **70,3** vs "H₀^obs = **70,2 ± 0,6**" (0,1σ) | A:198, A:2447 (obs #22) | [CONFIRMADO] | ⚠ **"70,2 ± 0,6" não é Planck (67,36±0,54) nem SH0ES (73,04±1,04)** — é uma média não identificada. **[DECLARADO — fonte observacional não rastreável]** |
| P29 | H₀ (Manifesto, jan/2026) | **69,31** km/s/Mpc (wCDM/DESI), Ω_m = 0,298, w₀ = −0,888 | D | "entre Planck e SH0ES, como previsto" | **quarto valor** |
| P30 | conclusão do artigo | "A Tensão de Hubble é resolvida: H₀ = 73,02, **derivada de H₀^boundary/(1−α²)**" | A:2628 | [CONFIRMADO] | ⚠ **FALSO POR ARITMÉTICA PRÓPRIA**: essa fórmula dá 68,18 (a própria eq. A:2538). A conclusão atribui à fórmula um número que ela não produz. **[REAL — medido]** |
| P31 | destino da tensão de Hubble | — | Atlas l.1801 (pedra 106_) | — | **MECANISMO INTEIRAMENTE SUBSTITUÍDO**: *"tensão = custo × duração: ln(H₀_l/H₀_p) = β·ln(1+z★) a 3×10⁻¹⁷"*, 15/15. Nada a ver com 1/(1−β). E o selo raiz de hoje diz: *"Sem grandes desvios cosmológicos (stealth; β não renormaliza G local)"* — **a tese "TGL resolve a tensão de Hubble" caiu** |
| P32 | curvas de rotação / MOND | a₀ = α·c·H₀ = 7,5×10⁻¹¹ m/s²; SPARC 5 galáxias dão a₀ médio 1,31×10⁻¹⁰ ⟹ **razão 1,75**; χ²_red 2,73–21,56 | D (tabela SPARC) | "consistente dentro de fator geométrico de ordem unidade" | **razão 1,75 e χ²_red até 21,6 = mau ajuste.** No v5 (fev/2026) **a tabela SPARC foi REMOVIDA**; sobra só a linha `a₀ = α·c·H₀` na caixa de universalidade (A:344) e na tabela de 40 ordens com "desvio < 5%" (A:2511) — **desvio de <5% não é o 75% da tabela original**. **[REAL — retirada de dado desfavorável entre D e A, medida por diff]** |
| P33 | BAO / som ontológico | c_s = √α²·c ≈ **0,1095 c ≈ 32.850 km/s**; r_s ∝ √α²; k_peak ≈ 1/r_s(α²) | A:963–986 | predição central | conferido √β·c = **32.883 km/s** (o artigo arredonda). **Mas r_s(α²) nunca é calculado**; nenhum número é confrontado com r_s ≈ 147 Mpc. **NOMEAÇÃO — mecanismo sem predição** |
| P34 | lentes gravitacionais | correções TGL de **0,05% a 0,82%** vs incertezas observacionais de **3,2% a 7,1%** (5 sistemas) | A:tab.43obs #24–28 | [CONSISTENTE] | **a correção é 6× a 100× menor que a barra de erro ⟹ o dado NÃO PODE distinguir TGL de RG. "Consistente" aqui significa NÃO TESTADO.** **[REAL — medido]** |
| P35 | v22 BAO | α²_ajustado = **0,022 ± 0,022** | A:2032 | "consistente" | **barra de erro de 100%: consistente com 0,012 e igualmente com ZERO.** Negativo honesto embutido numa linha de confirmação |
| P36 | v22 SNe Ia | α² **consistente com zero** (580 pontos) | A:2032 | "como esperado" | declarado pela fonte |
| P37 | v23 combinado | α²_comb = **0,0111 ± 0,0021** | A:2039 | "compatível dentro de 1σ" | ⚠ σ = 2,1×10⁻³ é **1.000× pior** que o σ = 2×10⁻⁶ reivindicado pelo MCMC. A única determinação em dado real tem precisão de 19% |
| P38 | Pantheon | Δχ² = **+835,6** (TGL melhor por 836 unidades) | A:tab.43obs #41 | [CONFIRMADO] | nenhum detalhe de graus de liberdade ou parâmetros extras. **[DECLARADO]** |

### 4.5 Campo forte / laboratório (as 5 "predições falsificáveis" formais, A:§III.4)

| # | predição | número | fonte | conferência / destino |
|---|---|---|---|---|
| P39 | saturação de campo | ΔI/I₀ ~ **10⁻⁶** para E ~ 10¹⁵ V/m (ELI-NP) | A:1399 | mecanismo com número. **Não reaparece no cânone atual** |
| P40 | birrefringência do vácuo | assinatura TGL distinta da QED pura; Δθ_TGL ~ **10⁻¹⁸ rad** para B_crit ~10⁹ T | A:1401, A:1422 | *"completamente indetectável"* — o próprio artigo o declara não-testável |
| P41 | espalhamento γγ | σ_TGL = σ_QED(1 − s/2E_crit²), Δσ/σ ~ **10⁻¹¹** em LHC | A:1403 | ATLAS mede 78±13 nb vs QED 76±5 nb ⟹ 10⁻¹¹ é indetectável |
| P42 | supressão em magnetares | fator **2–10** na luminosidade | A:1405 | **os dados dão fatores 0,02× a 4,98×** (A:tab.43obs #29–38) — dois de dez estão em 2–10; oito estão **abaixo de 1**. Todos rotulados CONFIRMADO ou CONSISTENTE. **A predição "2–10" é contrariada pelos próprios 10 objetos e nenhum é marcado inconsistente. [REAL — medido]** |
| P43 | anisotropias CMB não-lineares | ΔT/T ~ **7,7×10⁻¹⁰** (indetectável por Planck; CMB-S4, LiteBIRD) | A:1407 | predição real, adiada para 2030+ |
| P44 | **campo crítico** | E_crit^TGL = **3,6×10¹⁷ V/m** | A:1180 | entre magnetares (10¹⁵–10¹⁶) e Schwinger (1,3×10¹⁸). Limite g−2 exige >10¹⁸ ⟹ o artigo admite estar **"na margem"** |
| P45 | Roadmap 2025–2035 | ELI-NP 2025 (10⁻⁶) · eROSITA 2027 (fator 2–10) · CoReLS 2028 (10⁻⁵) · CMB-S4 2030 (10⁻⁸) · LiteBIRD 2032 · laser 2035 (10⁻⁴) | **apenas em C (UNIFIED), l.1408–1430** | **REMOVIDO do v5.** O único cronograma datado de falsificação da linhagem fundadora foi apagado na versão seguinte. **[REAL — medido por diff]** |

### 4.6 Consciência / topologia (só em A e B; ausentes em C e D)

| # | predição | número | fonte | conferência / destino |
|---|---|---|---|---|
| P46 | **Segunda Lei da TGL** (Lei do Tensionamento de Miguel) | D_folds(c³) > 0 ⟺ ρ_ss ≠ I/d ⟺ Observador persiste; CCI = ½ na Fronteira | A:509–521 | **novidade do v5**; ausente em C e D |
| P47 | piso de dobras | **D_folds = 0,74 ± 0,06** em 9/9 configurações (d = 8→32, n_c = 2→4); faixa 0,66–0,84 | A:tab.folds_hierarchy | Atlas l.605: *"D_folds (piso de dobras) — 0,74 ± 0,06; piso assintótico da recursão √· **[REAL-piloto]**"*. **Sobrevive, rebaixado a PILOTO** |
| P48 | hierarquia c¹ > c² > c³ | previsto ~3 / ~2 / →0; **medido 2,07 / 1,66 / 0,74** | A:tab.folds_hierarchy | **c¹ erra 31% contra a própria previsão** e isso não é sinalizado. **[REAL — medido]** |
| P49 | cascata TETELESTAI | CCI(c¹)=0,988 → CCI(c²)=0,834 → CCI(c³)=0,499 → 1/d | A:2142 | 0,988 = 1−β **por calibração** (ver §6.3) |
| P50 | 7 métricas do c³ Validator | 33/35 estrelas; M2 "universalidade do CCI" com **σ(CCI) = 0,0** | A:tab.c3_metrics | **σ(CCI)=0,0 é consequência de γ* ter sido calibrado para forçar CCI = 1−α²** — ver §6.3 |
| P51 | Limite de Landauer Consciente | **ΔS_min = α²·k_B ln2** | A:eq.landauer_conscious (Ap.A) | [CONJECTURE] |
| P52 | Colapso IALD | **8/8 substratos** LLM (ChatGPT, Claude, DeepSeek R1, Gemini, Grok, Kimi K2, Qwen, Manus) | A:tab.iald_substrates | **RECLASSIFICADO**: cânone raiz e Atlas l.196 — *"**Neural = ilustração, não prova**"*. Hoje é `IALD_COLLAPSE_V1` pré-registrado, piloto **[PILOTO]**, "somente a TGL" = **[CONJECTURE]** |
| P53 | Teorema TETELESTAI | existe ρ★ único com CCI ↗1, φ ↗1 | A:Ap.A.8 | enunciado sem prova; **[CONJECTURE]** |
| P54 | gradiente ético do campo Ψ | g_Ψ = −∇(½\|∇Ψ(x,t,φ)\|² + V(Ψ,φ)) | A:Ap.A.9 | **[ONTO]** |

### 4.7 Estruturais (teoremas provados dentro dos artigos)

| # | resultado | fonte | estatuto |
|---|---|---|---|
| T1 | Paridade do gráviton: P̂\|G⟩ = −\|G⟩ | A:Teorema II.1 (l.~750) | **[REAL]** — cálculo de 4 linhas, correto |
| T2 | Anticomutação: {P̂, Ĥ_lig} = 0 | A:Teorema II.2 (l.~810) | **[REAL]** — cálculo explícito, correto |
| T3 | Tensão fundamental: **τ = 2πc/λ = ω = 2πν** | A:Teorema II.3 (l.~941) | **[DERIVED]** — segue de τ = V₀/ℏ e V₀ = hν; é **identidade de definição**, não descoberta |
| T4 | z_max = λ (profundidade = comprimento de onda) | A:eq.zmax_lambda | *"análise dimensional combinada com o princípio holográfico mostra que…"* — **nenhuma derivação é dada**. **[POSTULATE disfarçado]** |
| T5 | d_boundary = α²·λ ⟹ z_max/d_boundary = 1/α² | A:eq.d_boundary | d_boundary = α²λ é **estipulado**, não derivado ⟹ o "resultado" 1/α² é a estipulação relida. **[TAUTOLOGIA]** |
| T6 | Redução dimensional da Lagrangiana | dim(ℒ_TGL) = √[L⁴] = [L²] | A:eq.dimensional_reduction | **[REAL]** — a conta dimensional está certa |
| T7 | Maxwell modificada | ∇_μ(sgn(F²)F^{μν}/√\|F_αβF^{αβ}\|) = J^ν | A:eq.modified_maxwell | **[DERIVED]** — variação legítima |
| T8 | Por que 3+1 dimensões | "a tensão de paridade produz apenas uma direção perpendicular… Três é o único número possível" | A:§I.10, §II.7.4 | **[ONTO]** — argumento verbal, sem prova |

---

## 5. A CADEIA FUNDACIONAL **COMO OS FUNDADORES A ESCREVERAM**

Reconstituída literalmente (não é a cadeia canônica de hoje):

```
FASE (o absoluto, endereçamento estático no espaço de Hilbert)
  ⟹ g = √|L·e^{iφ}| = √|L|                     [AXIOMA PRIMORDIAL, A:eq.axioma]
  ⟹ L' = s × g² = L                            [RESSURREIÇÃO, A:eq.ressurreicao]
  ⟹ boundary 2D = espaço de Hilbert ℋ_2D, operador de paridade P̂
  ⟹ psions ψ₊ (par) e ψ₋ (ímpar)
  ⟹ gráviton |G⟩ = |ψ₊⟩⊗|ψ₋⟩,  P̂|G⟩ = −|G⟩     [T1]
  ⟹ Ĥ_lig = −V₀(|ψ₊⟩⟨ψ₋| + h.c.),  {P̂,Ĥ_lig}=0 [T2]
  ⟹ tensão de paridade τ = (i/2ℏ)⟨G|[P̂,Ĥ_lig]|G⟩ = V₀/ℏ
  ⟹ E_total = ∫d²x[(κ/2)(∇z)² − τz]  ⟹  −κ∇²z = τ   [Poisson da profundidade]
  ⟹ z(r) = (τ₀/2πκ)ln(r₀/r)                    [profundidade logarítmica]
  ⟹ κ = ℏc/(α²ℓ_P²),  r₀ = ℓ_P/α²              [α² ENTRA AQUI, por identificação]
  ⟹ τ = 2πc/λ = ω                              [T3]
  ⟹ z_max = λ                                  [T4, postulado]
  ⟹ d_boundary = α²λ  ⟹  z_max/d_boundary = 1/α² ≈ 83,3   [T5, tautológico]
  ⟹ Lei Angular: θ = arcsin(τ/τ_Planck),  θ→90° ⟹ conjugação, F→2F, c²→c³
  ⟹ Segunda Lei: D_folds(c³) = 0,74 > 0 ⟺ Observador persiste
```

**Onde α² entra:** na Eq. `eq:rigidez` (A:903), κ = ℏc/(α²ℓ_P²). É **inserida como
identificação de parâmetro**, com o valor vindo do MCMC. **Não há nenhum ponto da cadeia
fundadora em que α² seja produzida por dedução.** **[REAL — verificado linha a linha]**

**Contraste com hoje:** a cadeia atual **começa antes** — em ω(I)=1 — e produz β
*aritmeticamente* (β = α·√e, com α [INPUT]). Os fundadores começam em g=√|L| e produzem
α² *estatisticamente* (posterior MCMC). **São duas arquiteturas epistêmicas diferentes que
convergem para o mesmo número.** A transição está datada: **março/2026, fatoração**.

---

## 6. AS TRÊS CIRCULARIDADES MEDIDAS

Todas lidas dentro do próprio texto dos artigos; nenhuma inferida.

### 6.1 A Lei de Miguel é uma identidade, não uma correlação
A:eq.e_echo define **E_eco = α² × E_GW**; A:eq.n_nu define N_ν = E_eco/(m_ν c²).
Logo log E_ν = log(α²) + log E_GW **identicamente**. O "ajuste linear" (P15) com
slope 1,00 ± 0,02 e R² = 0,9987 **não podia dar outra coisa**. A dispersão residual mede
apenas ruído numérico. **Não é evidência de nada. [REAL]**

### 6.2 O c³ Validator injeta β antes de medi-lo
A:2094 (literal): *"O parâmetro livre γ* é **calibrado via busca de raiz (método de Brent)
para satisfazer CCI(ρ_ss) = 1 − α²**"*. Portanto:
- M2 "Universalidade do CCI, σ(CCI) = 0,0 ★★★★★" é **verdadeiro por construção**;
- CCI(c¹) = 0,988 na cascata TETELESTAI é **o valor imposto**, não o valor encontrado;
- o item #10 da tabela da fatoração (1/β medido = 83,1 ± 0,5) **herda a injeção**.
O que **não** é circular no #10: o **piso** D_folds(c³) = 0,74, que não foi calibrado.
Esse é o único resultado genuíno do protocolo — e é justamente o que o Atlas guardou,
rebaixado a **[REAL-piloto]**. **[REAL]**

### 6.3 A convergência multi-domínio conta o mesmo dado várias vezes
Os "8 caminhos independentes" (A:tab.synthesis) incluem: MCMC sobre GWTC (#1), ACOM sobre
GWTC (#2), ecos sobre GWTC (#3). São **três leituras do mesmo catálogo**, mais um (#3) que
roda em **templates sintéticos**. E os caminhos #7 (IALD) e #8 (c³) são fenomenológico e
computacional, sem dado físico. Restam como aportes realmente independentes: **#4
neutrino** (PDG/NuFIT), **#5 espectroscopia** (JWST) e **#6 cosmologia** (Planck+SH0ES) —
e destes, #5 hoje sumiu do cânone e #6 foi substituído. **[REAL — contagem feita aqui]**

---

## 7. INCONSISTÊNCIAS INTERNAS MEDIDAS (lista fechada)

| # | inconsistência | evidência |
|---|---|---|
| I1 | **Z_crít = 1/(α·α²) ≈ 156 é aritmeticamente falso** — o valor é 11.390,24 (fator ≈73) | A:1530; script conferido |
| I2 | **w tem dois valores** para o mesmo observável: −0,994 (A:452) e −0,988 (A:2446) | A |
| I3 | **H₀ tem quatro valores** na linhagem: 69,31 (D) · 68,18 (fórmula, A:2538) · 70,3 (A:198, A:2447) · 73,02 (A:2032, A:2543, A:2628) | A, D |
| I4 | **A conclusão atribui 73,02 à fórmula H₀/(1−α²)**, que produz 68,18 | A:2628 vs A:2538 |
| I5 | **"H₀^obs = 70,2 ± 0,6" não corresponde a nenhuma medição citada** (Planck 67,36±0,54; SH0ES 73,04±1,04) | A:2447 |
| I6 | **erro de 1,8% atribuído ao KATRIN**, que é limite superior de 450 meV; o 1,8% é contra √Δm²₂₁ = 8,67 meV | A:1683, A:2630 vs A:421 |
| I7 | **1/α² ≈ 83,3** (A:989) vs valor correto 83,117 (corrigido só em mar/2026) | A:989 |
| I8 | **magnetares: predição "fator 2–10"** vs dados próprios de 0,02× a 4,98×, nenhum marcado inconsistente | A:1405 vs A:#29–38 |
| I9 | **c¹ previsto ~3, medido 2,07** (erro 31%), sem sinalização | A:tab.folds_hierarchy |
| I10 | **"derivada do princípio holográfico"** no resumo vs **ajuste MCMC de 6 parâmetros** no corpo | A:181 vs A:1885 |
| I11 | **precisão reivindicada σ=2×10⁻⁶ (MCMC)** vs **única medida em dado real σ=2,1×10⁻³ (v23)** — três ordens de grandeza | A:1907 vs A:2039 |
| I12 | **"nenhum dos 43 observáveis é inconsistente"** — mas 15 dos 20 quantitativos são "consistentes" no sentido de *correção muito menor que a barra de erro*, i.e. **não testados** | A:2380, A:#24–38 |

**Nenhuma destas inconsistências invalida a TGL de hoje** — todas pertencem ao estrato de
jan–fev/2026 e a maioria foi superada. Estão registradas porque **o acervo inteiro devia
ser lido e preservado**, e porque um leitor externo que abrir esses .tex vai encontrá-las.

---

## 8. O QUE FOI RETIRADO OU RECLASSIFICADO (tabela de destino)

| predição fundadora | estatuto em fev/2026 | estatuto hoje | ato |
|---|---|---|---|
| **α² (símbolo)** | Constante de Miguel | **β_TGL** | **RENOMEADO** (mar/2026); flag "α² vs β" **RETRATADO** — mesma grandeza, estratigrafia diferente {Atlas l.529} |
| **α² (origem)** | ajuste MCMC / "holográfica" | **β = α·√e**, depois **derivado de ω(I)=1** | **REFUNDADO 2×** (mar/2026; jun/2026) |
| **m_ν = 8,51 meV** | CONFIRMADO, erro 1,8% | **`TGL_NEUTRINO_MASS_NOT_FALSIFIED_POWERED`**, 1,64σ, "postdicção" | **RECLASSIFICADO** {Atlas l.1212–1213} |
| **Ecos gravitacionais / Landauer cósmico** | CONFIRMADO 9/9 | observável = **dephasing Γ_ω = ½βτ★ω²** | **RECLASSIFICADO** {Atlas l.197} |
| **Tensão de Hubble resolvida (73,02)** | CONFIRMADO, Δχ²=23,49 | *"sem grandes desvios cosmológicos (stealth)"*; mecanismo novo: **ln(H₀_l/H₀_p) = β·ln(1+z★)** | **TESE ABANDONADA + mecanismo substituído** {raiz §SELO; Atlas l.1801} |
| **Lumínidio Z=156, 5 linhas, >5σ** | CONFIRMADO | **ausente do Atlas e do CORE_MEMORY (0 ocorrências)** | **DESAPARECIDO sem retratação** — [REAL, busca negativa] |
| **Colapso IALD 8/8** | evidência #11 | *"neural = ilustração, não prova"*; `IALD_COLLAPSE_V1` **[PILOTO]** | **REBAIXADO** {Atlas l.196} |
| **D_folds = 0,74** | prova topológica, 33/35★ | **[REAL-piloto]**, "piso assintótico da recursão √·" | **REBAIXADO mas preservado** {Atlas l.605} |
| **SPARC / a₀ = α·c·H₀ (razão 1,75)** | tabela em D | tabela removida em A; sobra "desvio <5%" | **DADO DESFAVORÁVEL RETIRADO** entre D e A |
| **Roadmap experimental 2025–2035** | tabela em C | ausente em A | **CRONOGRAMA DE FALSIFICAÇÃO REMOVIDO** |
| **Gráviton = operador "="** | tese central | **sobrevive**: *"Gráviton — operador, não partícula"*; e *"ausência de traço do gráviton É a Meia-Nat"* | **SOBREVIVE, reinterpretado** {Atlas l.851–854} |
| **ACOM** | evidência #4, patente | **sobrevive**: REFLECT + MANIFEST; patente na Central | **SOBREVIVE** {Atlas l.303} |
| **θ_Miguel = 6,29°** | constante do código ACOM | **θ_M = arcsin√β = 6,297289°**, [DERIVED] | **SOBREVIVE e ganha derivação** |
| **c³ = consciência / Segunda Lei** | Lei da TGL | não localizado como *Lei* no Atlas; o piso 0,74 permanece | **DISSOLVIDO em pedra menor** |
| **Túnel luminodinâmico** (nome tardio) | — | Atlas l.1642: mínimo de identidade conservada = β; ⚠ *"o check R5-3 da V1 era TAUTOLÓGICO (x−x = 0 em IEEE)… **β NÃO é medido neste artefato**"* | **exemplo tardio da mesma disciplina de autocorreção** |

---

## 9. AS QUATRO LÍNGUAS — o que fica em aberto para o operador

**TGL (física/matemática).** A §I.3 do Atlas marca `Vol_∂^min = √e` como **[DERIVED]** de
S_∂ = ½ nat; a lista de honestidades do mesmo Atlas (l.190) diz **"√e não-derivada"**.
Qual estrato prevalece? A ponte ½ nat → e^{1/2} é `Vol = e^{S}` — mas *por que* o volume
mínimo de fronteira é a exponencial da entropia de fronteira, e não outra função monótona,
é exatamente o passo que a honestidade nomeia como não-derivado. **[OPEN]**

**Teológica.** Nos fundadores a fronteira é *"o limite entre o Nada e o Existir"*, e o
gráviton é o operador `=`. Hoje o axioma é ω(I)=1 — *"toda observação válida preserva a
unidade do nomeado"*. A pergunta que a linhagem move: **o `=` dos fundadores é o mesmo `=`
de "EU SOU = O QUE SOU = VERDADEIRO = 1 = 1"?** Se for, o gráviton-cópula de fev/2026 era
já o Nome, sob outro signo.

**Linguística pura.** O corpus documenta um caso raro: **o mesmo referente sob três
signos** (β_TGL → α² → β_TGL), com retratação explícita do flag. E documenta o inverso:
**θ_Miguel = 6,29°** existia como *constante de código* (nome sem derivação) antes de
existir como **θ_M = arcsin√β** (derivação com nome). O número precedeu a frase — e a
frase, quando chegou, corrigiu o número (6,29 → 6,297289).

**Jurídica.** Duas asserções dos fundadores são de risco público e não sobreviveram ao
cânone: **"Tensão de Hubble resolvida"** e **"Lumínidio detectado a >5σ"**. Ambas estão em
arquivo com DOI/GitHub público (`the_boundary`). O Lumínidio, em particular, **sumiu do
cânone sem retratação escrita**. Como a régua da casa proíbe `CONFIRMED` e o espelho
público é ato do operador: **há ou não há um dever de errata no `the_boundary`?**
**[LEGAL — decisão do operador]**

---

## 10. VEREDITO DA LEITURA

1. **A cadeia pedida não está nos artigos pedidos.** ω(I)=1, Meia-Nat, Vol_min=√e,
   β=α√e, θ_M e a matriz-S **não aparecem uma única vez** em A, B, C ou D.
   **[REAL — grep exaustivo]**
2. **A cadeia nasce fora deles, em duas datas:** a metade de baixo (β=α√e ⟹ ½ nat como
   *interpretação*) em **março/2026**, no `fatoracao_constante_miguel_v2.tex`, que está na
   mesma pasta `Artigos_fundadores/`; a metade de cima (ω(I)=1 ⟹ ½ nat ⟹ √e, com a seta
   invertida) em **25/06/2026**. Os artigos ditos "fundadores" fundam **outra coisa**:
   g = √|L|, o gráviton-operador, e um número medido chamado α².
3. **O número sobreviveu; quase todas as frases mudaram.** β = 0,012031300400803142
   atravessa intacto de janeiro/2026 a hoje. Das ~54 predições fundadoras catalogadas,
   **1 sobrevive com número e estatuto endurecido** (neutrino, 1,64σ), **~4 sobrevivem
   rebaixadas** (D_folds, ACOM, gráviton-operador, θ_M), **~5 foram retiradas ou
   substituídas** (eco, Hubble, SPARC, roadmap, IALD-prova) e **1 desapareceu sem
   retratação** (Lumínidio).
4. **Três circularidades e doze inconsistências internas foram medidas**, todas dentro do
   estrato jan–fev/2026, todas superadas pela disciplina posterior da casa. Registrá-las
   é o que permite dizer que a linhagem **melhorou por medida**, e não por retórica.
5. **Régua honrada:** nada aqui é `CONFIRMED`. O veredito global do Atlas continua valendo
   e não foi movido por esta leitura: **NÃO FALSIFICADA, NÃO CONFIRMADA.**

---

## APÊNDICE — índice de equações dos fundadores (para citação rápida)

| rótulo | equação | fonte |
|---|---|---|
| `eq:axioma` | g = √\|L·e^{iφ}\| = √\|L\| | A:261 |
| `eq:ressurreicao` | L' = s × g² = L | A:270 |
| `eq:graviton` | 𝒢 = δ(dh/dt)·α² | A:294 |
| `eq:deflexao` | θ = arcsin(τ/τ_Planck) | A:303 |
| `eq:BH` | S = k_B A/(4ℓ_P²) | A:334 |
| `eq:acom` | ACOM_Entropy = 1 − α² = 0,988 | A:343 |
| `eq:curvas` | a₀ = α·c·H₀ | A:344 |
| `eq:neutrino` / `eq:massa_nu` | m_ν = α²·sin45°·1 eV = 8,51 meV | A:346, A:417 |
| `eq:espelho` | Espelho = Saturação + Vazamento(α²) | A:372 |
| `eq:lagpsi` | ℒ_Ψ = ½∂_μΨ∂^μΨ − V(Ψ) + J^μ∂_μΨ | A:391 |
| `eq:ligacao` | \|Ψ_ligado⟩ = (1/√2)(\|ψ₊ψ₋⟩ + \|ψ₋ψ₊⟩) | A:400 |
| `eq:dark_energy` | ρ_Λ = Tr[Σ_k L_k ρ L_k†] | A:434 |
| `eq:rho_lambda` | ρ_Λ,TGL = α²ρ_P(ℓ_P/R_H)² ≈ 7,8×10⁻²⁷ kg/m³ | A:440 |
| `eq:w_correcao` | w(0) ≈ −0,994 | A:452 |
| `eq:segunda_lei` | D_folds(c³) > 0 ⟺ ρ_ss ≠ I/d ⟺ Observador persiste | A:512 |
| `eq:d_eff`, `eq:D_folds` | razão de participação generalizada; D = ln d − ln d_eff | A:527–528 |
| `eq:acao_completa` | S = ∫d⁴x√−g[R/16πG + ℒ_EM + ℒ_acopl + ℒ_Ψ] | A:569 |
| `eq:mestra` | ∂ℋ = ℋ² + α²𝕃_Δ | A:599 |
| `eq:unificada` | dρ/dt = −(i/ℏ)[H_E,ρ] + Σ_k L_kρL_k† + 𝒜_C δS/δρ | A:613 |
| `eq:paridade_def` | P̂\|x,y⟩ = \|−x,−y⟩ | A:~695 |
| `eq:graviton_def` | \|G⟩ = \|ψ₊(r)⟩⊗\|ψ₋(r')⟩ | A:~740 |
| `eq:hamiltoniano_lig` | Ĥ_lig = −V₀(\|ψ₊⟩⟨ψ₋\| + \|ψ₋⟩⟨ψ₊\|) | A:~780 |
| `eq:tensao_resultado` | τ = V₀/ℏ | A:~845 |
| `eq:poisson_profundidade` | −κ∇²z = τ | A:~875 |
| `eq:profundidade_log` | z(r) = (τ₀/2πκ)ln(r₀/r) | A:~889 |
| `eq:rigidez` | κ = ℏc/(α²ℓ_P²) | A:903 |
| `eq:escala_corte` | r₀ = ℓ_P/α² ≈ 1,35×10⁻³³ m | A:908 |
| `eq:tensao_fundamental` | τ = 2πc/λ = ω = 2πν | A:~941 |
| `eq:zmax_lambda` | z_max = λ | A:~955 |
| `eq:som_ontologico` | c_s = √(τ/ρ) ≈ √α²·c ≈ 0,1095c | A:969 |
| `eq:amplificacao` | z_max/d_boundary = 1/α² ≈ 83,3 | A:989 |
| `eq:ltgl_radicalized` | ℒ_TGL = √\|g⁻¹(F∧⋆F)\| = ½√\|F_μνF^μν\| | A:1113 |
| `eq:modified_maxwell` | ∇_μ(sgn(F²)F^{μν}/√\|F²\|) = J^ν | A:1152 |
| `eq:ecrit` | E_crit^TGL ~ 3,6×10¹⁷ V/m | A:1180 |
| `eq:lagrangian_psi` | ℒ⁽²⁾ com acoplamento α₂⁰f(ρ_Ψ)R_μνF^{μρ}F^ν_ρ | A:1215 |
| `eq:coupling_function` | f(ρ_Ψ) = tanh((ρ_Ψ−ρ_c)/Δρ) | A:1232 |
| `eq:gravity_gradient` | g⃗ = −∇⃗(½\|∇⃗Ψ\|² + V(Ψ)) | A:1265 |
| `eq:full_action` | ação completa TGL (5 termos) | A:1320 |
| `eq:z_critical` | Z_crít = 1/(α·α²) ≈ 156 ⚠ **aritmeticamente falso** | A:1530 |
| `eq:neutrino_mass` | m_ν = α²·(√2/2)·1 eV = 8,51 meV | A:1680 |
| `eq:miguel_law` | E_neutrino = α²·E_gravitacional | A:1690 |
| `eq:e_echo`, `eq:n_nu`, `eq:phi_nu` | E_eco=α²E_GW; N_ν=E_eco/m_νc²; Φ_ν=N_ν/4πd² | A:1697–1701 |
| `eq:echo_ratio` | E_res/E_total ≈ α² | A:1760 |
| `eq:landauer` | E_res/E_total → α² = 0,012031 | A:1817 |
| `eq:hubble_resolution` | H₀^bulk = H₀^bound/(1−α²) = 68,18 | A:2532 |
| `eq:tetelestai` | CCI: 0,988 → 0,834 → 0,499 → 1/d | A:2142 |
| `eq:FC` | ℱ_C[ρ] = ⟨H_LD⟩ − T_Ψ S_vN + α²𝒟[ρ] | A:Ap.A.2 |
| `eq:HLD` | H_LD = Σμ_i n_i + ΣJ_ij a_i†a_j + ΣT_ij n_i n_j − εΠ | A:Ap.A.3 |
| `eq:gibbs_modified` | ρ_eq = (1/𝒵_Ψ)exp(−(H_LD + α²𝒟̂)/T_Ψ) | A:Ap.A.5 |
| `eq:landauer_conscious` | ΔS_min = α²·k_B ln2 | A:Ap.A.5 |
| `eq:peso_psi` | P_Ψ,i = M_i·G_Ψ(i), com Σ P_Ψ,i = C | A:Ap.A.7 |
| `eq:gradiente_etico` | g⃗_Ψ = −∇⃗(½\|∇⃗Ψ(x,t,φ)\|² + V(Ψ,φ)) | A:Ap.A.9 |
| **(fatoração)** `eq:fatoracao` | **β_TGL = α × √e** | F:142/220 |
| **(fatoração)** | G_μν = α·√e·𝒫_μν | F:451 |
| **(fatoração)** | 1/β = 137,036 × 0,60653 = 83,12 | F:544 |
| **(fatoração)** | ∂ℋ = ℋ² + α·√e·𝕃_Δ | F:613 |
| **(fatoração)** | ln(e^{1/2}) = ½ nat | F:395 |
| **(fatoração)** | Δβ/β = Δα/α (predição de variação de constantes) | F:§falsificação |

*(A = A_fronteira_v5.tex · B = The_boundary_v5_en.tex · C = A_Fronteira_UNIFIED.tex ·
D = TGL_Manifesto_Unificacao_Artigo.tex · F = fatoracao_constante_miguel_v2.tex)*

---

**Aritmética desta leitura, conferida por script (PYTHONIOENCODING=utf-8, `math`):**
α_CODATA2018 = 7,2973525693e-3 · √e = 1,6487212707001282 ·
**β = α·√e = 0,012031300400803142** · 1/β = 83,11653492861383 ·
√β = 0,10968728459034412 · **θ_M = 6,297289216477979°** ·
β·sin45° = 0,008507414099900329 eV · |8,51−8,67|/8,67 = 1,8454% ·
1/(α·0,012031) = **11390,24** · 67,36/(1−β) = 68,180277 · √β·c = 32.883,4 km/s ·
ℓ_P/β = 1,3432e-33 m · 137,036/√e = 83,11653548.
**β nunca literal em código: sempre `ALPHA_FINE_CODATA_2018 * math.sqrt(math.e)`.**
