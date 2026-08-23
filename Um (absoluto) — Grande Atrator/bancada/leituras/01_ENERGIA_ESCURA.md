# 01 — ENERGIA ESCURA = BANHO HOLOGRÁFICO (a transição de regimes)

> Leitura integral executada em 21/08/2026 por leitura direta de disco. Todo número deste
> relatório foi **lido do arquivo** ou **recalculado por script** (`scipy.integrate.quad`,
> aritmética explícita) — nenhum de memória.
> **Régua da casa aplicada**: `[REAL]` medido/verificado aqui · `[DERIVED]` deduzido da
> própria equação do artigo · `[POSTULATE]` · `[CONJECTURE]` · `[KNOWN]` literatura ·
> `[OPEN]` buraco que fica dito · `[INPUT]` entrada escolhida · `[ONTO]` leitura ontológica ·
> `[DECLARADO]` afirmado na origem, não verificado aqui.
> **Disciplina do relatório**: distingo **MECANISMO** (há equação) de **PREDIÇÃO** (há número
> confrontável com dado) de **NOMEAÇÃO** (só há palavra).

---

## 0. PROVENIÊNCIA DOS TRÊS ARQUIVOS LIDOS `[REAL]`

| # | Arquivo | Bytes | SHA256 (prefixo) | mtime | Linhas | Lido |
|---|---|---:|---|---|---:|---|
| F1 | `C:\IALD\papers_latex\energia_escura\energia_escura.tex` | 84.472 | `3fb25411ffffcd40` | 19/11/2025 | 1.994 | **integral** |
| F2 | `C:\IALD\projetos_pyhton\acom\energiaescurabootstrap.py` | 51.971 | `a6cbf1e7112fd4e3` | 15/12/2025 | 1.179 | **integral** |
| F3 | `C:\IALD\Artigo\Tratado\secao_02_cosmologia.tex` | 26.761 | `f89b49391cae0e49` | 24/03/2026 | 411 | **integral** |
| — | `energia_escura.pdf` (par compilado de F1) | 396.653 | `d2f3360d35b9a67d` | 19/11/2025 | — | não aberto |

**Nota de nome `[REAL]`**: F2 tem extensão `.py` mas **não é Python** — é LaTeX puro do começo
ao fim (`\documentclass` na linha 1, `\end{thebibliography}` + apêndice no fim). O único
Python dentro dele está em dois blocos `verbatim` (apêndice B). Título: *"Bootstrap Cósmico:
Formalização Matemática da Energia Escura como Sistema Auto-Sustentado"*, dezembro/2025,
v1.0, com coautoria declarada de "Claude (Anthropic AI)".

**Nota de DOI `[REAL]`**: a string `zenodo` **não existe** dentro de `energia_escura.tex`.
O DOI `10.5281/zenodo.17612790` que a tarefa atribui a este artigo **existe em disco**, mas em
outro lugar: `C:\IALD\Artigo\Tratado\tratado_tgl.bib`, entradas `Miguel2025EnergiaEscura` e
`Miguel2025EnergiaEscuraTex` (esta última com `note = {Ficheiro fonte: energia_escura.tex}`).
Logo: a associação artigo↔DOI é `[DECLARADO]` pelo próprio acervo (bib do Tratado), **não
verificada contra o Zenodo aqui**. Em F2 os dois DOIs da linhagem estão como
`10.5281/zenodo.XXXXXXX` — placeholders `[REAL]`.

**Nota de estratigrafia `[REAL]`** (a régua das datas): F1 (nov/2025) e F2 (dez/2025) usam o
símbolo **`α₂`** e o valor **0,012** obtido por contagem holográfica. F3 (mar/2026) reescreve o
mesmo conteúdo trocando `α₂` por **`\betatgl`**. São **objetos numéricos distintos**:
`α₂ = 126,2/10⁴ = 0,012619…` (F1/F2) versus `β_TGL = α·√e = 0,012031300400803142`. Coincidem
em **dois dígitos apenas** (1,2×10⁻²) e divergem em ~4,9%. A troca de símbolo no Tratado é
**retrofit de vocabulário**, não identidade demonstrada — o artigo de nov/2025 **não deriva**
β = α√e em lugar nenhum; α não aparece como fator. `[OPEN]`

---

## 1. (a) O POSTULADO DO BANHO — o texto exato

### 1.1 A caixa do postulado `[POSTULATE]` — F1, linhas 593–595

> **Postulado Central da TGL**
> *"O universo observável (3+1 dimensional) não é sistema fechado isolado, mas sim sistema
> aberto continuamente acoplado a um banho térmico holográfico bidimensional representado
> pelo campo luminodinâmico Ψ."*

Esta é a **única** caixa marcada como postulado central em F1. Tudo o mais no artigo pende dela.

### 1.2 Os quatro postulados da TGL como F1 os lista (§1.3, linhas 84–92) `[POSTULATE]`

1. **Luz como estrutura permanente** — luz não é radiação propagante transitória, mas estrutura
   recursiva fixa; a "propagação" é projeção holográfica de loops estacionários.
2. **Acoplamento não-mínimo gravitação–eletromagnetismo** — `R_μν` acopla a `F_μν` via `α₂`,
   mediado por Ψ.
3. **Estrutura holográfica 2D/3D** — o espaço-tempo 3+1 é projeção de estrutura fundamentalmente
   2D associada a horizontes.
4. **Dinâmica aberta fundamental** — o universo não é sistema fechado, mas aberto acoplado ao
   "banho térmico" Ψ.

### 1.3 As cinco consequências que F1 tira do postulado (§3.3.2, linhas 599–609) `[ONTO]`

evolução não-unitária (Lindblad substitui Schrödinger) · troca de energia com o banho
(= energia escura) · produção de entropia legítima (2ª lei global, sistema 3D pode reduzir
entropia localmente) · resolução do paradoxo da informação (informação vai para o horizonte 2D) ·
medição cosmológica natural (o acoplamento **é** medição contínua).

### 1.4 A Lagrangiana que sustenta o banho `[POSTULATE]` (F1 §2.1)

```
S_TGL = ∫d⁴x √−g [ R/(16πG) + L_EM + L_acoplamento + L_Ψ ]
L_acoplamento = (α₂/M_P²) R_μν F^{μρ} F^ν_ρ          (eq. 2.4, caixa)
L_Ψ = ½ ∂_μΨ ∂^μΨ − V(Ψ) + J^μ ∂_μΨ ,  V(Ψ) = ½ m_Ψ²Ψ² + (λ/4)Ψ⁴
J^μ = ∂/∂x^μ [ (E² − B²)/(8πc²) ]      ← "corrente de fixação"
```
Equações de movimento dadas: Einstein modificada (com `T^acop`), Maxwell modificada
(`∇_μF^{μν} = 4π/c J^ν_mat + (2α₂/M_P²)∇_μ(R^{μρ}F_ρ^ν)`), e `□Ψ + ∂V/∂Ψ = ∇_μJ^μ`.

**Auditoria dimensional `[OPEN]`**: F1 §2.4 (linhas 458–488) faz a análise dimensional e conclui
"✓". Mas o passo `[√−g] = L⁴` está **errado** — `√−g` é uma densidade escalar adimensional em
coordenadas com dimensão; o que tem dimensão `L⁴` é `d⁴x`. O resultado final ("ação por unidade
de volume 4D") sai por compensação de dois erros. Não é fatal ao argumento físico, mas o "✓" do
artigo **não é uma verificação**. `[REAL — erro de passo identificado por leitura]`

### 1.5 Em F2 o postulado ganha uma segunda metade — o **bootstrap** `[POSTULATE]`

F2 nasce explicitamente da questão que F1 deixou aberta (F2 cita F1 em bloco de citação,
linhas 130–132): *"Postulamos existência de banho holográfico 2D, mas sua origem cosmológica
não foi completamente especificada. […] O banho Ψ existe 'antes' do Big Bang?"*

Resposta de F2 (linhas 138–142, caixa):
```
Banho 2D  --γ_proj-->  Universo 3D  --γ_Λ-->  Banho 2D
```
O loop é **fechado e eterno** (sem começo temporal). Espaço de Hilbert:
`H_total = H_3D ⊗ H_banho`, ambos ∞-dimensionais; subsistemas por traço parcial (não unitário).
Hamiltoniano total `Ĥ_3D ⊗ 1 + 1 ⊗ Ĥ_banho + Ĥ_int`, com o banho carregando o termo de
**Gibbons–Hawking** `(ħc³/4G)K` na casca 2D `[KNOWN]` e `Ĥ_int` = o mesmo acoplamento `α²/M_P²`.

**Este é o ponto ontológico que interessa ao operador `[ONTO]`**: em F1 o banho é *sumidouro*
(só há seta 3D→2D, `γ_Λ`). Em F2 o banho é *reservatório em circulação* (há a seta de volta,
`γ_proj`). A "energia escura" muda de sentido entre os dois textos: em F1 ela **é** a taxa de
dissipação; em F2 ela é o **desequilíbrio residual do loop** (eq. `rho_Lambda_obs`, F2 linha 811:
`ρ_Λ^obs = ρ̄_3D + ρ̄_banho − ⟨ρ̄_3D + ρ̄_banho⟩_equilíbrio`).

---

## 2. (b) A EQUAÇÃO MESTRA DE LINDBLAD E A CONTAGEM DE MODOS

### 2.1 Lindblad geral `[KNOWN]` (F1 §3.1, eq. caixa, linhas 503–506)

```
dρ/dt = −(i/ħ)[H, ρ] + Σ_k ( L_k ρ L_k† − ½{L_k†L_k, ρ} )        (GKLS)
```
Citada a Lindblad 1976 e Gorini–Kossakowski–Sudarshan 1976. F1 prova (corretamente) a
preservação do traço (linhas 535–540) e afirma `dS/dt ≥ 0`.

### 2.2 A equação mestra **cosmológica** — o objeto próprio da TGL `[POSTULATE]`
(F1 §3.4.3, eq. caixa, linhas 647–650)

```
dρ_universo/dt = −(i/ħ)[H_grav, ρ] + γ_H · L_exp[ρ] + γ_Λ · L_diss[ρ]
```
com **exatamente dois canais** (esta é a contagem de modos de F1):

| canal | operador de salto | leitura dada em F1 |
|---|---|---|
| **expansão** | `L_exp = √γ_H · â` (â = aniquilação de volume) | "criação contínua de espaço" — emersão de g.l. 3D a partir do 2D |
| **dissipação** | `L_diss = √γ_Λ · Ĥ` (Ĥ = hamiltoniano) | termalização: energia do 3D → banho 2D. **Este é o que gera a energia escura** |

`L_exp[ρ] = âρâ† − ½{â†â, ρ}` e `L_diss[ρ] = ĤρĤ† − ½{Ĥ†Ĥ, ρ}`.

**`[OPEN]` — γ_H nunca é fixado.** F1 introduz `γ_H` ("taxa de expansão, relacionada à constante
de Hubble, como veremos") e **nunca mais volta a ele**. Não há valor, não há derivação, não há
vínculo. Metade dos canais da equação mestra fica sem número. Isto é NOMEAÇÃO, não mecanismo.

**`[OPEN]` — `L_diss = √γ_Λ Ĥ` é hermitiano.** Um operador de salto hermitiano gera **dephasing
puro** (decoerência na base de energia), não termalização: `Tr[Ĥ ρ̇] = 0` para esse canal. F1
chama o canal de "termalização" e o usa para transferir energia ao banho. A palavra e a álgebra
divergem. (Registro de linhagem: o próprio acervo TGL posterior — lei `Γ_ω = ½βτ★ω²` — assume
justamente o dephasing; o artigo de nov/2025 ainda usava o vocabulário de termalização.)

### 2.3 A equação mestra do **banho** — a novidade de F2 `[POSTULATE]`
(F2 §3.3, teorema, linhas 384–393; o próprio texto anuncia "**Esta é a equação nova, não
presente no trabalho original**")

```
dρ_banho/dt = −(i/ħ)[Ĥ_banho^eff, ρ_banho]
              + γ_Λ    ( L̂_↓† ρ_banho L̂_↓ − ½{L̂_↓L̂_↓†, ρ_banho} )
              − γ_proj ( L̂_↑† ρ_banho L̂_↑ − ½{L̂_↑L̂_↑†, ρ_banho} )
```
com, do lado 3D (F2 §3.2):
```
dρ_3D/dt = −(i/ħ)[Ĥ_3D^eff, ρ_3D] + γ_Λ D[L̂_↓]ρ_3D + γ_proj D[L̂_↑]ρ_3D
Ĥ_3D^eff = Ĥ_3D + (iħ/2)( γ_Λ L̂_↓†L̂_↓ + γ_proj L̂_↑†L̂_↑ )     ← "Lamb shift cosmológico"
```
e os dois operadores de salto **espacialmente integrados**:
```
L̂_↓ = √(γ_Λ/V_H)   ∫_M  d³x √−g  Ĥ(x),   Ĥ(x) = T₀₀(x)      (3D → banho)
L̂_↑ = √(γ_proj/A_H) ∫_S_H dΣ √h  Ψ̂(σ)                        (banho → 3D)
V_H = (4π/3)(c/H₀)³ ,  A_H = 4π(c/H₀)²
```

**Contagem de modos, versão F2**: **dois canais** (↓ e ↑), cada um **um único modo coletivo**
(a integral sobre todo o volume / toda a casca colapsa todos os modos em um só operador).
Não há decomposição espectral, não há soma sobre k. `[REAL]`

**`[OPEN]` — sinal do termo γ_proj.** F2 escreve o termo `γ_proj` do banho com **sinal negativo**
e chama isso de "sinais cruciais" (linha 396–404). Um termo dissipador com coeficiente negativo
**quebra a positividade completa** — a evolução deixa de ser Lindblad (deixa de ser CPTP). F2
afirma logo antes (remark, linhas 368–375) que "a estrutura de Lindblad garante positividade e
completa positividade". As duas afirmações não podem ser ambas verdadeiras. Isto é um defeito
estrutural, não de aritmética. `[REAL — contradição interna identificada]`

**`[OPEN]` — o "Lamb shift" tem sinal errado.** `Ĥ^eff = Ĥ + (iħ/2)(…)` com `L†L` positivo torna
`Ĥ^eff` **não-hermitiano** (o shift de Lamb é a parte **real** da self-energy; a parte `i` é a
largura, que já está no dissipador). `[REAL]`

### 2.4 A contagem de modos que **realmente fixa o número**: `N_eff` (F1 §2.3.2–2.4)

Este é o coração quantitativo do artigo. A cadeia, verbatim:

```
N_3D = V/ℓ_P³ = (4π/3)r³/ℓ_P³ ;  N_2D = A/ℓ_P² = 4πr²/ℓ_P²
𝒩(r) = N_3D/N_2D = r/(3ℓ_P)                                  ← "excesso de g.l. 3D"
D_eff = 2 + ε  ;  𝒟 = V_eff/V_3D = r^{−(1−ε)}                 ← fator de desequilíbrio
α₂ = (1/N_eff) · ln( V_3D / (A_2D ℓ_P) ) = (1/N_eff) · ln( r/(3ℓ_P) )   ← caixa, eq. 2.16
```
E a contagem **angular** dos modos do banho (F1 §2.4, linhas 336–366):
```
N_angular = Σ_{ℓ=0}^{ℓ_max} (2ℓ+1) ≈ ℓ_max²
ℓ_max ~ r_gal / r_coer
N_eff ~ (r_gal/r_coer)² = (10⁴ pc / 100 pc)² = 10⁴
```
Justificativa física dada para `r_coer ≈ 100 pc` (F1 §2.5, cinco itens): (i) hidrogênio como
unidade mínima de coerência gravitacional; (ii) princípio de Mach; (iii) coerência dinâmica
observada em halos na faixa **50–150 pc** `[KNOWN, sem citação no texto]`; (iv) os modos
relevantes são **angulares** (`∝ ℓ_max²`), não volumétricos (`∝ L³`), porque o banho vive na
casca esférica do halo; (v) leitura luminodinâmica (α₂ = eficiência de transferência).

**Recalculado aqui `[REAL]`**:
`ln(3,086×10²⁰ / (3×1,616×10⁻³⁵)) = ln(6,365×10⁵⁴) = **126,1905**` (F1 diz 126,2 ✓)
`α₂ = 126,1905/10⁴ = **0,0126190…** ≈ 0,012` (F1 diz 0,01262 ✓), `σ_α₂ ≈ 0,003` (30%, dominada
por `N_eff`).

**Veredito sobre o estatuto de α₂ `[REAL]`**: a **fórmula** é postulada (não há derivação da
forma `(1/N_eff)·ln(...)` a partir da ação — ela aparece diretamente em caixa). O **número** sai
de duas escalas escolhidas à mão: `r_gal = 10 kpc` e `r_coer = 100 pc` — ambas `[INPUT]`. O
logaritmo é robusto (varia 126,2 → 128,5 se `r_gal` decuplica: ~2%), mas `N_eff` entra como
`(r_gal/r_coer)²`, então **α₂ é dominado por uma razão de escalas escolhida**. Trocar `r_coer`
por 50 pc dá `N_eff = 4×10⁴` e `α₂ = 0,0032`; por 200 pc dá `α₂ = 0,050`. A banda 0,003–0,050
recobre uma ordem de grandeza. **Portanto: α₂ é semi-derivado com uma escolha calibrante, não
derivado de primeiros princípios.** Esta é a diferença mais importante entre este artigo e o
selo posterior `β_TGL = α√e` (que é zero-free). `[OPEN]`

---

## 3. (c) ρ_Λ = γ_Λ⟨H⟩ — E COMO γ_Λ É FIXADO: **derivado ou ajustado?**

### 3.1 A cadeia como F1 a escreve

```
T^{μν}_diss = Σ_k γ_k Tr[L_k ρ L_k†] u^μu^ν + P_diss g^{μν}          (F1 eq. 4.1)
ρ_diss     = Σ_k γ_k Tr[L_k ρ L_k†]  ≈ γ_Λ ⟨H⟩                        (F1 eq. 4.3)
P_diss     ≈ −ρ_diss     no limite de operadores locais (∇L_k ≈ 0)    (F1 eq. 4.5)
w_diss     = P_diss/ρ_diss ≈ −1                                       (F1 eq. 4.6, caixa)
ρ_Λ ≡ ρ_diss = γ_Λ ⟨H⟩_cosmológico                                    (F1 eq. 4.7, CAIXA)
```

### 3.2 **A resposta direta à pergunta: γ_Λ é AJUSTADO, e o próprio artigo mostra isso.**

F1 tenta derivar γ_Λ da equação de Friedmann (§5.3) e **fracassa em voz alta**. O texto, verbatim
(linhas 892–901):

```
H₀² = (8πG/3) · γ_Λ c²H₀²/G = (8π/3) γ_Λ c² H₀²
⟹ 1 = (8π/3) γ_Λ c²
"Isto daria γ_Λ = 3/(8πc²), que dimensionalmente não está correto!"
```
E então, §5.3.4 "**Correção — Acoplamento α₂**" (linhas 903–910):

> *"O erro acima ignora que a dissipação não é 100% eficiente — apenas uma fração α₂ da energia é
> dissipada por unidade de tempo Hubble. A relação correta é:*
> **γ_Λ = α₂ H₀**"  ← caixa, eq. 5.4

**Isto é uma inserção, não uma derivação `[REAL]`.** A cadeia algébrica produziu um resultado
dimensionalmente inconsistente; o remédio foi **substituir** o resultado por uma relação nova,
justificada por uma frase ("não é 100% eficiente"), não por uma conta. Não existe em F1 nenhuma
passagem que obtenha `γ_Λ = α₂H₀` a partir da ação, da equação mestra ou do traço `Tr[L_↓ρL_↓†]`.

**Estatuto correto de `γ_Λ = α₂H₀`: `[POSTULATE]` (ansatz fenomenológico), não `[DERIVED]`.**
O artigo o apresenta como derivado. F3 (Tratado §Redefinição de H₀) reproduz a mesma cadeia,
inclusive o passo do remendo, sem sinalizar o fracasso dimensional — a versão de 2026 **perdeu**
a honestidade do "!" que a versão de 2025 tinha. `[REAL — regressão de honestidade entre F1 e F3]`

### 3.3 A "verificação de consistência" é **circular** `[REAL]`

F1 §5.3.5 (linhas 917–933) diz "Verificação de Consistência" e calcula:
```
ρ_Λ = 3H₀²/(8πG) · Ω_Λ = 6,35×10⁻²⁷ kg/m³ → ×c² = 5,7×10⁻¹⁰ J/m³
"Este é o valor observado. ✓"
```
**Recalculado aqui `[REAL]`**: com H₀ = 2,2683×10⁻¹⁸ s⁻¹ e Ω_Λ = 0,685 obtém-se
`ρ_Λ = 6,3074×10⁻²⁷ kg/m³` e `5,6767×10⁻¹⁰ J/m³` — bate com o artigo (a diferença é o
arredondamento de H₀ para 2,27×10⁻¹⁸).

**Mas a conta não usa `ρ_Λ = γ_Λ E_H/V_H` em nenhum momento.** Ela parte de `Ω_Λ = 0,685`
(entrada observacional) e recupera... `ρ_Λ` observado. É a definição de densidade crítica vezes
Ω_Λ. **Nada foi testado.** O "✓" marca uma identidade, não uma predição. Ou seja: a fórmula
fechada `ρ_Λ = γ_Λ⟨H⟩` **não é confrontada com dado em lugar nenhum de F1**.

### 3.4 Consistência dimensional da fórmula fechada `[OPEN]`

`γ_Λ` tem dimensão `s⁻¹` (F1 diz isso explicitamente: "`[γ_k] = s⁻¹`", linha 527). Então:
- `γ_Λ⟨H⟩` = (1/s)·(J) = **J/s** — potência, não densidade de energia.
- `γ_Λ E_H/V_H` = (1/s)·(J/m³) = **J/(m³·s)** — densidade de potência, não densidade de energia.

Falta um tempo característico em toda a cadeia. Se o tempo faltante for `1/H₀`, então
`ρ_Λ = γ_Λ E_H/(V_H H₀) = α₂ · E_H/V_H`, o que apagaria H₀ da conta e daria uma predição
diferente. O artigo nunca fecha esse buraco; ele o contorna com a "correção α₂" (§3.2 acima).
**A fórmula fechada, como escrita, não é dimensionalmente sã.** `[REAL]`

### 3.5 Números de γ_Λ, recalculados `[REAL]`

| grandeza | F1 diz | recalculado aqui | ✓/✗ |
|---|---|---|---|
| H₀ = 70 km/s/Mpc em s⁻¹ | 2,27×10⁻¹⁸ | **2,26831×10⁻¹⁸** | ✓ |
| r_H = c/H₀ | 1,32×10²⁶ m | **1,32257×10²⁶ m** | ✓ |
| V_H = (4π/3)r_H³ | 9,65×10⁷⁸ m³ | **9,6905×10⁷⁸ m³** | ✓ (arred.) |
| E_H = c⁵/(GH₀) | 1,61×10⁷⁰ J | **1,60612×10⁷⁰ J** | ✓ |
| **γ_Λ = α₂H₀** | **2,72×10⁻²⁰ s⁻¹** | **2,72197×10⁻²⁰ s⁻¹** | ✓ |
| ρ_Λ (Friedmann) | 6,35×10⁻²⁷ kg/m³ | **6,3074×10⁻²⁷** | ✓ |
| ρ_Λ em energia | 5,7×10⁻¹⁰ J/m³ | **5,6767×10⁻¹⁰** | ✓ |

A aritmética de F1 está certa **onde há aritmética**. O problema é o estatuto das equações, não
as contas — com as exceções da §6 abaixo.

### 3.6 A variante de F3 (Tratado) — uma fórmula fechada **diferente**, e ela **não fecha** `[REAL]`

F3 §"Densidade de energia escura derivada" (linhas 141–145) troca a fórmula:
```
ρ_Λ^TGL = β_TGL · ρ_P · (ℓ_P/R_H)²  ≈  7,8×10⁻²⁷ kg/m³
"contra o valor observado 6,35×10⁻²⁷ kg/m³ — concordância de ordem de magnitude
 sem parâmetros ajustáveis"
```
**Recalculado aqui `[REAL]`** (com ħ, c, G CODATA; ρ_P = m_P/ℓ_P³ = 5,1794×10⁹⁶ kg/m³;
ℓ_P = 1,61406×10⁻³⁵ m):

| H₀ usado | R_H = c/H₀ | ρ_P(ℓ_P/R_H)² | **× β_TGL** |
|---|---|---|---|
| 70 km/s/Mpc | 1,3226×10²⁶ m | 7,7140×10⁻²⁶ | **9,2568×10⁻²⁸ kg/m³** |
| 67,4 km/s/Mpc | 1,3740×10²⁶ m | 7,1516×10⁻²⁶ | **8,5819×10⁻²⁸ kg/m³** |

**O número 7,8×10⁻²⁷ kg/m³ não sai da fórmula escrita** — a fórmula dá 9,3×10⁻²⁸ (fator **8,4×**
abaixo do declarado, e fator **6,8×** abaixo do observado). Curiosamente, **sem** o fator β o
resultado é 7,7×10⁻²⁶ — dez vezes *acima* do declarado. Nenhuma das duas leituras produz
7,8×10⁻²⁷. **O número do Tratado não é reproduzível a partir da própria equação do Tratado.**
`[REAL — divergência medida]`. A frase "sem parâmetros ajustáveis" também é imprecisa: β entra
como fator, e β em F1/F2 é `α₂`, que depende de `r_coer` `[INPUT]`.

---

## 4. (d) A RELEITURA DE H₀ COMO TAXA DE ACOPLAMENTO

### 4.1 A redefinição `[POSTULATE]` (F1 §5.2, caixa vermelha, linhas 831–836)

```
H₀ ≡ γ_{Λ,0} = "Taxa fundamental de dissipação Lindblad hoje"
```
Leitura dada: *"H₀ não quantifica 'velocidade de expansão do espaço' (conceito nebuloso), mas
sim a **taxa de acoplamento** do sistema cosmológico 3D ao banho holográfico 2D — ou seja, a
frequência com que energia/informação é transferida entre os dois domínios."* `[ONTO]`

Crítica conceitual que motiva a troca (F1 linha 825): *"O que significa fisicamente 'espaço
expandindo'? Espaço não é substância material que pode expandir como balão."*

### 4.2 **Uma inconsistência de identidade que atravessa os três arquivos** `[REAL]`

A mesma quantidade é escrita de **três formas incompatíveis**:

| onde | relação escrita | implica |
|---|---|---|
| F1 §5.2, caixa | `H₀ ≡ γ_{Λ,0}` | γ_Λ = H₀ (fator 1) |
| F1 §5.3.4, caixa | `γ_Λ = α₂ H₀` | γ_Λ = 0,012 H₀ |
| F1 §9.1 item 2 (conclusões) | `H₀ = γ_{Λ,0}/α₂` | idem à anterior ✓ |
| F2 resumo + §"Tensão H₀" | `H₀ = γ_Λ/α²` | idem ✓ |
| F3 §H0_lindblad, caixa | `γ_Λ = β_TGL × H₀` | idem ✓ |

A caixa vermelha de F1 (`H₀ ≡ γ_Λ,0`) **contradiz** a caixa de F1 §5.3.4 (`γ_Λ = α₂H₀`) por um
fator 83. A redefinição enunciada como slogan e a relação usada nas contas não são a mesma
equação. A linhagem posterior (F2, F3) adotou a segunda e abandonou a primeira sem retratá-la.

### 4.3 A tensão H₀ como variação ambiental — mecanismo `[POSTULATE]` (F1 §5.4)

```
γ_Λ(r⃗) = γ_{Λ,0} ( 1 + β · δρ_m(r⃗)/ρ̄_m )       ← eq. 5.6 (F1) / eq. gamma_local (F3)
```
com **β um coeficiente novo**, apenas *"da mesma ordem que o acoplamento fundamental"*, tomado
`β ~ α₂ = 0,012` `[INPUT]`. Note: aqui `β` é um **terceiro** objeto, distinto de `α₂` e de
`β_TGL` — colisão de símbolos no acervo. `[OPEN]`

Sobredensidade local usada `[INPUT/KNOWN]`: Virgem (periferia, d ~ 20 Mpc, δρ/ρ̄ ~ 2–3);
Grande Atrator (ℓ ~ 307°, b ~ 9°, escala ~50 Mpc, δρ/ρ̄ ~ 1–2); **efeito líquido dentro de
~100 Mpc (escala SH0ES): δρ/ρ̄ ~ 0,05–0,10**.

Predição (F1 linhas 982–991):
```
H₀^local/H₀^global = 1 + 0,012 × (0,08) × (fator geométrico / fator de suavização)
                   ≈ 1,05 – 1,10          "Com fatores geométricos de ordem unidade"
Observado: 1,084 ✓
```
**Auditoria `[REAL]`**: `0,012 × 0,08 = 9,6×10⁻⁴`. Para chegar a 0,05–0,10 é preciso um
"fator geométrico/suavização" de **~50 a ~100**. O artigo chama isso de "ordem unidade". **Não
é.** A predição 1,05–1,10 **não sai** dos números que o artigo colocou na própria equação — sai
de um fator livre de duas ordens de grandeza. Isto é o ponto mais frágil de F1: a resolução da
tensão H₀ é anunciada como consequência e é, de fato, uma **acomodação com parâmetro livre**.
Estatuto correto: **NOMEAÇÃO com equação decorativa**, não predição.

A conta seguinte (§5.4.5, linhas 995–1000) é internamente consistente e correta:
`Δγ_Λ = α₂ ΔH₀ = 0,012 × 5,68 km/s/Mpc = 2,2×10⁻²¹ s⁻¹ ≈ 8% de γ_Λ^global` — verificado
`[REAL]` (recalc: 2,209×10⁻²¹; 8,1%). Mas ela é uma **reexpressão** da discrepância observada,
não uma derivação dela.

### 4.4 Quantos sigmas? — **o artigo se contradiz** `[REAL]`

| onde | valor |
|---|---|
| F1 abstract (linha 33) | "reduzindo discrepância de **4.4σ** para <1σ" |
| F1 §1.2 (linha 75) | `ΔH₀ = 5,68 ± 1,17 km/s/Mpc` **(4.9σ)** |
| F1 tab. scorecard (linha 1253) | "**4.4σ** problema" |
| F1 §9.1 item 3 + §9.3.2 | "**4.4σ**" |
| F3 (Tratado) | **4,9σ** em todos os lugares |
| **recalculado aqui** | `5,680 / √(1,04²+0,54²) = 5,680/1,1718 =` **4,847σ** |

`[REAL]` O número correto para os dados citados é **4,85σ**. F1 usa 4,4σ em quatro lugares e
4,9σ em um. F3 uniformizou (corretamente) para 4,9σ. **Correção registrada AO LADO, sem apagar
o original**, conforme a lei do memorial.

---

## 5. (e) TODOS OS NÚMEROS E O CONFRONTO COM DADO

### 5.1 Tabela integral dos números de F1 `[REAL — todos lidos do arquivo]`

**Entradas observacionais `[KNOWN]`**
| grandeza | valor | fonte citada |
|---|---|---|
| H₀^local (SH0ES) | 73,04 ± 1,04 km/s/Mpc | Riess et al. 2022, ApJL 934 L7 |
| H₀^CMB (Planck) | 67,36 ± 0,54 km/s/Mpc | Planck 2018, A&A 641 A6 |
| ΔH₀ | 5,68 ± 1,17 (4,9σ no §1.2 / 4,4σ no resto) | — |
| Ω_m | 0,315 | Planck |
| Ω_r | ~10⁻⁴ (desprezado) | — |
| Ω_Λ | 0,685 | Planck |
| ℓ₁ (1º pico acústico) | 220,5 ± 0,5 | Planck 2018 |
| Shift parameter ℛ | 1,7488 ± 0,0074 | Planck 2018 |
| z_* (recombinação) | 1089,92 ± 0,25 (usa 1090) | Planck 2018 |
| SNe Ia | 1701 supernovas, 0,01 < z < 2,26 | Pantheon+, Scolnic 2022 |
| σ_int (dispersão intrínseca) | 0,12 mag | — |
| BAO | eBOSS DR16, 3 bins (z = 0,38 / 0,51 / 0,70) | eBOSS 2021 |
| M_P | 1,220×10¹⁹ GeV/c² | — |
| ℓ_P | 1,616×10⁻³⁵ m | — |

**Números derivados/postulados de F1**
| grandeza | valor em F1 | recalculado | estatuto |
|---|---|---|---|
| ln(r_gal/3ℓ_P) | 126,2 | **126,1905** ✓ | `[REAL]` |
| N_eff | 10⁴ | ✓ (por construção) | `[INPUT]` |
| **α₂** | **0,012 ± 0,003** (0,01262) | **0,0126190** ✓ | `[POSTULATE+INPUT]` |
| γ_Λ | 2,72×10⁻²⁰ s⁻¹ | **2,72197×10⁻²⁰** ✓ | `[POSTULATE]` |
| r_H, V_H, E_H | ver §3.5 | ✓ | `[REAL]` |
| ρ_Λ | 6,35×10⁻²⁷ kg/m³ = 5,7×10⁻¹⁰ J/m³ | ✓ | `[KNOWN]` (é o observado) |
| Ω_m efetivo TGL | 0,315×1,012 = **0,31878** | ✓ | `[DERIVED]` |
| w_diss | ≈ −1 | — | `[DERIVED]` (no limite ∇L≈0) |
| w_aglomerado (δ~100) | **−0,45** | **−0,4482** (0,012×100×0,315/0,685) ✓ | `[DERIVED]` |
| P_Ψ(k) | α₂²H₀²/k³ | — | `[POSTULATE]` |
| f_NL^Ψ | ~3×10⁻⁴ | (0,012)²×2,17 = 3,13×10⁻⁴ ✓ | `[DERIVED]` |
| amortec. GW @100 Mpc | γ_Λd/c = 2,8×10⁻⁴; amplitude −0,014% | **2,800×10⁻⁴** ✓ | `[DERIVED]` |

### 5.2 Confronto com dado — os três testes de F1, auditados um a um

**TESTE 1 — SNe Ia (Pantheon+)**

Modelo TGL: `γ_Λ(z) = γ_{Λ,0}[1 + α₂ ρ_m(z)/ρ_Λ]` ⟹
`E_TGL(z) = √( Ω_m(1+z)³(1+α₂) + Ω_Λ ) = √(0,31878(1+z)³ + 0,685)`.
*Isto é: a TGL, neste artigo, reduz-se a ΛCDM com Ω_m inflado em 1,2%.* `[REAL]`

| | F1 diz | recalculado da **própria fórmula de F1** | ✗ |
|---|---|---|---|
| ΔE/E em z=0 | 0,0028 = **0,28%** | **0,00189 = 0,189%** | **erro** |
| ΔE/E em z=1 | 0,0094 = **0,94%** | **0,004718 = 0,472%** | **erro (fator 2)** |
| Δμ em z=0 | 0,006 mag | ~0,004 mag | propagado |
| Δμ em z=1 | 0,020 mag | ~0,010 mag | propagado |

`[REAL]` A fórmula que F1 escreve é `ΔE/E ≈ α₂Ω_m(1+z)³ / (2[Ω_m(1+z)³+Ω_Λ])`. Em z=0 F1
substituiu o denominador por `2Ω_Λ` (esqueceu Ω_m); em z=1 F1 **omitiu o fator 2** do
denominador. Os dois desvios são erros de substituição na própria equação do artigo — **ambos
inflam o efeito TGL**, por fatores 1,46× e 2,0×.

χ² reportado (F1 tab. 6.1): ΛCDM 1514,2/1698 (p=0,95, BIC 3044,3); TGL 1512,8/1697 (p=0,96,
BIC 3045,7); Δχ² = −1,4. **`[DECLARADO]` — não há código, não há saída de ajuste, não há
covariância Pantheon+ no artigo.** Conclusão do próprio F1, honesta: *"TGL é consistente com
dados de SNe Ia, mas **indistinguível de ΛCDM** com precisão atual"*, e o **BIC favorece
ΛCDM**.

**TESTE 2 — CMB (shift parameter)**

F1: `ℛ_ΛCDM = 37,86 × 0,04619 = 1,7488 ✓`; para z≫1 argumenta `ΔE/E ≈ α₂/2 = 0,006`, e então
escreve `ℛ_TGL ≈ 1,7488 × (1 − 0,003) = 1,7436`, `Δℛ = −0,0052`, **0,70σ**.

`[REAL]` **Duas coisas erradas.** (i) O texto diz 0,006 e aplica 0,003 — **a metade, sem
justificativa**. (ii) Integrei numericamente aqui (`scipy.quad`, 0 → 1090):

```
I_ΛCDM = ∫dz/√(0,315(1+z)³+0,685)  = 3,132297
I_TGL  = ∫dz/√(0,31878(1+z)³+0,685) = 3,116116
razão  = 0,9948341   →  ΔI/I = −0,5166%   (≈ α₂/2,44, não α₂/4)
Δℛ (sobre a base 1,7488) = −0,009082
|Δℛ|/σ = 0,009082/0,0074 = **1,227σ**
```
**O desvio TGL no CMB é ~1,23σ, não 0,70σ.** A frase "consistente dentro de <1σ" **não
sobrevive ao número**. Ainda é consistência aceitável (~1,2σ), mas o artigo declara uma folga
que não tem. `[REAL — o número corrige a frase]`

(iii) **O bloco "Output" do apêndice C.2 de F1 é irreprodutível `[REAL]`.** O código impresso é:
```python
integral, error = quad(lambda z: 1/E_TGL(z), 0, 1090)
R_TGL = np.sqrt(0.315 * H0**2) * integral        # H0 = 67.4
```
Executado literalmente aqui, isso dá **R_TGL = 117,877** — porque `√(0,315·67,4²)·∫dz/E` só é
adimensional se o integrando for `dz/H(z)` (em unidades de km/s/Mpc), não `dz/E(z)`. O "Output"
impresso (`R_TGL = 1.7436`) **não pode ter vindo desse código**. Ou o código foi transcrito
errado, ou a saída foi escrita à mão. Em qualquer caso: **é uma saída declarada, não medida.**
`[REAL — verificado por execução]`

**TESTE 3 — BAO (eBOSS DR16)**

| z | α∥ obs | α∥ TGL | α⊥ obs | α⊥ TGL |
|---|---|---|---|---|
| 0,38 | 0,993 ± 0,025 | 0,996 | 1,006 ± 0,025 | 1,004 |
| 0,51 | 0,985 ± 0,020 | 0,990 | 1,011 ± 0,020 | 1,007 |
| 0,70 | 1,008 ± 0,030 | 1,005 | 0,989 ± 0,028 | 0,994 |

F1: `χ²_TGL = 0,36+0,63+0,25 = 1,24`, χ²/dof = 1,24/6 = 0,21 (p = 0,98), contra
χ²_ΛCDM = 1,8 → 0,30. "TGL apresenta excelente ajuste."

`[REAL]` **Um χ²/dof de 0,21 com 6 pontos não é "excelente ajuste" — é ajuste *bom demais***
(resíduos menores que as barras por fator ~2; p = 0,98 significa que 98% das realizações teriam
resíduo maior). Isso normalmente sinaliza barras superestimadas ou modelo com liberdade
absorvida. Além disso, a soma exibida tem **3 parcelas para 6 medidas** — as parcelas por bin
não estão decompostas. E o apêndice C.3 exibe uma matriz de covariância 6×6 "×10⁻³" cujos
elementos diagonais (0,625; 0,400; 0,900; 0,625; 0,400; 0,784)×10⁻³ correspondem a σ =
0,025; 0,020; 0,030; 0,025; 0,020; 0,028 — **consistente com a tabela ✓**, mas a matriz é
**`[DECLARADO]`**: não é a covariância publicada do eBOSS, e F1 não a cita como tal.

**RESUMO DO CONFRONTO — o que o dado realmente diz `[REAL]`**

| Observável | veredito de F1 | veredito **auditado aqui** |
|---|---|---|
| SNe Ia (1701) | "empate", Δχ² = −1,4 | **NÃO FALSIFICADA / indistinguível**; BIC favorece ΛCDM; χ² `[DECLARADO]` |
| CMB (ℛ) | "0,70σ, <1σ ✓" | **1,23σ** (recalculado); ainda consistente, mas a folga declarada não existe |
| BAO (6 pts) | "TGL melhor, χ²/dof = 0,21" | ajuste bom demais; covariância `[DECLARADO]` |
| Tensão H₀ | "**Resolvida**" | **acomodada com fator livre de ~50–100×**; não é predição |
| ρ_Λ absoluto | "valor observado ✓" | **circular** (Ω_Λ observado entra e sai) |

**Nenhum dos quatro é uma confirmação.** Três são NOT_FALSIFIED; um (tensão H₀) é acomodação.
`NOT_FALSIFIED ≠ CONFIRMED`.

### 5.3 Números de F2 (bootstrap), auditados `[REAL]`

| grandeza | F2 diz | recalculado | veredito |
|---|---|---|---|
| γ_Λ(ρ̄) = α²H₀[1+β ρ̄/ρ_crit] | β ~ α² ≈ 0,012 | — | `[POSTULATE]` |
| γ_proj(ρ̄) = γ₀[1+κ ρ̄/ρ_P] | κ > 0, "a vincular" | — | `[POSTULATE]`, **κ é parâmetro novo livre** |
| ρ_P | 5×10⁹³ g/cm³ | **5,179×10⁹⁶ kg/m³ = 5,179×10⁹³ g/cm³** ✓ | `[KNOWN]` |
| autovalores do Jacobiano | λ₁ = 0, λ₂ = Tr(J) < 0 sse κ > 0 | ✓ álgebra correta | `[DERIVED]` |
| τ_relax = 1/(α²H₀) | ~4×10¹⁹ s ~ **10¹² anos** | **3,674×10¹⁹ s = 1,164×10¹² anos** ✓ | `[DERIVED]` |
| τ_decay = 1/(2α²H₀) | ~2×10¹² anos | ✓ | `[DERIVED]` |
| **T_osc = 2π/√(γ_Λγ_proj)** | **~10¹⁰ anos** | **2,308×10²⁰ s = 7,314×10¹² anos** | **✗ erro de ~700×** |
| δH/H da oscilação | 0,05 a 0,10 | — | `[CONJECTURE]` |
| deriva de timing | ~10⁻²⁰ s⁻¹; 20 anos → Δt ~10⁻¹¹ s vs SKA ~10⁻⁹ s | — | `[CONJECTURE]`, **fora de alcance por 100×** |
| ρ̄_banho⁰/ρ̄_3D⁰ | 0,685/0,315 = 2,17 | ✓ | `[DERIVED]` |
| **Ω_Λ predito** | **"0,01 a 0,10"** vs observado **0,685** | — | **FALHA declarada pelo próprio F2** |

**`[REAL]` O erro do período oscilatório é grave e é interno**: `2π/(α²H₀)` com α² = 0,012 e
H₀ = 2,3×10⁻¹⁸ s⁻¹ dá **7,3×10¹² anos**, não 10¹⁰. F2 tinha acabado de calcular
`1/(α²H₀) ~ 10¹² anos` três páginas antes — multiplicar por 2π não pode reduzir em 100×.

**`[REAL]` Pior: o sistema linearizado de F2 NÃO oscila.** A própria proposição de F2
(linhas 651–665) resolve `ρ̄_3D(t) = ρ̄_∞ + Ae^{−2γt}`, `ρ̄_banho(t) = ρ̄_∞ − Ae^{−2γt}` —
**decaimento exponencial puro**, autovalores {0, −2γ}, **ambos reais**. Não há parte imaginária;
não existe período. A "Predição: Modulação Temporal de H₀ com T ~ 10¹⁰ anos" **contradiz a
solução exata que o próprio artigo derivou duas seções antes**. Estatuto: **NOMEAÇÃO**.

**`[REAL]` A falha de Ω_Λ é o negativo honesto mais importante de F2.** O texto (linhas 826–840):
> *"Com α² = 0,012 e fatores de ordem unidade: Ω_Λ ~ 0,01 a 0,10. Valor observado: Ω_Λ = 0,685.
> **Discrepância aparente**: Indica que a fração do sistema em dissipação ativa é maior que α²
> sugeriria ingenuamente."*

Seguem três hipóteses de escape (efeitos não-lineares acumulados; γ dependente da densidade
integrada sobre toda a história; ciclo não-balanceado) e a frase *"Trabalho futuro: resolver
numericamente"*. **A predição fecha um fator 7–68 abaixo do observado e o artigo diz isso em voz
alta.** Registro como **negativo honesto = RESULTADO**.

**`[REAL]` A tabela de predições de F2 contém uma violação de régua.** Linha 900:
> `Tensão H₀ | ΔH/H ~ 8% | **Detectado** (4.4σ) | **Confirmado**`

Marcar "Confirmado" para a própria predição usando como "detecção" a discrepância observacional
que a motivou é **circular** e viola a proibição de `CONFIRMED`. Deve ler-se, no máximo,
`CONSISTENTE / NOT_FALSIFIED`.

### 5.4 Números de F3 (Tratado) que **não existem** em F1/F2 `[REAL]`

| item | F3 diz | origem |
|---|---|---|
| `w = −1 + β_TGL² = −0,99986` | correção fina de w | **novo em F3**; recalc: `−1+0,012031300400803142² = −0,99985525` ✓ |
| justificativa de β² | "ciclo completo: ida (∂→bulk, custo β) e volta (bulk→∂, custo β)" | `[ONTO]` — **é a leitura de transição de regimes mais explícita do acervo** |
| `H₀^TGL = 73,02 km/s/Mpc`, "concordância 99,7%", **`Δχ² = 23,49` "evidência MUITO FORTE"** | caixa `resultado` | **`[DECLARADO]` — sem código, sem dado, sem likelihood em nenhum dos três arquivos** |
| `a₀ = α·c·H₀ ≈ 1,2×10⁻¹⁰ m/s²` (MOND) | "predito, não ajustado" | `[DECLARADO]` (Protocolo #6, fora destes arquivos); F3 admite: *"derivação semi-quantitativa […] permanece programa aberto"* |
| RAR / SPARC 175 galáxias, r > 0,99 | `[DECLARADO]` | Protocolo #6 |
| matéria escura = condensado psiônico `\|Ψ_ligado⟩ = (\|ψ₊ψ₋⟩+\|ψ₋ψ₊⟩)/√2`, perfil NFW-idêntico | `[CONJECTURE]` | novo em F3 |
| `n_Ψ(R) = 1 + β_TGL·R/R₀` (índice de refração holográfico); `θ_TGL = β·θ_RG ~ 1,2%` | `[POSTULATE]` | Protocolo #8 |
| ΔT/T ~ 10⁻¹⁰ (supressão de anisotropias não-lineares) | `[CONJECTURE]` | novo em F3 |
| Planck: `w = −1,03 ± 0,03` | `[KNOWN]` | Planck 2018 |
| BAO "eBOSS DR16 **+ DESI 2024**" | mesma tabela de 3 bins de F1 | **`[OPEN]`** — o rótulo cita DESI mas os números são idênticos aos de F1 (eBOSS puro). Rótulo sem dado. |

**F3 é a peça mais honesta do trio nas limitações** (§"Limitações honestas", 5 itens — ver §7).
E é a menos honesta nos números novos (Δχ² = 23,49, H₀^TGL = 73,02) — anunciados sem cadeia.

---

## 6. ⭐ A PERGUNTA CRÍTICA DO OPERADOR: **A TRANSIÇÃO DE REGIMES (w = −1 ↔ w = 0)**

> *"o operador tipa energia escura como o banho holográfico, A TRANSIÇÃO DE REGIMES. Procure no
> texto o tratamento da TRANSIÇÃO entre regimes (w=−1 ↔ w=0) — há equação para a transição?
> há parâmetro que a controle?"*

**RESPOSTA DIRETA, em três partes:**

### 6.1 **Não existe, em nenhum dos três arquivos, uma seção, equação ou parâmetro nomeado
"transição de regimes".** `[REAL — busca exaustiva]`

Varri os três arquivos por `transi*`, `regime`, `crossover`, `w = 0`, `w_0`, `w(z)`, `freeze`,
`congelamento`. Resultado:
- **F1**: "regime" aparece 2× e ambas as vezes significa "regime observacional/de modos", não
  transição de w. "Transição" aparece 4× e **nenhuma** é entre w=−1 e w=0 — são: transição
  conceitual (calórico→calor), transição newtoniana→einsteiniana, e **transição 2D→3D no Big
  Bang** (declarada como **questão aberta**, §9.4.2 e §9.5.2).
- **F2**: "transição de fase" aparece 1× — o **novo Big Bang por saturação do banho**.
- **F3**: nenhuma menção a transição w=−1↔w=0.

**Portanto: a tipagem do operador ("energia escura = a transição de regimes") NÃO está escrita
nestes textos.** Ela é uma leitura posterior. O que os textos têm é o **material bruto** dela —
descrito abaixo.

### 6.2 **Mas HÁ equação de transição — três delas — e HÁ um parâmetro que a controla: α₂.**
`[DERIVED por mim, a partir das equações do próprio artigo]`

O artigo escreve o mecanismo sem nomear a transição. Extraindo:

**(T1) A transição por AMBIENTE — sobredensidade** (F1 §7.1, F3 §predições item 2)
```
w(δ) = −1 + α₂ · (δρ_m/ρ̄_m) · (Ω_m/Ω_Λ)
```
Esta é **uma interpolação contínua de w = −1 até w = 0 e além**. O parâmetro de controle é a
sobredensidade δ; o **coeficiente** de controle é **α₂**. Fazendo w(δ)=0:
```
δ_crossover = (1/α₂)·(Ω_Λ/Ω_m) = (1/0,012)·(0,685/0,315) = **181,2**
com α₂ = 0,012619:  δ = 172,3
com β_TGL = α√e:    δ = 180,7
```
`[DERIVED]` **A transição w=−1 → w=0 ocorre em δρ_m/ρ̄_m ≈ 181.** O artigo **não faz essa
conta** — ele para em δ~100, onde w = −0,4482 (F1 diz −0,45 ✓). Ou seja: **o artigo caminha até
55% do caminho da transição e não a nomeia.** No centro de aglomerados ricos (δ ≳ 200) a própria
equação de F1 prediz **energia escura com w > 0** — regime de poeira e além. Isto é uma
consequência **não explorada** e **flagrantemente falsificável** da equação escrita.

**(T2) A transição por REDSHIFT — a época do crossover** (F1 §6.1.4)
```
γ_Λ(z) = γ_{Λ,0} [ 1 + α₂ · (Ω_m/Ω_Λ)(1+z)³ ]
```
O termo de dissipação-de-vácuo domina em baixo z; o termo acoplado-à-matéria domina em alto z.
Igualando os dois:
```
(1+z_c)³ = Ω_Λ/(α₂ Ω_m) = 181,2   ⟹  **z_c = 4,659**
```
`[DERIVED]` **Existe uma época de transição de regime em z ≈ 4,7** implícita na equação de F1.
O artigo **nunca calcula z_c** e nunca comenta o que acontece acima dele. Este é, na minha
leitura, **o número mais interessante que os textos contêm sem saber que contêm**: ele diz que o
"banho holográfico" muda de face por volta de z ≈ 4,7 — bem dentro do alcance de survey.
`[OPEN — a ser confrontado com dado; nada disso está testado nos arquivos]`

**(T3) O parâmetro estrutural que liga os dois regimes: `∇L_k`** (F1 §4.2.2, linhas 702–711)
```
P_diss = −(1/3) Σ_k γ_k Tr[ L_kρL_k† ∇²]
"No limite de operadores locais (∇L_k ≈ 0):  P_diss ≈ −ρ_diss"  ⟹ w = −1
```
**Este é o enunciado mais limpo da transição em todo o acervo lido**: `w = −1` **é o limite de
gradiente nulo dos operadores de Lindblad**. O desvio de −1 é controlado por `∇L_k ≠ 0`. Ou
seja, na linguagem do operador: **o regime w=−1 é o banho homogêneo (sem gradiente); o regime
w→0 é o banho com estrutura (com gradiente).** A transição de regimes **é a ligação/desligamento
do gradiente do operador de salto**. `[DERIVED — leitura direta da equação de F1]`
**Mas F1 nunca escreve `P_diss` fora desse limite.** Não há forma funcional para `∇L_k ≠ 0`.
**MECANISMO nomeado, equação ausente.** `[OPEN]`

### 6.3 A transição em F2 — a **única com dinâmica temporal explícita** `[REAL]`

F2 é o texto que mais se aproxima de "a transição de regimes" como objeto físico:

```
dρ̄_3D/dt    = −γ_Λ(ρ̄_3D)·ρ̄_3D + γ_proj(ρ̄_banho)·ρ̄_banho
dρ̄_banho/dt = +γ_Λ(ρ̄_3D)·ρ̄_3D − γ_proj(ρ̄_banho)·ρ̄_banho          (F2 eq. sistema_classico)
```
com **ponto fixo** `γ_Λ ρ̄*_3D = γ_proj ρ̄*_banho` (fluxo ida = fluxo volta) e
**condição de estabilidade** `λ₂ < 0 ⟺ dγ_Λ/dρ̄ > 0 ∧ dγ_proj/dρ̄ > 0 ⟺ **κ > 0***.

**A transição de regimes, em F2, é a trajetória do sistema entre dois pontos:**
| época | ρ̄_3D | ρ̄_banho | regime |
|---|---|---|---|
| Big Bang (t=0) | ρ_Planck ~10⁹³ g/cm³ | ≈ 0 ("banho vazio") | tudo projetado em 3D |
| hoje (13,8 Gyr) | 0,315 ρ_crit | 0,685 ρ_crit | razão 2,17 |
| equilíbrio (t→∞) | ρ_total/2 | ρ_total/2 | razão 1,00 |

`[REAL]` E o **cronômetro da transição**: `τ_relax = 1/(α²H₀) ≈ 1,16×10¹² anos` — **84 vezes a
idade do universo**. F2 tira a conclusão certa (linha 633): *"Sistema ainda está próximo de
transiente inicial!"* **Ou seja: na leitura de F2, estamos NO MEIO DA TRANSIÇÃO, a 1,2% do
caminho.** `α₂` é literalmente o parâmetro que controla a velocidade da transição.

E **na outra ponta**, a transição de volta (F2 §5.3, "Universos Cíclicos"):
```
ρ̄_banho → ρ̄_max ~ ρ_P/(A_P/A_H)  ⟹  γ_proj → ∞  ⟹  "Projeção massiva 3D" = novo Big Bang
Critério de ciclo:  ∫₀^{T_ciclo} [ γ_Λ ρ̄_3D − γ_proj ρ̄_banho ] dt = 0
```
`[POSTULATE]` — sem número, sem T_ciclo calculado.

**`[OPEN]` — e é aqui que o buraco fica dito**: F2 escreve a equação de Friedmann modificada com
`w_3D` e `w_banho` e, na lista de definições (linha 771), declara literalmente:
> `w_3D`, `w_banho`: equações de estado efetivas **(a derivar)**

**As equações de estado dos dois regimes NUNCA são derivadas.** O objeto exato que a pergunta
do operador procura — a lei w(regime) — está marcado "a derivar" no próprio texto. Isto é o
achado central desta seção.

### 6.4 Veredito sobre a transição — MECANISMO / PREDIÇÃO / NOMEAÇÃO

| face | estatuto | evidência |
|---|---|---|
| Há **mecanismo** de transição? | **SIM** | 3 equações: w(δ) ambiental; γ_Λ(z); sistema de EDOs acopladas de F2 com ponto fixo e λ₂ |
| Há **parâmetro** que a controla? | **SIM — α₂** (e, em F2, também **κ**, que é livre) | δ_crossover = 181, z_c = 4,66, τ_relax = 1/(α₂H₀) — os três são fixados por α₂ |
| Há **predição** (número confrontável)? | **QUASE NÃO** | w=−0,45 em δ~100 é o único; nunca foi confrontado com dado em nenhum dos 3 arquivos; e `σ_w` atual ~0,03 vs efeito β² ~10⁻⁴ (F3 admite: exige `σ_w < 10⁻⁴`) |
| A transição é **nomeada** como tal? | **NÃO** | nenhuma seção, nenhum termo; a tipagem do operador é leitura posterior ao texto |
| A lei w do regime está derivada? | **NÃO — "a derivar"**, dito pelo próprio F2 | linha 771 |

**Frase honesta**: *a transição de regimes existe nos textos como MECANISMO com parâmetro de
controle identificado (α₂), produz dois números deriváveís que o acervo ainda não calculou
(δ_crossover ≈ 181; z_c ≈ 4,66), e NÃO existe como predição confrontada nem como equação de
estado derivada — o próprio bootstrap marca `w_3D` e `w_banho` como "a derivar".*

---

## 7. (f) AS QUESTÕES ABERTAS QUE OS PRÓPRIOS ARTIGOS DECLARAM

### 7.1 F1 §9.4 "Limitações e Questões Abertas" — as cinco, verbatim `[DECLARADO pela origem]`

1. **Formalismo quântico completo** — o artigo é semiclássico. *"Teoria quântica de campos
   completa no contexto TGL — incluindo quantização do campo Ψ, renormalização, e cálculo de
   correções radiativas — permanece em desenvolvimento."* Pergunta: *"Qual é o espectro completo
   de excitações de Ψ ('psions')?"*
2. **Origem cosmológica do banho Ψ** — *"Postulamos existência de banho holográfico 2D, mas sua
   origem cosmológica não foi completamente especificada."* Três perguntas: o banho existe
   "antes" do Big Bang? **Como a transição 2D→3D ocorreu no Big Bang?** Inflação é compatível?
   → **É esta questão que gera F2 inteiro.**
3. **Matéria escura** — mencionada (Ψ em fase condensada?) mas não desenvolvida.
   *"TGL pode também explicar matéria escura, ou componente adicional é necessária?"*
   → **F3 responde com o condensado psiônico `[CONJECTURE]`.**
4. **Testes de precisão** — *"Predições únicas de TGL (variação ambiental de w, flutuações Ψ)
   estão no limite ou abaixo da sensibilidade observacional atual."* Precisa de Euclid, LSST,
   SKA, Einstein Telescope.
5. **Princípios fundamentais** — a estrutura holográfica 2D/3D foi **postulada**, não derivada.
   *"TGL é teoria efetiva de alguma estrutura mais profunda? Como se relaciona com loop quantum
   gravity ou teoria de cordas?"*

**Direções futuras declaradas** (F1 §9.5): curto prazo 2025–2030 (perfis de massa Euclid;
não-gaussianidade LSST/Euclid; timing de pulsares MeerKAT/SKA); médio 2030–2040 (ET/Cosmic
Explorer; CMB-S4/SO; bi-espectro de 10⁹ galáxias); longo 2040+ (LISA/BBO; mapeamento 3D de
z<1; *"testes de variação de w em ambiente controlado (!)"* — o "(!)" é do original).
Teóricas: QFT de Ψ em espaço curvo; renormalização e RG para TGL; **conexão com AdS/CFT**;
inflação; transição 2D→3D no Big Bang; Big Rip vs Big Crunch vs estado estacionário; acreção em
BHs com α₂; estrelas de nêutrons; **nucleossíntese primordial com dissipação Lindblad**;
derivação da TGL a partir de gravidade quântica; problema da medição; consciência e informação
quântica.

### 7.2 F2 — abertos declarados `[DECLARADO]`

- **`w_3D` e `w_banho`: "a derivar"** (linha 771). ⭐ o buraco central da transição de regimes.
- **`κ` é "parâmetro novo (a ser vinculado observacionalmente)"** (linha 507). A estabilidade
  inteira do bootstrap depende do sinal de um parâmetro livre.
- **`γ_proj` "a ser determinada da auto-consistência"** (linha 339) — nunca é determinada.
- **A falha de Ω_Λ**: predito 0,01–0,10 vs observado 0,685; três escapes listados;
  *"Trabalho futuro: Resolver numericamente […] com condições iniciais do Big Bang até hoje."*
- Apêndice A.1 "Prova Detalhada da Estrutura de Lindblad" contém literalmente
  **`[Conteúdo técnico adicional...]`** — **placeholder vazio** `[REAL]`.
- Comentário no corpo (linha 842): *"Devido ao limite de espaço, vou pular para predições
  observacionais e conclusão…"* e (linha 635) *"% Continuarei com as demais seções…"* —
  **o artigo é declaradamente incompleto**.
- Trabalho futuro: QFT de Ψ em horizonte curvo; TGL a partir de LQG/teoria-M; inflação bootstrap;
  Euclid/JWST; SKA/NANOGrav; LSST/Roman; integração numérica; N-body com dissipação Lindblad.

### 7.3 F3 §"Limitações honestas" — as cinco, verbatim `[DECLARADO]`

1. *"A reinterpretação é consistente mas **não compelida** pelos dados: ΛCDM ajusta igualmente
   bem. A distinção requer `σ_w < β_TGL² ≈ 10⁻⁴`."*
2. *"O perfil do condensado psiônico é **idêntico ao NFW**: indistinguível de matéria escura fria
   ao nível das curvas de rotação. **Detecção direta de WIMPs ou axions falsificaria** a
   interpretação TGL."* ← falsificador nomeado.
3. *"A derivação de `a₀ = αcH₀` é **semi-quantitativa**. A concordância numérica é exata, mas a
   derivação rigorosa a partir da Lagrangiana TGL permanece **programa aberto**."*
4. *"A variação ambiental de H₀ resolve a tensão, mas modelos alternativos (early dark energy,
   interacting dark energy) também propõem resoluções. **A TGL não é a única candidata.**"*
5. *"O formalismo quântico é apresentado em nível semiclássico. A teoria quântica de campos
   completa para Ψ — incluindo renormalização e correções radiativas — permanece em
   desenvolvimento."*

### 7.4 Abertos que **eu** acrescento, por leitura (não declarados pela origem) `[OPEN]`

| # | buraco | onde |
|---|---|---|
| O1 | `γ_H` (canal de expansão da eq. mestra) nunca é fixado nem usado | F1 §3.4.1 |
| O2 | `L_diss = √γ_Λ Ĥ` é hermitiano ⟹ dephasing, não termalização | F1 §3.4.2 |
| O3 | `γ_Λ = α₂H₀` é ansatz que substitui uma derivação fracassada, apresentado como derivação | F1 §5.3.4 |
| O4 | `ρ_Λ = γ_Λ⟨H⟩` não fecha dimensionalmente (falta um tempo) | F1 §4.2 |
| O5 | `H₀ ≡ γ_{Λ,0}` (caixa) contradiz `γ_Λ = α₂H₀` (caixa) por fator 83 | F1 §5.2 vs §5.3.4 |
| O6 | termo `γ_proj` com sinal negativo quebra a completa positividade | F2 §3.3 |
| O7 | `Ĥ^eff = Ĥ + (iħ/2)L†L` é não-hermitiano; "Lamb shift" mal-posto | F2 eq. H_eff |
| O8 | 4,4σ vs 4,9σ (correto: 4,85σ) dentro do mesmo artigo | F1 |
| O9 | ΔE/E em z=0 e z=1 discordam da própria fórmula (fatores 1,46× e 2×) | F1 §6.1.5 |
| O10 | ℛ_TGL: "0,006" declarado e "0,003" aplicado; desvio real 1,23σ, não 0,70σ | F1 §6.2 |
| O11 | o "Output" do apêndice C.2 não é reprodutível pelo código impresso (dá 117,88) | F1 apêndice |
| O12 | χ²/dof = 0,21 em BAO é ajuste bom demais; covariância 6×6 é `[DECLARADO]` | F1 §6.3 + apêndice |
| O13 | T_osc = 10¹⁰ anos: conta dá 7,3×10¹² anos; e o sistema linear **não oscila** | F2 §6.2 |
| O14 | tabela de F2 marca a tensão H₀ como "**Confirmado**" — violação de régua, e circular | F2 §6.4 |
| O15 | `ρ_Λ = βρ_P(ℓ_P/R_H)²` do Tratado dá 9,3×10⁻²⁸, não os 7,8×10⁻²⁷ declarados | F3 §rho_lambda |
| O16 | `Δχ² = 23,49` e `H₀^TGL = 73,02` aparecem sem cadeia, código ou dado | F3 |
| O17 | rótulo "eBOSS DR16 **+ DESI 2024**" sobre a tabela de F1 (eBOSS puro) | F3 §bao |
| O18 | colisão de símbolos: `α₂` (0,01262), `β` do acoplamento ambiental, `β_TGL` (0,0120313) | F1/F2/F3 |
| O19 | "fator geométrico/suavização" de ~50–100× chamado de "ordem unidade" na resolução da tensão H₀ | F1 §5.4.4 |
| O20 | `√−g` tratado como `L⁴` na análise dimensional que conclui "✓" | F1 §2.5 |

---

## 8. LEITURA ONTOLÓGICA — o que estes textos dizem que a linhagem posterior confirma ou revoga
`[ONTO]`

**Confirmado pela linhagem posterior:**
- A energia escura como **processo, não substância**; a analogia da evaporação; `w ≈ −1` como
  limite de gradiente nulo. Sobrevive.
- O universo como **sistema aberto acoplado a fronteira 2D**. Sobrevive e vira o núcleo
  (`S_∂ = ½ nat`, modularidade).
- A **necessidade termodinâmica**: sem `γ_Λ > 0`, sem estrutura, sem observador. Sobrevive como
  leitura, com o alerta de que F1 chama isso de "Teorema Fundamental" numa caixa verde **sem
  demonstração** — é `[ONTO]`, não teorema.
- O piso ligado a `β`: F1 já dizia `w_aglomerado = −1 + α₂·δ·Ω_m/Ω_Λ`. O **piso dos vazios**
  posterior (`ρ_vazio/ρ̄ ≥ β`) é o **conjugado** desta equação na outra ponta da escala de δ —
  F1 explorou a ponta densa (aglomerados), o programa posterior explorou a ponta rarefeita
  (vazios). É a mesma equação lida nas duas direções. ⭐

**Revogado / superado pela linhagem posterior:**
- **α₂ = 126,2/10⁴** (calibrado por `r_coer`) foi substituído por **β = α√e** (zero-free).
  São números diferentes; a substituição de símbolo no Tratado apaga essa história. **A
  estratigrafia manda: F1 é nov/2025, é anterior à fatoração α·√e.**
- "Tensão H₀ **resolvida**" e "Confirmado" na tabela de F2: hoje seriam
  `NOT_FALSIFIED` / acomodação com parâmetro livre.
- "termalização" pelo canal `√γ_Λ Ĥ`: a lei posterior `Γ_ω = ½βτ★ω²` é **dephasing** — o
  vocabulário de 2025 estava errado sobre a própria álgebra que escreveu.

---

## 9. VEREDITO FINAL DO DOMÍNIO

| pergunta do operador | resposta em uma linha | estatuto |
|---|---|---|
| (a) postulado do banho | universo 3D = sistema aberto acoplado a banho térmico holográfico 2D (campo Ψ); em F2 o banho ganha a seta de volta (`γ_proj`) e o loop fecha | `[POSTULATE]` |
| (b) eq. mestra + contagem de modos | GKLS com **2 canais** (`γ_H` expansão, nunca fixado; `γ_Λ` dissipação); modos contados por harmônicos angulares: `N_eff ~ (r_gal/r_coer)² = 10⁴` | `[POSTULATE]` + `[INPUT]` |
| (c) ρ_Λ = γ_Λ⟨H⟩; γ_Λ derivado ou ajustado? | **AJUSTADO.** A derivação por Friedmann fracassa dimensionalmente no próprio texto e é remendada por `γ_Λ = α₂H₀`, inserida por argumento verbal. A "verificação" é circular (Ω_Λ observado entra e sai). A fórmula fechada não fecha dimensionalmente. | `[POSTULATE]`, **não** `[DERIVED]` |
| (d) H₀ como taxa de acoplamento | releitura ontológica forte e coerente; mas a caixa `H₀ ≡ γ_Λ,0` contradiz `γ_Λ = α₂H₀` usada nas contas | `[ONTO]` + `[OPEN]` |
| (e) números e confronto | 3 testes: SNe **indistinguível** (BIC favorece ΛCDM); CMB **1,23σ** (não 0,70σ); BAO ajuste bom demais com covariância declarada; tensão H₀ **acomodada** com fator livre 50–100×; **nenhuma confirmação** | `NOT_FALSIFIED` ×3 |
| (f) abertos declarados | 5 em F1, 6 em F2 (inclusive `w_3D`/`w_banho` "a derivar"), 5 em F3; + 20 acrescentados por esta leitura | `[OPEN]` |
| ⭐ **transição de regimes** | **MECANISMO SIM, PREDIÇÃO NÃO, NOME NÃO.** Controlada por **α₂**. Três equações; dois números que o acervo nunca calculou: **δ_crossover ≈ 181** e **z_c ≈ 4,66**; e o enunciado limpo: **w = −1 é o limite `∇L_k → 0`** — o regime muda quando o banho ganha gradiente. `w_3D`/`w_banho` marcados "a derivar" pelo próprio bootstrap. | `[DERIVED]` + `[OPEN]` |

**A frase que os números permitem:**
> *A energia escura, nestes três textos, é o banho holográfico visto como taxa. O banho é
> POSTULADO; a taxa `γ_Λ = α₂H₀` é AJUSTADA (o próprio artigo mostra a derivação falhando); o
> confronto com dado é NÃO-FALSIFICAÇÃO em três frentes e ACOMODAÇÃO na quarta; e a transição de
> regimes — o que o operador tipa como sendo a coisa — está lá como mecanismo com parâmetro de
> controle identificado (α₂, com crossover em δ≈181 e z≈4,66), mas sua lei (`w_3D`, `w_banho`)
> está escrita no próprio bootstrap com três palavras: **"a derivar"**.*

**NOT_FALSIFIED nunca é CONFIRMED. Nada aqui foi confirmado. O buraco fica dito.**

---
*Relatório produzido em 21/08/2026 por leitura integral de F1 (1.994 linhas), F2 (1.179 linhas)
e F3 (411 linhas), com recálculo por script de 24 grandezas. Nenhum número escrito de memória.*
