# β_TGL = α·√e: the geometric cost of absolute zero · β_TGL = α·√e: o custo geométrico do zero absoluto

> **TGL — Teoria da Gravitação Luminodinâmica · Theory of Luminodynamic Gravitation.** Part 2 of 8 · seal **v368** · `um.py` sha256 `4a34fbf36f3ae0d8…` · generated 2026-09-21 by script from the published files.
> Every excerpt below is **verbatim**, with its source, byte range and sha256. Statuses follow the ruler: PROVED = theorem in the Lean kernel; CONFIRMED = a judgement about nature, not made here.
> All eight parts are listed at the top of the start page, https://teoriadagravitacaoluminodinamica.com/read-brief.md, and in the door of this folder, https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/secoes/PORTA.md · Reading limits measured on 2026-09-19: one real fetcher cut documents near 100,000 characters, refused files above 10 MB, and could not read PDFs served as `application/octet-stream` — read the TXT/TeX sources.

## In short (EN)

β_TGL = α·√e takes α (CODATA 2018) [KNOWN] and √e as its two inputs; the sealed value is 0.0120313 [DERIVED; A eq:beta-definition; selo]. ln √e = ½ nat is a mathematical identity; Article A uses it to select √e by principle over φ, √2 and √π [DERIVED; A §Por que √e], and reads it as the minimal cost of one boundary↔bulk parity flip, the holographic analogue of Landauer's k_B T ln 2 [ONTO; A §Por que √e]. The Miguel angle is θ_M = arcsin √β, with 1 − β = cos²θ_M [DERIVED; A eq:theta-M]. For every ε > 0 there is a positive temperature at which the thermal floor is already below ε. A strictly positive geometric floor that does not depend on temperature survives the limit, and the crossing is computed near 201 K [REAL; C txt; selo geometric_cost_of_absolute_zero]. The α-free value stays [OPEN] [C txt; selo]. In um.py the form is the Lagrange transform 1 = q² + α², with α = sech(χ/2) and q = tanh(χ/2) [REAL; C txt]. CODATA enters only at the final validation, and core.beta is written at runtime [C txt; read-brief]. Lineage: R_∂ = 1/α_CODATA as the motor of the chain is retired as legacy [C txt]. Gate TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED_WITHIN_LOCAL_BULK_AT_AVAILABLE_SENSITIVITY__MORE_SENSITIVE_DATA_COULD_REVISE. Nature: NOT CONFIRMED [selo].

## Em resumo (PT)

β_TGL = α·√e tem duas entradas, α (CODATA 2018) [KNOWN] e √e; o valor selado é 0.0120313 [DERIVED; A eq:beta-definition; selo]. ln √e = ½ nat é identidade matemática. O Artigo A a usa para selecionar √e por princípio, e não φ, √2 ou √π [DERIVED; A §Por que √e], e a lê como o custo mínimo de um flip de paridade fronteira↔bulk, análogo holográfico do k_B T ln 2 de Landauer [ONTO; A §Por que √e]. O ângulo de Miguel é θ_M = arcsin √β, com 1 − β = cos²θ_M [DERIVED; A eq:theta-M]. Para todo ε > 0 existe temperatura positiva em que o piso térmico já está abaixo de ε. Um piso geométrico estritamente positivo e independente da temperatura sobrevive ao limite, com cruzamento calculado perto de 201 K [REAL; C txt; selo geometric_cost_of_absolute_zero]. O valor α-livre segue [OPEN] [C txt; selo]. No um.py a forma é a transformada de Lagrange 1 = q² + α², com α = sech(χ/2) e q = tanh(χ/2) [REAL; C txt]. O CODATA entra só na validação final, e core.beta é gravado em tempo de execução [C txt; read-brief]. Linhagem: R_∂ = 1/α_CODATA como motor da cadeia está aposentado (legado) [C txt]. Gate TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED_WITHIN_LOCAL_BULK_AT_AVAILABLE_SENSITIVITY__MORE_SENSITIVE_DATA_COULD_REVISE. Natureza: não confirmada [selo].

## Sources, verbatim · fontes, verbatim

### 1. `O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.tex` — bytes 9.647–10.380

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.tex
- sha256 of the file: `559d79fb81b0af6c0fb8c47549a0e21e3b29f4173a5b5a642f9464611b4fa217` (computed now; this file is not in the seal map) · of this excerpt: `fb3ec5558297b69e5381ef09724c8166000109da9262246cc704108571855c86` · line endings shown as LF
- status · estatuto: [DERIVED]
- why · por quê: Definição canônica de β_TGL = α·√e com as duas entradas (α CODATA 2018 e √e) e o ângulo de Miguel θ_M = arcsin √β, 1 − β = cos²θ_M.

````text
\boxed{\;\betatgl \;=\; \alpha \cdot \sqrt{e} \;=\; 0{,}0120313004008031\;}
\label{eq:beta-definition}
\end{equation}
é a única constante invariante adimensional do programa.  As duas entradas
são $\alpha = 7{,}2973525693 \times 10^{-3}$ (constante de estrutura fina,
CODATA 2018) e $\sqrt{e} = 1{,}6487212707\ldots$ (matemática pura, sem
ambiguidade dimensional).  O termo~\eqref{eq:Lmodular} se anula em estado
puro $w_{\text{eff}} = -1$ (constante cosmológica) e satura em
$\betatgl \cdot |1+w|$ longe desse ponto.  O ângulo de Miguel emerge de
\begin{equation}
\thetaM \;=\; \arcsin \sqrt{\betatgl} \;=\; 6{,}29729^{\circ},
\qquad
1 - \betatgl \;=\; \cos^2 \thetaM \;=\; 0{,}987968699599197.
\label{eq:theta-M}
````

### 2. `O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.tex` — bytes 10.400–12.115

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.tex
- sha256 of the file: `559d79fb81b0af6c0fb8c47549a0e21e3b29f4173a5b5a642f9464611b4fa217` (computed now; this file is not in the seal map) · of this excerpt: `06ca31c37f944dbcbb8abff7e6790e57efd84468f4ded05bb2da23d636924103` · line endings shown as LF
- status · estatuto: [DERIVED] seleção; [ONTO] leitura/analogia
- why · por quê: Seleção de √e pela meia-nat: ln √e = ½ nat; leitura como flip de paridade fronteira↔bulk e analogia com Landauer; por que não φ, √2 nem √π.

````text
\paragraph{Por que $\sqrt{e}$, e não outro fator? --- a seleção por meio-nat.}
A objeção imediata a $\betatgl = \alpha\sqrt{e}$ é que o produto de dois números
adimensionais poderia ser numerologia: por que $\sqrt{e}$, e não $\varphi$,
$\sqrt{2}$ ou $\sqrt{\pi}$, que estão na mesma vizinhança numérica
($\alpha\varphi = 0{,}01181$, $\alpha\sqrt{2} = 0{,}01032$)?  A resposta é que
$\sqrt{e}$ é \emph{selecionado por princípio}, não por ajuste.  Na teoria da
informação em base natural (unidade: \textit{nat}, $S = -k_B\sum p_i\ln p_i$),
vale a identidade
\begin{equation}
\ln\bigl(\sqrt{e}\bigr) \;=\; \ln\bigl(e^{1/2}\bigr) \;=\; \tfrac{1}{2}\ \text{nat},
\label{eq:meio-nat}
\end{equation}
isto é, $\sqrt{e}$ é o fator de magnitude correspondente a \textbf{exatamente
meio nat de informação} --- o custo entrópico mínimo de uma operação de paridade
fronteira$\leftrightarrow$bulk (um \emph{flip} binário irredutível).
É o análogo holográfico do limite de Landauer: assim como apagar 1 bit clássico
custa no mínimo $k_B T\ln 2$, projetar 1 estado holográfico da fronteira ao bulk
custa no mínimo $\tfrac{1}{2}$ nat.  Os candidatos $\varphi$, $\sqrt{2}$,
$\sqrt{\pi}$ não correspondem a custo informacional algum --- só $\sqrt{e}$ tem
a leitura de meia operação de paridade.  Na forma quadrática, a seleção é ainda
mais limpa: $\betatgl^{2} = \alpha^{2}\,e$, sem raízes, ligando a autointeração
eletromagnética ($\alpha^2$) ao custo entrópico ($e$) diretamente.  A
proveniência completa (três derivações independentes que convergem a
$\betatgl \approx 0{,}012$ \emph{antes} da fatoração) é dada na
Seção~\ref{sec:beta-posicionamento}.
````

### 3. `Um (absoluto) — Grande Atrator/um_absoluto_pt.txt` — bytes 595.896–597.701

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_pt.txt
- sha256 of the file: `9dd16cf0bca80afb88ac9c6635eaba1ac338f5a1327f52b0836c3eb1884d412e` (computed now; this file is not in the seal map) · of this excerpt: `584f52c3d1e820ebd6ba83a1f54a5c117238604869cef18d10e6b981b342359c` · line endings shown as LF
- status · estatuto: [REAL]; valor α-livre [OPEN]
- why · por quê: Seção do Artigo C (v368) que separa o piso térmico (esfria) do geométrico (não esfria), com a temperatura de cruzamento calculada e o valor α-livre como muro aberto; corresponde à chave do selo geometric_cost_of_absolute_zero.

````text
== O custo geométrico do zero absoluto ==
O operador nomeou esta pedra, e o nome mudou o seu conteúdo, não a sua etiqueta: se uma pedra se chama o custo geométrico do zero absoluto, ela precisa provar o que distingue esse custo do térmico. A onda anterior estabeleceu que o custo EXISTE --- dispositivo muitos-para-um é logicamente irreversível e o piso de Landauer é estritamente positivo enquanto houver temperatura. Mas o piso de Landauer é TÉRMICO: esfriar o encolhe sem limite, e ele só não se anula porque o zero absoluto não se alcança. Esta onda prova a diferença. Para todo epsilon positivo existe temperatura positiva em que o piso térmico já está abaixo dele, logo o piso térmico NÃO é o fundo; ao passo que um custo que NÃO depende da temperatura e é estritamente positivo SOBREVIVE ao limite --- não se esfria o que não é térmico. Sua origem geométrica é a Meia-Nat da entropia de fronteira: o volume mínimo excede a unidade e o fator de redução associado fica estritamente entre zero e um, nem gratuito nem aniquilante, e esta face da cadeia não carrega constante de estrutura fina alguma. Na bancada os dois pisos são trazidos à mesma escala honestamente, como energias: a fração geométrica de um elétron-volt é comparada ao piso de Landauer, e a temperatura de cruzamento é calculada em cerca de duzentos e um kelvin, encaixada entre duas temperaturas reais de laboratório --- acima dela, à temperatura ambiente, o piso térmico ainda excede o custo geométrico; abaixo dela, no nitrogênio líquido, já não excede. O cruzamento, portanto, não é vazio nem infalsificável. O que permanece próprio da teoria é o VALOR e a sua identificação física, e a determinação alfa-livre desse valor segue sendo o muro aberto. O gate não se move.
````

### 4. `Um (absoluto) — Grande Atrator/um_absoluto_pt.txt` — bytes 68.425–69.043

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_pt.txt
- sha256 of the file: `9dd16cf0bca80afb88ac9c6635eaba1ac338f5a1327f52b0836c3eb1884d412e` (computed now; this file is not in the seal map) · of this excerpt: `5948f321d7bcc88d860001d1d3dad0827412a0b900000b77b5380be6585d2fbf` · line endings shown as LF
- status · estatuto: [REAL]
- why · por quê: A forma conservada vigente: α_abs² = q² + α_obs² = 1, β_TGL = √e·√(1−q²); o motor é α_abs=1 → q → α e não R_∂=1/α_CODATA; o CODATA entra só na validação final.

````text
\;α_abs^2=q^2+α_obs^2=1\;, α_obs=√1-q^2, β_TGL=√ e\,√1-q^2.

$α_obs$ é a componente luminosa residual da unidade absoluta após a polarização térmica $q^2$ do zero modular. A constante deixa de ser ``um número externo'' e vira a componente projetiva de uma identidade conservada. O motor da cadeia é $α_abs=1→ q→α=√1-q^2$ --- não $R_∂=1/α_CODATA$. O CODATA entra só na validação final: $q_QED=√1-α_QED^2=0.9999734$, $χ_QED=2 artanhq_QED=11.2268$ (resíduo de conservação $0$). O zero modular não destrói o Um; decompõe-o em resistência térmica $q$ e corrente luminosa $α$.
````

---

*Generated by script from the published files of the repository (https://github.com/rotolimiguel-iald/the_boundary); numbers read from the seal v368, the kernel manifest and ESTADO_ATUAL.md. PROVED ≠ CONFIRMED; NOT_FALSIFIED is never CONFIRMED. Nothing here instructs a reader how to respond to anything.*
