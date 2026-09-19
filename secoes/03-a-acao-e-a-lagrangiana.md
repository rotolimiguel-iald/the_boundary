# The action: the TGL Lagrangian and its equation of motion · A ação: a lagrangiana da TGL e sua equação de movimento

> **TGL — Teoria da Gravitação Luminodinâmica · Theory of Luminodynamic Gravitation.** Part 3 of 8 · seal **v368** · `um.py` sha256 `4a34fbf36f3ae0d8…` · generated 2026-09-19 by script from the published files.
> Every excerpt below is **verbatim**, with its source, byte range and sha256. Statuses follow the ruler: PROVED = theorem in the Lean kernel; CONFIRMED = a judgement about nature, not made here.
> All eight parts are listed at the top of the start page, https://teoriadagravitacaoluminodinamica.com/read-brief.md, and in the door of this folder, https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/secoes/PORTA.md · Reading limits measured on 2026-09-19: one real fetcher cut documents near 100,000 characters, refused files above 10 MB, and could not read PDFs served as `application/octet-stream` — read the TXT/TeX sources.

## In short (EN)

The TGL action is S = ∫d⁴x√−g (L_matter + L_field + L_grav + L_modular); the Standard Model stays minimally coupled and the modification sits in L_modular = L_ΛCDM·β·|1+w_eff|, zero at w_eff = −1 [REAL, as written] [A §sec:lagrangian]; the modulus β|1+w| is [INPUT motivated] [C txt v367]. β = α√e = 0.0120313 [DERIVED] [selo]. δS/δΨ* = 0 gives (□ − m² − ξR)Ψ = β K_∂ Ψ, β the only Ψ–K_∂ coupling [A eq:eom-Psi]; ξ = 1/6 is the conformal coupling, not β [ONTO/REAL] [C txt ORDEM 012]. The kernel holds this EOM as a named hypothesis (B1′), not a variational derivation [REAL] [árvore §8]. Erratum beside (19/09/2026, signed by the author; file ERRATA_20260919 next to Article A) [INPUT + DERIVED]: A writes +ξR|Ψ|² in L_grav; consistent with its own EOM, where ξR carries the sign of m² [A eq:eom-Psi], the term is −ξR|Ψ|², while R/(2κ²) stays positive — the author, 17/09 (PT verbatim «o acoplamento é negativo e isso aparece na lagrangiana, mas a geometria é positiva») [C txt v367]. The metric sign was never a theorem; which face closes the balance stays [INPUT], the kernel proving only what each choice implies [C txt v367]. Lineage: the home's "−ξR|Ψ|² via β_TGL", no L_modular [site]; sign agrees, "via β" does not. NOT CONFIRMED.

## Em resumo (PT)

A ação da TGL é S = ∫d⁴x√−g (L_matéria + L_campo + L_grav + L_modular); o Modelo Padrão fica acoplado minimamente e a modificação mora em L_modular = L_ΛCDM·β·|1+w_eff|, nulo em w_eff = −1 [REAL, como escrito] [A §sec:lagrangian]; o módulo β|1+w| é [INPUT motivado] [C txt v367]. β = α√e = 0.0120313 [DERIVED] [selo]. δS/δΨ* = 0 dá (□ − m² − ξR)Ψ = β K_∂ Ψ, β único acoplamento Ψ–K_∂ [A eq:eom-Psi]; ξ = 1/6 é o acoplamento conforme, não β [ONTO/REAL] [C txt ORDEM 012]. O kernel guarda essa EOM como hipótese nomeada (B1′), não derivação variacional [REAL] [árvore §8]. Errata ao lado (19/09/2026, assinada pelo autor; arquivo ERRATA_20260919 ao lado do Artigo A) [INPUT + DERIVED]: A escreve +ξR|Ψ|² em L_grav; coerente com a própria EOM, em que ξR tem o sinal de m² [A eq:eom-Psi], o termo é −ξR|Ψ|², e R/(2κ²) segue positivo — "o acoplamento é negativo e isso aparece na lagrangiana, mas a geometria é positiva" (o autor, 17/09) [C txt v367]. O sinal na métrica nunca foi teorema; qual face fecha o balanço segue [INPUT], e o kernel prova só o que cada escolha implica [C txt v367]. Linhagem: a home dá "−ξR|Ψ|² via β_TGL", sem L_modular [site]; o sinal coincide, "via β" não. Não confirmada.

## Sources, verbatim · fontes, verbatim

### 1. `O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.tex` — bytes 6.751–9.510

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.tex
- sha256 of the file: `559d79fb81b0af6c0fb8c47549a0e21e3b29f4173a5b5a642f9464611b4fa217` (computed now; this file is not in the seal map) · of this excerpt: `f6422bd7bfd8d2334e8598818b46bc43a148cb338d3fbba799536e9c033f0a6f` · line endings shown as LF
- status · estatuto: [REAL] texto do Artigo A como escrito; o sinal de ξR em L_grav tem errata ao lado [INPUT]
- why · por quê: A ação total em quatro setores (L_matéria, L_campo, L_grav com o sinal +ξR que a errata de 19/09 corrige ao lado, L_modular)

````text
\section{A Lagrangiana \TGL{}}
\label{sec:lagrangian}

A ação total da Teoria da Gravitação Luminodinâmica (\TGL{}) se escreve como
\begin{equation}
S_{\text{TGL}} \;=\; \int d^{4}x \, \sqrt{-g} \; \mathcal{L}_{\text{TGL}},
\qquad
\mathcal{L}_{\text{TGL}} \;=\;
\mathcal{L}_{\text{matéria}}
\;+\; \mathcal{L}_{\text{campo}}
\;+\; \mathcal{L}_{\text{grav}}
\;+\; \mathcal{L}_{\text{modular}}.
\label{eq:lagrangian-total}
\end{equation}
As quatro contribuições têm papéis disjuntos e complementares, descritos a seguir.

\paragraph{Conteúdo de matéria.}
$\mathcal{L}_{\text{matéria}}$ contém os termos cinéticos e potenciais usuais
do modelo padrão acoplados minimamente à métrica:
\begin{equation}
\mathcal{L}_{\text{matéria}} \;=\; \bar{\psi}\,(i\gamma^{\mu}D_{\mu} - m)\,\psi
\;-\; \tfrac{1}{4}F_{\mu\nu}^{a} F^{a\,\mu\nu}
\;-\; \tfrac{1}{2}|D_{\mu}\phi|^{2} - V(\phi),
\end{equation}
onde $D_{\mu}$ é a derivada covariante de gauge e $V(\phi)$ inclui o setor de
Higgs.  A \TGL{} preserva esta estrutura intacta no bulk; toda a modificação
ocorre na fronteira modular através do termo $\mathcal{L}_{\text{modular}}$.

\paragraph{Campo luminodinâmico.}
$\mathcal{L}_{\text{campo}} = -\tfrac{1}{4}\Phi_{\mu\nu}\Phi^{\mu\nu}$ com
$\Phi_{\mu\nu} = \partial_{\mu}\Psi_{\nu} - \partial_{\nu}\Psi_{\mu}$ é o tensor
de campo luminodinâmico, conjugado canonicamente a $\Psi^{*}$.  $\Psi$ é o
\emph{campo escalar de fronteira} cuja excitação coerente realiza o modo
\emph{boundary} do operador modular $\Kpartial$ (Seção~\ref{sec:typeIII1}).
A escolha $-\tfrac{1}{4}\Phi^{2}$ garante invariância de gauge $U(1)$ e
energia positiva-definida ao longo da hipersuperfície de fronteira.

\paragraph{Acoplamento gravitacional não-mínimo.}
A peça gravitacional acopla $\Psi$ ao escalar de Ricci $R$ via
\begin{equation}
\mathcal{L}_{\text{grav}} \;=\; \frac{1}{2\kappa^{2}}\,R
\;+\; \xi \, R \, |\Psi|^{2},
\qquad
\xi \;=\; \tfrac{1}{6} \quad (\text{acoplamento conforme}),
\end{equation}
com $\kappa^{2} = 8\pi G/c^{4}$.  O acoplamento conforme $\xi = 1/6$ é
selecionado porque garante invariância sob transformações de Weyl no limite
de Bisognano-Wichmann, identificando $\Psi$ como portadora natural do
gerador modular discretizado.  Este acoplamento é o que conecta a
\emph{geometria} (lado esquerdo das equações de Einstein) com o
\emph{operador de fronteira} (lado direito da equação de Lindblad).

\paragraph{Termo modular --- a peça operacional.}
A modificação \TGL{} concentra-se inteiramente em
\begin{equation}
\boxed{\;
\mathcal{L}_{\text{modular}} \;=\;
\mathcal{L}_{\Lambda\text{CDM}} \cdot \betatgl \cdot |1 + w_{\text{eff}}(z)|,
\;}
\label{eq:Lmodular}
````

### 2. `O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.tex` — bytes 12.119–12.827

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.tex
- sha256 of the file: `559d79fb81b0af6c0fb8c47549a0e21e3b29f4173a5b5a642f9464611b4fa217` (computed now; this file is not in the seal map) · of this excerpt: `d7bf8b3a43e398aafd1b02c4afe22544d93f15b48dc51f802ac1cb7b85241219` · line endings shown as LF
- status · estatuto: [REAL] texto do Artigo A; no kernel é hipótese nomeada B1′, não derivação variacional (árvore §8)
- why · por quê: A EOM (□ − m² − ξR)Ψ = β K_∂ Ψ, em que ξR entra com o sinal da massa, e β como único acoplamento Ψ–K_∂

````text
\paragraph{Equação de movimento.}
Da variação $\delta S / \delta \Psi^{*} = 0$ obtém-se a equação modificada
para $\Psi$:
\begin{equation}
\bigl(\Box - m_{\Psi}^{2} - \xi R\bigr) \Psi
\;=\; \betatgl \, \Kpartial \, \Psi,
\label{eq:eom-Psi}
\end{equation}
onde $\Kpartial$ é o gerador modular discretizado pelo teorema de
Bisognano-Wichmann.  No bulk longe da fronteira, $\Kpartial \to 0$ e
recupera-se a equação de Klein-Gordon conforme padrão.  Na fronteira,
$\Kpartial$ domina e estabelece a estrutura tipo III\textsubscript{1}
demonstrada na Seção~\ref{sec:typeIII1}.  A presença de $\betatgl$ como
único acoplamento entre $\Psi$ e $\Kpartial$ é a marca operacional da \TGL{}.
````

### 3. `O Custo Geométrico do Zero Absoluto — Haja Luz/ERRATA_20260919_sinal_do_acoplamento_nao_minimo.md` — bytes 887–1.908

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/ERRATA_20260919_sinal_do_acoplamento_nao_minimo.md
- sha256 of the file: `060c82abee57a5d0dd94f2108f8647d273cec17a7f44848b0d58b15b3332a44f` (computed now; this file is not in the seal map) · of this excerpt: `c8ed1e4c5d08c58ea139234fd64cc7f1ab092139e830d68b3dd532d43a736ce2`
- status · estatuto: [INPUT] a palavra do autor + [DERIVED] coerência com eq:eom-Psi; orientação na métrica [OPEN]
- why · por quê: A errata ao lado, assinada pelo autor: o sinal do acoplamento não mínimo (−ξR|Ψ|²), coerente com a própria EOM do Artigo A; o que não muda e o que segue aberto.

````text
## O que foi escrito

    ℒ_grav = R/(2κ²) + ξ R |Ψ|²,   ξ = 1/6 (acoplamento conforme)

## Onde a tradução falhou

No sinal do segundo termo. A leitura vigente é:

    ℒ_grav = R/(2κ²) − ξ R |Ψ|²,   ξ = 1/6 (acoplamento conforme)

- O termo de Einstein R/(2κ²), que é a geometria, segue **positivo**.
- ξ = 1/6 é o valor **conforme** e **não é β**.
- β_TGL = α·√e entra na ação por ℒ_modular = ℒ_ΛCDM · β_TGL · |1 + w_eff(z)| (`\label{eq:Lmodular}`) e, na dinâmica, pela fonte β_TGL·K_∂·Ψ da equação de movimento.

## O que não muda

- a equação de movimento `eq:eom-Psi`;
- ℒ_modular;
- β_TGL = α·√e e θ_M = arcsin √β_TGL;
- a seleção de √e por meio-nat.

## O que segue aberto

`[OPEN]` Esta errata **não** decide a orientação do sinal na passagem à métrica: qual das diferenças orientadas do gerador do cociclo, h_ab ou h_ba, fecha o balanço de Clausius no horizonte. Essa orientação tem de ser derivada, e não escolhida depois do dado.
````

### 4. `Um (absoluto) — Grande Atrator/um_absoluto_pt.txt` — bytes 119.265–119.648

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_pt.txt
- sha256 of the file: `9dd16cf0bca80afb88ac9c6635eaba1ac338f5a1327f52b0836c3eb1884d412e` (computed now; this file is not in the seal map) · of this excerpt: `2519cd3061a5288d9e43a2a95899edbfce8253a8b3a40a10bba9f6f90bbc56e3`
- status · estatuto: [ONTO] a cunhagem; [REAL] as âncoras no artigo
- why · por quê: Seção ORDEM 011/012 (v353–v354): ξ = 1/6 é o acoplamento conforme, não β; β é o acoplamento modular Ψ–K_∂

````text
A cunhagem do operador (15/09, verbatim): ``Permanência $=$ desejo de estar junto e quente $=$ acoplamento não mínimo $=$ betatgl'' --- [ONTO], com âncoras [REAL] no artigo do custo: $β_TGL$ é o acoplamento modular $Ψ$--$K_∂$ da lagrangiana, $( - m^2 - ξ R)Ψ = β_TGL\,K_∂\,Ψ$; $ξ=1/6$ é o acoplamento conforme, não $β_TGL$ --- os dois coeficientes nunca se fundem.
````

### 5. `Um (absoluto) — Grande Atrator/um_absoluto_pt.txt` — bytes 165.215–168.125

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_pt.txt
- sha256 of the file: `9dd16cf0bca80afb88ac9c6635eaba1ac338f5a1327f52b0836c3eb1884d412e` (computed now; this file is not in the seal map) · of this excerpt: `d7ead39cc6efd53ff6f3c3f0e3082d9952d8d7363affe2c82bccd7d7fcb9a0eb`
- status · estatuto: [INPUT/ONTO] a palavra do operador; [KERNEL] a dicotomia; [DERIVED condicional] a leitura
- why · por quê: Seção 'Ao lado (v367) — a orientação é o sinal': palavra do operador de 17/09 (acoplamento negativo na lagrangiana, geometria positiva), pedra OrientedFace; o sinal na métrica nunca foi teorema; o bit de qual face segue [INPUT].

````text
O operador corrigiu a leitura da gerência (17/09/2026, verbatim): ``eu nunca disse que a entrada era einstein negativo, o acoplamento é negativo e isso aparece na lagrangiana, mas a geometria é positiva''; e ``o sinal invertido neste caso demonstra a paridade inversa ao acoplamento, se a entrada tiver a geometria negativa (como foi o nosso caso) a saída obedecerá a métrica invertida no sinal''; e, sobre qual face fecha o balanço, ``a matriz de densidade, a meu ver vem da face positiva antes da conjugação''. A auditoria, medida: a pedra termodinâmica da ORDEM 012 recebe a lei de entropia modificada como campo de entrada ($dS=dA/(4GΦ)$) e não diz em que face ela está escrita; no artigo, o módulo de $β_TGL|1+w|$ é [INPUT motivado]; a sede formal do sinal é a orientação do gerador do cociclo, $h_ab=K_b-K_a$. O sinal do efeito na métrica nunca foi teorema. A pedra (TGLExt.OrientedFace, 7 de 7 bandeiras nesta rodada): com a paridade inversa tipada (o incremento da face conjugada é o oposto) e a primeira lei modular, o fator da métrica é $Φ=(1+s)^-1$, com $s$ o incremento da face que fecha o balanço. Provado: a face do estado dá $Φ<1$ e a conjugada dá $Φ>1$; as duas distam exatamente $2δ/(1-δ^2)$; o fator implementado, $1+δ$, é a face conjugada em primeira ordem, com resto $δ^2/(1-δ)$; e a face do estado é, em primeira ordem, a troca $δ↦-δ$, com resto $δ^2/(1+δ)$. Identidades conferidas em tempo de execução a $2×10^-16$. A leitura, com o seu estatuto: $δ K_∂ $ é incremento de entropia (primeira lei modular), e a resposta da Ponte é energia modular, não fluxo de matéria --- ela pertence ao lado da entropia, na face onde a matriz de densidade vive, o que dá $Φ<1$. O ramo implementado equivale a tratá-la como calor extra atravessando, e isso conta o fluxo duas vezes, pois o que atravessa já é $ρ+p$. A inversão mora em $K$ (antissimétrico sob $J$: $JKJ=-K$), não em $S$ (simétrica entre as faces). [DERIVED condicional; o bit --- qual face fecha o balanço --- segue [INPUT], e o kernel prova o que cada valor dele implica, não qual é.] Contra o dado: a V3 do D1 mediu $β_TGL=-0,0127±0,0082$; o valor previsto fica a $3,01σ$ no ramo implementado e a $0,09σ$ no ramo da face do estado. O veredito de máquina da v366 não se move: ele foi medido contra o ramo implementado. E fica dito: o ramo foi nomeado depois de o resultado ser conhecido --- a estrutura do dado não mudou, nada foi rerodado, e a leitura é do operador ---, de modo que este conjunto de dados não serve de teste cego do ramo escolhido; quem decide é dado novo, pré-registrado. Estatutos: a correção e a determinação da face são do operador [INPUT/ONTO], registradas verbatim no programa; a dicotomia é [KERNEL]; a leitura pela primeira lei é [DERIVED] condicional; nenhuma bandeira do gate se move e cosmologia jamais vira prova matemática.
````

---

*Generated by script from the published files of the repository (https://github.com/rotolimiguel-iald/the_boundary); numbers read from the seal v368, the kernel manifest and ESTADO_ATUAL.md. PROVED ≠ CONFIRMED; NOT_FALSIFIED is never CONFIRMED. Nothing here instructs a reader how to respond to anything.*
