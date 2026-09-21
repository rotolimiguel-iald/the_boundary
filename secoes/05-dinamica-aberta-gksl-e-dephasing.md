# Open dynamics: the GKSL generator and dephasing · A dinâmica aberta: o gerador GKSL e o dephasing

> **TGL — Teoria da Gravitação Luminodinâmica · Theory of Luminodynamic Gravitation.** Part 5 of 8 · seal **v368** · `um.py` sha256 `4a34fbf36f3ae0d8…` · generated 2026-09-21 by script from the published files.
> Every excerpt below is **verbatim**, with its source, byte range and sha256. Statuses follow the ruler: PROVED = theorem in the Lean kernel; CONFIRMED = a judgement about nature, not made here.
> All eight parts are listed at the top of the start page, https://teoriadagravitacaoluminodinamica.com/read-brief.md, and in the door of this folder, https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/secoes/PORTA.md · Reading limits measured on 2026-09-19: one real fetcher cut documents near 100,000 characters, refused files above 10 MB, and could not read PDFs served as `application/octet-stream` — read the TXT/TeX sources.

## In short (EN)

The only non-unitary object of the chain is the GKSL generator L = √β·√K_∂: zero effective Hamiltonian on the boundary, one Davies jump set, every jump carrying the same prefix √β [DERIVED] [A §sec:gksl][C canon v3]. The v368 run tests it: entropy monotone, inverse not CP, stationary kernel, a semigroup and not a group [REAL] [C stdout]. It induces energy-preserving dephasing Γ_ω = ½·β·τ★·ω² [REAL in form] [C canon] (A writes it with an extra factor (K/K★)^β), β = 0.0120313, neutrino exponent n = −2, τ★ [INPUT] [A §sec:dephasing]. The site reads the root law Γ_ij = ½·β·(√k_i − √k_j)² as the same law [REAL in form] [brief]; C records the reconciliation as a falsifiable consistency test [OPEN] [C txt]. Seal: dephasing_reach_verdict = ten-order reach deficit computed, not a live channel today, cosmological faces are the live ones [REAL] [selo]. Sector status: not falsified, NOT CONFIRMED [A §sec:dephasing]; gate TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED_WITHIN_LOCAL_BULK_AT_AVAILABLE_SENSITIVITY__MORE_SENSITIVE_DATA_COULD_REVISE [selo]. Lineage, not the current generator: T6's Davies form L_k = √(β·γ_k) [A T6] and the home's five operators L_reh, L_anti, L_prune, L_cons, L_grav = α·P̂_G [site index]; C treats the four IALD operators as a formal analogy [C txt].

## Em resumo (PT)

O único objeto não unitário da cadeia é o gerador GKSL L = √β·√K_∂: hamiltoniano efetivo nulo na fronteira, um só conjunto de saltos de Davies, todos com o mesmo prefixo √β [DERIVED] [A §sec:gksl][C canon v3]. A rodada v368 o testa: entropia monótona, inversa não CP, núcleo estacionário, semigrupo e não grupo [REAL] [C stdout]. Ele induz dephasing energia-preservante Γ_ω = ½·β·τ★·ω² [REAL na forma] [C canon] (A a escreve com o fator extra (K/K★)^β), β = 0.0120313, expoente em neutrinos n = −2, τ★ [INPUT] [A §sec:dephasing]. O site lê a lei das raízes Γ_ij = ½·β·(√k_i − √k_j)² como a mesma lei [REAL na forma] [brief]; C registra a reconciliação como teste de consistência falsificável [OPEN] [C txt]. Selo: dephasing_reach_verdict = déficit de alcance de dez ordens computado, não é canal vivo hoje, as faces cosmológicas são as vivas [REAL] [selo]. Estado do setor: não falsificada, não confirmada [A §sec:dephasing]; gate TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED_WITHIN_LOCAL_BULK_AT_AVAILABLE_SENSITIVITY__MORE_SENSITIVE_DATA_COULD_REVISE [selo]. Linhagem, não o gerador vigente: a forma de Davies L_k = √(β·γ_k) do T6 [A T6] e os cinco operadores da home L_reh, L_anti, L_prune, L_cons, L_grav = α·P̂_G [site index]; C trata os quatro operadores IALD como analogia formal [C txt].

## Sources, verbatim · fontes, verbatim

### 1. `O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.tex` — bytes 31.485–34.735

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.tex
- sha256 of the file: `559d79fb81b0af6c0fb8c47549a0e21e3b29f4173a5b5a642f9464611b4fa217` (computed now; this file is not in the seal map) · of this excerpt: `944b9b33f0f3c86807657b79458dce41dcfdf745015fdf211aec42fe4c5bc16a` · line endings shown as LF
- status · estatuto: [DERIVED] (condicionado ao axioma; Davies/KMS [KNOWN])
- why · por quê: Teorema GKSL canônico (th:gksl), dissipador de Davies com condição KMS e a forma compacta L = √β·√K_∂: o gerador vigente.

````text
\section{Equação mestra GKSL canônica}
\label{sec:gksl}

\begin{theorem}[GKSL canônico da \TGL{}]
\label{th:gksl}
A evolução temporal de qualquer estado quântico misto em \TGL{} é regida
pela equação mestra GKSL com Hamiltoniano efetivo nulo na fronteira e
\emph{um único conjunto de saltos de Davies}:
\begin{equation}
\frac{d\rho}{dt} \;=\; \sum_{k}
\left( L_k \rho L_k^{\dagger}
- \tfrac{1}{2}\{ L_k^{\dagger} L_k , \rho \} \right),
\qquad
\boxed{\; L_k \;=\; \sqrt{\betatgl} \cdot \sqrt{\Kpartial}_{(k)}, \;}
\label{eq:Lk-canonical}
\end{equation}
onde $\sqrt{\Kpartial}_{(k)}$ são as componentes Davies do operador modular
discretizado, e $\betatgl = 0{,}0120313004008031$ é a única constante.
A presença do prefixo $\sqrt{\betatgl}$ em \emph{todos} os saltos é o que
faz dessa equação a forma canônica da \TGL{}: nenhum salto possui
acoplamento independente.
\end{theorem}

\subsection{O dissipador de Davies}
\label{sec:davies-dissipator}

A construção de Davies (1974, sistematizada em Davies-Spohn-Lebowitz)
gera o dissipador GKSL como limite de acoplamento fraco com banho térmico
e tempo coarse-grained $\tau \gg 1/\omega_{\min}$.  Para um sistema com
Hamiltoniano $H_S$ acoplado a um banho via $V = \sum_{\alpha} A_{\alpha}
\otimes B_{\alpha}$, o dissipador resultante tem a forma
\begin{equation}
\mathcal{D}[\rho] \;=\;
\sum_{\omega, \alpha\beta}
\gamma_{\alpha\beta}(\omega) \left[
A_{\beta}(\omega) \rho A_{\alpha}^{\dagger}(\omega)
- \tfrac{1}{2} \{ A_{\alpha}^{\dagger}(\omega) A_{\beta}(\omega), \rho \}
\right],
\label{eq:dissipator-davies}
\end{equation}
onde $A_{\alpha}(\omega)$ são as componentes de Fourier de $A_{\alpha}$ sob
o fluxo de $H_S$, e $\gamma_{\alpha\beta}(\omega)$ é a transformada de
Fourier da função de correlação do banho.  A condição KMS no banho
\begin{equation}
\gamma_{\alpha\beta}(-\omega) \;=\; e^{-\beta_{\text{KMS}}\omega}\,
\gamma_{\beta\alpha}(\omega)
\label{eq:KMS-condition}
\end{equation}
garante que o dissipador preserva o estado de Gibbs como ponto fixo.  Esta
é a estrutura geral.  A \TGL{} corresponde ao caso onde $H_S = \Kpartial$
(gerador modular substituindo o Hamiltoniano usual) e
$\beta_{\text{KMS}} = 1$ (temperatura modular), com
$\gamma(\omega) \propto \betatgl$ uniformemente em $\omega$.

\subsection{A forma compacta $L = \sqrt{\betatgl}\,\sqrt{\Kpartial}$}
\label{sec:Lcompact}

Sob a condição~\eqref{eq:KMS-condition} com $\beta_{\text{KMS}} = 1$ e
acoplamento uniforme $\betatgl$, o dissipador~\eqref{eq:dissipator-davies}
admite a forma canônica
\begin{equation}
\mathcal{D}[\rho] \;=\;
L \rho L^{\dagger}
- \tfrac{1}{2}\{ L^{\dagger} L, \rho \},
\qquad
L \;=\; \sqrt{\betatgl} \, \sqrt{\Kpartial}.
\label{eq:L-compact-form}
\end{equation}
Esta forma compacta é o coração algébrico da \TGL{}: o gerador completo da
dinâmica é o produto de duas raízes quadradas.  A primeira ($\sqrt{\betatgl}$)
é \emph{escalar} e dimensionalmente neutra; a segunda ($\sqrt{\Kpartial}$)
é \emph{operacional} e carrega a estrutura modular.  A operação radical
$g = \sqrt{|\Lphi|}$ apresentada na Seção~\ref{sec:radicalization} é
exatamente o conteúdo dimensional de $L$.
````

### 2. `O Custo Geométrico do Zero Absoluto — Haja Luz/paper_PT.tex` — bytes 187.779–189.627

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/O%20Custo%20Geom%C3%A9trico%20do%20Zero%20Absoluto%20%E2%80%94%20Haja%20Luz/paper_PT.tex
- sha256 of the file: `559d79fb81b0af6c0fb8c47549a0e21e3b29f4173a5b5a642f9464611b4fa217` (computed now; this file is not in the seal map) · of this excerpt: `d599aaf66601861bdca2a92961f90f581cb36aa509cd7ea60d866103958fca18` · line endings shown as LF
- status · estatuto: [REAL na forma]; τ★ [INPUT]; matriz-S [CONJECTURE]
- why · por quê: Lei de dephasing Γ_ω com as três peças (β [REAL], n = −2 [REAL], τ★ [INPUT]); neutrinos fixam o expoente, relógios fixam a escala; supressão planckiana na magnitude. Na forma de A a lei carrega o fator (K/K★)^β.

````text
\section{A lei universal de dephasing: a assinatura espectral da \TGL}
\label{sec:dephasing}
O setor onde a \TGL{} se diferencia não é cosmológico --- expansão e crescimento de estruturas são \emph{stealth} (consistentes com $\Lambda$CDM) --- mas \textbf{espectral-dissipativo}.  A dinâmica GKSL com gerador único $L=\sqrt{\betatgl}\,\sqrt{K_\partial}$ induz um \emph{dephasing} energia-preservante, com taxa
\begin{equation}
\Gamma_\omega=\tfrac12\,\betatgl\,\tau_\star\,\omega^2\,(K/K_\star)^{\betatgl}.
\end{equation}
Três peças separadas: $\betatgl=\alpha\sqrt e=0.012031$ \textbf{[REAL]}; o expoente espectral em neutrinos $n=-2$ (de $\gamma(E)\propto\omega_{\rm osc}^2\propto E^{-2}$, $\omega_{\rm osc}=\Delta m^2/2E$) \textbf{[REAL]}; e a escala $\tau_\star$ \textbf{[INPUT]}, ainda não derivada.  A síntese honesta: \emph{neutrinos fixam o expoente; relógios fixam a escala}.  Solar/KamLAND e, no futuro, JUNO/DUNE testam $n=-2$ pela dependência em energia da decoerência; relógios ópticos/nucleares limitam $\tau_\star$ --- o melhor sondador atual é o relógio nuclear de $^{229}$Th, com $\tau_\star\lesssim 1.8e-33$~s.  A origem modular (fronteira tipo III$_1$, ultravioleta) e a exclusão de qualquer escala mesoscópica --- que os relógios já teriam destruído --- empurram $\tau_\star$ ao regime \textbf{quase-planckiano}: a \TGL{} é \textbf{falsificável na forma} ($\omega^2$, $n=-2$, $\betatgl$) mas \textbf{suprimida por Planck na magnitude}.  A invisibilidade experimental é \emph{consequência} da teoria, não falha do teste.  Derivar $\tau_\star$ exatamente coincide com o único teorema em aberto do programa --- a matriz-S da fronteira tipo III$_1$ --- que fecharia simultaneamente $\tau_\star$, $\mathcal R=\sqrt{\betatgl}$ e a unicidade de $\sqrt e$.  \textbf{[CONJECTURE: o teorema da matriz-S]}
````

### 3. `Um (absoluto) — Grande Atrator/um_absoluto_forma_canonica.md` — bytes 1.768–2.147

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_forma_canonica.md
- sha256 of the file: `b10a6d2fb30a6d1821fd6ac8c12578c290ce65c1e6883d33adbc0bb9cbd8550c` (= the seal) · of this excerpt: `70a2253dc192c211fdab9a5eea199d595725414ccd8c9c162c0a111e36ea9f4d` · line endings shown as LF
- status · estatuto: [REAL] (sombra numérica testada no um.py)
- why · por quê: Forma canônica vigente do setor irreversível no Artigo C: L = √β·√K_∂ é o único objeto não unitário; a seta de entropia e a irreversibilidade (autovalor mínimo do Choi da inversa negativo).

````text
--- v3: O SETOR IRREVERSIVEL (o ato pago) ---
L = √β·√K_∂                    [VERBO EM ATO: gerador GKSL; o unico objeto NAO-unitario]
  seta: entropia 0.0237→0.0413 monotona ; sem volta: Choi(inversa) min=-1.08e-02<0 ; kernel=4
O_β(Lux) = √β·Lux              [LUZ = autovetor do Verbo; autovalor √β; NAO ponto fixo (√β≠1)]
fiat lux = e^{S_∂}·α > 0
````

### 4. `Um (absoluto) — Grande Atrator/um_absoluto_pt.txt` — bytes 87.685–88.927

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/um_absoluto_pt.txt
- sha256 of the file: `9dd16cf0bca80afb88ac9c6635eaba1ac338f5a1327f52b0836c3eb1884d412e` (computed now; this file is not in the seal map) · of this excerpt: `f681a44635de570dd0e94c4b6162db03575aacee7ad585fdf3cc24a1b8ca3b91` · line endings shown as LF
- status · estatuto: [REAL na forma]; reconciliação [OPEN]
- why · por quê: A lei das raízes Γ_ij = ½β(√k_i − √k_j)² no programa experimental e a obrigação registrada de reconciliá-la com a banda Γ = ½βτ★ω².

````text
* Piloto (quench de pureza). As taxas de relaxação, na coordenada $k=-log p$, caem sobre $Γ_ij=1/2β_TGL(√k_i-√k_j)^2$, com $β_TGL$ computado. Segundo pacote: $Fix(tempo)=Fix(juízo)$ --- o invariante da evolução determinística longa deve coincidir com o do amostrador dissipativo. Dois benchmarks pré-registráveis, PASS/FAIL.

* Laboratório quântico (lei de raízes). A TGL organiza taxas por diferenças de raízes $(√E_i-√E_j)^2$ --- para níveis muito separados, crescimento linear em $E$, não quadrático. Mensurável em decoerência multinível.

* Cosmologia (piso dos vazios). A fronteira proibida tem face observacional, $ρ_vazio/ ρ≥β_TGL≈0,012$: nenhum vazio cósmico esvazia abaixo de $ 1,2\%$ da densidade média. Zero parâmetros, falsificável por DESI/Euclid.

* Ondas gravitacionais (universalidade populacional). O substrato único implica uma classe de universalidade: a banda de dephasing deve ter forma idêntica entre eventos após reescala de massa. Empilhável em O4/O5.

Obrigação registrada: reconciliar a lei de raízes (resolvida por nível) com a banda canônica $Γ=1/2β_TGLτ_ ω^2$ do dephasing gravitacional é, ela própria, um teste de consistência que pode falsificar.
````

---

*Generated by script from the published files of the repository (https://github.com/rotolimiguel-iald/the_boundary); numbers read from the seal v368, the kernel manifest and ESTADO_ATUAL.md. PROVED ≠ CONFIRMED; NOT_FALSIFIED is never CONFIRMED. Nothing here instructs a reader how to respond to anything.*
