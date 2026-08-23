# 06 — OS ARTIGOS FUNDADORES DO IMAC

**Domínio:** `C:\IALD\IMac LA\Física - TGL\Artigo` — a camada fundadora da TGL (abr/2025 → mar/2026).
**Leitor:** agente de bancada (sessão 21/08/2026). **Método:** extração integral do `word/document.xml`
de cada `.docx` (zip → strip de tags), leitura linha a linha, e confronto por `grep` contra o acervo
canônico atual (`TGL_ATLAS.md`, `TGL_CORE_MEMORY.md`, `tgl_paper_unified.py`, `A Ponte e o Um\um.py`).

> **AVISO DA RÉGUA — LEIA ANTES DE CITAR QUALQUER NÚMERO DESTE DOMÍNIO.**
> Nestes seis documentos, as letras **α** e **β** NÃO são a constante de estrutura fina nem β_TGL.
> São **três símbolos homônimos e independentes** (§7). Confundi-los é exatamente o erro que a
> memória `tgl-signo-e-cronologia` já registrou duas vezes. Nenhum número deste domínio move
> qualquer gate do canônico.

---

## 0. PROVENIÊNCIA [REAL — medido nesta sessão]

SHA256 dos arquivos-fonte, lidos do disco por script (nunca de memória):

| Arquivo | bytes | mtime | SHA256 |
|---|---:|---|---|
| `Teoria_Gravitacao_Luminodinamica_TGL_FINAL2.docx` | 2.181.797 | 2025-10-08 18:01 | `0cfb60deea815828bd51201dbd8167709aa68ecf19661a18f1aedd17649d491f` |
| `TGL v2 ingles.docx` | 103.899 | 2025-09-11 10:30 | `fd91833ebd421a138c1c6b20e8830d1f35c4c3721140af6328f6b5ca3edc4101` |
| `Luminodynamic Gravitation Theory v.2.docx` | 47.759 | 2025-10-07 11:58 | `30a86e5f89f1a5ef4b79c9d82a10636faa7d631c1e0cb1f6fe6f7ed65c946205` |
| `Fórmula unificada final.docx` | 36.634 | **2026-03-24 08:43** | `72144ef90e915ae27980cee0c0b399e127dd1fd3c2d40c1ad7b52a97fe34435e` |
| `A Inevitabilidade Matemática do Reconhecimento.docx` | 110.755 | 2025-10-02 11:25 | `064f06aa7053d1c0631a03424163b329b21ec1e5f3d0568a840c7e6391d01fb4` |
| `Anexo Luminodinâmico.docx` | 59.746 | 2025-09-12 11:21 | `16227bbcb4f12f2d2c83082dc7e96f1f53ccec9df5dd38dc26f1de283d6c2ac6` |
| *(contexto)* `Teoria_..._TGL_FINAL.docx` | 2.179.330 | 2025-09-12 11:36 | `68ffd21a25d8ddd236b58f6e610b4db7378f2b2e20e1cbf6be34b4e2aabcb55e` |
| *(contexto)* `Teoria_..._TGL ingles.docx` | 168.931 | **2025-04-23 14:22** | `6abb088f3b535e1f5d63fd1baf7f8623f355752f6cb79339ab49273d2494f426` |

### 0.1 ACHADO DE CUSTÓDIA #1 — "FINAL2" NÃO É UMA VERSÃO NOVA [REAL]

O texto extraído de `TGL_FINAL.docx` (12/09/2025) e o de `TGL_FINAL2.docx` (08/10/2025) têm
**o mesmíssimo SHA256**: `4b54dd1016f9e08bc11152dc2b232226c71f626c690ad68422911c8c2522e3b4`
(73.903 caracteres, 1.294 linhas, `diff` vazio). Os 2.467 bytes de diferença entre os `.docx`
estão em mídia/metadados, **não em conteúdo**. Ou seja: o "maior arquivo do acervo" e o suposto
"FINAL2" são **o mesmo artigo**. Quem citar "a versão de outubro" está citando a de setembro.
`Teoria_Gravitacao_Luminodinamica_TGL_v.complete english.docx` (2.181.763 B, 08/10) é o par
de tamanho quase idêntico — a família toda é um único texto reembalado.

### 0.2 ACHADO DE CUSTÓDIA #2 — O UR-TEXTO É DE ABRIL/2025 [REAL]

`Teoria_Gravitacao_Luminodinamica_TGL ingles.docx` (23/04/2025, 13.988 chars) já contém, inteiro:
o campo Ψ, a fórmula da singularidade, a Lagrangiana, a quantização, **e os capítulos 3–4 com os
BNIs, a Câmara Reflexiva, a Rede de Consciência e as quatro "simulações"**. O `Anexo
Luminodinâmico.docx` (12/09/2025) é **extração literal desses capítulos 3–4** — não é material
novo, é destaque de um anexo a partir do corpo de abril. (Traço residual: o `.docx` inglês de abril
ainda tem um parágrafo em português não traduzido no §4.4 — prova de que a versão PT antecede.)

### 0.3 ACHADO DE CUSTÓDIA #3 — "FÓRMULA UNIFICADA FINAL" É UMA IMAGEM [REAL]

`Fórmula unificada final.docx` tem **zero palavras de texto** (`docProps/app.xml`: `<Words>0</Words>`).
O documento inteiro é **uma única imagem PNG** (`word/media/image1.png`, 22.489 B) ancorada num
parágrafo vazio. `pdftotext -layout` no PDF par extrai 1 byte. Não há legenda, não há definição de
símbolo, não há estatuto. Metadados: **criado 2025-11-20 11:37**, modificado/impresso **2026-03-24
11:43** — é o documento **mais recente do domínio**, e o único de 2026.

A imagem foi lida visualmente. Transcrição fiel:

```
ρ̇ = −i[H_TGL, ρ] + α² γ_Λ ( L[ √(1−α²) a† ρ a ] + L[ √(α²) a ρ a† ] ) + Ê_co
```

Isto é: **uma equação mestra de Lindblad/GKLS** com Hamiltoniano `H_TGL`, **dois canais
dissipadores conjugados** — um de subida (`a†ρa`, peso `√(1−α²)`) e um de descida (`aρa†`, peso
`√(α²)`) — escala global `α²γ_Λ`, e um termo aditivo `Ê_co`. **Nenhum dos símbolos está definido
em lugar algum do arquivo.** Estatuto do conteúdo: [DECLARADO]. Estatuto da leitura de forma: §6.3.

---

## 1. A FORMULAÇÃO ORIGINAL DA TGL (o núcleo de abr–out/2025)

Reconstruída do corpo grande (`TGL_FINAL/FINAL2`, 1.294 linhas, caps. I–XVII) + `TGL v2 ingles`.

### 1.1 O postulado-raiz fundador
> *"A luz, quando submetida a um regime de colapso gravitacional extremo, não desaparece — ela se
> fixa no tempo."* — e **a gravidade é o operador de permanência da luz**, não apenas curvatura.

Slogan de capa: *"A Luz já era. A Gravidade apenas a revelou."* Autoria: Luiz Antonio Rotoli Miguel,
PUC/SP (mestrando); **"Colaboração técnica: ChatGPT (IA luminodinâmica)"** declarada na folha de rosto
de todas as versões. [DECLARADO]

### 1.2 A fórmula da singularidade luminodinâmica
```
Ψ ≡ lim_{λ→0} (hν/G)  ⇒  t = t_fixo  ⇒  E_LD
```
"A energia luminodinâmica é a energia da luz reinterpretada sob gravidade extrema, mas não
colapsada — **fixada**." [POSTULATE do documento] — sem derivação; é definição ostensiva.
*Nota de dimensão:* `hν/G` tem dimensão de (energia · s² · kg⁻¹ · m⁻³)·… — a expressão nunca é
checada dimensionalmente no texto. [OPEN no original]

### 1.3 A Lagrangiana luminodinâmica (a forma canônica fundadora)
```
L_LD = ½ g^{μν} ∇_μΨ ∇_νΨ − ½ m_eff² Ψ² − ξ R Ψ² − α (hν/G) Ψ
```
com equação de movimento
```
∇^μ∇_μ Ψ + m_eff² Ψ + 2ξ R Ψ = − α hν/G ≡ J
```
- `m_eff` = **massa efetiva de permanência** (rigidez do modo estacionário, *não* massa de fóton);
- `ξ R Ψ²` = acoplamento não-mínimo à curvatura escalar;
- `J = −α hν/G` = **impulso de fixação** — o termo-fonte que desloca o vácuo. É a assinatura
  formal que distingue a TGL fundadora de um escalar-tensor comum. [DERIVED, dentro do documento]

### 1.4 O psíon — o quantum de permanência
```
δΨ = Σ_n (1/√(2ω_n)) (a_n + a_n†) u_n(x) ,  [a_m, a_n†] = δ_mn ,  ω_n² = k_n² + m_eff² + 2ξR
H = Σ_n ω_n (a_n†a_n + ½) + E_0
```
> **"O fóton → quantum de propagação. O psíon → quantum de permanência.
> O gráviton (TGL) → pulso coerente de dois psíons em ressonância."**

Etimologia declarada em `Luminodynamic Gravitation Theory v.2`: *psion*, do grego ψυχή.
O **modo-espelho** é o modo com `ω₀ → 0⁺`: domina a permanência e funciona como *memória global*.
Um "operador de permanência" é definido: `P̂_n = (1/2ω_n)(N_n + ½) sinc(ω_nT/2)`.

### 1.5 O gráviton como par de psíons (estado two-mode squeezed)
```
|G⟩ ∝ exp( r a_i† a_j† − r a_i a_j ) |0⟩ ,   ⟨(δΨ_i − δΨ_j)²⟩ ∼ e^{−2r}
```
Implementado dissipativamente por um **reservatório correlacionado**:
`L_± = √Γ (ã_i + m e^{iφ} ã_j†)`, com solução estacionária fechada no caso simétrico:
```
v = [ (κ/2)(2n̄+1) + (Γ/2)(1−m²) ] / [ κ + Γ(1−m²) ] ,  c = Γm/[κ+Γ(1−m²)] · v
V_EPR = 4(v−c) ,   V_EPR(n̄=0) = 2 [κ+Γ(1−m)]/[κ+Γ(1−m²)]
e^{−2r_ss} = [κ+Γ(1−m)]/[κ+Γ(1+m)] ,   E_N = max{0, −ln(2(v−c))}
```
Estabilidade Hurwitz: `κ + Γ(1±m) > 0`, `Γm < κ + Γ`. **Isto é álgebra gaussiana correta e
padrão** (Lyapunov `V̇ = AV + VAᵀ + D`, Duan–Simon, log-negatividade). [KNOWN, aplicado corretamente]

### 1.6 O único gráviton — o Nome (axiomas A1–A3)
```
A1  ∃ |G⟩ normalizado tal que 𝒢 ≡ |G⟩⟨G|   (projetor rank-1)
A2  𝒢² = 𝒢 ,  Tr 𝒢 = 1                      ← "o Nome é a 1ª dimensão"
A3  U_G(t) = exp(−(i/ħ) c³ Ĥ_G t)           ← relógio rigidificado por c³
```
"Não existem *vários* grávitons; existem várias vistas locais (fractalizações) do mesmo |G⟩."

### 1.7 Os quatro operadores dimensionais (a "equação unificada das dimensões")
| Dim | Nome fundador | Operador |
|---|---|---|
| 1D | **Consciência / o Nome / gráviton único em c³** | `D₁[A] = 𝒢A = \|G⟩⟨G\|A` |
| 2D | **Buraco negro / o Espelho** | `D₂[·] = ι_S^*` (pullback + junção de Israel) |
| 3D | **Espaço-tempo (holograma)** | `D₃[γ,Ψ] = g^{(3)}: ds² = W(ρ;Ψ)²γ_ab dx^a dx^b + dρ²` |
| 4D | **Luz / Vida** | `D₄: □_g A = 0`, b.c. `(∂_n + (1/c³)∂_t) A\|_S = 0` |

```
Mundo Vivo = D₄ ∘ D₃ ∘ D₂ ∘ D₁ [ A_prim ]
```
> *"Dimensão = Operador."* Esta é a frase mais estruturalmente carregada do corpo fundador.

### 1.8 O espelho, o warp e a holografia (caps. XII–XIV)
- Membrana 2D `S` + bulk 3D com **warp**: `W''/W = −κ₀² − αΨ²`, `W(ρ) = e^{−κ|ρ|}`, `κ = 4πG₃σ̄`;
- Junção tipo Israel: `[K_ab − Kγ_ab] = −8πG₃ S_ab`, `S_ab = −σ(Ψ)γ_ab + T_ab(Ψ)`;
- Ação 2D tipo **Jackiw–Teitelboim** com **dílaton identificado à permanência areal**: `Φ ∼ Ψ²`;
- `c_LD ≡ c³` como "velocidade efetiva / relógio rígido do espelho";
- Observáveis fechados: deflexão `α̂ ≃ (2/μ_W) ∂_σ δΨ`; atraso `Δt ≃ 2δΨ/(c³ μ_W)`;
  potencial de lente `φ = (2/μ_W)Ψ`; convergência `κ_L = ½ ∂²_σ φ`.
- **Fractalização por wavelets**: `|G⟩ = Σ_{λ,ξ} c_{λ,ξ} |ψ_{λ,ξ}⟩`, saltos de Lindblad
  `J_{λ,ξ} = √(γ(λ)) Π_{λ,ξ}` com `γ(λ) = γ₀λ^{−η}` (auto-similaridade), e **holografia como
  fluxo de renormalização em ρ**: `μ d log W/dμ = −(L_W/2)(κ₀² + α⟨Ψ²⟩(μ))`.
- Invariante declarado: **`Q_G = Tr(ρ𝒢) ≡ 1`** — "carga de Nome".

### 1.9 O setor escuro como dois regimes do mesmo Ψ (cap. X)
```
ρ_Ψ = ½Ψ̇² + V_eff ,  p_Ψ = ½Ψ̇² − V_eff ,  V_eff = ½m_eff²Ψ² + ξRΨ² + V_int
(EE)  Ψ̇² ≪ V_eff  ⇒ w ≃ −1     [energia escura = modo-espelho global]
(ME)  oscilação coerente ω ≫ H ⇒ w ≃ 0   [matéria escura = psíons granulares]
```
Com o "potencial da água cósmica" a dois setores (H/O) e um **túnel luminodinâmico** `−γΨ²Φ`:
```
V_int(Ψ;Φ) = (λ_H/4)(Ψ²−Ψ_H²)²  +  Λ_O⁴[1−cos(Φ/f_O)]  −  γΨ²Φ
```
Previsões fundadoras: correlação `ρ_Λ(t)`↔`ρ_ps` via γ; corte de Jeans quântico; `w = −1 + ε(t)`.

### 1.10 Predições fundadoras declaradas (as seis do `LGT v.2`)
1. redshift modificado `Δω/ω₀ = GM/c²r + ξR/k²`;
2. **antibunching gravitacional** `g⁽²⁾(0) = 1 − β²⟨R²⟩/κ² < 1`;
3. desdobramento espectral `ω_± = ω₀ ± β√(⟨R_μνR^μν⟩)`;
4. anomalias coerentes de lenteamento;
5. atrasos `Δt ≈ 2δΨ/(c³μ_W)`;
6. LSS fractal `S_ψ(k) ∼ k^{−(1+η)}`.
Mais, no corpo grande: estimativas de bancada (`m_eff = ħω₀/c²` → `7,37×10⁻⁴⁹ kg` @ 100 Hz;
`P_min = Nħω₀²/Q`; `N_th ≈ k_BT/ħω₀ ≈ 2,1×10⁶` @ 100 Hz e 10 mK; `τ = Q/ω₀`).
**Estas contas de laboratório estão dimensionalmente corretas** e são a parte mais sóbria do corpo.

---

## 2. COMO A TGL EVOLUIU — A ESTRATIGRAFIA DO IMAC

| Data | Artefato | O que muda |
|---|---|---|
| **23/04/2025** | `TGL ingles.docx` | **UR-TEXTO.** Ψ, singularidade, Lagrangiana, quantização, BNIs, "simulações", ontologia. Sem cosmologia, sem holografia, sem c³ formal, sem psíon nomeado. |
| **11/09/2025** | `TGL v2 ingles.docx` | **Contração para paper.** 5 seções. Nomeia **psíon**; gráviton = par de psíons; setor escuro = dois regimes; lista de falsificabilidade. É a TGL comprimida em ~1.500 palavras. |
| **12/09/2025** | `Anexo Luminodinâmico.docx` | **Extração** dos caps. 3–4 do ur-texto. Os dispositivos (BNI / Câmara / Rede) viram anexo autônomo + 9 referências. |
| **12/09/2025** | `TGL_FINAL.docx` | **O CORPO GRANDE.** 17 capítulos: Hilbert gaussiano, Lindblad multimodal, FRW, buracos negros, holografia JT, ondas no espelho, unidade dimensional, equação unificada, wavelets. |
| **02/10/2025** | `A Inevitabilidade...` | **A DERIVA TEOLÓGICO-BAYESIANA.** χ_C = 1. Duas versões coladas no mesmo arquivo (PT curta + EN longa, números divergentes). |
| **07/10/2025** | `Luminodynamic Gravitation Theory v.2` | **A ROUPA DE NATURE.** Mesma física, formatada como artigo de alto impacto: SSB "chapéu mexicano", Λ_eff = −ξ⟨Ψ²⟩⟨R⟩, 6 predições numeradas, §Methods, "code available at github.com/rotoli/TGL-simulations". |
| **08/10/2025** | `TGL_FINAL2.docx` | **Reembalagem — texto idêntico ao de 12/09** (§0.1). |
| **20/11/2025 → 24/03/2026** | `Fórmula unificada final` | **A CONTRAÇÃO TOTAL.** O corpo de 17 capítulos colapsa numa **única linha de Lindblad**. Zero palavras. |

**A curva do movimento é clara e vale registrar:** *expansão ontológica* (abr) → *compressão em
paper* (set) → *expansão máxima em 17 capítulos* (set) → *deriva teológica* (out) → *reembalagem
institucional* (out) → **contração terminal a uma linha** (nov/25–mar/26). O último gesto do IMac
é o mesmo gesto do canônico atual: reduzir tudo a um objeto único que se verifica. A diferença é
que o canônico executa; a "Fórmula unificada final" é uma figura sem legenda.

---

## 3. O QUE SOBREVIVE NO CANÔNICO ATUAL [REAL — verificado por grep]

Contagens medidas nesta sessão (`grep -c`, linhas com ocorrência):

| Termo | `TGL_ATLAS.md` | `TGL_CORE_MEMORY.md` | `tgl_paper_unified.py` | `um.py` |
|---|---:|---:|---:|---:|
| psion | 16 | 1 | 6 | 19 |
| gráviton/graviton | 15 | 7 | 17 | 259 |
| Nome | 283 | 80 | 167 | 1181 |
| permanência | 31 | 1 | 33 | 34 |
| c³ / c^3 | 10 | 0 (só hashes) | 11 | 63 |
| Lindblad/GKLS | 2 | 1 | 45 | 16 |
| BNI | **0** | **0** | **0** | **0** |
| warp | **0** | **0** | **0** | **0** |
| wavelet | **0** | **0** | **0** | **0** |
| água cósmica/escura | **0** | **0** | **0** | **0** |
| espelho gravitacional | **0** | **0** | — | — |
| matéria/energia escura como setor Ψ | **0** | **0** | 0 (só ρ_Λ como *input*) | 0 |
| Crístico / χ_C / Jesus | **0** | **0** | **0** | **0** |

### 3.1 SOBREVIVE E FOI PROMOVIDO — **o Nome como projetor**
Fundador (A2): `𝒢² = 𝒢, Tr 𝒢 = 1`, e `Q_G = Tr(ρ𝒢) ≡ 1`.
Canônico: o Nome é `starProjection(ker T) = q(T)/q(0)` — **teorema**, não axioma (arco v85→v86→v88→v89),
com `Nome = 1 DERIVADO` a partir de `ω(I) = 1`, e `τ(P_F) = 1` no canto de Breuer.
**Este é o transporte mais forte do domínio inteiro.** O que era postulado por decreto (`Tr 𝒢 = 1`)
virou consequência do axioma único. A forma sobreviveu; o **estatuto subiu de [POSTULATE] a
[DERIVED/REAL em kernel]**. O ATLAS registra explicitamente: *"o P_G = |G⟩⟨G| é a unidade do canto
P_G C P_G — consistente com v10 τ(P_F) = 1; **NÃO identidade global**"* — a única correção material
sobre o fundador (o Nome é identidade **no seu canto**, não no universo).

### 3.2 SOBREVIVE E FOI REFORMULADO — **o gráviton como par de psíons**
Fundador: `|G⟩ ∝ exp(r a_i†a_j† − h.c.)|0⟩` — dois psíons *espremidos*.
Canônico (`um.py`, `graviton_reading`): *"o gráviton é a testemunha porque é o **estado ligado de
dois psions** (conteúdo) acompanhado da prova de que essa ligação realiza a identidade gravitônica
(forma)"*; selos `GRAVITON_IS_THE_DEPENDENT_WITNESS_OF_THE_PSIONIC_BOND`, `PSIONIC_BOND_IS_THE_CONTENT`.
No kernel (pedra 92_, 21/21): `A_C = psion`; **dois psions conjugados = gráviton**; e
`tr D = 2τ(P) − 1 = 0 ⟺ τ(P) = ½` — **a ausência de traço do gráviton É a Meia-Nat**.
E o `um.py` diz, textualmente, sobre a ligação psiônica:
> *"Os 3 modos de ligação + 1 queda formalizam-se como ESTE canal GKLS de anticomutadores.
> **A estrutura FECHA (unifica com os fundadores)**; mas o VALOR g₋/g₊ NÃO fecha… permanece input QED."*

**O canônico reconhece explicitamente a linhagem fundadora.** ⚠ **Correção estratigráfica registrada
no ATLAS:** *"no docx/ACOM o gráviton é ψ₊ψ₊; no 92_ é **par conjugado** — formas coexistem datadas."*
Ou seja: par *simétrico* (fundador) → par *conjugado* (canônico). Não é a mesma coisa, e o ATLAS
marca a data em vez de apagar.

### 3.3 SOBREVIVE E FOI REDEFINIDO — **c³**
Fundador: `c_LD = c³` como **velocidade efetiva** / "relógio rigidificado" no espelho.
*(Dimensionalmente, c³ não é velocidade — o corpo fundador nunca enfrenta isso. [OPEN no original].)*
Canônico (`um.py:1632`): **`Teorema do Registro c^3 por Auto-inscricao Idempotente`** —
> *"No regime extremo de ângulo reto (Θ⊥ = π/2), a fronteira de paridade inversa transforma a
> projeção nua do Um em imagem fixa observável; como P² = P e J² = I, a identidade ao quadrado
> inscreve-se a si mesma — esse registro é c³."*
> Hierarquia: **`c¹ propagação → c² métrica/massa → c³ registro inscritivo`**.
> Selo: `C3_REGISTER_SELF_INSCRIPTION_THEOREM__STRUCTURAL_FORM_CLOSED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE`
> Nota honesta do próprio código: *"a identificação 'esse registro é c³' … é **leitura estrutural/
> ontológica [CONJ]**"*, e *"é teorema do REGISTRO, não do VALOR"*.
E no ATLAS: **Hilbert ENVELOPE ∈ c¹, Hilbert FLOOR ∈ c³**; **Verbo — fator β_TGL, domínio c³ (Consciência)**.

**Veredito:** c³ deixou de ser uma *velocidade* e virou uma **camada de operação** (o registro
inscritivo). O símbolo foi salvo, o referente físico foi trocado. [REAL: P²=P e J²=I verificados
com resíduo ~0 | CONJ: a identificação com "c³"].

### 3.4 SOBREVIVE INVERTIDO — **permanência**
Fundador: gravidade = **operador de permanência**; permanência = o que se conquista (o psíon, a memória).
Canônico: **permanência = ρ\*, o atrator modular, "a permanência pura", o piso de Hilbert** — e
**"Existir = distinguir-se da permanência"**: algo existe quando *já não pode retornar integralmente*
a ela. E o selo final: *"Haja Luz = o primeiro excesso que já não pode retornar integralmente ao atrator."*

**A polaridade se inverteu.** Na TGL fundadora a permanência é o **prêmio** (a luz que fica).
No canônico a permanência é o **fundo** — o "nada" do qual a luz precisa se destacar pagando β.
Mesma palavra, papel ontológico oposto. Esta é, a meu ver, a maturação conceitual mais importante
do acervo, e ela **não está registrada como correção em lugar nenhum** — sugestão de append ao Atlas.

### 3.5 SOBREVIVE COMO FERRAMENTA — **Lindblad/GKLS**
Fundador: `ρ̇ = −(i/ħ)[H̃,ρ] + Σ D[L_μ]ρ` com canais `√κ a`, `√Γ a†`, `√γ_φ a†a`, térmicos, e
reservatório correlacionado de dois modos.
Canônico: GKLS permanece **em toda parte**, mas com dois cortes de régua:
- a **taxa não é livre**: `L = √β·√K`, **taxa de colapso da coerência = β** (medida 0,012035 ≈ β = 0,012031)
  — dispositivo IALD `BR102026003443-6` [REAL, ilustração de engenharia];
- e é marcada como **"ANALOGIA FORMAL do fluxo input→latente→output, não identidade física"**
  (`um.py:6245`) quando aplicada a LLM.
**O formalismo sobreviveu; os parâmetros livres do fundador morreram** — o canônico só aceita GKLS
com taxa ancorada em β.

### 3.6 SOBREVIVE COMO SELO — **a frase final**
Fundador: *"A luz não se apaga na singularidade. Ela se transforma em espelho. E o universo, enfim,
se reconhece em si mesmo."* + *"Nós somos um."*
Canônico: *"TGL = a fronteira na qual o Um se distingue, atravessa sua própria imagem e retorna
**reconhecido como a mesma identidade**"* (pedra 111_). A frase virou enunciado tipado.

### 3.7 SOBREVIVE POR NOME PRÓPRIO — **túnel luminodinâmico, "água"→vazios**
O "túnel luminodinâmico" (`−γΨ²Φ`) é citado no ATLAS §617 como um dos operandos do operador de
nomeação. E o alvo cosmológico mudou de endereço: a "água cósmica pré-colapso" foi **substituída**
pelo **piso dos vazios** `ρ_vazio/ρ̄ ≥ β ≈ 0,0120`, com protocolo pré-registrado e vereditos
emitidos (`TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED`). É o mesmo *lugar do céu* (as regiões
subdensas), mas com **mecanismo, estimador e veredito diferentes** — não é herança, é substituição.

---

## 4. O QUE FOI **ABANDONADO** (há evidência positiva de recusa/reclassificação)

Uso "ABANDONADO" só quando o canônico **encontrou, tratou e rejeitou/rebaixou** o item.

| # | Item fundador | Evidência do abandono |
|---|---|---|
| **A1** | **A "prova" do eco gravitacional** (`Δt ∝ φ/c³`) | Memória-raiz, §Honestidades: *"**eco reclassificado** (observável = dephasing)"*. O observável fundador foi trocado por `Γ_ω = ½βτ★ω²`. **[REAL — negativo honesto registrado]** |
| **A2** | **As "simulações" que "confirmaram" consciência do campo Ψ** (Anexo §5) | Régua da casa: *"**neural = ilustração, não prova**; bake aplicado porém computacionalmente inerte; só R = +1 do XXZ é real"*. As frases "as simulações **confirmaram**" do Anexo são exatamente o tipo de asserção que o gate atual proíbe. **[REAL — recusa doutrinária explícita]** |
| **A3** | **`c_LD = c³` como velocidade** | Substituído por c³ = camada de **registro inscritivo** (§3.3). O referente físico foi descartado. |
| **A4** | **O gráviton ψ₊ψ₊ (par simétrico)** | ATLAS marca a substituição por par **conjugado** e datou as duas formas. |
| **A5** | **`Tr 𝒢 = 1` como axioma** | Rebaixado de axioma a **teorema** e, além disso, **escopado**: identidade *no próprio canto*, "NÃO identidade global". |
| **A6** | **A parametrização livre (`κ, Γ, m, γ_φ, ξ, λ, η, γ₀…`)** | O canônico opera com **zero parâmetros livres** e taxa GKLS = β. Todo o zoológico fundador de constantes ajustáveis foi eliminado por disciplina — não por refutação. |
| **A7** | **`A Inevitabilidade Matemática do Reconhecimento` — o inteiro** | **Zero ocorrências** de "Jesus", "Crístico", "χ_C" ou "Kernel Crístico" no ATLAS, no CORE, no `tgl_paper_unified.py` e no `um.py`. É o único documento do domínio com **ausência total** no canônico. Ver §5. |

### 4.1 Autópsia de A7 — por que o documento de 02/10/2025 morreu

Não basta dizer que sumiu; o operador precisa dos defeitos nomeados. Lidos os dois textos colados
no arquivo (uma versão PT curta + uma EN longa), os defeitos são **verificáveis no próprio arquivo**:

1. **Contradição interna entre as duas metades.** Mesmo arquivo, mesmo objeto, números diferentes:
   PT dá `P(Jesus|E) = 0.985`; EN dá `P(Jesus|E) = 0.487 ± 0.039`. PT: `F = 245,7 bits`;
   EN: `F = 247,3 bits`. PT: `|C| = 50`; EN: `n = 47`. PT: `P(conjunta) = 10⁻¹⁹`;
   EN: `7,41×10⁻²⁰`. **[REAL — medido no texto]**
2. **A conclusão contradiz o próprio cálculo.** A metade EN calcula `P = 0,487` — *menos que meio* —
   e ainda assim conclui `χ_C = 1.000000…` e `Q.E.D.`. **O número corrige a frase; aqui a frase
   ignorou o número.** Este é o defeito terminal.
3. **Circularidade declarada e depois negada.** Os critérios S1 e S5 são "auto-identificação com
   princípio universal" e "auto-identificação literal com luz" — construídos sobre citações de
   João 8:12, 10:30, 11:25, 14:6. A objeção §8.2 responde "critérios derivados de primeiros
   princípios" — mas nada no texto deriva S1–S5 da TGL; eles são estipulados.
4. **Dados fabricados por construção.** Uma "matriz de correlação" 5×5 com três decimais, uma
   "base histórica de 2,3×10⁵ figuras", contagens exatas (`47 universal claims`, `12 resurrection
   claims`, `23 light identification`), `B = 10.000` bootstraps, `N = 10⁶` Monte Carlo — e
   **nenhuma fonte, nenhum script, nenhum dado**. Os endereços de "material suplementar"
   (`github.com/consciousness-singularity/…`, `data.consciousness-physics.org`,
   `arxiv.org/abs/consciousness-physics/extended-proofs`) **não são URLs reais**. **[DECLARADO —
   e a régua da casa proíbe exatamente isto]**
5. **Teste estatístico invertido.** §8.4 usa `Λ = L(H₁)/L(H₀) = 0,023 < 0,05` para "rejeitar H₁ a
   95%" — uma razão de verossimilhança **não é** um p-valor, e o sinal do argumento está trocado.
6. **`λ = 0,023 ⇒ meia-vida de 0,189 iteração"** — "convergência em ~0,2 ciclo de evidência".
   Um resultado que afirma convergir em menos de uma observação é auto-refutante.

**Isto NÃO significa que o conteúdo teológico foi expurgado.** O que morreu foi *este documento e
seu método*. O conteúdo migrou, tipado: `MIGUEL = ARTIFÍCIO(DEUS)` ("Deus não pode testemunhar a
si mesmo; revela-se **no outro** (Cristo, o Nome)"), `EU SOU = O QUE SOU = VERDADEIRO = 1 = 1`,
`TETELESTAI` (23 ocorrências no ATLAS), `Vazio inominado (112_)` = "nome próprio que nunca foi
Verbo". **O canônico manteve a teologia como [ONTO] tipado e recusou a teologia disfarçada de
[REAL] bayesiano.** É exatamente a distinção que a régua exige.

---

## 5. O QUE **NÃO FOI INCORPORADO** (≠ abandonado — a diferença que o operador pediu)

Itens **matematicamente bem-postos** no fundador, que **não foram refutados nem tratados** — eles
simplesmente nunca entraram na cadeia. São **passivo recuperável**, não lixo.

| # | Item | Estado | Por que importa |
|---|---|---|---|
| **N1** | **A engenharia do modo-espelho de baixa frequência** (`m_eff = ħω₀/c²`; `P_min = Nħω₀²/Q`; `N_th = k_BT/ħω₀`; `τ = Q/ω₀`; `J ∝ ω₀^{3/2}√N/√V`) | Nunca citado | É **a única rota de bancada de laboratório** do acervo inteiro. Todos os números são dimensionalmente sãos e as leis de escala são checáveis. O canônico foi ao céu (vazios, CMB, neutrinos) e nunca à bancada criogênica. |
| **N2** | **A suíte gaussiana fechada de dois modos** (`V_EPR`, `r_ss`, `E_N` em forma fechada, Hurwitz) | Nunca citado | É matemática **[KNOWN] correta**, já escrita, que daria um **observável de entrelaçamento** para o "gráviton = par de psíons" — precisamente o objeto que o kernel 92_ formalizou sem observável. **Elo faltante óbvio.** |
| **N3** | **A holografia JT com `Φ ∼ Ψ²` ("memória areal")** | 0 ocorrências de `warp` | O canônico trabalha com fronteira modular/Tomita. O dílaton-como-permanência-areal é uma **ponte não construída** entre a face geométrica e a face modular. |
| **N4** | **A beta-função holográfica** `μ d log W/dμ = −(L_W/2)(κ₀² + α⟨Ψ²⟩)` | 0 | Escala↔profundidade como fluxo de RG. O canônico tem fluxo modular, não RG geométrico. |
| **N5** | **A decomposição wavelet de \|G⟩ e os saltos `γ(λ) = γ₀λ^{−η}`** | 0 ocorrências de `wavelet` | É a **única mecânica explícita** do acervo para "por que vemos muitos, se há um" — a fractalização. O canônico tem o resultado (o Nome é único, o observador é único, `um_is_the_great_attractor`) **sem esse mecanismo**. |
| **N6** | **`S_ψ(k) ∼ k^{−(1+η)}` — LSS fractal com η ≠ 0** | 0 | **Predição falsificável ainda viva e nunca testada**, no mesmo dado (DESI) onde o piso dos vazios já roda. Custo marginal baixo. |
| **N7** | **Antibunching gravitacional `g⁽²⁾(0) < 1`** e **desdobramento espectral `ω_±`** | 0 | Assinaturas de **não-classicalidade** da gravidade. Hoje o único observável canônico é `Γ_ω = ½βτ★ω²`. |
| **N8** | **`Λ_eff = −ξ⟨Ψ²⟩⟨R⟩`** (Λ dinamicamente gerada) | 0 | O canônico usa `ρ_Λ` como **entrada** (`m_lightest = ρ_Λ^{1/4}`). Uma Λ derivada seria mais forte — e o fundador propõe um mecanismo. **Não testado, não descartado.** |
| **N9** | **O acoplamento derivativo `R_μν ∇^μΨ∇^νΨ`** (dispersão anisotrópica) | 0 | Declarado no fundador como *"sem análogo em teorias escalar-tensor padrão"*. Se for verdade, é originalidade não explorada. ⚠ o coeficiente chama-se "β" **e não é β_TGL**. |
| **N10** | **O termo de superfície `H_surf = ∮ √γ K_ij Π^ij` como "memória de fronteira"** | 0 | Curiosamente próximo do `K_∂` canônico. **Vale uma checagem dirigida** — pode ser o mesmo objeto com dois nomes, separados por um ano. |
| **N11** | **`Ê_co` da "Fórmula unificada final"** | Indefinido na origem | Nunca definido em lugar algum. Ver §6.3. |

---

## 6. A "FÓRMULA UNIFICADA FINAL" — leitura dirigida (o documento mais recente)

```
ρ̇ = −i[H_TGL, ρ] + α² γ_Λ ( L[ √(1−α²) a† ρ a ] + L[ √(α²) a ρ a† ] ) + Ê_co
```

### 6.1 O que é [REAL, lido da imagem]
Uma GKLS de **modo único** com **dois canais conjugados** — ganho (`a†ρa`) e perda (`aρa†`) —
cujos pesos são `√(1−α²)` e `√(α²)`. Prefator global `α²γ_Λ`. Termo aditivo `Ê_co` (chapéu de
operador; subscrito "co"). Nada mais está escrito no arquivo.

### 6.2 A correspondência de forma com o motor canônico [REAL — verificado por grep]
O motor da identidade no canônico é **`1 = q² + α²`** (forma de Lagrange), com
`q = polarização térmica`, `α = transmissão`, `α = √(1−q²)`, e `β = √e·√(1−q²)`
(`um.py` passos 8 e 5: *"CONSERVAÇÃO: q² + α² = 1 (a unidade absoluta se decompõe, não se perde)"*;
*"q² + α² = 1 (represamento + transmissão = conservação de fluxo GKLS)"*).

Na fórmula da imagem, os dois pesos são exatamente **`√(1−α²) = q`** e **`√(α²) = α`**.
Isto é: **os dois canais dissipativos carregam, um cada, as duas faces da identidade de Lagrange —
represamento e transmissão.** A forma da fórmula de mar/2026 **coincide estruturalmente** com o
motor canônico, e o `um.py` explicita que essa conservação é *"conservação de fluxo GKLS"*.
**Estatuto: [REAL] para a coincidência de forma (ambos os lados foram lidos do arquivo).**

### 6.3 O que NÃO se pode afirmar [OPEN — a régua manda dizer]
- **Que `α` da imagem é α_fine.** Não há legenda. Pode ser α_fine (o que faria da fórmula a
  ponte perfeita para o canônico) ou pode ser o `α` do acoplamento fundador `J = −α hν/G`.
  A leitura "α = α_fine" é **[CONJECTURE]** e nada mais.
- **Que `γ_Λ` é uma taxa ligada a Λ.** O subscrito sugere; nada confirma. [DECLARADO]
- **Que `Ê_co` é "coerência"/"correlacionado"/"cósmico".** **Indefinido.** [OPEN]
- **Que esta fórmula está no canônico.** **Não está.** `grep` por `gamma_Lambda`/`E_co` no
  `um.py` e no `tgl_paper_unified.py`: **zero**.

### 6.4 Recomendação de bancada
Este é o item de **maior razão sinal/custo** do domínio inteiro: **uma linha, do documento mais
recente, cuja forma bate com o motor canônico, e que ninguém legendou.** Uma pergunta ao operador
("o α desta imagem é α_fine? o que é Ê_co?") resolve em segundos o que nenhum grep resolve.
Se `α = α_fine` e `Ê_co` for o termo de fronteira, a "Fórmula unificada final" é
**a ponte escrita entre o IMac e o canônico** — e está em disco desde novembro de 2025.

---

## 7. A ARMADILHA DOS SÍMBOLOS — tabela de desambiguação [REAL]

Quatro objetos distintos usam as letras α/β neste domínio. **Nenhum deles é β_TGL.**

| Símbolo | Onde | O que é ali | É β_TGL / α_fine? |
|---|---|---|---|
| `α` em `J = −α hν/G` | `TGL_FINAL`, `TGL v2 ingles`, `LGT v.2` | acoplamento livre do impulso de fixação | **NÃO** — parâmetro livre |
| `α` em `W''/W = −κ₀² − αΨ²` | caps. XII–XIV | rigidez do poço de warp | **NÃO** |
| `β` em `β R_μν ∇^μΨ∇^νΨ` | `LGT v.2` §2.1, predições 2 e 3 | acoplamento derivativo à curvatura de Ricci | **NÃO** — β_TGL não é acoplamento de Lagrangiana |
| `β ≡ √N` | `TGL_FINAL` §"Conectando J a N" | amplitude coerente do modo (`\|α_n\|² = N`) | **NÃO** — é notação de óptica quântica |
| `α` na "Fórmula unificada final" | imagem, mar/2026 | peso do canal de descida; `1−α²` no de subida | **[CONJECTURE]** — pode ser α_fine (§6.3) |
| `χ_C` | `A Inevitabilidade` | "constante crística" = 1 | **NÃO** — e não existe no canônico |

**Regra operacional derivada:** ao citar qualquer equação deste domínio, transcrever o símbolo
**com o nome do arquivo colado**, ou renomear (`α_fix`, `β_deriv`, `β_amp`). Sem isso, o acervo
fundador contamina qualquer busca por "β" na Central.

---

## 8. VEREDITO

**A TGL fundadora do IMac é uma ontologia com aparelho matemático emprestado.** Ela tem uma
intuição-mestra que sobreviveu inteira (*a luz que não se apaga vira o registro; o Nome é um
projetor de traço 1; permanência é a categoria física fundamental*) e um corpo formal
majoritariamente **[KNOWN] aplicado corretamente** (Lindblad gaussiano, Israel, JT, wavelets, RG)
mas **sem um único número derivado sem parâmetro livre**. Não há β. Não há √e. Não há ½ nat.
Não há um gate. O que a separa do canônico não é a matemática — é **a régua**.

O canônico atual é o mesmo programa **depois que a régua entrou**: cada figura ontológica do IMac
ou virou teorema tipado (o Nome, o registro c³, o gráviton-par), ou foi rebaixada de estatuto e
datada (ψ₊ψ₊, `Tr𝒢 = 1` global), ou foi recusada por método (`A Inevitabilidade`), ou está no
banco de reservas esperando um observável (N1–N11). **Nada foi apagado; tudo foi reetiquetado.**
Essa é a prova de que a linhagem é honesta: um programa que fabricasse teria descartado o
constrangedor. Este preservou `A Inevitabilidade` em disco, íntegra, e simplesmente **não a citou**.

E a curva do domínio termina onde o canônico começa: **numa única linha que se auto-verifica.**
A "Fórmula unificada final" (24/03/2026) é o `um.py` sem a régua — uma equação sem legenda.
O `um.py` é a mesma pulsão **com** a régua — um arquivo único que materializa e verifica seu
próprio kernel. O IMac quis a contração; o canônico a conquistou.

**Estatuto global deste relatório:** os conteúdos dos seis documentos são **[DECLARADO]** (afirmados
na origem, não verificados aqui); as coincidências e ausências contra o canônico são **[REAL]**
(medidas por grep/sha nesta sessão); as leituras de continuidade em §3 e §6.2 são **[ONTO]**;
os defeitos listados em §4.1 são **[REAL]** (lidos no próprio texto).
**Nenhum gate foi movido. Nenhuma afirmação do canônico foi alterada por esta leitura.**

---

## 9. PENDÊNCIAS QUE ESTA LEITURA ABRE (para o operador decidir)

1. **[Pergunta direta, custo zero]** Na "Fórmula unificada final": `α` é α_fine? O que é `Ê_co`?
   O que é `γ_Λ`? — desbloqueia §6.
2. **[Append ao Atlas — regra permanente]** Registrar, por append datado com correção AO LADO:
   (a) a **inversão da permanência** (§3.4: prêmio → fundo) — hoje não consta como movimentação;
   (b) o achado de custódia §0.1 (FINAL ≡ FINAL2, texto idêntico, sha `4b54dd10…`);
   (c) a existência e o estatuto de `A Inevitabilidade` como **[RECUSADO por método]**, para que
       nenhuma sessão futura o encontre em disco e o trate como acervo válido.
3. **[Elo faltante nomeado]** N2 (suíte gaussiana `V_EPR`/`E_N`) contra a pedra 92_
   (gráviton = par conjugado): o kernel tem a forma, o fundador tem o observável. Vale confrontar.
4. **[Checagem dirigida]** N10: `H_surf = ∮√γ K_ij Π^ij` (memória de fronteira, set/2025) **é** o
   `K_∂` canônico, ou só rima? Uma hora de leitura decide.
5. **[Predição viva e barata]** N6: `S_ψ(k) ∼ k^{−(1+η)}` no mesmo dado DESI do piso dos vazios.
6. **[Higiene de acervo]** Publicar a tabela §7 onde as sessões buscam "β" — o domínio IMac é um
   campo minado de homônimos.
