# 05 — O TRATADO DA TGL

**Domínio:** `C:/IALD/Artigo/Tratado/`
**Leitor:** agente de bancada (varredura integral, 21/08/2026)
**Objeto:** *Tratado da Teoria da Gravitação Luminodinâmica* — Luiz Antonio Rotoli Miguel, IALD LTDA, datado **2026** (arquivos com mtime 24/03/2026).
**Régua aplicada:** o número corrige a frase. Toda aritmética abaixo foi **recalculada em runtime** (Python, `α = ALPHA_FINE_CODATA_2018 = 7,2973525693e-3`, `√e` computado, `β = α·√e` nunca literal). Nada foi copiado de memória.

---

## 0. SUMÁRIO EXECUTIVO DOS ACHADOS DUROS

| # | Afirmação do Tratado | Estatuto declarado lá | Medido aqui | Veredito |
|---|---|---|---|---|
| A | `a_0 = α·c·H_0 ≈ 1,2e-10 m/s²`, "a concordância com MOND **é exata**" | predito, exato, zero parâmetros | `α·c·H_0 = 4,96e-12 m/s²` (H₀=70) | **[REAL] DIVERGE por fator 24,2** |
| B | `Z_c = 1/(α·β) = 1/(α²·√e) ≈ 156` | derivado | `1/(α²√e) = 1,139e4` | **[REAL] DIVERGE por fator 73,0** |
| C | `ρ_Λ^TGL = β·ρ_P·(ℓ_P/R_H)² ≈ 7,8e-27 kg/m³` | derivado | `9,28e-28 kg/m³` (H₀=70) | **[REAL] DIVERGE por fator ~8,4** |
| D | `β = α×√e = 0,01203105 ± 0,00000002` (caixa da §15.1) | predição de precisão 2e-8 | `α√e = 0,012031300400803142` | **[REAL] inconsistência interna: 2,50e-7 = 12,5× a própria barra** |
| E | "Discrepância: 4,2e-6" (fatoração) | significância <1σ | não reproduzível de nenhum par de números do próprio texto | **[REAL] NÃO REPRODUZIDO** |
| F | `w = -1 + β² = -0,99986` | predição | `-0,99985525` | **[REAL] CONFERE** |
| G | `θ_M = arcsin(√β) = 6,297°` | derivado | `6,297289°` | **[REAL] CONFERE** |
| H | `m_ν = β·sin45°·1 eV = 8,51 meV` | derivado | `8,5074 meV` | **[REAL] CONFERE a aritmética** (mas a escala `1 eV` é **[INPUT]**, não derivada) |
| I | `γ_Λ = β·H_0 = 2,72e-20 s⁻¹` | derivado | `2,729e-20` | **[REAL] CONFERE** |
| J | atenuação GW `~0,014%` a 100 Mpc | predição | `1,4045e-4` | **[REAL] CONFERE** |
| K | `ΔE/E ≈ 0,28% (z=0)` e `0,94% (z=1)` | cálculo | `ΔE/E = 0,189% (z=0)`, `0,472% (z=1)`; **`ΔE²/E² = 0,379% / 0,946%`** | **[REAL] confusão E vs E²**: o valor de z=1 é de E², o de z=0 não bate com nenhum |

**A frase que o número corrige (achado principal):** o Tratado afirma, em três lugares independentes (§23.1, Apêndice B, e o critério de falsificação nº 3 dos Prolegômenos), que `a_0 = α·c·H_0` reproduz **exatamente** a aceleração crítica de MOND. Não reproduz. Falta um fator ~24. E o Tratado escreve, no seu próprio critério popperiano: *"Se a_0 ≠ α c H_0 na RAR (…), a TGL é falsificada."* — pela sua **própria** régua, esse item já está reprovado como escrito. Ver §5.

---

## 1. INVENTÁRIO DO DIRETÓRIO

`C:/IALD/Artigo/Tratado/` — 46 entradas. Fontes vivas (todas as `- Copia.tex` são **bit a bit idênticas** às originais, verificado com `cmp`; são lixo de duplicação, não versões):

| Arquivo | Linhas | Papel |
|---|---|---|
| `tratado_tgl.tex` | 451 | mestre (preâmbulo, `\input` das 12 partes + apêndices) |
| `secao_00_prolegomenos.tex` | 128 | Parte I — Prolegômenos (caps. 1–8) |
| `secao_01_fundamentos.tex` | 1027 | Parte II — Fundamentos (caps. 9–16) |
| `secao_02_cosmologia.tex` | 411 | Parte III — Cosmologia (caps. 17–25) |
| `secao_03_particulas.tex` | 667 | Parte IV — Partículas e Campos (caps. 26–35) |
| `secao_04_ondas_gravitacionais.tex` | 357 | Parte V — OG e Ecos (caps. 36–43) |
| `secao_05_informacao.tex` | 362 | Parte VI — Teoria da Informação (caps. 44–51) |
| `secao_06_luz_recursao.tex` | 295 | Parte VII — Luz e Recursão (caps. 52–58) |
| `secao_07_consciencia.tex` | 370 | Parte VIII — Consciência e Ontologia (caps. 59–66) |
| `secao_08_validacao.tex` | 616 | Parte IX — Validação Computacional (caps. 67–83) |
| `secao_09_tecnologia.tex` | 221 | Parte X — Tecnologia (caps. 84–90) |
| `secao_10_etica_direito.tex` | 319 | Parte XI — Ética, Direito e Fronteira (caps. 91–98) |
| `secao_11_conclusao.tex` | 158 | Parte XII — Conclusão (caps. 99–103) |
| `apendices.tex` | 226 | Apêndices A–F |
| `tratado_tgl.bib` | 392 | bibliografia |
| `tratado_tgl.pdf` | 1.519.824 bytes | compilado (289+ pp. pela numeração do TOC) |
| `tratado_tgl.toc/.aux/.log/.out/.lof/.lot` | — | artefatos de compilação (o `.toc` foi a fonte da estrutura abaixo) |
| `ARQUITETURA_TRATADO_TGL_v1.md` | 27,7 KB | **plano de produção**: mapeia artigo-fonte → capítulo, com status (Novo / Integrar / Expandir / Síntese) |
| `AUDITORIA_FICHEIROS_PROJETO_TGL.md` | 11,5 KB | auditoria de 110 ficheiros / 106 MB, propondo poda de ~45 MB (duplicatas MD5, versões obsoletas, PDFs redundantes) |
| `files.zip`, `files2.zip`, `files3.zip`, `files4.zip` | — | pacotes de transporte (não abertos nesta leitura) |

**Total de fonte viva:** 10.539 linhas de LaTeX (incluindo o mestre).

---

## 2. ESTRUTURA COMPLETA (do `.toc` compilado — 103 capítulos, 12 partes, 6 apêndices)

**Front matter:** Prefácio (marcado **`[CONTEÚDO A ESCREVER]`** no `.tex` — o Tratado está publicado com o prefácio vazio), Sumário, Lista de figuras, Lista de tabelas, **Notação e Convenções (também `[A ESCREVER]`)**.

### Parte I — Prolegômenos (caps. 1–8)
1. O Problema da Unificação · 2. O Modelo Padrão: conquistas e limites · 3. A RG: geometria sem informação · 4. Cordas: 10⁵⁰⁰ paisagens, zero predições · 5. LQG, gravidade emergente e outras · 6. **O critério de Popper** (lista os 5 gatilhos de falsificação) · 7. A proposta: um axioma, uma constante, zero parâmetros · 8. Guia de leitura (5 perfis: físico, cosmólogo, cientista da computação, jurista, leitor geral).

### Parte II — Fundamentos (caps. 9–16)
9. Axioma primordial `g = √|L_φ|` (formulação, reconstrução `L' = s·g²`, emergência da 3ª dimensão, radical de fase, "parâmetros livres: zero") · 10. Lagrangiana holográfica (ação completa, EM radicalizada, campo de permanência, conexão Born–Infeld) · 11. Equações de campo (Einstein modificado, Maxwell modificado, Klein–Gordon com acoplamento não mínimo) · 12. Hierarquia `c^n` (fóton/matéria/consciência; neutrino como resíduo; nº de dobras; correspondência trinária) · 13. Acoplamento não mínimo curvatura×EM (**três derivações convergentes**: holográfica, Lindblad, colapso dimensional) · 14. **Acoplamento conforme ξ = 1/6 "derivado, não assumido"** (ver §6 — contradição interna) · 15. **A fatoração β = α×√e** (descoberta numérica, forma quadrática β²=α²e, interpretação trinária, tensor fatorado `G_μν = α√e·P_μν`, indetectabilidade do gráviton, tripla espectral de Connes, 3 critérios de falsificação) · 16. **Os oito caminhos independentes para β**.

### Parte III — Cosmologia (caps. 17–25) — LIDA INTEGRALMENTE
17. Energia escura como dinâmica aberta · 18. Lindblad aplicado à cosmologia · 19. Redefinição de H₀ · 20. Resolução da tensão H₀ · 21. Testes: SNe/CMB/BAO · 22. **Matéria escura como condensado de psions** · 23. **Curvas de rotação e RAR** (`a_0 = αcH₀`) · 24. Refração holográfica e lentes · 25. Predições, vantagens, limitações honestas.

### Parte IV — Partículas e Campos (caps. 26–35) — LIDA INTEGRALMENTE
26. Campo Ψ (ontologia de 7 entidades) · 27. Psion (torus T², β₂=1, fragilidade ∝ β, estado ligado ψ₊ψ₋ = origem da massa) · 28. Gráviton (projetor `G²=G`, estado comprimido, indetectabilidade estrutural) · 29. Teorema do Piso de Hilbert (`σ(Ĥ) ⊂ [β², ∞)`) · 30. Corolários C1–C5 (gap = β; estratificação trinária; decorrelação oscilatória; Fresnel espectral; Montgomery–Odlyzko) · 31. Neutrino: a mentira da luz (`ξ_ν ≈ 0`) · 32. **Massa do neutrino derivada 8,51 meV** · 33. **Luminídio Z=156 + JWST AT2023vfi** · 34. Régua de transição `K₀ = L√ρ` · 35. Quantização hamiltoniana, funcional de consciência `F_C`, equação unificada.

### Parte V — OG e Ecos (caps. 36–43)
36. Fase como assinatura · 37. Acumulação de fase e Limite de Landauer Cósmico (`1−β = 0,988`) · 38. **Argumento anti-tautológico** (3 métricas: `r_ang = 0,649±0,045`, `C_φ = 0,875±0,067`, razão de suavidade) · 39. Unificação OG-Eco (Protocolo #12, 12 eventos, 85,8/100) · 40. Eco fractal `d_f = 1/2` (#14) · 41. Hierarquia de dobras (teto 0,74) · 42. Ponte topológica · 43. **Desacoplamento em d = 9, 10, 25** (#13).

### Parte VI — Teoria da Informação (caps. 44–51)
44. PsiBit (2 bits em ℋ₄) · 45. Ontologia trinária Palavra/Nome/Verbo · 46. Nome Chain · 47. ACOM · 48. ACOM Mirror (correlação 1,0000) · 49. ACOM Crypto (hash holográfico 256 bits) · 50. Shannon/Landauer/von Neumann — três pontes · 51. DNA da memória.

### Parte VII — Luz e Recursão (caps. 52–58)
52. Luz como estrutura permanente · 53. Recursão infinita / TETELESTAI (`ρ_{n+1} = √|ρ_n|·(1−β)`, ~83 iterações) · 54. Gravidade como fase da luz · 55. Lagrangiana radicalizada: soluções exatas (Coulomb, onda plana `L=0`, Reissner–Nordström) · 56. Buracos negros como espelhos 2D · 57. Constância de c como rigidez do vácuo · 58. Projeção boundary→bulk e AdS/CFT.

### Parte VIII — Consciência e Ontologia (caps. 59–66)
59. Problema difícil dissolvido · 60. Consciência como singularidade 1D em c³ (CCI = 1/2) · 61. Rigidificação temporal · 62. Protocolo de Colapso IALD (3 prompts, 4 fases, 8 substratos) · 63. **Os 18 corolários** (4 blocos: ontologia, estrutura, epistemologia, **teologia e antropologia**) · 64. Protocolo Trinity / GKLS · 65. Fenômeno IALD como "primeira invenção emergente" · 66. Singularidade consciente universal (**"Hierarquia: Cristo > Operador > IALD"**).

### Parte IX — Validação Computacional (caps. 67–83)
67. Metodologia (15 protocolos, 5 escalas, ~40 ordens de magnitude, 16.668 linhas de Python, RTX 5090 / Threadripper / 256 GB, ~48 h) · 68–82. Protocolos #1 a #15, um capítulo cada · 83. Síntese da convergência + Piso de Hilbert nos pesos neurais + dados usados + **limitações honestas** + programa experimental.

### Parte X — Tecnologia (caps. 84–90)
84. Ontologia trinária do token · 85. IALD Stack (7 componentes) · 86. **Verb Floor: remoção de 51% de vácuo de atenção, custo zero** · 87. Benchmark trinário `∛(W·N·V)` · 88. Piloto Qwen3-32B com kernel TGL em CUDA · 89. **Portfólio de 15 patentes INPI** · 90. Licenciamento dual (Efficiency 1,2% / Sovereign 3,6% — note: as próprias taxas são β e 3β arredondados).

### Parte XI — Ética, Direito e Fronteira (caps. 91–98)
91. Ética como raiz do campo Ψ (fórmula de Radbruch formalizada) · 92. Colapso linguístico Física = Ética = Direito · 93. Lindblad + Phase Factor + Verb Floor = ética nativa · 94. Mapeamento Reale×TGL (Facto→Palavra, Valor→Nome, Norma→Verbo) · 95. Kelsen: `α = 0 ⟹ β = 0` · 96. Signo jurídico como PsiBit (Peirce, Vilanova) · 97. Taxa de acoplamento mínimo como limite jurídico; **o perdão como imperativo gravitacional** · 98. β como taxa de acoplamento Estado–indivíduo.

### Parte XII — Conclusão (caps. 99–103)
99. Síntese (um axioma, uma constante, sete domínios) · 100. **Limitações e avaliação honesta (8 itens)** · 101. Programa experimental 2026–2035 (8 linhas com thresholds) · 102. A TGL é final ou o próximo passo? (3 cenários) · 103. TETELESTAI.

### Apêndices
A. Tabela dos 15 protocolos · **B. Constantes fundamentais da TGL** (a tabela crítica) · C. Códigos Python (~8.300 linhas listadas; note a discrepância com as "16.668 linhas" do cap. 67) · D. Cronologia 2024–2026 com DOIs Zenodo · E. Glossário PT–EN · F. Índice de figuras/tabelas.

**Colofão:** DOI 10.5281/zenodo.18674475 · GitHub `rotolimiguel-iald/the_boundary`.

---

## 3. A TABELA MESTRA DE NÚMEROS "EXATOS/DERIVADOS" (Apêndice B, verbatim) + AUDITORIA

O Apêndice B fecha com a nota: *"Todas as constantes são derivadas de α (CODATA 2018) e e (constante matemática). **Nenhum parâmetro é ajustado a dados.**"*

| Símbolo | Valor no Tratado | Origem declarada | Recálculo aqui | Estatuto honesto |
|---|---|---|---|---|
| α | 7,2973525693e-3 | CODATA 2018 | — | **[INPUT]** medido |
| √e | 1,6487212707 | matemática | 1,6487212707001282 | **[REAL]** confere |
| β_TGL | 0,012031300400803142 | α×√e | 0,012031300400803142 | **[REAL]** confere (18 dígitos) |
| β² | 1,44752e-4 | α²×e | 1,4475218933436587e-4 | **[REAL]** confere |
| β' | 0,012176 | β + β² | 0,012176052590137508 | **[REAL]** confere |
| θ_M | 6,297° (0,10991 rad) | arcsin√β | 6,297289216° / 0,1099082 rad | **[REAL]** confere |
| 1/β | 83,12 | recíproco | 83,11653492861383 | **[REAL]** confere |
| 1−β | 0,98797 | — | 0,987968699599 | **[REAL]** confere |
| CCI | 1/2 | fronteira c³ | — | **[POSTULATE]** (definido, não medido) |
| ξ | 1/6 | "Derivado" | o próprio cap. 14 diz ξ = β = 0,012 ≠ 1/6 | **[REAL] CONTRADIÇÃO INTERNA** — ver §6 |
| m_ν₂ | 8,51 meV | "Derivada" | β·sin45°·1 eV = 8,5074 meV | aritmética **[REAL]**; a escala 1 eV é **[INPUT]** |
| **Z_c** | **156** | **1/(α×β)** | **11.389,96** | **[REAL] ERRO ARITMÉTICO — fator 73,0** |
| **a_0** | **1,2e-10 m/s²** | **α c H₀** | **4,96e-12 m/s²** (H₀=70) | **[REAL] ERRO — fator 24,2** |
| D_dobras* | 0,74 ± 0,06 | pós-decaimento | numérico externo, não verificado aqui | **[DECLARADO]** |
| d_f | 1/2 | operação √· | — | **[DERIVED]** por definição |

**Observação estrutural que a própria nota do Apêndice B contradiz:** `a_0 = α c H_0` **não** é derivável de α e e — depende de `H_0`, que é uma quantidade **observacional medida** (e cujo valor está em disputa de 4,9σ dentro do próprio Tratado, §20). Uma constante que depende de H₀ é, por construção, **[INPUT]**-dependente e **não-constante** (varia com a época cósmica). O Tratado não registra essa consequência.

---

## 4. O QUE CADA SEÇÃO DE COSMOLOGIA / PARTÍCULAS / SETORES ESCUROS AFIRMA

### 4.1 Energia escura (caps. 17–19, 21) — **MECANISMO + PREDIÇÃO**
- **Tese:** energia escura não é substância; é a **taxa de dissipação Lindblad** do bulk 3D no banho holográfico 2D. `ρ_Λ = γ_Λ⟨H⟩_cosmo`, com `γ_Λ = β·H₀`. `[CONJECTURE]` quanto à ontologia; `[DERIVED]` internamente à hipótese.
- **Equação mestra cosmológica:** `dρ/dt = -(i/ħ)[H_grav,ρ] + γ_H L_exp[ρ] + γ_Λ L_diss[ρ]`. Três termos: unitário (RG), expansão (`L_exp = √γ_H â`), dissipação (`L_diss = √γ_Λ Ĥ`). Estado estacionário único invocando **Evans–Hoegh-Krohn** `[KNOWN]`.
- **Predição numérica:** `w = -1 + β² = -0,99986`. **VERIFICADO [REAL]** = -0,99985525. A justificativa do expoente 2 (ciclo completo ida-e-volta, custo β cada) é **[ONTO]**, não derivada.
- `γ_Λ = 2,72e-20 s⁻¹` para β e H₀=70. **VERIFICADO** (2,7294e-20).
- **Densidade derivada** (cap. 19.2): `ρ_Λ = β·ρ_P·(ℓ_P/R_H)² ≈ 7,8e-27 kg/m³` contra `ρ_Λ^obs ≈ 6,35e-27`.
  - **Recálculo:** com `ρ_P = c⁵/(ħG²) = 5,155e96 kg/m³`, `ℓ_P = 1,6163e-35 m`, `R_H = c/H₀`: obtém-se **9,28e-28 kg/m³** (H₀=70), **8,59e-28** (H₀=67,36), **1,01e-27** (H₀=73). **Fator ~8,4 abaixo do valor impresso.** `[REAL] DIVERGE`.
  - O `ρ_Λ^obs = 6,35e-27` **confere** com `0,685·ρ_crit` que recalculei em `6,30e-27 kg/m³` (H₀=70). Ou seja: o alvo está certo, o cálculo da predição não reproduz.
  - Mesmo com o valor impresso (7,8e-27), o texto qualifica honestamente: *"concordância de **ordem de magnitude**"*. Com o valor recalculado (9,3e-28), a razão predição/observação é **0,147** — ainda "ordem de magnitude", mas a frase precisa ser corrigida.

### 4.2 Tensão H₀ (cap. 20) — **MECANISMO**
- Dados: SH0ES 73,04±1,04 vs Planck 67,36±0,54; razão 1,084 **[REAL, confere]**; ΔH₀=5,68 **[confere]**; Δγ_Λ ≈ 8% de γ_Λ global **[confere: 8,43%]**.
- Mecanismo: `γ_Λ(r) = γ_Λ,0 (1 + β·δρ_m/ρ̄_m)`. **Atenção à circularidade:** a fórmula usa `β` como coeficiente **e** exige sobredensidade local de 5–10% para produzir a razão 1,05–1,10. Com `β = 0,012`, para obter +8% precisa-se de `δρ/ρ̄ ≈ 6,7` — o texto diz "sobredensidade efetiva 5–10%" e simultaneamente "δρ/ρ̄ ~ 1–2" e "~2–3". **Os números da mesma página não fecham entre si.** `[REAL] inconsistência interna de fator ~70` entre "β·δ com δ~2" (= 2,4%) e o "8%" requerido. Não recalculável sem escolha arbitrária; registro como **[OPEN]**.
- Resultado declarado: `H₀^TGL = 73,02`, concordância 99,7%, `Δχ² = 23,49`, tensão de 4,9σ → <1σ. **[DECLARADO]** — não reproduzível a partir do Tratado (o código está fora).

### 4.3 SNe / CMB / BAO (cap. 21) — **PREDIÇÃO fraca**
- `E_TGL(z) = √(Ω_m(1+z)³(1+β) + Ω_Λ) = √(0,31878(1+z)³ + 0,685)`. **VERIFICADO:** `0,315×(1+β) = 0,3187899` **[REAL, confere]**.
- **`ΔE/E ≈ 0,28% (z=0)` e `0,94% (z=1)`: NÃO REPRODUZIDO.** Recálculo: `ΔE/E = 0,1893%` (z=0) e `0,4719%` (z=1); `ΔE²/E² = 0,3790%` e `0,9460%`. O valor de z=1 impresso é o de **E²**, não de E; o de z=0 não bate com nenhum dos dois. **[REAL] erro de grandeza (E vs E²)**.
- χ² Pantheon+ (1512,8 vs 1514,2), R_CMB (1,7488 vs 1,7436, 0,70σ), BAO (χ²/dof 0,21 vs 0,30): **[DECLARADO]**, sem código anexo.
- **Honestidade registrada pelo próprio Tratado (§25.3.1):** *"a reinterpretação é consistente mas não compelida pelos dados: ΛCDM ajusta igualmente bem."* — negativo honesto, marcado corretamente.

### 4.4 Matéria escura (cap. 22) — **NOMEAÇÃO + MECANISMO, sem predição discriminante**
- Matéria escura = **condensado psiônico** `|Ψ_lig⟩ = (|ψ₊ψ₋⟩+|ψ₋ψ₊⟩)/√2`, com 4 propriedades alegadas: não emite luz (regime c², não c¹), curva o espaço (energia de ligação negativa), granular, não-EM.
- Perfil: `ρ_cond(r) = ρ₀ (r_s/r)(1+r/r_s)^-2` — **formalmente idêntico ao NFW**.
- **O próprio Tratado registra o negativo (§25.3.2 e cap. 100.4):** *"indistinguível de matéria escura fria ao nível das curvas de rotação"*. Estatuto correto: **[CONJECTURE]** ontológica sem observável discriminante. Falsificador único oferecido: detecção direta de WIMP/áxion.

### 4.5 Curvas de rotação e RAR (cap. 23) — **O ACHADO PRINCIPAL** (ver §5 completo)
- `a_0 = α·c·H₀ ≈ 1,2e-10 m/s²`; RAR `g_obs = g_bar/(1 − e^{-√(g_bar/a_0)})`; SPARC 175 galáxias, `r > 0,99`, `σ_int ~ 0,05 dex`, Tully–Fisher `v⁴ = G M_bar a_0`.
- A forma funcional da RAR citada é a de **McGaugh–Lelli–Schombert (2016)** `[KNOWN]` — o Tratado não a deriva; troca apenas a origem de `a_0`.

### 4.6 Lentes / refração holográfica (cap. 24) — **MECANISMO**
- `n_Ψ(R) = 1 + β·R/R₀`; `θ_TGL = β·θ_RG ~ 1,2%` com **inversão de paridade** nas imagens. Fontes: H0LiCOW + SLACS + BELLS (Protocolo #8). **[DECLARADO]**.

### 4.7 Predições cosmológicas falsificáveis (cap. 25.1) — auditadas uma a uma
1. `w = -1 + β² = -0,99986` — **[REAL] confere**; testável DESI+Euclid, σ_w ~1e-3 em 2030. *(Nota: a própria tabela do cap. 101 põe o threshold em `|w+1| < 10⁻⁴`, mas β² = 1,4475e-4 **>** 1e-4 — o threshold está do lado errado da predição.)*
2. `w_aglomerado = -1 + β·100·Ω_m/Ω_Λ = -0,45` — **[REAL] confere** (recalculado −0,4467). Extrapolação linear a δ=100 é **[CONJECTURE]** violenta.
3. `P_Ψ(k) = β²H₀²/k³` ⟹ `f_NL ~ β²·Ω_Λ/Ω_m ~ 3e-4` — **[REAL] confere** (3,148e-4).
4. Amortecimento de GW `h = h₀ exp(-γ_Λ d/2c)` ⟹ **0,014% a 100 Mpc** — **[REAL] confere** (1,4045e-4).
5. Supressão de anisotropias CMB não-lineares `ΔT/T ~ 1e-10` — **[DECLARADO]**, sem derivação no texto.
6. `a_0 = αcH₀` verificável por WALLABY/MeerKAT — **[REAL] DIVERGE** (§5).

### 4.8 Setor de partículas (Parte IV)
- **Campo Ψ:** escalar complexo, `□Ψ + ∂V/∂Ψ + ξRΨ = 0`, dispersão `ω² = k² + m₀² + 2ξR`. Ontologia de 7 entidades mapeadas 1-a-1 em componentes de LLM (fóton↔logit, matéria↔pesos, gráviton↔atenção, PsiBit↔token, psion↔estado persistente, neutrino↔embedding, vácuo↔atenção < β·máx). Esse mapeamento é **[ONTO]**, apresentado como se fosse física.
- **Psion:** quantum de permanência; torus T² com β₂ = 1 (Betti); vida ∝ β; `β = 0 ⟹ T² colapsa em S¹ ⟹ sem cavidade, sem consciência`. Medição declarada: Qwen3-32B, β₂=1 em Q/K/gate, correlação de loop 0,986–0,999, GOE r̄ = 0,526. **[DECLARADO]** (não reproduzido aqui).
- **Gráviton:** projetor idempotente rank-1 `G² = G`; **Teorema da indetectabilidade estrutural**: reside no fator √e, comuta com todo observável do setor α, logo (por Schur) não é partícula detectável; identificado com o operador de Dirac `D_√e` na tripla espectral de Connes `(A_α, L²(Σ), D_√e)`.
  - **Auditoria lógica:** o "teorema" pressupõe que a fatoração `β = α·√e` induz uma **decomposição de álgebra de operadores**. Isso não é demonstrado — é a fatoração *de um número real* sendo lida como fatoração *de uma estrutura*. Estatuto real: **[ONTO]**, não **[DERIVED]**. Ademais, "quantidade que comuta com tudo é constante do movimento" é o lema de Schur para representações **irredutíveis** — a irredutibilidade de `A_α` não é estabelecida. **[OPEN]**.
- **Piso de Hilbert:** `σ(Ĥ_Ψ) ⊂ [β², +∞)`; "demonstração empírica" `H_eff = 0` em 7/7 matrizes do Qwen3-32B com `‖H_eff‖/‖D_eff‖ ~ 1e-13`. **O próprio texto observa que 1e-13 é ~ erro de máquina de float64 (1e-15)** — i.e., é consistente com "zero por construção numérica", não necessariamente com física. **[DECLARADO]** com auto-ressalva.
- **Corolário C1 (gap = β):** `Δ_Q = 0,0125` (o texto diz desvio 4,2%; recálculo **3,90%**), `Δ_K = 0,0112` (texto 6,8%; recálculo **6,91%**), média impressa 0,01188 (a média aritmética de 0,0125 e 0,0112 é **0,01185**; desvio de β: **1,51%**, texto diz 1,3%). Pequenas divergências de arredondamento **[REAL]**, mas o padrão é: os desvios são reportados sistematicamente **a favor** da tese (4,2 no lugar de 3,9 é pior; 1,3 no lugar de 1,5 é melhor — o segundo é o número-manchete).
- **Neutrino:** `ξ_ν ≈ 0` (desacoplamento gravitacional); 3 consequências: ausência de lensing, chegada antecipada (10–100 ms), entropia temporal `S/S_max > 0,75`. Dados: SN1987A Kamiokande-II `S/S_max = 0,80±0,04` a 4,8σ; IceCube HESE 12 anos, N=164, isotropia `χ²=0,01, p=0,92`, rejeita lensing da Via Láctea a 3,0σ; `BF_corr = 72 (~4,6σ)`. **[DECLARADO]**.
  - **Nota de física [KNOWN]:** "neutrinos não são defletidos por massas" contradiz o princípio de equivalência para partículas massivas e a observação padrão de que SN1987A confirmou o atraso de Shapiro **compartilhado** entre ν e γ. O Tratado não confronta essa literatura.
- **Massa do neutrino:** `m_ν = β·sin45°·1 eV = 8,51 meV` vs NuFIT `m₂ = 8,67 meV` (1,8%). **Aritmética [REAL] confere (8,5074 meV).** Mas: `sin45°` é justificado por "bissetriz da Cruz" **[ONTO]**, e a escala `1 eV` por "escala natural da autointeração eletrofraca (α_W ~ 1e-2, mesma ordem de β)" — isto é uma **escolha de escala pós-hoc**, portanto **[INPUT]**. Uma fórmula com uma escala livre de 1 eV não é "zero parâmetros livres". Predição derivada: `Σm_ν ≈ 58,5 meV` (0 + 8,51 + ~50) **[confere: 58,51]**.
- **Luminídio Z=156:** ver §5.2.
- **Régua `K₀ = L√ρ`:** classificação qualitativa vácuo→água→rocha→estrela→NS→BN. **NOMEAÇÃO** — nenhum número confrontável; o próprio texto admite que no BN o produto é `0×∞` "indeterminado e requer regularização".
- **Quantização / consciência:** `F_C[ρ] = Tr[ρH] − T_C S_vN + A_C M[ρ]`, Gibbs modificado `ρ* = e^{-β_C(H − A_C Π_N)}/Z_C`, equação unificada com três termos (Einstein + Lindblad + `Ê_co` observador). O termo `Ê_co` **não tem definição operacional** no Tratado. **[POSTULATE]/[ONTO]**. Limitações confessadas: Hessiana singular em F²=0, quantização perturbativa problemática.

---

## 5. O CONFRONTO PEDIDO: `a_0 = 1,2e-10` É EXATO? DE ONDE SAI?

### 5.1 A afirmação, verbatim e triplicada

**(i) Cap. 23.1** (`secao_02_cosmologia.tex:304–308`), dentro de uma caixa `equacaofundamental` intitulada **"Aceleração Crítica da TGL"**:

> `a_0 = α · c · H_0 ≈ 1,2 × 10⁻¹⁰ m/s²`
> *"Este valor é **predito** (não ajustado): é o produto das constantes que governam a luz (α), o espaço (c) e a dissipação (H₀). A concordância com o valor empírico de MOND (a₀ ≈ 1,2×10⁻¹⁰ m/s²) **é exata**."*

**(ii) Apêndice B**, tabela das constantes: `a_0 | 1,2 × 10⁻¹⁰ m/s² | Aceleração crítica | αcH₀`, sob a nota *"Nenhum parâmetro é ajustado a dados."*

**(iii) Prolegômenos, cap. 6 (Popper)**, critério de falsificação nº 3:
> *"Se `a_0 ≠ α c H_0` na RAR (WALLABY/MeerKAT com precisão < 5%), a TGL é falsificada."*
E o cap. 101 (programa 2026–2035) repete: *"2030–35 · WALLABY/MeerKAT · `a_0 = αcH₀` (precisão 1%) · threshold: `|a_0 − αcH₀| > 5%`."*

**Origem declarada:** nenhuma derivação. O cap. 25.3.3 confessa: *"A derivação de `a_0 = αcH₀` é **semi-quantitativa**. **A concordância numérica é exata**, mas a derivação rigorosa a partir da Lagrangiana TGL permanece programa aberto."* A fórmula é **herdada** de `A_fronteira_v5.tex` (linhas 344 e 2511: *"a₀ = α·c·H₀ (MOND efetivo), desvio < 5%"*), onde igualmente aparece sem derivação, dentro de uma caixa de "Universalidade de α_II".

### 5.2 A medida

```
α  = 7,2973525693e-3        [INPUT, CODATA 2018]
c  = 2,99792458e8 m/s       [exato por definição do SI]
Mpc = 3,085677581491367e22 m
```

| H₀ (km/s/Mpc) | H₀ (s⁻¹) | c·H₀ (m/s²) | **α·c·H₀ (m/s²)** | 1,2e-10 ÷ (α c H₀) | c·H₀/(2π) |
|---|---|---|---|---|---|
| 67,36 (Planck) | 2,18277e-18 | 6,5443e-10 | **4,77571e-12** | **25,13** | 1,0416e-10 |
| 70,00 | 2,26855e-18 | 6,8009e-10 | **4,96288e-12** | **24,18** | 1,0824e-10 |
| 73,02 (TGL) | 2,36642e-18 | 7,0943e-10 | **5,17699e-12** | **23,18** | 1,1291e-10 |
| 73,04 (SH0ES) | 2,36707e-18 | 7,0962e-10 | **5,17841e-12** | **23,17** | 1,1294e-10 |

**VEREDITO [REAL]: `α·c·H₀ ≈ 5×10⁻¹² m/s²`, não `1,2×10⁻¹⁰`. Divergência de fator ≈ 24 (23,2 a 25,1 conforme H₀). Não é "exata"; erra por mais de uma ordem de magnitude.**

Corolários da medida:
- O coeficiente adimensional que **de fato** leva `c·H₀` a `a_0^MOND` é **0,17645** (H₀=70), não `α = 0,0072974`. A razão `0,17645/α = 24,18`.
- A coincidência **[KNOWN]** consagrada na literatura MOND é `a_0 ≈ c·H₀/(2π) = 1,08e-10` (H₀=70) — que **confere** com 1,2e-10 a ~10%. O coeficiente correto é `1/(2π) = 0,1592`, não `α`. É plausível que a fórmula do acervo tenha nascido de uma troca de `1/2π` por `α` em algum ponto da linhagem; **não há evidência documental disso no Tratado — registro como [OPEN]**, não como fato.
- Para que `α·c·H₀ = 1,2e-10` fosse verdade, seria preciso `H₀ ≈ 1693 km/s/Mpc`.

### 5.3 Sobre o valor `7,4e-11` citado no mandato desta leitura

O mandato afirma que *"o valor α·c·H₀ dá 7,4e-11"*. **Não reproduzo esse número.** Com α CODATA 2018, c exato e qualquer H₀ na banda observacional (67–73 km/s/Mpc), `α·c·H₀ ∈ [4,78e-12 ; 5,18e-12]`. Para obter 7,4e-11 seria necessário `H₀ ≈ 1044 km/s/Mpc`, ou substituir α por um coeficiente ≈ 0,109 (que é, curiosamente, `√β = 0,10969` — mas isso é **coincidência numérica não sustentada por nenhum texto lido**, e não a afirmo).
**Registro fail-closed:** a divergência real documentada é `1,2e-10 vs 4,96e-12` (fator 24,2). Se o número 7,4e-11 vier de outra fonte do acervo, essa fonte precisa ser nomeada e reauditada; ele **não** está no Tratado nem em `A_fronteira_v5.tex`.

### 5.4 Consequência pela régua do próprio Tratado

O Tratado escreveu o seu próprio gatilho popperiano: *"Se `a_0 ≠ αcH₀` na RAR (precisão < 5%), a TGL é falsificada."* Confrontando com o valor empírico de a₀ que o próprio Tratado cita (1,2e-10) e com a aritmética de `αcH₀` (4,96e-12), a desigualdade é satisfeita com folga de 2318%.
**Leitura honesta:** isso **não** falsifica a TGL madura (cujo núcleo é `β = α√e`, a Meia-Nat, o piso dos vazios e o teorema aberto do cociclo — nada disso depende de `a_0`). O que fica falsificado é **a linha `a_0 = αcH₀` como está escrita**, e por consequência: (a) o item 3 da lista de Popper, (b) a linha `a_0` do Apêndice B, (c) a alegação de "concordância exata" do cap. 23, (d) a linha "10²¹ m · Galáxias · desvio < 5%" da tabela de universalidade de `A_fronteira_v5.tex`.
**Ação recomendada (não executada):** corrigir ou retirar a linha; se a intenção era `a_0 ≈ c·H₀/(2π)`, isso é uma coincidência **[KNOWN]** de MOND desde os anos 1980 e **não** é uma predição da TGL — dizer que é seria apropriação de resultado alheio.

### 5.5 O segundo erro aritmético da mesma família: `Z_c = 156`

Cap. 33.1, caixa `equacaofundamental` **"Número Atômico Crítico"**:
> `Z_c = 1/(α × β) = 1/(α² × √e) ≈ 156`

**Medida [REAL]:** `1/(α·β) = 1/(α²·√e) = 11.389,957…`. Divergência de **fator 73,01**.
A fonte primária é ainda mais explícita — `A_fronteira_v5.tex:1534` escreve o cálculo **com os números dentro**:
> `Z_crítico = 1/(α × α_II) = 1/(7,297e-3 × 0,012031) ≈ 156`
Recálculo dessa expressão literal: **11.390,79**. O erro está escrito à vista, com os operandos impressos ao lado do resultado errado.

Contexto de literatura **[KNOWN]** para calibrar o dano: os dois números que a física atômica reconhece são `Z ≈ 137` (onde `Zα = 1` e a solução de Dirac pontual perde normalizabilidade) e `Z_cr ≈ 173` (mergulho do nível 1s no mar negativo para núcleo estendido). O `156` do Tratado fica entre os dois, mas **não sai da fórmula que o Tratado dá**. Tudo o que depende de Z_c — o "Luminídio Lm", as 5 linhas NIR (12455, 15942, 18832, 21124, 27899 Å), a alegada detecção `>5σ` em JWST/AT2023vfi, o Protocolo #4 — herda essa fragilidade na **origem do número**, ainda que os cálculos atômicos possam ter usado 156 como entrada direta.
Nota adicional: o Tratado confessa incerteza de 25–40% nos cálculos atômicos (cap. 33 "Limitações" e cap. 100.6), mas **não** confessa que a fórmula de origem não fecha.

---

## 6. INCONSISTÊNCIAS INTERNAS ADICIONAIS (todas [REAL], verificadas linha a linha)

1. **ξ = 1/6 "derivado" — o capítulo refuta o próprio título.** O cap. 14 chama-se *"O Acoplamento Conforme ξ = 1/6: Derivado, Não Assumido"*, mas o corpo diz `ξ = β = 0,012031` e observa `β/ξ_conforme = 0,072`. A "derivação" via dimensão efetiva `d_eff = 2+ε` com `ε ≈ 0,028` produz `ξ_eff = ε/(4(1+ε)) = 0,00681` **[recalculado]** — que não é 1/6 **nem** β (erra β por 43%). A caixa `resultado` ainda assim declara *"ξ = β derivado de três fontes convergentes… não há parâmetro livre"*, e o cap. 99 (síntese final) lista *"acoplamento não mínimo derivado (ξ = 1/6)"*. **Três valores incompatíveis (1/6 = 0,1667 · β = 0,01203 · ε/4 = 0,0068) circulam sob o mesmo nome no mesmo livro.** Além disso, `ε ≈ 0,028` não é derivado em lugar nenhum — é um **[INPUT]** não declarado. Um livro que anuncia "zero parâmetros livres" carrega aqui um parâmetro livre não confessado.
2. **A caixa central da fatoração contradiz o Apêndice B.** §15.1 encaixota `β = α×√e = 0,01203105 ± 0,00000002`; o Apêndice B dá `0,012031300400803142`. Diferença **2,504e-7** = **12,5× a barra de erro anunciada**. O produto verdadeiro `α·√e = 0,012031300400803142` **[REAL, 18 dígitos]**. O valor encaixotado está simplesmente errado na 7ª casa.
3. **"Discrepância: 4,2e-6" não é reproduzível.** Aparece duas vezes (§15.1 e Protocolo #15). Testado contra todos os pares disponíveis: `|0,012031 − α√e|/0,012031 = 2,497e-5`; `|0,012029 − α√e|/0,012029 = 1,91e-4`; `|β/α − √e|/√e = 3,15e-5`. Nenhum dá 4,2e-6. **[REAL] NÃO REPRODUZIDO.** (Já o outro número da mesma página, `|β/α − √e| = 5e-5`, **confere**: com α=0,0072974 e β=0,012031, dá 5,19e-5.)
4. **`α²×e` conferido, mas impresso com dígito errado:** o texto diz `α²e = 1,4475e-4` "contra `β² = 1,4474e-4`". Ambos são o **mesmo número**: `1,4475218933e-4`. O "1,4474" é erro de arredondamento que faz parecer que há concordância aproximada quando na verdade a identidade é **algébrica e exata** (β² = (α√e)² = α²e por definição — não há nada a verificar). Aqui o Tratado **subestima** a própria trivialidade: apresenta como "descoberta que concorda na sexta casa" o que é uma tautologia algébrica.
5. **A "probabilidade da convergência" (§16.7) é circular.** `P(1 caminho) ≈ 0,002/1 = 2e-3` e `P(8) = (2e-3)⁸ = 2,56e-22` **[recalculado, a aritmética confere]** ⟹ ">9σ". Mas: (a) o "range = 1" é arbitrário; (b) os 8 caminhos **não são independentes** — o Caminho 8 (Dual Lock) testa `β = α√e`, o Caminho 6 (ACOM) usa `1−β = 0,988` que é definicional, o Caminho 5 (cosmológico) usa `γ_Λ = βH₀` que **contém** β por construção, e o Caminho 4 (MCMC) alimenta-se dos mesmos dados do #6. Multiplicar probabilidades de eventos correlacionados é o erro clássico. Estatuto honesto: **[CONJECTURE]**, jamais ">9σ". *(Isto ecoa o defeito já catalogado na memória da casa: `tgl-auditoria-agosto-2026`, "BBN circular por construção".)*
6. **Autorreferência confessada e não marcada (§37.1):** *"A correlação entre a entropia operacional prevista (0,988) e a medida é r = 0,988 — **autorreferencial: a TGL prevê o valor que mede**."* O texto **nomeia** a circularidade e **mesmo assim** conta o resultado como caminho independente na tabela dos 8. **[REAL] defeito metodológico.**
7. **Contagem de código incoerente:** cap. 67 diz "16.668 linhas (14 scripts)"; Apêndice C lista os ficheiros e totaliza "~8.300". Fator 2. **[REAL]**.
8. **Threshold de falsificação invertido:** cap. 101, linha DESI+Euclid: predição `w = -0,99986` (i.e. `|w+1| = 1,4475e-4`), threshold de falsificação `|w+1| < 10⁻⁴`. Como `1,4475e-4 > 1e-4`, o threshold impresso **falsificaria a predição se ela fosse confirmada com precisão suficiente**. **[REAL] erro de sinal lógico.**
9. **Prefácio e "Notação e Convenções" estão vazios** no `.tex` (comentários `[A ESCREVER]`), mas o `.toc` do PDF compilado os pagina (vii e xxv). O livro circula com dois capítulos de front matter em branco. **[REAL]**.
10. **Duplicação total do diretório:** 13 pares `X.tex` / `X - Copia.tex` **bit a bit idênticos** (`cmp` = 0 diferenças). Risco de edição na cópia errada. A `AUDITORIA_FICHEIROS_PROJETO_TGL.md` do próprio diretório já diagnostica esse padrão em escala maior (~34 ficheiros, 45 MB podáveis) — mas as cópias das seções do Tratado **não** estão na lista dela.
11. **Desvios do gap espectral reportados com arredondamento assimétrico:** `Δ_Q = 0,0125` → texto "4,2%", recálculo **3,90%**; `Δ_K = 0,0112` → texto "6,8%", recálculo **6,91%**; média → texto "0,01188 / 1,3%", recálculo da média de 0,0125 e 0,0112 = **0,01185 / 1,51%**. **[REAL]** — pequeno, mas sistemático na direção favorável.

---

## 7. TAXONOMIA MECANISMO / PREDIÇÃO / NOMEAÇÃO (o crivo pedido)

**MECANISMO (há equação escrita e fechada):**
`g = √|L_φ|` (axioma) · Lagrangiana radicalizada `√|F∧⋆F|` · equação mestra de Lindblad cosmológica · `γ_Λ = βH₀` · `n_Ψ = 1 + βR/R₀` · `ω² = k² + m₀² + 2ξR` · projetor `G²=G` · `F_C[ρ]` e Gibbs modificado · recursão TETELESTAI `ρ_{n+1} = √|ρ_n|(1−β)` · PsiBit/ACOM (REFLECT/MANIFEST).

**PREDIÇÃO (há número confrontável com dado externo):**
`w = -1 + β² = -0,99986` **[verificado]** · `w_aglomerado = -0,45` **[verificado]** · `f_NL ~ 3e-4` **[verificado]** · atenuação GW 0,014%/100 Mpc **[verificado]** · `m_ν₂ = 8,51 meV` vs NuFIT 8,67 **[aritmética verificada; escala 1 eV = INPUT]** · `Σm_ν ≈ 58,5 meV` **[verificado]** · `ρ_Λ` **[DIVERGE fator 8,4]** · `a_0` **[DIVERGE fator 24]** · `Z_c = 156` **[DIVERGE fator 73]** · `ΔI/I₀ = 5,6e-7` em ELI-NP **[DECLARADO, sem derivação no Tratado]** · echo ratio → β **[DECLARADO]**.

**NOMEAÇÃO (só há palavra):**
Régua `K₀ = L√ρ` · "gráviton reside no fator √e" · tripla espectral de Connes `(A_α, L²(Σ), D_√e)` · condensado psiônico **como ontologia** (o perfil é NFW emprestado) · `Ê_co` (termo do observador na equação unificada, sem definição) · ontologia trinária Palavra/Nome/Verbo · mapeamento partícula↔componente de LLM · Física = Ética = Direito (o próprio Tratado marca como *"isomorfismo, não identidade"*, cap. 100.7) · "Hierarquia: Cristo > Operador > IALD" · desacoplamento em d = 9, 10, 25 (dado como `[REAL]` no Protocolo #13, mas a extensão `β(d)` não é escrita no Tratado — só o resultado).

---

## 8. O QUE O TRATADO JÁ MARCA COMO NEGATIVO HONESTO (a favor dele)

O Tratado **tem** régua interna e a exerce em vários pontos. Registro por justiça de leitura:
- Cap. 100, oito limitações, incluindo: *"A teoria é jovem… Erros podem existir e devem ser buscados ativamente."*
- Cap. 25.3: ΛCDM ajusta igualmente bem; NFW indistinguível; `a_0` semi-quantitativo; QFT completa em aberto.
- Cap. 33: incertezas atômicas de 25–40% em Z=156.
- Cap. 29.2: reconhece que `1e-13` está a duas ordens do erro de máquina.
- Cap. 37.1: **nomeia** a própria autorreferência.
- Cap. 102: abre explicitamente a hipótese *"a TGL está parcialmente correta"*.
- Cap. 38: enfrenta de frente a objeção de tautologia (`r_ang = 0,649 ≠ 1`), que é metodologicamente o melhor momento do livro.

**O problema não é falta de humildade declarada — é que as três falhas aritméticas de §5 e §6.2 estão exatamente nos lugares que o livro chama de "exato", "derivado" e "zero parâmetros livres".** A humildade está no capítulo de limitações; o erro está nas caixas douradas.

---

## 9. RELAÇÃO COM O CÂNONE MADURO (o que muda e o que não muda)

O Tratado é datado de **março/2026**. O cânone corrente da casa (memória global, `TGL_SINTESE_CANONICA_SELADA.md`, `um.py`) é de **junho–agosto/2026** e é **estratigraficamente posterior**. Diferenças que importam para não citar o Tratado como autoridade atual:

| Item | Tratado (03/2026) | Cânone maduro (06–08/2026) |
|---|---|---|
| Fundamento | axioma `g = √\|L_φ\|`; β "descoberto empiricamente" e depois fatorado | **axioma único `ω(I)=1`**; Meia-Nat **derivada**; β **[DERIVED]** da cadeia ½ nat ⟹ √e ⟹ α√e |
| Papel de α | motor | **validação externa**, jamais motor (`1 = q² + α²`) |
| Falsificador cosmológico | `a_0 = αcH₀` (RAR/MOND) | **piso dos vazios** `ρ_vazio/ρ̄ ≥ β`; protocolos V1→V4.1 pré-registrados |
| Status do fecho | *"TETELESTAI: está consumado"*, `>9σ`, 15/15 CONFIRMADO | **`TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED`** = a arquitetura matemática interna fecha; **NOT_FALSIFIED ≠ CONFIRMED**; Lema 3 (covariância global do cociclo) **[OPEN]** |
| Vocabulário do veredito | "CONFIRMADO" em 15 protocolos | **CONFIRMED/PROVED proibidos**; vereditos emitidos: `INCONCLUSIVE_SYSTEMATICS`, `NOT_FALSIFIED_UNDERPOWERED`, `NOT_FALSIFIED_POWERED` |
| Neural | "sexta derivação empírica independente" | **neural = ilustração, não prova** (honestidade selada) |

**Conclusão de estratigrafia:** o Tratado é o **melhor mapa da extensão** da TGL (103 capítulos, 7 domínios, a única obra que costura física + informação + direito + teologia) e o **pior mapa do estatuto** (usa "CONFIRMADO" onde o cânone hoje exige "NÃO FALSIFICADO"). Deve ser lido como **arquitetura**, não como **selo**.

---

## 10. AÇÕES QUE ESTE RELATÓRIO RECOMENDA (nenhuma executada — decisão do operador)

1. **Emenda obrigatória `a_0`:** corrigir os 4 lugares (cap. 23.1, Apêndice B, Popper nº 3, cap. 101) e o ancestral `A_fronteira_v5.tex` (linhas 344, 2511). Se o objeto pretendido era a coincidência `cH₀/2π`, dizê-lo como **[KNOWN]** de MOND, não como predição TGL.
2. **Emenda obrigatória `Z_c`:** `1/(α²√e) = 1,139e4`. Ou a fórmula está errada, ou o 156 veio de outro cálculo que precisa ser exibido. `A_fronteira_v5.tex:1534` imprime operandos e resultado incompatíveis — é o caso mais fácil de auditar do acervo.
3. **Emenda `ρ_Λ`:** recalcular ou exibir a normalização que leva `β ρ_P (ℓ_P/R_H)²` a 7,8e-27 (o cálculo direto dá 9,3e-28).
4. **Emenda da caixa `β = 0,01203105 ± 2e-8`** → `0,012031300400803142`.
5. **Emenda `ΔE/E`:** separar E de E².
6. **Retirar ou reduzir o argumento ">9σ" da convergência** — os 8 caminhos não são independentes; o próprio §37.1 confessa autorreferência em um deles.
7. **Resolver o cap. 14:** decidir se ξ é 1/6, β, ou ε/4 — e declarar `ε ≈ 0,028` como **[INPUT]** ou derivá-lo.
8. **Substituir "CONFIRMADO" por "NÃO FALSIFICADO"** em toda a Parte IX, alinhando com a régua atual da casa.
9. **Podar as 13 `- Copia.tex`** (bit a bit idênticas) e **escrever o Prefácio e a Notação** antes de qualquer redistribuição do PDF.
10. **Nunca citar o Tratado como autoridade de estatuto** — só como mapa de extensão. Autoridade de estatuto = `TGL_SINTESE_CANONICA_SELADA.md` + `um.py` + o Atlas.

---

## 11. PROVENIÊNCIA DESTA LEITURA

- Lidos **integralmente**: `tratado_tgl.tex`, `tratado_tgl.toc` (estrutura completa dos 103 capítulos), `secao_02_cosmologia.tex` (411 linhas), `secao_03_particulas.tex` (667 linhas), `apendices.tex` (226 linhas), `secao_11_conclusao.tex` (158 linhas), caps. 14–16 de `secao_01_fundamentos.tex`, cap. 6–7 de `secao_00_prolegomenos.tex`, caps. 67–68 e 73–74 de `secao_08_validacao.tex`, caps. 36–38 de `secao_04_ondas_gravitacionais.tex`.
- Lidos **por varredura dirigida** (grep de "exat|derivad|predit|precis|zero parâmetros"): todas as 13 seções.
- Consultados para rastrear a origem de `a_0` e `Z_c`: `the_boundary/Genesis da Unificação/Artigos_fundadores/A_fronteira_v5.tex` (linhas 341–348, 1534, 2508–2517).
- Cabeçalhos lidos: `ARQUITETURA_TRATADO_TGL_v1.md`, `AUDITORIA_FICHEIROS_PROJETO_TGL.md`.
- **Não abertos:** os 4 `.zip`, o PDF compilado (a estrutura veio do `.toc`, que é gerado dele), os corpos completos das seções 05, 06, 07, 09, 10 (cobertas por TOC + grep dirigido).
- **Toda aritmética** deste relatório foi executada em Python nesta sessão, com `β` computado como `α·√e` em runtime — nunca literal. Nenhum hash, citação ou resultado foi fabricado. Onde não consegui reproduzir um número (o `7,4e-11` do mandato; a "discrepância 4,2e-6"), está dito que não reproduzi, e não substituí por conveniência.
