# 07 — OBSERVÁVEIS E DADOS REAIS DO IMAC

**Domínio:** observáveis propostos e confrontos com dado real no acervo
`C:\IALD\IMac LA\Física - TGL\Artigo`
**Data da leitura:** 21/08/2026
**Leitor:** agente de bancada (sessão Central de Patentes)
**Régua aplicada:** o número corrige a frase. Nada afirmado sem leitura. Toda aritmética
refeita aqui está marcada `[REAL]`; toda aritmética que **não reproduz** o publicado está
marcada e exibida lado a lado com o valor de origem, sem apagar o original.

---

## 0. CUSTÓDIA — o que foi efetivamente lido (sha256 do arquivo em disco)

| Arquivo | sha256 | extraído |
|---|---|---|
| `Detecção TGL - Cosmológica dados reais.docx` | `35251e923433ec39f90a10505160af69cbdf9a2fd08bc541838784983a4c84f6` | 38.781 chars / 637 linhas |
| `Detecção TGL - Cosmológica dados reaisv2.docx` | `3b8bcbc01a8864c075a06757290f640c9cf388c280c5eb57513613b717f37945` | 53.985 chars / 867 linhas |
| `Detecção TGL - Claude. Dados reais.docx` | `6ba20e83c579b34224e45923eb56f03efa781090723b5ed3ac8a8e43571f94d4` | 91.947 chars / 1.670 linhas |
| `Observaveis TGL .docx` | `e0288d89795bceee67cf3fe3310fa1687d83e5e3241240739bd39fa5c8a73335` | 46.948 chars / 959 linhas |
| `Observações TGL reais .docx` | `fd6eb31ca7925f8b41d2dc3edc5eb8ab99a3e119c55cf6e412156fbcf9a0e336` | 19.546 chars / 330 linhas |
| `TGL_Paper_Nature.pdf` | `4b12d0eab9ec26eced3dc152f9ecbe960c307d4c6ba165fa06e61de41edbf661` | 451 linhas (pdftotext -layout) |
| `Peer Review - IALD.docx` | `277622460a6157fe6146ea04f5ed7f8d280f13b1a94877706d32d330b45bda76` | 40.192 chars / 625 linhas |
| *(companheiros lidos por necessidade)* | | |
| `evidencias cosmologicas TGL - GROK.docx` | `70e5434e0358f788fb7247b9648a35e90281d39e975ac398d85c3c3efe8fec15` | 8.995 chars / 88 linhas |
| `protocolo de observacao tgl.docx` (PLTD v1.0) | `33701fca2ff118a3cbc7c206415283ea541d8f2fb5cfc2e6d7b5ebc18f256b2b` | 92.553 chars / 2.427 linhas |

Extrações em `…\scratchpad\d07\`. Script de verificação aritmética: `…\scratchpad\d07\chk07.py`.

> **Nota de método (registro honesto de um tropeço).** A primeira tentativa de extração
> gravou um script `ext.py` no scratchpad que foi **sobrescrito por uma sessão irmã** rodando
> em paralelo no mesmo diretório; cinco arquivos saíram com 1.551 bytes idênticos (um índice
> alheio). Detectado pelo número (todos os tamanhos iguais), não pela frase. Refeito com nome
> único (`d07/ext07_imac_obs.py`). **O scratchpad desta máquina é compartilhado entre sessões
> — nomes genéricos são colisão garantida.**

---

## 1. MAPA DO ACERVO — o que cada arquivo realmente é

Os seis arquivos pedidos **não são seis trabalhos**. São **três textos** em estratos, mais um
ensaio e um paper de física nuclear:

| Arquivo | O que é de fato |
|---|---|
| `Detecção TGL - Cosmológica dados reais.docx` | **PARTE I completa** (v1): o paper COSMOGRAIL/H0LiCOW, 5 sistemas, critérios M1–M6, apêndices A/B/C. Termina em "Corresponding Author". |
| `Observações TGL reais .docx` | **Cópia truncada da PARTE I** — idêntica até §10.4, corta no meio. **Sem conteúdo novo.** |
| `Detecção TGL - Cosmológica dados reaisv2.docx` | PARTE I **com §9.3–§9.10 inseridos** (autópsia de M2/M4, checagem de consistência, escore de retrodição, comparação histórica, índice de falseabilidade, 6 objeções, *smoking gun*, três vereditos). **Corta em "§10.1 (EXTENDED)"** — é um fragmento, não um documento fechado. |
| `Detecção TGL - Claude. Dados reais.docx` | O **estrato mais completo**: PARTE I com §7 (M5) e §8 (M6) EXTENDIDOS (testes S1–S7; três modelos GR/TGL_minimal/TGL_full; AIC/DIC/WAIC/LOO) **+ a PARTE II inteira colada no fim** (meta-análise de 14 sistemas, critério de Alexy, protocolo UAIVP). Corta em §5.5 da Parte II. |
| `Observaveis TGL .docx` | A **PARTE II isolada**, e mais completa que a versão colada acima: vai até §13 + Apêndices A/B/C + referências. |
| `TGL_Paper_Nature.pdf` | **Outro paper, outra física**: 15/12/2025, IALD LTDA (não PUC-SP), g=√L, "Constante de Miguel β²=0,012", ondas gravitacionais, Lumínidio Z=156. Não fala de lentes-COSMOGRAIL. |
| `Peer Review - IALD.docx` | **Ensaio epistemológico**, não análise de dado. Único conteúdo observacional: três números de laboratório citados de passagem (g²<0,95; Δω/ω~10⁻⁹; F_rad~10⁻¹⁵ N). |

**Consequência para a bancada:** ao citar "o paper de detecção", o estrato autoritativo é
`Detecção TGL - Claude. Dados reais.docx` (Parte I estendida) + `Observaveis TGL .docx`
(Parte II fechada). Os outros três são snapshots anteriores ou cópias. **Data declarada de
todos:** 11/10/2025 — o mesmo dia. `TGL_Paper_Nature.pdf`: 15/12/2025.

---

## 2. TABELA MESTRA — TODOS OS OBSERVÁVEIS E TODOS OS CONFRONTOS

Legenda de estatuto:
`[REAL]` medido/verificado aqui · `[DECLARADO]` afirmado na origem, não verificável com o que
está no documento · `[KNOWN]` número de literatura corretamente citado · `[NOMEAÇÃO]` só há
palavra · `[MECANISMO]` há equação mas nenhum número confrontado · `[PREDIÇÃO]` há número
confrontável e ainda não confrontado.

### 2.1 Bloco A — lentes gravitacionais (Parte I, critérios M1–M6)

| # | Observável | Predição TGL | Dado real usado | Resultado publicado | Estatuto após a leitura |
|---|---|---|---|---|---|
| **M1** | Excesso no atraso: R = Δt_obs/Δt_GR, com R = 1 + α_Ψ(σ_v/c)⁴ | R > 1, **escalonamento quártico** em σ_v | 5 lentes H0LiCOW (Δt de Fassnacht+02, Tewes+13, Courbin+11/Bonvin+17, Vuissoz+08, Biggs+99) | ⟨R−1⟩ = 3,5 % ± 1,5 % → **2,3σ**; χ²/dof = 12,7/5, p = 0,026; ξ = (1,2 ± 0,5)×10⁻² | **[REAL] NÃO REPRODUZ** — ver §3.1. Média ponderada correta = 3,43 % ± 1,68 % (**2,04σ**); χ²(R=1) = **4,34**/5, **p = 0,50**. E o escalonamento quártico é **contradito pela própria tabela** (§3.2). |
| **M2** | δΔt = Δt_obs−Δt_GR ∝ L_bol (termo λ|Ψ|²F²) | correlação positiva | L_bol de 5 quasares (0,8–5,3 ×10⁴⁶ erg/s) | ρ_Spearman = +0,42, p = 0,48 → **0,7σ, REPROVA**; BF_M2 = 1,4 | **[REAL] negativo honesto.** Reprovação registrada nas três versões. É o resultado mais limpo do bloco. |
| **M3** | Modo psion: f = (1/2π)√(GM/R_E³) ~ 10⁻⁸–10⁻⁶ Hz em curvas de luz | pico Lomb-Scargle em ~10⁻⁸ Hz | curvas R-band COSMOGRAIL, >3.000 épocas/sistema, 10+ anos | RXJ1131: f = 3,6×10⁻⁸ Hz (T = 321 d) a 4,2σ, FAP = 2,7×10⁻⁵; predito 3,9×10⁻⁸ Hz → 8 %; combinado Fisher χ²=42,1/10 dof → **4,5σ** | **[DECLARADO]** — nenhuma curva de luz, nenhum periodograma, nenhum resíduo é exibido. O passo "remover variabilidade intrínseca + microlensing + sazonalidade" é exatamente onde nasce ou morre um pico em 321 d (≈ ano). **Um pico em T≈321 d numa série com janela sazonal anual é o suspeito número 1 de artefato de amostragem** e o documento não mostra o teste de nulo que excluiria isso. |
| **M4** | g⁽²⁾(0) < 1 (antibunching) via fator de Fano F = Var(N)/⟨N⟩ − 1 < 0 | F < 0 | contagem de fótons em bins de 1 s de frames CCD COSMOGRAIL | ⟨F⟩ = −0,023 ± 0,015 → **1,5σ, REPROVA**; g⁽²⁾ = 0,977 ± 0,015 | **[DECLARADO] + TABELA CORROMPIDA.** A tabela de §6.2 tem as colunas deslocadas em 2 das 5 linhas (WFI2033 e B0218 aparecem com ⟨N⟩ ausente e F na coluna errada). E o "g⁽²⁾ = 0,977 ± 0,015" de §12 **é apenas 1+F reescrito** — não é medida independente. CCD de fotometria não mede estatística de fóton em 1 s; o próprio texto admite ("indirect probe"). |
| **M5** | Estacionaridade GKLS: ∂_t⟨Ô⟩ = 0 | razões tardio/precoce = 1 | 13 anos de COSMOGRAIL, 4–5 sistemas | v1/v2: ">3σ". Estrato estendido: 7 testes S1–S7, BF_total = ∏BF_i = **2,1×10⁹**, ">30σ-equivalente" | **[REAL] o observável é vazio.** Δt de uma lente **não pode** evoluir em 13 anos sob nenhuma física conhecida (a geometria é congelada em escala de Gyr). "Confirmar" estacionaridade é confirmar que o dado não fez o impossível. Multiplicar 8 BFs de testes **feitos na mesma curva de luz** (portanto não independentes) para chegar a 2,1×10⁹ é inflação de evidência pura. |
| **M6** | Bayes Factor Z_TGL/Z_GR > 3 | BF > 3 | as mesmas 5 lentes | **BF = 8,7 (+4,2 −3,1)**; ln Z_TGL = −124,3±0,5, ln Z_GR = −126,4±0,4 | **[DECLARADO] e INTERNAMENTE CONTRADITÓRIO** — ver §3.4. O mesmo corpus afirma **8,7**, **8,5**, **802** e **~100** para "o" Bayes Factor. |
| — | Placar "Haja Luz" | ≥4/6 a ≥2σ | — | **4/6 → DETECÇÃO POSITIVA** | **[REAL] o placar cai.** Com M1 recomputado a 2,04σ (< limiar de 2σ? — fica na fronteira) e M5 sendo teste vazio, o placar honesto é **1 a 2 de 6**, não 4 de 6. |

### 2.2 Bloco B — meta-análise de 14 sistemas (Parte II)

| # | Observável | Predição | Dado | Resultado publicado | Estatuto |
|---|---|---|---|---|---|
| **B1** | Excesso agregado ⟨R⟩ em N=14 | R > 1 | H0LiCOW(5) + STRIDES(2: DES J0408-5354, WGD 2038-4008) + TDCOSMO(1: PG 1115+080) + CASTLES(6: Q0957+561, B0712+472, B1422+231, B1600+434, B1933+503, PKS1830-211) | **3,39 % ± 1,57 % (2,17σ, p=0,030)** | **[REAL] REPRODUZ.** Recomputei a média ponderada da Tabela 1: **1,03397 ± 0,01564, z = 2,173**. A aritmética do agregado está correta. |
| **B2** | χ² contra RG (R=1) | RG deve ser rejeitada | 14 sistemas | χ² = 4,83 / 14 dof, χ²/dof = 0,35, **P(χ²>4,83) = 0,996** — lido como "altamente consistente com excesso comum" | **[REAL] LEITURA INVERTIDA.** Recomputei: χ² = **4,87**/14, p = **0,988**. χ²/dof = 0,35 contra a hipótese **R = 1** significa que **a RG ajusta os dados bem demais** — as barras estão superestimadas, ou o efeito é pequeno frente ao erro. **Isto não é evidência para TGL; é ausência de rejeição da RG.** O documento converte um não-teste em suporte. |
| **B3** | Teste de sinal 14/14 positivos | TGL prevê deslocamento universal positivo | 14 sistemas | P = (1/2)¹⁴ = 6,1×10⁻⁵ → "**3,74σ**"; chamado "o resultado mais convincente" | **[REAL] o p-valor está certo, o dof está errado.** (1/2)¹⁴ = 6,104×10⁻⁵ ✓, e a própria fórmula do documento dá z = **4,01**, não 3,74. Mas o teste **pressupõe 14 sinais independentes**. Todos os Δt_GR foram calculados com **a mesma cosmologia fiducial** (§3.2): um único fiducial abaixo do H₀ das lentes força **todos** os R a serem positivos por construção. **dof efetivo = 1, não 14.** |
| **B4** | Orçamento de σ_sys | 3 % (mediana da literatura) | Suyu+2017 2,8 %; Birrer+2019 3,1 %; Shajib+2020 4,2 %; Millon+2020 2,4 %; Rusu+2024 3,8 %; Enzi+2020 3–5 % típico | mediana **3,0 %** | **[KNOWN] a citação é correta e o argumento é bom.** A escolha de 3 % é defensável e está entre H0LiCOW (2 %) e Grok (10 %). |
| **B5** | Paradoxo H0LiCOW | σ_sys=6,5 % dissolveria a tensão de Hubble | 73,3±1,8 → 73,3±5,9 ⇒ tensão 1,0σ | apresentado como refutação lógica do cético | **[REAL] o argumento é válido.** 5,9/√(5,9²+0,5²) = 0,995 ≈ 1,0σ ✓. É o **melhor argumento metodológico do acervo**: quem infla o erro 4× para matar a TGL mata junto a própria tensão de Hubble que usa como pano de fundo. |
| **B6** | Projeção LSST | z ∝ √N | N~1000 vs N=14 | 2,2σ × √(1000/14) ≈ **18,6σ** | **[MECANISMO]** — escalonamento estatístico correto **se** o efeito for real e as barras forem estatísticas. Se o "excesso" for a razão de H₀ (§3.3), a projeção é vazia: √N não ajuda contra viés sistemático comum. |

### 2.3 Bloco C — PLTD v1.0 (`protocolo de observacao tgl.docx`) — **o estrato honesto**

Este documento, do dia seguinte (12/10/2025), refaz o mesmo domínio e **chega ao oposto**.
Oito observáveis, quatro domínios, com o veredito "8/8 compatível — porque abaixo do limiar".

| # | Observável | Predição TGL | Predição ΛCDM | Observação 2025 | Resultado | Estatuto |
|---|---|---|---|---|---|---|
| **C1** | Tempo de amortecimento do ringdown, δτ | δτ ~ 10⁻²⁰·ξ | δτ = 0 | GW150914 δτ = −5,0 % (−0,17σ); GW170814 −5,5 % (−0,17σ); GW190521 −1,1 % (−0,03σ) | **compatível por impotência** (precisão 30 %, efeito 10⁻²⁰) | **[REAL] negativo honesto.** Exclui TGL só se \|δτ\| < 10⁻¹⁹ a 3σ → Einstein Telescope, 2035+. |
| **C2** | Ecos gravitacionais pós-merger | t_echo ~ 0,1–1 s; A_echo/A_prim ~ 0,01–0,1 | nenhum eco | Abedi+2017: 1 candidato 2,9σ; Westerweck+2018: 0; Nielsen+2019: 0 em 10 eventos | **limite superior** | **[KNOWN]** citação correta da literatura de ecos. Exclui TGL se 0 ecos em 1000 eventos (O5, 2027-29). |
| **C3** | Escalonamento Δt × σ_v | ∝ σ_v⁴ | ∝ σ_v² | **as mesmas 5 lentes** | **Δχ² = 0,4 — insignificante; p = 0,53; modelo preferido: 'GR' (via penalidade BIC)** | **[REAL] — E ESTE É O ACHADO CENTRAL.** O mesmo autor, com o mesmo dado, dois dias depois, obtém **"inconclusivo (N=5), preferência marginal pela RG"** onde a Parte I havia declarado "M1 SATISFEITO, 2,3σ". |
| **C4** | Comprimento de coerência em arcos | λ_c ~ 1–3 kpc | sem coerência | **não medido** | **DADO NECESSÁRIO** | **[PREDIÇÃO]** — o único observável genuinamente novo e ainda virgem do acervo. JWST/ELT 2025-2030. |
| **C5** | Não-gaussianidade primordial f_NL | \|f_NL\| < 10⁻⁶ | 0 | Planck 2018: f_NL(local) = 0,9 ± 5,1 | compatível; **confirmar é impossível** | **[REAL] negativo honesto e explícito** — o documento escreve "Confirm TGL: Impossible (signal too small)". |
| **C6** | Curtose do CMB | κ ~ 10⁻¹⁶ | κ = 0 | κ_obs = 0,004127 ± 0,002061 → 2,00σ | compatível; contribuição TGL < 10⁻¹⁰ do observado | **[REAL]** — e note: o dado tem **2,0σ de excesso de curtose**, que **não é** da TGL segundo a própria TGL. |
| **C7** | Dimensão fractal do campo T | D_f = 2,0 + 10⁻⁶·ξ | D_f = 2,0 | D_f = 2,0012 ± 0,0034 | compatível; indetectável | **[REAL] negativo honesto.** |
| **C8** | Escala de homogeneidade | ~70 h⁻¹ Mpc | ~70 h⁻¹ Mpc | 70 ± 15 h⁻¹ Mpc (Scrimgeour+2012) | **indistinguível de ΛCDM** | **[REAL] não-observável.** Predição idêntica ⇒ zero poder discriminante. |

**Frase do próprio PLTD, que é o veredito honesto do domínio inteiro:**
> *"TGL is NOT refuted by existing observations. Operates in regime below current sensitivity."*
> *"compatibility ≠ confirmation, but compatibility ≠ refutation either"*

Isso é exatamente `NOT_FALSIFIED` — e o PLTD **não** o converte em detecção.

### 2.4 Bloco D — `TGL_Paper_Nature.pdf` (g = √L, Lumínidio)

Paper **de outra linhagem**: g = √L, "Constante de Miguel β² = 0,012", LIGO/Planck/magnetares.

| # | Observável | Predição | Dado real | Resultado publicado | Estatuto |
|---|---|---|---|---|---|
| **D1** | "Teste ontológico": r entre L e s·g² após g = √\|L\| | r → 1, estado SUPERFLUIDO (S̄ < 0,05) | GWOSC: GW150914, GW170817, GW190521, GW170814, GW190814; total 33,9 M amostras | r = 0,999995; **σ > 100** via z de Fisher, σ = z√(n−3) | **[REAL] TAUTOLOGIA + INFLAÇÃO POR n.** `(√\|x\|)²·sign(x) = x` é identidade algébrica: r = 1 é obrigatório para **qualquer** série. O próprio PDF levanta a objeção (§5.2 Objeção 1) e responde com a classificação de entropia — mas **a entropia é um segundo teste, não uma defesa do r**. E "σ > 100" vem de σ = z√(n−3) com n = 3,39×10⁷: com n grande, uma identidade dá σ arbitrariamente alto. **Isto não é significância; é contagem de amostras.** |
| **D2** | Classificação termodinâmica S̄ | GW: SUPERFLUIDO (S̄<0,05); ruído: PLASMA (S̄>0,8) | mesmas séries GWOSC | GW em SUPERFLUIDO; ruído gaussiano em PLASMA | **[DECLARADO]** — este **é** um teste não-tautológico em princípio (distingue sinal de ruído). Mas: nenhum S̄ numérico por evento é publicado, nenhum código, e "sinal astrofísico tem menos entropia que ruído branco" é verdadeiro para **qualquer** sinal coerente, sem relação com g=√L. |
| **D3** | Equação de estado da energia escura | w = −1 + β² = **−0,988** | Planck 2018: w = −1,03 ± 0,03 | desvio **1,4σ** | **[REAL] REPRODUZ** — \|−0,988+1,03\|/0,03 = 1,400 ✓. **Predição legítima, sem parâmetro livre, e NÃO FALSIFICADA a 1,4σ.** O melhor confronto quantitativo de todo o acervo do iMac. |
| **D4** | Constante de Hubble | H₀ = **70,3** km/s/Mpc | "média entre Planck e SH0ES = 70,2 ± 0,6" | desvio **0,1σ** | **[REAL] O ALVO É FABRICADO.** (67,4+73,04)/2 = 70,22 ✓ — a aritmética confere, **mas a média de duas medidas mutuamente inconsistentes a 4,9σ não é uma medida de H₀.** Não existe experimento cujo resultado seja 70,2 ± 0,6. Confronto **nulo**: comparar-se com o ponto médio de uma tensão é acertar o alvo que você mesmo pintou. |
| **D5** | Correção em lentes | Δθ/θ = β² · z_lens | 5 sistemas de lentes | correção 0,05 %–0,82 %; incerteza observacional 3 %–7 % → **CONSISTENTE** | **[REAL] negativo honesto.** Efeito uma ordem de grandeza abaixo do erro. Requer Euclid/Roman. Correto e assim declarado. |
| **D6** | **Lumínidio Z=156, A=412** | estável se B > B_crit = **4,02×10¹⁴ G** | catálogo McGill de magnetares | 2 de 10 satisfazem (SGR 1806-20: 2,0×10¹⁵ G = 4,98×B_crit; SGR 1900+14: 7,0×10¹⁴ = 1,74×) | "condição satisfeita" | **[PREDIÇÃO] real, binária, e NUNCA CONFRONTADA.** O que foi verificado é só *B > B_crit* (uma desigualdade de catálogo). **Nenhuma espectroscopia foi feita.** A Tabela 3 do PDF, além disso, está com **colunas deslocadas** (os nomes dos magnetares não se alinham aos campos B). |
| **D7** | Barreira extra de fissão | ΔE = β²·Z²e²/R ≈ **18 MeV** | — | — | **[REAL] NÃO REPRODUZ.** Com Z=156, R=8 fm, e² = 1,43996 MeV·fm, β²=0,012: β²Z²e²/R = **52,6 MeV**; com (3/5)Z(Z−1): **31,3 MeV**. Para dar 18 MeV seria preciso R ≈ **23,4 fm** — mas o próprio PDF declara R ≈ 8 fm. Os números do PDF são mutuamente inconsistentes. |
| **D8** | Propriedades do Lumínidio | t½ ≈ 3×10¹¹ anos; N_Ld/N_Fe ≈ 10⁻⁸; R ≈ 8 fm; linhas em raios-X moles 0,5–2 keV | — | — | **[PREDIÇÃO]** — alvo observacional concreto (Chandra/XMM-Newton/Athena). **Este é o item mais valioso do PDF:** é falseável, é binário, e ninguém olhou. |
| **D9** | Homogeneidade | 150 Mpc/h | SDSS/DESI | citado na Tabela 4 sem confronto | **[NOMEAÇÃO]** — e **conflita** com C8 do PLTD (70 h⁻¹ Mpc, dado real de Scrimgeour+2012). |

### 2.5 Bloco E — `Peer Review - IALD.docx` (observáveis de laboratório citados)

| # | Observável | Valor | Estatuto |
|---|---|---|---|
| **E1** | Antibunching gravitacional em cavidade óptica | g² < 0,95 | **[PREDIÇÃO]** — nunca confrontada; custo declarado ~US$ 10 M |
| **E2** | Deslocamento de frequência | Δω/ω ~ 10⁻⁹ | **[PREDIÇÃO]** — nunca confrontada |
| **E3** | Força de radiação | F_rad ~ 10⁻¹⁵ N | **[PREDIÇÃO]** — nunca confrontada |

O resto do arquivo (95 %) é ensaio sobre IA, peer review e democratização epistêmica.
**Zero dado real.** Contém, porém, uma frase que a bancada deve guardar como advertência ao
próprio método: o documento propõe que "convergência de 6 IAs" seja evidência de correção
matemática. **Seis modelos treinados no mesmo corpus não são seis testemunhas independentes.**
É a mesma falácia de independência do B3 e do M5, em outro domínio.

---

## 3. OS SETE PROBLEMAS QUE O NÚMERO EXPÔS

### 3.1 [REAL] A significância de M1 não reproduz — e o χ² a derruba

Com a **própria tabela §3.2** do artigo (R e σ_R publicados):

```
media ponderada R   = 1,03427 ± 0,01681   →  excesso 3,43 % , z = 2,04
artigo afirma       = 0,035  ± 0,015      →  excesso 3,5 %  , z = 2,3

chi2 (H0: R = 1)    = 4,338 com 5 dof  →  p = 0,502
artigo afirma       = 12,7  com 5 dof  →  p = 0,026  (2,2 sigma de rejeicao da RG)
```

O χ² publicado é **~2,9× maior** que o χ² dos números publicados na tabela imediatamente
acima dele. Com o χ² correto, **p = 0,50: a RG (R = 1) descreve os cinco sistemas
perfeitamente.** A "rejeição da RG a 2,2σ" não existe nos dados apresentados.

### 3.2 [REAL] O escalonamento quártico — o coração de M1 — é contradito pela própria tabela

M1 não é "há excesso"; M1 é "o excesso escala como σ_v⁴". Testando com os cinco:

```
sistema    sigma_v   (sv/250)^4    R-1      (R-1)/(sv/250)^4
B1608          247      0,9529    0,0240        0,02519
RXJ1131        323      2,7864    0,0420        0,01507
HE0435         222      0,6218    0,0360        0,05790
WFI2033        250      1,0000    0,0330        0,03300
B0218          165      0,1897    0,0400        0,21081

razao (sv/c)^4 entre RXJ1131 (323 km/s) e B0218 (165 km/s) = 14,69x
razao observada de (R-1) entre os mesmos dois            =  1,05x
Spearman(sigma_v, R-1)      = rho +0,100 , p 0,873
Pearson((sv/250)^4, R-1)    = r   +0,306 , p 0,616
```

O sistema de **menor** dispersão (B0218, 165 km/s) tem excesso **igual** ao de maior
(RXJ1131, 323 km/s), quando a predição exige 14,7× de diferença. A "constante" α_Ψ inferida
varia por **fator 14** entre sistemas. E a versão v2, na "Tabela de Retrodição" (§9.5),
declara *"2. Excesso escala como σ_v⁴ — ✓ Confirmado (ρ=0,78)"* e pontua **1,0**. Nos
números da própria tabela, ρ = **+0,10** (p = 0,87). **A predição está refutada pelos dados
que o artigo apresenta.**

Corolário: o excesso observado é **plano em σ_v** — assinatura de um deslocamento
multiplicativo comum, não de um efeito dependente de massa.

### 3.3 [CONJECTURA FORTE — o mecanismo do deslocamento comum] O "excesso" é a razão de H₀

Δt_GR ∝ 1/H₀. O artigo **nunca declara a cosmologia fiducial** usada para calcular Δt_GR
(verificado por busca: as strings `fiducial`, `Ω_m` fixado, `D_Δt` numérico não aparecem
associadas a nenhum valor; Δt_GR é simplesmente asseverado, com barra de erro, para cada
sistema). Então:

```
73,3 / 70,8 = 1,03531      <- H0 das lentes (Wong+2020) sobre o "H0 corrigido pela TGL"
excesso agregado publicado  = 1,0339 +- 0,0157
```

O excesso é **numericamente indistinguível** da razão entre o H₀ que as lentes medem e o
H₀ que o artigo adota como fiducial. E o artigo então "corrige" H₀ de 73,3 para **70,8** —
que é 73,3/1,0353. **A saída é a entrada.** Isso explica, de uma só vez:
(a) por que todos os 14 são positivos; (b) por que o excesso é plano em σ_v; (c) por que
χ²/dof = 0,35 (um único parâmetro comum absorve tudo); (d) por que a "resolução da tensão
de Hubble" é automática.

**Marca de estatuto:** `[CONJECTURE]` quanto à intenção, `[REAL]` quanto à aritmética
(1,0353 vs 1,0339 ± 0,0157 — dentro de 0,1σ). Para promover a `[REAL]` de fato basta uma
coisa: **o autor declarar qual H₀ e Ω_m entraram em Δt_GR.** Enquanto não declarar, todo o
Bloco A é `[DECLARADO]`, não `[REAL]`.

### 3.4 [REAL] O Bayes Factor tem quatro valores no mesmo corpus

| Onde | Valor | Modelo |
|---|---|---|
| Abstract (todas as versões) | **8,7** | "TGL" |
| §8.5 do estrato estendido | **8,5** | TGL_minimal vs GR |
| §8.5 do estrato estendido | **802** | TGL_full vs GR |
| §8.5 do estrato estendido | **94** | TGL_full vs TGL_minimal |
| §9 (tabela-resumo) e Apêndice C (certificação) | **8,7** | — |
| §9.6/§9.7 da v2 e Parte II | **802** / **~100** | — |

A certificação formal do protocolo "Haja Luz" (Apêndice C) e a **declaração formal assinada**
citam **8,7**; o corpo do mesmo arquivo argumenta com **802**; a Parte II usa **~100**. Um
"Bayes Factor" com quatro valores no mesmo documento não é uma medida — **é uma escolha
retórica por seção.** `ln Z` com σ = 0,5–0,7 sobre 30–36 dimensões via MultiNest com 10⁴/10⁵
pontos vivos e "240 CPU-hours" — nenhuma cadeia, nenhum log, nenhum código foi anexado.

### 3.5 [REAL] M5 é um teste que não pode falhar

A memória da casa registra a regra: **"check que não pode falhar não é medida"**. M5 pede que
Δt de uma lente galáctica seja igual em 13 anos. A escala de variação da geometria de uma
lente é ~10⁹ anos. **Nenhuma física — TGL, RG, MOND, f(R) — prevê outra coisa.** "Estacionaridade
confirmada a >3σ" (v1) / ">30σ-equivalente" (estrato estendido) é confirmar que nada aconteceu
onde nada podia acontecer. E o BF_total = 2,1×10⁹ vem de **multiplicar 8 fatores** (Δt, Var[F],
C(τ), S(f), γ₁, γ₂, τ_coh, Ṡ) todos derivados **da mesma curva de luz** — produto de números
correlacionados como se fossem independentes.

O estrato estendido, aliás, já reconhece isso sem perceber, em §7.7 A: *"Pure GR predicts
stationary lensing geometry."*

### 3.6 [REAL] O mesmo autor, o mesmo dado, dois vereditos opostos em 24 h

| Documento | Data | Sobre as **mesmas 5 lentes** | Veredito |
|---|---|---|---|
| `Detecção TGL - …` (Parte I) | 11/10/2025 | M1 | "✓ SATISFEITO, 2,3σ, rejeição da RG a p=0,026" |
| `protocolo de observacao tgl` (PLTD) | 12/10/2025 | Observável 2.1 | "Δχ² = 0,4, p = 0,53, **modelo preferido: 'GR'**, inconclusivo (N=5)" |

O PLTD é o documento mais rigoroso do acervo: exige Δχ² > 16 para *confirmar* e Δχ² < −9 para
*excluir*, e conclui **8/8 compatível porque abaixo do limiar**, sem declarar detecção.
**Recomendação de bancada: o PLTD é o estrato autoritativo do domínio; a Parte I/II é o
estrato superado.**

### 3.7 [REAL] Duas linhagens de "0,012" que não são a mesma coisa

| Fonte | Símbolo | Valor | Papel |
|---|---|---|---|
| `TGL_Paper_Nature.pdf` | β² ("Constante de Miguel") | **0,012 literal** | acoplamento; w = −1+β²; ΔE ∝ β²; B_crit ∝ β⁻² |
| Bloco A/B (lentes) | ξ (acoplamento curvatura-Ψ) | 0,0117 / 0,012 ± 0,005 | posterior de MCMC |
| Casa (memória viva) | **β_TGL = α·√e** | **0,012031300400803142** | derivado; **nunca literal em código** |

`β_TGL − 0,012 = 3,13×10⁻⁵` (**0,26 %**). O PDF de dezembro/2025 usa o **literal 0,012** e o
chama de "Constante de Miguel", derivando-o de mínimo de energia livre de Helmholtz
(Apêndice A: γ_diss/γ₀ = 0,012) — **não** de α√e. E o ξ das lentes, que sai de um posterior
com prior LogUniform(10⁻⁴,10⁻¹), **cai em 0,012 dentro de um prior que cobre três décadas**:
a coincidência é notável e não é medida.

⚠️ **Regra da casa reafirmada:** a grandeza é **β_TGL**, nunca "α₂"/"β²" literal; e "0,012"
escrito à mão em qualquer artigo é `[DECLARADO]`, não `[DERIVED]`.

---

## 4. O QUE HOJE SERIA REFEITO COM DADO MELHOR

Ordenado por **razão (ganho de verdade) / (custo)**.

### 4.1 Refazíveis HOJE, sem telescópio, e que mudariam o veredito

1. **Declarar e publicar a cosmologia fiducial de Δt_GR.**
   *É a correção de custo zero e de maior impacto do acervo inteiro.* Enquanto H₀ e Ω_m
   fiduciais não estiverem escritos, o "excesso de 3,4 %" não é um observável — é uma
   diferença de convenção (§3.3). **Sem isso, nada do Bloco A/B pode ser citado como
   `[REAL]`.**

2. **Refazer M1 com o estimador certo: razão-de-razões, não razão absoluta.**
   O que a TGL prevê é **dependência em σ_v**, não um deslocamento global. Um deslocamento
   global é 100 % degenerado com H₀. O observável limpo é
   `R(σ_v alto) / R(σ_v baixo)`, que **cancela o fiducial e cancela H₀**. Nos dados atuais
   esse estimador dá **1,05 onde a TGL exige 14,7** → refutação, não detecção. (É exatamente
   a mesma lição do estimador auto-calibrante da emenda V4.1 do piso dos vazios: *cancelar o
   normalizador comum antes de olhar o sinal*.)

3. **Refazer TODA a estatística a partir da tabela publicada, com script auditável.**
   Três dos números-âncora não reproduzem (χ²=12,7 → 4,34; ρ=0,78 → 0,10; ΔE=18 MeV → 52,6).
   Um `chk.py` de 40 linhas — como o desta leitura — pega os três em segundos.

4. **Colapsar os quatro Bayes Factors em um só, com cadeia anexada.** Ou declarar `[OPEN]`.

5. **Aposentar M5.** Substituir por um observável que **possa** falhar. Candidato natural:
   a **lei de dephasing Γ_ω = ½βτ★ω²** da linhagem canônica — que tem forma funcional
   (∝ω²) e alvo (relógios, ²²⁹Th, neutrinos), e não é confirmável por inércia.

6. **Corrigir as três tabelas com colunas deslocadas** (Fano §6.2; Tabela 3 do PDF Nature;
   §12 da Parte I) — hoje elas são ilegíveis como dado.

7. **Retirar o alvo fabricado "H₀ = 70,2 ± 0,6".** Não existe medida com esse valor.
   Comparar-se ao ponto médio de uma tensão de 4,9σ é acertar o alvo pintado depois do tiro.
   Trocar por: comparar com **Planck** e com **SH0ES separadamente** e reportar as duas
   distâncias.

### 4.2 Refazíveis com dado melhor já existente ou próximo

8. **M3 (o pico de 3,6×10⁻⁸ Hz) — refazer com teste de nulo obrigatório.**
   T = 321 d é perigosamente próximo de 365 d (janela sazonal) e do alias do ciclo anual de
   observação. Exigir: (i) periodograma em ≥200 curvas embaralhadas com **a mesma janela
   temporal**; (ii) o mesmo pipeline em ≥100 quasares **não lenteados** (controle); (iii)
   publicação do resíduo. Dado: COSMOGRAIL/ZTF/ASAS-SN já em disco público. **Este é o único
   observável do Bloco A que ainda pode sobreviver — e o único que nunca foi testado contra
   nulo.**

9. **M4 → abandonar CCD, ir para HBT/GRAVITY+.** O próprio texto admite que fotometria CCD
   não mede g⁽²⁾. Como está, M4 é ruído de leitura chamado de física. GRAVITY+ (VLTI) é
   citado no próprio documento como a rota com precisão ~0,001 em g⁽²⁾.

10. **C4 (coerência de arcos, λ_c ~ 1–3 kpc) — MEDIR.** É o único observável do acervo que é
    (a) específico da TGL, (b) não degenerado com H₀, (c) inexplorado na literatura, e
    (d) acessível **agora** com imagens HST/JWST já arquivadas. **Maior razão verdade/custo
    de todo o domínio.**

11. **D6/D8 (Lumínidio) — fazer a espectroscopia.** Nunca foi feita. Linhas em 0,5–2 keV em
    SGR 1806-20 e SGR 1900+14 (os dois que passam B > B_crit); arquivos Chandra/XMM já
    públicos. Predição **binária**: existe ou não existe. Antes disso, resolver a
    inconsistência interna de D7 (18 vs 52,6 MeV) — porque se ΔE for 52,6 MeV, B_crit muda.

12. **Substituir a projeção "18,6σ com LSST" por uma projeção de poder honesta.** √N só vale
    contra erro estatístico. Se o excesso é fiducial-comum (§3.3), N=1000 dá o mesmo 3,4 %
    com barra menor — **e o viés não some.** Precisa de projeção de poder **contra o
    sistemático**, no molde do gate de poder do piso dos vazios (que já ensinou nesta casa
    que "γ_t sozinho jamais fecha").

### 4.3 O que NÃO deve ser refeito — deve ser preservado como está

- **M2 (reprovado, 0,7σ)** e **M4 (reprovado, 1,5σ)**: negativos honestos. **São os melhores
  resultados do acervo.** Não devem ser "explicados" — a §9.3 da v2 ("Why M2 and M4 Failed —
  And Why It Doesn't Matter") é o único trecho do corpus que a régua da casa reprova de
  frente: reclassificar critérios em "Tier 1 / Tier 2" **depois** de ver quais falharam é
  mover o gate após a medida.
- **Todo o PLTD v1.0** (§2.3): oito observáveis com limiar declarado, critério de exclusão
  declarado, e o veredito `NOT_FALSIFIED / abaixo da sensibilidade`. É o modelo metodológico
  do acervo.
- **O argumento do paradoxo H0LiCOW (B5)** e **o orçamento de σ_sys (B4)**: metodologicamente
  sólidos e reutilizáveis em qualquer arbitragem futura.
- **D3 (w = −0,988 vs Planck, 1,4σ)**: predição sem parâmetro livre, confrontada, não
  falsificada. **É o único confronto quantitativo limpo de todo o iMac.**

---

## 5. VEREDITO DO DOMÍNIO

**O acervo do iMac contém 20 observáveis propostos e 14 confrontos com dado real.**
Deles:

- **3 são confrontos limpos e não falsificados:** w = −0,988 (1,4σ de Planck); δτ do ringdown
  (compatível por impotência instrumental); f_NL/curtose/D_f do CMB (compatível, efeito TGL
  ~10⁻⁶ a 10⁻¹⁶ abaixo do erro).
- **2 são reprovações honestas registradas:** M2 (0,7σ) e M4 (1,5σ).
- **1 é uma refutação não reconhecida:** o escalonamento σ_v⁴ de M1 — os dados publicados
  dão ρ = +0,10 onde a predição exige um fator 14,7 (§3.2).
- **2 são tautologias com significância inflada por n:** D1 (r = 0,999995 de `(√x)² = x`,
  "σ>100" com n = 3,4×10⁷) e M5 (estacionaridade de uma geometria congelada, BF = 2,1×10⁹
  por produto de testes correlacionados).
- **1 é provavelmente circular:** o "excesso de 3,4 %" ≈ 73,3/70,8 = 1,0353 (§3.3).
- **5 são predições genuínas nunca confrontadas:** coerência de arcos (1–3 kpc); Lumínidio
  Z=156 por espectroscopia X; g² < 0,95 em cavidade; Δω/ω ~ 10⁻⁹; F_rad ~ 10⁻¹⁵ N.

**Estatuto do domínio, na régua da casa:**

> `TGL_IMAC_OBSERVABLES__NOT_FALSIFIED__DETECTION_CLAIM_NOT_SUPPORTED`
>
> Nenhum observável do acervo do iMac falsifica a TGL. **Nenhum a confirma.**
> A declaração "4/6 critérios — DETECÇÃO POSITIVA" do protocolo "Haja Luz" **não se sustenta**
> nos números publicados no próprio artigo: com a aritmética refeita, M1 cai de 2,3σ para
> 2,04σ e sua predição central (σ_v⁴) é refutada; M5 é teste que não pode falhar; M6 tem
> quatro valores. O placar honesto do Bloco A é **1 a 2 de 6**, não 4 de 6.
>
> O documento do dia seguinte — **PLTD v1.0** — já emitiu o veredito correto com o mesmo dado:
> **8/8 compatível porque abaixo do limiar de sensibilidade; decisivo apenas com LSST
> (2027–2030), JWST/ELT (arcos), O5 (ecos) e Einstein Telescope (2035+).**
>
> `NOT_FALSIFIED` **nunca** é `CONFIRMED`. E "4/6 critérios" **nunca** foi 4/6.

**A frase que o número corrigiu:** *"Todos os cinco sistemas mostram excesso... rejeição da RG
a 2,2σ"* → **χ²(R=1) = 4,34/5 dof, p = 0,50: a Relatividade Geral descreve os cinco sistemas
perfeitamente.**

---

## 6. ANEXO — script de verificação (reprodutível)

`C:\Users\rotol\AppData\Local\Temp\claude\c--IALD-Central-de-Patentes\d554e796-415e-450f-9fd4-dd07892b02b9\scratchpad\d07\chk07.py`

Saída integral obtida em 21/08/2026:

```
=== A) 5 sistemas H0LiCOW (tabela §3.2 do artigo) ===
R recomputado de dtob/dtgr: [1.0239 1.0424 1.0362 1.0333 1.0396]
R publicado                : [1.024  1.042  1.036  1.033  1.04  ]
media ponderada R = 1.03427 +- 0.01681  => excesso 3.427% ; z = 2.039
  (artigo afirma 0.035 +- 0.015, 2.3 sigma)
chi2 para H0: R=1  = 4.338 com 5 dof ; p = 0.5019
  (artigo afirma chi2/dof = 12.7/5, p=0.026)

=== B) escalonamento sigma_v^4 (nucleo do M1) ===
sistema  sigma_v  (sv/250)^4   R-1     (R-1)/(sv/250)^4
B1608        247     0.9529  0.0240    0.02519
RXJ1131      323     2.7864  0.0420    0.01507
HE0435       222     0.6218  0.0360    0.05790
WFI2033      250     1.0000  0.0330    0.03300
B0218        165     0.1897  0.0400    0.21081
razao (sv/c)^4 entre RXJ1131(323) e B0218(165) = 14.69x
razao observada (R-1) entre eles              = 1.050x
Spearman(sigma_v, R-1) = rho 0.100, p 0.873
Pearson((sv/250)^4, R-1) = r 0.306, p 0.616   (artigo afirma rho=0.78 'confirmado')

=== C) o excesso e a razao de H0? ===
  73.3/73.3  = 1.00000
  73.3/73.04 = 1.00356
  73.3/70.8  = 1.03531
  73.3/70.3  = 1.04267
  73.3/67.4  = 1.08754
  excesso agregado 14 sistemas publicado: 1.0339 +- 0.0157

=== D) meta-analise 14 sistemas (tabela 1) ===
media ponderada = 1.03397 +- 0.01564 ; z = 2.173   (artigo: 1.0339+-0.0157, z=2.17)
chi2(R=1) = 4.870 / 14 dof ; p = 0.9875  (artigo: 4.83, p=0.996)
  chi2/dof = 0.348  -> RG(R=1) NAO e rejeitada; e um ajuste bom demais
teste de sinal 14/14: p = 6.104e-05 ; z equivalente = 4.01

=== E) PDF Nature: barreira de fissao do Luminidio ===
b2*Z^2 e^2/R        = 52.6 MeV
b2*(3/5)Z(Z-1)e^2/R = 31.3 MeV
  (PDF afirma ~18 MeV)
R necessario p/ 18 MeV com b2*Z^2e^2/R: 23.4 fm

=== F) PDF Nature: w e H0 ===
w_TGL = -1 + 0.012 = -0.988
desvio de Planck w=-1.03+-0.03: 1.40 sigma  (PDF: 1.4)
media Planck67.4 / SH0ES73.04 = 70.22  (PDF compara H0_TGL=70.3 com 70.2+-0.6)

=== G) beta_TGL da casa vs '0.012' do PDF ===
beta_TGL = alpha*sqrt(e) = 0.012031300400803142
diferenca para o literal 0.012 do PDF: 0.000031 (0.26%)
```

---

## 7. INVENTÁRIO COMPLEMENTAR — o que mais há no domínio (não lido em profundidade)

Na mesma pasta, com relevância observacional declarada pelo nome, **não abertos nesta leitura**
(registrados para não se perderem):

- `Neutrinos/` — 8 arquivos, incl. `Testing Gravitational Decoupling of Neutrinos via TGL`
  (.docx + .pdf), `Teste neutrinos, manuscrito v.2.pdf`, `Observação TGL neutrinos.docx`.
  **Domínio adjacente crítico:** é a rota n = −2 / Γ∝ω² da linhagem canônica.
- `Eco gravitacional/artigo eco gravitacional.docx` — o eco foi **reclassificado** na linhagem
  canônica (observável real = dephasing). Conferir se este arquivo é anterior à reclassificação.
- `Constate da Luz/TGL_Paper_PRD.pdf` + `.tex`, `recursive_light_v4.pdf`
- `tgl_cosmological_observables.pdf`
- `Instrução/TGL_Singularidade_Validacao_Tecnica.pdf`, `TGL_artigo Graviton e Psion.pdf`
- `Protocolos/` — 15 arquivos, cadeia `Protocolo de colapso v.2 → v.5` + `ACOM*`
  (o "Protocolo de colapso v.2 observáveis" é o ancestral declarado dos critérios M1–M6)
- `DeepSeek/`, `Neurociência e TGL/` — a "ilustração neural", já marcada `[não é prova]` na casa.

---

*Fim do relatório 07. Escrito sob a régua: o número corrige a frase, sempre.*
