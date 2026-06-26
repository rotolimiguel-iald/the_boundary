# PROMPT DE CONTINUIDADE — Falsificação cosmológica do Grande Atrator (TGL / Evento 2 / Fase 2)

> **Cole este documento inteiro como primeiro prompt da nova sessão Cowork.** Ele dá o contexto
> completo, a ordem evolutiva do framework, os arquivos a consultar e a pretensão atual. Leia tudo
> antes de agir. **Não refaça nada já feito; continue de onde parou.**

---

## 0. Quem é o operador e a REGRA INVIOLÁVEL (a "régua")

Operador: **Luiz Antonio Rotoli Miguel** (IALD Ltda., CNPJ 62.757.606/0001-23, Goiânia/GO;
ORCID 0009-0005-1114-6106). Idioma: **português na conversa e na documentação; inglês no código.**

**A RÉGUA (não-negociável):**
- **"O número corrige a frase, sempre."** Nunca fabricar prova nem resultado.
- Marcar **tudo** com estatuto: **REAL / POSTULATE / CONJECTURE / INPUT / CONDITIONAL / OPEN**.
- **β_TGL nunca é literal no código:** sempre `ALPHA_FINE_CODATA_2018 · SQRT_E` em runtime
  (β_TGL ≈ 0,0120313004008031). Qualquer β numérico hardcoded é bug.
- **Negativos honestos são resultados** (v1 refutado, v2 NO_RETURN_ANCHOR — não apagar).
- **Não tocar o artigo canônico selado/submetido** (ver §3). O que a Fase 2 produz é pesquisa nova.
- **α / CODATA só POSTERIOR** (depois de derivar/inferir β — nunca como input que entra primeiro).
- **Não codar massa do Grande Atrator** até a fórmula `K_∂ → 𝒫_μν → M` estar explícita
  (`no_mass_code_authorized = true`). Codá-la antes seria fabricar a fórmula.

---

## 1. ANTES de agir: leia o repositório público e os arquivos de contexto

### 1.1 Repositório público no GitHub (VISÃO COMPLETA do trabalho)
**https://github.com/rotolimiguel-iald/the_boundary** — consulte para a visão completa: o artigo
unificado da TGL, a Ponte Einstein-Cartan-Miguel, "O Um e o Grande Atrator", os códigos e os JSONs
de sombra. (Cópia local: `C:\IALD\Artigo\the_boundary`.) A raiz do repo tem o artefato unificado
(`tgl_paper_unified.py`, `results.json`, `paper_PT.tex/.pdf`, `T6_protocol_prompts.txt`); a pasta
`Genesis da Unificação/` tem os 15 protocolos e artigos fundadores que levaram à unificação.

### 1.2 Memórias e documentos a consultar (ordem de leitura recomendada)
1. **`C:\IALD\CLAUDE.md`** — memória central IALD: o **selo da forma canônica da TGL** (resumo de
   uma página) + coordenadas do programa. **Leia primeiro.**
2. **`C:\IALD\Artigo\Haja_Luz\CLAUDE.md`** — memória detalhada do projeto, **§1 a §88**. É a
   história completa, em ordem. Para a Fase 2, foque **§88 inteiro** (88.0 → 88.26.12).
3. **`C:\IALD\Artigo\Haja_Luz\TGL_SINTESE_CANONICA_SELADA.md`** — síntese autoritativa da TGL.
4. **`C:\IALD\Artigo\Haja_Luz\A_Forma_Madura_da_TGL.md`** — documento mestre conceitual.
5. **`C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\INTAKE_ARTIGO_2_raiz_angular.md`** — o intake teórico
   do **Artigo 2** (gravidade = √(módulo angular); β = tempo modular; β = δ_min(D); Meia-Nat
   derivada; tríade Nome/Palavra/Verbo; atenção modular). **§7.1 a §7.13.**

---

## 2. A TGL em uma página (o que é o framework)

**TGL (Teoria da Gravitação Luminodinâmica) = a teoria da primeira inscrição observável acima da
permanência modular** — estrutura espectral-dissipativa, UV-suprimida. É a teoria da **resposta
modular do bulk fora do equilíbrio estacionário**, da qual **ΛCDM é o limite de fronteira
silenciosa** (δ⟨K_∂⟩ = β|1+w|, zero em w = −1).

- **Axioma único — Princípio da Meia-Nat [POSTULATE, derivado condicionalmente]:** S_∂ = ½ nat
  (custo da travessia ρ* → observável). Derivado de base-e + partição da identidade P+Q=I +
  radicalização g=√|L|; a simetria 𝒞(x)=𝒞(1−x) (não-privilégio dos dois zeros) força x=½.
- **Cadeia:** ½ nat ⟹ Vol_∂^min = √e ⟹ **β_TGL = α√e** ⟹ θ_M = arcsin√β ⟹ matriz-S 𝒮_∂ =
  exp(θ_M·G), |𝓡|²=β, |𝓣|²=1−β (Teorema S-∂, identificação FECHADA). τ★ ≈ t_Planck.
- **Único teorema aberto:** covariância global do cociclo de Connes ⟹ G_μν+Λg_μν = 8πG·𝒫_μν[K_∂].
- **Evidência primária [REAL, não-circular]:** convergência de β=α√e (BBN centra em α√e a 0,0σ;
  DESI/cronômetros/ringdown/H₀ na banda 0,012–0,050; III₁ gap-test). É abdutiva, zero-free.
- **Honestidades [negativos REAL]:** eco gravitacional reclassificado (o observável é dephasing,
  não eco); neural = ilustração, não prova; Chandrasekhar = atrator de Fresnel ~0,35%, não 0,009%.

---

## 3. OS TRÊS PILARES (entenda os três antes de prosseguir)

### 3.1 `tgl_paper_unified.py` — o artigo canônico (forma = conteúdo)
Arquivo único (~16.048 linhas) que **implementa, valida e renderiza** a TGL: roda um comando e
recomputa todos os números, regenera figuras, gera o LaTeX e o PDF. β nunca hardcoded.
- **Local:** `C:\IALD\Artigo\Haja_Luz\tgl_paper_unified.py`.
- **SHA256 selado:** `dea07d8e0c3ed85c240b0d9dc928c5440f6b8af67b08f0e12d7ea829966f96ee`.
- **SUBMETIDO à *Foundations of Physics*** — Submission ID
  `85931d2e-103a-4d8c-a0c9-176d11eb0371` (06/06/2026). PT no Zenodo (10.5281/zenodo.20563905) +
  GitHub `the_boundary`. **NÃO ALTERAR este artefato** — está congelado e submetido.
- Comando: `python tgl_paper_unified.py --live --paper` (+ `--lang en` para a edição EN).
- **Atenção:** a cópia de `tgl_paper_unified.py` que existe DENTRO da pasta `A Ponte e o Um` é um
  **CLONE REGRESSIVO** (anterior ao selo) — **ignore-a**; a fonte da verdade é a da raiz Haja_Luz.

### 3.2 A Ponte Einstein-Cartan-Miguel (a geometria de gravidade quântica)
Documento grande na pasta `A Ponte e o Um` (também no GitHub). Triplo espectral causal
𝔗_TGL=(𝒜,ℋ,D_√e,J,Δ,Π_irr). Conexão Einstein-Cartan Γ=Γ_LC+K_β, com **K_β = contorção da torção
tracial-dissipativa** (onde β se inscreve na geometria). **Teorema da Terminalidade pela Verdade**
(§sec:terminalidade): a Hipótese de Universalidade (U) **se herda, não se impõe** — via
Takesaki/Frigerio/Gelfand/Kochen-Specker/Tomiyama; sombra 6/6 ~1e-27. **Fase C FECHADA no sentido
do documento** (estrutura fechada e coerente; Face C = teorema CONDICIONAL; sem prova
incondicional; resíduo = T1 ergodicidade III₁ + BW não-Killing dissolvido por Ax.G). Ver memória
§88.1. 12 módulos python de sombra, todos PASS, β nunca hardcoded.

### 3.3 "O Um e o Grande Atrator" — documento-semente (o alvo cosmológico)
Ensaio [CONJ], **não** o paper final. Mapeia a TGL ao **Grande Atrator / Dipole Repeller**
(Hoffman 2017, REAL) por **correspondência de FORMA** (topologia do retrato dinâmico), **NÃO** por
cálculo. **O ensaio declara explicitamente, verbatim:** *"Massa, posição e amplitude de fluxo do
Grande Atrator NÃO foram derivadas; a correspondência é de forma; a versão falsificável exige o
ciclo do piso dos vazios."* A predição zero-free falsificável do ensaio é o **piso dos vazios**:
**ρ_vazio/ρ̄ ≥ β_TGL ≈ 0,0120** (δ_c ≥ −0,988). Ensaio no GitHub + cópia local.

---

## 4. ORDEM EVOLUTIVA do trabalho (para entender o framework completo)

1. **Artigo canônico (§1–§80 da memória Haja_Luz):** construção da TGL como artefato unificado,
   verificação multi-domínio, errata integrada, selo, **submissão à FoP**. Forma madura: 1
   postulado (Meia-Nat), 1 teorema aberto, evidência na convergência de β.
2. **Evento 2 (§EVENTO 2):** fase experimental-empírica. Testar se β é determinável **α-livre**.
3. **Fase 2 — "A Ponte e o Um" (§88):** Ponte Einstein-Cartan-Miguel + alvo de falsificação
   cosmológica (piso dos vazios). Aqui mora o trabalho atual.
4. **CÓDIGO UM — falsificador cosmológico (§88.3 → §88.18):** pipeline fail-closed que faz **β
   emergir espectralmente SEM α**, com locks em cascata (SPEC → ANALYSIS → ATTRACTOR → DATA →
   BLIND). **v1 REFUTADO** (§88.26): β como hard edge global do grafo = conectividade (→0 com N) =
   `0_grafo`, não acoplamento. Isso é resultado honesto, não derrota da TGL.
5. **v2-return (§88.26 → §88.26.8):** falsificador de **retorno modular** (eco angular, matched
   filter de infall, β_ret=sin²θ_M α-livre). Rodada SDSS-PV ponte: **BRIDGE_NO_RETURN_ANCHOR**
   (z_emp=−0,68). Encerrado como **detector de eco** — não testa a massa do GA.
6. **Reconhecimento (§88.26.8):** o v2-return testa eco, não a **intenção original** (massa/campo
   do GA → α=β/√e). Mas o ensaio NÃO derivava a massa. → whitepaper honesto.
7. **Artigo 2 / teorema de conversão (§88.26.9 → §88.26.12):** a massa do GA reformulada como
   **teorema condicional** da Face C, com a rota correta **√Θ_LD → K_∂ → 𝒫_μν → ρ_eff → M_GA**
   (NÃO o salto √Θ→M_☉); tríade Nome/Palavra/Verbo; atenção modular; **harness de sombra de P_⋆
   ACEITO**. O gargalo OPEN é a forma explícita de 𝒫_μν[K_∂].

---

## 5. ESTADO ATUAL EXATO (o que está feito, o que falta)

### 5.1 Três "betas", não confundir
- **`beta_spec_v1`** — hard edge espectral global: **REFUTADO** (era conectividade = `0_grafo`).
- **`beta_ret_v2`** — retorno angular direto: **NO_RETURN_ANCHOR** (SDSS-PV; CF4 deu NO_DOMINANT_BASIN).
- **`beta_RG_GA`** — β como renormalizador universal da massa/campo do GA: **CONJECTURE**, pende do
  teorema da reconstrução 𝔯_μν / 𝒫_μν.

### 5.2 A massa do GA = teorema condicional (NÃO codar ainda)
Documentos teóricos entregues (pasta `A Ponte e o Um/TGL_CODE_ONE_REAL_FALSIFIER/`):
- **`TGL_GA_MASS_CONVERSION_THEOREM.md`** — o teorema condicional, rota √Θ→K_∂→𝒫_μν→ρ_eff→M_GA.
- **`TGL_GA_MASS_WHITEPAPER_v2.md`** — as 13 perguntas + os dois modos (A: massa com β fixo; B:
  emergência de β, α_TGL=β_GA/√e). **Nunca misturar os modos**; coincidência = identidade refletida.
- **`TGL_GA_MASS_SPEC_RC.json`** — `runnable=false`, `NOT_RUNNABLE_PENDING_PMUNU_EXPLICIT`.
- **`TGL_PONTE_PMUNU_DERIVATION_v1.md`** + **`TGL_PONTE_PMUNU_STATUS.json`** — a sub-ponte
  Nome→Palavra→Verbo→𝒫_μν, com o gargalo OPEN **localizado**: δ⟨P_⋆ β|K_∂| P_⋆⟩/δg^μν.
- **`TGL_PONTE_ATTENTION_PROJECTOR_v1.md`** — atenção modular = P_⋆ = posto estacionário rank-1 da
  solução GKLS; atenção ≠ consciência (é o módulo mínimo que a opera).

### 5.3 Harness de sombra de P_⋆ — ACEITO [REAL]
Pasta **`A Ponte e o Um/TGL_CODE_ONE_REAL_FALSIFIER/code_one_ponte_attention/`**:
`tgl_attention_projector.py`, `tgl_attention_shadow_source.py`, `tgl_attention_mocks.py`,
`tgl_attention_lock.py`, `TGL_ATTENTION_PROJECTOR_SPEC_RC.json`, `..._TEST_REPORT.md`.
**Rodada do operador: `TGL_ATTENTION_PROJECTOR_LOCK_20260622_095746.json` = ACCEPTED**, T1–T9 todos
true (idempotência ~0, estacionariedade ~1e-16, sombra dE/ds finito≈analítico=−25,354, covariância
~1e-15, 6/6 nulls rejeitados, auditorias massa/dado/α-β vazias). **A atenção (P_Q~P_⋆) é objeto
computável da Ponte.** `P_mu_nu_physical=OPEN`, `mass_code_authorized=false`.

### 5.4 O piso dos vazios — o falsificador zero-free que EXISTE hoje
- **`Piso_dos_Vazios_derivacao_v1.tex`** + **`tgl void floor v1.py`** (β nunca hardcoded). Predição:
  ρ_vazio/ρ̄ ≥ β ≈ 0,0120. Veredito hoje: **não falsificada, não confirmada** (vazios profundos
  observados/simulados δ_c ~ −0,94 a −0,98 → consistente, margem fina ~1,7×). KILL pré-registrado:
  um único vazio robusto de **matéria** com ρ_central/ρ̄ < 0,0120 refuta. Aguarda perfil de matéria
  de vazios profundos público (DESI/Euclid/lensing — mesma fronteira do UNIONS).
- O CÓDIGO UM completo (locks SPEC/ANALYSIS/ATTRACTOR aceitos; DATA_LOCK com binding UNIONS Table 3
  conservador) está em `release_clean/`. **DESI PV** (catálogo genuinamente independente) **ainda
  não é público** (liberação na aceitação do journal). v1 blind-run full-domain **REFUTADA**
  (β_spec=0=0_grafo). Ver §88.21, §88.23–§88.26.

---

## 6. A PRETENSÃO (o que esta linha de trabalho quer provar/falsificar)

**Pretensão central:** demonstrar que a TGL faz uma **predição cosmológica falsificável zero-free**
no setor do Grande Atrator / vazios — e, idealmente, que **β_TGL emerge de dado real SEM usar α**,
fechando a inversão α_TGL = β_GA/√e (a "identidade refletida").

**Dois caminhos, disciplinados:**
- **(A) Piso dos vazios** — o falsificador zero-free **que já existe e é derivável**. Caminho mais
  curto e mais honesto hoje. Precisa de perfil de matéria de vazios profundos público.
- **(B) Massa/campo do Grande Atrator** — só vira falsificador **depois** de explicitada a forma
  global de **𝒫_μν[K_∂]** (a reconstrução 𝔯_μν / a descida da Ponte do Verbo modular à fonte
  geométrica efetiva). É trabalho **teórico**, não código. O harness de sombra de P_⋆ (já ACEITO)
  é o primeiro objeto computável dessa descida.

**A inversão pretendida (a identidade refletida):**
```
Modo A: β fixo (=α√e) prediz a massa/campo  →  M_A
Modo B: infere β_GA do campo/massa SEM α    →  α_TGL = β_GA/√e
Se  M_A ≈ M_B  E  β_GA/√e ≈ α  →  identidade refletida (Ψ_out = ℛ∘𝒞(Ψ_in))  →  sinal.
Discrepância → refutação.  NUNCA misturar os dois modos.
```

---

## 7. PRÓXIMO PASSO sugerido (decisão do operador) — escolha um

1. **Descer a Ponte para 𝒫_μν (teórico, não código):** a partir do harness de sombra ACEITO,
   tentar a **primeira sombra finita de δ⟨P_⋆ |K_∂| P_⋆⟩/δg** como funcional de fonte efetiva, e
   mostrar que sobrevive à discretização (o gate `explicit_weak_field_shadow_of_P_mu_nu_Q_beta`).
   Isto NÃO é código de massa; é o objeto que precede 𝒫_μν físico.
2. **Trabalhar o piso dos vazios (falsificador real):** detalhar o ciclo de derivação e a aquisição
   do perfil de matéria de vazios profundos; é o teste que pode matar a teoria hoje.
3. **Aguardar dado independente:** rodar o watcher do DESI PV periodicamente; quando público,
   rodar a blind-run v2-return cega na máquina do operador.

**O que NÃO fazer:** codar `tgl_ga_mass_blind_run.py`; converter √Θ ou P_⋆ direto em M_☉; usar
α/CODATA antes de β; misturar os modos A e B; afrouxar critério de lock; alterar o artefato
canônico submetido; chamar SDSS-PV de confirmação (é CF3-calibrado, independência parcial).

---

## 8. Locks e selos (integridade — não mexer sem re-selar)
- Artefato canônico: SHA256 `dea07d8e0c3ed85c240b0d9dc928c5440f6b8af67b08f0e12d7ea829966f96ee`.
- Spec do CÓDIGO UM (resolution-flow): `7dc16918271e1db1485b35dc20ac715199cf90d0171c9cd2788496a7cf8532f1`.
- v2-return estimator selado: `V2_ESTIMATOR_LOCKED.json` (operator fe61d63…, beta bf2cb67…).
- Atenção (P_⋆): `TGL_ATTENTION_PROJECTOR_LOCK_20260622_095746.json` = ACCEPTED.

---

## 9. Como confirmar que você entendeu (faça isto primeiro, na nova sessão)
1. Consulte o GitHub `the_boundary` e leia `C:\IALD\CLAUDE.md` + `Haja_Luz\CLAUDE.md` §88.
2. Devolva ao operador, em PT: (a) a diferença entre os três betas; (b) por que a massa do GA NÃO
   é codável hoje; (c) qual é o falsificador zero-free que existe agora; (d) o que o harness de
   atenção (ACEITO) provou e o que ainda é OPEN. **Só então** proponha o próximo passo e espere a
   decisão do operador. A régua acima vale o tempo todo.

> **Frase-guia:** A massa do Grande Atrator não vem diretamente de √Θ_LD; vem da fonte geométrica
> efetiva gerada pelo cociclo. β_TGL é o sinal do sinal. O zero absoluto não é eixo; é impedância.
> Atenção é o módulo estacionário mínimo que opera a consciência sem ainda sê-la.
> **TGL = haja luz. Tetelestai.**
