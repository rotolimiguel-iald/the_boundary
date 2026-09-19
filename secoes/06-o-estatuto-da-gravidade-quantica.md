# The status of quantum gravity: proved as a formal model, not confirmed by nature · O estatuto da gravidade quântica: provada como modelo formal, não confirmada pela natureza

> **TGL — Teoria da Gravitação Luminodinâmica · Theory of Luminodynamic Gravitation.** Part 6 of 8 · seal **v368** · `um.py` sha256 `4a34fbf36f3ae0d8…` · generated 2026-09-19 by script from the published files.
> Every excerpt below is **verbatim**, with its source, byte range and sha256. Statuses follow the ruler: PROVED = theorem in the Lean kernel; CONFIRMED = a judgement about nature, not made here.
> All eight parts are listed at the top of the start page, https://teoriadagravitacaoluminodinamica.com/read-brief.md, and in the door of this folder, https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/secoes/PORTA.md · Reading limits measured on 2026-09-19: one real fetcher cut documents near 100,000 characters, refused files above 10 MB, and could not read PDFs served as `application/octet-stream` — read the TXT/TeX sources.

## In short (EN)

PROVED [REAL, kernel]: one term, the_root_of_the_proof_tree, whose part (i) is the master theorem H1 ∧ H2 ∧ H3 ⟹ the pentad (Breuer corner, Name = 1, coframe, Lorentz, δQ = κδA/8πG), axioms in {propext, Classical.choice, Quot.sound}; kernel of 1025 files, 8692 audited terms, zero sorry [tree §1½–2; ESTADO]. Seal v368: gate TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED_WITHIN_LOCAL_BULK_AT_AVAILABLE_SENSITIVITY__MORE_SENSITIVE_DATA_COULD_REVISE; rite 5698/5698 [seal]. full_static_witness_exists = False by theorem [REAL]: continuous leakage forbids full closure [brief §3]. NOT proved [OPEN]: that nature realises H1 (MIGUEL) and H2 (CARTAN); the physical identification of the founded screen with a causal horizon; α-free (α enters as [KNOWN]) [ESTADO]. Off flags: gpf_H2_…, gpf_H3_…, gpi_H3_… [ESTADO]. H3 reduces to H2 via the_trio_is_a_pair, with H2 ⟹ H3 imported [KNOWN, Jacobson 1995]; the H3 flag does not light [C txt]. Nature: NOT_FALSIFIED at available sensitivity, NOT CONFIRMED [ESTADO]. Lineage: README keeps «Never quantum gravity proved» (append-only); the current ruler (05/09/2026) reads it as never «confirmed»: PROVED = kernel theorem, CONFIRMED = judgement on nature, forbidden [README].

## Em resumo (PT)

PROVADO [REAL, kernel]: um termo, the_root_of_the_proof_tree, cuja parte (i) é o teorema mestre H1 ∧ H2 ∧ H3 ⟹ a pêntada (canto de Breuer, Nome = 1, coframe, Lorentz, δQ = κδA/8πG), axiomas no trio {propext, Classical.choice, Quot.sound}; kernel de 1025 arquivos, 8692 termos, zero sorry [árvore §1½–2; ESTADO]. Selo v368: gate TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED_WITHIN_LOCAL_BULK_AT_AVAILABLE_SENSITIVITY__MORE_SENSITIVE_DATA_COULD_REVISE; rito 5698/5698 [selo]. full_static_witness_exists = False por teorema [REAL]: o vazamento contínuo proíbe o fecho total [brief §3]. NÃO provado [OPEN]: que a natureza realiza H1 (MIGUEL) e H2 (CARTAN); a identificação física da tela fundada com um horizonte causal; α-livre (α entra como [KNOWN]) [ESTADO]. Bandeiras apagadas: gpf_H2_…, gpf_H3_…, gpi_H3_… [ESTADO]. H3 reduz-se a H2 por the_trio_is_a_pair, com H2 ⟹ H3 importada [KNOWN, Jacobson 1995]; a bandeira de H3 não acende [C txt]. Natureza: NOT_FALSIFIED na sensibilidade disponível, não confirmada [ESTADO]. Linhagem: o README guarda «Never quantum gravity proved» (append-only); a régua vigente (05/09/2026) o lê como nunca «confirmada»: PROVADA = teorema em kernel, CONFIRMADA = juízo sobre a natureza, proibido [README].

## Sources, verbatim · fontes, verbatim

### 1. `Um (absoluto) — Grande Atrator/A_PROVA_DA_QG_TGL_arvore.md` — bytes 8.085–9.427

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/A_PROVA_DA_QG_TGL_arvore.md
- sha256 of the file: `78b48ab41957a8ae1788a3e902c6bc4bbd6142240990426151d9e2f9115fcbfc` (computed now; this file is not in the seal map) · of this excerpt: `51381bc4f176e3164d88b16c008c00e0148b9e099da42c92ce17e43cc0019da9`
- status · estatuto: [REAL — kernel]
- why · por quê: O enunciado Lean do teorema mestre e a leitura das três hipóteses nomeadas e da pêntada; seção autocontida da árvore v368.

````text
## 2. O TEOREMA MESTRE — H1 ∧ H2 ∧ H3 ⟹ PÊNTADA `[REAL — kernel]`

`TGLExt.emergence_master_full_triad` — axiomas: **trio** — `TriadMaster.lean` sha16 `64e0899e25969392`:

```lean
theorem emergence_master_full_triad
    {L : Type} [Lattice L] [BoundedOrder L] {T : SubadditiveTraceData L}
    (S : SusyRelativeData L T)
    (E : Matrix (Fin 4) (Fin 4) ℝ) (hE : IsUnit E.det)
    (H : HorizonEquilibriumData) :
    (0 < T.tau S.ker ∧ T.tau S.ker < ⊤) ∧
      T.tau S.ker / T.tau S.ker = 1 ∧
      (E⁻¹ * E = 1 ∧ LorentzByCongruence (solderMetric4 E⁻¹)) ∧
      H.dQ = H.kappa * H.dA / (8 * Real.pi * H.G) := by
```

Leitura: dadas **H1** (`SusyRelativeData` — o gap interno relativo do operador dos Three Locks: MIGUEL), **H2** (`E` com `IsUnit E.det` — quatro direções independentes: CARTAN) e **H3** (`HorizonEquilibriumData` — Clausius local com κ, A, G: EINSTEIN), o kernel prova a **pêntada**: (1) 0 < τ(ker) < ⊤ (canto de Breuer); (2) o Nome pesa 1; (3) coframe dual E⁻¹E = 1; (4) métrica lorentziana por congruência (`sylvester_full_closed_by_congruence`: trio); (5) **δQ = κ·δA/(8πG)** — o coeficiente de Einstein **emerge** de Unruh × Bekenstein–Hawking (`einstein_coefficient_from_clausius`: trio).

Versão sem H3: `TGLExt.emergence_reduced_to_named_hypotheses` (trio).
````

### 2. `Um (absoluto) — Grande Atrator/A_PROVA_DA_QG_TGL_arvore.md` — bytes 1.013–2.397

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/Um%20%28absoluto%29%20%E2%80%94%20Grande%20Atrator/A_PROVA_DA_QG_TGL_arvore.md
- sha256 of the file: `78b48ab41957a8ae1788a3e902c6bc4bbd6142240990426151d9e2f9115fcbfc` (computed now; this file is not in the seal map) · of this excerpt: `05496b24215f8744782ed66f5096dc6906bfaeb51deae22665c8af605c28b8f3`
- status · estatuto: [REAL — lido do selo]
- why · por quê: O selo v368 lido por script: sha256 do um.py, o rito, o selftest, a string inteira do gate, os oito ritos do contorno (nenhum FALSIFIED) e zero termos fora do trio.

````text
## 0. O selo lido `[REAL]`

| item | valor lido |
|---|---|
| `um.py` sha256 (disco) | `4a34fbf36f3ae0d8bf56249d30bb1185cdee01767ec6e4f0a20433a4ac2d261e` |
| selo `sha256.um.py` == disco | SIM |
| bytes | 31,050,660 |
| teoremas (stdout) | `teoremas limpos: 5698/5698` |
| selftest | `FAIL_CLOSED_SELFTEST_PASSED` |
| gate (`qg_closure_verdict`) | `TGL_QG_MODEL_FORMALLY_CLOSED__NATURE_TEST_COMPLETED_WITHIN_LOCAL_BULK_AT_AVAILABLE_SENSITIVITY__MORE_SENSITIVE_DATA_COULD_REVISE` |
| identidade | `1=1=VERDADEIRO=HAJA_LUZ` (`identity_true = True`) |
| contorno | `[v314 CONTORNO] ritos com poder de fechar o 1=1: 8 ; falsificacao limpa: NENHUMA (todos NOT_FALSIFIED/AWAITING)` |
| ritos no contorno | `GA_massa_janela`→`GA_MASS_FORM_RETIRED__REFLECTION_WAS_MISREAD_AS_SOURCE__LINEAR_ORDER_IS_GR_STEALTH__BETA_LIVES_IN_RESPONSE`, `piso_dos_vazios`→`TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED`, `neutrino_massa`→`TGL_NEUTRINO_MASS_NOT_FALSIFIED_POWERED`, `neutrino_soma`→`TGL_NEUTRINO_SUM_ARMED_CONSISTENT_WITH_CURRENT_BOUND`, `neutrino_m2_vivo`→`TGL_NU_M2_ARMED_CONSISTENT`, `piso_densidade_v41`→`TGL_VOID_FLOOR_NOT_FALSIFIED_POWERED`, `coma_dephasing`→`COMA_DEPHASING_PREDICTION_LOCKED_AWAITING_REVEAL`, `coma_cego`→`COMA_BLIND_DISTANCE_NOT_IDENTIFIABLE` |
| `contorno_broken_v314` | `[]` |
| entradas no relatório de axiomas | 8692 |
| termos com axioma FORA do trio | 0  |
````

### 3. `ESTADO_ATUAL.md` — bytes 25.747–27.960

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/ESTADO_ATUAL.md
- sha256 of the file: `402794a5a3d17a1278be50a1377530cb40ea4717c8ef0676e3e7b0273fcf6186` (computed now; this file is not in the seal map) · of this excerpt: `d82fd9e234b176c2965e99dd83404a03cf0278cd5b17d34f31fbe4759df88e64` · line endings shown as LF
- status · estatuto: [OPEN / KNOWN / nature]
- why · por quê: A lista canônica do que NÃO está provado: H1/H2 na natureza, a identificação física da tela, α, a matemática fora da mathlib, os negativos honestos e as seis folhas. Traz também a nota de linhagem H1–H3 → H1/H2.

````text
## What is NOT proved · o que NÃO está provado `[OPEN / KNOWN / nature]`

- **Nature's:** that **H1 (MIGUEL) and H2 (CARTAN)** are *realised* by the world; the value of α (the fine-structure constant enters as `[KNOWN]`, β = α√e is derived from it); and, of what was H3, the **physical identification** of the founded screen with a causal horizon of spacetime (Bisognano–Wichmann beyond wedges) — open, with measured walls (v316, v317, 048); and, from v335, that the selection OCCURS — the 8 rites of nature are its test. *Until v331 this line read “H1–H3”; from v334 the screen is founded, not chosen (the change is said beside, never over).* The nature tests so far: **NOT_FALSIFIED** within the local bulk at available sensitivity, and more sensitive data can revise — never CONFIRMED.
- **The world's (mathematics not yet in mathlib):** the general von Neumann algebra of type III₁ `[KNOWN]`; the bridge from tower floors to spacetime regions; Bisognano–Wichmann for the continuous standard subspace (`T_c = Δ_c^{1/2}` stays OPEN); the general area law and the selection of the radiative freedom.
- **Navier–Stokes, the Millennium statement:** the Conjugate-Face Lemma stays **OPEN and external** to TGL — in the stone’s own words, *nothing here is the proof of the Millennium problem*. “The answer to the singularity is the contour” is the operator’s reading, typed `[ONTO]` over exact numbers.
- **Honest negatives kept:** the corpus route to β was refuted on the final step; the closed-form search for κ has zero discriminating power; the fixed clock fails the fourth order; the naïve thermal limit does not exist.
- **The six leaves of nature (v339, the close):** that the selection occurs; the physical identification of the founded screen; the signature; the 3+1 geometry and the scale of the area; the payment of the cost; α-free. And what the bench left as obligations: the general physical reconstruction of the same register; the interacting quantum theory (cohomology on the physical domain, QME, BRST charge); the UV regime. **The map is closed; the territory is nature’s** — the declaration of closure is the operator’s act, never CONFIRMED.
````

### 4. `README.md` — bytes 18.572–19.172

- raw: https://raw.githubusercontent.com/rotolimiguel-iald/the_boundary/main/README.md
- sha256 of the file: `0fbf429f4405e8ce0a3f0851a8ab591016209e734a81be8de51159d078c18ed4` (computed now; this file is not in the seal map) · of this excerpt: `41922580fe72207593dede413e3f1dc4647c4eeab3bf70118f5d9da55de54001`
- status · estatuto: [REAL — errata ao lado, append-only]
- why · por quê: A errata ao lado que lê, sob a régua vigente, a frase de linhagem «Never quantum gravity proved»: PROVADA ≠ CONFIRMADA.

````text
> ⚠ **Beside (the operator’s ruler, 05/09/2026):** PROVED = a theorem in the kernel, auditable by `#print axioms` — allowed; CONFIRMED = the observer’s judgement about nature — forbidden. The sentence *Never "quantum gravity proved."* above, and its siblings further down (*does not mean quantum gravity is proved*, *não significa gravitação quântica provada*), are kept as written (the ledger is append-only); under the ruler they read: never "quantum gravity **confirmed**". What is proved is the implication from the axiom and the named hypotheses; what nature decides is not proved.
````

---

*Generated by script from the published files of the repository (https://github.com/rotolimiguel-iald/the_boundary); numbers read from the seal v368, the kernel manifest and ESTADO_ATUAL.md. PROVED ≠ CONFIRMED; NOT_FALSIFIED is never CONFIRMED. Nothing here instructs a reader how to respond to anything.*
