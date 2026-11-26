import TGLExt.OAlgoritmoPenP

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# O livro consultado: nenhuma porta se tenta duas vezes no mesmo ponto

## O que é

`processoComLivro` é a busca com a poda em que a pilha guarda só pontos e as portas moram no livro, e o livro é consultado: num
ponto, só entram as portas limpas que o livro não registra naquele ponto (`limpasNoveisDe`). O ponto é o conteúdo do espectro
(`conteudoDe`: em cada variável, o valor da primeira equação do espectro que a contém), não a ordem do registro. Quando não resta
porta nova, a busca sobe ao ponto mais recente da pilha; a pilha vazia é o silêncio.

## O que fica provado [REAL]

1. `conteudoDe`, `mesmoPonto`, `tentadaNoPonto`, `limpasNoveisDe`, `lacoComLivro`, `processoComLivro` [DEF].
2. ★★ `o_eco_do_livro_e_fiel`: a busca só responde com ponto de custo zero.
3. ★★ `o_livro_nunca_repete`: os registros do livro (o conteúdo do ponto e a porta) são sempre distintos.
4. ★★ `a_cota_do_livro`: o tamanho do livro ao fim nunca passa de `aCotaDoLivro n`, o número de pares (conteúdo, porta), que é
   3ⁿ·2n: finito e exponencial no número de variáveis.
5. ★★ `a_energia_do_combustivel`: se o laço responde com silêncio, ou existe um estado de silêncio terminal (`oSilencioTerminal`:
   o ponto da descida tem custo, a primeira cláusula sem par existe e não resta porta nova), ou o combustível foi pago em portas e
   pontos: `2·|livro ao fim| + |pilha| ≥ combustível + 2·|livro no início|`.
6. ★★ `o_combustivel_da_luz_basta`: com combustível de pelo menos `2·aCotaDoLivro n + 1`, se o processo responde com silêncio,
   existe um estado de silêncio terminal.
7. ★★ `a_medida_a_maquina_do_livro_decide`: por `decide`, a busca com o livro responde com ponto nas mesmas sete fórmulas com
   solução da medida de `OAlgoritmoPenP` e cala na fórmula sem solução de uma variável.

## O que NÃO se afirma

* Os enunciados 5 e 6 afirmam que existe um estado de silêncio terminal, mas não que a corrida passou por ele. Como estão
  escritos, não provam que a corrida termina pelo detector antes de o combustível acabar; provam que o livro nunca passa de
  3ⁿ·2n registros e que, se a corrida responde com silêncio e não existe estado de silêncio terminal algum, ela registrou em
  portas pelo menos metade do combustível.
* Que o silêncio desta busca signifique que a fórmula não tem solução (a completude do veredito): não está provado neste módulo;
  está provado em `ODentroOuMorto` (`o_veredito_decide_com_a_cota`), com combustível de pelo menos `2·aCotaDoLivro n + 1`, pelo
  invariante do livro sadio (toda porta registrada está dentro do ramo corrente ou numa subárvore sem solução).
* Que combustível polinomial no tamanho da fórmula baste: segue [OPEN]; a cota desta pedra é exponencial.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OLivroConsultado

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno TGLExt.AMedidaDaArvore
open TGLExt.AFamiliaProfunda TGLExt.ADescidaPinadaEmMemoria TGLExt.ALeituraCorrespondente
open TGLExt.OAlgoritmoPenP

/-- [DEF] O conteúdo do espectro na variável: o valor que o espectro fixa nela, se fixa. (O ponto da
doutrina é o conteúdo, não a ordem do registro.) -/
def conteudoDe {n : ℕ} (S : Espectro2 n) (j : Fin n) : Option (ZMod 2) :=
  (S.find? (fun e => e.1 j)).map Prod.snd

/-- [DEF] O mesmo ponto: os dois espectros fixam o mesmo conteúdo. -/
def mesmoPonto {n : ℕ} (S₁ S₂ : Espectro2 n) : Bool :=
  (List.finRange n).all (fun j => decide (conteudoDe S₁ j = conteudoDe S₂ j))

/-- [DEF] O livro consultado: a porta foi tentada no ponto. -/
def tentadaNoPonto {n : ℕ} (livro : List (Espectro2 n × Literal n)) (S : Espectro2 n) (l : Literal n) :
    Bool :=
  livro.any (fun p => mesmoPonto p.1 S && mesmaLiteral p.2 l)

/-- [DEF] As portas novas do ponto: as limpas que o livro não registra no ponto. -/
def limpasNoveisDe {n : ℕ} (φ : CNF n) (livro : List (Espectro2 n × Literal n)) (S : Espectro2 n)
    (C : Clausula n) : List (Literal n) :=
  (limpasDe φ S C).filter (fun l => !tentadaNoPonto livro S l)

/-- [DEF] **O LAÇO COM O LIVRO CONSULTADO**: a pilha guarda só pontos; as portas moram no livro. A morte
registra a porta nova do ponto e empilha o ponto; o detector (limpas novas vazia) dispara a subida ao
ponto mais recente; de volta ao ponto, as portas tentadas não aparecem — nenhuma porta se tenta duas
vezes no mesmo ponto. -/
def lacoComLivro {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → List (Espectro2 n) → List (Espectro2 n × Literal n) →
      (Option (Fin n → Bool)) × Espectro2 n × List (Espectro2 n × Literal n)
  | 0, S, x, _, livro => (if custoCNF φ x = 0 then some x else none, S, livro)
  | fuel + 1, S, x, pilha, livro =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S, livro)
    else
      match primeiraSemPar φ w with
      | none => (none, S, livro)
      | some C =>
        match limpasNoveisDe φ livro S C with
        | [] =>
          match pilha with
          | [] => (none, S, livro)
          | S' :: pilha' => lacoComLivro φ fuel S' (ressurreicaoConstrutiva S') pilha' livro
        | limpasNovas =>
          match oMelhorDe φ S limpasNovas with
          | some l =>
            lacoComLivro φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
              (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
              (S :: pilha) ((S, l) :: livro)
          | none => (none, S, livro)

/-- [DEF] O processo com o livro consultado: o campo fala o fecho; o laço corre do livro vazio. -/
def processoComLivro {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :=
  let S₀ := fechoDoCampo φ (n + 2) []
  lacoComLivro φ fuel S₀ (ressurreicaoConstrutiva S₀) [] []

/-- ★★ **O ECO DO LIVRO É FIEL**: o laço com o livro consultado só responde com ponto de custo zero. -/
theorem o_eco_do_livro_e_fiel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool) (pilha : List (Espectro2 n))
      (livro : List (Espectro2 n × Literal n)) (S' : Espectro2 n)
      (livro' : List (Espectro2 n × Literal n)),
      lacoComLivro φ fuel S x pilha livro = (some w, S', livro') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w pilha livro S' livro' h
    unfold lacoComLivro at h
    rw [Prod.mk.injEq] at h
    obtain ⟨h1, -⟩ := h
    split at h1
    · rename_i hcond
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · simp at h1
  | succ f ih =>
    intro S x w pilha livro S' livro' h
    unfold lacoComLivro at h
    dsimp only [] at h
    split at h
    · rename_i hcond
      rw [Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · split at h
      · simp at h
      · split at h
        · split at h
          · simp at h
          · exact ih _ _ _ _ _ _ _ h
        · split at h
          · exact ih _ _ _ _ _ _ _ h
          · simp at h

/-- [DEF] A imagem do registro do livro: o conteúdo do ponto (como função) e a porta (como par). -/
def imagemDoLivro {n : ℕ} (p : Espectro2 n × Literal n) : (Fin n → Option (ZMod 2)) × (Fin n × Bool) :=
  (fun j => conteudoDe p.1 j, (p.2.var, p.2.pos))

/-- O mesmo ponto, na forma da igualdade ponto a ponto. -/
theorem mesmoPonto_sse {n : ℕ} (S₁ S₂ : Espectro2 n) :
    mesmoPonto S₁ S₂ = true ↔ ∀ j : Fin n, conteudoDe S₁ j = conteudoDe S₂ j := by
  unfold mesmoPonto
  rw [List.all_eq_true]
  constructor
  · intro h j
    have hj := h j (List.mem_finRange j)
    exact of_decide_eq_true hj
  · intro h j _
    exact decide_eq_true (h j)

/-- A imagem é fiel ao ponto e à porta: imagens iguais, mesmo ponto e mesma literal. -/
theorem imagem_fiel {n : ℕ} (p q : Espectro2 n × Literal n)
    (h : imagemDoLivro p = imagemDoLivro q) :
    mesmoPonto p.1 q.1 = true ∧ mesmaLiteral p.2 q.2 = true := by
  unfold imagemDoLivro at h
  rw [Prod.mk.injEq] at h
  obtain ⟨hpt, hpr⟩ := h
  constructor
  · rw [mesmoPonto_sse]
    intro j
    exact congrFun hpt j
  · unfold mesmaLiteral
    rw [Prod.mk.injEq] at hpr
    obtain ⟨hv, hpo⟩ := hpr
    rw [Bool.and_eq_true]
    exact ⟨beq_iff_eq.mpr hv, beq_iff_eq.mpr hpo⟩

/-- O livro consultado, na forma testemunhada: a porta foi tentada no ponto sse algum registro do livro
é daquele ponto e daquela porta. -/
theorem tentada_sse {n : ℕ} (livro : List (Espectro2 n × Literal n)) (S : Espectro2 n) (l : Literal n) :
    tentadaNoPonto livro S l = true ↔
      ∃ p : Espectro2 n × Literal n, p ∈ livro ∧ mesmoPonto p.1 S = true ∧ mesmaLiteral p.2 l = true := by
  unfold tentadaNoPonto
  rw [List.any_eq_true]
  constructor
  · intro h
    obtain ⟨p, hp, hpred⟩ := h
    exact ⟨p, hp, by rwa [Bool.and_eq_true] at hpred⟩
  · intro h
    obtain ⟨p, hp, h1, h2⟩ := h
    exact ⟨p, hp, by rw [Bool.and_eq_true]; exact ⟨h1, h2⟩⟩

/-- ★★ **O LIVRO NUNCA REPETE**: as imagens do livro (ponto × porta) são sempre distintas — nenhuma
porta se tenta duas vezes no mesmo ponto. -/
theorem o_livro_nunca_repete {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool) (pilha : List (Espectro2 n))
      (livro : List (Espectro2 n × Literal n)) (res : Option (Fin n → Bool)) (S' : Espectro2 n)
      (livro' : List (Espectro2 n × Literal n)),
      (livro.map imagemDoLivro).Nodup →
      lacoComLivro φ fuel S x pilha livro = (res, S', livro') →
      (livro'.map imagemDoLivro).Nodup := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x pilha livro res S' livro' hnd h
    unfold lacoComLivro at h
    rw [Prod.mk.injEq, Prod.mk.injEq] at h
    obtain ⟨_, _, hlv⟩ := h
    rw [← hlv]
    exact hnd
  | succ f ih =>
    intro S x pilha livro res S' livro' hnd h
    unfold lacoComLivro at h
    dsimp only [] at h
    split at h
    · -- o eco: o livro intacto
      rw [Prod.mk.injEq, Prod.mk.injEq] at h
      obtain ⟨_, _, hlv⟩ := h
      rw [← hlv]
      exact hnd
    · split at h
      · rw [Prod.mk.injEq, Prod.mk.injEq] at h
        obtain ⟨_, _, hlv⟩ := h
        rw [← hlv]
        exact hnd
      · rename_i C hC
        split at h
        · split at h
          · rw [Prod.mk.injEq, Prod.mk.injEq] at h
            obtain ⟨_, _, hlv⟩ := h
            rw [← hlv]
            exact hnd
          · -- a subida: o livro intacto
            exact ih _ _ _ _ _ _ _ hnd h
        · split at h
          · -- a morte: o livro cresce com imagem fresca
            rename_i hnn l hl
            have hlmem : l ∈ limpasNoveisDe φ livro S C := o_melhor_de_mem φ S _ _ hl
            unfold limpasNoveisDe at hlmem
            have hnao : tentadaNoPonto livro S l = false := by
              have h2 := (List.mem_filter.mp hlmem).2
              cases ht : tentadaNoPonto livro S l with
              | false => rfl
              | true => rw [ht] at h2; simp at h2
            have hfresh : imagemDoLivro (S, l) ∉ livro.map imagemDoLivro := by
              intro hmem
              rw [List.mem_map] at hmem
              obtain ⟨p, hp, himg⟩ := hmem
              obtain ⟨hpt, hpor⟩ := imagem_fiel p (S, l) himg
              have ht : tentadaNoPonto livro S l = true := (tentada_sse _ _ _).mpr ⟨p, hp, hpt, hpor⟩
              rw [ht] at hnao
              simp at hnao
            have hnd' : (((S, l) :: livro).map imagemDoLivro).Nodup := by
              rw [List.map_cons, List.nodup_cons]
              exact ⟨hfresh, hnd⟩
            exact ih _ _ _ _ _ _ _ hnd' h
          · -- o melhor de lista viva nunca é none — mas se calar, o livro segue intacto
            rw [Prod.mk.injEq, Prod.mk.injEq] at h
            obtain ⟨_, _, hlv⟩ := h
            rw [← hlv]
            exact hnd

/-- [DEF] **A COTA DO LIVRO**: o universo dos pontos com porta — finito e explícito. -/
def aCotaDoLivro (n : ℕ) : ℕ := Fintype.card ((Fin n → Option (ZMod 2)) × (Fin n × Bool))

/-- ★★ **A COTA DO LIVRO**: o livro da máquina cabe sempre no universo dos pontos com porta. -/
theorem a_cota_do_livro {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool)
    (res : Option (Fin n → Bool)) (S' : Espectro2 n) (livro' : List (Espectro2 n × Literal n))
    (h : lacoComLivro φ fuel (fechoDoCampo φ (n + 2) [])
      (ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) [])) [] [] = (res, S', livro')) :
    livro'.length ≤ aCotaDoLivro n := by
  have hnd := o_livro_nunca_repete φ _ _ _ _ _ _ _ _ (by simp) h
  have hcard : (livro'.map imagemDoLivro).toFinset.card ≤ Fintype.card
      ((Fin n → Option (ZMod 2)) × (Fin n × Bool)) := by
    rw [← Finset.card_univ]
    exact Finset.card_le_card (Finset.subset_univ _)
  rw [List.toFinset_card_of_nodup hnd, List.length_map] at hcard
  exact hcard

/-- [DEF] **O SILÊNCIO TERMINAL**: o estado do veredito — o custo fora do zero, a cláusula morta lida, e
nenhuma porta nova no ponto (o detector disparado com o livro consultado). -/
def oSilencioTerminal {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (livro : List (Espectro2 n × Literal n)) : Prop :=
  custoCNF φ (descerGenerico passoFiel φ S φ.length ([], x)).2 ≠ 0 ∧
    ∃ C : Clausula n, primeiraSemPar φ (descerGenerico passoFiel φ S φ.length ([], x)).2 = some C ∧
      limpasNoveisDe φ livro S C = []

/-- ★★ **A ENERGIA DO COMBUSTÍVEL**: se o laço responde com silêncio, ou existe um estado de silêncio
terminal (o enunciado não diz que a corrida passou por ele), ou o combustível se pagou em portas e pontos:
cada passo ou registra uma porta nova no livro ou sobe um ponto da pilha. -/
theorem a_energia_do_combustivel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool) (pilha : List (Espectro2 n))
      (livro : List (Espectro2 n × Literal n)) (S' : Espectro2 n)
      (livro' : List (Espectro2 n × Literal n)),
      lacoComLivro φ fuel S x pilha livro = (none, S', livro') →
      (∃ S₀ x₀ livro₀, oSilencioTerminal φ S₀ x₀ livro₀) ∨
        (2 * livro'.length + pilha.length ≥ fuel + 2 * livro.length) := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x pilha livro S' livro' h
    unfold lacoComLivro at h
    rw [Prod.mk.injEq, Prod.mk.injEq] at h
    obtain ⟨_, _, hlv⟩ := h
    subst hlv
    right
    omega
  | succ f ih =>
    intro S x pilha livro S' livro' h
    unfold lacoComLivro at h
    dsimp only [] at h
    split at h
    · -- o eco: some = none — impossível
      rw [Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      simp at h1
    · rename_i hcusto
      split at h
      · -- a primeiraSemPar não cala com custo fora do zero
        rename_i hsem
        exfalso
        have his := a_primeira_sem_par_existe φ _ hcusto
        rw [hsem] at his
        simp at his
      · rename_i C hC
        split at h
        · -- o detector disparou: a pilha decide
          rename_i hlimpas
          split at h
          · -- ★ O VEREDITO: o silêncio terminal — a pilha vazia com o detector disparado
            left
            exact ⟨S, x, livro, hcusto, C, hC, hlimpas⟩
          · -- a subida: o livro intacto, a pilha um a menos
            rename_i S₁ pilha₁
            have hih := ih S₁ (ressurreicaoConstrutiva S₁) pilha₁ livro S' livro' h
            cases hih with
            | inl hterm => exact Or.inl hterm
            | inr harit =>
              right
              rw [List.length_cons]
              omega
        · -- a morte: o livro cresce um, a pilha cresce um
          rename_i hnn
          split at h
          · rename_i l hl
            have hih := ih (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
              (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
              (S :: pilha) ((S, l) :: livro) S' livro' h
            cases hih with
            | inl hterm => exact Or.inl hterm
            | inr harit =>
              right
              rw [List.length_cons, List.length_cons] at harit
              omega
          · -- a escolha não trava em lista viva — impossível
            rename_i hmelhor
            exfalso
            have his := o_melhor_de_nao_vazio φ S _ hnn
            rw [hmelhor] at his
            simp at his

/-- ★★ **O COMBUSTÍVEL DA LUZ BASTA**: com combustível maior que o dobro do universo dos pares (conteúdo,
porta), se o processo responde com silêncio, existe um estado de silêncio terminal. O enunciado não liga
esse estado à corrida. -/
theorem o_combustivel_da_luz_basta {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool)
    (S' : Espectro2 n) (livro' : List (Espectro2 n × Literal n))
    (hcomb : 2 * aCotaDoLivro n + 1 ≤ fuel)
    (h : processoComLivro φ fuel x = (none, S', livro')) :
    ∃ (S₀ : Espectro2 n) (x₀ : Fin n → Bool) (livro₀ : List (Espectro2 n × Literal n)),
      oSilencioTerminal φ S₀ x₀ livro₀ := by
  unfold processoComLivro at h
  have hE := a_energia_do_combustivel φ fuel _ _ [] [] S' livro' h
  cases hE with
  | inl hterm => exact hterm
  | inr harit =>
    exfalso
    have hcota := a_cota_do_livro φ fuel x none S' livro' h
    simp at harit
    omega

/-- ★★ **A MEDIDA FINAL: A MÁQUINA DO LIVRO DECIDE** — os três muros, a família profunda, a sanidade e a
honestidade, re-medidos na máquina que consulta o livro: idêntica à do algoritmo no conjunto medido. -/
theorem a_medida_a_maquina_do_livro_decide :
    (processoComLivro phiTrap 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiCE 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiFofa2 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiTrap2 20 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiTrap3 30 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiCadeia 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiFofa 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 12 (fun _ => false)).1 = none := by
  decide

end TGLExt.OLivroConsultado

