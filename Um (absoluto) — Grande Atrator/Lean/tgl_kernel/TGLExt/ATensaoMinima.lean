import TGLExt.OCustoDoProcesso

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A tensão mínima: a escolha pelo custo do ponto de retorno

## O que é

A regra da primeira literal limpa responde com silêncio numa fórmula de três variáveis que tem solução (`phiCE`): o filtro do
campo não condena x₀, mas nenhuma solução tem x₀, e a regra o registra. Esta pedra escreve a regra da tensão mínima: entre as
literais limpas (de retorno útil e não condenadas pelo campo), a que deixa o ponto ressurreto com o menor custo; no empate, a
primeira.

## O que fica provado [REAL]

1. `custoDoRetorno`, `literalDoRetornoMinimo`, `lacoDaTensaoMinima`, `processoDaTensaoMinima` [DEF]: a regra e o processo.
2. ★ `o_minimo_existe_quando_ha_limpa`: havendo literal limpa, a escolha existe.
3. ★★ `o_eco_da_tensao_minima_e_fiel`: o laço só responde com ponto de custo zero.
4. ★★ `a_medida_a_primeira_limpa_falha_no_contraexemplo`: a regra da primeira limpa responde com silêncio em `phiCE`, que
   tem solução (por `decide`).
5. ★★ `a_medida_o_contraexemplo_ecoa_com_a_tensao_minima`: o processo da tensão mínima responde em `phiCE` com a solução
   (falso, verdadeiro, falso) (por `decide`).
6. As fórmulas pequenas das pedras anteriores, com a regra nova (por `decide`).

## O que NÃO se afirma

* Que a regra da tensão mínima responda com eco em toda fórmula que tem solução: a fórmula da armadilha do empate
  (`OMapaDosMuros.phiTrap`) tem solução e nela a regra responde com silêncio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ATensaoMinima

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso

/-- [DEF] O custo do retorno por uma literal: o custo do ponto ressurreto após registrá-la. A leitura do
sinal de retorno, antes da saída do espectro. -/
def custoDoRetorno {n : ℕ} (φ : CNF n) (S : Espectro2 n) (l : Literal n) : ℕ :=
  custoCNF φ (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))

/-- [DEF] A literal do retorno de tensão mínima: entre as limpas, a de menor custo do ponto ressurreto
(empate: a primeira). -/
def literalDoRetornoMinimo {n : ℕ} (φ : CNF n) (S : Espectro2 n) (C : Clausula n) : Option (Literal n) :=
  (C.filter (fun l => retornoUtil S l && !condenadaNoCampo φ S l)).foldl
    (fun acc l => match acc with
      | none => some l
      | some m => if custoDoRetorno φ S l < custoDoRetorno φ S m then some l else some m)
    none

/-- [DEF] O laço da tensão mínima: a mesma estrutura do laço da tensão, com a escolha pelo custo do
ponto de retorno. -/
def lacoDaTensaoMinima {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → (Option (Fin n → Bool)) × Espectro2 n
  | 0, S, x => (if custoCNF φ x = 0 then some x else none, S)
  | fuel + 1, S, x =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S)
    else
      match primeiraSemPar φ w with
      | none => (none, S)
      | some C =>
        match literalDoRetornoMinimo φ S C with
        | none => (none, S)
        | some l =>
          let S' := S ++ [(mascaraDeLiteral l, valorDeLiteral l)]
          lacoDaTensaoMinima φ fuel S' (ressurreicaoConstrutiva S')

/-- [DEF] O processo da tensão mínima: o campo fala o fecho primeiro; o laço mínimo corre a partir daí. -/
def processoDaTensaoMinima {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :
    (Option (Fin n → Bool)) × Espectro2 n :=
  let S₀ := fechoDoCampo φ (n + 2) []
  lacoDaTensaoMinima φ fuel S₀ (ressurreicaoConstrutiva S₀)

/-- ★ O mínimo existe quando há limpa: a escolha nunca trava enquanto há porta. (Auxiliar da
completude.) -/
theorem o_minimo_existe_quando_ha_limpa {n : ℕ} (φ : CNF n) (S : Espectro2 n) (C : Clausula n)
    (h : ∃ l ∈ C, retornoUtil S l = true ∧ condenadaNoCampo φ S l = false) :
    (literalDoRetornoMinimo φ S C).isSome = true := by
  unfold literalDoRetornoMinimo
  -- a filtrada é não-vazia
  obtain ⟨l, hlC, hret, hcond⟩ := h
  have hnl : (C.filter (fun l => retornoUtil S l && !condenadaNoCampo φ S l)) ≠ [] := by
    intro hvazio
    have hm : l ∈ C.filter (fun l => retornoUtil S l && !condenadaNoCampo φ S l) :=
      List.mem_filter.mpr ⟨hlC, by rw [Bool.and_eq_true]; exact ⟨hret, by rw [hcond]; rfl⟩⟩
    rw [hvazio] at hm
    exact absurd hm List.not_mem_nil
  -- o foldl com a função do mínimo sobre lista não-vazia dá some
  obtain ⟨l₀, resto, hl₀⟩ := List.exists_cons_of_ne_nil hnl
  rw [hl₀]
  suffices h : ∀ (resto : Clausula n) (acc : Option (Literal n)),
      acc.isSome = true →
      ((resto.foldl (fun acc l => match acc with
          | none => some l
          | some m => if custoDoRetorno φ S l < custoDoRetorno φ S m then some l else some m)
        acc).isSome = true) by
    rw [List.foldl_cons]
    exact h resto (some l₀) rfl
  intro resto
  induction resto with
  | nil => intro acc hacc; simpa using hacc
  | cons b bs ih =>
    intro acc hacc
    rw [List.foldl_cons]
    apply ih
    split
    · rfl
    · split
      · rfl
      · rfl

/-- ★★ **O ECO DA TENSÃO MÍNIMA É FIEL**: o laço novo só responde com ponto de custo zero. -/
theorem o_eco_da_tensao_minima_e_fiel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool) (S' : Espectro2 n),
      lacoDaTensaoMinima φ fuel S x = (some w, S') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w S' h
    unfold lacoDaTensaoMinima at h
    rw [Prod.mk.injEq] at h
    obtain ⟨h1, -⟩ := h
    split at h1
    · rename_i hcond
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · simp at h1
  | succ f ih =>
    intro S x w S' h
    unfold lacoDaTensaoMinima at h
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
        · simp at h
        · exact ih _ _ _ _ h

/-- [MEDIDA] A fórmula de três variáveis: (x₀∨x₁) ∧ (¬x₀∨x₁∨x₂) ∧ (¬x₀∨x₁∨¬x₂) ∧ (¬x₀∨¬x₁∨x₂) ∧
(¬x₀∨¬x₁∨¬x₂). A luz: (falso, verdadeiro, falso). -/
def phiCE : CNF 3 :=
  [[⟨0, true⟩, ⟨1, true⟩],
   [⟨0, false⟩, ⟨1, true⟩, ⟨2, true⟩],
   [⟨0, false⟩, ⟨1, true⟩, ⟨2, false⟩],
   [⟨0, false⟩, ⟨1, false⟩, ⟨2, true⟩],
   [⟨0, false⟩, ⟨1, false⟩, ⟨2, false⟩]]

/-- ★★ A regra da primeira limpa (o laço da tensão, `ATensaoFundamental`) responde com silêncio em `phiCE`, que tem
solução. -/
theorem a_medida_a_primeira_limpa_falha_no_contraexemplo :
    (processoDaTensao phiCE 5 (fun _ => false)).1 = none
    ∧ custoCNF phiCE (fun i => decide (i = 1)) = 0 := by
  decide

/-- ★★ O processo da tensão mínima responde em `phiCE` com a solução (falso, verdadeiro, falso). -/
theorem a_medida_o_contraexemplo_ecoa_com_a_tensao_minima :
    (processoDaTensaoMinima phiCE 5 (fun _ => false)).1 = some (fun i => decide (i = 1)) := by
  decide

/-- A sanidade com a regra nova: a fofa ecoa com a própria luz. -/
theorem a_medida_minima_fofa :
    (processoDaTensaoMinima phiFofa 5 (fun _ => false)).1 = some (fun _ => true) := by
  decide

/-- A sanidade: a fofa2 ecoa com a própria luz. -/
theorem a_medida_minima_fofa2 :
    (processoDaTensaoMinima phiFofa2 6 (fun _ => false)).1
      = some (fun i => decide (i = 0) || decide (i = 1)) := by
  decide

/-- A sanidade: a cadeia resolve; a insatisfazível cala; a phiMedida segue sã. -/
theorem a_medida_minima_sanidade :
    (processoDaTensaoMinima phiCadeia 5 (fun _ => false)).1.isSome = true
    ∧ (processoDaTensaoMinima ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 4 (fun _ => false)).1 = none
    ∧ ((processoDaTensaoMinima ([[⟨0, true⟩, ⟨1, true⟩], [⟨0, false⟩, ⟨1, false⟩]] : CNF 2) 5
        (fun _ => false)).1).isSome = true := by
  decide

end TGLExt.ATensaoMinima

