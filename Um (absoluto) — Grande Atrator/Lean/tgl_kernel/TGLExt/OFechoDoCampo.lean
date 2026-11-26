import TGLExt.OCampoFalaPrimeiro

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O fecho do campo: a propagação das literais forçadas

## O que é

Contra um espectro, uma literal pode estar proibida (a equação dela é inconsistente com o espectro), implicada (já se lê
dele) ou livre. Quando uma cláusula da fórmula tem todas as literais proibidas menos uma, e nenhuma implicada, a que resta
está forçada. O fecho do campo (`fechoDoCampo`) acrescenta essas equações ao espectro, uma a uma, até não haver mais; é a
propagação de cláusulas unitárias dos resolvedores conhecidos (Davis–Putnam–Logemann–Loveland, 1962) [KNOWN], escrita
sobre o espectro de paridades. O processo com o fecho (`processoComFecho`) faz o fecho primeiro e depois corre o laço com
retorno (`OSinalDeRetorno.lacoComRetorno`).

## O que fica provado [REAL]

1. `reducaoLiteral`, `literalProibida`, `literalImplicada`, `passoDePropagacao`, `passoDePropagacaoEm` [DEF].
2. ★ `proibida_falha_em_quem_serve`: uma literal proibida falha em todo ponto que serve o espectro.
3. ★ `passoDePropagacao_forcado`: a literal forçada de uma cláusula da fórmula é satisfeita por toda solução que serve o
   espectro.
4. ★★ `o_fecho_e_gratuito`: toda solução que serve o espectro serve também o fecho do campo.
5. As medidas, por `decide`: na cadeia `phiCadeia`, cuja única solução é o próprio ponto de partida (todas as variáveis
   falsas), o processo responde com ele; em `phiFofa` responde com a solução; na fórmula «x₀» e «não x₀» cala; em
   `phiFofa2`, que tem solução, cala.

## O que NÃO se afirma

* Que o processo responda com eco quando a fórmula tem solução: `phiFofa2` mede o contrário.
* O valor da morte com o eixo fora do span segue [OPEN].
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OFechoDoCampo

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro

/-- [DEF] A redução da equação de uma literal contra o escalonado do espectro. -/
def reducaoLiteral {n : ℕ} (S : Espectro2 n) (l : Literal n) : (Fin n → Bool) × ZMod 2 :=
  reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)

/-- [DEF] A literal está proibida pelo espectro: a sua equação é inconsistente com ele (reduz a 0 = 1). -/
def literalProibida {n : ℕ} (S : Espectro2 n) (l : Literal n) : Bool :=
  let r := reducaoLiteral S l
  (bitPivo r.1).isNone && decide (r.2 = 1)

/-- [DEF] A literal está implicada pelo espectro: a sua equação já é lida dele (reduz a 0 = 0). -/
def literalImplicada {n : ℕ} (S : Espectro2 n) (l : Literal n) : Bool :=
  let r := reducaoLiteral S l
  (bitPivo r.1).isNone && decide (r.2 = 0)

/-- [DEF] O passo de propagação numa cláusula: se alguma literal já está implicada, a cláusula já está
servida (nada a fazer); senão, se todas menos uma estão proibidas, a que resta está forçada. -/
def passoDePropagacao {n : ℕ} (S : Espectro2 n) (C : Clausula n) : Option (Literal n) :=
  if C.any (fun l => literalImplicada S l) then none
  else
    match C.filter (fun l => !literalProibida S l) with
    | [l] => some l
    | _ => none

/-- [DEF] O passo de propagação no campo: a primeira cláusula que força uma literal. -/
def passoDePropagacaoEm {n : ℕ} (S : Espectro2 n) : List (Clausula n) → Option (Literal n)
  | [] => none
  | C :: Cs => match passoDePropagacao S C with
    | some l => some l
    | none => passoDePropagacaoEm S Cs

/-- [DEF] O fecho do campo: a propagação iterada até estacionar (o fuel é a cota dos pivôs). -/
def fechoDoCampo {n : ℕ} (φ : CNF n) : ℕ → Espectro2 n → Espectro2 n
  | 0, S => S
  | fuel + 1, S =>
    match passoDePropagacaoEm S φ with
    | none => S
    | some l => fechoDoCampo φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])

/-- O passo no campo vem de uma cláusula do campo. (Auxiliar da gratuidade.) -/
theorem passoNoCampo_mem {n : ℕ} (S : Espectro2 n) (Cs : List (Clausula n)) (l : Literal n)
    (h : passoDePropagacaoEm S Cs = some l) :
    ∃ C ∈ Cs, passoDePropagacao S C = some l := by
  induction Cs with
  | nil => simp [passoDePropagacaoEm] at h
  | cons C Cs ih =>
    unfold passoDePropagacaoEm at h
    cases hp : passoDePropagacao S C with
    | some l' =>
      rw [hp] at h
      have hll : l' = l := Option.some.inj h
      rw [hll] at hp
      exact ⟨C, List.mem_cons_self, hp⟩
    | none =>
      rw [hp] at h
      obtain ⟨C', hC', hP⟩ := ih h
      exact ⟨C', List.mem_cons_of_mem C hC', hP⟩

/-- ★ **A PROIBIDA FALHA EM QUEM SERVE**: a literal inconsistente com o espectro falha em todo ponto que
o serve. (Auxiliar da força do passo.) -/
theorem proibida_falha_em_quem_serve {n : ℕ} (S : Espectro2 n) (y : Fin n → Bool) (l : Literal n)
    (hys : serveEspectro2 y S = true) (hpro : literalProibida S l = true) :
    l.vale y = false := by
  unfold literalProibida reducaoLiteral at hpro
  rw [Bool.and_eq_true] at hpro
  obtain ⟨hpiv, hval⟩ := hpro
  have hnone : bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1 = none := by
    cases hb : bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1 with
    | none => rfl
    | some i => rw [hb] at hpiv; simp at hpiv
  have hval1 : (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 = 1 :=
    of_decide_eq_true hval
  have hzero := mascara_zero_de_bitPivo_none _ hnone
  by_contra hnvale
  have hvale : l.vale y = true := by
    cases hv : l.vale y with
    | false => exact absurd hv hnvale
    | true => rfl
  have hyserve : serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
    show paridade2 y (mascaraDeLiteral l) = valorDeLiteral l
    rw [paridade2_mascaraLiteral]
    unfold valorDeLiteral
    unfold Literal.vale at hvale
    rw [beq_iff_eq] at hvale
    rw [hvale]
  have hy' : serveEspectro2 y (oEscalonado S) = true := (oEscalonado_preserva S y ⟨y, hys⟩).mpr hys
  have hpy := reduzirNova_preserva y (oEscalonado S) hy' (mascaraDeLiteral l, valorDeLiteral l)
  have hyred : serveEq y (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)) :=
    hpy.mpr hyserve
  have h10 : (0 : ZMod 2) = 1 := by
    have h1 : paridade2 y (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1
        = (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 := hyred
    rw [hzero, paridade2_zero_eixo, hval1] at h1
    exact h1
  exact (by decide : (0 : ZMod 2) ≠ 1) h10

/-- ★ **O PASSO DE PROPAGAÇÃO É FORÇADO**: se a cláusula do campo tem todas as literais proibidas menos
uma — e nenhuma implicada —, a que resta é servida por toda luz que serve o espectro. A propagação não
escolhe: lê a força do campo. -/
theorem passoDePropagacao_forcado {n : ℕ} (φ : CNF n) (S : Espectro2 n) (C : Clausula n) (l : Literal n)
    (y : Fin n → Bool) (hCφ : C ∈ φ) (hy : custoCNF φ y = 0)
    (hys : serveEspectro2 y S = true)
    (h : passoDePropagacao S C = some l) :
    serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
  unfold passoDePropagacao at h
  split at h
  · simp at h
  · split at h
    · rename_i l' hfiltro
      have hll : l' = l := Option.some.inj h
      rw [hll] at hfiltro
      -- da forma do filtro: l ∈ C, l livre; e toda outra literal de C é proibida
      have hlC : l ∈ C := by
        have hm : l ∈ C.filter (fun l => !literalProibida S l) := by
          rw [hfiltro]; exact List.mem_cons_self
        exact (List.mem_filter.mp hm).1
      have hout : ∀ l₂ ∈ C, l₂ ≠ l → literalProibida S l₂ = true := by
        intro l₂ hl₂C hne
        by_contra hpro
        have hpfalse : literalProibida S l₂ = false := Bool.eq_false_iff.mpr hpro
        have hpl : (!literalProibida S l₂) = true := by rw [hpfalse]; rfl
        have hm : l₂ ∈ C.filter (fun l => !literalProibida S l) :=
          List.mem_filter.mpr ⟨hl₂C, hpl⟩
        rw [hfiltro] at hm
        simp at hm
        exact hne hm
      -- a luz satisfaz a cláusula: alguma literal vale nela; as outras são proibidas — resta l
      have hCy : satisfeita C y = true := (custoCNF_zero_sse φ y).mp hy C hCφ
      unfold satisfeita at hCy
      rw [List.any_eq_true] at hCy
      obtain ⟨l₂, hl₂C, hl₂y⟩ := hCy
      by_cases hll2 : l₂ = l
      · rw [hll2] at hl₂y
        show paridade2 y (mascaraDeLiteral l) = valorDeLiteral l
        rw [paridade2_mascaraLiteral]
        unfold valorDeLiteral
        unfold Literal.vale at hl₂y
        rw [beq_iff_eq] at hl₂y
        rw [hl₂y]
      · have hpro := hout l₂ hl₂C hll2
        have hfalha := proibida_falha_em_quem_serve S y l₂ hys hpro
        rw [hfalha] at hl₂y
        exact Bool.noConfusion hl₂y
    · simp at h

/-- ★★ **O FECHO DO CAMPO É GRATUITO**: a propagação iterada das forças é servida por toda luz — o campo
inteiro falando não perde luz nenhuma, sem gate e sem morte. -/
theorem o_fecho_e_gratuito {n : ℕ} (φ : CNF n) (y : Fin n → Bool)
    (hy : custoCNF φ y = 0) :
    ∀ (fuel : ℕ) (S : Espectro2 n), serveEspectro2 y S = true →
      serveEspectro2 y (fechoDoCampo φ fuel S) = true := by
  intro fuel
  induction fuel with
  | zero => intro S h; exact h
  | succ f ih =>
    intro S h
    unfold fechoDoCampo
    split
    · exact h
    · rename_i l hpasso
      obtain ⟨C, hCφ, hC⟩ := passoNoCampo_mem S φ l hpasso
      have hserve := passoDePropagacao_forcado φ S C l y hCφ hy h hC
      apply ih
      rw [serveEspectro2_append]
      have h1 : serveEspectro2 y [(mascaraDeLiteral l, valorDeLiteral l)] = true := by
        unfold serveEspectro2
        rw [List.all_cons, List.all_nil]
        show ((paridade2 y (mascaraDeLiteral l) == valorDeLiteral l) && true) = true
        rw [beq_iff_eq.mpr hserve]
        rfl
      rw [h1, h]
      rfl

/-- [DEF] O processo inteiro: o campo fala o seu fecho primeiro; o laço com retorno corre a partir daí. -/
def processoComFecho {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :
    (Option (Fin n → Bool)) × Espectro2 n :=
  let S₀ := fechoDoCampo φ (φ.length + 1) []
  lacoComRetorno φ fuel S₀ (ressurreicaoConstrutiva S₀)

/-- [MEDIDA] A cadeia de forças: «não x₂», «x₂ ou não x₁», «x₁ ou não x₀»; a única solução tem as três variáveis falsas. -/
def phiCadeia : CNF 3 := [[⟨2, false⟩], [⟨2, true⟩, ⟨1, false⟩], [⟨1, true⟩, ⟨0, false⟩]]

/-- A medida: na cadeia de forças, partindo do ponto com as três variáveis falsas, que já é a solução, o processo
responde com ele. -/
theorem a_medida_a_cadeia_resolve_sem_morte :
    (processoComFecho phiCadeia 4 (fun _ => false)).1 = some (fun _ => false) := by
  decide

/-- A medida: a fofa segue ecoando com a própria luz, agora com o fecho antes. -/
theorem a_medida_o_fecho_deixa_a_fofa_ecoar :
    (processoComFecho phiFofa 5 (fun _ => false)).1 = some (fun _ => true) := by
  decide

/-- A medida da honestidade: a insatisfazível cala mesmo com o fecho do campo. -/
theorem a_medida_o_fecho_cala_na_insatisfazível :
    (processoComFecho ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 3 (fun _ => false)).1 = none := by
  decide

/-- ★★ A medida: com o fecho do campo primeiro, o processo ainda cala em `phiFofa2`, que tem solução. -/
theorem a_medida_o_gate_final_segue_sendo_o_valor_da_morte_transversal :
    (processoComFecho phiFofa2 6 (fun _ => false)).1 = none := by
  decide

end TGLExt.OFechoDoCampo

