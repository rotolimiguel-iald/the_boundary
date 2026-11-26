import TGLExt.OSolverGF2_Parte2

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O solver GF(2): a forma isolada e o ponto servido

## O que é

Um espectro é uma lista de equações de paridade sobre GF(2) (máscara, valor). Na forma ISOLADA, toda equação tem
pivô; o mesmo bit de pivô é a mesma equação; e o bit de pivô de uma equação não entra na máscara de outra. O ponto
construído (`pontoConstruido`) lê cada bit direto do espectro: no bit de pivô, o valor da primeira equação da lista
com esse pivô; nos demais bits, falso. Na forma isolada essa equação é a única com aquele pivô.

Este módulo é uma peça do resolvedor GF(2) construtivo que o transporte T pede (hipótese nomeada [OPEN] em
`ARessurreicao`); o transporte T não se prova aqui.

## O que fica provado [REAL]

1. `formaIsolada` [DEF]: a forma isolada, como definição; `formaIsolada_nil`: a lista vazia está na forma isolada.
2. `pontoConstruido_no_bit_de_nao_pivo`: num bit que não é pivô de nenhuma equação, o ponto construído é falso.
   `find?_de_unico`: se `e` está na lista, satisfaz o predicado e todo elemento da lista que o satisfaz é igual a
   `e`, então `find?` devolve `e` (único a menos de repetições de `e` na lista).
   `pontoConstruido_no_bit_de_pivo`: se uma equação é a única com aquele bit de pivô, o ponto construído nesse bit é
   o valor dela.
3. ★★ `pontoConstruido_serve`: se o espectro está na forma isolada, o ponto construído serve cada equação dele (a
   paridade do ponto na máscara é o valor da equação).

## O que NÃO se afirma

* Não se prova aqui que a inserção de uma equação preserva a forma isolada, nem a montagem do solver completo. Essas
  duas etapas ficam nomeadas como alvos abertos [OPEN]; não entram como axioma nem como sorry.
* Resolver um sistema consistente de equações de paridade sobre GF(2) por eliminação gaussiana é polinomial
  [KNOWN]; isso não é decidir SAT. Nada aqui diz que algum procedimento decida SAT.
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OSolverGF2_Parte3

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2

variable {n : ℕ}

/-- [DEF] A forma isolada: toda equação tem pivô; o mesmo bit de pivô é a mesma equação; e o bit de pivô
de uma não entra na máscara de outra. -/
def formaIsolada {n : ℕ} (pivots : Espectro2 n) : Prop :=
  (∀ p ∈ pivots, ∃ i : Fin n, bitPivo p.1 = some i)
    ∧ (∀ p₁ ∈ pivots, ∀ p₂ ∈ pivots, ∀ i : Fin n, bitPivo p₁.1 = some i → bitPivo p₂.1 = some i → p₁ = p₂)
    ∧ (∀ p₁ ∈ pivots, ∀ p₂ ∈ pivots, ∀ i j : Fin n,
        bitPivo p₁.1 = some i → bitPivo p₂.1 = some j → i ≠ j → p₁.1 j = false)

/-- A lista vazia está na forma isolada. -/
theorem formaIsolada_nil : formaIsolada ([] : Espectro2 n) := by
  refine ⟨?_, ?_, ?_⟩
  · intro p hp
    exact absurd hp (List.not_mem_nil)
  · intro p₁ hp₁
    exact absurd hp₁ (List.not_mem_nil)
  · intro p₁ hp₁
    exact absurd hp₁ (List.not_mem_nil)

/-- Num bit que não é pivô de nenhuma equação, o ponto construído é falso. -/
theorem pontoConstruido_no_bit_de_nao_pivo {n : ℕ} (pivots : Espectro2 n) (j : Fin n)
    (h : ∀ p ∈ pivots, bitPivo p.1 ≠ some j) :
    pontoConstruido pivots j = false := by
  unfold pontoConstruido
  have hnone : pivots.find? (fun p => decide (bitPivo p.1 = some j)) = none := by
    rw [List.find?_eq_none]
    intro p hp
    simp [h p hp]
  rw [hnone]

/-- Se `e` está na lista, satisfaz o predicado e todo elemento da lista que o satisfaz é igual a `e`, então
`find?` devolve `e` (único a menos de repetições de `e` na lista). -/
theorem find?_de_unico {α : Type} (l : List α) (p : α → Bool) (e : α)
    (he : e ∈ l) (hp : p e = true) (huni : ∀ a ∈ l, p a = true → a = e) :
    l.find? p = some e := by
  induction l with
  | nil => exact absurd he (List.not_mem_nil)
  | cons a as ih =>
    rw [List.find?_cons]
    by_cases ha : p a = true
    · rw [ha]
      exact congrArg some (huni a (List.mem_cons_self) ha)
    · have haf : p a = false := by simp [ha]
      rw [haf]
      apply ih
      · cases List.mem_cons.mp he with
        | inl h1 =>
          subst h1
          exact absurd hp ha
        | inr h2 => exact h2
      · intro b hb
        exact huni b (List.mem_cons_of_mem _ hb)

/-- Se a equação `e` é a única com o bit de pivô `j`, o ponto construído em `j` é o valor dela. -/
theorem pontoConstruido_no_bit_de_pivo {n : ℕ} (pivots : Espectro2 n) (e : (Fin n → Bool) × ZMod 2)
    (j : Fin n) (he : e ∈ pivots) (hb : bitPivo e.1 = some j)
    (hdist : ∀ p ∈ pivots, bitPivo p.1 = some j → p = e) :
    pontoConstruido pivots j = decide (e.2 = 1) := by
  unfold pontoConstruido
  have hfind : pivots.find? (fun p => decide (bitPivo p.1 = some j)) = some e := by
    apply find?_de_unico pivots _ e he (by simp [hb])
    intro p hp hpiv
    exact hdist p hp (of_decide_eq_true hpiv)
  rw [hfind]

/-- ★★ **O PONTO CONSTRUÍDO SERVE A FORMA ISOLADA**: se o espectro está na forma isolada, o ponto construído
serve cada equação dele; a paridade do ponto na máscara é o valor da equação. -/
theorem pontoConstruido_serve {n : ℕ} (pivots : Espectro2 n) (hiso : formaIsolada pivots) :
    ∀ e ∈ pivots, serveEq (pontoConstruido pivots) e := by
  intro e he
  obtain ⟨i, hb⟩ := hiso.1 e he
  show paridade2 (pontoConstruido pivots) e.1 = e.2
  unfold paridade2
  have houtros : ∀ j ∈ Finset.univ, j ≠ i →
      (if e.1 j && pontoConstruido pivots j then (1 : ZMod 2) else 0) = 0 := by
    intro j _ hji
    by_cases hbit : e.1 j = true
    · have hnao : ∀ p ∈ pivots, bitPivo p.1 ≠ some j := by
        intro p hp hpiv
        have hzero : e.1 j = false := hiso.2.2 e he p hp i j hb hpiv (Ne.symm hji)
        rw [hzero] at hbit
        simp at hbit
      rw [pontoConstruido_no_bit_de_nao_pivo pivots j hnao]
      simp
    · simp [hbit]
  rw [Finset.sum_eq_single i houtros (fun h => absurd (Finset.mem_univ i) h)]
  have hxi : pontoConstruido pivots i = decide (e.2 = 1) :=
    pontoConstruido_no_bit_de_pivo pivots e i he hb (fun p hp hpiv => hiso.2.1 p hp e he i hpiv hb)
  have hbit : e.1 i = true := bitPivo_spec e.1 i hb
  rw [hbit, hxi]
  simp only [Bool.true_and]
  cases zmod2_exaustivo e.2 with
  | inl h0 => rw [h0]; simp
  | inr h1 => rw [h1]; simp

end TGLExt.OSolverGF2_Parte3

