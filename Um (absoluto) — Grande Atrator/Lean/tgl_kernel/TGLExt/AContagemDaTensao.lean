import TGLExt.OPostoDoEscalonado

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A contagem da tensão: o laço registra no máximo n equações

## O que é

A contagem das equações que o laço da tensão (`ATensaoFundamental.lacoDaTensao`) registra, pelo posto da base escalonada
(`OPostoDoEscalonado`).

## O que fica provado [REAL]

1. `encontra_satisfaz`: o elemento que `find?` devolve satisfaz o predicado.
2. `lacoDaTensao_passo`: um passo do laço ou deixa o espectro como está (eco ou silêncio), ou registra uma literal de
   retorno útil e segue do espectro acrescido.
3. `inserirEscalonado_length_ge`, `oEscalonado_mono_snoc`: a inserção nunca encolhe a base.
4. ★ `lacoDaTensao_rank_mono`: o posto não decresce ao longo do laço.
5. ★★ `a_tensao_conta_as_mortes`: as equações que o laço registra não passam do crescimento do posto.
6. ★★ `a_tensao_faz_no_maximo_n_mortes`: com qualquer combustível, o laço registra no máximo n equações além das do
   espectro de partida.

## O que NÃO se afirma

* Que o laço responda com eco em toda fórmula que tem solução: a completude em geral segue [OPEN].
* O custo de cada passo na fita, ou que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AContagemDaTensao

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado

variable {n : ℕ}

/-- O `find?` satisfaz o predicado. (Micro-lema da extração do guarda do laço.) -/
theorem encontra_satisfaz {α : Type} (p : α → Bool) :
    ∀ (l : List α) (a : α), l.find? p = some a → p a = true := by
  intro l
  induction l with
  | nil => intro a h; simp [List.find?] at h
  | cons b bs ih =>
    intro a h
    cases hb : p b with
    | true =>
      rw [List.find?_cons_of_pos hb] at h
      have hba : b = a := Option.some.inj h
      rw [← hba]
      exact hb
    | false =>
      rw [List.find?_cons_of_neg (by simp [hb])] at h
      exact ih a h

/-- ★ **O PASSO DO LAÇO, DE UMA VEZ**: ou o espectro não muda (eco ou silêncio), ou o laço registra uma literal de
retorno útil (`retornoUtil`) e segue do espectro acrescido. -/
theorem lacoDaTensao_passo {n : ℕ} (φ : CNF n) (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool) :
    (lacoDaTensao φ (fuel + 1) S x).2 = S ∨
      ∃ (l : Literal n), retornoUtil S l = true ∧
        (lacoDaTensao φ (fuel + 1) S x).2 =
          (lacoDaTensao φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
            (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))).2 := by
  rw [lacoDaTensao.eq_2]
  dsimp only []
  split
  · left; rfl
  · split
    · left; rfl
    · split
      · left; rfl
      · rename_i C l hl
        right
        refine ⟨l, ?_, ?_⟩
        · unfold literalDoRetornoLimpo at hl
          have h2 := encontra_satisfaz _ _ _ hl
          rw [Bool.and_eq_true] at h2
          exact h2.1
        · simp_all

/-- A inserção nunca encolhe a base. -/
theorem inserirEscalonado_length_ge {n : ℕ} (e : (Fin n → Bool) × ZMod 2) (ps : Espectro2 n) :
    (inserirEscalonado e ps).length ≥ ps.length := by
  unfold inserirEscalonado
  dsimp only []
  split
  · exact Nat.le_refl _
  · rw [List.length_cons, List.length_map]
    exact Nat.le_add_right _ _

/-- O posto do acrescentado não decresce. -/
theorem oEscalonado_mono_snoc {n : ℕ} (S : Espectro2 n) (e : (Fin n → Bool) × ZMod 2) :
    (oEscalonado (S ++ [e])).length ≥ (oEscalonado S).length := by
  rw [oEscalonado_snoc]
  exact inserirEscalonado_length_ge _ _

/-- ★ O posto não decresce ao longo do laço da tensão. -/
theorem lacoDaTensao_rank_mono {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      (oEscalonado (lacoDaTensao φ fuel S x).2).length ≥ (oEscalonado S).length := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x
    show (oEscalonado S).length ≥ (oEscalonado S).length
    exact Nat.le_refl _
  | succ f ih =>
    intro S x
    rcases lacoDaTensao_passo φ f S x with h | ⟨l, hret, hstep⟩
    · simp [h]
    · rw [hstep]
      have hcresce := oEscalonado_cresce_um_com_retorno S l hret
      have hmono := ih (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
        (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
      omega

/-- ★★ **A CONTAGEM DA TENSÃO**: as equações que o laço registra não passam do crescimento do posto (cada registro
cresce o posto de exatamente um). -/
theorem a_tensao_conta_as_mortes {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      ((lacoDaTensao φ fuel S x).2).length + (oEscalonado S).length
        ≤ S.length + (oEscalonado (lacoDaTensao φ fuel S x).2).length := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x
    show S.length + (oEscalonado S).length ≤ S.length + (oEscalonado S).length
    exact Nat.le_refl _
  | succ f ih =>
    intro S x
    rcases lacoDaTensao_passo φ f S x with h | ⟨l, hret, hstep⟩
    · simp [h]
    · rw [hstep]
      have hcresce := oEscalonado_cresce_um_com_retorno S l hret
      have hih := ih (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
        (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
      have hlen : (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length = S.length + 1 := by
        rw [List.length_append]
        simp
      omega

/-- ★★ Com qualquer combustível, o laço da tensão registra no máximo n equações além das do espectro de partida; n é o
número de variáveis. -/
theorem a_tensao_faz_no_maximo_n_mortes {n : ℕ} (φ : CNF n) (fuel : ℕ) (S : Espectro2 n)
    (x : Fin n → Bool) :
    ((lacoDaTensao φ fuel S x).2).length ≤ S.length + n := by
  have h1 := a_tensao_conta_as_mortes φ fuel S x
  have h2 := oEscalonado_length_le (lacoDaTensao φ fuel S x).2
  omega

end TGLExt.AContagemDaTensao

