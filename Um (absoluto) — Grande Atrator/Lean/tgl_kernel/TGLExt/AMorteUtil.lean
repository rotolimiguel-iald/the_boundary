import TGLExt.OProcessoFinal

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A morte útil: o lado guardado guarda alguma solução

## O que é

A condição por morte da pedra `AVvem` (`morteSadia`) pede que o lado guardado guarde toda solução que serve o espectro.
A morte útil (`morteUtil`) pede menos: que o lado guardado guarde alguma. Esta pedra define a morte útil, liga as duas e
mede ambas em duas fórmulas de duas variáveis.

## O que fica provado [REAL]

1. `morteUtil` [DEF]: o lado guardado (a paridade `v` no eixo `a`) guarda alguma solução que serve o espectro.
2. `a_morte_sadia_e_util`: havendo solução que serve o espectro, a morte sadia é útil.
   `a_morte_util_guarda_alguma_luz`: com a morte útil, alguma solução serve o espectro acrescido da equação lida.
3. ★ As medidas, por `decide`: na fórmula «x₀ ou x₁» e «não x₀ ou não x₁» (`phiMedida`), no eixo das duas variáveis, a
   paridade 0 (a do canto) é inútil e a paridade 1 é útil; na fórmula «x₀ ou x₁» (`phiUm`), as duas paridades são úteis e
   nenhuma é sadia (`a_medida_uma_clausula_nenhum_lado_e_sadio`).

## O que NÃO se afirma

* Que a morte útil em cada morte baste para que o eco venha: nenhum teorema aqui o diz; segue [OPEN].
* Uma regra que escolha o lado útil para toda fórmula não é dada.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMorteUtil

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal

variable {n : ℕ}

/-- [DEF] **A morte útil**: o lado guardado (a paridade `v` no eixo `a`) guarda ALGUMA solução da fórmula que
serve o espectro. Mais fraca que a sadia. -/
def morteUtil {n : ℕ} (φ : CNF n) (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2) : Prop :=
  ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true ∧ paridade2 y a = v

/-- A sadia implica a útil, quando há luz servindo o espectro. (Auxiliar da ponte.) -/
theorem a_morte_sadia_e_util {n : ℕ} (φ : CNF n) (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2)
    (hsadia : morteSadia φ S a v)
    (hcons : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :
    morteUtil φ S a v := by
  obtain ⟨y, hy, hyserv⟩ := hcons
  exact ⟨y, hy, hyserv, hsadia y hy hyserv⟩

/-- A morte útil guarda a luz (a forma fraca da preservação). (Auxiliar do laço.) -/
theorem a_morte_util_guarda_alguma_luz {n : ℕ} (φ : CNF n) (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2)
    (hutil : morteUtil φ S a v) :
    ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y (S ++ [(a, v)]) = true := by
  obtain ⟨y, hy, hyserv, hpar⟩ := hutil
  refine ⟨y, hy, ?_⟩
  rw [serveEspectro2_append]
  have h1 : serveEspectro2 y [(a, v)] = true := by
    unfold serveEspectro2
    rw [List.all_cons, List.all_nil]
    show ((paridade2 y a == v) && true) = true
    rw [hpar]
    simp
  rw [h1]
  simp [hyserv]

/-! ## As medidas -/

/-- A fórmula de uma cláusula: `x₀ ∨ x₁`. -/
def phiUm : CNF 2 := [[⟨0, true⟩, ⟨1, true⟩]]

/-- O eixo das duas variáveis. -/
def eixoUm : Fin 2 → Bool := fun _ => true

/-- ★ Medida 1: na fórmula da medida (duas cláusulas), o lado do poço é INÚTIL. -/
theorem a_medida_o_lado_do_poco_e_inutil :
    ¬ morteUtil phiMedida [] eixoMedida 0 := by
  unfold morteUtil
  decide

/-- ★ Medida 2: na fórmula da medida, o lado oposto é ÚTIL. -/
theorem a_medida_o_lado_oposto_e_util :
    morteUtil phiMedida [] eixoMedida 1 := by
  unfold morteUtil
  decide

/-- ★ Medida 3: na fórmula de uma cláusula, o lado do canto (paridade 0) é ÚTIL (o `(V,V)` serve). -/
theorem a_medida_uma_clausula_o_lado_do_canto_e_util :
    morteUtil phiUm [] eixoUm 0 := by
  unfold morteUtil
  decide

/-- ★ Medida 4: na fórmula de uma cláusula, o lado oposto também é ÚTIL. -/
theorem a_medida_uma_clausula_o_lado_oposto_e_util :
    morteUtil phiUm [] eixoUm 1 := by
  unfold morteUtil
  decide

/-- ★ Medida 5: na fórmula de uma cláusula, NENHUM dos lados é sadio (a `(V,V)` e o canto têm paridades
opostas) — a sadiê é mais forte que o processo pede. -/
theorem a_medida_uma_clausula_nenhum_lado_e_sadio :
    ¬ morteSadia phiUm [] eixoUm 0 ∧ ¬ morteSadia phiUm [] eixoUm 1 := by
  unfold morteSadia
  decide

end TGLExt.AMorteUtil

