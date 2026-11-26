import TGLExt.AMorteUtil
import TGLExt.ACompletude
import TGLExt.AInvarianciaDaLeitura

set_option autoImplicit false

/-!
# As hipóteses nomeadas do laço final, medidas

## O que é

Os teoremas do laço final dizem algo sobre uma fórmula só quando a hipótese nomeada que tomam vale para ela. Este módulo
mede as duas formas dessa hipótese: a da morte sadia (`hgates` em `ACompletude.o_laco_preserva_a_luz`,
`AFechamentoFinal.o_fechamento` e `ACompletudeFinal.a_completude_final`) e a da morte útil (`hgates` em
`OFechoDoArco.o_laco_preserva_a_luz_util` e `OFechoDoArco.o_fecho_do_arco`).

## O que fica provado [REAL]

1. `phiUm_satisfazivel`: a fórmula `x₀ ∨ x₁` (`AMorteUtil.phiUm`) tem um ponto de custo zero.
2. ★ `a_hipotese_do_laco_falha_em_uma_clausula`: a hipótese da morte sadia é falsa para `x₀ ∨ x₁`. No ponto (F, F) a
   única cláusula está sem par; o eixo da morte é o das duas variáveis, fora do span da lista vazia; e a morte com a
   paridade oposta à do ponto não é sadia, porque (V, F) e (V, V) são soluções de paridades diferentes nesse eixo.
3. ★ `a_hipotese_util_do_laco_e_insatisfazivel`: a hipótese da morte útil é falsa para toda fórmula que tenha um ponto
   de custo não nulo cujo eixo de morte não é o eixo apagado. Ela quantifica sobre todo espectro, inclusive o de uma
   equação de eixo apagado e valor 1, que nenhum ponto serve; o eixo da morte não está no span desse espectro, e a morte
   útil pede um ponto que o sirva.
4. `a_hipotese_util_falha_em_uma_clausula`: em particular, para `x₀ ∨ x₁`.

## O que NÃO se afirma

* Nada aqui falsifica o laço final: os teoremas que tomam essas hipóteses continuam válidos; para as fórmulas acima eles
  valem no vazio.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMedidaDaHipoteseDoLaco

open TGLExt.TraducaoTuring TGLExt.AMorteEARessurreicao TGLExt.AVvem TGLExt.ACompletude TGLExt.AMorteUtil
open TGLExt.AparedeDoSpan TGLExt.AInvarianciaDaLeitura

/-- A fórmula `x₀ ∨ x₁` tem um ponto de custo zero. -/
theorem phiUm_satisfazivel : ∃ x : Fin 2 → Bool, custoCNF phiUm x = 0 :=
  ⟨fun _ => true, by decide⟩

/-- ★ A hipótese da morte sadia, com o enunciado de `ACompletude.o_laco_preserva_a_luz`, é falsa para a fórmula
`x₀ ∨ x₁`. -/
theorem a_hipotese_do_laco_falha_em_uma_clausula :
    ¬ (∀ (S : Espectro2 2) (w : Fin 2 → Bool),
        custoCNF phiUm w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe phiUm w) = true) →
        morteSadia phiUm S (eixoDaMorteDe phiUm w) (paridade2 w (eixoDaMorteDe phiUm w) + 1)) := by
  intro h
  have hw : custoCNF phiUm (fun _ => false) ≠ 0 := by decide
  have heixo : eixoDaMorteDe phiUm (fun _ => false) = eixoUm := by decide
  have hspan : ¬ (eixoNoSpanB (([] : Espectro2 2).map Prod.fst) (eixoDaMorteDe phiUm (fun _ => false)) = true) := by
    rw [heixo]
    decide
  have hs := h [] (fun _ => false) hw hspan
  have hpar : paridade2 (fun _ => false) eixoUm + 1 = 1 := by decide
  rw [heixo, hpar] at hs
  exact a_medida_uma_clausula_nenhum_lado_e_sadio.2 hs

/-- O xor de uma lista de eixos apagados é o eixo apagado. -/
theorem xorComb_de_apagados {n : ℕ} :
    ∀ sub : List (Fin n → Bool), (∀ x ∈ sub, x = fun _ => false) → xorComb sub = fun _ => false := by
  intro sub
  induction sub with
  | nil => intro _; rfl
  | cons x l ih =>
    intro hx
    rw [xorComb_cons, hx x (by simp), ih (fun y hy => hx y (by simp [hy])), xorPonto_zero_esq]

/-- ★ A hipótese da morte útil, com o enunciado de `OFechoDoArco.o_fecho_do_arco`, é falsa para toda fórmula que tenha
um ponto de custo não nulo cujo eixo de morte não é o eixo apagado. -/
theorem a_hipotese_util_do_laco_e_insatisfazivel {n : ℕ} (φ : CNF n) (w₀ : Fin n → Bool)
    (hw₀ : custoCNF φ w₀ ≠ 0) (ha : eixoDaMorteDe φ w₀ ≠ fun _ => false) :
    ¬ (∀ (S : Espectro2 n) (w : Fin n → Bool),
        custoCNF φ w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe φ w) = true) →
        morteUtil φ S (eixoDaMorteDe φ w) (paridade2 w (eixoDaMorteDe φ w) + 1)) := by
  intro h
  have hspan : ¬ (eixoNoSpanB (([((fun _ => false), (1 : ZMod 2))] : Espectro2 n).map Prod.fst)
      (eixoDaMorteDe φ w₀) = true) := by
    rw [eixoNoSpanB_iff]
    rintro ⟨sub, hsub, hx⟩
    apply ha
    rw [← hx]
    apply xorComb_de_apagados
    intro x hxs
    have hm := hsub hxs
    simpa using hm
  have hu := h _ w₀ hw₀ hspan
  unfold morteUtil at hu
  obtain ⟨y, _, hy, _⟩ := hu
  have hn : serveEspectro2 y [((fun _ => false), (1 : ZMod 2))] = false := by
    simp [serveEspectro2, paridade2]
  rw [hn] at hy
  exact Bool.false_ne_true hy

/-- Em particular, a hipótese da morte útil é falsa para `x₀ ∨ x₁`. -/
theorem a_hipotese_util_falha_em_uma_clausula :
    ¬ (∀ (S : Espectro2 2) (w : Fin 2 → Bool),
        custoCNF phiUm w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe phiUm w) = true) →
        morteUtil phiUm S (eixoDaMorteDe phiUm w) (paridade2 w (eixoDaMorteDe phiUm w) + 1)) :=
  a_hipotese_util_do_laco_e_insatisfazivel phiUm (fun _ => false) (by decide) (by decide)

end TGLExt.AMedidaDaHipoteseDoLaco
