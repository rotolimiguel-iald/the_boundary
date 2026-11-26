import TGLExt.ACompletude
import TGLExt.AMorteUtil

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O laço final sob a hipótese da morte útil

## O que é

A morte útil (`AMorteUtil.morteUtil`) pede que o lado guardado guarde alguma solução que serve o espectro. Esta pedra
toma como hipótese nomeada `hgates` que, em todo espectro e em todo ponto que não satisfaz e cujo eixo de morte está fora
do span, a morte com a paridade oposta à do ponto seja útil, e tira dela duas propriedades do laço final (`ACompletude`).

## O que fica provado [REAL]

1. ★ `o_laco_preserva_a_luz_util`: sob `hgates`, se alguma solução serve o espectro inicial, alguma solução serve o
   espectro final do laço.
2. ★ `o_fecho_do_arco`: sob `hgates` e havendo solução que serve o espectro inicial, quando o laço responde com um ponto,
   esse ponto tem custo zero; e alguma solução serve o espectro final.

## O que NÃO se afirma

* Que o laço responda com eco quando há solução: o enunciado não o diz.
* A hipótese `hgates` desta pedra é falsa para toda fórmula que tenha um ponto de custo não nulo cujo eixo de morte não é
  o eixo apagado (`AMedidaDaHipoteseDoLaco.a_hipotese_util_do_laco_e_insatisfazivel`); para essas fórmulas os dois
  teoremas valem no vazio.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OFechoDoArco

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AMorteUtil

variable {n : ℕ}

/-- Sob a hipótese `hgates`, se alguma solução serve o espectro inicial, alguma solução serve todo espectro que o
laço alcança. -/
theorem o_laco_preserva_a_luz_util {n : ℕ} (φ : CNF n)
    (hgates : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      custoCNF φ w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe φ w) = true) →
      morteUtil φ S (eixoDaMorteDe φ w) (paridade2 w (eixoDaMorteDe φ w) + 1)) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) →
      ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧
        serveEspectro2 y (lacoFinal φ fuel S x).2 = true := by
  intro fuel
  induction fuel with
  | zero => intro S x h; exact h
  | succ f ih =>
    intro S x h
    unfold lacoFinal
    dsimp only []
    split
    · rename_i hcond
      exact h
    · split
      · exact h
      · rename_i hcn hspan
        -- a morte útil guarda alguma luz
        have hnova : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧
            serveEspectro2 y (S ++ [(eixoDaMorteDe φ (descerGenerico passoFiel φ S φ.length ([], x)).2,
                paridade2 (descerGenerico passoFiel φ S φ.length ([], x)).2
                  (eixoDaMorteDe φ (descerGenerico passoFiel φ S φ.length ([], x)).2) + 1)]) = true :=
          a_morte_util_guarda_alguma_luz φ S _ _ (hgates S _ hcn hspan)
        exact ih _ _ hnova

/-- Sob a hipótese `hgates` e havendo solução que serve o espectro inicial: quando o laço final responde com um ponto,
esse ponto tem custo zero; e alguma solução serve o espectro final. Não diz que o laço responda. -/
theorem o_fecho_do_arco {n : ℕ} (φ : CNF n)
    (hgates : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      custoCNF φ w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe φ w) = true) →
      morteUtil φ S (eixoDaMorteDe φ w) (paridade2 w (eixoDaMorteDe φ w) + 1))
    (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool)
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :
    (∀ w : Fin n → Bool, ∀ S' : Espectro2 n, lacoFinal φ fuel S x = (some w, S') → custoCNF φ w = 0)
      ∧ (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧
          serveEspectro2 y (lacoFinal φ fuel S x).2 = true) := by
  constructor
  · intro w S' hw
    exact o_eco_do_laco_final_e_fiel φ fuel S x w S' hw
  · exact o_laco_preserva_a_luz_util φ hgates fuel S x hluz

end TGLExt.OFechoDoArco

