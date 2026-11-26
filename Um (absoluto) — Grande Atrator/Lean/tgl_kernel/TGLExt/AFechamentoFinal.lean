import TGLExt.ACompletude

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O laço final: o eco fiel e a luz guardada, num só enunciado

## O que fica provado [REAL]

1. `a_morte_faz_o_espectro_crescer`: acrescentar uma equação ao espectro aumenta o seu comprimento em um.
2. ★ `o_fechamento`: sob a hipótese nomeada `hgates` (a condição por morte em todo poço que não satisfaz e cujo eixo de
   morte está fora do span) e com o espectro inicial servido por toda solução, valem juntas as duas propriedades do laço
   final (`ACompletude`): quando o laço responde com um ponto, esse ponto tem custo zero; e o espectro final é servido por
   toda solução.

## O que NÃO se afirma

* Que o laço responda com eco quando há solução: o enunciado não diz que o laço responda; ele pode parar em silêncio ou
  esgotar o combustível. A completude do laço segue [OPEN].
* Nenhuma cota sobre o comprimento do espectro final está neste enunciado.
* A hipótese `hgates` não está provada.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AFechamentoFinal

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude

variable {n : ℕ}

/-- ★ A morte faz o espectro crescer: o espectro acrescentado tem uma equação a mais. (Auxiliar da
contagem colada.) -/
theorem a_morte_faz_o_espectro_crescer {n : ℕ} (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2) :
    (S ++ [(a, v)]).length = S.length + 1 := by
  simp

/-- Sob a hipótese `hgates` e com o espectro inicial servido por toda solução (`hguarda`): quando o laço final
responde com um ponto, esse ponto tem custo zero; e o espectro final é servido por toda solução. Não diz que o laço
responda. -/
theorem o_fechamento {n : ℕ} (φ : CNF n)
    (hgates : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      custoCNF φ w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe φ w) = true) →
      morteSadia φ S (eixoDaMorteDe φ w) (paridade2 w (eixoDaMorteDe φ w) + 1))
    (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool)
    (hguarda : ∀ y : Fin n → Bool, custoCNF φ y = 0 → serveEspectro2 y S = true) :
    (∀ w : Fin n → Bool, lacoFinal φ fuel S x = (some w, (lacoFinal φ fuel S x).2) →
      custoCNF φ w = 0)
      ∧ (∀ y : Fin n → Bool, custoCNF φ y = 0 →
          serveEspectro2 y (lacoFinal φ fuel S x).2 = true) := by
  constructor
  · intro w hw
    exact o_eco_do_laco_final_e_fiel φ fuel S x w _ hw
  · intro y hy
    exact o_laco_preserva_a_luz φ hgates fuel S x hguarda y hy

end TGLExt.AFechamentoFinal

