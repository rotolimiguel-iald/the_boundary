import TGLExt.ACompletude

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O ponto único que guarda a luz; o laço final, de novo

## O que fica provado [REAL]

1. ★ `o_poco_final_e_a_luz`: se todo ponto de custo zero serve o espectro (`hguarda`), algum ponto tem custo zero (`hcons`)
   e todo ponto que serve o espectro é `x` (`hunico`), então `x` tem custo zero.
2. `a_completude_final`: sob a hipótese nomeada `hgates` e com o espectro inicial servido por toda solução, quando o laço
   final responde com um ponto, esse ponto tem custo zero; e o espectro final é servido por toda solução. São as mesmas
   duas propriedades de `AFechamentoFinal.o_fechamento`.

## O que NÃO se afirma

* Que o laço chegue a um espectro servido por um único ponto, nem que responda com eco quando há solução: a completude do
  laço segue [OPEN].
* A hipótese `hgates` não está provada.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ACompletudeFinal

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude

variable {n : ℕ}

/-- ★ **O POÇO FINAL É A LUZ**: quando o resíduo guarda toda solução e segue habitado e é único, o ponto é
solução. (A forma do singleton, a ponte com a completude condicional.) -/
theorem o_poco_final_e_a_luz {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (hguarda : ∀ y : Fin n → Bool, custoCNF φ y = 0 → serveEspectro2 y S = true)
    (hcons : ∃ y : Fin n → Bool, custoCNF φ y = 0)
    (hunico : ∀ y : Fin n → Bool, serveEspectro2 y S = true → y = x) :
    custoCNF φ x = 0 := by
  obtain ⟨y, hy⟩ := hcons
  have hyserv : serveEspectro2 y S = true := hguarda y hy
  have hyx : y = x := hunico y hyserv
  subst hyx
  exact hy

/-- Sob a hipótese `hgates` e com o espectro inicial servido por toda solução (`hguarda`): quando o laço final
responde com um ponto, esse ponto tem custo zero; e o espectro final é servido por toda solução (as mesmas duas
propriedades de `o_fechamento`). Não diz que o laço responda. -/
theorem a_completude_final {n : ℕ} (φ : CNF n)
    (hgates : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      custoCNF φ w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe φ w) = true) →
      morteSadia φ S (eixoDaMorteDe φ w) (paridade2 w (eixoDaMorteDe φ w) + 1))
    (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool)
    (hguarda : ∀ y : Fin n → Bool, custoCNF φ y = 0 → serveEspectro2 y S = true) :
    (∀ w : Fin n → Bool, ∀ S' : Espectro2 n, lacoFinal φ fuel S x = (some w, S') → custoCNF φ w = 0)
      ∧ (∀ y : Fin n → Bool, custoCNF φ y = 0 →
          serveEspectro2 y (lacoFinal φ fuel S x).2 = true) := by
  constructor
  · intro w S' hw
    exact o_eco_do_laco_final_e_fiel φ fuel S x w S' hw
  · intro y hy
    exact o_laco_preserva_a_luz φ hgates fuel S x hguarda y hy

end TGLExt.ACompletudeFinal

