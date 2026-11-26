import TGLExt.ABateriaDaParede

set_option autoImplicit false
set_option maxHeartbeats 1600000

/-!
# A bateria da parede II: as outras quatro sementes

## O que é

A continuação da pedra `ABateriaDaParede`, com a mesma medida, nas sementes 2003, 2665, 3658 e 4651 (dez cláusulas cada).

## O que fica provado [REAL]

1. ★★ `bat5` a `bat8`: para essas quatro sementes, o processo da tensão responde com eco exatamente quando a fórmula tem
   solução (por `decide`); uma tem solução (3658) e três não.
2. ★★ `a_bateria_II_confere`: as quatro medidas juntas.

## O que NÃO se afirma

* Que o processo da tensão responda com eco em toda fórmula de três variáveis que tem solução: a bateria é uma amostra, e a
  variante de três variáveis da pedra `AMedidaDaHipoteseDaMontagem` tem solução e nela o processo responde com silêncio.
* Que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaDaParede2

open TGLExt.ABateriaDaParede TGLExt.ATensaoFundamental TGLExt.TraducaoTuring

/-- ★★ A segunda metade da bateria. -/
theorem bat5 : (processoDaTensao (geraCNF3 2003 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 2003 10) := by decide

theorem bat6 : (processoDaTensao (geraCNF3 2665 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 2665 10) := by decide

theorem bat7 : (processoDaTensao (geraCNF3 3658 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 3658 10) := by decide

theorem bat8 : (processoDaTensao (geraCNF3 4651 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 4651 10) := by decide

/-- ★★ A bateria II confere. -/
theorem a_bateria_II_confere :
    ((processoDaTensao (geraCNF3 2003 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 2003 10))
    ∧ ((processoDaTensao (geraCNF3 2665 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 2665 10))
    ∧ ((processoDaTensao (geraCNF3 3658 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 3658 10))
    ∧ ((processoDaTensao (geraCNF3 4651 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 4651 10)) :=
  ⟨bat5, bat6, bat7, bat8⟩

end TGLExt.ABateriaDaParede2

