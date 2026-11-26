import TGLExt.ABateriaDaParedeN4

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria da parede IV: as outras quatro sementes de quatro variáveis

## O que é

A continuação da pedra `ABateriaDaParedeN4`, com a mesma medida, nas sementes 789, 886, 1177 e 1468 (dezessete cláusulas cada);
nenhuma das quatro fórmulas tem solução.

## O que fica provado [REAL]

1. ★★ `n4_5` a `n4_8`: para essas quatro sementes, o processo da tensão responde com eco exatamente quando a fórmula tem solução
   (por `decide`); como nenhuma tem solução, nas quatro ele responde com silêncio.
2. ★★ `a_bateria_IV_confere`: as quatro medidas juntas.

## O que NÃO se afirma

* Que o processo da tensão responda com eco em toda fórmula de quatro variáveis que tem solução: a bateria é uma amostra, e a
  fórmula de quatro variáveis da pedra `AMedidaDaHipoteseDaMontagem` tem solução e nela o processo responde com silêncio.
* Que estas fórmulas estejam no limiar de satisfação do 3-SAT aleatório: o gerador mistura cláusulas unitárias, pares e triplas.
* Que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaDaParedeN4b

open TGLExt.ABateriaDaParedeN4 TGLExt.ATensaoFundamental TGLExt.TraducaoTuring

/-- ★★ A segunda metade da região crítica. -/
theorem n4_5 : (processoDaTensao (geraCNF4 789 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 789 17) := by decide

theorem n4_6 : (processoDaTensao (geraCNF4 886 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 886 17) := by decide

theorem n4_7 : (processoDaTensao (geraCNF4 1177 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 1177 17) := by decide

theorem n4_8 : (processoDaTensao (geraCNF4 1468 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 1468 17) := by decide

/-- ★★ A bateria IV confere. -/
theorem a_bateria_IV_confere :
    ((processoDaTensao (geraCNF4 789 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 789 17))
    ∧ ((processoDaTensao (geraCNF4 886 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 886 17))
    ∧ ((processoDaTensao (geraCNF4 1177 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 1177 17))
    ∧ ((processoDaTensao (geraCNF4 1468 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 1468 17)) :=
  ⟨n4_5, n4_6, n4_7, n4_8⟩

end TGLExt.ABateriaDaParedeN4b

