import TGLExt.ABateriaMinimaN5

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria sob a tensão mínima V: cinco variáveis, sementes 207, 401, 498 e 595

Quatro fórmulas do gerador de `ABateriaMinimaN5`, nenhuma com solução: o processo da tensão mínima responde com silêncio nas
quatro (por `decide`) [REAL].

## O que NÃO se afirma

* Que a regra da tensão mínima responda com eco em toda fórmula que tem solução: a bateria é uma amostra, e a
  fórmula da armadilha do empate (`OMapaDosMuros.phiTrap`) tem solução e nela a regra responde com silêncio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaMinimaN5b

open TGLExt.ABateriaDaParedeN4 TGLExt.ABateriaMinimaN5 TGLExt.ATensaoMinima TGLExt.TraducaoTuring

/-- ★★ A região crítica em n = 5, segunda metade (sementes 207, 401, 498, 595: todas sem luz). -/
theorem n5_5 : (processoDaTensaoMinima (geraCNF5 207 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 207 21) := by decide

theorem n5_6 : (processoDaTensaoMinima (geraCNF5 401 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 401 21) := by decide

theorem n5_7 : (processoDaTensaoMinima (geraCNF5 498 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 498 21) := by decide

theorem n5_8 : (processoDaTensaoMinima (geraCNF5 595 21) 7 (fun _ => false)).1.isSome
    = temLuz5 (geraCNF5 595 21) := by decide

/-- ★★ As quatro medidas juntas. -/
theorem a_bateria_VI_confere :
    ((processoDaTensaoMinima (geraCNF5 207 21) 7 (fun _ => false)).1.isSome = temLuz5 (geraCNF5 207 21))
    ∧ ((processoDaTensaoMinima (geraCNF5 401 21) 7 (fun _ => false)).1.isSome = temLuz5 (geraCNF5 401 21))
    ∧ ((processoDaTensaoMinima (geraCNF5 498 21) 7 (fun _ => false)).1.isSome = temLuz5 (geraCNF5 498 21))
    ∧ ((processoDaTensaoMinima (geraCNF5 595 21) 7 (fun _ => false)).1.isSome
        = temLuz5 (geraCNF5 595 21)) :=
  ⟨n5_5, n5_6, n5_7, n5_8⟩

end TGLExt.ABateriaMinimaN5b

