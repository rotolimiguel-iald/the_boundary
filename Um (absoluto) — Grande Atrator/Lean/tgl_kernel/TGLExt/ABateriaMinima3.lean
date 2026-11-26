import TGLExt.ABateriaMinima2

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria sob a tensão mínima III: quatro variáveis, sementes 789, 886, 1177 e 1468

As quatro fórmulas de `ABateriaDaParedeN4b`, nenhuma com solução, sob a regra da tensão mínima: o processo responde com
silêncio nas quatro (por `decide`) [REAL].

## O que NÃO se afirma

* Que a regra da tensão mínima responda com eco em toda fórmula que tem solução: a bateria é uma amostra, e a
  fórmula da armadilha do empate (`OMapaDosMuros.phiTrap`) tem solução e nela a regra responde com silêncio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaMinima3

open TGLExt.ABateriaDaParedeN4 TGLExt.ATensaoMinima TGLExt.TraducaoTuring

/-- ★★ A última metade da região crítica sob a regra mínima. -/
theorem bm9 : (processoDaTensaoMinima (geraCNF4 789 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 789 17) := by decide

theorem bm10 : (processoDaTensaoMinima (geraCNF4 886 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 886 17) := by decide

theorem bm11 : (processoDaTensaoMinima (geraCNF4 1177 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 1177 17) := by decide

theorem bm12 : (processoDaTensaoMinima (geraCNF4 1468 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 1468 17) := by decide

/-- ★★ A bateria mínima III confere. -/
theorem a_bateria_minima_III_confere :
    ((processoDaTensaoMinima (geraCNF4 789 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 789 17))
    ∧ ((processoDaTensaoMinima (geraCNF4 886 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 886 17))
    ∧ ((processoDaTensaoMinima (geraCNF4 1177 17) 6 (fun _ => false)).1.isSome
        = temLuz4 (geraCNF4 1177 17))
    ∧ ((processoDaTensaoMinima (geraCNF4 1468 17) 6 (fun _ => false)).1.isSome
        = temLuz4 (geraCNF4 1468 17)) :=
  ⟨bm9, bm10, bm11, bm12⟩

end TGLExt.ABateriaMinima3

