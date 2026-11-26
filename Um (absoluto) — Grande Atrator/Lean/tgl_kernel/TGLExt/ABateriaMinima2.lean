import TGLExt.ABateriaDaParedeN4b
import TGLExt.ABateriaMinima

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria sob a tensão mínima II: quatro variáveis, sementes 13, 110, 207 e 304

As quatro fórmulas de quatro variáveis e dezessete cláusulas de `ABateriaDaParedeN4`, sob a regra da tensão mínima: o processo
responde com eco exatamente quando a fórmula tem solução (por `decide`) [REAL].

## O que NÃO se afirma

* Que a regra da tensão mínima responda com eco em toda fórmula que tem solução: a bateria é uma amostra, e a
  fórmula da armadilha do empate (`OMapaDosMuros.phiTrap`) tem solução e nela a regra responde com silêncio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaMinima2

open TGLExt.ABateriaDaParedeN4 TGLExt.ATensaoMinima TGLExt.TraducaoTuring

/-- ★★ A região crítica sob a regra mínima, primeira metade. -/
theorem bm5 : (processoDaTensaoMinima (geraCNF4 13 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 13 17) := by decide

theorem bm6 : (processoDaTensaoMinima (geraCNF4 110 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 110 17) := by decide

theorem bm7 : (processoDaTensaoMinima (geraCNF4 207 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 207 17) := by decide

theorem bm8 : (processoDaTensaoMinima (geraCNF4 304 17) 6 (fun _ => false)).1.isSome
    = temLuz4 (geraCNF4 304 17) := by decide

/-- ★★ A bateria mínima II confere (primeira metade). -/
theorem a_bateria_minima_II_confere :
    ((processoDaTensaoMinima (geraCNF4 13 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 13 17))
    ∧ ((processoDaTensaoMinima (geraCNF4 110 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 110 17))
    ∧ ((processoDaTensaoMinima (geraCNF4 207 17) 6 (fun _ => false)).1.isSome = temLuz4 (geraCNF4 207 17))
    ∧ ((processoDaTensaoMinima (geraCNF4 304 17) 6 (fun _ => false)).1.isSome
        = temLuz4 (geraCNF4 304 17)) :=
  ⟨bm5, bm6, bm7, bm8⟩

end TGLExt.ABateriaMinima2

