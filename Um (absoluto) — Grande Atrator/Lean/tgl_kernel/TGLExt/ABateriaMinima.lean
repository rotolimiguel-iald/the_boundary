import TGLExt.ABateriaDaParede
import TGLExt.AContagemDaTensaoMinima

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria sob a tensão mínima I: três variáveis

As quatro fórmulas de três variáveis e dez cláusulas de `ABateriaDaParede` (sementes 17, 1010, 1341 e 1672), mais `phiCE`, sob a
regra da tensão mínima: o processo responde com eco exatamente quando a fórmula tem solução (por `decide`) [REAL].

## O que NÃO se afirma

* Que a regra da tensão mínima responda com eco em toda fórmula que tem solução: a bateria é uma amostra, e a
  fórmula da armadilha do empate (`OMapaDosMuros.phiTrap`) tem solução e nela a regra responde com silêncio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaMinima

open TGLExt.ABateriaDaParede TGLExt.ATensaoMinima TGLExt.TraducaoTuring

/-- ★★ As oito de n = 3 sob a regra mínima. -/
theorem bm1 : (processoDaTensaoMinima (geraCNF3 17 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 17 10) := by decide

theorem bm2 : (processoDaTensaoMinima (geraCNF3 1010 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 1010 10) := by decide

theorem bm3 : (processoDaTensaoMinima (geraCNF3 1341 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 1341 10) := by decide

theorem bm4 : (processoDaTensaoMinima (geraCNF3 1672 10) 5 (fun _ => false)).1.isSome
    = temLuz (geraCNF3 1672 10) := by decide

/-- ★★ A bateria mínima I confere. -/
theorem a_bateria_minima_I_confere :
    ((processoDaTensaoMinima (geraCNF3 17 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 17 10))
    ∧ ((processoDaTensaoMinima (geraCNF3 1010 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 1010 10))
    ∧ ((processoDaTensaoMinima (geraCNF3 1341 10) 5 (fun _ => false)).1.isSome = temLuz (geraCNF3 1341 10))
    ∧ ((processoDaTensaoMinima (geraCNF3 1672 10) 5 (fun _ => false)).1.isSome
        = temLuz (geraCNF3 1672 10)) :=
  ⟨bm1, bm2, bm3, bm4⟩

/-- ★★ `phiCE` na bateria: o processo da tensão mínima responde com eco. -/
theorem bmCE : (processoDaTensaoMinima phiCE 5 (fun _ => false)).1.isSome
    = temLuz phiCE := by decide

end TGLExt.ABateriaMinima

