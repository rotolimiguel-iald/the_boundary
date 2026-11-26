import TGLExt.ABateriaMinimaN5b

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A varredura ampla I: cinco variáveis, sementes da forma i·331 + 17

A medida `confere s` (o processo da tensão mínima responde com eco exatamente quando a fórmula `geraCNF5 s 21` tem solução) e a
varredura das sementes 17, 348, 679, 1010, 1341, 1672, 2003 e 2334: as oito conferem (por `decide`) [REAL]. As pedras
`AVarreduraAmpla2` e `AVarreduraAmpla3` fazem o mesmo com mais dezesseis sementes da mesma família.

## O que NÃO se afirma

* Que a regra da tensão mínima responda com eco em toda fórmula que tem solução: a bateria é uma amostra, e a
  fórmula da armadilha do empate (`OMapaDosMuros.phiTrap`) tem solução e nela a regra responde com silêncio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AVarreduraAmpla1

open TGLExt.ABateriaDaParede TGLExt.ABateriaMinimaN5 TGLExt.ATensaoMinima TGLExt.TraducaoTuring

/-- [MEDIDA] A conferência de uma semente: eco ⟺ luz. -/
def confere (s : ℕ) : Bool :=
  (processoDaTensaoMinima (geraCNF5 s 21) 7 (fun _ => false)).1.isSome == temLuz5 (geraCNF5 s 21)

/-- ★★ A varredura I: as sementes 17, 348, 679, 1010, 1341, 1672, 2003, 2334 — todas conferem. -/
theorem a_varredura_I_confere :
    confere 17 ∧ confere 348 ∧ confere 679 ∧ confere 1010
      ∧ confere 1341 ∧ confere 1672 ∧ confere 2003 ∧ confere 2334 := by
  decide

end TGLExt.AVarreduraAmpla1

