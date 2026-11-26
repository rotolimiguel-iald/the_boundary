import TGLExt.ADescidaPinadaEmMemoria
import TGLExt.ABateriaMinima3

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A bateria sob a busca com o livro I

As quatro fórmulas de três variáveis de `ABateriaDaParede` (sementes 17, 1010, 1341 e 1672), `phiCE`, `phiFofa2`, `phiFofa` e
`phiCadeia`, sob a busca com a poda e o livro: o processo responde com eco exatamente quando a fórmula tem solução, e responde
com silêncio numa fórmula sem solução de uma variável (por `decide`) [REAL].

## O que NÃO se afirma

* Que a busca responda com eco em toda fórmula que tem solução: a bateria é uma amostra, e isso segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ABateriaSobAMemoria1

open TGLExt.ABateriaDaParede TGLExt.ATensaoMinima TGLExt.AFamiliaProfunda TGLExt.AMedidaDaArvore
open TGLExt.ADescidaPinadaEmMemoria TGLExt.OMapaDosMuros TGLExt.TraducaoTuring
open TGLExt.ACartografiaDaParede TGLExt.AParedeDoLadoOposto TGLExt.OFechoDoCampo

/-- [MEDIDA] A conferência sob a máquina final. -/
def confereM3 (φ : CNF 3) : Bool :=
  (processoComPodaM φ 12 (fun _ => false)).1.isSome == temLuz φ

/-- ★★ A bateria n = 3 sob a máquina final: as quatro, todas conferem. -/
theorem a_bateria_n3_sob_a_memoria :
    confereM3 (geraCNF3 17 10) ∧ confereM3 (geraCNF3 1010 10)
      ∧ confereM3 (geraCNF3 1341 10) ∧ confereM3 (geraCNF3 1672 10) := by
  decide

/-- [MEDIDA] A conferência sob a máquina final, em n = 2. -/
def confereM2 (φ : CNF 2) : Bool :=
  (processoComPodaM φ 12 (fun _ => false)).1.isSome == (cubo 2).any (fun x => decide (custoCNF φ x = 0))

/-- ★★ Os muros e o contraexemplo sob a máquina final: todos com luz, todos com eco. -/
theorem os_muros_sob_a_memoria :
    confereM3 phiCE ∧ confereM3 phiFofa2 ∧ confereM2 phiFofa ∧ confereM3 phiCadeia := by
  decide

/-- ★★ A honestidade sob a máquina final: a insatisfazível cala. -/
theorem a_honestidade_sob_a_memoria :
    (processoComPodaM ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 12 (fun _ => false)).1 = none := by
  decide

end TGLExt.ABateriaSobAMemoria1

