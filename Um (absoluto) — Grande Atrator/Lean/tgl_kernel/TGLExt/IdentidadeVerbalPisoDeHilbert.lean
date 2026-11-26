import Mathlib

set_option autoImplicit false

/-!
# A identidade verbal no piso de Hilbert

A implicação "mesma leitura e mesmo nome é identidade" de `IdentidadeVerbal`, sem matriz e sem dimensão finita:
vale em QUALQUER espaço com produto interno.

* `IsReadingOf K y d` — `d` é a leitura de `y` pelo leitor `K`: `d` está no código e o que
  sobra (`y - d`) é ortogonal ao código. (A forma de `IsFrobProjection` da v143, num ponto.)
* `on_the_hilbert_floor` — mesma leitura e mesma norma ⟹ igualdade.
* `the_reading_returns_the_name_on_the_hilbert_floor` — para toda isometria linear `V`
  (o transporte): se a leitura de `V d` é `d`, então `V d = d`.

HONESTIDADE: é álgebra de espaço com produto interno (Pitágoras); `H` não precisa ser completo
(é mais geral que Hilbert). "A mesma implicação" da versão de matrizes é por analogia de forma:
não há aqui ponte formal entre os dois arquivos. Não
constrói o leitor, não diz qual é o transporte físico, não é a prova global. Sem sorry, sem axiom.
-/

namespace TGLExt.IdentidadeVerbalPisoDeHilbert

open scoped InnerProductSpace

variable {𝕜 H : Type*} [RCLike 𝕜] [NormedAddCommGroup H] [InnerProductSpace 𝕜 H]

/-- `d` é a leitura de `y` pelo leitor `K`. -/
def IsReadingOf (K : Submodule 𝕜 H) (y d : H) : Prop :=
  d ∈ K ∧ ∀ w ∈ K, ⟪y - d, w⟫_𝕜 = 0

/-- ★★★ NO PISO DE HILBERT: mesma leitura e mesmo nome é identidade. -/
theorem on_the_hilbert_floor {K : Submodule 𝕜 H} {y d : H}
    (hread : IsReadingOf K y d) (hname : ‖y‖ = ‖d‖) : y = d := by
  have h0 : ⟪y - d, d⟫_𝕜 = 0 := hread.2 d hread.1
  have h1 : ⟪d, y - d⟫_𝕜 = 0 := inner_eq_zero_symm.mp h0
  have hp := norm_add_sq_eq_norm_sq_add_norm_sq_of_inner_eq_zero d (y - d) h1
  have e : d + (y - d) = y := by abel
  rw [e, hname] at hp
  have hz : ‖y - d‖ * ‖y - d‖ = 0 := by linarith
  have hn : ‖y - d‖ = 0 := mul_self_eq_zero.mp hz
  exact sub_eq_zero.mp (norm_eq_zero.mp hn)

/-- ★★★ A LEITURA DEVOLVE O NOME ⟹ IDENTIDADE, para toda isometria linear. -/
theorem the_reading_returns_the_name_on_the_hilbert_floor {K : Submodule 𝕜 H}
    (V : H →ₗᵢ[𝕜] H) {d : H} (hread : IsReadingOf K (V d) d) : V d = d :=
  on_the_hilbert_floor hread (V.norm_map d)

end TGLExt.IdentidadeVerbalPisoDeHilbert
