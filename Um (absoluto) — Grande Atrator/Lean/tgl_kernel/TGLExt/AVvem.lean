import TGLExt.OSolverGF2_Parte4

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A condição por morte, medida

## O que é

Na descida com espectro, cada morte guarda um lado: a paridade `v` num eixo `a`. A condição por morte (`morteSadia`) é que
o lado guardado guarde toda solução da fórmula que serve o espectro. Esta pedra define a condição e a mede numa fórmula de
duas variáveis.

## O que fica provado [REAL]

1. `morteSadia` [DEF]: o lado guardado (a paridade `v` no eixo `a`) guarda toda solução que serve o espectro.
2. ★ `a_morte_do_lado_do_poco_e_insidia` (por `decide`): na fórmula «x₀ ou x₁» e «não x₀ ou não x₁» (`phiMedida`), cujas
   soluções são (V, F) e (F, V), guardar no eixo das duas variáveis a paridade 0, a do canto (F, F), não é sadio: as duas
   soluções têm paridade 1.
3. ★ `a_morte_do_lado_oposto_e_sadia_aqui` (por `decide`): na mesma fórmula, guardar a paridade 1 é sadio.
4. `o_residuo_final_com_a_luz`: um ponto de custo zero dá SAT. As hipóteses de servir o espectro e de ser o único ponto
   que satisfaz e serve não entram na prova.

## O que NÃO se afirma

* A condição por morte em geral, para toda fórmula e todo poço, não está provada; que, havendo solução, o eco venha segue
  [OPEN].
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AVvem

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4

variable {n : ℕ}

/-- [DEF] **A morte sadia**: o lado guardado (a paridade `v` no eixo `a`) guarda toda solução da fórmula
que serve o espectro. (A condição por morte do V-vem.) -/
def morteSadia {n : ℕ} (φ : CNF n) (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2) : Prop :=
  ∀ x : Fin n → Bool, custoCNF φ x = 0 → serveEspectro2 x S = true → paridade2 x a = v

/-! ## A medida -/

/-- A fórmula da medida: `x₀ ∨ x₁` e `¬x₀ ∨ ¬x₁` — as soluções são (V,F) e (F,V), as duas de paridade 1. -/
def phiMedida : CNF 2 := [[⟨0, true⟩, ⟨1, true⟩], [⟨0, false⟩, ⟨1, false⟩]]

/-- O eixo da medida: as duas variáveis. -/
def eixoMedida : Fin 2 → Bool := fun _ => true

/-- ★ **A MORTE DO LADO DO POÇO É INSIDIA**: na fórmula da medida, guardar a paridade 0 (o lado do canto)
perde toda solução. (Medido por `decide`.) -/
theorem a_morte_do_lado_do_poco_e_insidia :
    ¬ morteSadia phiMedida [] eixoMedida 0 := by
  unfold morteSadia
  decide

/-- ★ **A MORTE DO LADO OPOSTO É SADIA AQUI**: guardar a paridade 1 (o lado oposto ao canto) guarda toda
solução. (Medido por `decide`.) -/
theorem a_morte_do_lado_oposto_e_sadia_aqui :
    morteSadia phiMedida [] eixoMedida 1 := by
  unfold morteSadia
  decide

/-- Um ponto de custo zero dá SAT. As hipóteses `hserve` (o ponto serve o espectro) e `hguarda` (ele é o único que
satisfaz e serve) não entram na prova. -/
theorem o_residuo_final_com_a_luz {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (hsat : custoCNF φ x = 0) (hserve : serveEspectro2 x S = true)
    (hguarda : ∀ y : Fin n → Bool, custoCNF φ y = 0 → serveEspectro2 y S = true → y = x) :
    SAT φ :=
  ⟨x, (custoCNF_zero_sse φ x).mp hsat⟩

end TGLExt.AVvem

