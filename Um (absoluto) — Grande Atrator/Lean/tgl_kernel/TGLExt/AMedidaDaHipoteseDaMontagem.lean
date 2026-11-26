import TGLExt.AMontagemFinal

set_option autoImplicit false

/-!
# A medida da hipótese da montagem

## O que é

A hipótese nomeada dos dois teoremas da montagem (`AMontagemFinal.o_processo_da_tensao_ecoa` e
`AMontagemFinal.o_processo_completo_ecoa`), escrita como proposição (`HipoteseDaMontagem`) e medida numa fórmula pequena
e satisfazível.

## O que fica provado [REAL]

1. `HipoteseDaMontagem` [DEF]: para todo espectro S, todo ponto w que o serve, toda cláusula C da fórmula que w falha e toda
   literal l de C que é retorno útil e que o campo não condena, existe uma solução que serve S e a equação de l. É a
   hipótese dos dois teoremas da montagem, palavra por palavra.
2. `phiMontagem` [DEF]: a fórmula de quatro variáveis (x₀ ∨ x₁) ∧ (¬x₀ ∨ x₂ ∨ x₃) ∧ (¬x₀ ∨ x₂ ∨ ¬x₃) ∧ (¬x₀ ∨ ¬x₂ ∨ x₃) ∧
   (¬x₀ ∨ ¬x₂ ∨ ¬x₃).
3. `phiMontagem_satisfazivel`: ela tem solução (x₀ falso, x₁ verdadeiro).
4. `a_literal_x0_nao_e_condenada`: no espectro vazio, o campo não condena a literal x₀ da primeira cláusula; registrar x₀ e
   fazer a propagação não deixa cláusula com uma literal só nem cláusula toda proibida.
5. `nenhuma_solucao_tem_x0`: nenhuma solução de `phiMontagem` tem x₀ verdadeiro.
6. ★★ `a_hipotese_da_montagem_falha`: `HipoteseDaMontagem phiMontagem` é falsa; para essa fórmula satisfazível, os dois
   teoremas da montagem valem no vazio.
7. ★★ `o_processo_da_tensao_cala_na_formula_da_medida` e `o_processo_da_tensao_cala_com_mais_combustivel`: partindo do
   ponto com todas as variáveis falsas, o processo da tensão responde com silêncio nessa fórmula satisfazível, com o
   combustível n + 1 dos teoremas da montagem e com combustível 50.
8. ★★ `phiMontagem3_satisfazivel` e `o_processo_da_tensao_cala_em_tres_variaveis`: a variante de três variáveis,
   (x₀ ∨ x₁) e as quatro cláusulas (¬x₀ ∨ ±x₁ ∨ ±x₂), tem solução, e nela o processo da tensão responde com silêncio, com o
   combustível n + 1.

## O que NÃO se afirma

* Nada sobre fórmulas em geral, nem sobre a máquina de Turing.
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMedidaDaHipoteseDaMontagem

open TGLExt.TraducaoTuring TGLExt.AMorteEARessurreicao TGLExt.OSolverGF2_Parte2 TGLExt.OSinalDeRetorno
open TGLExt.ATensaoFundamental

/-- [DEF] A hipótese nomeada dos dois teoremas da montagem, como proposição. -/
def HipoteseDaMontagem {n : ℕ} (φ : CNF n) : Prop :=
  ∀ (S : Espectro2 n) (w : Fin n → Bool) (C : Clausula n) (l : Literal n),
    serveEspectro2 w S = true → satisfeita C w = false → C ∈ φ → l ∈ C →
    retornoUtil S l = true → condenadaNoCampo φ S l = false →
    ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true ∧
      serveEq y (mascaraDeLiteral l, valorDeLiteral l)

/-- [DEF] A fórmula da medida: (x₀ ∨ x₁) e as quatro cláusulas (¬x₀ ∨ ±x₂ ∨ ±x₃), que juntas impedem x₀ sem que nenhuma fique
com uma literal só. -/
def phiMontagem : CNF 4 :=
  [[⟨0, true⟩, ⟨1, true⟩],
   [⟨0, false⟩, ⟨2, true⟩, ⟨3, true⟩],
   [⟨0, false⟩, ⟨2, true⟩, ⟨3, false⟩],
   [⟨0, false⟩, ⟨2, false⟩, ⟨3, true⟩],
   [⟨0, false⟩, ⟨2, false⟩, ⟨3, false⟩]]

/-- A fórmula da medida tem solução: x₀ falso, x₁ verdadeiro. -/
theorem phiMontagem_satisfazivel : ∃ y : Fin 4 → Bool, custoCNF phiMontagem y = 0 :=
  ⟨fun i => decide (i = 1), by decide⟩

/-- No espectro vazio, o campo não condena a literal x₀. -/
theorem a_literal_x0_nao_e_condenada : condenadaNoCampo phiMontagem [] ⟨0, true⟩ = false := by
  decide

/-- No espectro vazio, a literal x₀ é retorno útil. -/
theorem a_literal_x0_e_retorno_util : retornoUtil ([] : Espectro2 4) ⟨0, true⟩ = true := by
  decide

/-- Nenhuma solução da fórmula da medida tem x₀ verdadeiro. -/
theorem nenhuma_solucao_tem_x0 : ∀ y : Fin 4 → Bool, custoCNF phiMontagem y = 0 → y 0 = false := by
  decide

/-- ★★ A hipótese da montagem é falsa para a fórmula da medida, que é satisfazível. -/
theorem a_hipotese_da_montagem_falha : ¬ HipoteseDaMontagem phiMontagem := by
  intro h
  obtain ⟨y, hy, -, hl⟩ := h [] (fun _ => false) [⟨0, true⟩, ⟨1, true⟩] ⟨0, true⟩
    (by decide) (by decide) List.mem_cons_self List.mem_cons_self a_literal_x0_e_retorno_util a_literal_x0_nao_e_condenada
  have h0 := nenhuma_solucao_tem_x0 y hy
  unfold serveEq at hl
  rw [paridade2_mascaraLiteral] at hl
  simp only [h0] at hl
  exact absurd hl (by decide)

/-- ★★ Partindo do ponto com todas as variáveis falsas, o processo da tensão responde com silêncio na fórmula da medida, com o
combustível n + 1 = 5 dos teoremas da montagem. -/
theorem o_processo_da_tensao_cala_na_formula_da_medida :
    (processoDaTensao phiMontagem 5 (fun _ => false)).1.isSome = false := by
  decide

/-- ★★ O mesmo silêncio com combustível 50. -/
theorem o_processo_da_tensao_cala_com_mais_combustivel :
    (processoDaTensao phiMontagem 50 (fun _ => false)).1.isSome = false := by
  decide

/-- [DEF] A variante de três variáveis: (x₀ ∨ x₁) e as quatro cláusulas (¬x₀ ∨ ±x₁ ∨ ±x₂). -/
def phiMontagem3 : CNF 3 :=
  [[⟨0, true⟩, ⟨1, true⟩],
   [⟨0, false⟩, ⟨1, true⟩, ⟨2, true⟩],
   [⟨0, false⟩, ⟨1, true⟩, ⟨2, false⟩],
   [⟨0, false⟩, ⟨1, false⟩, ⟨2, true⟩],
   [⟨0, false⟩, ⟨1, false⟩, ⟨2, false⟩]]

/-- A variante de três variáveis tem solução: x₀ falso, x₁ verdadeiro, x₂ falso. -/
theorem phiMontagem3_satisfazivel : ∃ y : Fin 3 → Bool, custoCNF phiMontagem3 y = 0 :=
  ⟨fun i => decide (i = 1), by decide⟩

/-- ★★ Em três variáveis, partindo do ponto com todas as variáveis falsas, o processo da tensão responde com silêncio na
variante, com o combustível n + 1 = 4 dos teoremas da montagem. -/
theorem o_processo_da_tensao_cala_em_tres_variaveis :
    (processoDaTensao phiMontagem3 4 (fun _ => false)).1.isSome = false := by
  decide

end TGLExt.AMedidaDaHipoteseDaMontagem
