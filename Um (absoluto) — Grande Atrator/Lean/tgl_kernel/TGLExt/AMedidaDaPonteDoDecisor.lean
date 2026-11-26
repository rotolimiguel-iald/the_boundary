import TGLExt.APonteDoDecisor
import TGLExt.OMapaDosMuros
import TGLExt.AMedidaDaHipoteseDaMontagem

set_option autoImplicit false

/-!
# A medida da hipótese da ponte do decisor

## O que é

A ponte do decisor (`APonteDoDecisor.o_processo_e_o_alvo_se_correspondem`) e a montagem do laço da tensão mínima
(`AContagemDaTensaoMinima.o_processo_minimo_completo_ecoa`) têm a mesma hipótese nomeada `hgate`. Esta pedra mostra que essa
hipótese é exatamente `HipoteseDaMontagem` (a de `AMontagemFinal`), sem conversão, e a mede em duas fórmulas satisfazíveis: a
de três variáveis do contraexemplo (`ATensaoMinima.phiCE`) e a da armadilha do empate (`OMapaDosMuros.phiTrap`).

## O que fica provado [REAL]

1. ★ `a_ponte_vale_sob_a_hipotese_da_montagem` e `a_montagem_minima_vale_sob_a_hipotese_da_montagem`: os dois teoremas
   aplicam-se a toda fórmula que satisfaz `HipoteseDaMontagem`; a hipótese deles é essa proposição, por definição.
2. ★★ `a_hipotese_falha_no_contraexemplo`: a hipótese é falsa em `phiCE`, que tem solução. A literal x₀ da cláusula
   (x₀ ∨ x₁) é retornável e não é condenada pelo campo no espectro vazio, e nenhuma solução tem x₀ verdadeiro.
3. ★★ `a_regra_minima_ecoa_onde_a_hipotese_falha`: em `phiCE` o processo da tensão mínima responde com a solução, e a
   hipótese é falsa. A hipótese não descreve a regra da tensão mínima: ela pede que toda literal limpa leve a uma solução,
   qualquer que seja a regra de escolha.
4. ★★ `a_hipotese_falha_na_armadilha`: a hipótese é falsa em `phiTrap`, que tem solução; nela o processo da tensão mínima
   responde com silêncio (`OMapaDosMuros.a_medida_a_tensao_minima_e_refutada_pela_armadilha`).

## O que NÃO se afirma

* Que a ponte do decisor ou a montagem estejam erradas: são implicações corretas; a hipótese delas é que é falsa nas
  fórmulas medidas.
* Nada sobre o laço com a poda (`APodaFechaOContorno`), que não usa esta hipótese.
* Que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMedidaDaPonteDoDecisor
open TGLExt.TraducaoTuring TGLExt.AMorteEARessurreicao TGLExt.OSolverGF2_Parte2 TGLExt.OSinalDeRetorno
open TGLExt.ATensaoFundamental TGLExt.OFechoDoCampo TGLExt.ATensaoMinima TGLExt.AContagemDaTensaoMinima
open TGLExt.APonteDoDecisor TGLExt.OMapaDosMuros TGLExt.AMedidaDaHipoteseDaMontagem TGLExt.APonteAoTM2

/-- ★ A ponte do decisor vale sob a hipótese da montagem: a hipótese dela é `HipoteseDaMontagem`, por definição. -/
theorem a_ponte_vale_sob_a_hipotese_da_montagem {n : ℕ} (φ : CNF n) (h : HipoteseDaMontagem φ) :
    (processoDaTensaoMinima φ (n + 1) (fun _ => false)).1.isSome = true ↔ satNaFitaLocal (aFormulaNaFita φ) :=
  o_processo_e_o_alvo_se_correspondem φ h

/-- ★ A montagem do laço da tensão mínima vale sob a mesma hipótese. -/
theorem a_montagem_minima_vale_sob_a_hipotese_da_montagem {n : ℕ} (φ : CNF n) (h : HipoteseDaMontagem φ)
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0) :
    (processoDaTensaoMinima φ (n + 1) (fun _ => false)).1.isSome = true :=
  o_processo_minimo_completo_ecoa φ h hluz

/-- Nenhuma solução de `phiCE` tem x₀ verdadeiro. -/
theorem nenhuma_solucao_do_contraexemplo_tem_x0 : ∀ y : Fin 3 → Bool, custoCNF phiCE y = 0 → y 0 = false := by
  decide

/-- ★★ A hipótese da montagem é falsa em `phiCE`, que tem solução. -/
theorem a_hipotese_falha_no_contraexemplo : ¬ HipoteseDaMontagem phiCE := by
  intro h
  obtain ⟨y, hy, -, hl⟩ := h [] (fun _ => false) [⟨0, true⟩, ⟨1, true⟩] ⟨0, true⟩
    (by decide) (by decide) List.mem_cons_self List.mem_cons_self (by decide) (by decide)
  have h0 := nenhuma_solucao_do_contraexemplo_tem_x0 y hy
  unfold serveEq at hl
  rw [paridade2_mascaraLiteral] at hl
  simp only [h0] at hl
  exact absurd hl (by decide)

/-- ★★ Em `phiCE` o processo da tensão mínima responde com a solução (falso, verdadeiro, falso), e a hipótese da montagem
é falsa. -/
theorem a_regra_minima_ecoa_onde_a_hipotese_falha :
    (processoDaTensaoMinima phiCE 5 (fun _ => false)).1 = some (fun i => decide (i = 1))
    ∧ ¬ HipoteseDaMontagem phiCE :=
  ⟨a_medida_o_contraexemplo_ecoa_com_a_tensao_minima, a_hipotese_falha_no_contraexemplo⟩

/-- Nenhuma solução de `phiTrap` tem x₀ verdadeiro. -/
theorem nenhuma_solucao_da_armadilha_tem_x0 : ∀ y : Fin 4 → Bool, custoCNF phiTrap y = 0 → y 0 = false := by
  decide

/-- ★★ A hipótese da montagem é falsa em `phiTrap`, que tem solução. -/
theorem a_hipotese_falha_na_armadilha : ¬ HipoteseDaMontagem phiTrap := by
  intro h
  obtain ⟨y, hy, -, hl⟩ := h [] (fun _ => false) [⟨0, true⟩, ⟨1, true⟩] ⟨0, true⟩
    (by decide) (by decide) List.mem_cons_self List.mem_cons_self (by decide) (by decide)
  have h0 := nenhuma_solucao_da_armadilha_tem_x0 y hy
  unfold serveEq at hl
  rw [paridade2_mascaraLiteral] at hl
  simp only [h0] at hl
  exact absurd hl (by decide)

end TGLExt.AMedidaDaPonteDoDecisor
