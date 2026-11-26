import TGLExt.OSinalDeRetorno

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O campo fala primeiro: as cláusulas unitárias, e o silêncio do retorno

## O que é

Uma cláusula de uma literal só força o bit dessa literal em toda solução. O processo com o campo primeiro
(`processoComCampoPrimeiro`) registra no espectro inicial as equações dessas cláusulas (`oCampoFala`) e corre a partir
daí o laço com retorno (`OSinalDeRetorno.lacoComRetorno`).

## O que fica provado [REAL]

1. ★ `a_unitaria_e_leitura_forcada`: toda solução serve a equação de uma cláusula unitária da fórmula.
2. ★★ `o_campo_fala_e_gratuito`: toda solução serve o espectro das cláusulas unitárias.
3. ★★ `o_silencio_do_retorno_e_a_ausencia_da_luz`: se o ponto `w` serve o espectro e falha uma cláusula da fórmula, e
   nenhuma literal dessa cláusula tem equação nova e consistente com o espectro, então nenhuma solução serve o espectro.
4. As medidas, por `decide`: a equação unitária de `phiFofa2` é servida pela solução dela; com o campo primeiro, o
   processo responde com a solução em `phiFofa`, cala na fórmula «x₀» e «não x₀», e cala em `phiFofa2`, que tem solução.

## O que NÃO se afirma

* Que o processo responda com eco quando a fórmula tem solução: `phiFofa2` mede o contrário. O silêncio do retorno diz
  que nenhuma solução serve o espectro daquele momento, não que a fórmula não tenha solução.
* O valor da morte com o eixo fora do span segue [OPEN]; a propagação iterada das cláusulas unitárias não é feita aqui.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OCampoFalaPrimeiro

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno

/-- [DEF] As cláusulas unitárias do campo: as de uma literal só. -/
def unitariasDoCampo {n : ℕ} (φ : CNF n) : List (Literal n) :=
  φ.filterMap (fun C => match C with | [l] => some l | _ => none)

/-- [DEF] O campo fala primeiro: o espectro das leituras forçadas. -/
def oCampoFala {n : ℕ} (φ : CNF n) : Espectro2 n :=
  (unitariasDoCampo φ).map (fun l => (mascaraDeLiteral l, valorDeLiteral l))

/-- ★ **A UNITÁRIA É LEITURA FORÇADA**: a cláusula de uma literal só força o bit — toda luz serve a
equação dela. Registrar a unitária não é escolha: é o campo falando. -/
theorem a_unitaria_e_leitura_forcada {n : ℕ} (φ : CNF n) (l : Literal n)
    (hl : [l] ∈ φ) (y : Fin n → Bool) (hy : custoCNF φ y = 0) :
    serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
  have hCy : satisfeita [l] y = true := (custoCNF_zero_sse φ y).mp hy [l] hl
  unfold satisfeita at hCy
  rw [List.any_eq_true] at hCy
  obtain ⟨l', hl', hly⟩ := hCy
  have hll : l' = l := by
    simp at hl'
    exact hl'
  have hserve : serveEq y (mascaraDeLiteral l', valorDeLiteral l') := by
    show paridade2 y (mascaraDeLiteral l') = valorDeLiteral l'
    rw [paridade2_mascaraLiteral]
    unfold valorDeLiteral
    unfold Literal.vale at hly
    rw [beq_iff_eq] at hly
    rw [hly]
  rw [hll] at hserve
  exact hserve

/-- ★★ **O CAMPO FALA DE GRAÇA**: o espectro das unitárias é servido por toda luz — as leituras forçadas
não perdem luz nenhuma. -/
theorem o_campo_fala_e_gratuito {n : ℕ} (φ : CNF n) (y : Fin n → Bool)
    (hy : custoCNF φ y = 0) : serveEspectro2 y (oCampoFala φ) = true := by
  unfold serveEspectro2 oCampoFala
  rw [List.all_eq_true]
  intro e he
  rw [List.mem_map] at he
  obtain ⟨l, hl, hle⟩ := he
  unfold unitariasDoCampo at hl
  rw [List.mem_filterMap] at hl
  obtain ⟨C, hCφ, hCl⟩ := hl
  have hlunit : [l] ∈ φ := by
    cases hC : C with
    | nil => rw [hC] at hCl; simp at hCl
    | cons a Cs =>
      cases Cs with
      | nil =>
        rw [hC] at hCl
        simp at hCl
        rw [← hCl]
        exact hC ▸ hCφ
      | cons b Cs' => rw [hC] at hCl; simp at hCl
  have hserve := a_unitaria_e_leitura_forcada φ l hlunit y hy
  show (paridade2 y e.1 == e.2) = true
  rw [← hle]
  exact beq_iff_eq.mpr hserve

/-- ★★ Se o ponto serve o espectro e falha uma cláusula da fórmula, e nenhuma literal dessa cláusula retorna, então
nenhuma solução serve o espectro. Não diz que a fórmula não tenha solução: uma morte anterior pode ter excluído do
espectro todas as soluções. -/
theorem o_silencio_do_retorno_e_a_ausencia_da_luz {n : ℕ} (φ : CNF n) (S : Espectro2 n)
    (w : Fin n → Bool) (C : Clausula n) (hCφ : C ∈ φ)
    (hpoco : serveEspectro2 w S = true) (hfalha : satisfeita C w = false)
    (hsem : literalDeRetorno S C = none) :
    ¬ ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true := by
  intro hluz
  obtain ⟨l, hlC, hret⟩ := o_sinal_de_retorno_existe φ S w C hCφ hpoco hfalha hluz
  unfold literalDeRetorno at hsem
  rw [List.find?_eq_none] at hsem
  exact hsem l hlC hret

/-- [DEF] O processo com o campo falando primeiro: as leituras forçadas entram no espectro inicial; o
laço com retorno corre a partir daí. -/
def processoComCampoPrimeiro {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :
    (Option (Fin n → Bool)) × Espectro2 n :=
  lacoComRetorno φ fuel (oCampoFala φ) (ressurreicaoConstrutiva (oCampoFala φ))

/-- A medida: a leitura forçada da fofa2 (o campo diz x₂ = falso) é servida pela luz. -/
theorem a_medida_o_campo_da_fofa2_e_servido_pela_luz :
    serveEspectro2 luzFofa2 (oCampoFala phiFofa2) = true := by
  decide

/-- A medida: com o campo falando primeiro, a fofa segue ecoando com a própria luz. -/
theorem a_medida_o_campo_primeiro_ecoa_na_fofa :
    (processoComCampoPrimeiro phiFofa 5 (fun _ => false)).1 = some (fun _ => true) := by
  decide

/-- A medida da honestidade: a insatisfazível cala mesmo com o campo falando primeiro. -/
theorem a_medida_o_campo_primeiro_cala_na_insatisfazível :
    (processoComCampoPrimeiro ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 3 (fun _ => false)).1 = none := by
  decide

/-- ★★ A medida: mesmo com as cláusulas unitárias registradas primeiro, o processo cala em `phiFofa2`, que tem solução:
a morte com o eixo fora do span e a paridade oposta exclui a solução. -/
theorem a_medida_o_gate_final_e_o_valor_da_morte_transversal :
    (processoComCampoPrimeiro phiFofa2 6 (fun _ => false)).1 = none := by
  decide

end TGLExt.OCampoFalaPrimeiro

