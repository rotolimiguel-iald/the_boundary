import TGLExt.OSilencioVivo

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A parede do lado oposto: o laço final cala numa fórmula satisfazível

## O que é

O laço final registra, em cada ponto que não satisfaz, a máscara da primeira cláusula sem par com a paridade oposta à do
ponto. Esta pedra mede, por `decide`, uma fórmula de duas variáveis em que essa regra perde a solução.

## O que fica provado [REAL]

1. `phiFofa` [DEF]: «x₀ ou x₁», «x₀ ou não x₁» e «não x₀ ou x₁», cuja única solução é (V, V). `a_luz_existe_na_fofa`: (V, V)
   tem custo zero. `o_canto_nao_e_luz`: o canto (F, F) falha uma cláusula. `o_eixo_da_morte_no_canto`: o eixo da morte no
   canto é a máscara das duas variáveis.
2. ★ `a_luz_toda_do_lado_do_canto`: toda solução tem, nesse eixo, a paridade do canto (0). ★ `a_morte_oposta_no_canto_mata_a_luz`:
   a morte com a paridade 1 nesse eixo não é útil.
3. ★★ `a_medida_o_laco_cala_com_luz`: a partir do canto, com combustível 3, o laço final responde em silêncio, embora a fórmula
   tenha solução.
4. ★★ `a_parede_do_lado_oposto`: existe fórmula com solução, e um ponto de custo não nulo com eixo de morte fora do span,
   em que a morte com a paridade oposta não é útil.

## O que NÃO se afirma

* Nada aqui mede outra regra de registro além da que o laço final usa.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AParedeDoLadoOposto

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo

/-- [MEDIDA] A fórmula-fofa: três cláusulas de duas variáveis cuja única solução é o ponto todo aceso.
(A luz existe e é única — qualquer corte transversal que a erre mata TODA a luz.) -/
def phiFofa : CNF 2 := [[⟨0, true⟩, ⟨1, true⟩], [⟨0, true⟩, ⟨1, false⟩], [⟨0, false⟩, ⟨1, true⟩]]

/-- A luz existe: o ponto todo aceso satisfaz a fofa. -/
theorem a_luz_existe_na_fofa : custoCNF phiFofa (fun _ => true) = 0 := by decide

/-- O canto não é luz: falha exatamente uma cláusula. -/
theorem o_canto_nao_e_luz : custoCNF phiFofa (fun _ => false) = 1 := by decide

/-- O eixo lido no canto é a máscara cheia das duas variáveis. -/
theorem o_eixo_da_morte_no_canto :
    eixoDaMorteDe phiFofa (fun _ => false) = (fun _ => true) := by decide

/-- ★ **TODA A LUZ DO LADO DO CANTO**: qualquer solução da fofa tem, no eixo da morte do canto, a mesma
paridade do canto — o lado oposto está vazio de luz. -/
theorem a_luz_toda_do_lado_do_canto :
    ∀ y : Fin 2 → Bool, custoCNF phiFofa y = 0 → paridade2 y (fun _ => true) = 0 := by
  decide

/-- ★ **A MORTE OPOSTA MATA A LUZ**: no canto da fofa, a equação que a regra registra (a máscara cheia,
o lado oposto) não guarda solução nenhuma — a morte útil é impossível naquele lado. -/
theorem a_morte_oposta_no_canto_mata_a_luz :
    ¬ morteUtil phiFofa [] (fun _ => true) (1 : ZMod 2) := by
  intro h
  obtain ⟨y, hy, -, hpar⟩ := h
  have h0 := a_luz_toda_do_lado_do_canto y hy
  rw [h0] at hpar
  exact (by decide : (0 : ZMod 2) ≠ 1) hpar

/-- ★★ **A MEDIDA DO PROCESSO**: na fórmula-fofa, partindo do canto com fuel de sobra, o laço final
responde SILÊNCIO — com a luz existente. A primeira morte mata a luz; a segunda cai no span e o processo
cala para sempre. -/
theorem a_medida_o_laco_cala_com_luz :
    (lacoFinal phiFofa 3 [] (fun _ => false)).1 = none := by
  decide

/-- ★★ **A PAREDE DO LADO OPOSTO**: existe configuração — poço sem par, eixo transversal ao espectro,
luz existente — em que a morte no lado oposto não guarda luz nenhuma. O gate `hgates` do fecho do arco
não é realizado pela regra da máscara + lado oposto: a regra da morte útil pede OUTRA leitura. -/
theorem a_parede_do_lado_oposto :
    ∃ (φ : CNF 2) (w : Fin 2 → Bool),
      custoCNF φ w ≠ 0
      ∧ ¬ eixoNoSpanB (([] : Espectro2 2).map Prod.fst) (eixoDaMorteDe φ w) = true
      ∧ ¬ morteUtil φ [] (eixoDaMorteDe φ w) (paridade2 w (eixoDaMorteDe φ w) + 1)
      ∧ (∃ y : Fin 2 → Bool, custoCNF φ y = 0) := by
  refine ⟨phiFofa, (fun _ => false), ?_, ?_, ?_, ?_⟩
  · rw [o_canto_nao_e_luz]; decide
  · rw [o_eixo_da_morte_no_canto]; decide
  · rw [o_eixo_da_morte_no_canto]
    have hpar0 : paridade2 (fun _ => false : Fin 2 → Bool) (fun _ => true) = 0 := by decide
    rw [hpar0]
    exact a_morte_oposta_no_canto_mata_a_luz
  · exact ⟨(fun _ => true), a_luz_existe_na_fofa⟩

end TGLExt.AParedeDoLadoOposto

