import TGLExt.AParedeDoLadoOposto

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A cartografia da parede: duas regras locais medidas, e o lado útil existe

## O que é

Depois da parede da máscara cheia (`AParedeDoLadoOposto`), esta pedra mede a regra que registra o eixo de um só literal da
cláusula sem par, e mostra que, havendo solução que serve o espectro, algum lado útil sempre existe em todo eixo.

## O que fica provado [REAL]

1. ★ `a_luz_sempre_tem_um_lado`: havendo solução que serve o espectro, para todo eixo existe uma paridade cuja morte é útil
   (a paridade dessa solução).
2. `a_medida_o_literal_guarda_na_fofa`: em `phiFofa`, o eixo da variável 0 com a paridade 1 é útil.
3. `phiFofa2` [DEF], de três variáveis, com a única solução (V, V, F) (`a_luz_existe_na_fofa2`); o canto falha uma cláusula
   (`o_canto_nao_e_luz_2`); ★ toda solução lê zero no eixo da variável 2, a do primeiro literal da cláusula sem par no canto
   (`a_luz_toda_fora_do_primeiro_literal`), e a morte com a paridade 1 nesse eixo não é útil
   (`a_morte_do_primeiro_literal_mata_a_luz_2`).
4. ★★ `a_medida_o_laco_cala_na_fofa2`: a partir do canto, com combustível 4, o laço final responde em silêncio, embora a
   fórmula tenha solução.
5. ★★ `a_parede_do_literal_unico`: existe fórmula com solução, ponto de custo não nulo e eixo de um só literal, fora do span,
   em que a morte com a paridade oposta não é útil.

## O que NÃO se afirma

* `a_luz_sempre_tem_um_lado` escolhe o lado pela própria solução; uma regra que escolha o lado útil sem conhecer a solução
  não é dada; segue [OPEN].
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ACartografiaDaParede

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto

variable {n : ℕ}

/-- ★ **O TEOREMA DO LADO CERTO**: havendo luz servida, para TODO eixo existe um valor que guarda luz —
a paridade da própria luz. A existência é trivial e construtiva; o conteúdo do gate é a LEITURA: computar
o lado do campo, sem visitar a luz. -/
theorem a_luz_sempre_tem_um_lado {n : ℕ} (φ : CNF n) (S : Espectro2 n) (a : Fin n → Bool)
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :
    ∃ v : ZMod 2, morteUtil φ S a v := by
  obtain ⟨y, hy, hyserv⟩ := hluz
  exact ⟨paridade2 y a, y, hy, hyserv, rfl⟩

/-- [MEDIDA] A máscara de uma variável: o eixo de um literal só. -/
def mascaraX0 : Fin 2 → Bool := fun i => decide (i = 0)

/-- Medida positiva: na fofa, o eixo da variável 0 no lado oposto ao canto GUARDA a luz. (A regra do
literal único funciona aí — a parede dela é outra fórmula.) -/
theorem a_medida_o_literal_guarda_na_fofa : morteUtil phiFofa [] mascaraX0 (1 : ZMod 2) :=
  ⟨(fun _ => true), a_luz_existe_na_fofa, rfl, by decide⟩

/-- [MEDIDA] A fórmula-armadilha 2: luz única no ponto das duas primeiras acesas; o canto falha só a
primeira cláusula; todos os vizinhos do canto sem melhora; e a primeira literal da cláusula morta é x₂ —
que a luz NÃO flipa. -/
def phiFofa2 : CNF 3 := [[⟨2, true⟩, ⟨0, true⟩, ⟨1, true⟩], [⟨2, false⟩], [⟨0, false⟩, ⟨1, true⟩],
  [⟨0, true⟩, ⟨1, false⟩]]

/-- A máscara da primeira literal da cláusula morta da armadilha 2. -/
def mascaraX2 : Fin 3 → Bool := fun i => decide (i = 2)

/-- A luz da armadilha 2: as duas primeiras acesas. -/
def luzFofa2 : Fin 3 → Bool := fun i => decide (i = 0) || decide (i = 1)

/-- A luz existe na armadilha 2. -/
theorem a_luz_existe_na_fofa2 : custoCNF phiFofa2 luzFofa2 = 0 := by decide

/-- O canto da armadilha 2 falha exatamente uma cláusula. -/
theorem o_canto_nao_e_luz_2 : custoCNF phiFofa2 (fun _ => false) = 1 := by decide

/-- ★ **TODA A LUZ FORA DO PRIMEIRO LITERAL**: qualquer solução da armadilha 2 lê zero na máscara de
x₂ — o lado oposto ao poço, nesse eixo, está vazio de luz. -/
theorem a_luz_toda_fora_do_primeiro_literal :
    ∀ y : Fin 3 → Bool, custoCNF phiFofa2 y = 0 → paridade2 y mascaraX2 = 0 := by
  decide

/-- ★ A morte no lado oposto, no eixo da primeira literal, não guarda luz nenhuma na armadilha 2. -/
theorem a_morte_do_primeiro_literal_mata_a_luz_2 :
    ¬ morteUtil phiFofa2 [] mascaraX2 (1 : ZMod 2) := by
  intro h
  obtain ⟨y, hy, -, hpar⟩ := h
  have h0 := a_luz_toda_fora_do_primeiro_literal y hy
  rw [h0] at hpar
  exact (by decide : (0 : ZMod 2) ≠ 1) hpar

/-- ★★ **A MEDIDA DO PROCESSO NA ARMADILHA 2**: do canto, com fuel de sobra, o laço final responde
SILÊNCIO — com a luz existente. -/
theorem a_medida_o_laco_cala_na_fofa2 :
    (lacoFinal phiFofa2 4 [] (fun _ => false)).1 = none := by
  decide

/-- ★★ **A PAREDE DO LITERAL ÚNICO**: existe configuração — eixo de uma variável só, poço sem par,
transversal ao espectro, luz existente — em que a morte no lado oposto não guarda luz nenhuma. A regra do
literal único também não realiza o gate. -/
theorem a_parede_do_literal_unico :
    ∃ (φ : CNF 3) (w a : Fin 3 → Bool),
      (∃ i : Fin 3, a = fun j => decide (j = i))
      ∧ custoCNF φ w ≠ 0
      ∧ ¬ eixoNoSpanB (([] : Espectro2 3).map Prod.fst) a = true
      ∧ ¬ morteUtil φ [] a (paridade2 w a + 1)
      ∧ (∃ y : Fin 3 → Bool, custoCNF φ y = 0) := by
  refine ⟨phiFofa2, (fun _ => false), mascaraX2, ⟨2, rfl⟩, ?_, ?_, ?_,
    ⟨luzFofa2, a_luz_existe_na_fofa2⟩⟩
  · rw [o_canto_nao_e_luz_2]; decide
  · decide
  · have hpar0 : paridade2 (fun _ => false : Fin 3 → Bool) mascaraX2 = 0 := by decide
    rw [hpar0]
    exact a_morte_do_primeiro_literal_mata_a_luz_2

end TGLExt.ACartografiaDaParede

