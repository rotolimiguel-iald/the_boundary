import TGLExt.AMaquinaVerificaCorrida

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# As medidas da fita e a cota da corrida

## O que fica provado [REAL]

1. `aSomaDosIndices`, `aSomaDosLiterais` [DEF]: a soma dos índices (com um por literal) e o número de literais da fórmula.
2. ★ `codificaLiteral_length`, `codificaClausula_length`, `codificaFormula_length`, `codificaEntrada_length`,
   `flatMap_codificaClausula_length`, `codificaFormula_length_reverse`: os comprimentos das codificações na fita.
3. ★ `litNaPilha_length`, `flatMap_litNaPilha_length`, `flatMap_clausulaNaPilha_length`, `pilhaDaCorrida_length_le`: os comprimentos
   na pilha; a pilha da corrida não é maior que a fórmula codificada.
4. ★ `sum_map_cotaLit`, `aSomaDosLiterais_le_indices`, `aSomaDosIndices_reverse`, `aSomaDosLiterais_reverse` (auxiliares da
   contabilidade).
5. ★★ `cotaBloco_le` e `cotaCorrida_le`: as cotas do bloco e da corrida em termos da soma dos índices, do número de literais e do
   comprimento da testemunha.
6. ★★ `a_cota_da_corrida_em_L`: com L o comprimento da entrada codificada, a corrida custa no máximo L² + 13L passos.

## O que NÃO se afirma

* Uma cota de passos para decidir SAT: as medidas contam os passos da verificação de uma testemunha dada.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMaquinaVerifica

open TGLExt TGLExt.TraducaoTuring TGLExt.APonteAoTM2 TGLExt.ATensaoMinima Turing.TM2.Stmt Turing
open TGLExt.AVerificadoraTM2

/-- [DEF] A soma dos índices com o custo de um por literal. -/
def aSomaDosIndices (φf : List (List (ℕ × Bool))) : ℕ :=
  (φf.map (fun C => (C.map (fun p => p.1 + 1)).sum)).sum

/-- [DEF] A soma dos literais. -/
def aSomaDosLiterais (φf : List (List (ℕ × Bool))) : ℕ := (φf.map List.length).sum

/-- ★ **A MEDIDA DO LITERAL NA FITA**. -/
theorem codificaLiteral_length (p : ℕ × Bool) :
    (codificaLiteral p).length = p.1 + 1 := by
  rw [codificaLiteral, List.length_cons, List.length_replicate]

/-- ★ **A MEDIDA DA CLÁUSULA NA FITA**. -/
theorem codificaClausula_length (C : List (ℕ × Bool)) :
    (codificaClausula C).length = (C.map (fun p => p.1 + 1)).sum + 1 := by
  rw [codificaClausula, List.length_append, List.length_flatMap]
  show (C.map (fun p => (codificaLiteral p).length)).sum + [Simbolo.fimC].length = _
  simp only [codificaLiteral_length, List.length_cons, List.length_nil]

/-- ★ **A MEDIDA DA FÓRMULA NA FITA**. -/
theorem codificaFormula_length (φf : List (List (ℕ × Bool))) :
    (codificaFormula φf).length = aSomaDosIndices φf + φf.length + 1 := by
  rw [codificaFormula, List.length_append, List.length_flatMap]
  simp only [codificaClausula_length, List.length_cons, List.length_nil]
  show (φf.map (fun C => (C.map (fun p => p.1 + 1)).sum + 1)).sum + 1
      = (φf.map (fun C => (C.map (fun p => p.1 + 1)).sum)).sum + φf.length + 1
  induction φf with
  | nil => rfl
  | cons C Cs ih =>
    rw [List.map_cons, List.sum_cons, List.map_cons, List.sum_cons, List.length_cons]
    omega

/-- ★ **A MEDIDA DA ENTRADA NA FITA**. -/
theorem codificaEntrada_length (φf : List (List (ℕ × Bool))) (bits : List Bool) :
    (codificaEntrada (φf, bits)).length = (codificaFormula φf).length + 1 + bits.length := by
  rw [codificaEntrada, List.length_append, List.length_append, List.length_cons, List.length_nil,
    codificaBits, List.length_map]

/-- ★ **A MEDIDA DO LITERAL NA PILHA**. -/
theorem litNaPilha_length (p : ℕ × Bool) : (litNaPilha p).length = p.1 + 1 := by
  rw [litNaPilha, List.length_append, List.length_replicate, List.length_cons, List.length_nil]

/-- ★ **A MEDIDA DO FLATMAP NA PILHA**. -/
theorem flatMap_litNaPilha_length (Ls : List (ℕ × Bool)) :
    (Ls.flatMap litNaPilha).length = (Ls.map (fun p => p.1 + 1)).sum := by
  rw [List.length_flatMap]
  show (Ls.map (fun p => (litNaPilha p).length)).sum = (Ls.map (fun p => p.1 + 1)).sum
  simp only [litNaPilha_length]

/-- ★ **A SOMA COM A CONSTANTE** (auxiliar da contabilidade). -/
theorem sum_map_cotaLit (w : List Simbolo) :
    ∀ (C : List (ℕ × Bool)),
      (C.map (cotaLit w)).sum = 3 * (C.map (fun p => p.1 + 1)).sum + C.length * (w.length + 4) := by
  intro C
  induction C with
  | nil => simp
  | cons p ps ih =>
    rw [List.map_cons, List.sum_cons, cotaLit, List.map_cons, List.sum_cons, List.length_cons,
      Nat.add_mul, Nat.one_mul, ih]
    omega

/-- ★★ **A COTA DO BLOCO EM TERMOS DOS ÍNDICES**. -/
theorem cotaBloco_le (w : List Simbolo) (C : List (ℕ × Bool)) :
    cotaBloco w C ≤ 4 * ((C.map (fun p => p.1 + 1)).sum) + C.length * (w.length + 6) + 2 := by
  simp only [cotaBloco]
  rw [sum_map_cotaLit, flatMap_litNaPilha_length]
  have hexp : C.length * (w.length + 6) = C.length * (w.length + 4) + 2 * C.length := by
    rw [show w.length + 6 = w.length + 4 + 2 from by omega, Nat.mul_add, Nat.mul_comm C.length 2]
  omega

/-- ★★ **A COTA DA CORRIDA EM TERMOS DOS ÍNDICES**. -/
theorem cotaCorrida_le (w : List Simbolo) :
    ∀ (φf : List (List (ℕ × Bool))),
      cotaCorrida w φf ≤ 4 * aSomaDosIndices φf + aSomaDosLiterais φf * (w.length + 6)
        + 3 * φf.length := by
  intro φf
  induction φf with
  | nil => simp [cotaCorrida, aSomaDosIndices, aSomaDosLiterais]
  | cons C Cs ih =>
    have hb := cotaBloco_le w C
    simp only [cotaCorrida, aSomaDosIndices, aSomaDosLiterais, List.map_cons, List.map_nil,
      List.sum_cons, List.length_cons] at hb ih ⊢
    rw [Nat.add_mul]
    omega

/-- ★ **OS LITERAIS SÃO MENORES QUE OS ÍNDICES** (cada literal custa ao menos um símbolo). -/
theorem aSomaDosLiterais_le_indices (φf : List (List (ℕ × Bool))) :
    aSomaDosLiterais φf ≤ aSomaDosIndices φf := by
  induction φf with
  | nil => rfl
  | cons C Cs ih =>
    simp only [aSomaDosLiterais, aSomaDosIndices, List.map_cons, List.sum_cons] at ih ⊢
    have hC : C.length ≤ (C.map (fun p => p.1 + 1)).sum := by
      induction C with
      | nil => rfl
      | cons p ps ih2 => rw [List.map_cons, List.sum_cons, List.length_cons]; omega
    omega

/-- ★ **A SOMA DOS ÍNDICES NÃO VÊ A ORDEM** (auxiliar da contabilidade). -/
theorem aSomaDosIndices_reverse (Cs : List (List (ℕ × Bool))) :
    aSomaDosIndices Cs.reverse = aSomaDosIndices Cs := by
  simp only [aSomaDosIndices, List.map_reverse, List.sum_reverse]

/-- ★ **A SOMA DOS LITERAIS NÃO VÊ A ORDEM** (auxiliar da contabilidade). -/
theorem aSomaDosLiterais_reverse (Cs : List (List (ℕ × Bool))) :
    aSomaDosLiterais Cs.reverse = aSomaDosLiterais Cs := by
  simp only [aSomaDosLiterais, List.map_reverse, List.sum_reverse]

/-- ★ **A MEDIDA DO FLATMAP DAS CLÁUSULAS NA FITA** (auxiliar). -/
theorem flatMap_codificaClausula_length (Cs : List (List (ℕ × Bool))) :
    (Cs.flatMap codificaClausula).length = aSomaDosIndices Cs + Cs.length := by
  rw [List.length_flatMap]
  show (Cs.map (fun C => (codificaClausula C).length)).sum = aSomaDosIndices Cs + Cs.length
  induction Cs with
  | nil => rfl
  | cons C Cs' ih =>
    simp only [codificaClausula_length, aSomaDosIndices, List.map_cons, List.sum_cons,
      List.length_cons] at ih ⊢
    omega

/-- ★ **A MEDIDA DO FLATMAP DAS CLÁUSULAS NA PILHA** (auxiliar). -/
theorem flatMap_clausulaNaPilha_length (Cs : List (List (ℕ × Bool))) :
    (Cs.flatMap clausulaNaPilha).length = aSomaDosIndices Cs + Cs.length := by
  rw [List.length_flatMap]
  show (Cs.map (fun C => (clausulaNaPilha C).length)).sum = aSomaDosIndices Cs + Cs.length
  induction Cs with
  | nil => rfl
  | cons C Cs' ih =>
    simp only [clausulaNaPilha, List.length_cons, flatMap_litNaPilha_length, List.map_reverse,
      List.sum_reverse, aSomaDosIndices, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    omega

/-- ★ **A MEDIDA DA FÓRMULA NÃO VÊ A ORDEM** (auxiliar da contabilidade). -/
theorem codificaFormula_length_reverse (Cs : List (List (ℕ × Bool))) :
    (codificaFormula Cs.reverse).length = (codificaFormula Cs).length := by
  rw [codificaFormula_length, codificaFormula_length, aSomaDosIndices_reverse, List.length_reverse]

/-- ★★ **A MEDIDA DA PILHA DA CORRIDA**: a pilha não passa da fita da fórmula. -/
theorem pilhaDaCorrida_length_le (Cs : List (List (ℕ × Bool))) :
    (pilhaDaCorrida Cs).length ≤ (codificaFormula Cs).length := by
  cases Cs with
  | nil =>
    rw [pilhaDaCorrida, codificaFormula]
    simp
  | cons C Cs' =>
    have h1 := flatMap_clausulaNaPilha_length Cs'
    have h2 := flatMap_codificaClausula_length Cs'
    have h3 := codificaClausula_length C
    rw [pilhaDaCorrida, codificaFormula, List.flatMap_cons, List.length_append, List.length_append,
      List.length_append, flatMap_litNaPilha_length, List.map_reverse, List.sum_reverse, h1, h2, h3]
    simp only [aSomaDosIndices, List.map_cons, List.sum_cons, List.length_cons, List.length_nil]
    omega

/-- ★★ **A COTA DA CORRIDA É QUADRÁTICA NA FITA**: com L o comprimento da entrada codificada,
a corrida custa no máximo L² + 13L. -/
theorem a_cota_da_corrida_em_L (φf : List (List (ℕ × Bool))) (bits : List Bool) :
    cotaCorrida (codificaBits bits) φf
      ≤ (codificaEntrada (φf, bits)).length * (codificaEntrada (φf, bits)).length
        + 13 * (codificaEntrada (φf, bits)).length := by
  have hL := cotaCorrida_le (codificaBits bits) φf
  have h2 := aSomaDosLiterais_le_indices φf
  have h3 : (codificaBits bits).length = bits.length := by rw [codificaBits, List.length_map]
  have h4 : aSomaDosIndices φf ≤ (codificaEntrada (φf, bits)).length := by
    rw [codificaEntrada_length, codificaFormula_length]
    omega
  have h5 : bits.length ≤ (codificaEntrada (φf, bits)).length := by
    rw [codificaEntrada_length]
    omega
  have h6 : φf.length ≤ (codificaEntrada (φf, bits)).length := by
    rw [codificaEntrada_length, codificaFormula_length]
    omega
  nlinarith [hL, h2, h3, h4, h5, h6]

end TGLExt.AMaquinaVerifica

