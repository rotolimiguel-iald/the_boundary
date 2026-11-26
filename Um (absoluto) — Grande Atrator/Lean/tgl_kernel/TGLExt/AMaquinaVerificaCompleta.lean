import TGLExt.AMaquinaVerificaCorrida
import TGLExt.AMaquinaVerificaMedidas

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A máquina verifica: a correção geral da verificadora TM2, com a cota polinomial

## O que fica provado [REAL]

1. ★ `a_config_inicial`: a configuração inicial, por nome: a entrada na pilha de entrada, as demais vazias.
2. ★ `codificaBits_ne_sep`, `reverse_codificaLiteral`, `reverse_flatMap`, `reverse_codificaClausula`, `flatMap_clausula_cons`
   (auxiliares) e `a_formula_na_pilha`: a fórmula codificada, invertida, é a pilha da corrida.
3. ★ `o_bit_da_testemunha`: o símbolo na posição da variável é «um» se e só se o bit é verdadeiro: a leitura da máquina é a leitura
   da casa.
4. ★ `o_presenca_da_testemunha`: há símbolo na posição da variável se e só se há o bit: a guarda da presença na língua da casa.
5. ★ `a_condicao_e_a_da_fita`, `a_clausula_e_a_da_fita`, `a_densidade_da_fita`: para todo índice, o casamento da máquina é o da casa,
   e o veredito é o de `verificaNaFita`.
6. ★★★ `a_maquina_verifica`: para toda fórmula e toda testemunha, existe combustível de no máximo L² + 20L + 10 passos, sendo L o
   comprimento da entrada codificada, com o qual a verificadora TM2 para na configuração final limpa, com o veredito de
   `verificaNaFita` na saída.

## O que NÃO se afirma

* Esta máquina recebe a leitura, não a procura: não decide SAT sem testemunha.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMaquinaVerifica

open TGLExt TGLExt.TraducaoTuring TGLExt.APonteAoTM2 TGLExt.ATensaoMinima Turing.TM2.Stmt Turing
open TGLExt.AVerificadoraTM2

/-- ★ **A CONFIGURAÇÃO INICIAL, POR NOME**: a entrada na pilha de entrada, as demais vazias. -/
theorem a_config_inicial (e : List Simbolo) :
    Turing.initList aVerificadora e
      = aConfig (some Fase.copiaEntrada) ⟨Fase.copiaEntrada, false, false, none, none⟩
          e [] [] [] [] [] := by
  simp only [Turing.initList, aConfig]
  congr 1
  funext k
  cases k <;> rfl

/-- ★ **A TESTEMUNHA NÃO TEM DIVISÓRIA** (auxiliar da colheita). -/
theorem codificaBits_ne_sep (bits : List Bool) : ∀ s₀ ∈ codificaBits bits, s₀ ≠ Simbolo.sep := by
  intro s₀ hs
  rw [codificaBits, List.mem_map] at hs
  obtain ⟨b, _, rfl⟩ := hs
  cases b <;> decide

/-- ★ **O LITERAL INVERTIDO** (auxiliar da pilha). -/
theorem reverse_codificaLiteral (p : ℕ × Bool) :
    (codificaLiteral p).reverse = litNaPilha p := by
  rw [codificaLiteral, List.reverse_cons, List.reverse_replicate]
  rfl

/-- ★ **O FLATMAP INVERTIDO** (auxiliar da pilha). -/
theorem reverse_flatMap {α β : Type} (f : α → List β) :
    ∀ l : List α, (l.flatMap f).reverse = l.reverse.flatMap (fun a => (f a).reverse) := by
  intro l
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.flatMap_cons, List.reverse_append, List.reverse_cons, ih, List.flatMap_append]
    show l.reverse.flatMap (fun a => (f a).reverse) ++ (f a).reverse
      = l.reverse.flatMap (fun a => (f a).reverse) ++ ((f a).reverse ++ [])
    rw [List.append_nil]

/-- ★ **A CLÁUSULA INVERTIDA É A CLÁUSULA NA PILHA** (auxiliar). -/
theorem reverse_codificaClausula (C : List (ℕ × Bool)) :
    (codificaClausula C).reverse = clausulaNaPilha C := by
  rw [codificaClausula, List.reverse_append, reverse_flatMap, clausulaNaPilha]
  show (Simbolo.fimC :: []) ++ C.reverse.flatMap (fun p => (codificaLiteral p).reverse)
      = Simbolo.fimC :: C.reverse.flatMap litNaPilha
  simp only [reverse_codificaLiteral]
  rfl

/-- ★ **A FÓRMULA INVERTIDA É A PILHA DA CORRIDA**: o terminador da fórmula à frente, as
cláusulas na ordem invertida, cada uma com o seu terminador. -/
theorem a_formula_na_pilha (φf : List (List (ℕ × Bool))) :
    (codificaFormula φf).reverse
      = Simbolo.fimF :: φf.reverse.flatMap clausulaNaPilha := by
  rw [codificaFormula, List.reverse_append, reverse_flatMap]
  show (Simbolo.fimF :: []) ++ φf.reverse.flatMap (fun C => (codificaClausula C).reverse)
      = Simbolo.fimF :: φf.reverse.flatMap clausulaNaPilha
  simp only [reverse_codificaClausula]
  rfl

/-- ★ **A PILHA NÃO-VAZIA ABRE COM O TERMINADOR** (auxiliar do início). -/
theorem flatMap_clausula_cons (L : List (List (ℕ × Bool))) (hL : L ≠ []) :
    L.flatMap clausulaNaPilha = Simbolo.fimC :: pilhaDaCorrida L := by
  cases L with
  | nil => exact absurd rfl hL
  | cons D Ds =>
    rw [List.flatMap_cons, clausulaNaPilha]
    rfl

/-- ★ **O BIT DA TESTEMUNHA**: o símbolo na posição da variável é «um» se, e só se, o bit é
verdadeiro — a leitura da máquina é a leitura da casa. -/
theorem o_bit_da_testemunha (bits : List Bool) :
    ∀ (i : ℕ), (((codificaBits bits).drop i).head? == some Simbolo.um)
      = (bits[i]? == some true) := by
  induction bits with
  | nil =>
    intro i
    cases i <;> rfl
  | cons b bs ih =>
    intro i
    cases i with
    | zero => cases b <;> rfl
    | succ i => exact ih i

/-- ★ **A PRESENÇA DA TESTEMUNHA**: há símbolo na posição da variável se, e só se, há o bit — a
guarda da presença na língua da casa. -/
theorem o_presenca_da_testemunha (bits : List Bool) :
    ∀ (i : ℕ), ((codificaBits bits).drop i).head?.isSome = (bits[i]?).isSome := by
  induction bits with
  | nil =>
    intro i
    cases i <;> rfl
  | cons b bs ih =>
    intro i
    cases i with
    | zero => rfl
    | succ i => exact ih i

/-- ★ **A CONDIÇÃO É A DA FITA**: o casamento da máquina (a guarda da presença, o símbolo e a
polaridade) é o casamento da casa (o bit na posição) — para todo índice, sem alcance hipotecado. -/
theorem a_condicao_e_a_da_fita (bits : List Bool) (p : ℕ × Bool) :
    (((codificaBits bits).drop p.1).head?.isSome &&
      (((codificaBits bits).drop p.1).head? == some Simbolo.um) == p.2)
      = (bits[p.1]? == some p.2) := by
  rw [o_bit_da_testemunha, o_presenca_da_testemunha]
  cases hb : bits[p.1]? with
  | none => rfl
  | some b => cases b <;> cases p.2 <;> rfl

/-- ★ **A CLÁUSULA É A DA FITA** (auxiliar do veredito). -/
theorem a_clausula_e_a_da_fita (bits : List Bool) :
    ∀ (C : List (ℕ × Bool)),
      C.any (fun p => (bits[p.1]? == some p.2))
        = C.any (fun p => ((codificaBits bits).drop p.1).head?.isSome &&
            (((codificaBits bits).drop p.1).head? == some Simbolo.um) == p.2) := by
  intro C
  induction C with
  | nil => rfl
  | cons p ps ih =>
    rw [List.any_cons, List.any_cons, a_condicao_e_a_da_fita bits p, ih]

/-- ★ **O VEREDITO É O DA RÉGUA**: a `verificaNaFita` na língua da máquina. -/
theorem a_densidade_da_fita (φf : List (List (ℕ × Bool))) (bits : List Bool) :
    verificaNaFita φf bits
      = φf.all (fun C => C.any (fun p =>
          ((codificaBits bits).drop p.1).head?.isSome &&
            (((codificaBits bits).drop p.1).head? == some Simbolo.um) == p.2)) := by
  induction φf with
  | nil => rfl
  | cons C Cs ih =>
    simp only [verificaNaFita, List.all_cons]
    rw [a_clausula_e_a_da_fita bits C,
      show Cs.all (fun C' => C'.any (fun p => (bits[p.1]? == some p.2)))
        = Cs.all (fun C' => C'.any (fun p =>
            ((codificaBits bits).drop p.1).head?.isSome &&
              (((codificaBits bits).drop p.1).head? == some Simbolo.um) == p.2)) from ih]

/-- ★★★ **A MÁQUINA VERIFICA, COM A COTA E A CASA EM ORDEM**: para toda fórmula e toda
testemunha — todo par, sem alcance hipotecado — existe combustível polinomial com o qual a
verificadora TM2 para na configuração final limpa, com o veredito da casa na saída. -/
theorem a_maquina_verifica (φf : List (List (ℕ × Bool))) (bits : List Bool) :
    ∃ fuel : ℕ, fuel ≤ (codificaEntrada (φf, bits)).length * (codificaEntrada (φf, bits)).length
        + 20 * (codificaEntrada (φf, bits)).length + 10 ∧
      correr aVerificadora (Turing.initList aVerificadora (codificaEntrada (φf, bits))) fuel
        = some (aConfig none ⟨Fase.copiaEntrada, false, false, none, none⟩ [] [] [] [] []
            [verificaNaFita φf bits]) := by
  have hentrada : (codificaEntrada (φf, bits)).reverse
      = (codificaBits bits).reverse ++ (Simbolo.sep :: (codificaFormula φf).reverse) := by
    rw [codificaEntrada, List.reverse_append, List.reverse_append]
    rfl
  cases φf with
  | nil =>
    refine ⟨(codificaEntrada ([], bits)).length + 1 +
      (((codificaBits bits).reverse).length + 1 + (3 + ((codificaBits bits).length + 4))), ?_, ?_⟩
    · have hL : 2 ≤ (codificaEntrada ([], bits)).length := by
        rw [codificaEntrada_length, codificaFormula_length]
        simp [aSomaDosIndices]
      have hB : bits.length ≤ (codificaEntrada ([], bits)).length := by
        rw [codificaEntrada_length]
        omega
      have hcb : (codificaBits bits).length = bits.length := by rw [codificaBits, List.length_map]
      rw [List.length_reverse, hcb]
      nlinarith
    rw [a_config_inicial, correr_add, copiaEntrada_fecha]
    simp only [Option.bind_some]
    rw [List.append_nil, hentrada, correr_add,
      colheTestemunha_fecha _ _ _ _ _ _ _
        (fun s₀ hs => codificaBits_ne_sep bits s₀ (List.mem_reverse.1 hs))]
    simp only [Option.bind_some]
    rw [List.reverse_reverse, List.append_nil, a_formula_na_pilha]
    show correr aVerificadora
        (aConfig (some Fase.inicioClausula)
          (⟨Fase.copiaEntrada, false, false, some Simbolo.sep, none⟩ : EstadoV) []
          [Simbolo.fimF] (codificaBits bits) [] [] [])
        (3 + ((codificaBits bits).length + 4)) = _
    rw [correr_add, inicioClausula_vazio_fecha]
    simp only [Option.bind_some]
    have h0 : ([] : List Simbolo).length = 0 := rfl
    rw [show (codificaBits bits).length + 4
        = 1 + (([] : List Simbolo).length + 1) + ((codificaBits bits).length + 1) + 1 from by
        rw [h0]; omega, a_limpeza_fecha]
    rfl
  | cons C Cs =>
    have hne : (C :: Cs).reverse ≠ [] := by
      rw [List.reverse_cons]
      simp
    rcases hrev : (C :: Cs).reverse with _ | ⟨D, Ds⟩
    · rw [List.reverse_eq_nil_iff] at hrev
      nomatch hrev
    · obtain ⟨fuel₃, v', fim, hfim, hfuel₃, hEq⟩ := a_corrida_verifica
        { { (⟨Fase.copiaEntrada, false, false, none, none⟩ : EstadoV) with
            lido := some Simbolo.sep } with lido := some Simbolo.fimC, falha := true }
        (codificaBits bits) [] D Ds
        rfl
      have hverd : (if (D :: Ds).all (fun C' => C'.any (fun p =>
              ((codificaBits bits).drop p.1).head?.isSome &&
                (((codificaBits bits).drop p.1).head? == some Simbolo.um) == p.2))
            then true else false) = verificaNaFita (C :: Cs) bits := by
        rw [← hrev, List.all_reverse, ← a_densidade_da_fita]
        cases verificaNaFita (C :: Cs) bits <;> rfl
      refine ⟨(codificaEntrada (C :: Cs, bits)).length + 1 +
        (((codificaBits bits).reverse).length + 1 +
          (2 + (fuel₃ + (1 + (fim.length + 1) + ((codificaBits bits).length + 1) + 1)))), ?_, ?_⟩
      · have hf3 : fuel₃ ≤ (codificaEntrada (C :: Cs, bits)).length
            * (codificaEntrada (C :: Cs, bits)).length
            + 13 * (codificaEntrada (C :: Cs, bits)).length := by
          rw [← hrev] at hfuel₃
          rw [cotaCorrida_reverse] at hfuel₃
          exact hfuel₃.trans (a_cota_da_corrida_em_L (C :: Cs) bits)
        have hfm : fim.length ≤ (codificaFormula (C :: Cs)).length := by
          rw [← hrev] at hfim
          exact hfim.trans ((pilhaDaCorrida_length_le _).trans (by
            rw [codificaFormula_length_reverse]))
        have hB : bits.length ≤ (codificaEntrada (C :: Cs, bits)).length := by
          rw [codificaEntrada_length]
          omega
        have hF : (codificaFormula (C :: Cs)).length
            ≤ (codificaEntrada (C :: Cs, bits)).length := by
          rw [codificaEntrada_length]
          omega
        have hrevlen : ((codificaBits bits).reverse).length = bits.length := by
          rw [List.length_reverse, codificaBits, List.length_map]
        have hcb : (codificaBits bits).length = bits.length := by rw [codificaBits, List.length_map]
        rw [hrevlen, hcb]
        nlinarith
      rw [a_config_inicial, correr_add, copiaEntrada_fecha]
      simp only [Option.bind_some]
      rw [List.append_nil, hentrada, correr_add,
        colheTestemunha_fecha _ _ _ _ _ _ _
          (fun s₀ hs => codificaBits_ne_sep bits s₀ (List.mem_reverse.1 hs))]
      simp only [Option.bind_some]
      rw [List.reverse_reverse, List.append_nil, a_formula_na_pilha,
        flatMap_clausula_cons _ hne, correr_add, inicioClausula_fecha]
      simp only [Option.bind_some]
      rw [hrev, correr_add, hEq]
      simp only [Option.bind_some]
      rw [a_limpeza_fecha, hverd]

end TGLExt.AMaquinaVerifica

