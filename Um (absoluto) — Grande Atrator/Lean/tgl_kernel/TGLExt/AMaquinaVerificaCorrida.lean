import TGLExt.AMaquinaVerificaFinal

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A máquina verifica: o bloco da cláusula e a corrida, com as cotas de passos

## O que fica provado [REAL]

1. `litNaPilha`, `clausulaNaPilha`, `pilhaDaCorrida` [DEF]: o literal, a cláusula e a corrida na pilha (a fita invertida).
2. `cotaLit`, `cotaBloco`, `cotaCorrida` [DEF]: as cotas de passos do literal, da cláusula e da corrida; ★ `cotaBloco_reverse` e
   `cotaCorrida_reverse`: as cotas não dependem da ordem.
3. ★ `litNaPilha_ne_fimC` e `flatMap_litNaPilha_ne_fimC` (auxiliares do salto): o conteúdo de uma cláusula na pilha não tem terminador.
4. ★★ `o_bloco_casa_medio`: com a falha armada e o resto da pilha depois do terminador, os literais processam-se, com qualquer índice,
   dentro do alcance ou além, em no máximo `cotaBloco` passos; o casamento desarma a falha, salta o resto da cláusula e reabre o
   ciclo; o descasamento de todos os literais escreve o veredito falso.
5. ★★ `o_bloco_casa_ultimo`: no fim da pilha, o casamento desarma a falha e o salto encontra o fim: a máquina escreve o veredito
   verdadeiro.
6. ★★★ `a_corrida_verifica`: a corrida decide a lista das cláusulas, em no máximo `cotaCorrida` passos: a máquina chega à parada com o
   veredito verdadeiro se todas as cláusulas têm casamento, e com o falso senão.

## O que NÃO se afirma

* A correção geral e o tempo da verificadora, para toda fórmula e toda testemunha: estão em `AMaquinaVerificaCompleta`
  (`a_maquina_verifica`, com combustível de no máximo L² + 20L + 10, sendo L o comprimento da entrada codificada) e em
  `AMaquinaVerificaHabitante` (`a_verificadora_em_tempo_polinomial`).
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMaquinaVerifica

open TGLExt TGLExt.TraducaoTuring TGLExt.APonteAoTM2 TGLExt.ATensaoMinima Turing.TM2.Stmt Turing
open TGLExt.AVerificadoraTM2

/-- [DEF] O literal na pilha (a fita invertida): os ticks do índice e o marcador da polaridade. -/
def litNaPilha (p : ℕ × Bool) : List Simbolo :=
  List.replicate p.1 Simbolo.tick ++ [if p.2 then Simbolo.litT else Simbolo.litF]

/-- [DEF] A cláusula na pilha: o terminador à frente dos literais invertidos. -/
def clausulaNaPilha (C : List (ℕ × Bool)) : List Simbolo :=
  Simbolo.fimC :: C.reverse.flatMap litNaPilha

/-- [DEF] A pilha da corrida: a primeira cláusula já aberta, as demais com os terminadores. -/
def pilhaDaCorrida : List (List (ℕ × Bool)) → List Simbolo
  | [] => []
  | C :: Cs => C.reverse.flatMap litNaPilha ++ Cs.flatMap clausulaNaPilha

/-- [DEF] A cota do combustível do literal: linear no índice e na testemunha. -/
def cotaLit (w : List Simbolo) (p : ℕ × Bool) : ℕ := 3 * (p.1 + 1) + (w.length + 1) + 3

/-- [DEF] A cota do bloco: a soma dos literais, o salto e os terminadores. -/
def cotaBloco (w : List Simbolo) (Ls : List (ℕ × Bool)) : ℕ :=
  (Ls.map (cotaLit w)).sum + (Ls.flatMap litNaPilha).length + 2 * Ls.length + 2

/-- [DEF] A cota da corrida: a soma dos blocos e um passo por cláusula. -/
def cotaCorrida (w : List Simbolo) (Cs : List (List (ℕ × Bool))) : ℕ :=
  (Cs.map (cotaBloco w)).sum + Cs.length

/-- ★ **A COTA DO BLOCO NÃO VÊ A ORDEM** (auxiliar da contabilidade). -/
theorem cotaBloco_reverse (w : List Simbolo) (Ls : List (ℕ × Bool)) :
    cotaBloco w Ls.reverse = cotaBloco w Ls := by
  simp only [cotaBloco, List.map_reverse, List.sum_reverse, List.length_reverse,
    List.length_flatMap]

/-- ★ **A COTA DA CORRIDA NÃO VÊ A ORDEM** (auxiliar da contabilidade). -/
theorem cotaCorrida_reverse (w : List Simbolo) (Cs : List (List (ℕ × Bool))) :
    cotaCorrida w Cs.reverse = cotaCorrida w Cs := by
  simp only [cotaCorrida, List.map_reverse, List.sum_reverse, List.length_reverse]

/-- ★ **O LITERAL NA PILHA NÃO TEM TERMINADOR** (auxiliar do salto). -/
theorem litNaPilha_ne_fimC (p : ℕ × Bool) : ∀ s₀ ∈ litNaPilha p, s₀ ≠ Simbolo.fimC := by
  intro s₀ hs
  have h : s₀ = Simbolo.tick ∨ s₀ = (if p.2 then Simbolo.litT else Simbolo.litF) := by
    rw [litNaPilha, List.mem_append] at hs
    cases hs with
    | inl h => exact Or.inl (List.eq_of_mem_replicate h)
    | inr h => exact Or.inr (List.mem_singleton.1 h)
  rcases h with rfl | rfl
  · decide
  · cases p.2 <;> decide

/-- ★ **A CLÁUSULA NA PILHA NÃO TEM TERMINADOR NO CONTEÚDO** (auxiliar do salto). -/
theorem flatMap_litNaPilha_ne_fimC (Ls : List (ℕ × Bool)) :
    ∀ s₀ ∈ Ls.flatMap litNaPilha, s₀ ≠ Simbolo.fimC := by
  intro s₀ hs
  rw [List.mem_flatMap] at hs
  obtain ⟨p, _, h⟩ := hs
  exact litNaPilha_ne_fimC p s₀ h

/-- ★★ **O BLOCO DA CLÁUSULA, NO MEIO DA PILHA**: com a falha armada e o resto após o terminador,
os literais processam-se — todo índice, dentro ou além; o casamento desarma, salta o resto da
cláusula e reabre o ciclo no resto; o descasamento integral escreve a mentira com o terminador
intacto. -/
theorem o_bloco_casa_medio (v : EstadoV) (w resto' : List Simbolo) (out : List Bool) :
    ∀ (Ls : List (ℕ × Bool)), v.falha = true →
      ∃ fuel : ℕ, ∃ v' : EstadoV, fuel ≤ cotaBloco w Ls ∧
        (correr aVerificadora
          (aConfig (some Fase.leMarcador) v []
            (Ls.flatMap litNaPilha ++ Simbolo.fimC :: resto') w [] [] out) fuel
        = some (if Ls.any (fun p => ((w.drop p.1).head?.isSome &&
              (((w.drop p.1).head? == some Simbolo.um) == p.2))) then
            aConfig (some Fase.leMarcador) v' [] resto' w [] [] out
          else aConfig (some Fase.escreveMentira) v' [] (Simbolo.fimC :: resto') w [] [] out))
        ∧ (Ls.any (fun p => ((w.drop p.1).head?.isSome &&
              (((w.drop p.1).head? == some Simbolo.um) == p.2))) = true → v'.falha = true) := by
  intro Ls
  induction Ls generalizing v with
  | nil =>
    intro hfalha
    refine ⟨1, { v with lido := some Simbolo.fimC }, by simp [cotaBloco], ?_, ?_⟩
    · show correr aVerificadora (aConfig (some Fase.leMarcador) v [] (Simbolo.fimC :: resto')
          w [] [] out) 1 = _
      rw [correr, passo_leMarcador_fimC, hfalha, if_pos rfl]
      rfl
    · intro h
      simp at h
  | cons p Ls' ih =>
    intro hfalha
    obtain ⟨fuelLit, hfuelLit, hLit⟩ := o_literal_cobre v w
      (Ls'.flatMap litNaPilha ++ Simbolo.fimC :: resto') out p.1 p.2
    have hforma : (p :: Ls').flatMap litNaPilha ++ Simbolo.fimC :: resto'
        = List.replicate p.1 Simbolo.tick ++
          ((if p.2 then Simbolo.litT else Simbolo.litF) ::
            (Ls'.flatMap litNaPilha ++ Simbolo.fimC :: resto')) := by
      rw [List.flatMap_cons, List.append_assoc, litNaPilha, List.append_assoc]
      rfl
    rw [hforma]
    cases hcond : ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2)) with
    | true =>
      refine ⟨fuelLit + ((Ls'.flatMap litNaPilha).length + 1 + 1),
        (⟨Fase.procuraBit, true, p.2, some Simbolo.fimC, (w.drop p.1).head?⟩ : EstadoV), ?_, ?_, ?_⟩
      · simp only [cotaBloco, cotaLit, List.map_cons, List.sum_cons, List.flatMap_cons,
          List.length_append, List.length_cons]
        omega
      · rw [List.any_cons, hcond, Bool.true_or, if_pos rfl, correr_add, hLit]
        simp only [Option.bind_some]
        rw [hcond, if_pos rfl, correr_add,
          saltaLiterais_fecha ⟨Fase.procuraBit, false, p.2, none, (w.drop p.1).head?⟩ w [] [] out
            (Ls'.flatMap litNaPilha) (Simbolo.fimC :: resto') (flatMap_litNaPilha_ne_fimC Ls')]
        simp only [Option.bind_some]
        show correr aVerificadora (aConfig (some Fase.inicioClausula)
            { (⟨Fase.procuraBit, false, p.2, none, (w.drop p.1).head?⟩ : EstadoV) with
              lido := some Simbolo.fimC } []
            (Simbolo.fimC :: resto') w [] [] out) 1 = _
        rw [correr, passo_inicioClausula]
        rfl
      · intro _
        rfl
    | false =>
      obtain ⟨fuel', v', hfuel', hEq, hf'⟩ :=
        ih ⟨Fase.procuraBit, v.falha, p.2, none, (w.drop p.1).head?⟩ hfalha
      refine ⟨fuelLit + fuel', v', ?_, ?_, ?_⟩
      · simp only [cotaBloco, cotaLit, List.map_cons, List.sum_cons, List.flatMap_cons,
          List.length_append, List.length_cons] at hfuel' ⊢
        omega
      · rw [List.any_cons, hcond, Bool.false_or, correr_add, hLit]
        simp only [Option.bind_some]
        rw [hcond, if_neg (by decide)]
        exact hEq
      · rw [List.any_cons, hcond, Bool.false_or]
        exact hf'

/-- ★★ **O BLOCO DA CLÁUSULA, NO FIM DA PILHA**: com a falha armada e o vazio depois dos literais,
o casamento desarma e o salto encontra o fim — escreve a verdade; o descasamento integral escreve
a mentira. -/
theorem o_bloco_casa_ultimo (v : EstadoV) (w : List Simbolo) (out : List Bool) :
    ∀ (Ls : List (ℕ × Bool)), v.falha = true →
      ∃ fuel : ℕ, ∃ v' : EstadoV, fuel ≤ cotaBloco w Ls ∧
        correr aVerificadora
          (aConfig (some Fase.leMarcador) v [] (Ls.flatMap litNaPilha) w [] [] out) fuel
        = some (if Ls.any (fun p => ((w.drop p.1).head?.isSome &&
              (((w.drop p.1).head? == some Simbolo.um) == p.2))) then
            aConfig (some Fase.escreveVerdade) v' [] [] w [] [] out
          else aConfig (some Fase.escreveMentira) v' [] [] w [] [] out) := by
  intro Ls
  induction Ls generalizing v with
  | nil =>
    intro hfalha
    refine ⟨1, { v with lido := none }, by simp [cotaBloco], ?_⟩
    show correr aVerificadora (aConfig (some Fase.leMarcador) v [] [] w [] [] out) 1 = _
    rw [correr, passo_leMarcador_nil, hfalha, if_pos rfl]
    rfl
  | cons p Ls' ih =>
    intro hfalha
    obtain ⟨fuelLit, hfuelLit, hLit⟩ := o_literal_cobre v w (Ls'.flatMap litNaPilha ++ []) out p.1 p.2
    have hforma : (p :: Ls').flatMap litNaPilha
        = List.replicate p.1 Simbolo.tick ++
          ((if p.2 then Simbolo.litT else Simbolo.litF) :: (Ls'.flatMap litNaPilha ++ [])) := by
      rw [List.flatMap_cons, litNaPilha, List.append_assoc]
      show List.replicate p.1 Simbolo.tick ++
          ((if p.2 then Simbolo.litT else Simbolo.litF) :: Ls'.flatMap litNaPilha)
        = List.replicate p.1 Simbolo.tick ++
          ((if p.2 then Simbolo.litT else Simbolo.litF) :: (Ls'.flatMap litNaPilha ++ []))
      simp only [List.append_nil]
    rw [hforma]
    cases hcond : ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2)) with
    | true =>
      refine ⟨fuelLit + ((Ls'.flatMap litNaPilha).length + 1),
        { (⟨Fase.procuraBit, false, p.2, none, (w.drop p.1).head?⟩ : EstadoV) with
          lido := none }, ?_, ?_⟩
      · simp only [cotaBloco, cotaLit, List.map_cons, List.sum_cons, List.flatMap_cons,
          List.length_append, List.length_cons]
        omega
      rw [List.any_cons, hcond, Bool.true_or, if_pos rfl, correr_add, hLit]
      simp only [Option.bind_some]
      rw [hcond, if_pos rfl,
        saltaLiterais_fecha ⟨Fase.procuraBit, false, p.2, none, (w.drop p.1).head?⟩ w [] [] out
          (Ls'.flatMap litNaPilha) ([] : List Simbolo) (flatMap_litNaPilha_ne_fimC Ls')]
    | false =>
      obtain ⟨fuel', v', hfuel', hEq⟩ :=
        ih ⟨Fase.procuraBit, v.falha, p.2, none, (w.drop p.1).head?⟩ hfalha
      refine ⟨fuelLit + fuel', v', ?_, ?_⟩
      · simp only [cotaBloco, cotaLit, List.map_cons, List.sum_cons, List.flatMap_cons,
          List.length_append, List.length_cons] at hfuel' ⊢
        omega
      rw [List.any_cons, hcond, Bool.false_or, correr_add, hLit]
      simp only [Option.bind_some]
      rw [hcond, if_neg (by decide), List.append_nil]
      exact hEq

/-- ★★★ **A CORRIDA VERIFICA**: a lista das cláusulas decide-se; a máquina para com a saída —
a luz se todas as cláusulas têm o casamento, a ausência senão. -/
theorem a_corrida_verifica (v : EstadoV) (w : List Simbolo) (out : List Bool) :
    ∀ (C : List (ℕ × Bool)) (Cs : List (List (ℕ × Bool))), v.falha = true →
      ∃ fuel : ℕ, ∃ v' : EstadoV, ∃ fim : List Simbolo,
        fim.length ≤ (pilhaDaCorrida (C :: Cs)).length ∧ fuel ≤ cotaCorrida w (C :: Cs) ∧
        correr aVerificadora
          (aConfig (some Fase.leMarcador) v [] (pilhaDaCorrida (C :: Cs)) w [] [] out) fuel
        = some (aConfig (some Fase.parada) v' [] fim w [] []
            ((if (C :: Cs).all (fun C' => C'.any (fun p =>
                ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2))))
              then true else false) :: out)) := by
  intro C Cs
  induction Cs generalizing C v with
  | nil =>
    intro hfalha
    have hpilha : pilhaDaCorrida [C] = C.reverse.flatMap litNaPilha := by
      rw [pilhaDaCorrida]
      show C.reverse.flatMap litNaPilha ++ [] = C.reverse.flatMap litNaPilha
      rw [List.append_nil]
    rw [hpilha]
    obtain ⟨fuel', v', hfuel', hEq⟩ := o_bloco_casa_ultimo v w out C.reverse hfalha
    cases hany : (C.reverse.any (fun p => ((w.drop p.1).head?.isSome &&
        (((w.drop p.1).head? == some Simbolo.um) == p.2)))) with
    | true =>
      have hverd : [C].all (fun C' => C'.any (fun p =>
          ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2))))
          = true := by
        simp only [List.all_cons]
        rw [← List.any_reverse, hany, Bool.true_and]
        rfl
      refine ⟨fuel' + 1, v', [], by simp [pilhaDaCorrida], ?_, ?_⟩
      · rw [cotaBloco_reverse] at hfuel'
        simp only [cotaCorrida, cotaBloco, List.map_cons, List.map_nil, List.sum_cons,
          List.sum_nil, List.length_cons, List.length_nil] at hfuel' ⊢
        omega
      · rw [correr_add, hEq]
        simp only [Option.bind_some]
        rw [hany, if_pos rfl, correr, passo_escreveVerdade, hverd, if_pos rfl]
        rfl
    | false =>
      have hverd : [C].all (fun C' => C'.any (fun p =>
          ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2))))
          = false := by
        simp only [List.all_cons]
        rw [← List.any_reverse, hany, Bool.false_and]
      refine ⟨fuel' + 1, v', [], by simp [pilhaDaCorrida], ?_, ?_⟩
      · rw [cotaBloco_reverse] at hfuel'
        simp only [cotaCorrida, cotaBloco, List.map_cons, List.map_nil, List.sum_cons,
          List.sum_nil, List.length_cons, List.length_nil] at hfuel' ⊢
        omega
      · rw [correr_add, hEq]
        simp only [Option.bind_some]
        rw [hany, if_neg (by decide), correr, passo_escreveMentira, hverd, if_neg (by decide)]
        rfl
  | cons C₂ Cs' ih =>
    intro hfalha
    have hpilha : pilhaDaCorrida (C :: C₂ :: Cs')
        = C.reverse.flatMap litNaPilha ++ (Simbolo.fimC :: pilhaDaCorrida (C₂ :: Cs')) := by
      rw [pilhaDaCorrida, List.flatMap_cons, clausulaNaPilha]
      rfl
    rw [hpilha]
    obtain ⟨fuel', v', hfuel', hEq, hf'⟩ := o_bloco_casa_medio v w (pilhaDaCorrida (C₂ :: Cs'))
      out C.reverse hfalha
    cases hany : (C.reverse.any (fun p => ((w.drop p.1).head?.isSome &&
        (((w.drop p.1).head? == some Simbolo.um) == p.2)))) with
    | true =>
      obtain ⟨fuel'', v'', fim, hfim, hfuel'', hEq''⟩ := ih v' C₂ (hf' hany)
      have hverd : (C :: C₂ :: Cs').all (fun C' => C'.any (fun p =>
          ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2))))
          = (C₂ :: Cs').all (fun C' => C'.any (fun p =>
            ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2)))) := by
        simp only [List.all_cons]
        rw [← List.any_reverse, hany, Bool.true_and]
      refine ⟨fuel' + fuel'', v'', fim, ?_, ?_, ?_⟩
      · simp only [pilhaDaCorrida, List.flatMap_cons, clausulaNaPilha, List.length_append,
          List.length_cons] at hfim ⊢
        omega
      · rw [cotaBloco_reverse] at hfuel'
        simp only [cotaCorrida, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
          List.length_cons, List.length_nil] at hfuel' hfuel'' ⊢
        omega
      rw [correr_add, hEq]
      simp only [Option.bind_some]
      rw [hany, if_pos rfl, hverd]
      exact hEq''
    | false =>
      have hverd : (C :: C₂ :: Cs').all (fun C' => C'.any (fun p =>
          ((w.drop p.1).head?.isSome && (((w.drop p.1).head? == some Simbolo.um) == p.2))))
          = false := by
        simp only [List.all_cons]
        rw [← List.any_reverse, hany, Bool.false_and]
      refine ⟨fuel' + 1, v', Simbolo.fimC :: pilhaDaCorrida (C₂ :: Cs'), ?_, ?_, ?_⟩
      · simp only [pilhaDaCorrida, List.flatMap_cons, clausulaNaPilha, List.length_append,
          List.length_cons]
        omega
      · rw [cotaBloco_reverse] at hfuel'
        simp only [cotaCorrida, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil,
          List.length_cons, List.length_nil] at hfuel' ⊢
        omega
      rw [correr_add, hEq]
      simp only [Option.bind_some]
      rw [hany, if_neg (by decide), correr, passo_escreveMentira, hverd, if_neg (by decide)]
      rfl

end TGLExt.AMaquinaVerifica

