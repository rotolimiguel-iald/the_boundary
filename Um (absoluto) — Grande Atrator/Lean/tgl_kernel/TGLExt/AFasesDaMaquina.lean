import TGLExt.AMaquinaTM2DaVerificacao

set_option autoImplicit false
set_option maxHeartbeats 800000
set_option maxRecDepth 8192

/-!
# A simulação da máquina TM2: as fases elementares — passos, laços de movimento, partida e parada

## O que é

A máquina `aMaquinaTM2` (pedra `AMaquinaTM2DaVerificacao`) é a máquina TM2 de cinco pilhas desenhada para
verificar uma CNF com uma testemunha dada na entrada (`codificaEntrada p`: a fórmula `p.1` e os bits `p.2`, nas
codificações fixas de `APonteAoTM2`). Em `AMaquinaTM2DaVerificacao`, a simulação geral é a proposição
`ASimulacaoCompleta`, usada como hipótese em `o_objetivo_se_reduz_a_simulacao`; a sua prova está em
`OLacoDasClausulas` (e, por outro caminho, em `SimTM2.simulacao_completa`). Esta pedra prova as fases elementares dessa simulação: o
passo de cada rótulo dos laços de movimento, os laços que movem símbolos entre pilhas (`preparar`, `reverter`,
`limparFormula`, `limparEntrada`), a partida (da configuração inicial ao laço das cláusulas) e a parada (da
limpeza ao `halt` com o bit da resposta).

## A forma da prova

As configurações se escrevem com as cinco pilhas nomeadas (`cfgStacks`, `cfg`, `cfgH`), e os laços raciocinam
por componente. Cada lema de passo reescreve a cabeça da pilha lida e fecha a igualdade de configurações ponto
a ponto (`cfg_fecha` + `fin_cases`). Os lemas de laço são `def` (o `EvalsToInTime` do Mathlib é uma estrutura
em `Type`, não uma proposição), cada um uma indução na lista movida, com a cota de uma invocação de rótulo por
símbolo, mais a detecção do fim. A composição é por `StateTransition.EvalsToInTime.trans`; a igualdade das
configurações finais, por `evals_congr`; a folga da cota, por `custo_mono`. Nenhum laço é provado por
`decide`: as listas são variáveis (pilhas simbólicas). O estado é (carregador, bandeira, rota); a bandeira
atravessa as limpezas intacta, porque, no estado, elas alteram só a rota.

## O que fica provado

1. Os passos elementares: `passo_preparar_sep`, `passo_preparar_move`, `passo_reverter_move`,
   `passo_reverter_vazio`, `passo_limparFormula_move`, `passo_limparFormula_vazio`, `passo_limparEntrada_move`,
   `passo_limparEntrada_vazio`, `passo_escrever`.
2. Os laços: `fase_preparar` (a fórmula vai à pilha de trabalho, invertida, e a divisória é consumida),
   `fase_reverter` (a inversão da inversão: a fórmula fica na pilha própria, na ordem original),
   `fase_limparFormula`, `fase_limparEntrada` (esvaziam as pilhas preservando a bandeira). Cada um com a cota
   de passos `comprimento da lista + 1`.
3. `sep_nao_entra_na_formula`: a divisória nunca aparece na fórmula codificada.
4. ★ `a_partida`: da configuração inicial do modelo, em no máximo `2·(Lf + 1)` passos (`Lf` = comprimento da
   fórmula codificada) a máquina está no laço das cláusulas — a fórmula na pilha própria, os bits da
   testemunha na entrada, as demais pilhas vazias.
5. ★ `a_parada`: do início da limpeza, com a pilha de trabalho, o depósito e a saída vazios, em no máximo
   `F + E + 3` passos a máquina para exatamente na configuração final do modelo (`haltList`), com a bandeira
   na pilha de saída.

As cotas de passos são cotas (`EvalsToInTime`: passos ≤ teto), não igualdades.

## O que NÃO se afirma

* O ciclo do literal (as caminhadas dos ticks e dos bits, a comparação, o restaurar) e o laço das cláusulas
  não são provados aqui: estão em `ACicloDoLiteral` e `OLacoDasClausulas`.
* Nenhuma máquina é dita decidir SAT em tempo polinomial: a máquina simulada é a verificadora, que recebe a
  testemunha na entrada. Decidir SAT sem testemunha, em tempo polinomial, na máquina de fita segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AFasesDaMaquina

open TGLExt.APonteAoTM2 TGLExt.AMaquinaTM2DaVerificacao StateTransition

/-- O tipo das cinco pilhas (a saída é booleana; as demais, do alfabeto). -/
abbrev Pilhas := ∀ k : PilhaK, List (stackAlphabet k)

/-- [DEF] As pilhas por nome — a forma em que a simulação raciocina. -/
def cfgStacks (e fo t d : List Simbolo) (sa : List Bool) : Pilhas
  | .entrada => e
  | .saida => sa
  | .formula => fo
  | .trabalho => t
  | .deposito => d

/-- [DEF] A configuração corrente com as pilhas nomeadas. -/
def cfg (l : Rotulo) (v : Estado) (e fo t d : List Simbolo) (sa : List Bool) :
    Turing.TM2.Cfg stackAlphabet Rotulo Estado :=
  ⟨some l, v, cfgStacks e fo t d sa⟩

/-- [DEF] A configuração de parada com as pilhas nomeadas. -/
def cfgH (e fo t d : List Simbolo) (sa : List Bool) : Turing.TM2.Cfg stackAlphabet Rotulo Estado :=
  ⟨none, (none, false, false), cfgStacks e fo t d sa⟩

/-- A igualdade de configurações passa ao `some` (congruência). (Auxiliar.) -/
theorem algum_inj {A B : Turing.TM2.Cfg stackAlphabet Rotulo Estado} (h : A = B) :
    (some A : Option (Turing.TM2.Cfg stackAlphabet Rotulo Estado)) = some B :=
  congrArg some h

/-- O fechamento da igualdade de configurações, ponto a ponto: rótulo, estado, e as pilhas por `fin_cases`.
(Auxiliar.) -/
theorem cfg_fecha {l₁ l₂ : Option Rotulo} {v₁ v₂ : Estado} {S₁ S₂ : Pilhas}
    (hl : l₁ = l₂) (hv : v₁ = v₂) (h : ∀ k, S₁ k = S₂ k) :
    (⟨l₁, v₁, S₁⟩ : Turing.TM2.Cfg stackAlphabet Rotulo Estado) = ⟨l₂, v₂, S₂⟩ :=
  (Turing.TM2.Cfg.mk.injEq ..).mpr ⟨hl, hv, funext h⟩

/-- Folga na cota: os passos são os mesmos, o teto cresce. (Auxiliar da composição.) -/
def custo_mono {a : Turing.TM2.Cfg stackAlphabet Rotulo Estado}
    {c : Option (Turing.TM2.Cfg stackAlphabet Rotulo Estado)} {m m' : ℕ}
    (h : EvalsToInTime aMaquinaTM2.step a c m) (hm : m ≤ m') : EvalsToInTime aMaquinaTM2.step a c m' :=
  ⟨h.toEvalsTo, Nat.le_trans h.steps_le_m hm⟩

/-- Um passo é uma avaliação em tempo 1. (Auxiliar da composição.) -/
def passo_unico {a b : Turing.TM2.Cfg stackAlphabet Rotulo Estado}
    (h : aMaquinaTM2.step a = some b) : EvalsToInTime aMaquinaTM2.step a (some b) 1 :=
  ⟨⟨1, h⟩, Nat.le_refl 1⟩

/-- Igualdade de configuração final: os passos são os mesmos, o destino é reescrito. (Auxiliar da composição.) -/
def evals_congr {a : Turing.TM2.Cfg stackAlphabet Rotulo Estado}
    {c₁ c₂ : Option (Turing.TM2.Cfg stackAlphabet Rotulo Estado)} {m : ℕ}
    (h : c₁ = c₂) (hE : EvalsToInTime aMaquinaTM2.step a c₁ m) : EvalsToInTime aMaquinaTM2.step a c₂ m :=
  h ▸ hE

/-- A cabeça da lista invertida como primeira escolha. (Auxiliar dos estados dos laços.) -/
theorem cabeca_reverse_cons (v : Option Simbolo) (s : Simbolo) (F : List Simbolo) :
    (F.reverse.head? <|> some s) = ((s :: F).reverse.head? <|> v) := by
  cases F with
  | nil => rfl
  | cons s' F' =>
    have hne : (s' :: F').reverse ≠ [] := by simp
    simp [List.reverse_cons, List.head?_append, hne]

/-- ★ A divisória nunca aparece na fórmula codificada: a preparação para no lugar certo. -/
theorem sep_nao_entra_na_formula (φf : List (List (ℕ × Bool))) :
    Simbolo.sep ∉ codificaFormula φf := by
  intro h
  simp only [codificaFormula, List.mem_append, List.mem_flatMap, List.mem_singleton] at h
  rcases h with ⟨C, hCφ, hC⟩ | hF
  · simp only [codificaClausula, List.mem_append, List.mem_flatMap, List.mem_singleton] at hC
    rcases hC with ⟨p, hpC, hp⟩ | hFimC
    · simp only [codificaLiteral, List.mem_cons, List.mem_replicate] at hp
      rcases hp with hmar | ⟨_, htic⟩
      · split_ifs at hmar <;> exact Simbolo.noConfusion hmar
      · exact Simbolo.noConfusion htic
    · exact Simbolo.noConfusion hFimC
  · exact Simbolo.noConfusion hF

/-! ## Os passos elementares -/

/-- O passo da preparação ao ver a divisória: consome-a e vai reverter. -/
theorem passo_preparar_sep (v : Estado) (rest fo t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .preparar v (Simbolo.sep :: rest) fo t d sa)
      = some (cfg .reverter (v.1, true, v.2.2) rest fo t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- O passo da preparação ao ver um símbolo da fórmula: move-o à pilha de trabalho. -/
theorem passo_preparar_move (v : Estado) (s : Simbolo) (rest fo t d : List Simbolo) (sa : List Bool)
    (hs : s ≠ Simbolo.sep) :
    aMaquinaTM2.step (cfg .preparar v (s :: rest) fo t d sa)
      = some (cfg .preparar (some s, false, v.2.2) rest fo (s :: t) d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons, Option.getD_some]
  simp only [show (some s = some Simbolo.sep) ↔ False from by simp [hs]]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- O passo da reversão com a pilha de trabalho vazia: vai ao laço das cláusulas. -/
theorem passo_reverter_vazio (v : Estado) (e fo d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .reverter v e fo [] d sa)
      = some (cfg .lerClausula (v.1, true, v.2.2) e fo [] d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_nil]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> rfl

/-- O passo da reversão com símbolo: move-o à pilha da fórmula. -/
theorem passo_reverter_move (v : Estado) (e fo : List Simbolo) (s : Simbolo) (rest d : List Simbolo)
    (sa : List Bool) :
    aMaquinaTM2.step (cfg .reverter v e fo (s :: rest) d sa)
      = some (cfg .reverter (some s, false, v.2.2) e (s :: fo) rest d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons, Option.getD_some]
  simp only [show ((some s : Option Simbolo) = none) ↔ False from by simp]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- O passo da limpeza da fórmula com símbolo: descarta-o, com a bandeira intacta. -/
theorem passo_limparFormula_move (v : Estado) (e : List Simbolo) (s : Simbolo) (rest t d : List Simbolo)
    (sa : List Bool) :
    aMaquinaTM2.step (cfg .limparFormula v e (s :: rest) t d sa)
      = some (cfg .limparFormula (v.1, v.2.1, false) e rest t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons]
  simp only [show ((some s : Option Simbolo) = none) ↔ False from by simp]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- O passo da limpeza da fórmula vazia: vai limpar a entrada, com a bandeira intacta. -/
theorem passo_limparFormula_vazio (v : Estado) (e t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .limparFormula v e [] t d sa)
      = some (cfg .limparEntrada (v.1, v.2.1, true) e [] t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_nil]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> rfl

/-- O passo da limpeza da entrada com símbolo: descarta-o, com a bandeira intacta. -/
theorem passo_limparEntrada_move (v : Estado) (s : Simbolo) (rest fo t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .limparEntrada v (s :: rest) fo t d sa)
      = some (cfg .limparEntrada (v.1, v.2.1, false) rest fo t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_cons, List.tail_cons]
  simp only [show ((some s : Option Simbolo) = none) ↔ False from by simp]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-- O passo da limpeza da entrada vazia: vai escrever, com a bandeira intacta. -/
theorem passo_limparEntrada_vazio (v : Estado) (fo t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .limparEntrada v [] fo t d sa)
      = some (cfg .escrever (v.1, v.2.1, true) [] fo t d sa) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgStacks,
    List.head?_nil]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> rfl

/-- O passo da escrita: a bandeira vai à pilha de saída, o estado inicial é reposto, e a máquina para. -/
theorem passo_escrever (v : Estado) (e fo t d : List Simbolo) (sa : List Bool) :
    aMaquinaTM2.step (cfg .escrever v e fo t d sa)
      = some (cfgH e fo t d (v.2.1 :: sa)) := by
  simp only [aMaquinaTM2, Turing.FinTM2.step, Turing.TM2.step, oPrograma, Turing.TM2.stepAux, cfg, cfgH,
    cfgStacks]
  apply algum_inj
  apply cfg_fecha rfl rfl
  intro k; fin_cases k <;> simp [cfgStacks, Function.update]

/-! ## Os laços de movimento -/

/-- ★ O LAÇO DA PREPARAÇÃO: a fórmula sai da entrada para a pilha de trabalho (invertida), a divisória é
consumida, e a máquina vai reverter. Cota: `F.length + 1` passos (um por símbolo, mais a divisória). -/
def fase_preparar (v : Estado) (fo t d : List Simbolo) (sa : List Bool) (F B : List Simbolo)
    (hsep : Simbolo.sep ∉ F) :
    EvalsToInTime aMaquinaTM2.step (cfg .preparar v (F ++ [Simbolo.sep] ++ B) fo t d sa)
      (some (cfg .reverter (F.reverse.head? <|> v.1, true, v.2.2) B fo (F.reverse ++ t) d sa)) (F.length + 1) := by
  induction F generalizing v t with
  | nil =>
    have h1 := passo_preparar_sep v B fo t d sa
    refine evals_congr ?_ (passo_unico h1)
    simp [List.reverse_nil, List.nil_append]
  | cons s F' ih =>
    have hs : s ≠ Simbolo.sep := by
      intro hss
      apply hsep
      rw [hss]
      exact List.mem_cons_self
    have hsep' : Simbolo.sep ∉ F' := fun hm => hsep (List.mem_cons_of_mem s hm)
    have h1 := passo_preparar_move v s (F' ++ [Simbolo.sep] ++ B) fo t d sa hs
    have h2 := ih (some s, false, v.2.2) (s :: t) hsep'
    refine evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2)
    apply algum_inj
    apply cfg_fecha rfl
    · show (F'.reverse.head? <|> (some s, false, v.2.2).1, true, (some s, false, v.2.2).2.2)
        = ((s :: F').reverse.head? <|> v.1, true, v.2.2)
      rw [cabeca_reverse_cons]
    · intro k; fin_cases k <;> simp [cfgStacks, List.reverse_cons, List.append_assoc, List.singleton_append]

/-- ★ O LAÇO DA REVERSÃO: a inversão da inversão — a fórmula fica na pilha própria, na ordem original. Cota:
`W.length + 1` passos (um por símbolo, mais a pilha vazia). -/
def fase_reverter (v : Estado) (e fo d : List Simbolo) (sa : List Bool) (W : List Simbolo) :
    EvalsToInTime aMaquinaTM2.step (cfg .reverter v e fo W d sa)
      (some (cfg .lerClausula (W.reverse.head? <|> v.1, true, v.2.2) e (W.reverse ++ fo) [] d sa))
      (W.length + 1) := by
  induction W generalizing v fo with
  | nil =>
    have h1 := passo_reverter_vazio v e fo d sa
    refine evals_congr ?_ (passo_unico h1)
    simp [List.reverse_nil, List.nil_append]
  | cons s W' ih =>
    have h1 := passo_reverter_move v e fo s W' d sa
    have h2 := ih (some s, false, v.2.2) (s :: fo)
    refine evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2)
    apply algum_inj
    apply cfg_fecha rfl
    · show (W'.reverse.head? <|> (some s, false, v.2.2).1, true, (some s, false, v.2.2).2.2)
        = ((s :: W').reverse.head? <|> v.1, true, v.2.2)
      rw [cabeca_reverse_cons]
    · intro k; fin_cases k <;> simp [cfgStacks, List.reverse_cons, List.append_assoc, List.singleton_append]

/-- ★ O LAÇO DA LIMPEZA DA FÓRMULA: esvazia a pilha da fórmula preservando a bandeira. Cota: `F.length + 1` passos. -/
def fase_limparFormula (v : Estado) (e t d : List Simbolo) (sa : List Bool) (F : List Simbolo) :
    EvalsToInTime aMaquinaTM2.step (cfg .limparFormula v e F t d sa)
      (some (cfg .limparEntrada (v.1, v.2.1, true) e [] t d sa)) (F.length + 1) := by
  induction F generalizing v with
  | nil => exact passo_unico (passo_limparFormula_vazio v e t d sa)
  | cons s F' ih =>
    have h1 := passo_limparFormula_move v e s F' t d sa
    have h2 := ih (v.1, v.2.1, false)
    exact EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2

/-- ★ O LAÇO DA LIMPEZA DA ENTRADA: esvazia a entrada preservando a bandeira. Cota: `E.length + 1` passos. -/
def fase_limparEntrada (v : Estado) (fo t d : List Simbolo) (sa : List Bool) (E : List Simbolo) :
    EvalsToInTime aMaquinaTM2.step (cfg .limparEntrada v E fo t d sa)
      (some (cfg .escrever (v.1, v.2.1, true) [] fo t d sa)) (E.length + 1) := by
  induction E generalizing v with
  | nil => exact passo_unico (passo_limparEntrada_vazio v fo t d sa)
  | cons s E' ih =>
    have h1 := passo_limparEntrada_move v s E' fo t d sa
    have h2 := ih (v.1, v.2.1, false)
    exact EvalsToInTime.trans _ _ _ _ _ _ (passo_unico h1) h2

/-! ## A partida e a parada -/

/-- A configuração inicial do modelo é a forma nomeada da partida, pilha a pilha. (Auxiliar.) -/
theorem initList_stacks (l : List Simbolo) (k : PilhaK) :
    (Turing.initList aMaquinaTM2 l).stk k = cfgStacks l [] [] [] [] k := by
  fin_cases k <;> simp [Turing.initList, aMaquinaTM2, cfgStacks]

/-- A configuração inicial do modelo é a forma nomeada da partida. (Auxiliar.) -/
theorem initList_cfg (l : List Simbolo) :
    Turing.initList aMaquinaTM2 l = cfg .preparar (none, false, false) l [] [] [] [] := by
  apply (Turing.TM2.Cfg.mk.injEq ..).mpr
  refine ⟨?_, ?_, funext (initList_stacks l)⟩
  · show (some aMaquinaTM2.main : Option Rotulo) = some Rotulo.preparar; rfl
  · show aMaquinaTM2.initialState = (none, false, false); rfl

/-- A configuração final da parada, com tudo vazio, é a do modelo, pilha a pilha. (Auxiliar.) -/
theorem haltList_stacks (b : Bool) (k : PilhaK) :
    (Turing.haltList aMaquinaTM2 [b]).stk k = cfgStacks [] [] [] [] [b] k := by
  fin_cases k <;> simp [Turing.haltList, aMaquinaTM2, cfgStacks]

/-- A configuração final da parada, com tudo vazio, é a do modelo. (Auxiliar.) -/
theorem cfgH_haltList (b : Bool) :
    cfgH [] [] [] [] [b] = Turing.haltList aMaquinaTM2 [b] := by
  apply (Turing.TM2.Cfg.mk.injEq ..).mpr
  refine ⟨rfl, ?_, funext fun k => (haltList_stacks b k).symm⟩
  · show (none, false, false) = aMaquinaTM2.initialState; rfl

/-- ★ **A PARTIDA**: da configuração inicial do modelo, em no máximo `2·(Lf + 1)` passos a máquina está pronta
para o laço das cláusulas — a fórmula na pilha própria, a testemunha na entrada, as demais pilhas vazias. -/
def a_partida (p : List (List (ℕ × Bool)) × List Bool) :
    EvalsToInTime aMaquinaTM2.step (Turing.initList aMaquinaTM2 (codificaEntrada p))
      (some (cfg .lerClausula
        ((codificaFormula p.1).head? <|> ((codificaFormula p.1).reverse.head? <|> none), true, false)
        (codificaBits p.2) (codificaFormula p.1) [] [] []))
      (2 * ((codificaFormula p.1).length + 1)) := by
  rw [initList_cfg]
  have h1 := fase_preparar (none, false, false) [] [] [] [] (codificaFormula p.1) (codificaBits p.2)
    (sep_nao_entra_na_formula p.1)
  have h2 := fase_reverter ((codificaFormula p.1).reverse.head? <|> none, true, false)
    (codificaBits p.2) [] [] [] ((codificaFormula p.1).reverse ++ [])
  refine custo_mono (evals_congr ?_ (EvalsToInTime.trans _ _ _ _ _ _ h1 h2)) ?_
  · apply algum_inj
    apply cfg_fecha rfl
    · show (((codificaFormula p.1).reverse ++ []).reverse.head? <|>
          ((codificaFormula p.1).reverse.head? <|> none), true, false)
        = ((codificaFormula p.1).head? <|> ((codificaFormula p.1).reverse.head? <|> none), true, false)
      simp [List.reverse_reverse, List.append_nil]
    · intro k; fin_cases k <;> simp [cfgStacks, List.reverse_reverse, List.append_nil]
  · simp only [List.length_reverse, List.length_append, List.length_nil]; omega

/-- ★ **A PARADA**: do início da limpeza (com a resposta na bandeira), em no máximo `F + E + 3` passos a
máquina para exatamente na configuração final do modelo — o `haltList` com a bandeira na saída e as
demais pilhas vazias. -/
def a_parada (v : Estado) (E F : List Simbolo) :
    EvalsToInTime aMaquinaTM2.step (cfg .limparFormula v E F [] [] [])
      (some (Turing.haltList aMaquinaTM2 [v.2.1])) (F.length + E.length + 3) := by
  have h1 := fase_limparFormula v E [] [] [] F
  have h2 := fase_limparEntrada (v.1, v.2.1, true) [] [] [] [] E
  have h3 := passo_escrever (v.1, v.2.1, true) [] [] [] [] []
  have hcomp := EvalsToInTime.trans _ _ _ _ _ _ h1 (EvalsToInTime.trans _ _ _ _ _ _ h2 (passo_unico h3))
  refine custo_mono (evals_congr ?_ hcomp) ?_
  · apply algum_inj
    show cfgH [] [] [] [] [v.2.1] = Turing.haltList aMaquinaTM2 [v.2.1]
    exact cfgH_haltList v.2.1
  · omega

end TGLExt.AFasesDaMaquina

