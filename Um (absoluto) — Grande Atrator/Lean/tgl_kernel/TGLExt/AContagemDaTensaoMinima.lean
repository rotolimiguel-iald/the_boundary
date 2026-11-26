import TGLExt.ATensaoMinima

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# A contagem e a montagem da tensão mínima

## O que é

A contagem e a montagem condicional, escritas para o laço da tensão mínima: a literal escolhida é limpa; o laço registra no
máximo n equações; e, sob a hipótese nomeada `hgate`, com solução que serve o espectro e combustível bastante, o laço responde
com eco.

## O que fica provado [REAL]

1. `foldl_some_mem`: a dobra do mínimo devolve sempre um membro da lista ou a semente.
2. ★ `o_minimo_e_limpo`: a literal escolhida pelo mínimo é limpa (de retorno útil e não condenada).
3. `lacoDaTensaoMinima_passo` e `lacoDaTensaoMinima_passo_completo`: a análise do passo.
4. ★★ `a_tensao_minima_faz_no_maximo_n_mortes`: o laço registra no máximo n equações.
5. ★★ `o_processo_da_tensao_minima_ecoa` e `o_processo_minimo_completo_ecoa`: a montagem condicional, sob `hgate`.

## O que NÃO se afirma

* Que `hgate` valha: ela é, por definição, a hipótese da montagem (`AMedidaDaHipoteseDaMontagem.HipoteseDaMontagem`) e é
  falsa em fórmulas satisfazíveis de três e de quatro variáveis (`AMedidaDaPonteDoDecisor`); nelas a montagem vale no vazio.
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AContagemDaTensaoMinima

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima

variable {n : ℕ}

/-- A dobra do mínimo devolve um membro da lista ou a semente. (Auxiliar da limpeza da escolha.) -/
theorem foldl_some_mem {n : ℕ} (φ : CNF n) (S : Espectro2 n) :
    ∀ (Cs : Clausula n) (acc : Option (Literal n)) (l : Literal n),
      (Cs.foldl (fun acc l => match acc with
          | none => some l
          | some m => if custoDoRetorno φ S l < custoDoRetorno φ S m then some l else some m)
        acc = some l) →
      acc = some l ∨ l ∈ Cs := by
  intro Cs
  induction Cs with
  | nil =>
    intro acc l h
    left
    simpa using h
  | cons c cs ih =>
    intro acc l h
    rw [List.foldl_cons] at h
    have hstep := ih _ l h
    rcases hstep with hacc | hmem
    · -- o passo devolveu some l
      by_cases hacc0 : acc = none
      · rw [hacc0] at hacc
        dsimp only [] at hacc
        have hcl : c = l := Option.some.inj hacc
        subst hcl
        exact Or.inr (List.mem_cons.2 (Or.inl rfl))
      · obtain ⟨m, hm⟩ : ∃ m, acc = some m := by
          cases hacc1 : acc with
          | none => exact absurd hacc1 hacc0
          | some m => exact ⟨m, rfl⟩
        rw [hm] at hacc
        dsimp only [] at hacc
        by_cases hlt : custoDoRetorno φ S c < custoDoRetorno φ S m
        · rw [if_pos hlt] at hacc
          have hcl : c = l := Option.some.inj hacc
          subst hcl
          exact Or.inr (List.mem_cons.2 (Or.inl rfl))
        · rw [if_neg hlt] at hacc
          have hml : m = l := Option.some.inj hacc
          subst hml
          left
          exact hm
    · exact Or.inr (List.mem_cons.2 (Or.inr hmem))

/-- ★ A literal escolhida pelo mínimo é limpa: retornável e não-condenada. (Auxiliar da contagem e da
montagem.) -/
theorem o_minimo_e_limpo {n : ℕ} (φ : CNF n) (S : Espectro2 n) (C : Clausula n) (l : Literal n)
    (h : literalDoRetornoMinimo φ S C = some l) :
    retornoUtil S l = true ∧ condenadaNoCampo φ S l = false := by
  unfold literalDoRetornoMinimo at h
  have hmem := foldl_some_mem φ S _ none l h
  -- a semente é none: a escolhida está na filtrada
  have hmem2 : l ∈ C.filter (fun l => retornoUtil S l && !condenadaNoCampo φ S l) := by
    rcases hmem with hnone | hmem
    · simp at hnone
    · exact hmem
  have h2 := (List.mem_filter.mp hmem2).2
  rw [Bool.and_eq_true] at h2
  obtain ⟨h1, h2'⟩ := h2
  refine ⟨h1, ?_⟩
  cases hc : condenadaNoCampo φ S l with
  | false => rfl
  | true => simp [hc] at h2'

/-- A análise do passo do laço mínimo. -/
theorem lacoDaTensaoMinima_passo {n : ℕ} (φ : CNF n) (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool) :
    (lacoDaTensaoMinima φ (fuel + 1) S x).2 = S ∨
      ∃ (l : Literal n), retornoUtil S l = true ∧
        (lacoDaTensaoMinima φ (fuel + 1) S x).2 =
          (lacoDaTensaoMinima φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
            (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))).2 := by
  rw [lacoDaTensaoMinima.eq_2]
  dsimp only []
  split
  · left; rfl
  · split
    · left; rfl
    · split
      · left; rfl
      · rename_i C hC x2 l hl
        right
        refine ⟨l, ?_, rfl⟩
        exact (o_minimo_e_limpo φ S C l hl).1

/-- ★★ **A CONTAGEM DA TENSÃO MÍNIMA**: o laço corrigido registra no máximo n equações. -/
theorem a_tensao_minima_conta_as_mortes {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      ((lacoDaTensaoMinima φ fuel S x).2).length + (oEscalonado S).length
        ≤ S.length + (oEscalonado (lacoDaTensaoMinima φ fuel S x).2).length := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x
    show S.length + (oEscalonado S).length ≤ S.length + (oEscalonado S).length
    exact Nat.le_refl _
  | succ f ih =>
    intro S x
    rcases lacoDaTensaoMinima_passo φ f S x with h | ⟨l, hret, hstep⟩
    · simp [h]
    · rw [hstep]
      have hcresce := oEscalonado_cresce_um_com_retorno S l hret
      have hih := ih (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
        (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
      have hlen : (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length = S.length + 1 := by
        rw [List.length_append]; simp
      omega

/-- ★★ O laço mínimo é finito: no máximo n mortes. -/
theorem a_tensao_minima_faz_no_maximo_n_mortes {n : ℕ} (φ : CNF n) (fuel : ℕ) (S : Espectro2 n)
    (x : Fin n → Bool) :
    ((lacoDaTensaoMinima φ fuel S x).2).length ≤ S.length + n := by
  have h1 := a_tensao_minima_conta_as_mortes φ fuel S x
  have h2 := oEscalonado_length_le (lacoDaTensaoMinima φ fuel S x).2
  omega

/-- A tricotomia do passo do laço mínimo, no nível do par. -/
theorem lacoDaTensaoMinima_passo_completo {n : ℕ} (φ : CNF n) (fuel : ℕ) (S : Espectro2 n)
    (x : Fin n → Bool) :
    (∃ w : Fin n → Bool, custoCNF φ w = 0 ∧ lacoDaTensaoMinima φ (fuel + 1) S x = (some w, S)) ∨
    (lacoDaTensaoMinima φ (fuel + 1) S x = (none, S)) ∨
    (∃ (l : Literal n) (C : Clausula n),
      primeiraSemPar φ (descerGenerico passoFiel φ S φ.length ([], x)).2 = some C ∧
      literalDoRetornoMinimo φ S C = some l ∧
      lacoDaTensaoMinima φ (fuel + 1) S x =
        lacoDaTensaoMinima φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
          (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))) := by
  rw [lacoDaTensaoMinima.eq_2]
  dsimp only []
  split
  · rename_i hcond
    left
    exact ⟨_, hcond, rfl⟩
  · split
    · right; left; rfl
    · split
      · right; left; rfl
      · rename_i C hC x2 l hl
        right; right
        exact ⟨l, C, hC, hl, by simp_all⟩

/-- ★★ **A MONTAGEM PARA O LAÇO CORRIGIDO**: sob o gate (toda literal limpa de cláusula morta preserva
alguma luz), com luz servida e fuel cobrindo o que falta de posto + 1, o laço da tensão mínima ecoa. -/
theorem o_processo_da_tensao_minima_ecoa {n : ℕ} (φ : CNF n)
    (hgate : ∀ (S : Espectro2 n) (w : Fin n → Bool) (C : Clausula n) (l : Literal n),
      serveEspectro2 w S = true → satisfeita C w = false → C ∈ φ → l ∈ C →
      retornoUtil S l = true → condenadaNoCampo φ S l = false →
      ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true ∧
        serveEq y (mascaraDeLiteral l, valorDeLiteral l)) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      serveEspectro2 x S = true →
      (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) →
      (oEscalonado S).length + fuel ≥ n + 1 →
      (lacoDaTensaoMinima φ fuel S x).1.isSome = true := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x hx hluz hcomb
    have h := oEscalonado_length_le S
    omega
  | succ f ih =>
    intro S x hx hluz hcomb
    rw [lacoDaTensaoMinima.eq_2]
    dsimp only []
    set w := (descerGenerico passoFiel φ S φ.length ([], x)).2 with hw
    have hpoco : serveEspectro2 w S = true := o_poco_fiel_serve φ S x hx
    cases hc0 : custoCNF φ w with
    | zero =>
      rfl
    | succ c =>
      rw [if_neg (by simp [hc0])]
      -- a cláusula morta existe
      have hpos : 0 < custoCNF φ w := by rw [hc0]; exact Nat.zero_lt_succ c
      unfold custoCNF at hpos
      rw [List.countP_pos_iff] at hpos
      obtain ⟨C₀, hC₀φ, hC₀sat⟩ := hpos
      cases hP : primeiraSemPar φ w with
      | none =>
        unfold primeiraSemPar at hP
        rw [List.find?_eq_none] at hP
        exact absurd hC₀sat (hP C₀ hC₀φ)
      | some C =>
        dsimp only []
        have hfalha : satisfeita C w = false := by
          unfold primeiraSemPar at hP
          have h2 := encontra_satisfaz _ _ _ hP
          cases hs : satisfeita C w with
          | false => rfl
          | true => simp [hs] at h2
        have hCφ : C ∈ φ := by
          unfold primeiraSemPar at hP
          exact (encontra_mem _ _ _ hP).1
        -- o retorno limpo existe, logo a escolha mínima é some
        obtain ⟨l₀, hl₀C, hret₀, hcond₀⟩ := o_retorno_limpo_existe φ S w C hCφ hpoco hfalha hluz
        obtain ⟨l, hl⟩ : ∃ l, literalDoRetornoMinimo φ S C = some l := by
          have h0 := o_minimo_existe_quando_ha_limpa φ S C ⟨l₀, hl₀C, hret₀, hcond₀⟩
          exact Option.isSome_iff_exists.mp h0
        rw [hl]
        dsimp only []
        -- a escolhida pelo mínimo é limpa
        have hlimpa := o_minimo_e_limpo φ S C l hl
        have hlC : l ∈ C := by
          unfold literalDoRetornoMinimo at hl
          have hmem := foldl_some_mem φ S _ none l hl
          rcases hmem with hnone | hmem
          · simp at hnone
          · exact (List.mem_filter.mp hmem).1
        -- o gate transporta a luz
        obtain ⟨y', hy'0, hy'S, hy'eq⟩ := hgate S w C l hpoco hfalha hCφ hlC hlimpa.1 hlimpa.2
        have hluz' : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧
            serveEspectro2 y (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true := by
          refine ⟨y', hy'0, ?_⟩
          rw [serveEspectro2_append]
          have h1 : serveEspectro2 y' [(mascaraDeLiteral l, valorDeLiteral l)] = true := by
            unfold serveEspectro2
            rw [List.all_cons, List.all_nil]
            show ((paridade2 y' (mascaraDeLiteral l) == valorDeLiteral l) && true) = true
            rw [beq_iff_eq.mpr hy'eq]
            rfl
          rw [h1, hy'S]
          rfl
        have hcresce := oEscalonado_cresce_um_com_retorno S l hlimpa.1
        have hluz'' := hluz'
        obtain ⟨y'', _, hy''serve⟩ := hluz''
        have hx' : serveEspectro2 (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
            (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true :=
          o_solver_serve _ ⟨y'', hy''serve⟩
        have hih := ih (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
          (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])) hx' hluz' (by omega)
        exact hih

/-- ★★ **O PROCESSO MÍNIMO COMPLETO ECOA**: o corolário no processo inteiro. -/
theorem o_processo_minimo_completo_ecoa {n : ℕ} (φ : CNF n)
    (hgate : ∀ (S : Espectro2 n) (w : Fin n → Bool) (C : Clausula n) (l : Literal n),
      serveEspectro2 w S = true → satisfeita C w = false → C ∈ φ → l ∈ C →
      retornoUtil S l = true → condenadaNoCampo φ S l = false →
      ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true ∧
        serveEq y (mascaraDeLiteral l, valorDeLiteral l))
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0) :
    (processoDaTensaoMinima φ (n + 1) (fun _ => false)).1.isSome = true := by
  unfold processoDaTensaoMinima
  dsimp only []
  obtain ⟨y, hy0⟩ := hluz
  have h0 : serveEspectro2 y ([] : Espectro2 n) = true := by
    unfold serveEspectro2
    simp
  have hyfecho : serveEspectro2 y (fechoDoCampo φ (n + 2) []) = true :=
    o_fecho_e_gratuito φ y hy0 (n + 2) [] h0
  have hx : serveEspectro2 (ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) []))
      (fechoDoCampo φ (n + 2) []) = true := o_solver_serve _ ⟨y, hyfecho⟩
  have hle : (oEscalonado (fechoDoCampo φ (n + 2) [])).length + (n + 1) ≥ n + 1 := by omega
  exact o_processo_da_tensao_minima_ecoa φ hgate (n + 1) _ _ hx ⟨y, hy0, hyfecho⟩ hle

end TGLExt.AContagemDaTensaoMinima

