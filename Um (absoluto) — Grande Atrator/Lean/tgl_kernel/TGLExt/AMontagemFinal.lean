import TGLExt.AContagemDaTensao

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A montagem final: a completude condicional do processo da tensão

## O que é

A montagem dos teoremas das pedras anteriores (o retorno limpo, o posto do escalonado e a contagem da tensão) numa
completude condicional: sob uma hipótese nomeada (`hgate`), o laço da tensão e o processo da tensão respondem com eco.

## O que fica provado [REAL]

1. `encontra_mem`: o elemento que `find?` devolve é membro da lista e satisfaz o predicado.
2. ★ `todos_bits_sao_pivos`: com o posto completo (n), todo bit é pivô da base.
3. ★ `eixo_de_pivo_unitario`: com o posto completo, todo eixo da base é a máscara unitária do seu pivô.
4. ★★ `servidor_unico_com_base_completa`: com o posto completo, dois pontos que servem o espectro coincidem.
5. ★★ `a_base_completa_fixa_o_ponto`: com o posto completo, um ponto que serve o espectro tem custo zero quando alguma
   solução também o serve.
6. `lacoDaTensao_passo_completo`: um passo do laço responde com um ponto de custo zero, ou cala, ou registra a literal do
   retorno limpo e segue.
7. ★★ `o_processo_da_tensao_ecoa`: sob a hipótese nomeada `hgate` (para todo espectro, toda literal de retorno útil que o
   campo não condena, numa cláusula que um ponto servidor falha, deixa alguma solução que serve o espectro e a equação
   dela), havendo solução que serve o espectro e combustível que cubra o que falta de posto mais um, o laço responde com
   eco.
8. ★★ `o_processo_completo_ecoa`: sob a mesma hipótese, havendo solução, o processo da tensão responde com eco.

## O que NÃO se afirma

* Que a hipótese `hgate` valha: na fórmula satisfazível de quatro variáveis da pedra `AMedidaDaHipoteseDaMontagem` ela é
  falsa, e ali o processo da tensão responde com silêncio; para essa fórmula, os dois teoremas valem no vazio.
* Que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AMontagemFinal

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao

variable {n : ℕ}

/-- O `find?` é membro e satisfaz o predicado. -/
theorem encontra_mem {α : Type} (p : α → Bool) :
    ∀ (l : List α) (a : α), l.find? p = some a → a ∈ l ∧ p a = true := by
  intro l
  induction l with
  | nil => intro a h; simp [List.find?] at h
  | cons b bs ih =>
    intro a h
    cases hb : p b with
    | true =>
      rw [List.find?_cons_of_pos hb] at h
      have hba : b = a := Option.some.inj h
      subst hba
      exact ⟨List.mem_cons_self, hb⟩
    | false =>
      rw [List.find?_cons_of_neg (by simp [hb])] at h
      obtain ⟨hm, hp⟩ := ih a h
      exact ⟨List.mem_cons_of_mem b hm, hp⟩

/-- ★ Com o posto completo, todo bit é pivô da base. -/
theorem todos_bits_sao_pivos {n : ℕ} (S : Espectro2 n)
    (hfull : (oEscalonado S).length = n) (j : Fin n) :
    ∃ p ∈ oEscalonado S, bitPivo p.1 = some j := by
  classical
  have hiso := oEscalonado_isolada S
  obtain ⟨htem, hunico, _⟩ := hiso
  rcases Nat.eq_zero_or_pos n with hn | hn
  · subst hn
    exact j.elim0
  · haveI : NeZero n := NeZero.of_pos hn
    -- a imagem dos bits de pivô cobre o universo
    have hinj : Set.InjOn (fun p : (Fin n → Bool) × ZMod 2 => (bitPivo p.1).getD 0)
        (oEscalonado S).toFinset := by
      intro p₁ hp₁ p₂ hp₂ h
      simp only [Finset.mem_coe, List.mem_toFinset] at hp₁ hp₂
      dsimp only at h
      obtain ⟨i₁, hi₁⟩ := htem p₁ hp₁
      obtain ⟨i₂, hi₂⟩ := htem p₂ hp₂
      have hf₁ : (bitPivo p₁.1).getD 0 = i₁ := by rw [hi₁]; rfl
      have hf₂ : (bitPivo p₂.1).getD 0 = i₂ := by rw [hi₂]; rfl
      rw [hf₁, hf₂] at h
      subst h
      exact hunico p₁ hp₁ p₂ hp₂ i₁ hi₁ hi₂
    have hnodup := oEscalonado_nodup S
    have hcard : ((oEscalonado S).toFinset.image
        (fun p : (Fin n → Bool) × ZMod 2 => (bitPivo p.1).getD 0)).card = n := by
      rw [Finset.card_image_of_injOn hinj, List.toFinset_card_of_nodup hnodup]
      exact hfull
    have huniv : (oEscalonado S).toFinset.image
        (fun p : (Fin n → Bool) × ZMod 2 => (bitPivo p.1).getD 0) = Finset.univ :=
      Finset.eq_univ_of_card _ (by rw [hcard, Fintype.card_fin])
    have hj : j ∈ (oEscalonado S).toFinset.image
        (fun p : (Fin n → Bool) × ZMod 2 => (bitPivo p.1).getD 0) := by
      rw [huniv]; exact Finset.mem_univ j
    rw [Finset.mem_image] at hj
    obtain ⟨p, hp, hpj⟩ := hj
    rw [List.mem_toFinset] at hp
    obtain ⟨i, hi⟩ := htem p hp
    refine ⟨p, hp, ?_⟩
    have hfi : (bitPivo p.1).getD 0 = i := by rw [hi]; rfl
    rw [hfi] at hpj
    subst hpj
    exact hi

/-- ★ Com o posto completo, todo eixo da base é a máscara unitária do seu pivô. -/
theorem eixo_de_pivo_unitario {n : ℕ} (S : Espectro2 n)
    (hfull : (oEscalonado S).length = n)
    (p : (Fin n → Bool) × ZMod 2) (hp : p ∈ oEscalonado S) (j : Fin n) (hj : bitPivo p.1 = some j) :
    p.1 = fun i => decide (i = j) := by
  have hiso := oEscalonado_isolada S
  obtain ⟨htem, hunico, hfora⟩ := hiso
  funext i
  by_cases hij : i = j
  · rw [hij]
    rw [bitPivo_spec p.1 j hj]
    simp
  · -- i é pivô de outra entrada; a máscara de p não a contém
    obtain ⟨p', hp', hi'⟩ := todos_bits_sao_pivos S hfull i
    have hjj : j ≠ i := fun h => hij h.symm
    have h0 := hfora p hp p' hp' j i hj hi' hjj
    rw [h0]
    simp [decide_eq_false hij]

/-- ★★ **A BASE COMPLETA FIXA O PONTO**: com o posto completo, dois servidores do espectro coincidem —
o resíduo é um ponto só. -/
theorem servidor_unico_com_base_completa {n : ℕ} (S : Espectro2 n)
    (hfull : (oEscalonado S).length = n)
    (y z : Fin n → Bool)
    (hy : serveEspectro2 y S = true) (hz : serveEspectro2 z S = true) :
    y = z := by
  have hcons : ∃ a : Fin n → Bool, serveEspectro2 a S = true := ⟨y, hy⟩
  have hy' : serveEspectro2 y (oEscalonado S) = true := (oEscalonado_preserva S y hcons).mpr hy
  have hz' : serveEspectro2 z (oEscalonado S) = true := (oEscalonado_preserva S z hcons).mpr hz
  funext j
  obtain ⟨p, hp, hpj⟩ := todos_bits_sao_pivos S hfull j
  have hunit := eixo_de_pivo_unitario S hfull p hp j hpj
  -- servindo a equação do pivô j: o bit é o valor
  unfold serveEspectro2 at hy' hz'
  rw [List.all_eq_true] at hy' hz'
  have hyb := hy' p hp
  have hzb := hz' p hp
  rw [beq_iff_eq] at hyb hzb
  -- paridade2 y p.1 = p.2 com p.1 unitária
  rw [hunit] at hyb hzb
  -- a máscara unitária lê o bit
  have hley : paridade2 y (fun i => decide (i = j)) = (if y j then (1 : ZMod 2) else 0) :=
    paridade2_mascaraLiteral (⟨j, true⟩ : Literal n) y
  have hlez : paridade2 z (fun i => decide (i = j)) = (if z j then (1 : ZMod 2) else 0) :=
    paridade2_mascaraLiteral (⟨j, true⟩ : Literal n) z
  rw [hley] at hyb
  rw [hlez] at hzb
  have : (if y j then (1 : ZMod 2) else 0) = (if z j then (1 : ZMod 2) else 0) := by
    rw [hyb, hzb]
  exact if_zmod2_inj this

/-- ★★ Com o posto completo, um ponto que serve o espectro tem custo zero quando alguma solução também o serve (os dois
coincidem). -/
theorem a_base_completa_fixa_o_ponto {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w y : Fin n → Bool)
    (hfull : (oEscalonado S).length = n)
    (hpoco : serveEspectro2 w S = true)
    (hy : custoCNF φ y = 0) (hys : serveEspectro2 y S = true) :
    custoCNF φ w = 0 := by
  have hwy : w = y := servidor_unico_com_base_completa S hfull w y hpoco hys
  rw [hwy]
  exact hy

/-- A tricotomia do passo do laço, no nível do par. -/
theorem lacoDaTensao_passo_completo {n : ℕ} (φ : CNF n) (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool) :
    (∃ w : Fin n → Bool, custoCNF φ w = 0 ∧ lacoDaTensao φ (fuel + 1) S x = (some w, S)) ∨
    (lacoDaTensao φ (fuel + 1) S x = (none, S)) ∨
    (∃ (l : Literal n) (C : Clausula n),
      primeiraSemPar φ (descerGenerico passoFiel φ S φ.length ([], x)).2 = some C ∧
      literalDoRetornoLimpo φ S C = some l ∧
      lacoDaTensao φ (fuel + 1) S x =
        lacoDaTensao φ fuel (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
          (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))) := by
  rw [lacoDaTensao.eq_2]
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

/-- ★★ **A MONTAGEM FINAL** — a completude condicional do laço da tensão: sob a hipótese nomeada `hgate` (toda literal de
retorno útil que o campo não condena, numa cláusula que um ponto servidor falha, deixa alguma solução que serve o espectro e a
equação dela), havendo solução que serve o espectro e combustível que cubra o que falta de posto mais um, o laço responde
com eco. A hipótese é falsa numa fórmula satisfazível de quatro variáveis (`AMedidaDaHipoteseDaMontagem`). -/
theorem o_processo_da_tensao_ecoa {n : ℕ} (φ : CNF n)
    (hgate : ∀ (S : Espectro2 n) (w : Fin n → Bool) (C : Clausula n) (l : Literal n),
      serveEspectro2 w S = true → satisfeita C w = false → C ∈ φ → l ∈ C →
      retornoUtil S l = true → condenadaNoCampo φ S l = false →
      ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true ∧
        serveEq y (mascaraDeLiteral l, valorDeLiteral l)) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      serveEspectro2 x S = true →
      (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) →
      (oEscalonado S).length + fuel ≥ n + 1 →
      (lacoDaTensao φ fuel S x).1.isSome = true := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x hx hluz hcomb
    have h := oEscalonado_length_le S
    omega
  | succ f ih =>
    intro S x hx hluz hcomb
    rw [lacoDaTensao.eq_2]
    dsimp only []
    set w := (descerGenerico passoFiel φ S φ.length ([], x)).2 with hw
    have hpoco : serveEspectro2 w S = true := o_poco_fiel_serve φ S x hx
    cases hc0 : custoCNF φ w with
    | zero =>
      rfl
    | succ c =>
      rw [if_neg (by simp [hc0])]
      -- a cláusula morta existe (o custo é positivo)
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
        -- os fatos da cláusula morta
        have hfalha : satisfeita C w = false := by
          unfold primeiraSemPar at hP
          have h2 := encontra_satisfaz _ _ _ hP
          cases hs : satisfeita C w with
          | false => rfl
          | true => simp [hs] at h2
        have hCφ : C ∈ φ := by
          unfold primeiraSemPar at hP
          exact (encontra_mem _ _ _ hP).1
        -- o retorno limpo existe (a luz está servida)
        obtain ⟨l₀, hl₀C, hret₀, hcond₀⟩ := o_retorno_limpo_existe φ S w C hCφ hpoco hfalha hluz
        obtain ⟨l, hl⟩ : ∃ l, literalDoRetornoLimpo φ S C = some l := by
          have hlsome : (literalDoRetornoLimpo φ S C).isSome = true := by
            unfold literalDoRetornoLimpo
            rw [List.find?_isSome]
            exact ⟨l₀, hl₀C, by rw [Bool.and_eq_true]; exact ⟨hret₀, by rw [hcond₀]; rfl⟩⟩
          exact Option.isSome_iff_exists.mp hlsome
        rw [hl]
        dsimp only []
        -- os fatos da literal escolhida
        have hfatos : retornoUtil S l = true ∧ condenadaNoCampo φ S l = false := by
          unfold literalDoRetornoLimpo at hl
          have h2 := (encontra_mem _ _ _ hl).2
          rw [Bool.and_eq_true] at h2
          obtain ⟨h1, h2'⟩ := h2
          refine ⟨h1, ?_⟩
          cases hcc : condenadaNoCampo φ S l with
          | false => rfl
          | true => simp [hcc] at h2'
        have hlC : l ∈ C := by
          unfold literalDoRetornoLimpo at hl
          exact (encontra_mem _ _ _ hl).1
        -- o gate: alguma luz segue servida no espectro acrescentado
        obtain ⟨y', hy'0, hy'S, hy'eq⟩ := hgate S w C l hpoco hfalha hCφ hlC hfatos.1 hfatos.2
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
        -- o posto cresce um
        have hcresce := oEscalonado_cresce_um_com_retorno S l hfatos.1
        -- o ressurreto serve
        have hluz'' := hluz'
        obtain ⟨y'', _, hy''serve⟩ := hluz''
        have hx' : serveEspectro2 (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))
            (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true :=
          o_solver_serve _ ⟨y'', hy''serve⟩
        -- a indução
        have hih := ih (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])
          (ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])) hx' hluz' (by omega)
        exact hih

/-- ★★ O corolário no processo inteiro (o fecho do campo primeiro, depois o laço da tensão): sob a hipótese nomeada `hgate`,
havendo solução, o processo da tensão responde com eco. -/
theorem o_processo_completo_ecoa {n : ℕ} (φ : CNF n)
    (hgate : ∀ (S : Espectro2 n) (w : Fin n → Bool) (C : Clausula n) (l : Literal n),
      serveEspectro2 w S = true → satisfeita C w = false → C ∈ φ → l ∈ C →
      retornoUtil S l = true → condenadaNoCampo φ S l = false →
      ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true ∧
        serveEq y (mascaraDeLiteral l, valorDeLiteral l))
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0) :
    (processoDaTensao φ (n + 1) (fun _ => false)).1.isSome = true := by
  unfold processoDaTensao
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
  exact o_processo_da_tensao_ecoa φ hgate (n + 1) _ _ hx ⟨y, hy0, hyfecho⟩ hle

end TGLExt.AMontagemFinal

