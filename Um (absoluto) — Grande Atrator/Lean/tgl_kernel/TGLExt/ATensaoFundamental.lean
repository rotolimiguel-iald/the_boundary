import TGLExt.OFechoDoCampo

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O laço da tensão: a literal de retorno que o campo não condena

## O que é

Uma literal está condenada pelo campo (`condenadaNoCampo`) quando registrar a equação dela e fazer o fecho do campo
(`OFechoDoCampo.fechoDoCampo`) deixa alguma cláusula com todas as literais proibidas. O laço da tensão (`lacoDaTensao`)
registra, na cláusula sem par do ponto que desceu, a primeira literal de retorno que o campo não condena
(`literalDoRetornoLimpo`): o valor registrado é o bit da literal, e o lugar é escolhido pelo campo. Sem literal assim, ele
para em silêncio. O processo da tensão (`processoDaTensao`) faz o fecho do campo primeiro e corre esse laço.

## O que fica provado [REAL]

1. ★★ `a_condenada_nao_tem_luz`: se a literal está condenada, nenhuma solução serve o espectro acrescido da equação dela.
2. ★★ `o_sinal_de_retorno_da_luz`: se o ponto `w` serve o espectro e falha uma cláusula da fórmula, e a solução `y` serve
   o espectro, a literal da cláusula que `y` satisfaz é de retorno.
3. ★★ `o_retorno_limpo_existe`: nas mesmas condições, havendo solução que serve o espectro, existe literal de retorno que
   o campo não condena.
4. ★★ `o_eco_da_tensao_e_fiel`: quando o laço da tensão responde com um ponto, esse ponto tem custo zero.
5. As medidas, por `decide`: o processo da tensão responde com a solução em `phiFofa`, em `phiFofa2` (em que os laços
   anteriores calavam) e na cadeia `phiCadeia` (cuja solução é o próprio ponto de partida); responde com uma solução na
   fórmula «x₀ ou x₁» e «não x₀ ou não x₁»; e cala na fórmula «x₀» e «não x₀», que não tem solução.

## O que NÃO se afirma

* Que o processo da tensão responda com eco em toda fórmula que tem solução: a primeira literal limpa na ordem da cláusula
  pode não levar a uma solução, e uma literal não condenada não garante, sozinha, que alguma solução sirva o espectro
  acrescido (o fecho do campo não é completo). A completude em geral segue [OPEN].
* O número de passos do processo não é contado aqui.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ATensaoFundamental

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo

/-- [DEF] A cláusula totalmente proibida pelo espectro: a contradição do campo. -/
def clausulaCondenada {n : ℕ} (S : Espectro2 n) (C : Clausula n) : Bool :=
  C.all (fun l => literalProibida S l)

/-- [DEF] O campo condena o espectro: alguma cláusula ficou totalmente proibida. -/
def campoCondena {n : ℕ} (φ : CNF n) (S : Espectro2 n) : Bool :=
  φ.any (fun C => clausulaCondenada S C)

/-- [DEF] A literal condenada pelo campo: registrá-la e fechar o campo gera a contradição. -/
def condenadaNoCampo {n : ℕ} (φ : CNF n) (S : Espectro2 n) (l : Literal n) : Bool :=
  campoCondena φ (fechoDoCampo φ (n + 2) (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]))

/-- [DEF] A literal do retorno limpo: a porta por onde a informação volta E o campo não condena. -/
def literalDoRetornoLimpo {n : ℕ} (φ : CNF n) (S : Espectro2 n) (C : Clausula n) : Option (Literal n) :=
  C.find? (fun l => retornoUtil S l && !condenadaNoCampo φ S l)

/-- [DEF] O laço da tensão: a morte é sempre a literal do retorno limpo (o valor é o bit da literal; o lugar, o da
literal limpa). Sem retorno limpo, o silêncio: então nenhuma solução serve o espectro daquele momento (o contrapositivo
de `o_retorno_limpo_existe`), o que não diz que a fórmula não tenha solução. -/
def lacoDaTensao {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → (Option (Fin n → Bool)) × Espectro2 n
  | 0, S, x => (if custoCNF φ x = 0 then some x else none, S)
  | fuel + 1, S, x =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S)
    else
      match primeiraSemPar φ w with
      | none => (none, S)
      | some C =>
        match literalDoRetornoLimpo φ S C with
        | none => (none, S)
        | some l =>
          let S' := S ++ [(mascaraDeLiteral l, valorDeLiteral l)]
          lacoDaTensao φ fuel S' (ressurreicaoConstrutiva S')

/-- [DEF] O processo da tensão: o campo fala o seu fecho primeiro; o laço da tensão corre a partir daí. -/
def processoDaTensao {n : ℕ} (φ : CNF n) (fuel : ℕ) (x : Fin n → Bool) :
    (Option (Fin n → Bool)) × Espectro2 n :=
  let S₀ := fechoDoCampo φ (n + 2) []
  lacoDaTensao φ fuel S₀ (ressurreicaoConstrutiva S₀)

/-- ★★ **A CONDENADA NÃO TEM LUZ** (a sanidade do filtro): se registrar a literal e fechar o campo gera
a contradição (uma cláusula totalmente proibida), nenhuma luz serve o espectro acrescentado. O campo
nunca condena a luz. -/
theorem a_condenada_nao_tem_luz {n : ℕ} (φ : CNF n) (S : Espectro2 n) (l : Literal n)
    (hcond : condenadaNoCampo φ S l = true) :
    ¬ ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧
      serveEspectro2 y (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true := by
  intro h
  obtain ⟨y, hy0, hys⟩ := h
  unfold condenadaNoCampo campoCondena at hcond
  rw [List.any_eq_true] at hcond
  obtain ⟨C', hC'φ, hC'⟩ := hcond
  have hyfecho : serveEspectro2 y
      (fechoDoCampo φ (n + 2) (S ++ [(mascaraDeLiteral l, valorDeLiteral l)])) = true :=
    o_fecho_e_gratuito φ y hy0 (n + 2) (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) hys
  unfold clausulaCondenada at hC'
  rw [List.all_eq_true] at hC'
  have hCy : satisfeita C' y = true := (custoCNF_zero_sse φ y).mp hy0 C' hC'φ
  unfold satisfeita at hCy
  rw [List.any_eq_true] at hCy
  obtain ⟨l', hl'C, hl'y⟩ := hCy
  have hpro := hC' l' hl'C
  have hfalha := proibida_falha_em_quem_serve _ y l' hyfecho hpro
  rw [hfalha] at hl'y
  exact Bool.noConfusion hl'y

/-- ★★ **O SINAL DE RETORNO DA LUZ** (a forma testemunhada): a literal que a luz satisfaz é retornável —
consistente (a luz a serve) e nova (o poço a falha). -/
theorem o_sinal_de_retorno_da_luz {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool)
    (C : Clausula n) (hCφ : C ∈ φ)
    (hpoco : serveEspectro2 w S = true) (hfalha : satisfeita C w = false)
    (y : Fin n → Bool) (hy0 : custoCNF φ y = 0) (hys : serveEspectro2 y S = true) :
    ∃ l : Literal n, l ∈ C ∧ l.vale y = true ∧ retornoUtil S l = true := by
  have hCy : satisfeita C y = true := (custoCNF_zero_sse φ y).mp hy0 C hCφ
  unfold satisfeita at hCy
  rw [List.any_eq_true] at hCy
  obtain ⟨l, hlC, hly⟩ := hCy
  refine ⟨l, hlC, hly, ?_⟩
  have hlyeq : serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
    show paridade2 y (mascaraDeLiteral l) = valorDeLiteral l
    rw [paridade2_mascaraLiteral]
    unfold valorDeLiteral
    unfold Literal.vale at hly
    rw [beq_iff_eq] at hly
    rw [hly]
  have hconsS : ∃ z : Fin n → Bool, serveEspectro2 z S = true := ⟨y, hys⟩
  have hy' : serveEspectro2 y (oEscalonado S) = true := (oEscalonado_preserva S y hconsS).mpr hys
  have hw' : serveEspectro2 w (oEscalonado S) = true := (oEscalonado_preserva S w hconsS).mpr hpoco
  unfold retornoUtil
  by_contra hnao
  cases hbit : bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1 with
  | some i =>
    exact hnao (by rw [hbit]; rfl)
  | none =>
    have hzero := mascara_zero_de_bitPivo_none _ hbit
    have hpy := reduzirNova_preserva y (oEscalonado S) hy' (mascaraDeLiteral l, valorDeLiteral l)
    have hpw := reduzirNova_preserva w (oEscalonado S) hw' (mascaraDeLiteral l, valorDeLiteral l)
    have hvy : serveEq y (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)) :=
      hpy.mpr hlyeq
    have hv0 : (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 = 0 := by
      have h1 : paridade2 y (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1
          = (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 := hvy
      rw [hzero, paridade2_zero_eixo] at h1
      exact h1.symm
    have hwserve : serveEq w (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)) := by
      show paridade2 w (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1
          = (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2
      rw [hzero, paridade2_zero_eixo, hv0]
    have hweq : serveEq w (mascaraDeLiteral l, valorDeLiteral l) := hpw.mp hwserve
    have hlvale : l.vale w = true := by
      have h1 : paridade2 w (mascaraDeLiteral l) = valorDeLiteral l := hweq
      rw [paridade2_mascaraLiteral] at h1
      unfold valorDeLiteral at h1
      unfold Literal.vale
      rw [beq_iff_eq]
      exact if_zmod2_inj h1
    have hsat : satisfeita C w = true := by
      unfold satisfeita
      rw [List.any_eq_true]
      exact ⟨l, hlC, hlvale⟩
    rw [hsat] at hfalha
    exact Bool.noConfusion hfalha

/-- ★★ **O RETORNO LIMPO EXISTE**: havendo luz servida e o poço falhando a cláusula, existe literal
retornável NÃO condenada pelo campo. O campo nunca fecha todas as portas enquanto há luz. -/
theorem o_retorno_limpo_existe {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool)
    (C : Clausula n) (hCφ : C ∈ φ)
    (hpoco : serveEspectro2 w S = true) (hfalha : satisfeita C w = false)
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :
    ∃ l : Literal n, l ∈ C ∧ retornoUtil S l = true ∧ condenadaNoCampo φ S l = false := by
  obtain ⟨y, hy0, hys⟩ := hluz
  obtain ⟨l, hlC, hly, hret⟩ := o_sinal_de_retorno_da_luz φ S w C hCφ hpoco hfalha y hy0 hys
  refine ⟨l, hlC, hret, ?_⟩
  by_contra hcond
  have hcond2 : condenadaNoCampo φ S l = true := by
    cases hc : condenadaNoCampo φ S l with
    | true => rfl
    | false => exact absurd hc hcond
  -- a luz serve o espectro acrescentado da equação da literal
  have hyserve : serveEspectro2 y (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true := by
    rw [serveEspectro2_append]
    have h1 : serveEspectro2 y [(mascaraDeLiteral l, valorDeLiteral l)] = true := by
      unfold serveEspectro2
      rw [List.all_cons, List.all_nil]
      show ((paridade2 y (mascaraDeLiteral l) == valorDeLiteral l) && true) = true
      have hlyeq : serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
        show paridade2 y (mascaraDeLiteral l) = valorDeLiteral l
        rw [paridade2_mascaraLiteral]
        unfold valorDeLiteral
        unfold Literal.vale at hly
        rw [beq_iff_eq] at hly
        rw [hly]
      rw [beq_iff_eq.mpr hlyeq]
      rfl
    rw [h1, hys]
    rfl
  exact a_condenada_nao_tem_luz φ S l hcond2 ⟨y, hy0, hyserve⟩

/-- ★★ **O ECO DA TENSÃO É FIEL**: o laço da tensão só responde com ponto de custo zero. -/
theorem o_eco_da_tensao_e_fiel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool) (S' : Espectro2 n),
      lacoDaTensao φ fuel S x = (some w, S') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w S' h
    unfold lacoDaTensao at h
    rw [Prod.mk.injEq] at h
    obtain ⟨h1, -⟩ := h
    split at h1
    · rename_i hcond
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · simp at h1
  | succ f ih =>
    intro S x w S' h
    unfold lacoDaTensao at h
    dsimp only [] at h
    split at h
    · rename_i hcond
      rw [Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · split at h
      · simp at h
      · split at h
        · simp at h
        · exact ih _ _ _ _ h

/-- ★★ A medida: a fofa ecoa com a própria luz, pelo laço da tensão. -/
theorem a_medida_a_tensao_encontra_a_luz_na_fofa :
    (processoDaTensao phiFofa 5 (fun _ => false)).1 = some (fun _ => true) := by
  decide

/-- ★★ **A MEDIDA DA PAREDE ATRAVESSADA**: a fofa2 — onde toda regra anterior calava ou matava a luz —
ECOA com a própria luz pelo laço da tensão. -/
theorem a_medida_a_tensao_atravessa_a_fofa2 :
    (processoDaTensao phiFofa2 6 (fun _ => false)).1
      = some (fun i => decide (i = 0) || decide (i = 1)) := by
  decide

/-- A medida da cadeia: partindo do ponto com as três variáveis falsas, que já é a solução, o processo da tensão
responde com ele. -/
theorem a_medida_a_tensao_resolve_a_cadeia :
    (processoDaTensao phiCadeia 5 (fun _ => false)).1 = some (fun _ => false) := by
  decide

/-- A medida da honestidade: a insatisfazível cala — o laço da tensão nunca inventa luz. -/
theorem a_medida_a_tensao_cala_na_insatisfazível :
    (processoDaTensao ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 4 (fun _ => false)).1 = none := by
  decide

/-- A medida de sanidade: a phiMedida segue com luz. -/
theorem a_medida_a_tensao_sana_na_phiMedida :
    ((processoDaTensao ([[⟨0, true⟩, ⟨1, true⟩], [⟨0, false⟩, ⟨1, false⟩]] : CNF 2) 5
      (fun _ => false)).1).isSome = true := by
  decide

end TGLExt.ATensaoFundamental

