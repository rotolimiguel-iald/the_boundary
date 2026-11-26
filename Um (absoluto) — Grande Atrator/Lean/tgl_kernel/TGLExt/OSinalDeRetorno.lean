import TGLExt.ACartografiaDaParede

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O laço com retorno: no span, a literal de retorno da cláusula sem par

## O que é

O laço com retorno (`lacoComRetorno`) é o laço final com outro ramo para o caso em que o eixo da morte já está no span dos
eixos inscritos: em vez de calar, lê da cláusula sem par a primeira literal cuja equação é nova e consistente para o
espectro (`literalDeRetorno`, `retornoUtil`), registra essa equação e recomeça do ponto que o resolvedor construtivo
devolve. Com o eixo fora do span, registra a paridade oposta à do ponto, como o laço final.

## O que fica provado [REAL]

1. `paridade2_mascaraLiteral`: a máscara de uma literal lê o bit da variável dela.
2. ★★ `o_sinal_de_retorno_existe`: se o ponto `w` serve o espectro e falha uma cláusula `C` da fórmula, e alguma solução
   serve o espectro, alguma literal de `C` tem equação nova e consistente com o espectro.
3. ★★ `o_eco_do_retorno_e_fiel`: quando o laço com retorno responde com um ponto, esse ponto tem custo zero.
4. ★★ `a_medida_o_retorno_encontra_a_luz_na_fofa`: em `phiFofa`, a partir do canto, com combustível 4, o laço com retorno
   responde com a solução (V, V), onde o laço final calava.
5. `a_medida_o_retorno_recusa_a_insatisfazível`: na fórmula «x₀» e «não x₀», que não tem solução, ele responde em silêncio.
6. `a_medida_a_fofa2_segue_parede_transversal`: em `phiFofa2`, a partir do canto, com combustível 5, ele responde em
   silêncio, embora a fórmula tenha solução: a morte com o eixo fora do span e a paridade oposta perde a solução antes do
   retorno.

## O que NÃO se afirma

* Que o laço com retorno responda com eco quando há solução: `phiFofa2` mede o contrário.
* A escolha do valor na morte com o eixo fora do span segue [OPEN].
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OSinalDeRetorno

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede

/-- [DEF] A máscara de uma literal: o eixo de uma variável só. -/
def mascaraDeLiteral {n : ℕ} (l : Literal n) : Fin n → Bool := fun j => decide (j = l.var)

/-- [DEF] O valor da literal como bit de paridade: a equação força a variável ao sinal pedido. -/
def valorDeLiteral {n : ℕ} (l : Literal n) : ZMod 2 := if l.pos then 1 else 0

/-- [DEF] O sinal de retorno de uma literal: a sua equação é informação NOVA para o espectro (o eixo
reduzido contra o escalonado não se anula). Se reduzisse a (0,1), a literal seria inconsistente com o
espectro; a (0,0), redundante — e aí todo servidor, inclusive o poço, a satisfaria. -/
def retornoUtil {n : ℕ} (S : Espectro2 n) (l : Literal n) : Bool :=
  (bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1).isSome

/-- [DEF] A literal de retorno da cláusula morta: a primeira que o espectro recebe como nova. -/
def literalDeRetorno {n : ℕ} (S : Espectro2 n) (C : Clausula n) : Option (Literal n) :=
  C.find? (fun l => retornoUtil S l)

/-- [DEF] O laço com o sinal de retorno: no caso do eixo já inscrito (o span), em vez de calar, lê da
cláusula morta a literal de retorno — a porta por onde a informação volta —, registra a equação dela e
ressuscita. (O caso transversal segue com a paridade flipada — o gate nomeado.) -/
def lacoComRetorno {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → (Option (Fin n → Bool)) × Espectro2 n
  | 0, S, x => (if custoCNF φ x = 0 then some x else none, S)
  | fuel + 1, S, x =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S)
    else
      let a := eixoDaMorteDe φ w
      if eixoNoSpanB (S.map Prod.fst) a then
        match primeiraSemPar φ w with
        | none => (none, S)
        | some C =>
          match literalDeRetorno S C with
          | none => (none, S)
          | some l =>
            let S' := S ++ [(mascaraDeLiteral l, valorDeLiteral l)]
            lacoComRetorno φ fuel S' (ressurreicaoConstrutiva S')
      else
        let S' := S ++ [(a, paridade2 w a + 1)]
        lacoComRetorno φ fuel S' (ressurreicaoConstrutiva S')

/-- A injeção do bit na paridade: os dois lados do if só coincidem se os bits coincidirem. (Auxiliar do
retorno.) -/
theorem if_zmod2_inj {b c : Bool} :
    (if b then (1 : ZMod 2) else 0) = (if c then (1 : ZMod 2) else 0) → b = c := by
  intro h
  cases b <;> cases c <;> simp at h ⊢

/-- A máscara de uma literal lê o bit dela. (Auxiliar do retorno.) -/
theorem paridade2_mascaraLiteral {n : ℕ} (l : Literal n) (y : Fin n → Bool) :
    paridade2 y (mascaraDeLiteral l) = (if y l.var then (1 : ZMod 2) else 0) := by
  unfold paridade2 mascaraDeLiteral
  rw [Finset.sum_eq_single l.var]
  · simp
  · intro b _ hb
    simp [decide_eq_false hb]
  · intro h
    simp at h

/-- ★★ **O SINAL DE RETORNO EXISTE**: havendo luz servida e o poço falhando a cláusula, alguma literal
da cláusula é sinal de retorno — a sua equação é informação nova consistente com o espectro. A leitura
do campo nunca fecha todas as portas enquanto há luz. -/
theorem o_sinal_de_retorno_existe {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool)
    (C : Clausula n) (hCφ : C ∈ φ)
    (hpoco : serveEspectro2 w S = true) (hfalha : satisfeita C w = false)
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :
    ∃ l : Literal n, l ∈ C ∧ retornoUtil S l = true := by
  obtain ⟨y, hy0, hyserv⟩ := hluz
  have hCy : satisfeita C y = true := (custoCNF_zero_sse φ y).mp hy0 C hCφ
  unfold satisfeita at hCy
  rw [List.any_eq_true] at hCy
  obtain ⟨l, hlC, hly⟩ := hCy
  refine ⟨l, hlC, ?_⟩
  -- a luz serve a equação da literal
  have hlyeq : serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
    show paridade2 y (mascaraDeLiteral l) = valorDeLiteral l
    rw [paridade2_mascaraLiteral]
    unfold valorDeLiteral
    unfold Literal.vale at hly
    rw [beq_iff_eq] at hly
    rw [hly]
  -- servidores do escalonado
  have hconsS : ∃ z : Fin n → Bool, serveEspectro2 z S = true := ⟨y, hyserv⟩
  have hy' : serveEspectro2 y (oEscalonado S) = true := (oEscalonado_preserva S y hconsS).mpr hyserv
  have hw' : serveEspectro2 w (oEscalonado S) = true := (oEscalonado_preserva S w hconsS).mpr hpoco
  unfold retornoUtil
  by_contra hnao
  cases hbit : bitPivo (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1 with
  | some i =>
    exact hnao (by rw [hbit]; rfl)
  | none =>
    have hzero := mascara_zero_de_bitPivo_none _ hbit
    -- a redução preserva o servir, nos dois lados
    have hpy := reduzirNova_preserva y (oEscalonado S) hy' (mascaraDeLiteral l, valorDeLiteral l)
    have hpw := reduzirNova_preserva w (oEscalonado S) hw' (mascaraDeLiteral l, valorDeLiteral l)
    -- a luz serve a reduzida: logo o valor reduzido é 0
    have hvy : serveEq y (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)) :=
      hpy.mpr hlyeq
    have hv0 : (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 = 0 := by
      have h1 : paridade2 y (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1
          = (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2 := hvy
      rw [hzero, paridade2_zero_eixo] at h1
      exact h1.symm
    -- o poço serve a reduzida (0 = 0), logo serve a equação da literal — contradição com a falha
    have hwserve : serveEq w (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)) := by
      show paridade2 w (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).1
          = (reduzirNova (mascaraDeLiteral l) (valorDeLiteral l) (oEscalonado S)).2
      rw [hzero, paridade2_zero_eixo, hv0]
    have hweq : serveEq w (mascaraDeLiteral l, valorDeLiteral l) := hpw.mp hwserve
    -- da equação servida, a literal vale no poço
    have hlvale : l.vale w = true := by
      have h1 : paridade2 w (mascaraDeLiteral l) = valorDeLiteral l := hweq
      rw [paridade2_mascaraLiteral] at h1
      unfold valorDeLiteral at h1
      unfold Literal.vale
      rw [beq_iff_eq]
      exact if_zmod2_inj h1
    -- a cláusula valeria no poço — contradição
    have : satisfeita C w = true := by
      unfold satisfeita
      rw [List.any_eq_true]
      exact ⟨l, hlC, hlvale⟩
    rw [this] at hfalha
    exact Bool.noConfusion hfalha

/-- ★★ **O ECO DO RETORNO É FIEL**: o laço com retorno só responde com ponto de custo zero. (A
fidelidade: o processo nunca inventa luz.) -/
theorem o_eco_do_retorno_e_fiel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool) (S' : Espectro2 n),
      lacoComRetorno φ fuel S x = (some w, S') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w S' h
    unfold lacoComRetorno at h
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
    unfold lacoComRetorno at h
    dsimp only [] at h
    split at h
    · rename_i hcond
      rw [Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · split at h
      · split at h
        · simp at h
        · split at h
          · simp at h
          · exact ih _ _ _ _ h
      · exact ih _ _ _ _ h

/-- ★★ **A MEDIDA DO RETORNO NA FOFA**: onde o laço original calava com a luz viva, o laço com retorno
responde com a própria luz — o ponto todo aceso. (O silêncio terminal está curado aí.) -/
theorem a_medida_o_retorno_encontra_a_luz_na_fofa :
    (lacoComRetorno phiFofa 4 [] (fun _ => false)).1 = some (fun _ => true) := by
  decide

/-- A medida da honestidade: numa fórmula sem luz, o laço com retorno cala — nunca inventa eco. -/
theorem a_medida_o_retorno_recusa_a_insatisfazível :
    (lacoComRetorno ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 3 [] (fun _ => false)).1 = none := by
  decide

/-- ★★ A medida da fronteira: na armadilha 2, o laço com retorno segue calando — a morte TRANSVERSAL
com o lado fixo mata a luz antes do retorno. O gate que resta é exatamente o valor da morte transversal,
lido do campo. -/
theorem a_medida_a_fofa2_segue_parede_transversal :
    (lacoComRetorno phiFofa2 5 [] (fun _ => false)).1 = none := by
  decide

end TGLExt.OSinalDeRetorno

