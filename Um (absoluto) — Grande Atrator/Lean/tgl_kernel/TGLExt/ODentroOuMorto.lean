import TGLExt.OLivroConsultado

set_option autoImplicit false
set_option maxHeartbeats 1600000
set_option maxRecDepth 8192

/-!
# Dentro ou morto: a completude do veredito da busca com o livro

## O que é

A busca com o livro consultado (`processoComLivro`) recebe aqui o invariante que a faz decidir (`EstadoDaLuzComLivro`): a luz serve o
espectro corrente ou um ponto da pilha; o ponto é servo fiel; a casa é habitada; os espectros são de equações de literais; a pilha é
encadeada; o livro é consistente; e o livro é sadio: toda porta registrada ou tem o seu ramo como prefixo do corrente (dentro) ou a sua
subárvore não tem solução (morto). A correção do silêncio não vem de um atestado do «não»: vem da manutenção do invariante de que a luz
nunca se perdeu.

## O que fica provado [REAL]

1. ★ `serve_transfer`: servir é do conteúdo; espectros de literais do mesmo ponto têm os mesmos servidores.
2. `Config`, `passoMaquina`, `aCorrida`, `dentroDoPonto`, `mortoNoPonto`, `livroSadio`, `pilhaEncadeada`, `EstadoDaLuzComLivro` [DEF].
3. ★ `o_estado_inicial_tem_luz`: com a fórmula satisfazível, a configuração inicial satisfaz o invariante.
4. ★ `existe_divergencia`: servindo um prefixo e não o todo, há o ponto onde a luz diverge.
5. ★★ `a_luz_exige_limpa_nova`: com a luz servida no espectro, o livro sadio e os registros consistentes, o detector do silêncio com o
   livro consultado não dispara: há sempre porta nova.
6. ★★★ `a_manutencao`: o invariante sobrevive a cada passo da máquina do livro, na morte e na subida.
7. ★ `o_terminal_impossivel`: com o invariante e a pilha vazia, o detector do silêncio terminal não pode estar disparado.
8. ★★★ `a_luz_nunca_cala_o_veredito_final`: com o invariante, se o laço responde com silêncio, ou o silêncio é o terminal com o
   invariante (que o item 7 exclui), ou o combustível foi pago em portas e pontos, com a luz intacta.
9. ★★★ `o_veredito_decide_com_a_cota`: se a fórmula tem solução e o combustível é pelo menos `2·aCotaDoLivro n + 1`, a busca com o
   livro, partindo do ponto com todas as variáveis falsas, responde com um ponto.
10. ★★ `o_veredito_e_honesto`: a busca só responde com ponto de custo zero.
11. ★★ `a_medida_o_veredito_decide`: por `decide`, a busca responde com ponto em cinco fórmulas com solução e cala na fórmula sem
    solução de uma variável.

Com 9 e 10: com combustível de pelo menos `2·aCotaDoLivro n + 1`, a busca responde com ponto se e só se a fórmula tem solução; dentro
dessa cota, o silêncio significa que não há solução.

## O que NÃO se afirma

* Que combustível polinomial no tamanho da fórmula baste: a cota é `2·aCotaDoLivro n + 1`, exponencial no número de variáveis
  (3ⁿ·2n pares); a cota polinomial segue [OPEN].
* Que algum processo, ou a máquina de Turing sem o leitor, decida SAT em tempo polinomial: segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ODentroOuMorto

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil
open TGLExt.OSilencioVivo TGLExt.AParedeDoLadoOposto TGLExt.ACartografiaDaParede TGLExt.OSinalDeRetorno
open TGLExt.OCampoFalaPrimeiro TGLExt.OFechoDoCampo TGLExt.ATensaoFundamental TGLExt.OPostoDoEscalonado
open TGLExt.AContagemDaTensao TGLExt.AMontagemFinal TGLExt.OCustoDoProcesso TGLExt.ATensaoMinima
open TGLExt.AContagemDaTensaoMinima TGLExt.OMapaDosMuros TGLExt.APodaFechaOContorno TGLExt.AMedidaDaArvore
open TGLExt.AFamiliaProfunda TGLExt.ADescidaPinadaEmMemoria TGLExt.ALeituraCorrespondente
open TGLExt.OAlgoritmoPenP TGLExt.OLivroConsultado

/-- Servir a equação da literal é a literal valer (a volta). -/
theorem vale_de_serveEq {n : ℕ} (l : Literal n) (y : Fin n → Bool)
    (h : serveEq y (mascaraDeLiteral l, valorDeLiteral l)) : l.vale y = true := by
  unfold serveEq at h
  rw [paridade2_mascaraLiteral] at h
  unfold valorDeLiteral at h
  unfold Literal.vale
  rw [beq_iff_eq]
  cases hy : y l.var with
  | false => cases hp : l.pos with
    | false => rfl
    | true => rw [hy, hp] at h; simp at h
  | true => cases hp : l.pos with
    | false => rw [hy, hp] at h; simp at h
    | true => rfl

/-- Servir a equação da literal é a literal valer (a forma bicondicional). -/
theorem serveEq_sse_vale {n : ℕ} (l : Literal n) (y : Fin n → Bool) :
    serveEq y (mascaraDeLiteral l, valorDeLiteral l) ↔ l.vale y = true :=
  ⟨vale_de_serveEq l y, serveEq_de_vale l y⟩

/-- [DEF] O espectro da máquina é só de equações de literais. -/
def soLiterais {n : ℕ} (S : Espectro2 n) : Prop :=
  ∀ e ∈ S, ∃ l : Literal n, e = (mascaraDeLiteral l, valorDeLiteral l)

/-- Servir um espectro de literais é toda literal dele valer. -/
theorem serveEspectro2_literais_sse {n : ℕ} (S : Espectro2 n) (hlit : soLiterais S)
    (y : Fin n → Bool) :
    serveEspectro2 y S = true ↔ ∀ l : Literal n,
      (mascaraDeLiteral l, valorDeLiteral l) ∈ S → l.vale y = true := by
  unfold serveEspectro2
  rw [List.all_eq_true]
  constructor
  · intro hserve l hl
    have hserveq : serveEq y (mascaraDeLiteral l, valorDeLiteral l) := by
      have h1 := hserve _ hl
      rw [beq_iff_eq] at h1
      exact h1
    exact vale_de_serveEq l y hserveq
  · intro hvale e he
    obtain ⟨l, hle⟩ := hlit e he
    rw [hle] at he ⊢
    rw [beq_iff_eq]
    exact serveEq_de_vale _ _ (hvale l he)

/-- O conteúdo alguma: há equação cobrindo a variável com aquele valor. -/
theorem conteudoDe_some {n : ℕ} (S : Espectro2 n) (j : Fin n) (v : ZMod 2)
    (h : conteudoDe S j = some v) :
    ∃ e ∈ S, e.1 j = true ∧ e.2 = v := by
  unfold conteudoDe at h
  cases hf : S.find? (fun e => e.1 j) with
  | none => rw [hf] at h; simp at h
  | some e =>
    rw [hf] at h
    simp at h
    have hmem : e ∈ S := List.mem_of_find?_eq_some hf
    have hbit : (fun e => e.1 j) e = true := encontra_mem _ _ _ hf |>.2
    exact ⟨e, hmem, hbit, h⟩

/-- O servidor único do conteúdo: num espectro de literais consistente, duas equações na mesma variável
têm o mesmo valor. -/
theorem o_valor_e_unico_no_ponto {n : ℕ} (S : Espectro2 n) (hlit : soLiterais S)
    (hcons : ∃ z : Fin n → Bool, serveEspectro2 z S = true) (j : Fin n)
    (e₁ e₂ : (Fin n → Bool) × ZMod 2) (he₁ : e₁ ∈ S) (he₂ : e₂ ∈ S)
    (hj₁ : e₁.1 j = true) (hj₂ : e₂.1 j = true) : e₁.2 = e₂.2 := by
  obtain ⟨z, hz⟩ := hcons
  obtain ⟨l₁, rfl⟩ := hlit e₁ he₁
  obtain ⟨l₂, rfl⟩ := hlit e₂ he₂
  have hserve := (serveEspectro2_literais_sse S hlit z).mp hz
  have hv₁ := hserve l₁ he₁
  have hv₂ := hserve l₂ he₂
  -- l₁.var = j = l₂.var: a máscara unitária só cobre a própria variável
  have hvar₁ : l₁.var = j := by
    have h1 : mascaraDeLiteral l₁ j = true := hj₁
    unfold mascaraDeLiteral at h1
    rw [decide_eq_true_eq] at h1
    exact h1.symm
  have hvar₂ : l₂.var = j := by
    have h1 : mascaraDeLiteral l₂ j = true := hj₂
    unfold mascaraDeLiteral at h1
    rw [decide_eq_true_eq] at h1
    exact h1.symm
  -- os dois valem em z na mesma variável: o mesmo sinal
  unfold Literal.vale at hv₁ hv₂
  rw [beq_iff_eq] at hv₁ hv₂
  rw [hvar₁] at hv₁
  rw [hvar₂] at hv₂
  have hpos : l₁.pos = l₂.pos := by rw [← hv₁, ← hv₂]
  unfold valorDeLiteral
  rw [hpos]

/-- ★ **SERVIR É DO CONTEÚDO**: dois espectros de literais do mesmo ponto, um consistente, têm os mesmos
servidores. -/
theorem serve_transfer {n : ℕ} (S₁ S₂ : Espectro2 n) (y : Fin n → Bool)
    (hlit₁ : soLiterais S₁) (hlit₂ : soLiterais S₂)
    (hcons₂ : ∃ z : Fin n → Bool, serveEspectro2 z S₂ = true)
    (hmp : mesmoPonto S₁ S₂ = true)
    (hserve : serveEspectro2 y S₁ = true) :
    serveEspectro2 y S₂ = true := by
  rw [mesmoPonto_sse] at hmp
  rw [serveEspectro2_literais_sse S₂ hlit₂ y]
  intro l hl
  -- a equação de l em S₂: o conteúdo na var de l é o valor de l
  have hcont₂ : conteudoDe S₂ l.var = some (valorDeLiteral l) := by
    have hbit : (mascaraDeLiteral l) l.var = true := by
      show decide (l.var = l.var) = true
      rw [decide_eq_true_eq]
    have hfind : (S₂.find? (fun e => e.1 l.var)).isSome = true := by
      rw [List.find?_isSome]
      exact ⟨(mascaraDeLiteral l, valorDeLiteral l), hl, hbit⟩
    unfold conteudoDe
    cases hf : S₂.find? (fun e => e.1 l.var) with
    | none => rw [hf] at hfind; simp at hfind
    | some e =>
      simp only [Option.map_some, Option.some.injEq]
      -- e cobre l.var; pela unicidade do valor no ponto, e.2 = valorDeLiteral l
      have hmem : e ∈ S₂ := List.mem_of_find?_eq_some hf
      have hbit' : e.1 l.var = true := (encontra_mem _ _ _ hf).2
      exact o_valor_e_unico_no_ponto S₂ hlit₂ hcons₂ l.var e (mascaraDeLiteral l, valorDeLiteral l)
        hmem hl hbit' hbit
  -- o mesmo conteúdo em S₁: há equação na var de l com o valor de l
  have hcont₁ := hmp l.var
  rw [hcont₂] at hcont₁
  obtain ⟨e₁, he₁, hj₁, hv₁⟩ := conteudoDe_some S₁ l.var (valorDeLiteral l) hcont₁
  obtain ⟨l₁, rfl⟩ := hlit₁ e₁ he₁
  -- a máscara de l₁ cobre l.var: l₁.var = l.var
  have hvar : l₁.var = l.var := by
    have h1 : mascaraDeLiteral l₁ l.var = true := hj₁
    unfold mascaraDeLiteral at h1
    rw [decide_eq_true_eq] at h1
    exact h1.symm
  -- o valor: valorDeLiteral l₁ = valorDeLiteral l → l₁.pos = l.pos
  have hv₁' : valorDeLiteral l₁ = valorDeLiteral l := hv₁
  have hpos : l₁.pos = l.pos := by
    unfold valorDeLiteral at hv₁'
    cases hp₁ : l₁.pos with
    | false =>
      cases hp : l.pos with
      | false => rfl
      | true => rw [hp₁, hp] at hv₁'; simp at hv₁'
    | true =>
      cases hp : l.pos with
      | false => rw [hp₁, hp] at hv₁'; simp at hv₁'
      | true => rfl
  -- l₁ = l, e está em S₁: l vale em y
  have hleq : l₁ = l := by
    cases l₁
    cases l
    simp at hvar hpos
    rw [hvar, hpos]
  have hvl₁ := (serveEspectro2_literais_sse S₁ hlit₁ y).mp hserve l₁ he₁
  rw [hleq] at hvl₁
  exact hvl₁

/-- [DEF] **DENTRO DO PONTO**: o ramo da porta registrada é prefixo do espectro corrente. -/
def dentroDoPonto {n : ℕ} (S : Espectro2 n) (p : Espectro2 n × Literal n) : Prop :=
  (p.1 ++ [(mascaraDeLiteral p.2, valorDeLiteral p.2)]) <+: S

/-- [DEF] **MORTO NO PONTO**: a subárvore da porta registrada não tem luz. -/
def mortoNoPonto {n : ℕ} (φ : CNF n) (p : Espectro2 n × Literal n) : Prop :=
  ¬ ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧
    serveEspectro2 y (p.1 ++ [(mascaraDeLiteral p.2, valorDeLiteral p.2)]) = true

/-- [DEF] **O LIVRO SADIO** (a forma forte): toda porta registrada ou está dentro do ramo corrente,
ou dentro de um ramo pendente da pilha, ou morreu sem luz. -/
def livroSadio {n : ℕ} (φ : CNF n) (S : Espectro2 n) (pilha : List (Espectro2 n))
    (livro : List (Espectro2 n × Literal n)) : Prop :=
  ∀ p ∈ livro, dentroDoPonto S p ∨ (∃ e ∈ pilha, dentroDoPonto e p) ∨ mortoNoPonto φ p

/-- A pilha encadeada: cada entrada é prefixo de todas as anteriores (a mais recente é a mais
profunda). (A forma que a subida consome.) -/
def pilhaEncadeada {n : ℕ} : List (Espectro2 n) → Prop
  | [] => True
  | a :: l => (∀ e ∈ l, e <+: a) ∧ pilhaEncadeada l

/-- A mesma literal é a mesma: campos iguais, estruturas iguais. -/
theorem mesmaLiteral_eq {n : ℕ} (a b : Literal n) (h : mesmaLiteral a b = true) : a = b := by
  unfold mesmaLiteral at h
  rw [Bool.and_eq_true] at h
  obtain ⟨hv, hp⟩ := h
  rw [beq_iff_eq] at hv hp
  cases a
  cases b
  simp at hv hp
  rw [hv, hp]

/-- O prefixo é membro: o ramo registrado está no espectro, a porta é lida nele. (Auxiliar do
bootstrap.) -/
theorem membro_de_dentro {n : ℕ} (S : Espectro2 n) (p : Espectro2 n × Literal n)
    (h : dentroDoPonto S p) :
    (mascaraDeLiteral p.2, valorDeLiteral p.2) ∈ S := by
  unfold dentroDoPonto at h
  obtain ⟨t, ht⟩ := h
  rw [← ht]
  exact List.mem_append_left _ (List.mem_append_right _ (List.mem_singleton_self _))

/-- A cláusula falsificada: toda literal dela falha no ponto. (Auxiliar do bootstrap.) -/
theorem falsificada_toda_falha {n : ℕ} (C : Clausula n) (w : Fin n → Bool)
    (h : satisfeita C w = false) (l : Literal n) (hl : l ∈ C) : l.vale w = false := by
  unfold satisfeita at h
  rw [List.any_eq_false] at h
  have h2 := h l hl
  cases hv : l.vale w with
  | true => exact absurd hv h2
  | false => rfl

/-- O mesmo ponto é simétrico. -/
theorem mesmoPonto_comm {n : ℕ} (S₁ S₂ : Espectro2 n) :
    mesmoPonto S₁ S₂ = mesmoPonto S₂ S₁ := by
  unfold mesmoPonto
  congr 1
  funext j
  have h : (conteudoDe S₁ j = conteudoDe S₂ j) = (conteudoDe S₂ j = conteudoDe S₁ j) := propext eq_comm
  simp only [h]

/-- [DEF] **A configuração da máquina do livro**: o espectro, o ponto, a pilha de pontos, o livro de
portas. -/
structure Config (n : ℕ) where
  esp : Espectro2 n
  ponto : Fin n → Bool
  pilha : List (Espectro2 n)
  livro : List (Espectro2 n × Literal n)

/-- [DEF] **O passo da máquina**: a transição — o eco (estaciona), o silêncio terminal (estaciona), a
subida ao ponto mais recente, ou a morte com o registro no livro. `none` é a máquina estacionada. -/
def passoMaquina {n : ℕ} (φ : CNF n) (c : Config n) : Option (Config n) :=
  let w := (descerGenerico passoFiel φ c.esp φ.length ([], c.ponto)).2
  if custoCNF φ w = 0 then none
  else
    match primeiraSemPar φ w with
    | none => none
    | some C =>
      match limpasNoveisDe φ c.livro c.esp C with
      | [] =>
        match c.pilha with
        | [] => none
        | S₁ :: pilha₁ => some ⟨S₁, ressurreicaoConstrutiva S₁, pilha₁, c.livro⟩
      | novas =>
        match oMelhorDe φ c.esp novas with
        | some l => some ⟨c.esp ++ [(mascaraDeLiteral l, valorDeLiteral l)],
            ressurreicaoConstrutiva (c.esp ++ [(mascaraDeLiteral l, valorDeLiteral l)]),
            c.esp :: c.pilha, (c.esp, l) :: c.livro⟩
        | none => none

/-- [DEF] **A corrida**: a iteração do passo — `none` depois de estacionar. -/
def aCorrida {n : ℕ} (φ : CNF n) (c₀ : Config n) : ℕ → Option (Config n)
  | 0 => some c₀
  | k + 1 => (aCorrida φ c₀ k).bind (passoMaquina φ)

/-- [DEF] **O ESTADO DA LUZ COM O LIVRO** — o invariante completo da busca com o livro: a luz no espectro ou na
pilha; o ponto servo fiel; a casa habitada; os espectros de literais; a pilha encadeada; o livro
consistente; e o livro sadio (dentro do corrente, dentro de um pendente, ou morto). -/
structure EstadoDaLuzComLivro {n : ℕ} (φ : CNF n) (c : Config n) : Prop where
  luz : (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y c.esp = true) ∨
    (∃ S₁ : Espectro2 n, S₁ ∈ c.pilha ∧ ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧
      serveEspectro2 y S₁ = true)
  pocoServe : serveEspectro2 c.ponto c.esp = true
  habitada : ∃ z : Fin n → Bool, serveEspectro2 z c.esp = true
  literais : soLiterais c.esp ∧ (∀ S₁ : Espectro2 n, S₁ ∈ c.pilha → soLiterais S₁)
  pilhaHabitada : ∀ S₁ : Espectro2 n, S₁ ∈ c.pilha → ∃ z : Fin n → Bool, serveEspectro2 z S₁ = true
  encadeada : (∀ e : Espectro2 n, e ∈ c.pilha → e <+: c.esp) ∧
    pilhaEncadeada c.pilha
  livroConsistente : ∀ p : Espectro2 n × Literal n, p ∈ c.livro →
    soLiterais p.1 ∧ ∃ z : Fin n → Bool, serveEspectro2 z p.1 = true
  sadio : livroSadio φ c.esp c.pilha c.livro
  protegida : ∀ D : Espectro2 n, (∃ l : Literal n, (D, l) ∈ c.livro) →
    (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y D = true) →
    D ∈ c.pilha ∨ D = c.esp
  aForma : (fechoDoCampo φ (n + 2) [] <+: c.esp) ∧
    (∀ e : Espectro2 n, e ∈ c.pilha → fechoDoCampo φ (n + 2) [] <+: e) ∧
    (∀ (D : Espectro2 n) (l : Literal n),
      (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]) <+: c.esp →
      (fechoDoCampo φ (n + 2) []).length < (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length →
      (D, l) ∈ c.livro)

/-- O prefixo tem comprimento limitado. (Auxiliar geral.) -/
theorem prefixo_length {n : ℕ} (a b : Espectro2 n) (h : a <+: b) : a.length ≤ b.length := by
  obtain ⟨t, ht⟩ := h
  rw [← ht, List.length_append]
  omega

/-- O prefixo de prefixo é prefixo. (Auxiliar geral.) -/
theorem prefixo_trans {n : ℕ} (a b c : Espectro2 n) (h₁ : a <+: b) (h₂ : b <+: c) : a <+: c := by
  obtain ⟨t₁, ht₁⟩ := h₁
  obtain ⟨t₂, ht₂⟩ := h₂
  exact ⟨t₁ ++ t₂, by rw [← ht₂, ← ht₁, List.append_assoc]⟩

/-- Servir o todo é servir o prefixo. (Auxiliar geral.) -/
theorem serve_prefixo {n : ℕ} (S₁ S₂ : Espectro2 n) (y : Fin n → Bool)
    (h : S₁ <+: S₂) (hs : serveEspectro2 y S₂ = true) : serveEspectro2 y S₁ = true := by
  obtain ⟨t, ht⟩ := h
  rw [← ht] at hs
  rw [serveEspectro2_append] at hs
  rw [Bool.and_eq_true] at hs
  exact hs.1

/-- Todo espectro é o take do seu prefixo maior. (Auxiliar da comparabilidade.) -/
theorem prefixo_take {n : ℕ} (a b : Espectro2 n) (h : a <+: b) : b.take a.length = a := by
  obtain ⟨t, ht⟩ := h
  rw [← ht]
  exact List.take_left

/-- Prefixos do mesmo espectro são comparáveis por comprimento. (Auxiliar da região.) -/
theorem prefixo_comparavel {n : ℕ} (a b c : Espectro2 n) (ha : a <+: c) (hb : b <+: c)
    (hle : a.length ≤ b.length) : a <+: b := by
  have hta := prefixo_take a c ha
  have htb := prefixo_take b c hb
  have h1 : b.take a.length = a := by
    have h2 : (c.take b.length).take a.length = c.take a.length := by
      rw [List.take_take, Nat.min_eq_left hle]
    rw [htb] at h2
    rw [h2, hta]
  refine ⟨b.drop a.length, ?_⟩
  have h2 := List.take_append_drop a.length b
  rw [h1] at h2
  exact h2

/-- ★ **A DIVERGÊNCIA EXISTE**: servindo um prefixo mas não o todo, há o ponto onde a luz diverge —
o último prefixo servido e a primeira porta que ela recusa. (Auxiliar da região.) -/
theorem existe_divergencia {n : ℕ} (y : Fin n → Bool) :
    ∀ (P₀ suf : Espectro2 n), serveEspectro2 y P₀ = true →
      ¬ serveEspectro2 y (P₀ ++ suf) = true → soLiterais (P₀ ++ suf) →
      ∃ (D : Espectro2 n) (l : Literal n) (rest : Espectro2 n),
        P₀ ++ suf = D ++ (mascaraDeLiteral l, valorDeLiteral l) :: rest ∧
        P₀ <+: D ∧
        serveEspectro2 y D = true ∧
        ¬ serveEspectro2 y (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true := by
  intro P₀ suf
  induction suf generalizing P₀ with
  | nil =>
    intro hserveP hnao hlit
    rw [List.append_nil] at hnao
    exact absurd hserveP hnao
  | cons e suf' ih =>
    intro hserveP hnao hlit
    by_cases hse : serveEspectro2 y (P₀ ++ [e]) = true
    · have hS : (P₀ ++ [e]) ++ suf' = P₀ ++ (e :: suf') := List.append_assoc _ _ _
      rw [← hS] at hnao
      have hlit' : soLiterais ((P₀ ++ [e]) ++ suf') := by rwa [← hS] at hlit
      obtain ⟨D, l, rest, hD, hPD, hs1, hs2⟩ := ih (P₀ ++ [e]) hse hnao hlit'
      exact ⟨D, l, rest, by rw [← hS]; exact hD,
        prefixo_trans P₀ (P₀ ++ [e]) D ⟨[e], rfl⟩ hPD, hs1, hs2⟩
    · have hmem : e ∈ P₀ ++ (e :: suf') := List.mem_append_right _ List.mem_cons_self
      obtain ⟨l, hl⟩ := hlit e hmem
      exact ⟨P₀, l, suf', by rw [hl], ⟨[], by rw [List.append_nil]⟩, hserveP,
        by rw [hl] at hse; exact hse⟩

/-- O fecho só acrescenta equações de literais. (Auxiliar do estado inicial.) -/
theorem fecho_so_literais {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n), soLiterais S → soLiterais (fechoDoCampo φ fuel S) := by
  intro fuel
  induction fuel with
  | zero => intro S h; unfold fechoDoCampo; exact h
  | succ f ih =>
    intro S h
    unfold fechoDoCampo
    split
    · exact h
    · rename_i l hpasso
      apply ih
      intro e he
      rw [List.mem_append, List.mem_singleton] at he
      cases he with
      | inl hS => exact h e hS
      | inr hS =>
        subst hS
        exact ⟨l, rfl⟩

/-- ★ **O ESTADO INICIAL TEM LUZ**: com luz na fórmula, a configuração inicial satisfaz o invariante
completo. -/
theorem o_estado_inicial_tem_luz {n : ℕ} (φ : CNF n)
    (hSAT : SAT φ) :
    EstadoDaLuzComLivro φ ⟨fechoDoCampo φ (n + 2) [],
      ressurreicaoConstrutiva (fechoDoCampo φ (n + 2) []), [], []⟩ := by
  obtain ⟨y, hy⟩ := hSAT
  have hy0 : custoCNF φ y = 0 := (custoCNF_zero_sse φ y).mpr hy
  have h0 : serveEspectro2 y ([] : Espectro2 n) = true := by unfold serveEspectro2; simp
  have hyfecho : serveEspectro2 y (fechoDoCampo φ (n + 2) []) = true :=
    o_fecho_e_gratuito φ y hy0 (n + 2) [] h0
  refine ⟨Or.inl ⟨y, hy0, hyfecho⟩, o_solver_serve _ ⟨y, hyfecho⟩, ⟨y, hyfecho⟩,
    ⟨fecho_so_literais φ _ [] (fun e he => absurd he List.not_mem_nil),
      fun S₁ h => absurd h List.not_mem_nil⟩,
    fun S₁ h => absurd h List.not_mem_nil,
    ⟨fun e he => absurd he List.not_mem_nil, trivial⟩,
    fun p hp => absurd hp List.not_mem_nil,
    fun p hp => absurd hp List.not_mem_nil,
    fun D hD _ => by
      obtain ⟨l, hmem⟩ := hD
      exact absurd hmem List.not_mem_nil,
    ⟨⟨[], by rw [List.append_nil]⟩, fun e he => absurd he List.not_mem_nil, fun D l hD hcomp => ?_⟩⟩
  have hle := prefixo_length _ _ hD
  change (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length ≤ (fechoDoCampo φ (n + 2) []).length at hle
  rw [List.length_append, List.length_singleton] at hle hcomp
  omega

/-- ★★ **A LUZ EXIGE LIMPA NOVA** — o bootstrap: com a luz servida no espectro, o livro sadio e os
registros de literais consistentes, o detector com o livro consultado NÃO dispara — há sempre porta
nova. O dentro morre no poço; o morto morre na transferência. Uma subárvore com luz nunca é abandonada
sem eco. -/
theorem a_luz_exige_limpa_nova {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool)
    (C : Clausula n) (hCφ : C ∈ φ)
    (hpoco : serveEspectro2 w S = true) (hfalha : satisfeita C w = false)
    (hlit : soLiterais S)
    (hluz : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true)
    (livro : List (Espectro2 n × Literal n))
    (pilha : List (Espectro2 n))
    (hsadio : livroSadio φ S pilha livro)
    (henc : ∀ e ∈ pilha, e <+: S)
    (hlitLivro : ∀ p : Espectro2 n × Literal n, p ∈ livro →
      soLiterais p.1 ∧ ∃ z : Fin n → Bool, serveEspectro2 z p.1 = true) :
    limpasNoveisDe φ livro S C ≠ [] := by
  intro hvazio
  obtain ⟨y, hy0, hys⟩ := hluz
  obtain ⟨l, hll, hly⟩ := a_porta_da_luz_nas_limpas φ S w C hCφ hpoco hfalha y hy0 hys
  have hmem : l ∈ limpasNoveisDe φ livro S C := by
    unfold limpasNoveisDe
    rw [List.mem_filter]
    refine ⟨hll, ?_⟩
    cases ht : tentadaNoPonto livro S l with
    | false => rfl
    | true =>
      exfalso
      obtain ⟨p, hp, hpt, hpor⟩ := (tentada_sse livro S l).mp ht
      have hpeq : p.2 = l := mesmaLiteral_eq _ _ hpor
      have hpoço_mata : ∀ (E : Espectro2 n), dentroDoPonto E p → serveEspectro2 w E = true → False := by
        intro E hdentro hserveE
        have hmemE := membro_de_dentro E p hdentro
        rw [hpeq] at hmemE
        have hvw : l.vale w = true := by
          unfold serveEspectro2 at hserveE
          rw [List.all_eq_true] at hserveE
          exact vale_de_serveEq l w ((beq_iff_eq).mp (hserveE _ hmemE))
        have hCmem : l ∈ C := by
          unfold limpasDe at hll
          exact (List.mem_filter.mp hll).1
        have hwf := falsificada_toda_falha C w hfalha l hCmem
        rw [hvw] at hwf
        exact Bool.noConfusion hwf
      have hsad := hsadio p hp
      cases hsad with
      | inl hdentro => exact hpoço_mata S hdentro hpoco
      | inr hresto =>
        cases hresto with
        | inl hpil =>
          obtain ⟨e, he, hdentro⟩ := hpil
          exact hpoço_mata e hdentro (serve_prefixo e S w (henc e he) hpoco)
        | inr hmorto =>
          obtain ⟨hlitp, z, hz⟩ := hlitLivro p hp
          have hmp : mesmoPonto S p.1 = true := by rw [mesmoPonto_comm]; exact hpt
          have hyP : serveEspectro2 y p.1 = true :=
            serve_transfer S p.1 y hlit hlitp ⟨z, hz⟩ hmp hys
          have hyPa : serveEspectro2 y (p.1 ++ [(mascaraDeLiteral l, valorDeLiteral l)]) = true :=
            serveEspectro2_append_de_vale p.1 y l hyP hly
          unfold mortoNoPonto at hmorto
          rw [hpeq] at hmorto
          exact hmorto ⟨y, hy0, hyPa⟩
  rw [hvazio] at hmem
  exact List.not_mem_nil hmem

/-- A equação determina a literal: máscaras e valores iguais, literais iguais. (Auxiliar da
aForma.) -/
theorem eqLiteral_inj {n : ℕ} (l₁ l₂ : Literal n)
    (h : (mascaraDeLiteral l₁, valorDeLiteral l₁) = (mascaraDeLiteral l₂, valorDeLiteral l₂)) :
    l₁ = l₂ := by
  rw [Prod.mk.injEq] at h
  obtain ⟨hm, hv⟩ := h
  have hvar : l₁.var = l₂.var := by
    have h2 : mascaraDeLiteral l₂ l₁.var = true := by
      have h1 : mascaraDeLiteral l₁ l₁.var = mascaraDeLiteral l₂ l₁.var := by rw [hm]
      rw [← h1]
      show decide (l₁.var = l₁.var) = true
      rw [decide_eq_true_eq]
    have h3 : decide (l₁.var = l₂.var) = true := h2
    rw [decide_eq_true_eq] at h3
    exact h3
  have hpos : l₁.pos = l₂.pos := by
    unfold valorDeLiteral at hv
    cases hp1 : l₁.pos with
    | false =>
      cases hp2 : l₂.pos with
      | false => rfl
      | true => rw [hp1, hp2] at hv; simp at hv
    | true =>
      cases hp2 : l₂.pos with
      | false => rw [hp1, hp2] at hv; simp at hv
      | true => rfl
  cases l₁
  cases l₂
  simp at hvar hpos
  rw [hvar, hpos]

/-- Prefixo com o comprimento do todo é o todo. (Auxiliar da aForma.) -/
theorem prefixo_de_mesmo_comprimento {n : ℕ} (a b : Espectro2 n) (h : a <+: b)
    (hlen : a.length = b.length) : a = b := by
  have hta := prefixo_take a b h
  rw [hlen, List.take_length] at hta
  exact hta.symm

/-- ★★★ **A MANUTENÇÃO DO PASSO**: o invariante da luz sobrevive a cada passo da máquina do livro —
na morte e na subida. Na subida, o livro sadio se mantém pelo argumento da região: o ramo com luz tem
a sua divergência protegida na pilha (a protegida), e o detector não dispara onde há luz (o bootstrap). -/
theorem a_manutencao {n : ℕ} (φ : CNF n) (c : Config n) (hI : EstadoDaLuzComLivro φ c) :
    ∀ c', passoMaquina φ c = some c' → EstadoDaLuzComLivro φ c' := by
  intro c' hp
  unfold passoMaquina at hp
  dsimp only [] at hp
  have hpoco : serveEspectro2 (descerGenerico passoFiel φ c.esp φ.length ([], c.ponto)).2 c.esp
      = true :=
    o_poco_fiel_serve φ c.esp c.ponto hI.pocoServe
  split at hp
  · simp at hp
  · rename_i hcusto
    split at hp
    · simp at hp
    · rename_i C hC
      obtain ⟨hCφ, hCfalha⟩ := a_primeira_sem_par_mem φ _ C hC
      split at hp
      · -- o detector disparou: limpas novas vazia — o espectro corrente está escuro
        rename_i hlimpas
        have hescuro : ¬ (∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y c.esp = true) :=
          fun hluz => a_luz_exige_limpa_nova φ c.esp _ C hCφ hpoco hCfalha hI.literais.1 hluz
            c.livro c.pilha hI.sadio hI.encadeada.1 hI.livroConsistente hlimpas
        split at hp
        · simp at hp
        · -- ★ A SUBIDA
          rename_i x₀ S₁ pilha₁ heq
          have hcp : c' = ⟨S₁, ressurreicaoConstrutiva S₁, pilha₁, c.livro⟩ := (Option.some.inj hp).symm
          subst hcp
          have hhead := hI.pilhaHabitada S₁ (by rw [heq]; exact List.mem_cons_self)
          have hlit₁ := hI.literais.2 S₁ (by rw [heq]; exact List.mem_cons_self)
          have hfecho₁ := hI.aForma.2.1 S₁ (by rw [heq]; exact List.mem_cons_self)
          refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
          · cases hI.luz with
            | inl hluz => exact absurd hluz hescuro
            | inr hna =>
              obtain ⟨S₂, he₂, y, hy0, hyS₂⟩ := hna
              rw [heq] at he₂
              rw [List.mem_cons] at he₂
              cases he₂ with
              | inl heq => subst heq; exact Or.inl ⟨y, hy0, hyS₂⟩
              | inr htail => exact Or.inr ⟨S₂, htail, y, hy0, hyS₂⟩
          · exact o_solver_serve _ hhead
          · exact hhead
          · exact ⟨hlit₁, fun S₂ he => hI.literais.2 S₂ (by rw [heq]; exact List.mem_cons_of_mem _ he)⟩
          · intro S₂ he; exact hI.pilhaHabitada S₂ (by rw [heq]; exact List.mem_cons_of_mem _ he)
          · constructor
            · intro e he
              have henc2 := hI.encadeada.2
              rw [heq] at henc2
              exact henc2.1 e he
            · have henc2 := hI.encadeada.2
              rw [heq] at henc2
              exact henc2.2
          · intro p hp2; exact hI.livroConsistente p hp2
          · -- ★ o livro sadio na subida: o argumento da região
            intro p hp2
            have hsad := hI.sadio p hp2
            cases hsad with
            | inl hdentro =>
              by_cases hramo : dentroDoPonto S₁ p
              · exact Or.inl hramo
              · -- o ramo ficou na região: ou morreu sem luz, ou a sua luz tem a divergência protegida
                by_cases hnl : mortoNoPonto φ p
                · exact Or.inr (Or.inr hnl)
                · unfold mortoNoPonto at hnl
                  push_neg at hnl
                  obtain ⟨y', hy'0, hy'⟩ := hnl
                  have hnaoS : ¬ serveEspectro2 y' c.esp = true := by
                    intro hsc
                    exact hescuro ⟨y', hy'0, hsc⟩
                  obtain ⟨suf, hsuf⟩ := hdentro
                  have hlitS : soLiterais ((p.1 ++ [(mascaraDeLiteral p.2, valorDeLiteral p.2)]) ++ suf) := by
                    rw [hsuf]; exact hI.literais.1
                  rw [← hsuf] at hnaoS
                  obtain ⟨D, l, rest, hD, hPD, hserveD, hnaoD⟩ :=
                    existe_divergencia y' _ suf hy' hnaoS hlitS
                  -- D' com luz, no livro: a protegida o guarda na pilha
                  have hpre : (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]) <+: c.esp := by
                    refine ⟨rest, ?_⟩
                    have htr : (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]) ++ rest =
                        D ++ (mascaraDeLiteral l, valorDeLiteral l) :: rest := List.append_assoc _ _ _
                    rw [htr, ← hD, hsuf]
                  have hlen : (fechoDoCampo φ (n + 2) []).length <
                      (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length := by
                    by_contra hn
                    have hle : (D ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length ≤
                        (fechoDoCampo φ (n + 2) []).length := by omega
                    have hpre2 : (D ++ [(mascaraDeLiteral l, valorDeLiteral l)])
                        <+: fechoDoCampo φ (n + 2) [] :=
                      prefixo_comparavel _ _ _ hpre hI.aForma.1 hle
                    have hyf : serveEspectro2 y' (fechoDoCampo φ (n + 2) []) = true := by
                      have h0 : serveEspectro2 y' ([] : Espectro2 n) = true := by
                        unfold serveEspectro2; simp
                      exact o_fecho_e_gratuito φ y' hy'0 (n + 2) [] h0
                    exact hnaoD (serve_prefixo _ _ y' hpre2 hyf)
                  have hlivroD : ∃ l' : Literal n, (D, l') ∈ c.livro :=
                    ⟨l, hI.aForma.2.2 D l hpre hlen⟩
                  have hprot := hI.protegida D hlivroD ⟨y', hy'0, hserveD⟩
                  cases hprot with
                  | inl hmem =>
                    rw [heq] at hmem
                    rw [List.mem_cons] at hmem
                    cases hmem with
                    | inl heq =>
                      subst heq
                      exact absurd hPD hramo
                    | inr htail =>
                      apply Or.inr
                      apply Or.inl
                      exact ⟨D, htail, hPD⟩
                  | inr heq =>
                    rw [heq] at hserveD
                    exact absurd ⟨y', hy'0, hserveD⟩ hescuro
            | inr hresto =>
              cases hresto with
              | inl hpil =>
                obtain ⟨e, he, hdentro⟩ := hpil
                rw [heq] at he
                rw [List.mem_cons] at he
                cases he with
                | inl heq => subst heq; exact Or.inl hdentro
                | inr htail => exact Or.inr (Or.inl ⟨e, htail, hdentro⟩)
              | inr hmorto => exact Or.inr (Or.inr hmorto)
          · intro D hD hluzD
            have hprot := hI.protegida D hD hluzD
            cases hprot with
            | inl hmem =>
              rw [heq] at hmem
              rw [List.mem_cons] at hmem
              cases hmem with
              | inl heq => subst heq; right; rfl
              | inr htail => left; exact htail
            | inr heq =>
              rw [heq] at hluzD
              exact absurd hluzD hescuro
          · constructor
            · exact hfecho₁
            · constructor
              · intro e he; exact hI.aForma.2.1 e (by rw [heq]; exact List.mem_cons_of_mem _ he)
              · intro D l hD hcomp
                exact hI.aForma.2.2 D l
                  (prefixo_trans _ _ _ hD (hI.encadeada.1 S₁ (by rw [heq]; exact List.mem_cons_self)))
                  hcomp
      · -- ★ A MORTE
        rename_i hnn
        split at hp
        · rename_i l hl
          have hcp : c' = ⟨c.esp ++ [(mascaraDeLiteral l, valorDeLiteral l)],
              ressurreicaoConstrutiva (c.esp ++ [(mascaraDeLiteral l, valorDeLiteral l)]),
              c.esp :: c.pilha, (c.esp, l) :: c.livro⟩ := (Option.some.inj hp).symm
          subst hcp
          have hlmem : l ∈ limpasNoveisDe φ c.livro c.esp C := o_melhor_de_mem φ c.esp _ l hl
          have hret : retornoUtil c.esp l = true := by
            unfold limpasNoveisDe at hlmem
            exact retornoUtil_das_limpas φ c.esp C l (List.mem_filter.mp hlmem).1
          have hcons' := a_porta_limpa_mantem_a_casa c.esp l hI.habitada hret
          refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
          · cases hI.luz with
            | inl hluz =>
              obtain ⟨y, hy0, hys⟩ := hluz
              cases hlv : l.vale y with
              | true => exact Or.inl ⟨y, hy0, serveEspectro2_append_de_vale _ _ l hys hlv⟩
              | false => exact Or.inr ⟨c.esp, List.mem_cons_self, y, hy0, hys⟩
            | inr hna =>
              obtain ⟨S₂, he₂, y, hy0, hyS₂⟩ := hna
              exact Or.inr ⟨S₂, List.mem_cons_of_mem _ he₂, y, hy0, hyS₂⟩
          · exact o_solver_serve _ hcons'
          · exact hcons'
          · constructor
            · intro e he
              rw [List.mem_append, List.mem_singleton] at he
              cases he with
              | inl hS => exact hI.literais.1 e hS
              | inr hS => subst hS; exact ⟨l, rfl⟩
            · intro S₂ he
              rw [List.mem_cons] at he
              cases he with
              | inl heq => subst heq; exact hI.literais.1
              | inr htail => exact hI.literais.2 S₂ htail
          · intro S₂ he
            rw [List.mem_cons] at he
            cases he with
            | inl heq => subst heq; exact hI.habitada
            | inr htail => exact hI.pilhaHabitada S₂ htail
          · constructor
            · intro e he
              rw [List.mem_cons] at he
              cases he with
              | inl heq =>
                subst heq
                exact ⟨[(mascaraDeLiteral l, valorDeLiteral l)], rfl⟩
              | inr htail =>
                exact prefixo_trans e c.esp _ (hI.encadeada.1 e htail)
                  ⟨[(mascaraDeLiteral l, valorDeLiteral l)], rfl⟩
            · exact ⟨hI.encadeada.1, hI.encadeada.2⟩
          · intro p hp2
            rw [List.mem_cons] at hp2
            cases hp2 with
            | inl heq =>
              subst heq
              exact ⟨hI.literais.1, hI.habitada⟩
            | inr htail => exact hI.livroConsistente p htail
          · intro p hp2
            rw [List.mem_cons] at hp2
            cases hp2 with
            | inl heq =>
              subst heq
              left
              exact ⟨[], by rw [List.append_nil]⟩
            | inr htail =>
              have hsad := hI.sadio p htail
              cases hsad with
              | inl hdentro =>
                left
                exact prefixo_trans _ _ _ hdentro
                  ⟨[(mascaraDeLiteral l, valorDeLiteral l)], rfl⟩
              | inr hresto =>
                cases hresto with
                | inl hpil =>
                  obtain ⟨e, he, hdentro⟩ := hpil
                  exact Or.inr (Or.inl ⟨e, List.mem_cons_of_mem _ he, hdentro⟩)
                | inr hmorto => exact Or.inr (Or.inr hmorto)
          · intro D hD hluzD
            obtain ⟨l', hl'⟩ := hD
            have hl2' := (List.mem_cons.mp hl' : (D, l') = (c.esp, l) ∨ (D, l') ∈ c.livro)
            cases hl2' with
            | inl heq =>
              rw [Prod.mk.injEq] at heq
              obtain ⟨h1, -⟩ := heq
              subst h1
              left
              exact List.mem_cons_self
            | inr htail =>
              have hprot := hI.protegida D ⟨l', htail⟩ hluzD
              cases hprot with
              | inl hmem => left; exact List.mem_cons_of_mem _ hmem
              | inr heq =>
                subst heq
                left
                exact List.mem_cons_self
          · constructor
            · exact prefixo_trans _ _ _ hI.aForma.1
                ⟨[(mascaraDeLiteral l, valorDeLiteral l)], rfl⟩
            · constructor
              · intro e he
                rw [List.mem_cons] at he
                cases he with
                | inl heq => subst heq; exact hI.aForma.1
                | inr htail => exact hI.aForma.2.1 e htail
              · intro D l₂ hD hcomp
                have hlen : (D ++ [(mascaraDeLiteral l₂, valorDeLiteral l₂)]).length ≤
                    (c.esp ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length :=
                  prefixo_length _ _ hD
                rw [List.length_append, List.length_singleton] at hlen
                cases Nat.lt_or_ge (D ++ [(mascaraDeLiteral l₂, valorDeLiteral l₂)]).length
                    (c.esp ++ [(mascaraDeLiteral l, valorDeLiteral l)]).length with
                | inl hlt =>
                  have hle : (D ++ [(mascaraDeLiteral l₂, valorDeLiteral l₂)]).length
                      ≤ c.esp.length := by
                    simp only [List.length_append, List.length_singleton] at hlt hlen ⊢
                    omega
                  have hpre : (D ++ [(mascaraDeLiteral l₂, valorDeLiteral l₂)]) <+: c.esp :=
                    prefixo_comparavel _ _ _ hD
                      ⟨[(mascaraDeLiteral l, valorDeLiteral l)], rfl⟩ hle
                  exact List.mem_cons_of_mem _ (hI.aForma.2.2 D l₂ hpre hcomp)
                | inr hge =>
                  have heqLen : D.length = c.esp.length := by
                    simp only [List.length_append, List.length_singleton] at hge hlen
                    omega
                  have heq2 := prefixo_de_mesmo_comprimento _ _ hD (by
                    simp only [List.length_append, List.length_singleton]
                    omega)
                  obtain ⟨hD2, hl2⟩ := List.append_inj heq2 heqLen
                  subst hD2
                  have hl3 : l₂ = l := eqLiteral_inj l₂ l (List.singleton_inj.mp hl2)
                  subst hl3
                  exact List.mem_cons_self
        · simp at hp

/-- ★ **O TERMINAL É IMPOSSÍVEL COM A LUZ**: com o invariante e a pilha vazia, o detector não pode
estar disparado — a luz no espectro exige porta nova (o bootstrap). -/
theorem o_terminal_impossivel {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (livro : List (Espectro2 n × Literal n))
    (hI : EstadoDaLuzComLivro φ ⟨S, x, [], livro⟩)
    (hterm : oSilencioTerminal φ S x livro) : False := by
  obtain ⟨hcusto, C, hC, hlimpas⟩ := hterm
  have hpoco : serveEspectro2 (descerGenerico passoFiel φ S φ.length ([], x)).2 S = true :=
    o_poco_fiel_serve φ S x hI.pocoServe
  obtain ⟨hCφ, hCfalha⟩ := a_primeira_sem_par_mem φ _ C hC
  cases hI.luz with
  | inl hluz =>
    exact a_luz_exige_limpa_nova φ S _ C hCφ hpoco hCfalha hI.literais.1 hluz livro []
      hI.sadio hI.encadeada.1 hI.livroConsistente hlimpas
  | inr hna =>
    obtain ⟨S₂, he₂, -⟩ := hna
    exact absurd he₂ List.not_mem_nil

/-- ★★★ **A LUZ NUNCA CALA: A FORMA FINAL** — com o invariante, o silêncio da máquina do livro ou é o
veredito terminal COM o invariante (que o bootstrap mata), ou desce ao fundo do combustível COM a luz
intacta E a contabilidade (cada passo pago em porta ou ponto). (A completude do veredito + a energia,
numa só cadeia.) -/
theorem a_luz_nunca_cala_o_veredito_final {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool) (pilha : List (Espectro2 n))
      (livro : List (Espectro2 n × Literal n)) (S' : Espectro2 n)
      (livro' : List (Espectro2 n × Literal n)),
      EstadoDaLuzComLivro φ ⟨S, x, pilha, livro⟩ →
      lacoComLivro φ fuel S x pilha livro = (none, S', livro') →
      (∃ (S₀ : Espectro2 n) (x₀ : Fin n → Bool) (livro₀ : List (Espectro2 n × Literal n)),
        EstadoDaLuzComLivro φ ⟨S₀, x₀, [], livro₀⟩ ∧ oSilencioTerminal φ S₀ x₀ livro₀) ∨
      (∃ (S₀ : Espectro2 n) (x₀ : Fin n → Bool) (pilha₀ : List (Espectro2 n))
        (livro₀ : List (Espectro2 n × Literal n)),
        EstadoDaLuzComLivro φ ⟨S₀, x₀, pilha₀, livro₀⟩ ∧
          lacoComLivro φ 0 S₀ x₀ pilha₀ livro₀ = (none, S', livro') ∧
          2 * livro₀.length + pilha.length ≥ fuel + 2 * livro.length) := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x pilha livro S' livro' hI h
    exact Or.inr ⟨S, x, pilha, livro, hI, h, by omega⟩
  | succ f ih =>
    intro S x pilha livro S' livro' hI h
    unfold lacoComLivro at h
    dsimp only [] at h
    split at h
    · rw [Prod.mk.injEq, Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      simp at h1
    · rename_i hcusto
      split at h
      · rename_i hsem
        exfalso
        have his := a_primeira_sem_par_existe φ _ hcusto
        rw [hsem] at his
        simp at his
      · rename_i C hC
        split at h
        · -- o detector disparou
          rename_i hlimpas
          split at h
          · -- ★ O SILÊNCIO TERMINAL — com o invariante
            left
            exact ⟨S, x, livro, hI, hcusto, C, hC, hlimpas⟩
          · -- a subida: o livro intacto, a pilha um a menos
            rename_i pilha₀ S₁ pilha₁
            have hstep : passoMaquina φ ⟨S, x, S₁ :: pilha₁, livro⟩ =
                some ⟨S₁, ressurreicaoConstrutiva S₁, pilha₁, livro⟩ := by
              unfold passoMaquina
              dsimp only []
              rw [if_neg hcusto]
              split
              · rename_i hsem
                rw [hsem] at hC
                simp at hC
              · rename_i C₂ hC2
                rw [hC2] at hC
                have hCC : C₂ = C := Option.some.inj hC
                rw [← hCC] at hlimpas
                split
                · rename_i hlimpas2
                  rfl
                · rename_i hnn
                  exact absurd hlimpas hnn
            have hI' := a_manutencao φ _ hI _ hstep
            have hih := ih S₁ (ressurreicaoConstrutiva S₁) pilha₁ livro S' livro' hI' h
            cases hih with
            | inl hterm => exact Or.inl hterm
            | inr hbot =>
              obtain ⟨S₀, x₀, pilha₀, livro₀, hI₀, h0, harit⟩ := hbot
              exact Or.inr ⟨S₀, x₀, pilha₀, livro₀, hI₀, h0, by
                rw [List.length_cons] at *
                omega⟩
        · -- a morte: o livro e a pilha crescem um
          rename_i hnn
          split at h
          · rename_i l hl
            have hstep : passoMaquina φ ⟨S, x, pilha, livro⟩ =
                some ⟨S ++ [(mascaraDeLiteral l, valorDeLiteral l)],
                  ressurreicaoConstrutiva (S ++ [(mascaraDeLiteral l, valorDeLiteral l)]),
                  S :: pilha, (S, l) :: livro⟩ := by
              unfold passoMaquina
              dsimp only []
              rw [if_neg hcusto]
              split
              · rename_i hsem
                rw [hsem] at hC
                simp at hC
              · rename_i C₂ hC2
                rw [hC2] at hC
                have hCC : C₂ = C := Option.some.inj hC
                rw [← hCC] at hl hnn
                split
                · rename_i hlimpas2
                  exact absurd hlimpas2 hnn
                · split
                  · rename_i l' hl'
                    rw [hl'] at hl
                    have hll : l' = l := Option.some.inj hl
                    subst hll
                    rfl
                  · rename_i hnone2
                    rw [hnone2] at hl
                    simp at hl
            have hI' := a_manutencao φ _ hI _ hstep
            have hih := ih _ _ _ _ _ _ hI' h
            cases hih with
            | inl hterm => exact Or.inl hterm
            | inr hbot =>
              obtain ⟨S₀, x₀, pilha₀, livro₀, hI₀, h0, harit⟩ := hbot
              exact Or.inr ⟨S₀, x₀, pilha₀, livro₀, hI₀, h0, by
                rw [List.length_cons, List.length_cons] at *
                omega⟩
          · rename_i hmelhor
            exfalso
            have his := o_melhor_de_nao_vazio φ S _ hnn
            rw [hmelhor] at his
            simp at his

/-- ★★★ **O VEREDITO DECIDE COM A COTA**: satisfazível, a máquina do livro com o combustível da luz
ECOA — o silêncio é impossível dentro da cota: o terminal é morto pelo bootstrap (o invariante), e o
fundo é morto pela contabilidade contra a cota do livro. A máquina do livro decide o lado do sim. -/
theorem o_veredito_decide_com_a_cota {n : ℕ} (φ : CNF n)
    (hSAT : SAT φ) (fuel : ℕ) (hcomb : 2 * aCotaDoLivro n + 1 ≤ fuel) :
    (processoComLivro φ fuel (fun _ => false)).1.isSome = true := by
  generalize hp : processoComLivro φ fuel (fun _ => false) = R
  obtain ⟨res, S', livro'⟩ := R
  cases res with
  | some w => rfl
  | none =>
    exfalso
    unfold processoComLivro at hp
    dsimp only [] at hp
    have hmain := a_luz_nunca_cala_o_veredito_final φ fuel _ _ [] [] S' livro'
      (o_estado_inicial_tem_luz φ hSAT) hp
    cases hmain with
    | inl hterm =>
      obtain ⟨S₀, x₀, livro₀, hI₀, hterm0⟩ := hterm
      exact o_terminal_impossivel φ S₀ x₀ livro₀ hI₀ hterm0
    | inr hbot =>
      obtain ⟨S₀, x₀, pilha₀, livro₀, hI₀, h0, harit⟩ := hbot
      -- o fundo devolve o próprio estado: livro' = livro₀
      have hlv : livro₀ = livro' := by
        unfold lacoComLivro at h0
        split at h0
        · simp at h0
        · rw [Prod.mk.injEq, Prod.mk.injEq] at h0
          obtain ⟨-, -, hlv⟩ := h0
          exact hlv
      rw [hlv] at harit
      have hcota := a_cota_do_livro φ fuel (fun _ => false) none S' livro' hp
      simp at harit
      omega

/-- ★★ **O VEREDITO É HONESTO**: a máquina do livro nunca inventa luz — o eco só sai do custo zero. -/
theorem o_veredito_e_honesto {n : ℕ} (φ : CNF n) (fuel : ℕ) (x w : Fin n → Bool)
    (h : (processoComLivro φ fuel x).1 = some w) : custoCNF φ w = 0 := by
  unfold processoComLivro at h
  dsimp only [] at h
  exact o_eco_do_livro_e_fiel φ _ _ _ _ _ _ _ _ (Prod.ext_iff.mpr ⟨h, rfl⟩)

/-- ★★ **A MEDIDA FINAL**: os muros, a família e a honestidade — a máquina do livro decide
o conjunto medido. -/
theorem a_medida_o_veredito_decide :
    (processoComLivro phiTrap 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiCE 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiFofa2 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiTrap2 20 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro phiCadeia 12 (fun _ => false)).1.isSome = true
    ∧ (processoComLivro ([[⟨0, true⟩], [⟨0, false⟩]] : CNF 1) 12 (fun _ => false)).1 = none := by
  decide

end TGLExt.ODentroOuMorto

