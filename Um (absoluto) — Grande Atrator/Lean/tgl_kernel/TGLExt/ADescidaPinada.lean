import TGLExt.TraducaoTuring

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A descida pinada: o estado-espectro, o estacionamento e a meia-nat espectral

## O que a pedra define

O estado-espectro: uma equação espectral é um eixo (as variáveis lidas) e um valor lido; um ponto serve a
equação quando a paridade dos seus bits acesos no eixo é o valor; o espectro é a lista das equações lidas,
e um ponto serve o espectro quando serve cada equação. A iteração de um passo sobre os pontos, a involução
que vira um bit, a dinâmica da descida (passo, eixo da morte, ressurreição) e o contrato `GatesDaDescida`.

Leitura [INPUT/ONTO]: o rastro da imagem é o neutrino, o gradiente, a descida pinada; o passo remove o que
não tem par; no poço o ponto morre pagando um bit e ressurge do outro lado, lido do espectro acumulado; o
eixo da morte é lido do ponto morto, não escolhido. A remoção fixa o eixo da projeção transversal no
ângulo oposto (a primeira lei); pagar um bit é o acoplamento não mínimo; a ressurreição se dá com
integridade suficiente; a dinâmica é a máquina fixada antes da fórmula. Essas leituras não são teoremas.

## O que fica provado [REAL no kernel]

1. `serveEspectro_cons`: servir `e :: S` é servir `e` e servir `S`. Leitura [ONTO]: a leitura nunca
   desmente a leitura.
2. `a_descida_estaciona` (com o auxiliar `a_descida_estaciona_aux`): sob a hipótese `hprog` (fora da altura
   zero, o passo ou está pinado ou reduz `custoCNF φ` estritamente), existe `k ≤ custoCNF φ x` tal que o
   ponto `iterar passo k x` tem altura zero ou é ponto fixo do passo (o poço).
3. `virar_duplo` e `paridade_virar`: virar duas vezes o bit `j` devolve o ponto; com `a j = true`, virar o
   bit `j` troca a paridade no eixo `a`.
4. ★ `paridade_equilibrada`: com `a j = true` para algum `j`, os pontos do cubo `Fin n → Bool` com paridade
   `true` no eixo `a` e os com paridade `false` têm o mesmo cardinal (a involução `virar j` é a bijeção).
   Leitura [ONTO]: a meia-nat espectral; uma leitura com eixo povoado corta o cubo ao meio, vale um bit.
5. O contrato `GatesDaDescida`, com os campos D (`progresso`), G (`crescimento`) e T (`transporte`), é uma
   estrutura de hipóteses; nenhum campo é provado aqui.

## O que NÃO se afirma

* O contrato `GatesDaDescida`, como está escrito, não é cumprido por NENHUMA dinâmica e nenhum `T`:
  `TGLExt.AVacuidadeDosContratos.gatesDaDescida_insatisfazivel` [REAL]. Na fórmula da cláusula vazia todo
  ponto tem custo 1, o progresso obriga a pinagem, e o crescimento pede que a ressurreição sirva um espectro
  contraditório. Logo um teorema que tome `GatesDaDescida` como hipótese vale no vazio e não diz nada sobre
  fórmula alguma. Nenhum teorema deste arquivo o toma como hipótese.
* `paridade_equilibrada` fala do cubo inteiro, não de um espaço residual recortado por um espectro: não se
  prova aqui que uma equação nova divida ao meio os pontos que servem um espectro já lido.
* A hipótese `hprog` de `a_descida_estaciona` é cumprida por qualquer passo que fique sempre pinado (por
  exemplo a identidade), e então `k = 0` já serve [DERIVED]. O teorema não diz que o ponto alcançado tenha
  altura zero: o poço é ponto fixo, não veredito.
* A forma clássica da descida (a busca local) paga exponencial no pior caso [KNOWN]; isso não se prova
  aqui. As paredes do passo ingênuo estão medidas em `TGLExt.AparedeDoPassoIngenuo` [REAL]:
  `parede_d_estrito_medida` (um passo que leva a altura de 1 a 2), `parede_do_ciclo_medida` (um ciclo de
  dois pontos de altura 1) e `o_gate_D_estrito_e_falso_para_o_passo_ingenuo` (o gate D estrito falha para
  esse passo em `n = 2`).
* Nada aqui decide SAT nem dá cota de tempo na máquina de fita; o campo T é uma proposição arbitrária, não
  um teorema sobre a fita. Decidir SAT sem testemunha, em tempo polinomial, na máquina de fita segue [OPEN];
  o gate do Instituto segue de pé e nomeado.
* Nada sobre a natureza. O gate não se move.

Sem `sorry` e sem declaração `axiom` neste arquivo.
-/

namespace TGLExt.ADescidaPinada

open TGLExt.TraducaoTuring

variable {n : ℕ}

/-! ## O estado-espectro -/

/-- [DEF] A equação espectral: o eixo (as variáveis lidas) e o valor lido. Leitura [ONTO]: é o que a morte
lê no poço. -/
structure EquacaoEspectral (n : ℕ) where
  eixo : Fin n → Bool
  valor : Bool

/-- [DEF] A paridade do ponto no eixo: a paridade do número de bits acesos onde o eixo está aceso. -/
def paridade {n : ℕ} (x : Fin n → Bool) (a : Fin n → Bool) : Bool :=
  ((Finset.univ.filter fun i => a i && x i).card % 2 == 1)

/-- [DEF] O ponto serve a equação: a paridade no eixo é o valor lido. -/
def serveEquacao {n : ℕ} (x : Fin n → Bool) (e : EquacaoEspectral n) : Bool :=
  paridade x e.eixo == e.valor

/-- [DEF] O espectro: a lista das leituras feitas. Leitura [ONTO]: cada morte acrescenta uma equação. -/
abbrev Espectro (n : ℕ) := List (EquacaoEspectral n)

/-- [DEF] O ponto serve o espectro: serve cada equação. (O espaço residual é o conjunto dos que servem.) -/
def serveEspectro {n : ℕ} (x : Fin n → Bool) (S : Espectro n) : Bool :=
  S.all (serveEquacao x)

/-- [REAL] Servir mais uma equação é servir a equação e o espectro. -/
theorem serveEspectro_cons {n : ℕ} (x : Fin n → Bool) (e : EquacaoEspectral n) (S : Espectro n) :
    serveEspectro x (e :: S) = (serveEquacao x e && serveEspectro x S) := by
  simp [serveEspectro, List.all_cons]

/-! ## O estacionamento dinâmico -/

/-- [DEF] A iteração do passo: `k` aplicações do passo ao ponto. -/
def iterar (passo : (Fin n → Bool) → Fin n → Bool) : ℕ → (Fin n → Bool) → (Fin n → Bool)
  | 0, x => x
  | k + 1, x => iterar passo k (passo x)

/-- [REAL] Auxiliar do estacionamento, por indução numa cota `m` da altura. -/
theorem a_descida_estaciona_aux (passo : (Fin n → Bool) → Fin n → Bool) (φ : CNF n)
    (hprog : ∀ x, custoCNF φ x ≠ 0 → passo x = x ∨ custoCNF φ (passo x) < custoCNF φ x) :
    ∀ m : ℕ, ∀ x : Fin n → Bool, custoCNF φ x ≤ m →
      ∃ k : ℕ, k ≤ custoCNF φ x ∧
        (custoCNF φ (iterar passo k x) = 0 ∨ passo (iterar passo k x) = iterar passo k x) := by
  intro m
  induction m with
  | zero =>
    intro x hx
    have h0 : custoCNF φ x = 0 := Nat.eq_zero_of_le_zero hx
    exact ⟨0, Nat.zero_le _, Or.inl h0⟩
  | succ m ihm =>
    intro x hx
    by_cases h0 : custoCNF φ x = 0
    · exact ⟨0, Nat.zero_le _, Or.inl h0⟩
    · cases hprog x h0 with
      | inl hp => exact ⟨0, Nat.zero_le _, Or.inr hp⟩
      | inr hdec =>
        have hle : custoCNF φ (passo x) ≤ m := by omega
        obtain ⟨k, hk, hkc⟩ := ihm (passo x) hle
        exact ⟨k + 1, by omega, hkc⟩

/-- ★ [REAL] **A DESCIDA ESTACIONA**: com um passo que, fora da altura zero, ou está pinado ou reduz a
altura estritamente, existe `k ≤ custoCNF φ x` tal que o ponto `iterar passo k x` tem altura zero (o
veredito) ou é ponto fixo do passo (o poço). A altura é o custo da fórmula do kernel. -/
theorem a_descida_estaciona (passo : (Fin n → Bool) → Fin n → Bool) (φ : CNF n)
    (hprog : ∀ x, custoCNF φ x ≠ 0 → passo x = x ∨ custoCNF φ (passo x) < custoCNF φ x)
    (x : Fin n → Bool) :
    ∃ k : ℕ, k ≤ custoCNF φ x ∧
      (custoCNF φ (iterar passo k x) = 0 ∨ passo (iterar passo k x) = iterar passo k x) :=
  a_descida_estaciona_aux passo φ hprog (custoCNF φ x) x (le_refl _)

/-! ## A meia-nat espectral -/

/-- [DEF] A involução da fibra: virar o bit `j` do ponto. -/
def virar {n : ℕ} (j : Fin n) (x : Fin n → Bool) : Fin n → Bool := Function.update x j (! x j)

/-- [REAL] Virar duas vezes devolve o ponto. -/
theorem virar_duplo {n : ℕ} (j : Fin n) (x : Fin n → Bool) : virar j (virar j x) = x := by
  ext i
  by_cases h : i = j
  · subst h; simp [virar, Function.update_self, Bool.not_not]
  · simp [virar, Function.update_of_ne h]

/-- [REAL] O filtro da paridade sob a involução: com o bit `j` aceso no eixo, o filtro do ponto virado é o
filtro do ponto menos `j` (se `x j`) ou mais `j` (se não). Auxiliar de `paridade_virar`. -/
theorem filtro_virar {n : ℕ} (x a : Fin n → Bool) (j : Fin n) (hj : a j = true) :
    (Finset.univ.filter fun i => a i && virar j x i)
      = (if x j then (Finset.univ.filter fun i => a i && x i).erase j
        else insert j (Finset.univ.filter fun i => a i && x i)) := by
  ext i
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_erase, Finset.mem_insert]
  by_cases h : i = j
  · subst h
    cases hx : x i <;> simp [virar, Function.update_self, hj, hx]
  · have hupd : virar j x i = x i := Function.update_of_ne h _ _
    rw [hupd]
    by_cases hx : x j <;> simp [hx, h]

/-- ★ [REAL] Virar um bit do eixo troca a paridade. -/
theorem paridade_virar {n : ℕ} (x a : Fin n → Bool) (j : Fin n) (hj : a j = true) :
    paridade (virar j x) a = ! paridade x a := by
  unfold paridade
  rw [filtro_virar x a j hj]
  by_cases hx : x j
  · have hmem : j ∈ Finset.univ.filter (fun i => a i && x i) := by
      simp [Finset.mem_filter, hj, hx]
    rw [if_pos hx, Finset.card_erase_of_mem hmem]
    set c := (Finset.univ.filter (fun i => a i && x i)).card with hc_def
    have hpos : 1 ≤ c := Finset.card_pos.mpr ⟨j, hmem⟩
    have h2 := Nat.mod_two_eq_zero_or_one c
    cases h2 with
    | inl h0 =>
      have hc1 : (c - 1) % 2 = 1 := by omega
      simp [h0, hc1]
    | inr h1 =>
      have hc1 : (c - 1) % 2 = 0 := by omega
      simp [h1, hc1]
  · have hnmem : j ∉ Finset.univ.filter (fun i => a i && x i) := by
      simp [Finset.mem_filter, hx]
    rw [if_neg (by simp [hx]), Finset.card_insert_of_notMem hnmem]
    set c := (Finset.univ.filter (fun i => a i && x i)).card with hc_def
    have h2 := Nat.mod_two_eq_zero_or_one c
    cases h2 with
    | inl h0 =>
      have hc1 : (c + 1) % 2 = 1 := by omega
      simp [h0, hc1]
    | inr h1 =>
      have hc1 : (c + 1) % 2 = 0 := by omega
      simp [h1, hc1]

/-- ★★ [REAL] **A MEIA-NAT ESPECTRAL**: com um eixo povoado (`a j = true`), os pontos do cubo com paridade
`true` e os com paridade `false` têm o mesmo cardinal; a involução de virar o bit `j` é a bijeção. O
enunciado é sobre o cubo inteiro, não sobre o espaço residual de um espectro. Leitura [ONTO]: uma leitura
com eixo povoado corta o cubo ao meio, vale um bit. -/
theorem paridade_equilibrada {n : ℕ} (a : Fin n → Bool) (j : Fin n) (hj : a j = true) :
    Fintype.card {x : Fin n → Bool // paridade x a = true}
      = Fintype.card {x : Fin n → Bool // paridade x a = false} := by
  apply Fintype.card_congr
  refine ⟨fun x => ⟨virar j x.1, ?_⟩, fun y => ⟨virar j y.1, ?_⟩, ?_, ?_⟩
  · rw [paridade_virar x.1 a j hj]; simp [x.2]
  · rw [paridade_virar y.1 a j hj]; simp [y.2]
  · intro x; exact Subtype.ext (virar_duplo j x.1)
  · intro y; exact Subtype.ext (virar_duplo j y.1)

/-! ## O contrato dos gates -/

/-- [DEF] A dinâmica da descida: o passo, o eixo da morte e a ressurreição, como funções quaisquer da
fórmula. Leitura [ONTO]: o passo é a remoção do que não tem par; o eixo da morte é a solução transversal no
poço, lida do ponto morto; a ressurreição é o ponto do outro lado, lido do espectro acumulado. -/
structure DinamicaDaDescida (n : ℕ) where
  passo : CNF n → (Fin n → Bool) → Fin n → Bool
  eixoDaMorte : CNF n → (Fin n → Bool) → (Fin n → Bool)
  ressurreicao : CNF n → Espectro n → (Fin n → Bool)

/-- [DEF] O poço é o estacionamento dinâmico: o passo pinado não move o ponto. -/
def estaPinado {n : ℕ} (d : DinamicaDaDescida n) (φ : CNF n) (x : Fin n → Bool) : Prop :=
  d.passo φ x = x

/-- ★★ [DEF] **OS GATES DA DESCIDA**: um contrato de hipóteses nomeadas; nenhum campo é provado aqui.

- **D (`progresso`):** fora da altura zero, o passo ou está pinado ou reduz a altura estritamente.
- **G (`crescimento`):** para todo espectro `S` e todo ponto pinado `x` de altura não nula, a ressurreição
  em `S` serve `S` acrescido da equação nova (o eixo da morte de `x` e a paridade de `x` nesse eixo).
- **T (`transporte`):** uma proposição dada como parâmetro; leitura [INPUT]: o custo de computar o passo,
  o eixo e a ressurreição na fita.

Limite [REAL]: nenhuma dinâmica e nenhum `T` cumprem este contrato como está escrito
(`TGLExt.AVacuidadeDosContratos.gatesDaDescida_insatisfazivel`); um teorema que o tome como hipótese vale
no vazio e não diz nada sobre fórmula alguma. -/
structure GatesDaDescida {n : ℕ} (d : DinamicaDaDescida n) (T : Prop) : Prop where
  progresso : ∀ (φ : CNF n) (x : Fin n → Bool),
    custoCNF φ x ≠ 0 → estaPinado d φ x ∨ custoCNF φ (d.passo φ x) < custoCNF φ x
  crescimento : ∀ (φ : CNF n) (S : Espectro n) (x : Fin n → Bool),
    estaPinado d φ x → custoCNF φ x ≠ 0 →
    serveEspectro (d.ressurreicao φ S)
      (S ++ [⟨d.eixoDaMorte φ x, paridade x (d.eixoDaMorte φ x)⟩]) = true
  transporte : T

end TGLExt.ADescidaPinada

