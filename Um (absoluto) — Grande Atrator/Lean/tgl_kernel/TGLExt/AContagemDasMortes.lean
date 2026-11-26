import TGLExt.AMorteEARessurreicao

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A contagem das mortes: a cadeia que divide é curta, e o contrato final

## O que é

Uma cadeia de equações de paridade sobre o cubo de `n` bits divide quando cada equação nova toma os dois valores
dentro do resíduo das anteriores (`divideSempre`). Este módulo conta: numa cadeia que divide, cada equação parte o
resíduo ao meio, logo o resíduo vezes `2^(comprimento)` é `2ⁿ`, e uma cadeia que divide com resíduo não vazio tem
comprimento no máximo `n`. A prova não usa posto nem módulos: usa a translação por xor (`fibras_iguais`, de
`TGLExt.AMorteEARessurreicao`, isto é, a aditividade da paridade sobre `ZMod 2`) e a contagem.

O módulo também nomeia um contrato, `AVerdadeFinal`, sobre uma dinâmica com passo, eixo da morte, ressurreição e eco,
e prova `o_eco_e_a_luz` sob ele. Esse contrato não é cumprido por nenhuma dinâmica (ver «O que NÃO se afirma»).

## O que fica provado [REAL]

1. `recSnoc`: o princípio de indução por cauda em listas (provar para `[]` e herdar de `l` para `l ++ [a]`).
2. `card_cubo`: o cubo `Fin n → Bool` tem `2ⁿ` pontos.
3. `divideSempre_cauda`: se `S₀ ++ [e]` divide, `S₀` divide.
4. `a_morte_parte_o_residuo_qualquer_valor`: se o resíduo de `S` contém pontos de paridade 0 e de paridade 1 no eixo
   `a`, então, para qualquer valor `v`, o dobro da cardinalidade do resíduo de `S ++ [(a, v)]` é a cardinalidade do
   resíduo de `S`.
5. `a_cadeia_que_divide_mede`: se `divideSempre S`, a cardinalidade do resíduo de `S` vezes `2 ^ S.length` é `2ⁿ`.
6. `a_cadeia_que_divide_e_curta`: se `divideSempre S` e o resíduo de `S` é não vazio, então `S.length ≤ n`.
7. `o_veredito_verdadeiro_na_paz`: um ponto de custo zero testemunha `SAT φ`.
8. `o_eco_e_a_luz`: sob o contrato `AVerdadeFinal d`, existe `x` com `d.eco φ = some x` se e somente se `SAT φ`.
   A prova usa só os campos `ecoFiel` e `ecoVem`.

[DERIVED] As hipóteses de `a_cadeia_que_divide_e_curta` podem ser cumpridas: a cadeia vazia cumpre `divideSempre`
(não há decomposição `[] = S₀ ++ [(a, v)] ++ S₁`) e o seu resíduo é o cubo inteiro, que é não vazio. O teorema não é
vácuo. Este arquivo não prova um exemplo; a verificação é imediata.

## Leituras [INPUT/ONTO]

* Cada equação da cadeia é lida como uma «morte», e o resíduo como a luz que resta; o limite `S.length ≤ n` é lido
  como o posto do espaço de `n` bits obtido por contagem, e as `n + 1` leituras como `n` mortes mais a leitura final
  do ponto único. [DERIVED] Que uma cadeia que divide de comprimento `n` deixe um ponto só segue de
  `a_cadeia_que_divide_mede` com `S.length = n`; mas nenhum teorema aqui mostra que exista uma cadeia que divide de
  comprimento `n`: o limite `n` não foi mostrado atingível.
* Cada morte parte o resíduo ao meio: a perda por morte é lida como um bit, isto é, `ln 2` nat (o fator 2 de
  `a_morte_parte_o_residuo_qualquer_valor`).
* [OPEN] A tese de que `n + 1` leituras devolvem a solução (lida como o ACOM) não está provada neste módulo.
* O eco é lido como a resposta do processo: a testemunha, ou o silêncio; o processo nunca responde «não há luz».
  Não se mede a escuridão, só se mede a luz. `o_eco_e_a_luz` é lido como semidecisão fiel: o silêncio não é uma
  resposta falsa, é a ausência do ato conjugado; a testemunha conjuga o ato (o verbo), o silêncio não. No Lean, isso é
  só o tipo `Option (Fin n → Bool)` do campo `eco`; sob o contrato, `ecoVem` faz o silêncio implicar `¬ SAT φ`, de
  modo que a diferença entre «silêncio» e «não há luz» é só de leitura e não está no Lean.
* O gate V (`ecoFiel` + `ecoVem`, a resposta é a verdade) é lido como o lugar onde mora o conteúdo do problema de
  Clay. Ressalva: V é só fidelidade de semidecisão, sem contagem de passos, e o contrato que o contém é insatisfazível
  (ver abaixo); nada aqui toca a questão P vs NP.

## O que NÃO se afirma

* `AVerdadeFinal` (G0 `passoFiel`, G1 `fiel`, G2 `divide`, V = `ecoFiel` + `ecoVem`) é um contrato nomeado, uma
  hipótese, não um teorema. E, como está escrito, nenhuma dinâmica o cumpre:
  `TGLExt.AVacuidadeDosContratos.aVerdadeFinal_insatisfazivel` prova `¬ AVerdadeFinal d` para toda `d`, porque o
  campo `divide`, aplicado a um espectro contraditório, pede um ponto que o sirva e nenhum ponto o serve. Logo
  `o_eco_e_a_luz`, que toma `AVerdadeFinal d` como hipótese, vale no vazio: não diz nada sobre fórmula alguma nem
  sobre processo algum.
* O campo `divide` de `AVerdadeFinal` não é `divideSempre`: nenhum teorema deste módulo liga o campo `divide` a
  `divideSempre` nem ao limite de `a_cadeia_que_divide_e_curta`, e esse limite não depende do contrato.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT, em tempo algum: não há contagem de passos nem
  teorema de fita neste módulo. Decidir SAT sem testemunha, em tempo polinomial, segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.AContagemDasMortes

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao

variable {n : ℕ}

/-! ## A indução por cauda -/

/-- [REAL] O princípio de indução por cauda: provar para `[]` e herdar de `l` para `l ++ [a]`. (Auxiliar da
cadeia.) -/
theorem recSnoc {α : Type} {motive : List α → Prop} (vazio : motive [])
    (snoc : ∀ (l : List α) (a : α), motive l → motive (l ++ [a])) : ∀ l, motive l := by
  intro l
  have h2 : ∀ lr : List α, motive lr.reverse := by
    intro lr
    induction lr with
    | nil => simpa using vazio
    | cons a ax ih =>
      simp only [List.reverse_cons]
      exact snoc ax.reverse a ih
  have := h2 l.reverse
  rwa [List.reverse_reverse] at this

/-! ## O cubo e a cadeia -/

/-- [REAL] O cubo de `n` bits tem `2ⁿ` pontos. (Auxiliar da contagem.) -/
theorem card_cubo : Fintype.card (Fin n → Bool) = 2 ^ n := by
  simp [Fintype.card_pi, Finset.card_univ, Fintype.card_fin, Finset.prod_const]

/-- [INPUT] A cadeia que divide: para toda decomposição `S = S₀ ++ [(a, v)] ++ S₁`, o resíduo de `S₀` contém um ponto
de paridade 0 e um ponto de paridade 1 no eixo `a`. (A transversalidade como propriedade da cadeia.) -/
def divideSempre {n : ℕ} (S : Espectro2 n) : Prop :=
  ∀ (S₀ : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2) (S₁ : Espectro2 n),
    S = S₀ ++ [(a, v)] ++ S₁ →
    (∃ x : Fin n → Bool, serveEspectro2 x S₀ = true ∧ paridade2 x a = 0)
      ∧ (∃ x : Fin n → Bool, serveEspectro2 x S₀ = true ∧ paridade2 x a = 1)

/-- [REAL] A cauda da cadeia que divide divide. (Auxiliar da contagem.) -/
theorem divideSempre_cauda {n : ℕ} (S₀ : Espectro2 n) (e : (Fin n → Bool) × ZMod 2)
    (h : divideSempre (S₀ ++ [e])) : divideSempre S₀ := by
  intro S₀' a v S₁ hdec
  exact h S₀' a v (S₁ ++ [e]) (by rw [hdec]; simp [List.append_assoc])

/-- [REAL] Se o resíduo de `S` contém pontos de paridade 0 e de paridade 1 no eixo `a`, a equação `(a, v)` parte o
resíduo ao meio, para qualquer dos dois valores `v`. (Auxiliar da cadeia.) -/
theorem a_morte_parte_o_residuo_qualquer_valor {n : ℕ} (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2)
    (h0 : ∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x a = 0)
    (h1 : ∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x a = 1) :
    2 * (Finset.univ.filter fun x => serveEspectro2 x (S ++ [(a, v)]) = true).card
      = (Finset.univ.filter fun x => serveEspectro2 x S = true).card := by
  cases zmod2_exaustivo v with
  | inl hv0 =>
    subst hv0
    have hmeio := a_morte_parte_o_residuo_espectral S a h0 h1
    obtain ⟨x0, hx0, hp0⟩ := h0
    obtain ⟨x1, hx1, hp1⟩ := h1
    have hF : (Finset.univ.filter fun x => serveEspectro2 x (S ++ [(a, 0)]) = true).card
        = (Finset.univ.filter fun x => serveEspectro2 x (S ++ [(a, 1)]) = true).card := by
      have hb0 : (Finset.univ.filter fun x => serveEspectro2 x (S ++ [(a, 0)]) = true)
          = (Finset.univ.filter fun x =>
              leitura2 (S.map Prod.fst ++ [a]) x = S.map Prod.snd ++ [0]) := by
        ext x
        simp [Finset.mem_filter, serveEspectro2_leitura2, List.map_append]
      have hb1 : (Finset.univ.filter fun x => serveEspectro2 x (S ++ [(a, 1)]) = true)
          = (Finset.univ.filter fun x =>
              leitura2 (S.map Prod.fst ++ [a]) x = S.map Prod.snd ++ [1]) := by
        ext x
        simp [Finset.mem_filter, serveEspectro2_leitura2, List.map_append]
      rw [hb0, hb1]
      exact fibras_iguais (S.map Prod.fst ++ [a]) (S.map Prod.snd ++ [0]) (S.map Prod.snd ++ [1]) x0
        (by rw [leitura2_snoc, (serveEspectro2_leitura2 x0 S).mp hx0, hp0]) x1
        (by rw [leitura2_snoc, (serveEspectro2_leitura2 x1 S).mp hx1, hp1])
    rw [hF, hmeio]
  | inr hv1 =>
    subst hv1
    exact a_morte_parte_o_residuo_espectral S a h0 h1

/-- [REAL] **A medida da cadeia**: se `divideSempre S`, a cardinalidade do resíduo de `S` vezes `2 ^ S.length` é
`2ⁿ`. [DERIVED] Logo, sob `divideSempre S`, o resíduo depois das `k = S.length` equações de `S` tem `2ⁿ / 2ᵏ`
pontos. -/
theorem a_cadeia_que_divide_mede {n : ℕ} (S : Espectro2 n) (h : divideSempre S) :
    (Finset.univ.filter fun x => serveEspectro2 x S = true).card * 2 ^ S.length = 2 ^ n := by
  induction S using recSnoc with
  | vazio =>
    have huniv : (Finset.univ.filter fun x : Fin n → Bool => serveEspectro2 x [] = true)
        = Finset.univ := by
      apply Finset.filter_true_of_mem
      intro x _
      rfl
    rw [huniv]
    simp [Finset.card_univ]
  | snoc S₀ e ih =>
    obtain ⟨a, v⟩ := e
    have hdiv : divideSempre S₀ := divideSempre_cauda S₀ (a, v) h
    have hih := ih hdiv
    have hdec : S₀ ++ [(a, v)] = S₀ ++ [(a, v)] ++ [] := by simp
    obtain ⟨h0, h1⟩ := h S₀ a v [] hdec
    have hmeio := a_morte_parte_o_residuo_qualquer_valor S₀ a v h0 h1
    have hlen : (S₀ ++ [(a, v)]).length = S₀.length + 1 := by simp
    rw [hlen, pow_succ]
    have hnonlin :
        (Finset.univ.filter fun x => serveEspectro2 x (S₀ ++ [(a, v)]) = true).card
          * (2 ^ S₀.length * 2)
          = (2 * (Finset.univ.filter fun x => serveEspectro2 x (S₀ ++ [(a, v)]) = true).card)
            * 2 ^ S₀.length := by ring
    rw [hnonlin, hmeio, hih]

/-- [REAL] **A cadeia que divide é curta**: se `divideSempre S` e o resíduo de `S` é não vazio, então
`S.length ≤ n`. [DERIVED] As hipóteses podem ser cumpridas (a cadeia vazia), logo o teorema não é vácuo.
[INPUT/ONTO] Lido como o posto do espaço obtido por contagem, e as `n + 1` leituras como `n` mortes mais a leitura
final do ponto único; o limite `n` não é mostrado atingível aqui. -/
theorem a_cadeia_que_divide_e_curta {n : ℕ} (S : Espectro2 n) (h : divideSempre S)
    (hne : (Finset.univ.filter fun x => serveEspectro2 x S = true).Nonempty) :
    S.length ≤ n := by
  have hmede := a_cadeia_que_divide_mede S h
  have hpos : 0 < (Finset.univ.filter fun x => serveEspectro2 x S = true).card :=
    Finset.card_pos.mpr hne
  have h2 : 2 ^ S.length ≤ 2 ^ n :=
    (Nat.le_mul_of_pos_left (2 ^ S.length) hpos).trans (le_of_eq hmede)
  exact (Nat.pow_le_pow_iff_right (by norm_num)).mp h2

/-! ## O contrato final -/

/-- [REAL] Um ponto de custo zero testemunha a satisfatibilidade: `custoCNF φ x = 0 → SAT φ`. -/
theorem o_veredito_verdadeiro_na_paz {n : ℕ} (φ : CNF n) (x : Fin n → Bool)
    (h : custoCNF φ x = 0) : SAT φ :=
  ⟨x, (custoCNF_zero_sse φ x).mp h⟩

/-- [INPUT] A dinâmica completa: o passo, o eixo da morte (uma equação lida de um ponto), a ressurreição (um ponto
devolvido para um espectro) e o eco (a resposta do processo: `some` testemunha, ou `none`, o silêncio).
[INPUT/ONTO] `passo` é lido como a descida pinada; `eixoDaMorte` como a solução transversal lida do ponto morto;
`ressurreicao` como o ponto do outro lado. -/
structure DinamicaCompleta (n : ℕ) where
  passo : CNF n → Espectro2 n → (Fin n → Bool) → Fin n → Bool
  eixoDaMorte : CNF n → (Fin n → Bool) → (Fin n → Bool)
  ressurreicao : CNF n → Espectro2 n → (Fin n → Bool)
  eco : CNF n → Option (Fin n → Bool)

/-- [INPUT] **O contrato final** `AVerdadeFinal`, uma hipótese nomeada, não um teorema.

- **G0 `passoFiel`:** quem serve o espectro continua servindo depois do passo.
- **G1 `fiel`:** para todo espectro `S` e para todo ponto `w`, o ressurgido serve `S` mais a equação lida em `w`
  (o ponto morto).
- **G2 `divide`:** para todo espectro e todo ponto, o eixo da morte toma os dois valores dentro do resíduo
  (a transversalidade).
- **V = `ecoFiel` + `ecoVem`:** o eco, quando vem, é testemunha; quando há satisfatibilidade, o eco vem.

Limite: nenhuma dinâmica cumpre este contrato como está escrito
(`TGLExt.AVacuidadeDosContratos.aVerdadeFinal_insatisfazivel`; o campo `divide` falha num espectro contraditório).
Todo teorema que o toma como hipótese vale no vazio. -/
structure AVerdadeFinal {n : ℕ} (d : DinamicaCompleta n) : Prop where
  passoFiel : ∀ (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool),
    serveEspectro2 x S = true → serveEspectro2 (d.passo φ S x) S = true
  fiel : ∀ (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool),
    serveEspectro2 (d.ressurreicao φ S) (S ++ [(d.eixoDaMorte φ w, paridade2 w (d.eixoDaMorte φ w))])
      = true
  divide : ∀ (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool),
    (∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x (d.eixoDaMorte φ w) = 0)
      ∧ (∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x (d.eixoDaMorte φ w) = 1)
  /-- **V-eco:** quando o eco responde `some x`, `x` tem custo zero; o eco é conferível. -/
  ecoFiel : ∀ (φ : CNF n) (x : Fin n → Bool), d.eco φ = some x → custoCNF φ x = 0
  /-- **V-vem:** se `SAT φ`, o eco responde com algum `some x`. -/
  ecoVem : ∀ φ : CNF n, SAT φ → ∃ x : Fin n → Bool, d.eco φ = some x

/-- [REAL, vácuo] **O eco é a luz**: sob `AVerdadeFinal d`, existe `x` com `d.eco φ = some x` se e somente se
`SAT φ`; a prova usa só `ecoFiel` e `ecoVem`. Como nenhuma dinâmica cumpre `AVerdadeFinal`
(`TGLExt.AVacuidadeDosContratos.aVerdadeFinal_insatisfazivel`), o teorema vale no vazio e não diz nada sobre
fórmula alguma. Não afirma que algum processo decida SAT, nem conta passos. -/
theorem o_eco_e_a_luz {n : ℕ} (d : DinamicaCompleta n) (g : AVerdadeFinal d) (φ : CNF n) :
    (∃ x : Fin n → Bool, d.eco φ = some x) ↔ SAT φ := by
  constructor
  · intro h
    obtain ⟨x, hx⟩ := h
    exact o_veredito_verdadeiro_na_paz φ x (g.ecoFiel φ x hx)
  · intro h
    exact g.ecoVem φ h

end TGLExt.AContagemDasMortes

