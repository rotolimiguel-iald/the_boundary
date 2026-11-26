import TGLExt.OProcessoDaLuz

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O passo fiel: o passo que só move dentro do resíduo, com a terminação e a preservação do espectro

O passo fiel `passoFiel φ S` olha a primeira cláusula sem par de `x`; estaciona se não há cláusula sem par,
se a cláusula é vazia, se a sua sombra já foi lida, ou se virar a variável do primeiro literal faria o ponto
deixar de servir o espectro `S`; caso contrário registra a sombra nova e vira essa variável. A descida
`descerGenerico passo2 φ S k st` itera um passo `passo2` k vezes a partir de `st`.

## O que fica provado nesta pedra [REAL]

1. A descida genérica, para qualquer passo `passo2`:
   * `pinado_persiste_generico`: se `st` é ponto fixo do passo, a descida fica em `st` para todo `k`.
   * `descer_add_generico`: `b + a` passos são `b` passos seguidos de `a` passos.
   * `descer_invariante_generico`: sob a hipótese `hAPI` (cada passo ou devolve o próprio estado, ou
     acrescenta à lista lida a sombra, ainda não lida, de uma cláusula de `φ`), para todo `k`, toda lista
     inicial `leu` e todo ponto `x`: se `leu` é sem repetição (`leu.Nodup`) e está contida em
     `φ.map sombraDe`, então, após `k` passos a partir de `(leu, x)`, a lista lida segue sem repetição e
     contida em `φ.map sombraDe`, e ou a descida já passou por um ponto fixo em algum `j ≤ k`, ou o
     comprimento da lista lida é `leu.length + k`.
   * `a_descida_generica_estaciona`: sob `hAPI`, partindo de `([], x)`, o estado após `φ.length` passos é
     ponto fixo do passo.
2. `passoFiel_ou_estaciona_ou_cresce`: o passo fiel cumpre `hAPI`.
3. `a_descida_fiel_estaciona`: o estado após `φ.length` passos fiéis, a partir de `([], x)`, é ponto fixo
   de `passoFiel φ S`.
4. `o_passo_fiel_preserva`: se `x` serve `S`, o ponto devolvido por `passoFiel φ S leu x` serve `S`.
5. `a_descida_fiel_preserva`: se `x` serve `S`, todo ponto de `descerGenerico passoFiel φ S k ([], x)`
   serve `S`, para todo `k`.
6. `o_poco_fiel_serve`: se `x` serve `S`, o ponto após `φ.length` passos fiéis serve `S` (o caso
   `k = φ.length` do item 5).

## Leitura [INPUT]

A descida fiel lê-se como a descida pinada: a leitura nunca desmente a leitura; o passo só se move
dentro do resíduo do espectro. Nenhum teorema desta pedra liga `descerGenerico passoFiel` às definições
de `TGLExt.ADescidaPinada`; a frase é leitura, não identidade provada. «Estacionar» é o nome dado ao
ponto fixo do passo.

## Os gates nomeados da descida (contratos, não teoremas)

* G0, o poço serve o espectro: no laço de `OProcessoDaLuz`, é a hipótese `hpoco` (sobre a descida
  `descerComEspectro φ`). Esta pedra prova um enunciado vizinho para a descida fiel (itens 4 a 6), não
  `hpoco`.
* G1, a ressurreição fiel: a hipótese `hfiel` do laço; pode ser cumprida
  (`TGLExt.AVacuidadeDosContratos.hfiel_habitado`).
* G2, a leitura divide: a hipótese `hdiv` do laço.
* V-vem: quando há luz, o eco vem.
* T: o transporte na fita.

## O que NÃO se afirma

* O ponto fixo da descida fiel não é afirmado satisfazer `φ`: o passo fiel também estaciona quando virar a
  variável quebraria o espectro, ou quando a sombra já foi lida. Nada aqui decide SAT, nem dá custo de
  tempo na fita.
* `o_poco_fiel_serve` é sobre a descida `descerGenerico passoFiel φ S`, que recebe o espectro `S`; não é
  sobre a descida do laço `descerComEspectro φ` de `OProcessoDaLuz`, que não recebe o espectro. Não
  descarrega a hipótese `hpoco` (G0) do laço; não se afirma que `hpoco` fique provada para o passo fiel.
  `hpoco` é falsa para toda fórmula com `n ≥ 1` (`TGLExt.AVacuidadeDosContratos.hpoco_falso`: a descida
  do laço não recebe o espectro), e `hdiv` (G2) e `hpoco` juntas não valem para nenhum `n` e nenhuma
  fórmula (`TGLExt.AVacuidadeDosContratos.gates_do_processo_insatisfaziveis`); um teorema que as tome
  juntas como hipótese vale por vacuidade e nada diz sobre fórmula alguma. `hfiel` (G1) pode ser cumprida
  (`hfiel_habitado`).
* Esta pedra não toma como hipótese `hdiv`, `hpoco`, `GatesDaDescida`, `GatesDaMorte` nem `AVerdadeFinal`;
  a única hipótese dos teoremas genéricos, `hAPI`, é cumprida pelo passo fiel
  (`passoFiel_ou_estaciona_ou_cresce`).
* `o_poco_fiel_serve` pede que o ponto de partida `x` já sirva `S`; não afirma que exista tal ponto.
* Não se afirmam aqui G1, G2, V-vem nem T.
* O gate do Instituto segue de pé e nomeado: decidir SAT sem testemunha, em tempo polinomial, na máquina
  de fita segue [OPEN].
* Nada sobre a natureza.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OPassoFiel

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz

variable {n : ℕ}

/-! ## A descida genérica (a terminação para qualquer passo com a propriedade da API) -/

/-- [DEF] A descida genérica com combustível. -/
def descerGenerico {n : ℕ}
    (passo2 : CNF n → Espectro2 n → List (List (Fin n)) → (Fin n → Bool) → EstadoDescida n)
    (φ : CNF n) (S : Espectro2 n) : ℕ → EstadoDescida n → EstadoDescida n
  | 0, st => st
  | k + 1, st => passo2 φ S (descerGenerico passo2 φ S k st).1 (descerGenerico passo2 φ S k st).2

/-- A pinagem persiste, na forma genérica. -/
theorem pinado_persiste_generico {n : ℕ}
    (passo2 : CNF n → Espectro2 n → List (List (Fin n)) → (Fin n → Bool) → EstadoDescida n)
    (φ : CNF n) (S : Espectro2 n) (st : EstadoDescida n)
    (hp : passo2 φ S st.1 st.2 = st) :
    ∀ k : ℕ, descerGenerico passo2 φ S k st = st := by
  intro k
  induction k with
  | zero => rfl
  | succ k ih =>
    show passo2 φ S (descerGenerico passo2 φ S k st).1 (descerGenerico passo2 φ S k st).2 = st
    rw [ih]
    exact hp

/-- A iteração compõe, na forma genérica. -/
theorem descer_add_generico {n : ℕ}
    (passo2 : CNF n → Espectro2 n → List (List (Fin n)) → (Fin n → Bool) → EstadoDescida n)
    (φ : CNF n) (S : Espectro2 n) (a b : ℕ) (st : EstadoDescida n) :
    descerGenerico passo2 φ S (b + a) st = descerGenerico passo2 φ S a (descerGenerico passo2 φ S b st) := by
  induction a with
  | zero => rfl
  | succ k ih =>
    rw [Nat.add_succ]
    have e1 : descerGenerico passo2 φ S (b + k + 1) st
        = passo2 φ S (descerGenerico passo2 φ S (b + k) st).1
            (descerGenerico passo2 φ S (b + k) st).2 := rfl
    have e2 : descerGenerico passo2 φ S (k + 1) (descerGenerico passo2 φ S b st)
        = passo2 φ S (descerGenerico passo2 φ S k (descerGenerico passo2 φ S b st)).1
            (descerGenerico passo2 φ S k (descerGenerico passo2 φ S b st)).2 := rfl
    rw [e1, e2, ih]

/-- O invariante do laço, na forma genérica. -/
theorem descer_invariante_generico {n : ℕ}
    (passo2 : CNF n → Espectro2 n → List (List (Fin n)) → (Fin n → Bool) → EstadoDescida n)
    (φ : CNF n) (S : Espectro2 n)
    (hAPI : ∀ (leu : List (List (Fin n))) (x : Fin n → Bool),
      passo2 φ S leu x = (leu, x) ∨
        ∃ (l : Literal n) (C' : Clausula n), (l :: C') ∈ φ ∧ sombraDe (l :: C') ∉ leu ∧
          ∃ y : Fin n → Bool, passo2 φ S leu x = (sombraDe (l :: C') :: leu, y)) :
    ∀ (k : ℕ) (leu : List (List (Fin n))) (x : Fin n → Bool),
      leu.Nodup → leu ⊆ φ.map sombraDe →
      (descerGenerico passo2 φ S k (leu, x)).1.Nodup
        ∧ (descerGenerico passo2 φ S k (leu, x)).1 ⊆ φ.map sombraDe
        ∧ ((∃ j : ℕ, j ≤ k ∧
              passo2 φ S (descerGenerico passo2 φ S j (leu, x)).1 (descerGenerico passo2 φ S j (leu, x)).2
                = descerGenerico passo2 φ S j (leu, x))
            ∨ (descerGenerico passo2 φ S k (leu, x)).1.length = leu.length + k) := by
  intro k
  induction k with
  | zero => intro leu x hnd hsub; exact ⟨hnd, hsub, Or.inr rfl⟩
  | succ k ih =>
    intro leu x hnd hsub
    obtain ⟨hnd0, hsub0, hcas0⟩ := ih leu x hnd hsub
    cases hAPI (descerGenerico passo2 φ S k (leu, x)).1 (descerGenerico passo2 φ S k (leu, x)).2 with
    | inl hpark =>
      have hid : descerGenerico passo2 φ S (k + 1) (leu, x) = descerGenerico passo2 φ S k (leu, x) :=
        hpark
      refine ⟨?_, ?_, Or.inl ⟨k, Nat.le_succ k, hpark⟩⟩
      · rw [hid]; exact hnd0
      · rw [hid]; exact hsub0
    | inr hmove =>
      obtain ⟨l, C', hmem, hfresh, y, hpasso⟩ := hmove
      have hnd1 : (sombraDe (l :: C') :: (descerGenerico passo2 φ S k (leu, x)).1).Nodup :=
        List.nodup_cons.mpr ⟨hfresh, hnd0⟩
      have hsub1 : (sombraDe (l :: C') :: (descerGenerico passo2 φ S k (leu, x)).1) ⊆ φ.map sombraDe := by
        intro s hs
        rw [List.mem_cons] at hs
        cases hs with
        | inl heq => rw [heq]; exact List.mem_map_of_mem hmem
        | inr hs' => exact hsub0 hs'
      have hid : descerGenerico passo2 φ S (k + 1) (leu, x)
          = (sombraDe (l :: C') :: (descerGenerico passo2 φ S k (leu, x)).1, y) := hpasso
      refine ⟨?_, ?_, ?_⟩
      · rw [hid]; exact hnd1
      · rw [hid]; exact hsub1
      · cases hcas0 with
        | inl hparked =>
          obtain ⟨j, hj, hp⟩ := hparked
          exact Or.inl ⟨j, by omega, hp⟩
        | inr hlen0 =>
          right
          rw [hid, List.length_cons, hlen0]
          omega

/-- A terminação genérica: para qualquer passo que ou devolve o próprio estado ou registra sombra nova da
fórmula, o estado após `φ.length` passos a partir de `([], x)` é ponto fixo do passo. -/
theorem a_descida_generica_estaciona {n : ℕ}
    (passo2 : CNF n → Espectro2 n → List (List (Fin n)) → (Fin n → Bool) → EstadoDescida n)
    (φ : CNF n) (S : Espectro2 n)
    (hAPI : ∀ (leu : List (List (Fin n))) (x : Fin n → Bool),
      passo2 φ S leu x = (leu, x) ∨
        ∃ (l : Literal n) (C' : Clausula n), (l :: C') ∈ φ ∧ sombraDe (l :: C') ∉ leu ∧
          ∃ y : Fin n → Bool, passo2 φ S leu x = (sombraDe (l :: C') :: leu, y))
    (x : Fin n → Bool) :
    passo2 φ S (descerGenerico passo2 φ S φ.length ([], x)).1
        (descerGenerico passo2 φ S φ.length ([], x)).2
      = descerGenerico passo2 φ S φ.length ([], x) := by
  obtain ⟨hnd, hsub, hcas⟩ :=
    descer_invariante_generico passo2 φ S hAPI φ.length [] x List.nodup_nil (List.nil_subset _)
  cases hcas with
  | inl hparked =>
    obtain ⟨j, hj, hp⟩ := hparked
    have hcom : φ.length = j + (φ.length - j) := (Nat.add_sub_of_le hj).symm
    have hid : descerGenerico passo2 φ S φ.length ([], x) = descerGenerico passo2 φ S j ([], x) := by
      rw [hcom, descer_add_generico, pinado_persiste_generico passo2 φ S _ hp (φ.length - j)]
    rw [hid]
    exact hp
  | inr hlen =>
    have hlen0 : (descerGenerico passo2 φ S φ.length ([], x)).1.length = φ.length := by
      simp at hlen
      omega
    cases hAPI (descerGenerico passo2 φ S φ.length ([], x)).1
        (descerGenerico passo2 φ S φ.length ([], x)).2 with
    | inl hp => exact hp
    | inr hmove =>
      obtain ⟨l, C', hmem, hfresh, y, hpasso⟩ := hmove
      have hsombra : sombraDe (l :: C') ∈ φ.map sombraDe := List.mem_map_of_mem hmem
      have htudo : φ.map sombraDe ⊆ (descerGenerico passo2 φ S φ.length ([], x)).1 := by
        intro s hs
        by_contra hfora
        have hnd' : ((descerGenerico passo2 φ S φ.length ([], x)).1 ++ [s]).Nodup := by
          rw [List.nodup_append]
          refine ⟨hnd, List.nodup_singleton _, ?_⟩
          intro sombra2 hs1 sombra3 hmem2
          rw [List.mem_singleton] at hmem2
          subst hmem2
          intro hcontra
          subst hcontra
          exact hfora hs1
        have hsub' : (descerGenerico passo2 φ S φ.length ([], x)).1 ++ [s] ⊆ φ.map sombraDe := by
          intro s2 hs2
          rw [List.mem_append] at hs2
          cases hs2 with
          | inl h1 => exact hsub h1
          | inr h2 =>
            rw [List.mem_singleton] at h2
            rw [h2]
            exact hs
        have hcompr := List.Subperm.length_le (List.Nodup.subperm hnd' hsub')
        rw [List.length_append, List.length_singleton, List.length_map, hlen0] at hcompr
        omega
      exact absurd (htudo hsombra) hfresh

/-! ## O passo fiel -/

/-- [DEF] O passo fiel: só move dentro do resíduo; estaciona na paz, na cláusula vazia, na sombra já lida,
ou quando o movimento faria o ponto deixar de servir o espectro. -/
def passoFiel {n : ℕ} (φ : CNF n) (S : Espectro2 n) (leu : List (List (Fin n))) (x : Fin n → Bool) :
    EstadoDescida n :=
  match primeiraSemPar φ x with
  | none => (leu, x)
  | some [] => (leu, x)
  | some (l :: C') =>
    if sombraDe (l :: C') ∈ leu then (leu, x)
    else if serveEspectro2 (virar l.var x) S then (sombraDe (l :: C') :: leu, virar l.var x)
    else (leu, x)

/-- A análise do passo fiel: ou estaciona, ou registra a sombra nova e move dentro do resíduo. -/
theorem passoFiel_ou_estaciona_ou_cresce {n : ℕ} (φ : CNF n) (S : Espectro2 n)
    (leu : List (List (Fin n))) (x : Fin n → Bool) :
    passoFiel φ S leu x = (leu, x) ∨
      ∃ (l : Literal n) (C' : Clausula n), (l :: C') ∈ φ ∧ sombraDe (l :: C') ∉ leu ∧
        ∃ y : Fin n → Bool, passoFiel φ S leu x = (sombraDe (l :: C') :: leu, y) := by
  cases h : primeiraSemPar φ x with
  | none => exact Or.inl (by simp [passoFiel, h])
  | some C =>
    cases C with
    | nil => exact Or.inl (by simp [passoFiel, h])
    | cons l C' =>
      by_cases hs : sombraDe (l :: C') ∈ leu
      · exact Or.inl (by simp [passoFiel, h, hs])
      · by_cases hf : serveEspectro2 (virar l.var x) S = true
        · exact Or.inr ⟨l, C', List.mem_of_find?_eq_some h, hs, virar l.var x,
            by simp [passoFiel, h, hs, hf]⟩
        · exact Or.inl (by simp [passoFiel, h, hs, hf])

/-- A descida fiel estaciona: o estado após `φ.length` passos fiéis é ponto fixo do passo fiel (a
terminação genérica instanciada). O ponto fixo não é afirmado satisfazer `φ`. -/
theorem a_descida_fiel_estaciona {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool) :
    passoFiel φ S (descerGenerico passoFiel φ S φ.length ([], x)).1
        (descerGenerico passoFiel φ S φ.length ([], x)).2
      = descerGenerico passoFiel φ S φ.length ([], x) :=
  a_descida_generica_estaciona passoFiel φ S (passoFiel_ou_estaciona_ou_cresce φ S) x

/-- O passo fiel preserva o espectro: se `x` serve `S`, o ponto depois do passo continua servindo `S`. -/
theorem o_passo_fiel_preserva {n : ℕ} (φ : CNF n) (S : Espectro2 n) (leu : List (List (Fin n)))
    (x : Fin n → Bool) (hx : serveEspectro2 x S = true) :
    serveEspectro2 (passoFiel φ S leu x).2 S = true := by
  cases h : primeiraSemPar φ x with
  | none => simpa [passoFiel, h] using hx
  | some C =>
    cases C with
    | nil => simpa [passoFiel, h] using hx
    | cons l C' =>
      by_cases hs : sombraDe (l :: C') ∈ leu
      · simpa [passoFiel, h, hs] using hx
      · by_cases hf : serveEspectro2 (virar l.var x) S = true
        · simpa [passoFiel, h, hs, hf] using hf
        · simpa [passoFiel, h, hs, hf] using hx

/-- A descida fiel preserva o espectro: começando de um ponto que serve `S`, todo ponto da descida fiel
serve `S`. -/
theorem a_descida_fiel_preserva {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (hx : serveEspectro2 x S = true) :
    ∀ k : ℕ, serveEspectro2 (descerGenerico passoFiel φ S k ([], x)).2 S = true := by
  intro k
  induction k with
  | zero => exact hx
  | succ k ih =>
    show serveEspectro2 (passoFiel φ S (descerGenerico passoFiel φ S k ([], x)).1
        (descerGenerico passoFiel φ S k ([], x)).2).2 S = true
    exact o_passo_fiel_preserva φ S _ _ ih

/-- O poço fiel serve o espectro: se `x` serve `S`, o ponto após `φ.length` passos fiéis serve `S`.
É sobre a descida fiel `descerGenerico passoFiel φ S`, não sobre a descida do laço `descerComEspectro φ`:
não descarrega a hipótese `hpoco` de `OProcessoDaLuz` (falsa para `n ≥ 1`,
`TGLExt.AVacuidadeDosContratos.hpoco_falso`). -/
theorem o_poco_fiel_serve {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (hx : serveEspectro2 x S = true) :
    serveEspectro2 (descerGenerico passoFiel φ S φ.length ([], x)).2 S = true :=
  a_descida_fiel_preserva φ S x hx φ.length

end TGLExt.OPassoFiel

