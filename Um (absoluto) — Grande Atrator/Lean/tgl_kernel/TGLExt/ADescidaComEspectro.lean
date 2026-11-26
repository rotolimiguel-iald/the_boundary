import TGLExt.AParedeDoPassoIngenuo

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A descida com espectro (camada ii): o estacionamento provado

## O que a pedra define

O passo da descida com o espectro no laço: o estado é o par (registro das sombras lidas, ponto). A sombra
de cada cláusula usada fica registrada, e o ponto estaciona quando não há cláusula sem par (o veredito),
quando a primeira cláusula sem par é vazia, ou quando a sombra da primeira cláusula sem par já foi lida
(o poço).

Leitura [INPUT/ONTO]: o estado da descida é o espectro (o rastro-neutrino), não o ponto; a leitura nunca
desmente a leitura.

## O que fica provado nesta camada [REAL no kernel]

1. `passo_ou_estaciona_ou_cresce`: a análise do passo; ou estaciona, ou registra a sombra nova de uma
   cláusula da fórmula e move o ponto.
2. `pinado_persiste` e `descer_add`: a pinagem persiste; a iteração compõe.
3. `descer_invariante`: o invariante do laço; as sombras lidas ficam distintas e dentro das sombras das
   cláusulas; ou já estacionou em algum passo `j ≤ k`, ou o comprimento do registro é o inicial mais `k`.
4. ★★ `a_descida_com_espectro_estaciona`: a terminação; em no máximo `|φ|` passos (o número de cláusulas),
   partindo do registro vazio, o estado alcançado é ponto fixo do passo. Pela definição do passo, esse ponto
   fixo é o veredito (nenhuma cláusula sem par), a cláusula vazia ou o poço [DERIVED da definição].

O registro das sombras é uma lista de listas de variáveis; não é o tipo `Espectro n` de `ADescidaPinada`, e
nenhuma `DinamicaComEspectro` é instanciada aqui.

Leitura [ONTO]: a morte sempre tem onde morrer.

## O que NÃO se afirma

* Não se afirma que o poço seja sempre alcançado: o teorema diz que a descida estaciona num ponto fixo. No
  poço há uma cláusula sem par; estacionar no poço a partir de um ponto não prova que a fórmula seja
  insatisfazível, e esta pedra não decide SAT. Sem o registro, o passo ingênuo pode subir ou ciclar
  (`AParedeDoPassoIngenuo`).
* O que acontece no poço (a morte, a ressurreição, o crescimento do espectro das equações) não é tratado
  aqui; os gates G e T seguem como hipóteses nomeadas no contrato da camada (i).
* O gate D′ (`gateDlinha`, em `AParedeDoPassoIngenuo`), o progresso da altura DENTRO de uma descida, não é
  tratado aqui; esta pedra prova a terminação.
* Nada sobre a natureza e nada sobre P versus NP: o gate do Instituto segue de pé e nomeado.

Sem `sorry` e sem declaração `axiom` neste arquivo.
-/

namespace TGLExt.ADescidaComEspectro

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo

variable {n : ℕ}

/-- [DEF] A sombra da cláusula: as variáveis que ela lê, na ordem do escriba. -/
def sombraDe {n : ℕ} (C : Clausula n) : List (Fin n) := C.map Literal.var

/-- [DEF] O estado da descida: o registro das sombras lidas e o ponto atual. -/
abbrev EstadoDescida (n : ℕ) := List (List (Fin n)) × (Fin n → Bool)

/-- [DEF] O passo com o espectro no laço: para na paz; estaciona na cláusula vazia (poço imediato);
estaciona no poço quando a sombra da primeira cláusula sem par já foi lida; senão registra a sombra e vira
o bit do primeiro literal. -/
def passoComEspectro {n : ℕ} (φ : CNF n) (leu : List (List (Fin n))) (x : Fin n → Bool) :
    EstadoDescida n :=
  match primeiraSemPar φ x with
  | none => (leu, x)
  | some [] => (leu, x)
  | some (l :: C') =>
    if sombraDe (l :: C') ∈ leu then (leu, x)
    else (sombraDe (l :: C') :: leu, virar l.var x)

/-- ★ A análise do passo: ou estaciona, ou registra a sombra nova de uma cláusula da fórmula e move. -/
theorem passo_ou_estaciona_ou_cresce {n : ℕ} (φ : CNF n) (leu : List (List (Fin n))) (x : Fin n → Bool) :
    passoComEspectro φ leu x = (leu, x) ∨
      ∃ (l : Literal n) (C' : Clausula n), (l :: C') ∈ φ ∧ sombraDe (l :: C') ∉ leu ∧
        passoComEspectro φ leu x = (sombraDe (l :: C') :: leu, virar l.var x) := by
  cases h : primeiraSemPar φ x with
  | none => exact Or.inl (by simp [passoComEspectro, h])
  | some C =>
    cases C with
    | nil => exact Or.inl (by simp [passoComEspectro, h])
    | cons l C' =>
      by_cases hs : sombraDe (l :: C') ∈ leu
      · exact Or.inl (by simp [passoComEspectro, h, hs])
      · exact Or.inr ⟨l, C', List.mem_of_find?_eq_some h, hs, by simp [passoComEspectro, h, hs]⟩

/-- [DEF] A descida com combustível: `k` aplicações do passo (cada aplicação é o passo do estado anterior). -/
def descerComEspectro {n : ℕ} (φ : CNF n) : ℕ → EstadoDescida n → EstadoDescida n
  | 0, st => st
  | k + 1, st => passoComEspectro φ (descerComEspectro φ k st).1 (descerComEspectro φ k st).2

/-- A pinagem persiste: estacionado, o estado não sai mais do lugar. -/
theorem pinado_persiste {n : ℕ} (φ : CNF n) (st : EstadoDescida n)
    (hp : passoComEspectro φ st.1 st.2 = st) :
    ∀ k : ℕ, descerComEspectro φ k st = st := by
  intro k
  induction k with
  | zero => rfl
  | succ k ih =>
    show passoComEspectro φ (descerComEspectro φ k st).1 (descerComEspectro φ k st).2 = st
    rw [ih]
    exact hp

/-- A iteração compõe: `b + a` passos são `a` passos depois de `b`. -/
theorem descer_add {n : ℕ} (φ : CNF n) (a b : ℕ) (st : EstadoDescida n) :
    descerComEspectro φ (b + a) st = descerComEspectro φ a (descerComEspectro φ b st) := by
  induction a with
  | zero => rfl
  | succ k ih =>
    rw [Nat.add_succ]
    have e1 : descerComEspectro φ (b + k + 1) st
        = passoComEspectro φ (descerComEspectro φ (b + k) st).1 (descerComEspectro φ (b + k) st).2 := rfl
    have e2 : descerComEspectro φ (k + 1) (descerComEspectro φ b st)
        = passoComEspectro φ (descerComEspectro φ k (descerComEspectro φ b st)).1
            (descerComEspectro φ k (descerComEspectro φ b st)).2 := rfl
    rw [e1, e2, ih]

/-- ★ O invariante do laço: as sombras lidas ficam distintas e dentro das sombras das cláusulas; ou já
estacionou em algum passo `j ≤ k`, ou o comprimento do registro é o inicial mais `k` (um por passo). -/
theorem descer_invariante {n : ℕ} (φ : CNF n) :
    ∀ (k : ℕ) (leu : List (List (Fin n))) (x : Fin n → Bool),
      leu.Nodup → leu ⊆ φ.map sombraDe →
      (descerComEspectro φ k (leu, x)).1.Nodup
        ∧ (descerComEspectro φ k (leu, x)).1 ⊆ φ.map sombraDe
        ∧ ((∃ j : ℕ, j ≤ k ∧
              passoComEspectro φ (descerComEspectro φ j (leu, x)).1 (descerComEspectro φ j (leu, x)).2
                = descerComEspectro φ j (leu, x))
            ∨ (descerComEspectro φ k (leu, x)).1.length = leu.length + k) := by
  intro k
  induction k with
  | zero => intro leu x hnd hsub; exact ⟨hnd, hsub, Or.inr rfl⟩
  | succ k ih =>
    intro leu x hnd hsub
    obtain ⟨hnd0, hsub0, hcas0⟩ := ih leu x hnd hsub
    cases passo_ou_estaciona_ou_cresce φ (descerComEspectro φ k (leu, x)).1
        (descerComEspectro φ k (leu, x)).2 with
    | inl hpark =>
      have hid : descerComEspectro φ (k + 1) (leu, x) = descerComEspectro φ k (leu, x) := hpark
      refine ⟨?_, ?_, Or.inl ⟨k, Nat.le_succ k, hpark⟩⟩
      · rw [hid]; exact hnd0
      · rw [hid]; exact hsub0
    | inr hmove =>
      obtain ⟨l, C', hmem, hfresh, hpasso⟩ := hmove
      have hnd1 : (sombraDe (l :: C') :: (descerComEspectro φ k (leu, x)).1).Nodup :=
        List.nodup_cons.mpr ⟨hfresh, hnd0⟩
      have hsub1 : (sombraDe (l :: C') :: (descerComEspectro φ k (leu, x)).1) ⊆ φ.map sombraDe := by
        intro s hs
        rw [List.mem_cons] at hs
        cases hs with
        | inl heq => rw [heq]; exact List.mem_map_of_mem hmem
        | inr hs' => exact hsub0 hs'
      have hid : descerComEspectro φ (k + 1) (leu, x)
          = (sombraDe (l :: C') :: (descerComEspectro φ k (leu, x)).1,
              virar l.var (descerComEspectro φ k (leu, x)).2) := hpasso
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

/-- ★★ **A DESCIDA COM ESPECTRO ESTACIONA**: com combustível igual ao número de cláusulas `|φ|`, partindo
do registro vazio, o estado alcançado é ponto fixo do passo: a descida está pinada no veredito (nenhuma
cláusula sem par) ou no poço (a primeira cláusula sem par é vazia ou tem a sombra já lida). Razão: cada
passo que não estaciona registra uma sombra nova, e as sombras registradas são distintas e estão entre as
sombras das cláusulas. Leitura [ONTO]: a morte sempre tem onde morrer. -/
theorem a_descida_com_espectro_estaciona {n : ℕ} (φ : CNF n) (x : Fin n → Bool) :
    passoComEspectro φ (descerComEspectro φ φ.length ([], x)).1
        (descerComEspectro φ φ.length ([], x)).2
      = descerComEspectro φ φ.length ([], x) := by
  obtain ⟨hnd, hsub, hcas⟩ :=
    descer_invariante φ φ.length [] x List.nodup_nil (List.nil_subset _)
  cases hcas with
  | inl hparked =>
    obtain ⟨j, hj, hp⟩ := hparked
    have hcom : φ.length = j + (φ.length - j) := (Nat.add_sub_of_le hj).symm
    have hid : descerComEspectro φ φ.length ([], x) = descerComEspectro φ j ([], x) := by
      rw [hcom, descer_add, pinado_persiste φ _ hp (φ.length - j)]
    rw [hid]
    exact hp
  | inr hlen =>
    have hlen0 : (descerComEspectro φ φ.length ([], x)).1.length = φ.length := by
      simp at hlen
      omega
    cases h : primeiraSemPar φ (descerComEspectro φ φ.length ([], x)).2 with
    | none => simp [passoComEspectro, h, Prod.eta]
    | some C =>
      cases C with
      | nil => simp [passoComEspectro, h, Prod.eta]
      | cons l C' =>
        have hmem : (l :: C') ∈ φ := List.mem_of_find?_eq_some h
        have hcontem : sombraDe (l :: C') ∈ (descerComEspectro φ φ.length ([], x)).1 := by
          by_contra hfora
          have hnd' : ((descerComEspectro φ φ.length ([], x)).1 ++ [sombraDe (l :: C')]).Nodup := by
            rw [List.nodup_append]
            refine ⟨hnd, List.nodup_singleton _, ?_⟩
            intro sombra hs1 sombra2 hmem2
            rw [List.mem_singleton] at hmem2
            subst hmem2
            intro hcontra
            subst hcontra
            exact hfora hs1
          have hsub' : (descerComEspectro φ φ.length ([], x)).1 ++ [sombraDe (l :: C')]
              ⊆ φ.map sombraDe := by
            intro s hs
            rw [List.mem_append] at hs
            cases hs with
            | inl hs1 => exact hsub hs1
            | inr hs2 =>
              rw [List.mem_singleton] at hs2
              rw [hs2]
              exact List.mem_map_of_mem hmem
          have hcompr := List.Subperm.length_le (List.Nodup.subperm hnd' hsub')
          rw [List.length_append, List.length_singleton, List.length_map, hlen0] at hcompr
          omega
        simp [passoComEspectro, h, hcontem, Prod.eta]

end TGLExt.ADescidaComEspectro

