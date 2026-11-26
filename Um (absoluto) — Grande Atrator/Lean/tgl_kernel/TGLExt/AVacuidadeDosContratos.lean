import TGLExt.OPassoFiel

set_option autoImplicit false

/-!
# A vacuidade dos contratos da descida, como estão escritos

## O que é

Um contrato nomeado é uma hipótese: um teorema que o toma diz algo sobre as dinâmicas que o cumprem. Se nenhuma
dinâmica o cumpre, o teorema vale no vazio e não diz nada sobre fórmula alguma. Este módulo mede, no kernel, quais
contratos e hipóteses da cadeia da descida podem ser cumpridos como estão escritos.

## O que fica provado [REAL]

1. `aVerdadeFinal_insatisfazivel`, `gatesDaMorte_insatisfazivel`, `gatesDaDescida_insatisfazivel`: nenhuma dinâmica
   cumpre `AVerdadeFinal`, `GatesDaMorte` ou `GatesDaDescida`. Os três pedem que um espectro contraditório seja
   servido: uma equação de eixo vazio com valor 1, que nenhum ponto serve (`ninguem_serve2`, `ninguem_serve`); o
   terceiro, também na fórmula da cláusula vazia, em que todo ponto tem custo 1.
2. `gates_do_processo_insatisfaziveis`: as hipóteses `hdiv` e `hpoco` dos dois teoremas do laço de `OProcessoDaLuz`
   (aqui `HDiv` e `HPoco`, com o mesmo enunciado) não valem juntas para nenhum `n` e nenhuma fórmula. `hpoco_falso`:
   com `n ≥ 1`, a descida do laço não recebe o espectro, e o mesmo poço teria de servir dois espectros que se
   contradizem. `hdiv_falso_na_paz`: numa fórmula com um ponto em paz, o eixo da morte desse ponto é nulo e a sua
   paridade nunca é 1. `hdiv_falso_n0`: com `n = 0`, toda paridade é 0.
3. `hfiel_habitado`: a hipótese `hfiel` (a ressurreição fiel) é cumprida por uma ressurreição que devolve um ponto do
   resíduo quando ele existe (escolha clássica).

## O que NÃO se afirma

* Nada aqui falsifica a descida, a morte ou a ressurreição: mede a redação dos contratos. Os teoremas das pedras que
  não tomam esses contratos como hipótese não são afetados.
* Nada aqui diz que algum processo decida SAT, em tempo algum. Decidir SAT sem testemunha, em tempo polinomial, na
  máquina de fita segue [OPEN].
* Nada sobre a natureza. O gate não se move.
-/

namespace TGLExt.AVacuidadeDosContratos

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel

/-- Nenhum ponto serve o espectro de uma equação com eixo vazio e valor 1 (a paridade no eixo vazio é 0). -/
theorem ninguem_serve2 {n : ℕ} (x : Fin n → Bool) :
    serveEspectro2 x [((fun _ => false), (1 : ZMod 2))] = false := by
  simp [serveEspectro2, paridade2]

/-- O mesmo na forma booleana da camada (i). -/
theorem ninguem_serve {n : ℕ} (x : Fin n → Bool) :
    serveEspectro x [⟨fun _ => false, true⟩] = false := by
  simp [serveEspectro, serveEquacao, paridade]

/-- `AVerdadeFinal` não é cumprido por nenhuma dinâmica: o campo `divide`, num espectro contraditório, pede um ponto
que o sirva. -/
theorem aVerdadeFinal_insatisfazivel {n : ℕ} (d : DinamicaCompleta n) : ¬ AVerdadeFinal d := by
  intro g
  obtain ⟨⟨x, hx, _⟩, _⟩ := g.divide [] [((fun _ => false), (1 : ZMod 2))] (fun _ => false)
  rw [ninguem_serve2] at hx
  exact Bool.false_ne_true hx

/-- `GatesDaMorte` não é cumprido por nenhuma dinâmica, pelo mesmo motivo. -/
theorem gatesDaMorte_insatisfazivel {n : ℕ} (d : DinamicaDaMorte n) : ¬ GatesDaMorte d := by
  intro g
  obtain ⟨⟨x, hx, _⟩, _⟩ := g.divide [] [((fun _ => false), (1 : ZMod 2))] (fun _ => false)
  rw [ninguem_serve2] at hx
  exact Bool.false_ne_true hx

/-- `GatesDaDescida` não é cumprido por nenhuma dinâmica: na fórmula da cláusula vazia todo ponto tem custo 1, o
progresso obriga a pinagem, e o crescimento pede que a ressurreição sirva um espectro contraditório. -/
theorem gatesDaDescida_insatisfazivel {n : ℕ} (d : DinamicaDaDescida n) (T : Prop) :
    ¬ GatesDaDescida d T := by
  intro g
  have hc : ∀ y : Fin n → Bool, custoCNF ([[]] : CNF n) y = 1 := by
    intro y; simp [custoCNF, satisfeita]
  have hpin : estaPinado d [[]] (fun _ => false) := by
    rcases g.progresso [[]] (fun _ => false) (by rw [hc]; omega) with h | h
    · exact h
    · rw [hc, hc] at h; omega
  have h := g.crescimento [[]] [⟨fun _ => false, true⟩] (fun _ => false) hpin (by rw [hc]; omega)
  simp [serveEspectro, serveEquacao, paridade] at h

/-- A hipótese `hdiv` de `OProcessoDaLuz`, com o mesmo enunciado. -/
def HDiv {n : ℕ} (φ : CNF n) : Prop :=
  ∀ (S : Espectro2 n) (w : Fin n → Bool),
    serveEspectro2 w S = true → (eixoDaMorteDe φ w) ∉ S.map Prod.fst →
    (∃ y : Fin n → Bool, serveEspectro2 y S = true ∧ paridade2 y (eixoDaMorteDe φ w) = 0)
      ∧ (∃ y : Fin n → Bool, serveEspectro2 y S = true ∧ paridade2 y (eixoDaMorteDe φ w) = 1)

/-- A hipótese `hpoco` de `OProcessoDaLuz`, com o mesmo enunciado (a descida do laço não recebe o espectro). -/
def HPoco {n : ℕ} (φ : CNF n) : Prop :=
  ∀ (S : Espectro2 n) (x : Fin n → Bool), (∃ y, serveEspectro2 y S = true) →
    serveEspectro2 (descerComEspectro φ φ.length ([], x)).2 S = true

/-- `HDiv` falha em toda fórmula com um ponto em paz: o eixo da morte desse ponto é a máscara vazia, cuja paridade é
sempre 0, e o espectro vazio o serve. -/
theorem hdiv_falso_na_paz {n : ℕ} (φ : CNF n) (w : Fin n → Bool) (hw : primeiraSemPar φ w = none) :
    ¬ HDiv φ := by
  intro h
  obtain ⟨-, ⟨y, -, hy⟩⟩ := h [] w (by simp [serveEspectro2]) (by simp)
  simp [eixoDaMorteDe, hw, paridade2] at hy

/-- `HDiv` falha em toda fórmula com `n = 0`: a paridade em `Fin 0` é a soma vazia. -/
theorem hdiv_falso_n0 (φ : CNF 0) : ¬ HDiv φ := by
  intro h
  obtain ⟨-, ⟨y, -, hy⟩⟩ := h [] (fun i => i.elim0) (by simp [serveEspectro2]) (by simp)
  simp [paridade2] at hy

/-- `HPoco` falha em toda fórmula com `n ≥ 1`: os espectros `[(e0, 0)]` e `[(e0, 1)]` são ambos servidos, e o mesmo
poço, que não depende do espectro, teria de servir os dois. -/
theorem hpoco_falso {m : ℕ} (φ : CNF (m + 1)) : ¬ HPoco φ := by
  intro h
  let e0 : Fin (m + 1) → Bool := fun i => decide (i = 0)
  have hp0 : paridade2 (fun _ => false) e0 = 0 := by simp [paridade2]
  have hp1 : paridade2 e0 e0 = 1 := by simp [paridade2, e0, Finset.sum_ite_eq']
  have h0 := h [(e0, 0)] (fun _ => false) ⟨fun _ => false, by simp [serveEspectro2, hp0]⟩
  have h1 := h [(e0, 1)] (fun _ => false) ⟨e0, by simp [serveEspectro2, hp1]⟩
  simp only [serveEspectro2, List.all_cons, List.all_nil, Bool.and_true, beq_iff_eq] at h0 h1
  rw [h0] at h1
  exact absurd h1 (by decide)

/-- ★ As hipóteses `hdiv` e `hpoco` do laço não valem juntas para nenhum `n` e nenhuma fórmula. -/
theorem gates_do_processo_insatisfaziveis : ∀ (n : ℕ) (φ : CNF n), ¬ (HDiv φ ∧ HPoco φ)
  | 0, φ => fun h => hdiv_falso_n0 φ h.1
  | _ + 1, φ => fun h => hpoco_falso φ h.2

/-- A ressurreição que devolve um ponto do resíduo, quando ele existe (escolha clássica). -/
noncomputable def ressurreicaoPorEscolha {n : ℕ} (_φ : CNF n) (S : Espectro2 n) : Fin n → Bool := by
  classical
  exact if h : ∃ y, serveEspectro2 y S = true then Classical.choose h else fun _ => false

/-- A hipótese `hfiel` (a ressurreição fiel) é cumprida pela ressurreição por escolha. -/
theorem hfiel_habitado {n : ℕ} (φ : CNF n) :
    ∀ (S : Espectro2 n) (w : Fin n → Bool),
      serveEspectro2 w S = true → (eixoDaMorteDe φ w) ∉ S.map Prod.fst →
      serveEspectro2 (ressurreicaoPorEscolha φ (S ++ [(eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))]))
        (S ++ [(eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))]) = true := by
  intro S w hw _
  have hex : ∃ y, serveEspectro2 y (S ++ [(eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))]) = true :=
    ⟨w, by simp [serveEspectro2, List.all_append] at hw ⊢; exact hw⟩
  unfold ressurreicaoPorEscolha
  rw [dif_pos hex]
  exact Classical.choose_spec hex

end TGLExt.AVacuidadeDosContratos
