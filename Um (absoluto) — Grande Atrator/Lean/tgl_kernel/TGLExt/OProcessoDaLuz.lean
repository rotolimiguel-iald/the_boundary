import TGLExt.AContagemDasMortes

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O processo da luz: o laço completo e o eco fiel por construção

## O que é

O laço descer–morrer–ressurgir (`lacoDoProcesso`): desce com o espectro (`descerComEspectro`); se o ponto alcançado
tem custo zero, responde com ele; se o eixo da morte desse ponto já está no espectro, devolve `none`; senão acrescenta
ao espectro a equação da morte (o eixo e a paridade do ponto) e recomeça a partir do ponto devolvido pela
ressurreição `R`, com um combustível a menos. A resposta está em `Option`; o espectro final é devolvido junto.
A descida usada no laço, com combustível `φ.length` e registro vazio, estaciona num ponto fixo do passo
(`TGLExt.ADescidaComEspectro.a_descida_com_espectro_estaciona`) [REAL, no módulo citado].

Leitura [INPUT/ONTO]: o processo nunca responde «não há luz»; responde com o eco ou não responde. Não se mede a
escuridão, só se mede a luz. `none` não afirma que a fórmula não tem testemunha.

As três hipóteses nomeadas dos itens 3 e 4 [INPUT]: `hdiv` (a divisão na morte: a equação da morte toma os valores
0 e 1 em pontos que servem o espectro), `hfiel` (a ressurreição fiel: o ponto devolvido por `R` serve o espectro
acrescido) e `hpoco` (o poço que serve o espectro: o ponto alcançado pela descida serve todo espectro que tem
algum ponto que o serve).

## O que fica provado

1. [REAL] `o_eco_do_processo_e_fiel`, sem hipótese alguma: se `lacoDoProcesso R φ fuel S x = (some w, S')`, então
   `custoCNF φ w = 0`. O laço confere o custo antes de responder.
2. [REAL] `divideSempre_snoc`: se `S` é uma cadeia que divide e a equação nova toma os valores 0 e 1 em pontos que
   servem `S`, então `S ++ [(a, v)]` é uma cadeia que divide.
3. [REAL, vale no vazio; ver abaixo] `o_processo_preserva_a_cadeia`: sob as hipóteses `hdiv`, `hfiel` e `hpoco`,
   para todo combustível, todo `S` que divide e tem um ponto que o serve, o espectro final do laço divide e tem um
   ponto que o serve.
4. [REAL, vale no vazio; ver abaixo] `o_processo_tem_as_mortes_contadas`: sob as mesmas três hipóteses e as mesmas
   condições sobre `S`, o espectro final do laço tem comprimento no máximo `n`.

## O que NÃO se afirma

* As hipóteses `hdiv` e `hpoco` não valem juntas para nenhum `n` e nenhuma fórmula
  (`TGLExt.AVacuidadeDosContratos.gates_do_processo_insatisfaziveis`). Com `n ≥ 1`, `hpoco` é falsa sozinha
  (`TGLExt.AVacuidadeDosContratos.hpoco_falso`): a descida do laço não recebe o espectro, e o mesmo ponto teria de
  servir dois espectros que se contradizem. Por isso `o_processo_preserva_a_cadeia` e
  `o_processo_tem_as_mortes_contadas` valem no vazio: não dizem nada sobre fórmula alguma. A hipótese `hfiel` pode
  ser cumprida (`TGLExt.AVacuidadeDosContratos.hfiel_habitado`).
* Não se prova que, com combustível `n + 1`, o ramo do esgotamento do combustível (`| 0, S, _ => (none, S)`) seja
  inalcançável; isso segue [OPEN].
* `OPassoFiel.o_poco_fiel_serve` trata da descida `descerGenerico passoFiel φ S`, não da descida do laço
  `descerComEspectro φ`; não cumpre `hpoco`.
* Não se prova que, quando a fórmula tem testemunha, o laço responde com ela: a completude segue [OPEN].
  Só a fidelidade da resposta (item 1) está provada.
* Nada aqui diz que algum processo, ou a máquina de fita, decida SAT em tempo polinomial; isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OProcessoDaLuz

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes

variable {n : ℕ}

/-- [INPUT] O laço do processo: desce com o espectro (`descerComEspectro`); se o ponto alcançado tem custo zero,
responde com ele; se o eixo da morte desse ponto já está no espectro, devolve `none`; senão acrescenta a equação
da morte (o eixo e a paridade do ponto) e recomeça do ponto devolvido por `R`, com um combustível a menos.
Devolve a resposta e o espectro final. A descida interna estaciona num ponto fixo do passo
(`ADescidaComEspectro.a_descida_com_espectro_estaciona`). -/
def lacoDoProcesso (R : CNF n → Espectro2 n → (Fin n → Bool)) (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → (Option (Fin n → Bool)) × Espectro2 n
  | 0, S, _ => (none, S)
  | fuel + 1, S, x =>
    let w := (descerComEspectro φ φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S)
    else
      let a := eixoDaMorteDe φ w
      if a ∈ S.map Prod.fst then (none, S)
      else lacoDoProcesso R φ fuel (S ++ [(a, paridade2 w a)]) (R φ (S ++ [(a, paridade2 w a)]))

/-- [REAL] O eco é fiel por construção: se o laço responde com `some w`, então `custoCNF φ w = 0`. Sem
hipótese: o laço confere o custo antes de responder. -/
theorem o_eco_do_processo_e_fiel (R : CNF n → Espectro2 n → (Fin n → Bool)) (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool) (S' : Espectro2 n),
      lacoDoProcesso R φ fuel S x = (some w, S') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w S' h
    simp [lacoDoProcesso] at h
  | succ f ih =>
    intro S x w S' h
    unfold lacoDoProcesso at h
    simp only [] at h
    split at h
    · rename_i hcond
      rw [Prod.mk.injEq] at h
      obtain ⟨h1, -⟩ := h
      have hw := Option.some.inj h1
      subst hw
      exact hcond
    · split at h
      · simp at h
      · rename_i hcn hsombra
        exact ih _ _ _ _ h

/-- [REAL] A cadeia cresce pela equação que divide: se `S` divide e a equação nova toma os valores 0 e 1 em pontos
que servem `S`, então `S ++ [(a, v)]` divide. (Auxiliar do laço.) -/
theorem divideSempre_snoc {n : ℕ} (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2)
    (hS : divideSempre S)
    (hdiv : (∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x a = 0)
      ∧ (∃ x : Fin n → Bool, serveEspectro2 x S = true ∧ paridade2 x a = 1)) :
    divideSempre (S ++ [(a, v)]) := by
  intro S₀ a' v' S₁ hdec
  by_cases hS₁ : S₁ = []
  · subst hS₁
    rw [List.append_nil] at hdec
    obtain ⟨h1, h2⟩ := append_snod_inj hdec
    rw [Prod.mk.injEq] at h2
    obtain ⟨haa, hvv⟩ := h2
    subst haa
    subst hvv
    subst h1
    exact hdiv
  · have hex : ∃ S₁' u, S₁ = S₁' ++ [u] := by
      cases hr : S₁.reverse with
      | nil =>
        simp [List.reverse_eq_nil_iff] at hr
        exact absurd hr hS₁
      | cons u R => exact ⟨R.reverse, u, by rw [← List.reverse_reverse S₁, hr]; simp⟩
    obtain ⟨S₁', u, rfl⟩ := hex
    rw [show S₀ ++ [(a', v')] ++ (S₁' ++ [u]) = (S₀ ++ [(a', v')] ++ S₁') ++ [u] from by
      simp [List.append_assoc]] at hdec
    obtain ⟨hS0, -⟩ := append_snod_inj hdec
    exact hS S₀ a' v' S₁' hS0

/-- [REAL, condicional] Sob `hdiv`, `hfiel` e `hpoco`: se `S` divide e tem um ponto que o serve, o espectro final
do laço divide e tem um ponto que o serve. `hdiv` e `hpoco` não valem juntas para nenhum `n` e nenhuma fórmula
(`AVacuidadeDosContratos.gates_do_processo_insatisfaziveis`); o teorema vale no vazio. -/
theorem o_processo_preserva_a_cadeia (R : CNF n → Espectro2 n → (Fin n → Bool)) (φ : CNF n)
    (hdiv : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      serveEspectro2 w S = true → (eixoDaMorteDe φ w) ∉ S.map Prod.fst →
      (∃ y : Fin n → Bool, serveEspectro2 y S = true ∧ paridade2 y (eixoDaMorteDe φ w) = 0)
        ∧ (∃ y : Fin n → Bool, serveEspectro2 y S = true ∧ paridade2 y (eixoDaMorteDe φ w) = 1))
    (hfiel : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      serveEspectro2 w S = true → (eixoDaMorteDe φ w) ∉ S.map Prod.fst →
      serveEspectro2 (R φ (S ++ [(eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))]))
        (S ++ [(eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))]) = true)
    (hpoco : ∀ (S : Espectro2 n) (x : Fin n → Bool), (∃ y, serveEspectro2 y S = true) →
      serveEspectro2 (descerComEspectro φ φ.length ([], x)).2 S = true) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      divideSempre S → (∃ y, serveEspectro2 y S = true) →
      divideSempre (lacoDoProcesso R φ fuel S x).2
        ∧ (∃ y, serveEspectro2 y (lacoDoProcesso R φ fuel S x).2 = true) := by
  intro fuel
  induction fuel with
  | zero => intro S x hS hserv; exact ⟨hS, hserv⟩
  | succ f ih =>
    intro S x hS hserv
    unfold lacoDoProcesso
    simp only []
    split
    · exact ⟨hS, hserv⟩
    · split
      · exact ⟨hS, hserv⟩
      · rename_i hcn hsombra
        have hw : serveEspectro2 (descerComEspectro φ φ.length ([], x)).2 S = true := hpoco S x hserv
        have hdivS := hdiv S _ hw hsombra
        have hnova : divideSempre (S ++ [(eixoDaMorteDe φ (descerComEspectro φ φ.length ([], x)).2,
            paridade2 (descerComEspectro φ φ.length ([], x)).2
              (eixoDaMorteDe φ (descerComEspectro φ φ.length ([], x)).2))]) :=
          divideSempre_snoc S _ _ hS hdivS
        have hnovaServ : ∃ y, serveEspectro2 y (S ++ [(eixoDaMorteDe φ
            (descerComEspectro φ φ.length ([], x)).2,
            paridade2 (descerComEspectro φ φ.length ([], x)).2
              (eixoDaMorteDe φ (descerComEspectro φ φ.length ([], x)).2))]) = true :=
          ⟨_, hfiel S _ hw hsombra⟩
        exact ih _ _ hnova hnovaServ

/-- [REAL, condicional] Sob `hdiv`, `hfiel` e `hpoco`, e se `S` divide e tem um ponto que o serve, o espectro final
do laço tem comprimento no máximo `n`. `hdiv` e `hpoco` não valem juntas para nenhum `n` e nenhuma fórmula
(`AVacuidadeDosContratos.gates_do_processo_insatisfaziveis`); o teorema vale no vazio. -/
theorem o_processo_tem_as_mortes_contadas (R : CNF n → Espectro2 n → (Fin n → Bool)) (φ : CNF n)
    (hdiv : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      serveEspectro2 w S = true → (eixoDaMorteDe φ w) ∉ S.map Prod.fst →
      (∃ y : Fin n → Bool, serveEspectro2 y S = true ∧ paridade2 y (eixoDaMorteDe φ w) = 0)
        ∧ (∃ y : Fin n → Bool, serveEspectro2 y S = true ∧ paridade2 y (eixoDaMorteDe φ w) = 1))
    (hfiel : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      serveEspectro2 w S = true → (eixoDaMorteDe φ w) ∉ S.map Prod.fst →
      serveEspectro2 (R φ (S ++ [(eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))]))
        (S ++ [(eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))]) = true)
    (hpoco : ∀ (S : Espectro2 n) (x : Fin n → Bool), (∃ y, serveEspectro2 y S = true) →
      serveEspectro2 (descerComEspectro φ φ.length ([], x)).2 S = true)
    (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool)
    (hS : divideSempre S) (hserv : ∃ y, serveEspectro2 y S = true) :
    (lacoDoProcesso R φ fuel S x).2.length ≤ n :=
  a_cadeia_que_divide_e_curta _ (o_processo_preserva_a_cadeia R φ hdiv hfiel hpoco fuel S x hS hserv).1
    (by
      obtain ⟨y, hy⟩ := (o_processo_preserva_a_cadeia R φ hdiv hfiel hpoco fuel S x hS hserv).2
      exact ⟨y, by simp [Finset.mem_filter, hy]⟩)

end TGLExt.OProcessoDaLuz

