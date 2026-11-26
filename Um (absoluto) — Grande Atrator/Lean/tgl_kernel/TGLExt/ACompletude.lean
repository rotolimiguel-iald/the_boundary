import TGLExt.OProcessoFinal

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O laço final: o eco fiel e a luz guardada sob a hipótese nomeada

## O que é

O laço final (`lacoFinal`) desce fiel a partir do ponto dado; se o poço satisfaz a fórmula, responde com ele; se o eixo da
morte no poço está no span dos eixos já lidos (`eixoNoSpanB`, a checagem computável), para em silêncio; senão, registra no
espectro a paridade OPOSTA à do poço nesse eixo e recomeça do ponto que o resolvedor construtivo devolve. O combustível
limita o número de voltas.

## O que fica provado [REAL]

1. `eixoNoSpanB_sadia`: se a checagem computável dispara, o eixo está no span dos eixos.
2. ★ `o_eco_do_laco_final_e_fiel`: quando o laço responde com um ponto, esse ponto tem custo zero (é testemunha).
3. ★ `o_laco_preserva_a_luz`: sob a hipótese nomeada `hgates` (em todo poço que não satisfaz e cujo eixo de morte está fora
   do span, a morte com a paridade oposta é sadia), se o espectro inicial é servido por toda solução, todo espectro que o
   laço alcança também é.
4. `a_completude_condicional`: se uma solução serve o espectro e todo ponto que serve o espectro é `x`, então `x` é
   solução. Este teorema não envolve o laço nem a hipótese `hgates`.

## O que NÃO se afirma

* Que o laço responda com eco quando há solução: ele pode parar em silêncio (eixo no span) ou esgotar o combustível; a
  completude do laço segue [OPEN].
* A hipótese `hgates` (a condição por morte em geral) não está provada.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ACompletude

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal

variable {n : ℕ}

/-- [DEF] A checagem do span, computável: o eixo é o xor de uma sub-lista dos eixos. -/
def eixoNoSpanB {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool) : Bool :=
  axes.sublists.any (fun sub => decide (xorComb sub = a))

/-- ★ A checagem computável é sadia: se ela dispara, o eixo está mesmo no span. (Auxiliar do processo; a
volta — a completude da checagem — é a invariância do xor por reordenação, nomeada.) -/
theorem eixoNoSpanB_sadia {n : ℕ} (axes : List (Fin n → Bool)) (a : Fin n → Bool) :
    eixoNoSpanB axes a = true → eixoNoSpan axes a := by
  unfold eixoNoSpanB eixoNoSpan
  rw [List.any_eq_true]
  intro h
  obtain ⟨sub, hsub, hdec⟩ := h
  exact ⟨sub, (List.mem_sublists.mp hsub).subset, by simpa using hdec⟩

/-- [DEF] O laço final: desce fiel; se o poço satisfaz, o eco; senão, se o eixo da morte não é transversal,
o silêncio; senão, a morte no lado OPOSTO (a paridade flipada) e a ressurreição construída. Devolve o eco
(ou o silêncio) e o espectro final, para a contagem. -/
def lacoFinal {n : ℕ} (φ : CNF n) :
    ℕ → Espectro2 n → (Fin n → Bool) → (Option (Fin n → Bool)) × Espectro2 n
  | 0, S, x => (if custoCNF φ x = 0 then some x else none, S)
  | fuel + 1, S, x =>
    let w := (descerGenerico passoFiel φ S φ.length ([], x)).2
    if custoCNF φ w = 0 then (some w, S)
    else
      let a := eixoDaMorteDe φ w
      let v := paridade2 w a + 1
      if eixoNoSpanB (S.map Prod.fst) a then (none, S)
      else lacoFinal φ fuel (S ++ [(a, v)]) (ressurreicaoConstrutiva (S ++ [(a, v)]))

/-- ★ O eco do laço final é fiel por construção: responde só com ponto de altura zero. -/
theorem o_eco_do_laco_final_e_fiel {n : ℕ} (φ : CNF n) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x w : Fin n → Bool) (S' : Espectro2 n),
      lacoFinal φ fuel S x = (some w, S') → custoCNF φ w = 0 := by
  intro fuel
  induction fuel with
  | zero =>
    intro S x w S' h
    unfold lacoFinal at h
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
    unfold lacoFinal at h
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
      · rename_i hcn hspan
        exact ih _ _ _ _ h

/-- ★ **O LAÇO PRESERVA A LUZ**: sob a morte sadia e a transversalidade em cada poço, todo espectro
alcançado pelo laço guarda toda solução; e a solução servida inicial segue servida. -/
theorem o_laco_preserva_a_luz {n : ℕ} (φ : CNF n)
    (hgates : ∀ (S : Espectro2 n) (w : Fin n → Bool),
      custoCNF φ w ≠ 0 → ¬ (eixoNoSpanB (S.map Prod.fst) (eixoDaMorteDe φ w) = true) →
      morteSadia φ S (eixoDaMorteDe φ w) (paridade2 w (eixoDaMorteDe φ w) + 1)) :
    ∀ (fuel : ℕ) (S : Espectro2 n) (x : Fin n → Bool),
      (∀ y : Fin n → Bool, custoCNF φ y = 0 → serveEspectro2 y S = true) →
      ∀ y : Fin n → Bool, custoCNF φ y = 0 →
        serveEspectro2 y (lacoFinal φ fuel S x).2 = true := by
  intro fuel
  induction fuel with
  | zero => intro S x hguarda y hy; exact hguarda y hy
  | succ f ih =>
    intro S x hguarda y hy
    unfold lacoFinal
    dsimp only []
    split
    · rename_i hcond
      exact hguarda y hy
    · split
      · exact hguarda y hy
      · rename_i hcn hspan
        -- a morte: a equação sadia guarda a luz
        have hnova : ∀ y : Fin n → Bool, custoCNF φ y = 0 →
            serveEspectro2 y (S ++ [(eixoDaMorteDe φ (descerGenerico passoFiel φ S φ.length ([], x)).2,
                paridade2 (descerGenerico passoFiel φ S φ.length ([], x)).2
                  (eixoDaMorteDe φ (descerGenerico passoFiel φ S φ.length ([], x)).2) + 1)]) = true := by
          intro y hy
          have hpar := hgates S _ hcn hspan y hy (hguarda y hy)
          rw [serveEspectro2_append]
          have h1 : serveEspectro2 y [(eixoDaMorteDe φ (descerGenerico passoFiel φ S φ.length ([], x)).2,
                paridade2 (descerGenerico passoFiel φ S φ.length ([], x)).2
                  (eixoDaMorteDe φ (descerGenerico passoFiel φ S φ.length ([], x)).2) + 1)] = true := by
            unfold serveEspectro2
            rw [List.all_cons, List.all_nil]
            show ((paridade2 y _ == _) && true) = true
            rw [hpar]
            simp
          rw [h1]
          simp [hguarda y hy]
        exact ih _ _ hnova y hy

/-- Se uma solução serve o espectro (`hcons`) e todo ponto que serve o espectro é `x` (`hunico`), então `x` é
solução. Não envolve o laço nem a hipótese da morte sadia. -/
theorem a_completude_condicional {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (hcons : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true)
    (hunico : ∀ y : Fin n → Bool, serveEspectro2 y S = true → y = x) :
    custoCNF φ x = 0 := by
  obtain ⟨y, hy, hyserv⟩ := hcons
  have hyx : y = x := hunico y hyserv
  subst hyx
  exact hy

end TGLExt.ACompletude

