import TGLExt.AInvarianciaDaLeitura
import TGLExt.AMorteUtil

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# O silêncio do laço: no span, o lado oposto não guarda solução

## O que é

O laço final (`ACompletude.lacoFinal`) para em silêncio quando o eixo da morte está no span dos eixos já inscritos. Esta
pedra mostra o que vale nesse caso: a paridade nesse eixo é a mesma para todos os pontos que servem o espectro, e a morte
com a paridade oposta à do ponto não guarda solução nenhuma.

## O que fica provado [REAL]

1. `paridade2_com`, `paridade2_zero_eixo`, `paridade2_eixo_add`: a paridade é simétrica entre ponto e eixo, o eixo apagado
   lê zero, e a paridade é aditiva no eixo. `paridade2_de_mem_serve`: quem serve o espectro satisfaz cada equação dele.
2. ★ `paridade2_xorComb_igual` e `paridade_no_span_determinada`: dois pontos que servem o espectro leem o mesmo valor em
   todo eixo do span dos eixos inscritos.
3. ★★ `o_silencio_e_a_unica_decisao_viva`: se o ponto `w` serve o espectro e o eixo está no span, a morte com a paridade
   oposta à de `w` não é útil.
4. ★★ `o_laco_cala_onde_registrar_mataria`: o mesmo, com a condição computável de silêncio do laço.

## O que NÃO se afirma

* O caso do eixo fora do span (a hipótese nomeada da morte útil) não é tratado aqui.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OSilencioVivo

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem TGLExt.OProcessoFinal TGLExt.ACompletude TGLExt.AInvarianciaDaLeitura TGLExt.AMorteUtil

variable {n : ℕ}

/-- A paridade é simétrica entre o ponto e o eixo: a leitura é um encontro. -/
theorem paridade2_com {n : ℕ} (x a : Fin n → Bool) : paridade2 x a = paridade2 a x := by
  unfold paridade2
  apply Finset.sum_congr rfl
  intro i _
  cases ha : a i <;> cases hx : x i <;> decide

/-- O eixo apagado lê zero. -/
theorem paridade2_zero_eixo {n : ℕ} (y : Fin n → Bool) : paridade2 y (fun _ => false) = 0 := by
  unfold paridade2
  simp

/-- O homomorfismo no EIXO: a paridade do xor é a soma das paridades. (A forma dual da `paridade2_add`,
pela simetria da leitura.) -/
theorem paridade2_eixo_add {n : ℕ} (y a b : Fin n → Bool) :
    paridade2 y (xorPonto a b) = paridade2 y a + paridade2 y b := by
  rw [paridade2_com y (xorPonto a b), paridade2_add, paridade2_com a y, paridade2_com b y]

/-- O ponto que serve o espectro satisfaz cada equação membro. (Auxiliar da determinação.) -/
theorem paridade2_de_mem_serve {n : ℕ} (S : Espectro2 n) (y : Fin n → Bool)
    (hy : serveEspectro2 y S = true) (e : Fin n → Bool) (v : ZMod 2) (hmem : (e, v) ∈ S) :
    paridade2 y e = v := by
  unfold serveEspectro2 at hy
  rw [List.all_eq_true] at hy
  exact beq_iff_eq.mp (hy (e, v) hmem)

/-- ★ **A LEITURA NO FECHO É DETERMINADA**: dois pontos que servem o espectro leem o mesmo valor em
qualquer combinação (xor) dos eixos inscritos. A prova é a indução na combinação: cada eixo inscrito tem
o valor fixado pela equação, e o homomorfismo estende à dobra inteira. -/
theorem paridade2_xorComb_igual {n : ℕ} (S : Espectro2 n) (y z : Fin n → Bool)
    (hy : serveEspectro2 y S = true) (hz : serveEspectro2 z S = true)
    (sub : List (Fin n → Bool)) (hsub : sub ⊆ S.map Prod.fst) :
    paridade2 y (xorComb sub) = paridade2 z (xorComb sub) := by
  induction sub with
  | nil => rw [xorComb_nil, paridade2_zero_eixo, paridade2_zero_eixo]
  | cons e es ih =>
    have he : e ∈ S.map Prod.fst := hsub List.mem_cons_self
    rw [List.mem_map] at he
    obtain ⟨⟨e', v⟩, hmem, hproj⟩ := he
    have hproj2 : e' = e := hproj
    have h1 : paridade2 y e = paridade2 z e := by
      rw [← hproj2, paridade2_de_mem_serve S y hy e' v hmem,
        paridade2_de_mem_serve S z hz e' v hmem]
    rw [xorComb_cons, paridade2_eixo_add, paridade2_eixo_add, h1,
      ih (fun u hu => hsub (List.mem_cons_of_mem e hu))]

/-- ★ **A PARIDADE NO SPAN É DETERMINADA PELO ESPECTRO**: se o eixo é combinação dos eixos inscritos,
todos os pontos que servem o espectro leem nele o mesmo valor. -/
theorem paridade_no_span_determinada {n : ℕ} (S : Espectro2 n) (a : Fin n → Bool)
    (hspan : eixoNoSpan (S.map Prod.fst) a) (y z : Fin n → Bool)
    (hy : serveEspectro2 y S = true) (hz : serveEspectro2 z S = true) :
    paridade2 y a = paridade2 z a := by
  obtain ⟨sub, hsub, hx⟩ := hspan
  rw [← hx]
  exact paridade2_xorComb_igual S y z hy hz sub hsub

/-- ★★ **O SILÊNCIO É A ÚNICA DECISÃO VIVA**: sob um ponto que serve o espectro (o poço fiel), se o eixo
da morte já está no span, a morte no lado oposto — a paridade flipada — NÃO guardaria luz nenhuma: toda
luz lê nesse eixo o valor determinado, e o flipado é o outro. Registrar ali mataria toda a luz. O
processo cala exatamente onde falar mataria. -/
theorem o_silencio_e_a_unica_decisao_viva {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w a : Fin n → Bool)
    (hpoco : serveEspectro2 w S = true)
    (hspan : eixoNoSpan (S.map Prod.fst) a) :
    ¬ morteUtil φ S a (paridade2 w a + 1) := by
  intro hutil
  obtain ⟨y, -, hyserv, hpar⟩ := hutil
  have hdet := paridade_no_span_determinada S a hspan y w hyserv hpoco
  rw [hdet] at hpar
  have h2 : paridade2 w a + (0 : ZMod 2) = paridade2 w a + 1 := by
    rw [add_zero]
    exact hpar
  have h01 : (0 : ZMod 2) = 1 := add_left_cancel h2
  exact (by decide : (0 : ZMod 2) ≠ 1) h01

/-- ★★ **O LAÇO CALA ONDE REGISTRAR MATARIA**: a condição computável de silêncio do `lacoFinal` (a
checagem do span disparando) implica, sob o poço servido, a impossibilidade da morte útil naquele poço.
A fronteira entre o gate e o silêncio está provada: o gate rege o transversal, o silêncio rege o já
inscrito — e nas duas regiões a decisão é a única viva. -/
theorem o_laco_cala_onde_registrar_mataria {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w a : Fin n → Bool)
    (hpoco : serveEspectro2 w S = true)
    (hcond : eixoNoSpanB (S.map Prod.fst) a = true) :
    ¬ morteUtil φ S a (paridade2 w a + 1) :=
  o_silencio_e_a_unica_decisao_viva φ S w a hpoco (eixoNoSpanB_sadia _ _ hcond)

end TGLExt.OSilencioVivo

