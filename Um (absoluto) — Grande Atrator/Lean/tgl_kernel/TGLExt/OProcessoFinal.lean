import TGLExt.AVvem

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A morte sadia guarda a luz

## O que é

A condição por morte (`morteSadia`, em `AVvem`) diz que o lado guardado na morte (a paridade `v` no eixo `a`) guarda toda
solução da fórmula que serve o espectro. Esta pedra mostra o que essa condição garante quando vale.

## O que fica provado [REAL]

1. ★ `a_morte_sadia_guarda_a_luz`: se a morte é sadia e há uma solução da fórmula que serve o espectro, há uma solução da
   fórmula que serve o espectro acrescido da equação lida `(a, v)`.
2. `o_vvem_condicional`: dados uma solução que serve o espectro (`hcons`) e a unicidade do ponto que serve o espectro
   (`hunico`), a fórmula é satisfazível. A conclusão (SAT) já decorre de `hcons`; o teorema não diz que o ponto final
   satisfaça a fórmula nem que algum processo o devolva.

## O que NÃO se afirma

* A condição por morte em geral, para toda fórmula e todo poço, não está provada; que, havendo solução, o eco venha segue
  [OPEN].
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.OProcessoFinal

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel
open TGLExt.ARessurreicao TGLExt.AparedeDoSpan TGLExt.AFibraECoclasse TGLExt.ADualidade
open TGLExt.OSolverGF2 TGLExt.OSolverGF2_Parte2 TGLExt.OSolverGF2_Parte3 TGLExt.OSolverGF2_Parte4
open TGLExt.AVvem

variable {n : ℕ}

/-- ★ **A MORTE SADIA GUARDA A LUZ**: se a morte é sadia (o lado guardado guarda toda solução) e havia
solução servindo o espectro, há solução servindo o espectro acrescentado da equação lida. -/
theorem a_morte_sadia_guarda_a_luz {n : ℕ} (φ : CNF n) (S : Espectro2 n) (a : Fin n → Bool) (v : ZMod 2)
    (hsadia : morteSadia φ S a v)
    (hcons : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true) :
    ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y (S ++ [(a, v)]) = true := by
  obtain ⟨y, hy, hyserv⟩ := hcons
  refine ⟨y, hy, ?_⟩
  rw [serveEspectro2_append]
  have hpar := hsadia y hy hyserv
  have h1 : serveEspectro2 y [(a, v)] = true := by
    unfold serveEspectro2
    rw [List.all_cons, List.all_nil]
    show ((paridade2 y a == v) && true) = true
    rw [hpar]
    simp
  rw [h1]
  simp [hyserv]

/-- Dados uma solução que serve o espectro (`hcons`) e a unicidade do ponto que serve (`hunico`), a fórmula é
satisfazível. A conclusão já decorre de `hcons`; `hunico` não a fortalece. -/
theorem o_vvem_condicional {n : ℕ} (φ : CNF n) (S : Espectro2 n) (x : Fin n → Bool)
    (hcons : ∃ y : Fin n → Bool, custoCNF φ y = 0 ∧ serveEspectro2 y S = true)
    (hunico : ∀ y : Fin n → Bool, serveEspectro2 y S = true → y = x) :
    SAT φ := by
  obtain ⟨y, hy, hyserv⟩ := hcons
  have hyx : y = x := hunico y hyserv
  subst hyx
  exact ⟨y, (custoCNF_zero_sse φ y).mp hy⟩

end TGLExt.OProcessoFinal

