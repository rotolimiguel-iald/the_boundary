import TGLExt.OPassoFiel

set_option autoImplicit false
set_option maxHeartbeats 800000

/-!
# A ressurreição (o gate G1): a consistência da morte e a ressurreição fiel por escolha

## O que é

O gate G1 (a ressurreição fiel) pede um ponto que sirva o espectro acumulado acrescentado da equação lida na
morte. Se um ponto `w` serve o espectro `S`, ele serve também `S` acrescentado da equação `(a, paridade2 w a)`
para qualquer eixo `a`, porque essa equação é, por definição, a paridade dele nesse eixo. Logo o espectro
depois da morte de `w` é habitado, e a ressurreição por escolha clássica o serve. [ONTO] Leitura: o ponto
morto testemunha a própria morte.

## O que fica provado [REAL]

1. `serveEspectro2_append`: servir a junção `S₁ ++ S₂` é servir `S₁` e servir `S₂` (auxiliar).
2. ★ `a_morte_e_consistente`: se `w` serve `S`, então `w` serve `S ++ [(a, paridade2 w a)]`, para todo eixo `a`.
3. `ressurreicaoEscolha` [INPUT, definição]: um ponto que serve o espectro, escolhido por `Classical.choose`
   quando existe; senão, o ponto nulo. ★ `a_ressurreicao_fiel_existe`: quando o espectro é habitado, a
   ressurreição por escolha o serve.
4. ★★ `o_g1_fecha_com_o_g0`: para toda fórmula `φ`, todo espectro `S` e todo ponto `w` que sirva `S` (hipótese
   local `hw`), a ressurreição por escolha serve `S ++ [aMorte φ w]`.

## O que se segue daí [DERIVED]

* A hipótese `hw` é satisfazível: com `S = []`, qualquer `w` serve `S`, porque `serveEspectro2 w []` é `true`
  por definição. Nenhum teorema desta pedra enuncia isso.
* Com `aMorte φ w = (eixoDaMorteDe φ w, paridade2 w (eixoDaMorteDe φ w))`, o enunciado de `o_g1_fecha_com_o_g0`
  implica o da hipótese `hfiel` de `OProcessoDaLuz` para a ressurreição `fun _ S => ressurreicaoEscolha S`: é
  mais forte, porque dispensa a premissa de o eixo ser novo. Esta pedra não traz o teorema que faz essa
  instanciação; o termo de kernel que habita `hfiel` é `TGLExt.AVacuidadeDosContratos.hfiel_habitado`.

## O que NÃO se afirma

* Apesar do nome, `o_g1_fecha_com_o_g0` não usa o G0 nem a hipótese `hpoco`: toma só `hw`, que o ponto `w` sirva
  `S`. `o_poco_fiel_serve` (`OPassoFiel`) trata da descida `descerGenerico passoFiel φ S`, não da descida do laço
  `descerComEspectro φ`, e não descarrega `hpoco`.
* A hipótese `hfiel` é cumprível (`TGLExt.AVacuidadeDosContratos.hfiel_habitado`). Mas os teoremas do laço em
  `OProcessoDaLuz` que a tomam tomam também `hdiv` e `hpoco`, e essas duas não valem juntas para nenhum `n` e
  nenhuma fórmula (`TGLExt.AVacuidadeDosContratos.gates_do_processo_insatisfaziveis`). Para `n ≥ 1` a falha é de
  `hpoco` (`hpoco_falso`, sobre `CNF (m+1)`: a descida do laço não recebe o espectro); para `n = 0` é de `hdiv`
  (`hdiv_falso_n0`). Esses teoremas valem no vazio e não dizem nada sobre fórmula alguma; provar `hfiel` não muda
  isso.
* A forma aqui é a da existência, por escolha clássica (`Classical.choice`, um dos três axiomas do trio
  `propext`, `Classical.choice`, `Quot.sound`). Nenhum resolvedor GF(2) construtivo é dado nem medido.
* O transporte T, hipótese nomeada [OPEN]: um resolvedor GF(2) construtivo e polinomial (a eliminação com a
  correção) que cumpra o G1. Não está provado nem formalizado nesta pedra.
* Nada aqui trata da divisão na morte (`hdiv`, o G2) nem do eco.
* Nada aqui diz que algum processo, ou a máquina de Turing, decida SAT em tempo polinomial. Isso segue [OPEN].
* Nada sobre a natureza. O gate não se move.

Sem sorry, sem axioma novo.
-/

namespace TGLExt.ARessurreicao

open TGLExt.TraducaoTuring TGLExt.ADescidaPinada TGLExt.AparedeDoPassoIngenuo TGLExt.ADescidaComEspectro
open TGLExt.AMorteEARessurreicao TGLExt.AContagemDasMortes TGLExt.OProcessoDaLuz TGLExt.OPassoFiel

variable {n : ℕ}

/-- Servir a junção é servir as duas partes. (Auxiliar da consistência.) [REAL] -/
theorem serveEspectro2_append {n : ℕ} (x : Fin n → Bool) (S₁ S₂ : Espectro2 n) :
    serveEspectro2 x (S₁ ++ S₂) = (serveEspectro2 x S₁ && serveEspectro2 x S₂) := by
  simp [serveEspectro2, List.all_append]

/-- ★ **A MORTE É CONSISTENTE** [REAL]: se `w` serve `S`, então `w` serve `S` acrescentado da equação
`(a, paridade2 w a)`, para todo eixo `a`; o espectro depois da morte é habitado pelo próprio ponto morto.
[ONTO] Leitura: o ponto de morte é o ponto lido pelo ângulo, e ele testemunha a equação que nele se lê. -/
theorem a_morte_e_consistente {n : ℕ} (S : Espectro2 n) (w : Fin n → Bool) (a : Fin n → Bool)
    (hw : serveEspectro2 w S = true) :
    serveEspectro2 w (S ++ [(a, paridade2 w a)]) = true := by
  rw [serveEspectro2_append, hw]
  simp [serveEspectro2]

/-- [INPUT, definição] A ressurreição por escolha: um ponto que serve o espectro, escolhido por
`Classical.choose` quando o espectro é habitado; senão, o ponto nulo. (Forma da existência; nenhum resolvedor
GF(2) construtivo é dado aqui; o resolvedor construtivo e polinomial é o transporte T, hipótese nomeada [OPEN].) -/
noncomputable def ressurreicaoEscolha {n : ℕ} (S : Espectro2 n) : Fin n → Bool :=
  if h : ∃ x : Fin n → Bool, serveEspectro2 x S = true then Classical.choose h else fun _ => false

/-- ★ **A RESSURREIÇÃO FIEL EXISTE** [REAL], G1 na forma da existência: quando o espectro é habitado, a
ressurreição por escolha serve o espectro. -/
theorem a_ressurreicao_fiel_existe {n : ℕ} (S : Espectro2 n)
    (h : ∃ x : Fin n → Bool, serveEspectro2 x S = true) :
    serveEspectro2 (ressurreicaoEscolha S) S = true := by
  unfold ressurreicaoEscolha
  rw [dif_pos h]
  exact Classical.choose_spec h

/-- ★★ **O G1 FECHA** [REAL]: se `w` serve `S` (hipótese local `hw`, satisfazível), a ressurreição por escolha
serve `S ++ [aMorte φ w]`. [DERIVED] Implica o enunciado da hipótese `hfiel` de `OProcessoDaLuz` para essa
ressurreição (é mais forte: dispensa a premissa de o eixo ser novo); a instanciação não é um teorema desta pedra. Não
usa o G0 nem `hpoco`; os teoremas do laço que tomam `hfiel` tomam também `hdiv` e `hpoco`, que juntas são
insatisfazíveis (`TGLExt.AVacuidadeDosContratos.gates_do_processo_insatisfaziveis`), e por isso valem no vazio. -/
theorem o_g1_fecha_com_o_g0 {n : ℕ} (φ : CNF n) (S : Espectro2 n) (w : Fin n → Bool)
    (hw : serveEspectro2 w S = true) :
    serveEspectro2 (ressurreicaoEscolha (S ++ [aMorte φ w])) (S ++ [aMorte φ w]) = true := by
  apply a_ressurreicao_fiel_existe
  exact ⟨w, a_morte_e_consistente S w (eixoDaMorteDe φ w) hw⟩

end TGLExt.ARessurreicao

