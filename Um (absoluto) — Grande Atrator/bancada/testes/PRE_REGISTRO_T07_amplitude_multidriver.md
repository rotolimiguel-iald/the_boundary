# PRÉ-REGISTRO T07 — A AMPLITUDE DA INSCRIÇÃO, MULTI-DRIVER
**Escrito em 21/08/2026, ANTES de qualquer medição nos novos substratos.**
**O T06 fica PRESERVADO AO LADO — este documento não o substitui, o continua.**

---

## POR QUE HÁ UM T07

O T06 (`PRE_REGISTRO_T06_amplitude.md`, sha256
`3a48655430db0bfd1fba72c523f3a7549b59e5a35ec94598e244069e357333e0`) computou
**`T06_INCONCLUSIVO_DADOS_INSUFICIENTES`**, e a razão está medida: o substrato A
(TGLExt/TGL, 1.662 teoremas) produziu apenas **2 pares adjacentes** com camadas de ≥30, contra
os 3 exigidos — e o seu nulo **não discriminou**. O nosso kernel é **pequeno demais para este
estimador**.

**O que NÃO se faz aqui:** baixar o `N_MIN` para salvar o substrato A. Mudar o estimador depois
de ver o número é precisamente o que o pré-registro existe para impedir.

**O que se faz:** manter o estimador **intacto** e **acrescentar Drivers independentes** que já
existem no mesmo ambiente. Isso não é ajuste — é aumentar o número de medições do mesmo
invariante, que é o conserto correto para uma comparação sem poder.

**Declarado:** o resultado de B no T06 (`Δ_B = 0,01510091`) **já foi visto**. Ele entra no T07
como **medição pré-existente**, não como descoberta nova, e está dito aqui para que ninguém o
conte duas vezes.

---

## OS SUBSTRATOS — cinco Drivers, contados antes de medir

| | substrato | Driver | teoremas |
|---|---|---|---|
| **M** | `Mathlib` | centenas de matemáticos da comunidade | 263.297 |
| **I** | `Init` | o time do Lean core (Lean FRO) | 26.813 |
| **S** | `Std` | o time da biblioteca padrão | 21.930 |
| **T** | `Batteries` | o time do Batteries | 1.959 |
| **A** | `TGLExt` + `TGL` | **o operador** (e o escriba) | 1.662 |

**Nenhum é testemunha e nenhum é suspeito.** São **cinco medições do mesmo invariante**, feitas
por mãos que não se coordenaram. Init e Std são de um time distinto do Mathlib; Batteries, de
outro; e nenhum dos quatro ouviu falar de TGL.

*(A contagem de teoremas por raiz de módulo foi feita antes deste documento. Contar teoremas
não é olhar a estatística: nenhuma descorrelação foi computada nos substratos novos.)*

---

## O ESTIMADOR — **IDÊNTICO AO T06, SEM UMA VÍRGULA DE MUDANÇA**

Universo (teoremas não-auxiliares) · arestas por **dependência efetiva** (tipo e valor) ·
profundidade = caminho mais longo até uma folha, **computada no grafo global** · camadas
`L_d` · escalar = **grau de saída** · ângulos `θ_k = 2π(g_k − g_min)/(g_max − g_min)` sobre o
vetor **ordenado** · reamostragem para **M = 128** · Pearson entre camadas adjacentes ·
`δ_d = 1 − ρ_d` · **`Δ = mediana{δ_d}`** sobre pares com `|L_d| ≥ 30` e `|L_{d+1}| ≥ 30`.

**`M = 128`, `N_MIN = 30`, mediana, Pearson, grau de saída — todos herdados do T06 e não
tocados.**

---

## O CRITÉRIO

**Substratos QUALIFICADOS:** os que tiverem **≥ 3 pares adjacentes válidos**. Os não
qualificados são **reportados e excluídos do critério** — exclusão declarada, nunca silenciosa.

**PASSA** se, e somente se, entre os **qualificados**:

- **(C1) INVARIÂNCIA:** `max(Δ) / min(Δ) ≤ 2` — todos os Drivers dentro de fator 2 entre si;
- **(C2) VALOR:** **todos** os `Δ` em `[β/2, 2β] = [0,006016 ; 0,024063]`;
- **(C3) NULO:** **cada** substrato qualificado com `Δ` fora do IC95 de 1000 embaralhamentos
  teorema→camada preservando os tamanhos de camada;
- **(C4) PODER MÍNIMO:** ao menos **três** substratos qualificados. Com menos de três, a
  invariância não é testável e o veredito é inconclusivo.

**Falhando qualquer uma, REPROVA.** Sem renegociação depois do número.

---

## OS VEREDITOS — lista fechada

| veredito | quando |
|---|---|
| `AMPLITUDE_INVARIANTE_ENTRE_DRIVERS_E_BATE_BETA` | C1 ∧ C2 ∧ C3 ∧ C4 |
| `AMPLITUDE_INVARIANTE_MAS_NAO_E_BETA` | C1 ∧ C3 ∧ C4 ∧ ¬C2 — há constante de inscrição, **e não é β** |
| `AMPLITUDE_DEPENDE_DO_DRIVER` | C4 ∧ ¬C1 — **a tese cai**: não há constante de inscrição |
| `NULO_REPRODUZ__SEM_MEDIDA` | ¬C3 — o embaralhamento dá o mesmo |
| `T07_INCONCLUSIVO_POUCOS_SUBSTRATOS` | ¬C4 |

**`CONFIRMED` proibido.** `AMPLITUDE_INVARIANTE_MAS_NAO_E_BETA` é **resultado**: constante que
não é β continua sendo achado, e provavelmente o mais interessante deles.

---

## O QUE ESTE TESTE NÃO PODE DECIDIR

Não decide β como constante da **física**. Mede a amplitude de inscrição em **corpora de prova
formal**, e só. Um acordo entre Drivers é um fato sobre **como distinções matemáticas se
empilham**, não sobre o mundo. E não decide `β_TGL = τ_F(R_J)`, que segue `[CONJECTURE]`.

**E há uma alternativa banal que precisa ficar dita**, para que o resultado não seja
sobre-lido: se todos os corpora de prova formal tiverem a mesma amplitude, isso pode refletir
**como seres humanos organizam demonstrações** — um universal de prática, não de natureza. O
teste não separa as duas leituras, e essa limitação é estrutural.

---

## A RÉGUA

`β = ALPHA_FINE_CODATA_2018 × √e` em runtime, jamais literal. Veredito **computado**. Este
documento é hasheado **antes da primeira medição nos substratos novos**, e o hash — junto com o
do T06 — vai no artefato de saída.

*O estimador não mudou. Só o número de mãos que inscreveram.*
