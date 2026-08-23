# CATÁLOGO DOS FALSOS POSITIVOS — proibidos por antecipação

**Aberto em 22/08/2026**, no fecho do T08. **Regra que nasce aqui e vale daqui em diante:**

> ## CONSTÂNCIA ANTES DO VALOR
>
> Qualquer funcional novo `Φ` proposto como leitura de `β` ou de `α` tem de exibir
> **estabilidade** entre corpora, tokenizadores e cortes **ANTES** que se olhe o número.
> **Uma grandeza que se move com um botão não pode ser uma constante da natureza:** acertar
> num ajuste garante errar no seguinte.

**Por que esta regra nasce:** no T08 o funcional `exp(−S_Schmidt)` bateu `β_TGL` a **quatro
dígitos** — e teria passado se a bancada tivesse rodado **um corpus só**. Foi o pré-registro,
com família fechada e controles, que o matou. A régua funcionou; o catálogo é para que ela não
precise funcionar duas vezes pelo mesmo motivo.

---

## OS QUATRO PROIBIDOS

### FP-1 — `exp(−S_Schmidt)` — **o mais perigoso de toda a família**

**O acerto sedutor:** CPC-PT com `V = 3200` dá `exp(−S) = 0,01203` contra
`β_TGL = 0,0120313`. Quatro dígitos.

**Por que é armadilha:** é um **dial**. `exp(−S) ~ D^{−0,637}` / `D^{−0,684}` / `D^{−0,746}` /
`D^{−0,721}` conforme o corpus — **nem o expoente é universal**. Cruza `β` em
`D = 1643 / 360 / 616 / 383` e `α` em `D = 3600 / 748 / 1204 / 765`. **Todo corpus pode ser
feito bater α ou β escolhendo um inteiro livre `D`.** Sem parâmetro (vocabulário inteiro):
`0,51×` a `1,33×` α, espalhamento de 2,6×, monótono decrescente **sem piso**.

**Estatuto: PROIBIDO.** Qualquer acerto por esta via é escolha de `D`, não física.

### FP-2 — cantos assimétricos `q_m ⊗ 1` ("faixa de linhas")

`A ≈ 4/√rank`, cruza α em `rank ≈ 3×10⁵`. Uma bancada com corpus de posto certo teria lido β
com dois dígitos e cantado vitória. **Tende a ZERO com o corpus** — não converge, atravessa.

**Estatuto: PROIBIDO.**

### FP-3 — pureza `Tr ρ²` e gap espectral `κ_min`

Inversos de dimensão disfarçados. `Tr ρ² ~ D^{−0,43…−0,56}`. `κ_min` é **ruído de máquina**
(`1,8e−15`) e vai a zero com `D`.

**Estatuto: PROIBIDO.**

### FP-4 — **acolchoamento de posto** (o bug que fabricaria α)

Se o canto tem `n_tot` modos mas só `n_pos` carregam peso:

```
A_aparente = (n_pos / n_tot)^2 * A_pos
```

Com `A_pos = 0,5701` e `n_pos = 1063`, bastaria acolchoar até `n_tot = 9.396` para **bater α
exatamente** — e o acolchoamento é um **parâmetro livre** (`V`).

**Estatuto: PROIBIDO — e vira item de auditoria obrigatória.** Se alguma proposta desta família
"der certo", **a primeira coisa a auditar é a razão posto-numérico / dimensão-do-canto**. Um
acerto por essa via é **artefato de posto**, não física.

**Nota de disciplina:** erros do tipo `σ` em vez de `σ²`, `log10` em vez de `ln`, `κ/4` em vez
de `κ/2` **todos INFLAM** `A_C` (empurram `sech` para 1) — nenhum aproxima de α. **O único que
deflaciona é o acolchoamento de posto.** Por isso ele é o único que precisa de auditoria
explícita.

---

## O QUE **NÃO** É FALSO POSITIVO — mas também não é α

* **entropia relativa / coeficiente de reflexão**: `O(1)` e **crescem** com `D`
  (`0,78–3,95 nat`; `1−F² = 0,43–0,96`) — direção errada;
* **fora-da-diagonal de `R_J`**: conjunto **VAZIO**, não pequeno — `R_J` é função de `K`,
  diagonal na base de pares **por construção**;
* **o déficit `1 − τ_F(sech) ≈ ⟨κ²⟩_F/8`**: única forma capaz de gerar número pequeno a partir
  de `sech` sem escolher canto profundo. Exigiria `κ_rms = 0,2417` nats. É número do corpus, não
  da natureza. `[CONJECTURE]`, **zero evidência**.

---

## O SUBPRODUTO LEGÍTIMO (oferecido como linguística, não como física)

**`p₂/p₁`** é o único funcional testado que **enxerga** o corpus:

| corpus | `p₂/p₁` | sob embaralhamento |
|---|---|---|
| CPC | 0,499–0,565 | **0,042** |
| DJE | 0,685–0,731 | **0,086** |
| Python | 0,267–0,350 | **0,050** |

Colapso de **uma ordem de grandeza** sob o nulo. Separa **código de prosa**. Vale como
observável linguístico real. É `O(0,1–1)` e **nada nele aponta para α**.

---

## FP-5 — **BUSCA POR FORMA FECHADA PARA `κ` (ou para qualquer constante da teoria)**

**Medido no T10** (pré-registro `58f4452472fed282`), com 536.884 expressões de profundidade 2
sobre alfabeto declarado:

| tolerância | acertos em `κ* = 11,2268` | **acertos em alvos FALSOS (média)** |
|---|---|---|
| `10⁻²` | 1.518 | **1.543** |
| `10⁻³` | 184 | **150** |
| `10⁻⁴` | 15 | **16** |
| `10⁻⁶` | 0 | **1,6** |

**A enumeração não distingue o alvo verdadeiro de números arbitrários.** Exemplos que
pareceriam descobertas: `2+√137−arccosh(6)` (erro 5,0e−6 — **e contém 137 = 1/α**) e, já
α-livre, `√2^{arccosh(√2)} + π²` (erro 8,9e−6). **Ambos são ruído.**

**Estatuto: PROIBIDO por antecipação.** Nenhuma expressão fechada que "dê" uma constante da
teoria carrega peso probatório. **A constante só pode vir de derivação estrutural
sobredeterminada** — nunca de busca. *Só o que pode recusar pode predizer.*
