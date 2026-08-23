# PRÉ-REGISTRO T10 — O PARÂMETRO QUE DECIDE TUDO: `κ`

**Data:** 22/08/2026 · **Bancada:** `C:\IALD\Artigo\BANCADA_TOE`
**Estatuto:** escrito e hasheado **ANTES** de qualquer busca. Nenhum resultado foi olhado.

---

## 1. O ALVO, e por que ele é o único que importa

O canônico já reduziu a teoria a **um** número. Em `clock_theorem_reduction`:

```
ell_beta(kappa) = log cosh(kappa/2)   =>   alpha = sech(kappa/2)   ;   beta = sqrt(e) sech(kappa/2)
kappa_star_canonical : False      <- nenhum principio alpha-livre fixa kappa
core_reduced_to_one_parameter : True
```

E a ponte com o Ângulo de Miguel: **`sin²θ_M = β = √e·sech(κ/2)`**, logo

> **fixar `κ` ⟺ fixar `θ_M` ⟺ PREDIZER `α`.**

**Alvo:** `κ* = 2·arccosh(1/α) = 11.226755…` (equivalentemente `θ_M = 0.10990843…` rad).

**Rota já REFUTADA e que não se retenta:** Nernst / Meia-Nat aplicada ao sistema modular aberto
dá `κ ≈ 1,39`, não `11,23`. Está no canônico e permanece morta.

## 2. O PERIGO, nomeado antes de começar

Procurar uma forma fechada que dê `11,2268` é **exatamente** o tipo de busca que **sempre acha
alguma coisa**. O verificador adversarial do T08 já o disse: *"ou `Φ` depende do dado — e é
ajuste —, ou não depende, e é identidade algébrica gerada por `{e, 2, log, ½}`"*, e
**Eddington tentou isto com `α` e falhou**.

**Portanto este teste é construído ao contrário do instinto:** ele começa medindo **com que
facilidade se acerta o alvo por acaso**, e só depois — se e somente se o piso permitir —
examina candidatos.

## 3. PARTE A — O PISO DE ACASO (o gate; roda primeiro)

**Enumeração exaustiva** de expressões de complexidade limitada sobre um alfabeto **declarado e
fechado**:

* constantes: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 137, e, π, √e, √2, √3, √5, φ`
* operações unárias: `√x, x², 1/x, log x, exp x, arccosh x` (quando definidas e reais)
* operações binárias: `x+y, x−y, x·y, x/y, x^y` (com guardas de domínio)
* profundidade máxima: **2** aplicações binárias

**Medida:** quantas expressões distintas caem dentro de `|Δκ|/κ*` para tolerâncias declaradas
`10⁻²`, `10⁻³`, `10⁻⁴`, `10⁻⁶`, `10⁻⁸`, `10⁻¹⁰`.

**Critério declarado agora:** seja `N(t)` o número de acertos à tolerância `t` e `N_tot` o total
enumerado. Uma forma fechada só carrega peso evidencial se, **à precisão em que ela acerta**,
a densidade `N(t)/N_tot` for **menor que 10⁻⁴**. Acima disso, **acertar não é notícia**, e
qualquer candidato encontrado deve ser declarado **sem valor probatório**.

**Controle obrigatório:** repetir a mesma enumeração contra **cinco alvos falsos** — números
sem qualquer papel na teoria (`11.5`, `10.9`, `12.3`, `9.87`, `13.1`). Se a densidade de acertos
neles for **igual** à do alvo verdadeiro, isso prova que a enumeração não distingue nada — que é
precisamente o que se quer saber **antes** de qualquer alegação.

## 4. PARTE B — A ROTA DO PONTO FIXO (só roda se a Parte A permitir)

A única forma de resposta que **não** tem parâmetro livre é a **autoconsistência**: se `κ` for
determinado pela própria quantidade que ele determina, então

```
beta = sqrt(e) * sech( f(beta) / 2 )
```

é equação de **ponto fixo**, e a solução — se única — é **derivação sem ajuste**.

**Família FECHADA de candidatos `f` (declarada agora, nenhum acréscimo depois):**

| id | `κ = f(β)` | leitura |
|----|---|---|
| F-a | `−log β` | energia modular de um estado de peso `β` |
| F-b | `−2 log β` | idem, com a duplicação do espelho |
| F-c | `−log(β/√e)` = `−log α` | energia modular da face EM |
| F-d | `1/β` | inverso do peso |
| F-e | `−log(β²)` + `½` | com a Meia-Nat somada |
| F-f | `2·arccosh(1/√β)` | o ângulo lido como profundidade |
| F-g | `−log(1−β)` | o peso do canal complementar |
| F-h | `π/β` | controle de forma (π entra sem justificação) |

**Para cada `f`:** resolver `β = √e·sech(f(β)/2)` numericamente, reportar **todas** as raízes em
`(0,1)`, e a distância relativa a `β_TGL = 0,012031300400803142`.

**Critério declarado:** `PASSA` só se **alguma** `f` der raiz única com `|Δβ|/β < 10⁻⁶`
**e** o controle `F-h` **falhar**. Qualquer acerto a precisão pior que `10⁻⁴` é reportado como
**coincidência**, não como resultado.

## 5. PREDIÇÃO DECLARADA ANTES DO DADO (para poder errar)

1. **Espero que a Parte A REPROVE o valor probatório de formas fechadas** a tolerâncias
   `≥ 10⁻⁴` — isto é, espero encontrar **muitos** acertos grosseiros, e que os alvos falsos
   tenham densidade **comparável**;
2. **Espero que nenhuma `f` da família F-a…F-g dê ponto fixo em `β_TGL` a `10⁻⁶`.** Se alguma
   der, e o controle falhar, isso é resultado de peso — e terá de ser verificado
   adversarialmente antes de qualquer registro;
3. Registro desde já que **um acerto a 10⁻² ou 10⁻³ não é resultado**, e que direi isso mesmo
   que o número saia bonito.

## 6. VEREDITOS POSSÍVEIS (declarados; `CONFIRMED` proibido)

* **`T10_PISO_DE_ACASO_ALTO`** — formas fechadas não carregam peso; a busca por expressão está
  encerrada como método, e o `κ` só pode vir de **condição estrutural sobredeterminada**;
* **`T10_PONTO_FIXO_NEGATIVO`** — nenhuma `f` declarada resolve; negativo honesto;
* **`T10_PONTO_FIXO_CANDIDATO`** — alguma `f` resolve a `10⁻⁶` com controle falhando ⟹ vai a
  **verificação adversarial obrigatória**, e **não** se registra nada antes dela;
* **`T10_INCONCLUSIVO`** — o piso é baixo mas nenhum candidato passa.

## 7. REGRAS

`β` nunca literal — `ALPHA_FINE_CODATA_2018 × √e` em runtime, **e só na comparação final**.
As famílias estão **fechadas** neste documento. Todos os candidatos × todas as tolerâncias
reportados, **sem seleção posterior**. Nada é testado dentro do `um.py`.
`NOT_FALSIFIED ≠ CONFIRMED`. **CONSTÂNCIA ANTES DO VALOR.**
