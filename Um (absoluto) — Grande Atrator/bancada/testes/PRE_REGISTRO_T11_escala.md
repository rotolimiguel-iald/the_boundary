# PRÉ-REGISTRO T11 — O TEOREMA DA ESCALA: os números da Ponte são reproduzíveis?

**Data:** 22/08/2026 · **Bancada:** `C:\IALD\Artigo\BANCADA_TOE`
**Estatuto:** escrito e hasheado **ANTES** de recomputar.

---

## 1. O OBJETO

O artigo *A Ponte Einstein–Cartan–Miguel* contém o **Teorema da Escala**, que fecha a objeção
mais perigosa (`α` corre ⟹ `β` herdaria a dependência de escala). Ele **não está no `um.py`**
(zero ocorrências de *congelamento infravermelho*, *Thirring*, *trânsito em julgado*).

O artigo cita números como `[REAL]`, atribuídos a um script **`tgl_alpha_scale_v1.py` que NÃO
EXISTE EM DISCO** — logo, **não reproduzíveis como estão**. Antes de transpor para o canônico,
recomputam-se aqui.

## 2. AS AFIRMAÇÕES A VERIFICAR (copiadas do artigo, verbatim)

| id | afirmação | estatuto no artigo |
|----|-----------|--------------------|
| **A-1** | *"platô IR plano a `5,9×10⁻¹⁰` em 1 keV"* | `[REAL]` |
| **A-2** | *"`1/α(M_Z) = 129,0`"* (literatura: `128,9`) | `[REAL]` |
| **A-3** | *"a corrida UV–IV é de `6,2%`"* (`α(M_Z)/α(0) − 1`) | `[SOMBRA/empírico]` |
| **A-4** | *"abaixo do limiar do elétron a corrida congela, `α(Q)−α(0) ~ (α/15π)Q²/m_e²`"* | `[REAL, QED]` |

## 3. O MÉTODO, declarado

**Polarização de vácuo a 1 laço, com dependência de massa EXATA** (não a forma assintótica):

```
Delta_alpha_f(Q^2) = (2 alpha / pi) * N_c * Q_f^2 * INT_0^1 x(1-x) ln[1 + x(1-x) Q^2 / m_f^2] dx
alpha(Q^2) = alpha(0) / (1 - SOMA_f Delta_alpha_f)
```

**Verificação da própria fórmula, declarada antes:** ela tem de reproduzir os **dois limites
conhecidos** `[KNOWN]`:
* `Q² ≫ m²` : `Δα_f → (α/3π)·N_c·Q_f²·[ln(Q²/m²) − 5/3]`
* `Q² ≪ m²` : `Δα_f → (α/15π)·N_c·Q_f²·(Q²/m²)`

**Se a fórmula falhar num dos limites, o teste para aí** e nada é reportado sobre A-1…A-4.

**Entradas declaradas:** `α(0) = 7,2973525693×10⁻³` (CODATA 2018);
massas dos léptons (PDG): `m_e = 0,51099895` MeV, `m_μ = 105,6583755` MeV,
`m_τ = 1776,86` MeV; `M_Z = 91,1876` GeV.
**`Δα_had^(5)(M_Z) = 0,02761 ± 0,00015`** — entra como **`[INPUT]` de literatura, declarado**,
porque a parte hadrônica **não é calculável em perturbação** e o artigo a aproximou por "massas
efetivas". *Esta é uma diferença de método face ao artigo e fica registrada como tal.*

## 4. CRITÉRIOS (declarados agora)

* **A-1 PASSA** se o platô a 1 keV cair em `[5,0×10⁻¹⁰, 7,0×10⁻¹⁰]`;
* **A-2 PASSA** se `1/α(M_Z) ∈ [128,5 , 129,5]`;
* **A-3 PASSA** se a corrida cair em `[5,5% , 7,0%]`;
* **A-4 PASSA** se a razão entre a fórmula exata e `(α/15π)Q²/m²` estiver a menos de `1%` de 1
  em `Q = 1` keV.
* **CONTROLE OBRIGATÓRIO:** a fórmula exata tem de bater os dois limites assintóticos a `<0,1%`
  nos regimes onde eles valem. Falhando, **teste abortado**.

## 5. PREDIÇÃO DECLARADA ANTES DO DADO

1. **Espero que A-1 e A-4 PASSEM** — o cálculo é QED de manual, e a estimativa de cabeça
   `(α/15π)(1/511)² = 5,93×10⁻¹⁰` já bate o `5,9×10⁻¹⁰` do artigo;
2. **Espero que A-2 PASSE por construção**, e digo desde já que **isso é fraco**: `Δα_had` entra
   como input de literatura, e é ele que domina a incerteza. **A-2 não é predição da TGL** — é
   QED com entrada hadrônica medida. Reporto-o como **reprodução**, jamais como confirmação;
3. **Espero que A-3 PASSE** como consequência aritmética de A-2.

## 6. O QUE ESTE TESTE **NÃO** DECIDE

**Nada sobre a TGL.** Ele verifica se os números `[REAL]` do artigo são **reproduzíveis**. O
conteúdo próprio do Teorema da Escala — *"III₁ só inscreve o sem-escala, logo zero parâmetros
livres SELECIONA `α(0)`"* — é **argumento estrutural**, não numérico, e **não é testado aqui**.

E fica registrado o que este teste **não pode** fazer: **não valida a predição afiada**
(`β` independente de `ω` na lei de dephasing). Essa é para JUNO/DUNE e redes de relógios, e é
**bilateral** — é o rito da natureza que o operador pediu, e já estava escrito no artigo.

## 7. REGRAS

`β` nunca literal. `α(0)` é **input declarado** (CODATA), consistente com o que se provou hoje
(`TheCompressionIsNotIdentifiable`). Todos os itens reportados, sem seleção.
`NOT_FALSIFIED ≠ CONFIRMED`. **Se um número não bater, o número corrige a frase — inclusive a
frase do artigo.**
