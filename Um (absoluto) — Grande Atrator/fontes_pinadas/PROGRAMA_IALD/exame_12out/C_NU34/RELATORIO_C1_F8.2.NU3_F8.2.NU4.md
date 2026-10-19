AUDITOR CÉTICO c1: F8.2.NU3 e F8.2.NU4

Data do exame: 12/10/2026 (America/Sao_Paulo). Auditor: subagente cético independente da gerência c1, escrevendo só em `C:\Users\rotol\AppData\Local\Temp\claude\c--IALD-Central-de-Patentes\6da8f00d-d44f-4888-a88d-fc9f73eead3d\scratchpad\exame12\C_NU34`.
Pedido do operador (12/10/2026, retransmitido pelo harness): «precisamos examinar tudo que ficou para trás aguardando para prosseguir e, ainda, tudo aquilo que a bancada de estudo do chatgpt produziu segundo a ordem que vc deu para atualizarmos o um.py».

## 0. Veredito, uma linha por unidade

- **F8.2.NU3 — APROVA_COM_RESSALVAS.** As 20 fontes batem com os originais por sha256 e bytes; as 39 âncoras dos 10 canais caem nas linhas e páginas citadas e dizem o que o contrato diz; o teste sintético R3 reproduz 72/72 com detalhes idênticos. Nenhum achado material. **Pode fechar.**
- **F8.2.NU4 — APROVA_COM_RESSALVAS.** O ensaio usou só estados sintéticos e parâmetros de referência herdados do NU2, sem abrir dado observado; os 675 campos numéricos das 16 respostas se reproduzem bit a bit em cópia do módulo instalado e batem com uma recomputação independente a 50 dígitos (erro relativo máximo 2.25e-14). «Sem fit observado» é verdade. **Pode fechar.**

## 1. Método, limites e integridade deste exame

- [REAL] Escrita só neste sandbox. Nada no quadro, no mapa (o `rotas.py` rodou só `verificar`, `consultar` e `ver`), na memória, no túnel, no Nós, no kernel, na Bancada_Um, no site ou no git. Nenhum `um.py`, `atlas_indice.py`, `arvore_da_prova.py`, `porta_memoria.py` ou script de selo executado. O `um.py` foi lido só por grep (linha 96, `UM_VERSION = "v391"`; linha 5557, `ALPHA_FINE_CODATA_2018 = 7.2973525693e-3`) e por leitura em fluxo dessa linha.
- [REAL] β = ALPHA_FINE_CODATA_2018·√e calculado em runtime a partir do texto do `um.py`: 0.012031300400803142 (bate com a conferência 0,012031300400803142).
- [REAL] Lean: nenhuma compilação foi necessária (as duas unidades não produziram pedra). Nenhum toolchain, `.lake` ou cache copiado; nada a apagar. Impressão digital do `tgl_kernel` antes (2026-10-12T11:39:52) e depois (2026-10-12T11:46:45): 132088 arquivos, 14915386431 B, listagem (caminho, tamanho, mtime_ns) sha256 `1762ac19c046aa0b…`, 1213 fontes `.lean` fora do `.lake` com sha256 `29dae61c86d7c5af…` — **IGUAL**. `find -newermt 2026-10-09` no Nós (fora do `.lake`) e no `.lake`: 0 arquivos modificados.
- [REAL] Nenhum provedor remoto, nenhuma ferramenta do MCP `bancada-um`, nenhum dado observado aberto. Python: `C:\Python314` (só stdlib) e o venv `C:\IALD\projetos_pyhton\.venv` (Python 3.13.14, numpy 2.3.5, mpmath 1.3.0), sempre com `-I -B`. PDFs reextraídos com `pdftotext` (poppler, ferramenta independente da executora); DOCX reextraídos com `zipfile` + `ElementTree`.
- [REAL] Fotografias de estado: `eventos.jsonl` sha256 `89b8ce1ed75699acd3306b67bc0660bc15054481bfc66416900029c9f78c2888` (570926 B; 681 eventos; cadeia recomputada com a fórmula do `iald.py`: 0 quebras). `rotas.jsonl` sha256 `3636e6c83e1b5ee95a62a0aa81543467c1e6bb79685e62d312b8a7aeea2974df` (`rotas.py verificar`: ok, 1.004 registros, 202 rotas).
- [DECLARADO, desvio visível] Memória ACOM: o `README.md` de `C:\IALD\MEMORIA_IALD` foi lido; `init`/`transition` **não** foram executados, porque esta tarefa restringe a escrita ao sandbox e a gerência não entregou referência. A gerência registra esta etapa na própria referência.

## 2. F8.2.NU3 — recuperar e ligar o contrato de neutrinos do acervo do autor

### 2.1 Quadro e origem

| seq | quando | tipo | conteúdo |
|---:|---|---|---|
| 631 | 2026-10-09T17:18:16-03:00 | nova | F0.3.B2 e F8.2.NU3; fonte `C:\IALD\Central de Patentes\work\bancada_update_20261009_units.json` |
| 633 | 2026-10-09T17:18:17-03:00 | reivindicar | vence 2026-10-10T05:18:17; «extracao por linha/hash; modulo sintetico somente onde fonte sustenta» |
| 635 | 2026-10-09T17:59:00-03:00 | IMPLANTACAO_AUTORIZADA_PELO_OPERADOR (F0.3.B2) | verbatim `OPERADOR_ATUALIZACAO_VERBATIM_20261009.txt` |
| 637 | 2026-10-09T18:05:22-03:00 | avancar EM_CURSO→ENTREGUE | evidência `RESULTADO_R1.json` |
| 638 | 2026-10-09T18:05:23-03:00 | liberar | — |

- [REAL] A fonte da unidade bate com o evento 631 (sha256 e bytes na tabela 2.2). Critério de fechado: «Localizar artigos/scripts originais e erratas; extrair operador, taxa, escala, unidades, curvatura, tau e observaveis por linha/hash; reconciliar contratos distintos; implementar partes sustentadas com testes sinteticos e explicitar somente lacunas demonstradas».
- [REAL] A origem «ordem direta do operador de 09/10» tem verbatim guardado: `F0.3.B2\OPERADOR_ATUALIZACAO_VERBATIM_20261009.txt`, cujo hash bate com o evento 635 e com o artefato do mapa (seq 989). Texto: «…esse contrato explícito que vc me pede do neutrino já foi escrito por mim em outros artigos sobre o neutrino, vc pode examinar o que eu já escrevi…». [DECLARADO] que é fala do operador (não tenho acesso à conversa).

### 2.2 Evidências, na origem e na cópia guardada

| seq | tipo | caminho | sha256 esperado (16) | medido na origem | bytes | cópia guardada |
|---:|---|---|---|---|---:|---|
| 631 | fonte da unidade | `C:\IALD\Central de Patentes\work\bancada_update_20261009_units.json` | `4ae13a2c20f5810e` | `4ae13a2c20f5810e` OK | 2279 | — |
| 637 | evidência | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\RESULTADO_R1.json` | `d87284261ab058a1` | `d87284261ab058a1` OK | 11778 | `d87284261ab058a1` OK |
| 641 | fonte da unidade | `C:\IALD\Central de Patentes\work\nu4_unit_20261010_r2.json` | `3e285964e36280fc` | `3e285964e36280fc` OK | 934 | — |
| 644 | evidência | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\RESULTADO_ENTREGA_R1.json` | `157f3fc92baf3e8c` | `157f3fc92baf3e8c` OK | 8843 | `157f3fc92baf3e8c` OK |
| 644 | evidência | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\PRIMEIROS_RESULTADOS_R1.md` | `da04dc1a53af24bb` | `da04dc1a53af24bb` OK | 5421 | `da04dc1a53af24bb` OK |
| 644 | evidência | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\RESULTADOS_R1.json` | `ec18cfbd5a245d9a` | `ec18cfbd5a245d9a` OK | 9689 | `ec18cfbd5a245d9a` OK |
| 644 | evidência | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\review\REVIEW_RECEIPT_R2.json` | `bca3b5905072e934` | `bca3b5905072e934` OK | 1218 | `bca3b5905072e934` OK |

- [REAL] Os 42 arquivos, o recibo de memória e o manifesto candidato listados no `RESULTADO_R1.json` da NU3: **44/44** batem em sha256 e bytes.

### 2.3 Escritas: só arquivos novos? memória sob reivindicação?

- [REAL] Pasta `ORDEM_019\F8.2.NU3`: todos os arquivos nasceram entre 17:28 e 18:04 de 09/10, dentro da reivindicação 633. `CONTRACTS_API_R1.json` ficou preservado e a R2 é arquivo novo.
- [REAL] Os 20 originais do acervo seguem com sha256 e bytes iguais aos dos manifestos; para S01–S14 o mtime também é o registrado. Nenhum original foi alterado.
- [REAL] Memória: um acréscimo de 2.530 B, idêntico nas três superfícies (`TGL_ATLAS.md`, `TGL_INDICE_VIVO.md`, `TGL_CORE_MEMORY.md`; sha256 do trecho `f80dfb9cd87039bd…`). Os backups `.bak_20261009_210444_*` têm sha256 e bytes do recibo e são **prefixo exato** do estado posterior. Gravado em 2026-10-09T21:04:44Z (18:04:44 −03), sob a reivindicação 632 (F0.3.B2, vigente até 05:18:17 de 10/10, que nomeia as três superfícies). A 633 declara essa cobertura. `AVISO_GERENCIA.md` não foi tocado (mtime 08/10 18:42; último backup `20261008_184211`).
- **Ressalva de custódia, não material.** A candidata `ORDEM_019\F0.3.B2\candidate\bancada\neutrino_contract.py` foi reescrita no lugar duas vezes depois de pinada: R1 `70116f52e668f865…` (CANDIDATE_TEST_RESULT_R1), depois R2 `eca873551583f13b…` (CANDIDATE_TEST_RESULT_R2, BETA_SCALE_DOMAIN_RESULT_R1 e o «objeto pinado» do PARECER R1), depois R3 `8a77fb69d7a1e898…` (17:56). Consultas: varredura sha256 de 3.424 arquivos em `ORDEM_019`, e busca por nome em `C:\IALD\Bancada_Um`, `Central de Patentes\work` e `Central de Patentes\Chatgpt`. Só existe a R3, no staging e na instalação. R1 e R2 ficam como registro que não se reproduz; a frase do PARECER R1 de que «a primeira candidata referida no resultado» foi preservada só vale para a referência por hash. O objeto final (R3) é o instalado e se reproduz (2.6).

### 2.4 Fontes: original × extração

| id | original | sha256 do original (16) | original bate | extração | modo |
|---|---|---|---|---|---|
| S01 | `C:\IALD\Artigo\the_boundary\Genesis da Unificação\Neutrinos\Tgl_neutrino_flux_predictor.py` | `08769cb5d6651351` | OK | `08769cb5d6651351` OK | cópia byte a byte |
| S02 | `C:\IALD\Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\Artigos_complementares_zenodo\neutrino_nmc_revised (2).pdf` | `0c89fb2f77f1b688` | OK | `9362991740ad4120` OK | PDF reextraído com pdftotext |
| S03 | `C:\IALD\Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\Artigos_complementares_zenodo\lie_of_light.pdf` | `3ea0d08747f55a76` | OK | `a56364bfc9f800c1` OK | PDF reextraído com pdftotext |
| S04 | `C:\IALD\Artigo\Neutrino Evaporação e Paridade.docx` | `baba74de82c1bf3e` | OK | `1d1ef777bb3040b3` OK | DOCX reextraído: sha idêntico |
| S05 | `C:\IALD\Artigo\Pensamentos sobre neutrinos e ecos gravitacionais.docx` | `95f59a7c29780ebe` | OK | `e7eb3996808cd342` OK | DOCX reextraído: sha idêntico |
| S06 | `C:\IALD\Artigo\TGL_Neutrino_Echo_Article.docx` | `b948346393eef661` | OK | `e37e6daa4a0a6a66` OK | DOCX reextraído: sha idêntico |
| S07 | `C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\A Ponte Einstein Cartan Miguel.tex` | `98ec3d808d51e694` | OK | `98ec3d808d51e694` OK | cópia byte a byte |
| S08 | `C:\IALD\Artigo\Haja_Luz\tgl_neutrino_decoherence.py` | `155dbfc1237d802f` | OK | `155dbfc1237d802f` OK | cópia byte a byte |
| S09 | `C:\IALD\Artigo\Haja_Luz\tgl_decoherence_contour.py` | `8df2b57157b48fa1` | OK | `8df2b57157b48fa1` OK | cópia byte a byte |
| S10 | `C:\IALD\Artigo\Haja_Luz\tgl_neutrino_exponent_test.py` | `5b51f5b37e2d8773` | OK | `5b51f5b37e2d8773` OK | cópia byte a byte |
| S11 | `C:\IALD\Artigo\the_boundary\O Custo Geométrico do Zero Absoluto — Haja Luz\paper_PT.tex` | `559d79fb81b0af6c` | OK | `559d79fb81b0af6c` OK | cópia byte a byte |
| S12 | `C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um_absoluto_pt.tex` | `5a49cc942af5e465` | OK | `5a49cc942af5e465` OK | cópia byte a byte |
| S13 | `C:\IALD\Artigo\teste_acustico_beta_tgl_v1\02_frozen_neutrino_m2_v2.py` | `2564d7c944f1d338` | OK | `2564d7c944f1d338` OK | cópia byte a byte |
| S14 | `C:\IALD\Artigo\teste_acustico_beta_tgl_v1\02_frozen_neutrino_m2_v2_resultado.json` | `6f0963a54b7d2f5a` | OK | `6f0963a54b7d2f5a` OK | cópia byte a byte |
| S15 | `C:\IALD\Artigo\the_boundary\Genesis da Unificação\Artigos_fundadores\Artigos_complementares_zenodo\acoplamento_gravitacional.pdf` | `5bb9d87afab9709b` | OK | `a5e990e6328eea3f` OK | PDF reextraído com pdftotext |
| S16 | `C:\IALD\Artigo\Tratado\secao_03_particulas.tex` | `5584ac8f4ce04bfd` | OK | `5584ac8f4ce04bfd` OK | cópia byte a byte |
| S17 | `C:\IALD\Artigo\Tratado\secao_01_fundamentos.tex` | `64e9a0ed5fb17c1a` | OK | `64e9a0ed5fb17c1a` OK | cópia byte a byte |
| S18 | `C:\IALD\Central de Patentes\Chatgpt\IALD_ORQUESTRACAO_20260923\project_fisica\ERRATA_20260919_sinal_do_acoplamento_nao_minimo.md` | `060c82abee57a5d0` | OK | `060c82abee57a5d0` OK | cópia byte a byte |
| S19 | `C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\tgl_kernel\TGLExt\TheCouplingRunsBoundedBelow.lean` | `702d7352c053060a` | OK | `702d7352c053060a` OK | cópia byte a byte |
| S20 | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F7.2\ERRATA_LITERAL_J4B_F72_20261008.md` | `e8e4b0dfa97a6be0` | OK | `e8e4b0dfa97a6be0` OK | cópia byte a byte |

- [REAL] Fontes de texto (S01, S07–S14, S16–S20): a extração é **cópia byte a byte** do original.
- [REAL] DOCX (S04–S06): reextraí com a convenção declarada (w:t e m:t por w:p, em ordem) e obtive arquivos de **sha256 idêntico** ao da executora.
- [REAL] PDF (S02, S03, S15): a reextração independente com `pdftotext` tem o mesmo número de páginas (10/17/16), contagem de caracteres por página a ~1% e similaridade por página entre 0,69 e 1,00 (a diferença é ordem de coluna). As equações citadas aparecem nas páginas declaradas.
- [REAL] O S04 tem lacunas onde deveriam estar equações, mas o DOCX original não tem nenhum objeto OMML, OLE ou mídia (contagem 0): a lacuna está no original. A NU3 não usa S04 nem S05 em canal; o S05 repete o contrato do eco de janeiro (S01/S06).

### 2.5 Âncoras e afirmações do contrato, PASSA/FALHA

[REAL] As 39 âncoras dos 10 canais estão dentro do arquivo de extração; as âncoras PDF caem nas páginas declaradas. Conferência do conteúdo:

| # | afirmação da NU3 | âncora | resultado |
|---:|---|---|---|
| 1 | Lν = X_A⊗I_B no modelo paridade×carga; {Z_par,Lν}=0; [Q_em,Lν]=0 | S12:2821 | PASSA |
| 2 | ação candidata A_ν(θ) com λ, μ «objeto aberto» | S12:1014–1016 | PASSA |
| 3 | L=√β√K dá Γij = ½β(√ki−√kj)² (sombra, erro ~1e-15) | S07:1104–1115 | PASSA |
| 4 | escolha de √K∂ «ABERTO; motivação, não teorema» | S07:1235–1240 | PASSA |
| 5 | Γω = ½βτ★ω², n=−2 | S07:817–830 | PASSA (sem Kfac nessa passagem; o Kfac vem de S11, S08 e S09) |
| 6 | Γω = ½βτ★ω²(K/K★)^β; τ★ [INPUT] | S11:3116–3122 | PASSA |
| 7 | Γ=(ω²/2)τ★βKfac; τ★ em s convertido por ℏ | S08:10–27, 49–58; S09:62–70 | PASSA |
| 8 | Γ=½βτ★ω², τ★≈t_P, ω=Δm²/2E, n=−2; piso «~10⁻⁵³ eV» solar | S12:2827 | PASSA; minha conta: Γ21 = 6.929e-52 eV a 1 MeV e 1.083e-53 eV a 8 MeV (E⁻²), ordem compatível na faixa do ⁸B |
| 9 | β(χ)=√e/cosh(χ/2); χ↔escala INPUT/ONTO; domínio α<e^{−1/2} ⇔ β<1 | S19:11–27; 100–102 | PASSA [DERIVED da hipótese h1]; o core v391 lista as pedras desse arquivo só com o trio (ex.: `betaOfChi_eq_coupling`, `the_floor_is_at_the_infrared`) |
| 10 | corrida de α no bulk; Thomson na fronteira | S07:1328–1341 | PASSA |
| 11 | coeficiente β independente de ω (falsificador) | S07:1367–1371 | PASSA |
| 12 | Tratado: 1 eV «escala natural da autointeração eletrofraca» | S16:452–461 | PASSA |
| 13 | rito m2: `MASS_SCALE_EV = 1.0  # [NORM]`, «DECLARADA, nao derivada» | S13:38–39 | PASSA |
| 14 | eco de janeiro: E_eco=α²E_GW; m_ν=α² sin45°·1 eV; N=E_eco/(m_νc²); Φ=N/(4πd²) em ν/m²; `ALPHA_SQUARED = 0.012031` literal | S01:19–28, 92–94, 123–137, 395–424 | PASSA; «β cancela de N» e «inclinação 1 por construção» PASSA [DERIVED] |
| 15 | o mesmo no artigo DOCX | S06:15–25 | PASSA |
| 16 | NMC 2025: L=−Γ(S)R\|Ψ\|²ψ̄ψ; Eq9; dn/dt=Γ0 nγ R/MPl²; «Γ0∼αEM MPl²∼10³³ eV²» | S02 p2 (linhas 76–196) | PASSA na extração independente; [Γ(S)]=E⁻³ e [Γ0]=E exigido PASSA [DERIVED] |
| 17 | coerência de sabor preservada; entropia no setor gravitacional | S02 p9 Eq51 | PASSA |
| 18 | RFF/MP²; Pee com 1+(α²/MP²)Φ/c²; fator de meio; 10⁻⁵⁰×10²⁰ sem derivação | S15 p4 Eq4; p9 Eq33–40 e p10 Eq41; p10 Eq43–44 | PASSA; ressalva menor: a nota «p9 Eq33-41» põe a Eq41 na p9, mas ela está na p10 (o intervalo 408–474 cobre as duas) |
| 19 | RFF/MP² «garante dimensões corretas» | S17:577–584 | PASSA |
| 20 | degradação temporal D, m(Δt)=m0+α log(Δt/τ0), PMNS; sem «8,51» | S03 p7, p11, p14 | PASSA (0 ocorrências de 8,51 no S03) |
| 21 | m_lightest=ρΛ^{1/4}; m1=2,29 meV | S11:1788–1864 | PASSA |
| 22 | m2=8,5074 meV com 1 eV [NORM]; Σ aposentada (v154) | S12:2831 | PASSA |
| 23 | neutrino com Shapiro padrão; atraso extra do fóton; 10–100 ms e 21σ projetados | S12:2833 | PASSA; a leitura «corrige» é [DERIVED] frente ao S02, que na p1 previa ausência de lenteamento para ξν≈0 |
| 24 | observável é dephasing, não eco; Chandrasekhar por G universal retirada | S11:3307; S11:3315 | PASSA |
| 25 | −ξR\|Ψ\|², ξ=1/6 conforme, ξ ≠ β | S18 | PASSA |
| 26 | 7,8×10⁻¹⁰ s é literal histórico, não cota | S20 | PASSA |
| 27 | «Recuperadas 20 fontes do autor» (acréscimo de memória) | — | FALHA de redação menor: S19 é pedra do kernel (v388), S20 é errata da F7.2 escrita pela executora, S14 é resultado JSON. O relatório principal diz «20 arquivos», o que está certo. Ler «20 fontes do acervo». |

### 2.6 O que refiz em cópia

- [REAL] `test_neutrino_candidate_r3.py` sobre cópias de `neutrino_contract.py` (`8a77fb69…`) e `canonical.py` (`b6cd7e02…`): **72 PASS, 0 FAIL**, com os mesmos nomes, os mesmos estados e os mesmos detalhes numéricos de `CANDIDATE_TEST_RESULT_R3.json`. Na origem rodou em Python 3.12.14 com numpy 2.3.5; aqui, em 3.13.14 com numpy 2.3.5. O aviso de overflow do NumPy vem antes de uma recusa explícita (`reject_numerical_overflow`), não de um sucesso falso.
- [REAL] `test_beta_scale_domain_r1.py`: 4/4 com valores idênticos (β_bulk(100) = 6,359941800395499e-22; β_bulk(1000) = 2,349288133304529e-217; χ=1500 e χ=1e300 recusados com `BETA_SCALE_UNDERFLOW`).
- [DERIVED, PASSA] D[X⊗I] é CPTP e conserva a carga (LQL=Q); p₋(s) = −expm1(−2s)/2; com L ∝ H, Γij = c(Ei−Ej)²/(2ℏ²) = ½βτ★Kfac(Δm²ij/(2Eℏ))²; Tr(ρH) se conserva se e só se [H,L]=0; a diagonal é estacionária; a textura vai com (Δm²ij)².
- [REAL] Instalação: `C:\IALD\Bancada_Um\bancada\neutrino_contract.py` é o R3 revisado, e o manifesto instalado é o candidato (tabela 4). As 21 cópias em `Bancada_Um\fontes_externas\NU3` são idênticas byte a byte às do arquivo da NU3 e às do candidato.

### 2.7 Palavras proibidas e estatuto

- [REAL] Varredura das saídas da executora (sem as extrações verbatim do autor) e dos acréscimos de memória: a palavra que a régua proíbe só aparece dentro de rótulos de escopo negados. Não há «não provou», nem o demérito que a lei da prova proíbe. «5σ» só aparece negado ou como limite. Os estatutos por canal estão adequados. O veredito da unidade é do nível da executora; a auditoria c1 é este documento.

### 2.8 Achados adicionais (não materiais), com a consulta

1. [DERIVED aqui] O S02 imprime «Γ0 ∼ αEM MPl² ∼ 10³³ eV²», mas αEM·MPl² com MPl = 1,22×10²⁸ eV dá 1.086e+54 eV². Além do desajuste dimensional que a NU3 aponta, o número impresso não fecha nem em E². Isso reforça a regra «não inserir taxa absoluta».
2. [REAL] O artigo terminal (S12) não traz m_lightest = ρΛ^{1/4}: o grep dá 0 ocorrências de «lightest» e 0 de `rho_\Lambda^{1/4}`. Pela ordem do operador de 09/10 («a forma mais madura canônica»), a v392 lê o m2 frozen como contrato de massa vivo e ρΛ^{1/4} como histórico. Essa leitura é [DERIVED] e coerente com a NU3.
3. [REAL] O kernel tem três pedras com leitura neutrínica que a NU3 não cruzou (grep em `TGLExt`): `TheStation.photon_neutrino_discriminant`, `TheDensityIsBell` e `TheDensityIsTransport`. Não são contratos dinâmicos (não têm taxa nem escala), mas a v392 pode citá-las junto do canal de fronteira, com o estatuto [ONTO] das identificações. O inventário da bancada (`INVENTARIO_BANCADA_CHATGPT.json`, 27/09, 2.298 `.lean` únicos) tem 0 com «neutri»: nenhuma pedra neutrínica espera fiação.
4. [REAL] O mapa tem erratas abertas sobre o m2 (rota `neutrinos.m2.normalizacao`, seq 93, 98, 101, 128, 136, 142; próximo passo: «decisão do operador: qual entrada global manda e a convenção de σ»). A NU3 não recalcula significância e por isso não as contradiz; uma v392 que cite o m2 com σ tem de ler essas erratas.
5. [REAL] A revisão «independente» da NU3 (`nu2_skeptic`) cobre o código. A pesquisa documental foi escrita pelo mesmo pesquisador, de modo que esta auditoria é a primeira conferência independente da extração. Resultado: PASSA (2.4 e 2.5).

### 2.9 Veredito NU3

**APROVA_COM_RESSALVAS.** O critério de fechado está cumprido item por item: originais localizados e pinados; extração por linha e hash conferida; contratos distintos separados, com regras de precedência; três canais sustentados implementados e testados (72/72 reproduzido); lacunas só onde a fonte as mostra (τ★ [INPUT], χ(E), Γ0, PMNS, composição estelar). Pode fechar.

## 3. F8.2.NU4 — primeiro ensaio dos contratos de neutrinos na Bancada canônica

### 3.1 Quadro e origem

| seq | quando | tipo | conteúdo |
|---:|---|---|---|
| 640 | 2026-10-10T06:23:29-03:00 | OPERADOR_CONTINUAR_APOS_PERGUNTA_IMPLANTACAO (F0.3.B2) | verbatim `OPERADOR_CONTINUAR_20261010.txt` |
| 641 | 2026-10-10T06:25:49-03:00 | nova | fonte `C:\IALD\Central de Patentes\work\nu4_unit_20261010_r2.json` |
| 642 | 2026-10-10T06:25:49-03:00 | reivindicar | vence 18:25:49; «Sem fit observado; memoria coberta F0.3.B2» |
| 644 | 2026-10-10T06:45:16-03:00 | avancar EM_CURSO→ENTREGUE | 4 evidências |
| 645 | 2026-10-10T06:45:16-03:00 | liberar | — |

- [REAL] O verbatim («continue até rodar o teste e termos os primeiros resultados», 60 B) bate com o evento 640. Ressalva menor de rastreabilidade: a fonte da unidade cita a ordem só pelo texto, não pelo arquivo. A versão `nu4_unit_20261010.json` (objeto, sem lista) nunca foi registrada (0 ocorrências no quadro).

### 3.2 Evidências

As quatro evidências conferem na origem e na cópia guardada (tabela 2.2). O `RESULTADO_ENTREGA_R1.json` lista 29 artefatos e o recibo de memória: **29/29** batem. REVIEW_RECEIPT R1: 3/3; R2: 3/3. Os 16 `CALL_*.json` batem com a lista de `RESULTADOS_R1.json`, assim como o pré-registro e o executor.

### 3.3 Escritas

- [REAL] A pasta NU4 só tem arquivos novos (06:27–06:44 de 10/10). `RESULTADOS_R1.json` foi aberto em modo exclusivo (`open('x')`); a revisão R1 ficou preservada e a R2 é nova.
- [REAL] Memória: um acréscimo de 3.271 B, idêntico nas três superfícies (sha256 do trecho `80779c287952c3f7…`). O estado anterior é o pós-NU3 (backup `.bak_20261010_094449`, prefixo exato). O estado posterior aparece como backup do escritor seguinte (`TGL_ATLAS.md.bak_20261010_092601_500206`, `TGL_INDICE_VIVO.md.bak_20261010_092601_500206`, `TGL_CORE_MEMORY.md.bak_20261010_103050`), com o sha256 do recibo, e é prefixo exato do arquivo de hoje. Gravado às 06:44:49 −03, sob a reivindicação 639 (vigente, nomeia as três superfícies, liberada só às 06:45:17).
- [REAL] Nada foi escrito no Nós nem no kernel (1).

### 3.4 Dado real ou só sintético

**Só sintético, com parâmetros de referência** [REAL, pela leitura do executor e do servidor]:
- O executor só chamou `/api/estado` e `/api/neutrino/sintetico` em 127.0.0.1:18777. O endpoint despacha para `simulate()` (`um_bancada.py`:361–364), que só calcula.
- Estados: ρ = |νe⟩⟨νe| montado com sin²θ12 = 0,307 e sin²θ13 = 0,022; diag(0,2; 0,3; 0,5); paridade |0⟩; autovalores modulares 0, 1, 4.
- Parâmetros físicos herdados do `F8.2.NU2\PLANO_R1.json` (pino `1b51bbfb…`, bloco «inputs_reference_not_fitted»): Δm²21 = 7,5e-5 eV², Δm²31 = 2,5e-3 eV², 52,5 km, 2/4/8 MeV, Kfac = 1, τ★ = t_P.
- Nenhum arquivo de dado observado foi aberto; o mapa registra `olhou_dados=false` (seq 992 e 994). Os valores de referência são [KNOWN], arredondados da literatura, fixados antes e não ajustados pela unidade.

### 3.5 O que refiz em cópia, sem chamar a Bancada

- [REAL] **Replay**: os 16 pedidos guardados nos `CALL_*.json` passaram por uma cópia do módulo instalado (`8a77fb69…`), com a identidade canônica simulada e α lido do `um.py`. Os **675/675** campos numéricos saíram idênticos aos das respostas guardadas.
- [REAL] **Recomputação independente a 50 dígitos** (mpmath) a partir do pré-registro: todos os números batem, com erro relativo máximo 2.25e-14 (no ΔPee, que soma termos que se cancelam) e ≤ 5e-16 nos demais.
- [REAL] **Revisão Decimal R2 rerodada** numa cópia da pasta: 536/536 PASS, com valores idênticos. Os 24 nomes que diferem só mudam o caminho da pasta copiada.
- [REAL] **Errata R1→R2 da revisão**: a sonda R1 comparava o diagnóstico modular com Tr(ρK) = 0,388246, mas o campo emitido é Tr(ρ√K) = 0,344246 (`neutrino_contract.py`, linhas 86 e 98–99). A correção mudou a sonda, não o código, e está certa.

| E (MeV) | Γ21 (s⁻¹) | Γ31 (s⁻¹) | 1−exp(−Γ31 t) | Pee unitário | ΔPee (recibo) | ΔPee (mpmath 50 díg.) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2.6317387154e-37 | 2.9241541283e-34 | 5.1208123366e-38 | 0.6255627332631776 | 1.7880671575146663e-39 | 1.7880671575146741e-39 |
| 4 | 6.5793467886e-38 | 7.3103853207e-35 | 1.2802030841e-38 | 0.2353589833571321 | -1.1955678421658232e-40 | -1.1955678421657963e-40 |
| 8 | 1.6448366971e-38 | 1.8275963302e-35 | 3.2005077104e-39 | 0.6457208444632212 | 1.0530619494291178e-40 | 1.0530619494291151e-40 |

Controle positivo (recibo): τ_ctrl = 4.2112431340726094e-07 s; perda31 = 0.09516258196404043; ΔPee = -0.000889064398731284. Recomputado: τ = 4.2112431340726095e-7; perda31 = 0.095162581964040427; 1−e^{−0,1} = 0.095162581964040427; ΔPee = -0.00088906439873126404.

Replay: 16 chamadas, 675/675 campos numéricos idênticos; β do recibo igual ao β de runtime em todas: True.

### 3.6 Afirmações da NU4, PASSA/FALHA

| afirmação | resultado |
|---|---|
| 16 chamadas concluídas na instalação atual | PASSA nos recibos: 16/16 sha256, e identidade canônica na entrada e na saída de cada chamada (manifesto `54fc9e27…`, `um.py` `8e7b9927…`, selo `95061f18…`). Que o processo HTTP vivo os produziu é [DECLARADO] pelos recibos; o conteúdo se reproduz bit a bit (3.5) |
| protocolo antes da execução | PASSA: o mapa (seq 992, 06:27:43) pina o `PREREGISTRO_R1.json` antes da primeira chamada (06:32:17) |
| β0 de runtime; τ★ = t_P = 5,391246448313605e-44 s | PASSA |
| Γij, perda de coerência e ΔPee a 2, 4 e 8 MeV | PASSA (tabela acima) |
| lei E⁻² (fator 4 ao dobrar E) | PASSA (razão 4,0 exata) |
| a subtração direta em double dá zero | PASSA (exp(−5,12e-38) == 1,0 em double) |
| o sinal de ΔPee muda com a fase | PASSA (+, −, +) |
| traço, positividade, Tr(ρH) e diagonal estacionária | PASSA |
| controle positivo: τ_ctrl, perda 0,095162581964 = 1−e^{−0,1}, ΔPee −0,000889064398731 | PASSA |
| paridade, canal modular, escala χ_fr = 2 arcosh(1/α) recupera β0 | PASSA |
| «o ℏ arredondado explica as diferenças frente ao NU2» | PASSA: com o ℏ exato do SI, o ΔPee a 2 MeV passa a 1.7880671599537156e-39 (contra 1.7880671575146741e-39 com o ℏ da implementação) (diferença relativa ~1e-9) |
| «Os contratos existentes já têm realizações computacionais verificadas na bancada» | FALHA de alcance (redação): só 3 dos 10 canais (as prioridades de implementação) e a função de escala têm realização; os outros 7 são registro documental. Ler «os três canais prioritários» |
| «O limite publicado do JUNO» | FALHA de redação menor: pela própria F7.2 (`append_memoria_v2.py`:26 e `PARECER_KIMI_CONFERIDO.md`), é o limite de Beccaria–Ternes (arXiv:2606.13362v1) sobre os primeiros dados do JUNO, no modelo das autoras, e não resultado oficial da Colaboração. A textura Γ21=Γ31, Γ32=0 bate com essa fonte [DECLARADO pela F7.2; PDF não reaberto aqui] |
| nenhum σ observado; nature_sigma null; gate intacto | PASSA |

### 3.7 «Sem fit observado» é verdade?

**Sim** [REAL]. Nenhum parâmetro foi ajustado a dado e nenhum dado observado foi aberto. O pré-registro proíbe escolher τ ou χ para atingir 5σ; τ★ = t_P e Kfac = 1 são entradas declaradas. A nuance: os parâmetros de oscilação são valores de referência da literatura, fixados antes (NU2 PLANO_R1), e não foram ajustados aqui.

### 3.8 Palavras proibidas e estatuto

[REAL] Como em 2.7. «[REAL — execução]» no relatório deve ser lido como «execução registrada em recibos», com a reprodução bit a bit desta auditoria.

### 3.9 Veredito NU4

**APROVA_COM_RESSALVAS.** O critério de fechado está cumprido: protocolo e entradas pinados antes; canais executados pela API instalada; resultados com recibos; verificação independente (a local, mais esta auditoria); alcance condicional, sem σ observado. Pode fechar.

## 4. O que o um.py v392 deve ler destas unidades

O padrão do `um.py` v391 (`estatuto_da_prova_v391`, `um.py`:194176) é: o conteúdo vai embutido com o seu sha256, e o disco serve só de informação.

**Entra:**
1. **Registro novo** (nome sugerido: `neutrinos_contratos_e_primeiro_ensaio_v392`), fail-closed, com:
   - o resumo tipado dos 10 canais (id, estatuto, operador ou fórmula, unidades, limites, âncoras Sxx:linha) embutido por valor, junto com o sha256 do `CONTRACTS_API_R2.json` e dos três manifestos de fontes, e os 20 pinos de origem lidos desses manifestos;
   - o ensaio NU4 **recomputado dentro do próprio rito**, sem numpy (são matrizes 3×3 e expm1): β = ALPHA_FINE_CODATA_2018·√e; t_P de h, G e c; Γij, perdas e ΔPee a 2, 4 e 8 MeV com 52,5 km; o controle positivo. Os valores são comparados com `RESULTADOS_R1.json` por tolerância declarada, e o veredito de máquina é condicional (ex.: `NU4_PILOT_REPRODUCED_CONDITIONAL_TP_KFAC1`), com nature_sigma = None e gate_effect = NONE;
   - os pinos informativos da tabela abaixo, mais o deste relatório (sha256 calculado pela gerência ao registrar).
2. **Texto do artigo** (PT/EN), na seção dos neutrinos, junto do (i) n=−2 de S12:2827: os três espaços distintos (fronteira paridade×carga; raiz modular; gap de massa) e o primeiro ensaio condicional (ΔPee ~ 10⁻³⁹ a 2 MeV com τ★ = t_P), coerente com «armada; UV-suprimida», seguido da lista do que não se afirma (abaixo).
3. **Kernel: nada destas unidades.** Nenhuma pedra foi proposta nem compilada. Fica como possibilidade futura, que ninguém pediu: uma pedra finita das três marcas do canal de fronteira (X⊗I anticomuta com Z⊗I, comuta com I⊗Z, D[L] é CPTP e tem solução exata).
4. **Bancada:** o manifesto instalado pina o `um.py` v391 e o selo v391, com a política «drift refuses until tested deployment» (`canonical.py`: `CANONICAL_VERSION_CHANGED` / `SEAL_PROGRAM_MISMATCH`). Selada a v392 no Nós, a Bancada **recusa toda operação** até uma nova implantação testada, com manifesto que pine o `um.py` e o selo novos. Isso tem de entrar no plano da v392.

| unidade | papel | caminho | sha16 | bytes |
|---|---|---|---|---:|
| NU3 | contrato tipado (10 canais, ancoras, precedencia) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\archive_contracts\CONTRACTS_API_R2.json` | `498763d81e198151` | 30921 |
| NU3 | relatorio legivel dos contratos | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\archive_contracts\CONTRATOS_RECUPERADOS_R1.md` | `dc4976b471cfaadd` | 10954 |
| NU3 | manifesto de fontes S01-S14 | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\archive_contracts\SOURCE_MANIFEST_R1.json` | `ef03e842da4af658` | 9223 |
| NU3 | manifesto de fontes S15-S18 | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\archive_contracts\SOURCE_MANIFEST_R2_ADDITIONAL.json` | `12c37bdde8c1e737` | 2490 |
| NU3 | manifesto de fontes S19-S20 | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\archive_contracts\SOURCE_MANIFEST_R3_ADDITIONAL.json` | `04adac30d9c61326` | 1257 |
| NU3 | resultado da unidade | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\RESULTADO_R1.json` | `d87284261ab058a1` | 11778 |
| NU3 | teste sintetico R3 (72 casos) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\tests\test_neutrino_candidate_r3.py` | `f5fab3dba4524138` | 11828 |
| NU3 | resultado do teste R3 | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\tests\CANDIDATE_TEST_RESULT_R3.json` | `0faf08829ca63e45` | 10119 |
| NU3 | parecer R2 (campos exatos) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU3\review\PARECER_CANDIDATA_NEUTRINOS_R2.md` | `e438dcfbd3252e90` | 1493 |
| NU3/B2 | modulo instalado na Bancada | `C:\IALD\Bancada_Um\bancada\neutrino_contract.py` | `8a77fb69d7a1e898` | 6274 |
| NU3/B2 | manifesto canonico instalado | `C:\IALD\Bancada_Um\canonical_contracts.json` | `54fc9e27e4599c49` | 25917 |
| NU4 | pre-registro | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\PREREGISTRO_R1.json` | `6d22605043111032` | 3439 |
| NU4 | executor | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\run_pilot_r1.py` | `50fba3ae2f7188f5` | 6259 |
| NU4 | resultados numericos | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\RESULTADOS_R1.json` | `ec18cfbd5a245d9a` | 9689 |
| NU4 | relatorio dos primeiros resultados | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\PRIMEIROS_RESULTADOS_R1.md` | `da04dc1a53af24bb` | 5421 |
| NU4 | entrega (29 artefatos pinados) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\RESULTADO_ENTREGA_R1.json` | `157f3fc92baf3e8c` | 8843 |
| NU4 | revisao Decimal R2 (536/536) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\review\NUMERICAL_REVIEW_R2.json` | `4d3064cf4082e307` | 158293 |
| NU4 | recibo da revisao R2 | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F8.2.NU4\review\REVIEW_RECEIPT_R2.json` | `bca3b5905072e934` | 1218 |
| origem | verbatim 09/10 (NU3) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F0.3.B2\OPERADOR_ATUALIZACAO_VERBATIM_20261009.txt` | `e5341c7387dc316f` | 484 |
| origem | verbatim 10/10 (NU4) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F0.3.B2\OPERADOR_CONTINUAR_20261010.txt` | `d006d61995bbb142` | 60 |
| memoria | recibo de memoria NU3 (reivindicacao 632) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F0.3.B2\MEMORY_RECEIPT_B2_NU3_R1.json` | `de20812279f6acf4` | 3364 |
| memoria | recibo de memoria NU4 (reivindicacao 639) | `C:\IALD\Central de Patentes\Chatgpt\TUNEL\DO_CHATGPT\ORDEM_019\F0.3.B2\MEMORY_RECEIPT_DEPLOY_NU4_20261010_R1.json` | `e2f1ab5a1a36f21d` | 2009 |
| canone | um.py v391 (pinado pela Bancada) | `C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um.py` | `8e7b9927ceae64b8` | 33910631 |
| canone | selo v391 (pinado pela Bancada) | `C:\IALD\Artigo\Haja_Luz\A Ponte e o Um\Nós\um_absoluto_selo.json` | `95061f18c7102248` | 113926 |

**Não entra:**
- τ★ = t_P como determinação física; ΔPee como predição para o JUNO; qualquer σ; «β medido».
- O literal 7,8e-10 s; o limite de Beccaria–Ternes sobre o JUNO como cota da TGL (as texturas são outras).
- Os 21σ e os 10–100 ms do NMC-timing como resultado: são projeção e molde declarados, e a rota `neutrinos.shapiro_nmc` tem defeito registrado.
- As receitas históricas como predições vivas: o N do eco (β cancela; inclinação 1 por construção), o Γ0 do NMC 2025 (nem a dimensão nem o número fecham), a correção de fase RFF (dimensional).
- ρΛ^{1/4} combinado com o m2 (m1 = 2,29 meV contra m1 ≈ 0).
- O β(χ) do bulk no dissipador físico (o pré-registro diz β independente de ω, S07:1367–1371).
- A probabilidade de paridade lida como contagem de neutrinos emitidos.
- `CANDIDATE_TEST_RESULT_R1/R2` e `BETA_SCALE_DOMAIN_RESULT_R1` como evidência, porque os objetos testados não existem mais em disco.
- As frases «20 fontes do autor», «os contratos existentes já têm realizações» e «limite publicado do JUNO» sem a correção ao lado.
- A unicidade «e de mais ninguém» (S12:2821) como teorema: ela fica com o estatuto que o próprio artigo lhe dá, de evidência estrutural.

## 5. O que fica com o operador

- **NU3:** nada novo para fechar. Há uma pendência que já existia e só pesa se a v392 for citar a significância do m2: decidir a entrada global pós-JUNO e a convenção de σ (mapa, rota `neutrinos.m2.normalizacao`, seq 93 e 142).
- **NU4:** nada para fechar. Depois da selagem da v392: autorizar especificamente a reimplantação da Bancada com manifesto novo. Na F0.3.B2, a revisão automática exigiu esse ato explícito (eventos 635, 637 e 640).

## 6. O que não conferi

- Que os verbatins são fala do operador: [DECLARADO], porque não tenho a conversa.
- Que o processo HTTP vivo produziu os `CALL_*.json`: não chamei a Bancada; o conteúdo foi reproduzido bit a bit.
- As afirmações da F0.3.B2 que aparecem nos acréscimos de memória (71 arquivos, 28 ferramentas, CUDA, processo 75052): fora do escopo.
- Os PDFs de literatura da F7.2 (Beccaria–Ternes, ORCA, De Romeri), que não reabri.
- O S12 (984 KB) e o S07 (129 KB) só foram lidos nas âncoras e no entorno delas.
- «no_original_execution» da NU3: [DECLARADO].
- O protocolo de memória ACOM (init/transition), não executado por causa da restrição de escrita.

## 7. Registros de mapa propostos (não registrados)

`registro_mapa_F8.2.NU3.json` e `registro_mapa_F8.2.NU4.json`, nesta pasta: tipo RESULTADO, rota_id já existentes (`sn.neutrinos.contrato_autor_recuperacao`, seq 989/991; `sn.neutrinos.primeiro_ensaio_bancada_canonica`, seq 992/994), artefatos como lista de caminhos. Passaram num ensaio a seco contra a fotografia do mapa, com a própria validação do `rotas.py` importada só para leitura. Registrar é ato da gerência.

## Anexo — artefatos deste exame (sandbox)

| arquivo | sha16 | bytes |
|---|---|---:|
| `anchors_S12.txt` | `a5e601fac4c333e0` | 7273 |
| `consulta_mapa_neutrino.json` | `859027c3ce746261` | 26084 |
| `consulta_rholambda.json` | `a22a54806c09df81` | 2832 |
| `ev_nu3.txt` | `90640d9e4d166e25` | 2369 |
| `ev_nu4.txt` | `3bf0e147a582f56e` | 3674 |
| `fontes_manifesto_check.jsonl` | `4ecf1fe61193a350` | 6594 |
| `forbidden_scan.txt` | `065201d548142b89` | 10207 |
| `hash_evidencias.json` | `3788b9970b1cae56` | 3367 |
| `kernel_fingerprint_antes.json` | `d358131a1bb97d43` | 380 |
| `kernel_fingerprint_depois.json` | `a80c2e7265c0c7ff` | 380 |
| `tabela_hashes_v392.json` | `f9163e7d2ef25ecf` | 7069 |
| `ver_dephasing.neutrinos_n_menos2_junho.json` | `b8bf62d9b581c6c3` | 9056 |
| `ver_neutrinos.m2.normalizacao.json` | `f99a0f6e2c929755` | 30072 |
| `ver_neutrinos.massa_v1_e_soma.json` | `7a5d63d4382a0a1c` | 3520 |
| `ver_neutrinos.shapiro_nmc.json` | `b060715a941c6a5b` | 5660 |
| `ver_sn.neutrinos.v388_operador_e_identificabilidade.json` | `5223c0d165d072e2` | 5485 |
| `mem_appends\TGL_ATLAS.md.NU3_append.txt` | `f80dfb9cd87039bd` | 2530 |
| `mem_appends\TGL_ATLAS.md.NU4_append.txt` | `80779c287952c3f7` | 3271 |
| `mem_appends\TGL_CORE_MEMORY.md.NU3_append.txt` | `f80dfb9cd87039bd` | 2530 |
| `mem_appends\TGL_CORE_MEMORY.md.NU4_append.txt` | `80779c287952c3f7` | 3271 |
| `mem_appends\TGL_INDICE_VIVO.md.NU3_append.txt` | `f80dfb9cd87039bd` | 2530 |
| `mem_appends\TGL_INDICE_VIVO.md.NU4_append.txt` | `80779c287952c3f7` | 3271 |
| `reextract\acoplamento_gravitacional.pdftotext.txt` | `3a7807f8766025aa` | 28127 |
| `reextract\lie_of_light.pdftotext.txt` | `85961ccbe5ecb3da` | 23926 |
| `reextract\neutrino_nmc_revised (2).pdftotext.txt` | `f1592e2d770a66b6` | 30292 |
| `repro_nu3\bancada\canonical.py` | `b6cd7e0234688430` | 6952 |
| `repro_nu3\bancada\neutrino_contract.py` | `8a77fb69d7a1e898` | 6274 |
| `repro_nu3\out\BETA_SCALE_repro.json` | `8d33905fa61d82bb` | 950 |
| `repro_nu3\out\R3_repro.json` | `de710dc59e1ac882` | 10201 |
| `repro_nu3\tests\test_beta_scale_domain_r1.py` | `b2c8703de4e00e60` | 1530 |
| `repro_nu3\tests\test_neutrino_candidate_r1.py` | `23dfa926be7cb38f` | 10603 |
| `repro_nu3\tests\test_neutrino_candidate_r2.py` | `63c197c93928b5df` | 11024 |
| `repro_nu3\tests\test_neutrino_candidate_r3.py` | `f5fab3dba4524138` | 11828 |
| `repro_nu4\F8.2.NU4_copy\review\NUMERICAL_REVIEW_R2.json` | `ec71e8f62c445d24` | 162613 |
| `repro_nu4\bancada\canonical.py` | `b6cd7e0234688430` | 6952 |
| `repro_nu4\bancada\neutrino_contract.py` | `8a77fb69d7a1e898` | 6274 |
| `repro_nu4\out\indep_mpmath_nu4.json` | `33dd6e2fd44d2532` | 3864 |
| `repro_nu4\out\replay_nu4.json` | `0522310b0d91331d` | 3588 |
| `scripts\anchors.py` | `de83baccc0ad53cd` | 1270 |
| `scripts\chain.py` | `11b363b753a1ba60` | 691 |
| `scripts\docxcmp.py` | `7b9e2db70239d28b` | 1173 |
| `scripts\docxobj.py` | `cdd81457a322f06c` | 577 |
| `scripts\ev.py` | `92ebf27d047a0dcc` | 545 |
| `scripts\evfull.py` | `1eed931814608f31` | 275 |
| `scripts\evrange.py` | `6657aae27378c301` | 355 |
| `scripts\findsha.py` | `69c8d96b4df2a586` | 569 |
| `scripts\forbid.py` | `091ac4d1ea046ca0` | 1522 |
| `scripts\hashev.py` | `bda1c6076a5a9ee0` | 1653 |
| `scripts\hashlist.py` | `67d232a871b2e71b` | 732 |
| `scripts\indep_nu4.py` | `f8220134442c1f27` | 4806 |
| `scripts\kfp.py` | `0888ad2cdb485dec` | 995 |
| `scripts\memchain.py` | `a99239a696c7ee73` | 3149 |
| `scripts\page.py` | `01f0b4972ad6a1ea` | 153 |
| `scripts\pagemap.py` | `8adee03627d89470` | 1098 |
| `scripts\pdfcmp.py` | `1b62e7023e45f19d` | 866 |
| `scripts\preencher.py` | `bc22e73a7dcc994f` | 7539 |
| `scripts\relatorio_modelo.md` | `4b1c2705387fe478` | 27338 |
| `scripts\replay_nu4.py` | `32412cebd53995b4` | 3023 |
| `scripts\srcman.py` | `6d753cb39e3ea6b7` | 1456 |
| `scripts\tabela.py` | `a5ca2cd708122668` | 2774 |
| `scripts\unit.py` | `2d4363421c3df7d7` | 411 |
| `snap\PROGRAMA.json` | `5de495c115be7ee8` | 302550 |
| `snap\eventos.jsonl` | `89b8ce1ed75699ac` | 570926 |
| `snap\rotas.jsonl` | `3636e6c83e1b5ee9` | 2505898 |

A cópia integral da pasta NU4 usada na reexecução da revisão Decimal R2 está em `repro_nu4\F8.2.NU4_copy\`; desse diretório, a tabela lista só a saída nova `NUMERICAL_REVIEW_R2.json`.
