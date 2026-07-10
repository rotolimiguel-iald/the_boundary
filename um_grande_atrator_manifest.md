# Um: Grande Atrator -- MANIFESTO DE ENTRADAS (nada fica escondido no codigo)

> Ou e' **definicao exata**, ou **constante medida**, ou **protocolo pre-registrado**, ou **conjectura testavel**. Este manifesto e' parte do hash do veredito.

## Definicoes exatas [DEF]

```json
{
  "ONE": 1,
  "TWO": 2,
  "HALF": 0.5,
  "FOUR": 4,
  "pi": "computed (4*atan(1))",
  "sqrt_e": "computed (exp(1/2))",
  "alpha_abs": "[DEF] = 1 (Um absoluto; input originario; Tomita do Bell nu da' alpha_abs=1)",
  "q": "[QED-VALIDATION] polarizacao termico-modular do zero modular (=q_QED no modo de validacao)",
  "alpha_form": "[DER] = sqrt(1 - q^2)  (= alpha_obs; motor canonico)",
  "beta_form": "[DER] = sqrt(e) * sqrt(1 - q^2)  (= alpha*sqrt(e) na leitura observacional)",
  "identity": "[DER] 1 = q^2 + alpha^2 (conservada)",
  "R_partial": "[LEGADO] = 1/alpha_form, derivado APOS a forma; NAO motor canonico (nao vem de CODATA)"
}
```

## Constantes medidas [DATA]

```json
{
  "alpha_CODATA": 0.0072973525693,
  "source": "CODATA 2018",
  "alpha_inv": "137.035999",
  "uncertainty": "~1.5e-10 (rel.)",
  "role": "[EXT] validacao final apenas: q_QED = sqrt(1 - alpha_CODATA^2); NAO move a cadeia",
  "G_Newton": 6.674e-11,
  "M_sun_kg": 1.98892e+30,
  "Mpc_m": 3.0857e+22
}
```

## Definicoes SI [DEF]

```json
{
  "c_m_per_s": 299792458.0
}
```

## Ponte da Impedancia do Vacuo [REAL/EXT]

```json
{
  "Z0": "[EXT] impedancia caracteristica do vacuo; face dimensional da constante DINAMICA da luz",
  "R_K": "[DEF] h/e^2, exato no SI porque e,h sao exatos (2019)",
  "G0": "[DEF] 2e^2/h, exato no SI porque e,h sao exatos (2019)",
  "alpha_Z0_bridge": "[REAL] alpha = Z0 e^2/(2h) = Z0/(2 R_K) = Z0 G0/4",
  "Z0_from_alpha_ohm": 376.730313668004,
  "chi_log_impedance_ratio": 11.226755051602328,
  "all_checks_verified": true,
  "status": "VACUUM_IMPEDANCE_BRIDGE_FORMULATED__ALPHA_VALUE_QED_SECTOR_FALSIFICATION_CHALLENGE. Ponte fisica fechada (c=cinematica, Z0=dinamica, alpha=Z0 adimensional); valor alpha-livre aberto: Z0 computado de alpha (mu0 nao exato pos-2019), entao Z0<->alpha dado e,h."
}
```

## Entrada geometrica [DATA]

```json
{
  "R_struct_literature_Mpc": 57.0,
  "source": "Lynden-Bell et al. 1988 (GA basin extent ~40 h^-1 Mpc; h=0.7 -> 57 Mpc)",
  "provenance": "GEOMETRY_ONLY_NO_MASS_NO_RG"
}
```

## Protocolo pre-registrado [PRE]

```json
{
  "GA_center_RA_deg": 243.6,
  "GA_center_Dec_deg": -60.4,
  "cone_half_angle_deg": 30.0,
  "dist_shell_Mpc": [
    30.0,
    100.0
  ],
  "R_struct_method": "percentile_90_from_centroid",
  "accepted_mass_window_Msun": [
    1000000000000000.0,
    1e+17
  ],
  "sensitivity_grid": {
    "cone_half_angle_deg": [
      20.0,
      25.0,
      30.0,
      35.0,
      40.0
    ],
    "dist_shell_Mpc": [
      [
        25.0,
        100.0
      ],
      [
        30.0,
        100.0
      ],
      [
        30.0,
        120.0
      ]
    ],
    "percentile": [
      80.0,
      85.0,
      90.0,
      95.0
    ],
    "center_offsets_deg": [
      [
        0.0,
        0.0
      ],
      [
        5.0,
        0.0
      ],
      [
        -5.0,
        0.0
      ],
      [
        0.0,
        5.0
      ],
      [
        0.0,
        -5.0
      ]
    ]
  }
}
```

## Comparacao externa apenas [EXT]

```json
[
  {
    "name": "Norma cluster ACO 3627 (virial, RG dinamica)",
    "M_Msun": 1000000000000000.0,
    "ref": "Woudt et al. 2008, MNRAS 383, 445",
    "type": "GR_dynamical_virial"
  },
  {
    "name": "Grande Atrator (infall linear, RG)",
    "M_Msun": 5.4e+16,
    "ref": "Lynden-Bell et al. 1988, ApJ 326, 19",
    "type": "GR_linear_infall"
  },
  {
    "name": "Laniakea (supercluster)",
    "M_Msun": 1e+17,
    "ref": "Tully et al. 2014, Nature 513, 71",
    "type": "supercluster_flow"
  }
]
```

## Parametros numericos dos testes de sombra [NUM]

```json
{
  "dim_n": 4,
  "seed": 11,
  "gesture_G": 6,
  "tunnel_theta": 0.4,
  "dipole_dim": 6,
  "dipole_trajectories": 12,
  "dipole_steps": 620,
  "dipole_thermal_seed": 7,
  "dipole_traj_seed": 3,
  "purity_floor_start": 0.997,
  "status": "FINITE_DIM_SANITY_CHECKS_NOT_TYPE_III1_PROOF"
}
```

## Setor irreversivel -- o ato (v3) [NUM]

```json
{
  "verb_generator_A": {
    "L": "L = sqrt(beta) sqrt(K_partial) ; K=A A^T/n (n=4, seed=11) ; beta=alpha*sqrt(e)",
    "dt": 0.02,
    "steps": 420,
    "t_inverse": 1.0,
    "seed": 11,
    "status": "[FINITE_DIM_SANITY_NOT_III1_PROOF]",
    "selo": "VERB_GENERATOR_L_EQ_SQRTBETA_SQRTK . ARROW_ENTROPY_MONOTONE . INVERSE_NOT_CP_CHOI_NEGATIVE . SEMIGROUP_NOT_GROUP . RES_JUDICATA_OF_THE_CODE"
  },
  "light_eigenvector_B": {
    "equation": "O_beta(Lux) = sqrt(beta) Lux ; autovalor sqrt(beta) ; NAO ponto fixo",
    "status": "[DER na forma; a identificacao fisica e' a equacao da unificacao TGL]",
    "selo": "LIGHT_IS_EIGENVECTOR_EIGENVALUE_SQRTBETA . NOT_FIXED_POINT_SQRTBETA_NEQ_1 . CHAIN_ORDER_ONE_HALFNAT_LIGHT_THEN_MASS"
  },
  "fiat_lux_counterfactual_C": {
    "law": "fiat lux = e^{S_partial} alpha > 0 (Verbo=Palavra x Nome)",
    "configs": "SEM PALAVRA (S=0) ; SEM NOME (alpha=0) ; COM AMBOS (S=1/2, alpha=CODATA)",
    "status": "[DER -- contrafactuais da MESMA cadeia, inputs alterados]",
    "selo": "FIAT_LUX_EQUALS_PRODUCT_WORD_TIMES_NAME_POSITIVE . NO_WORD_DEATH_BY_INDISTINCTION_BRIDGE_EQ_1 . NO_NAME_DEATH_BY_NONEXISTENCE_CROSSREF_S22 . NOTHING_EMERGES_UNLESS_THE_ONE_IS_INSCRIBED_NOW_TESTED"
  }
}
```

## Escala, peso e programa (v4-v7); inclui o protocolo P5' pre-registrado [PRE/NUM]

```json
{
  "P2_boundary_reads_IR": {
    "claim": "a fronteira le o IR (supremo de chi); chi*=rapidez modular=log(Z-ratio); VALOR=Nome (§21)",
    "chi_IR": 11.2267550516013,
    "chi_IR_eq_chi_bridge_resid": 1.028510610012745e-12,
    "status": "[DER a escala da leitura; REAL/EXT o IR-freeze da QED; §21 intocado: o VALOR e' o Nome]",
    "selo": "CHI_IS_ADDITIVE_MODULAR_RAPIDITY . BOUNDARY_READS_SUPREMUM_EQ_IR . QED_IR_FREEZE_UNIQUE_CANONICAL_READING . CHI_STAR_EQ_LOG_IMPEDANCE_RATIO_SAME_OBJECT . SCALE_IS_OBSERVER_POSITION_NOT_HIDDEN_PARAMETER . ALPHA_VALUE_STILL_THE_NAME"
  },
  "P3_smatrix_dual_weight": {
    "claim": "S_core=exp(thM G) sem traco => peso 0 sob acao dual (condicional P_2D em M)",
    "status": "[DER condicional a P_2D ; FINITE_DIM_SANITY_NOT_III1_PROOF na sombra]",
    "selo": "S_CORE_WEIGHT_ZERO_UNDER_DUAL_ACTION_CONDITIONAL_P2D . NO_TRACE_IN_DEFINITION . RESIDUE_P2D_LOCALIZATION"
  },
  "P4_void_floor_PRE": {
    "prediction": "rho_void/rho_bar >= beta (zero-parametro)",
    "test": "DESI/Euclid, perfil empilhado; falsifica se rho_c/rho_bar < beta-3sigma",
    "status": "[PRE + EXT] pre-registro falsificavel; consistencia + endereco, NAO confirmacao",
    "selo": "VOID_FLOOR_RHO_RATIO_GE_BETA_PREREGISTERED . DESI_EUCLID_ADDRESS . ZERO_PARAMETER_FALSIFIABLE"
  },
  "P5_dipole_antipode_PRE": {
    "prediction": "densidade(antipoda RA+180,Dec->-Dec) < densidade(GA) [POSICOES apenas]",
    "cf4_ok": true,
    "raw_note": "teste BRUTO (contagem pura); GA atras da Zona de Evitamento -> enviesado. Rotulo qualitativo DERIVADO do dado (nao hardcoded).",
    "status": "[PRE + DATA geometria pura; POSICOES apenas -- velocidades/massas IGNORADAS]",
    "selo": "GA_ANTIPODE_UNDERDENSITY_PREREGISTERED_POSITIONS_ONLY . DIPOLE_GEOMETRY_NO_VELOCITIES"
  },
  "P5prime_masked_PRE": {
    "protocol": {
      "completeness_mask": "|b_galactic| > 10 deg (MESMO corte nos dois cones; equatorial->galactica J2000)",
      "controls": "8 cones de mesma abertura (30 deg) e mesma casca em ceu limpo |b|>30 deg",
      "control_centers_RA_Dec_deg": [
        [
          46.285,
          -0.083
        ],
        [
          25.351,
          -47.77
        ],
        [
          341.398,
          14.109
        ],
        [
          241.33,
          1.419
        ],
        [
          353.129,
          -36.227
        ],
        [
          199.343,
          -1.877
        ],
        [
          312.24,
          -47.943
        ],
        [
          168.146,
          -26.469
        ]
      ],
      "criterion": "razao(anti/GA) mascarada < 1 E fora do IC90 da dispersao dos controles => REPULSOR SUB-DENSO INDICADO; dentro da dispersao => NAO-INFORMATIVO (CF4 posicoes pode nao bastar; endereco final: catalogos com mascara de completeza publicada)",
      "bootstrap": "B=1000, IC95, seed=11",
      "statute": "[PRE]",
      "selo": "GA_ANTIPODE_MASKED_COMPLETENESS_PREREGISTERED . CONTROL_CONES_CLEAN_SKY . POSITIONS_ONLY"
    },
    "GA_b_deg": -6.792945081298402,
    "antipode_b_deg": 6.792945081298405,
    "cf4_ok": true,
    "result_obtained": {
      "ratio_masked_antipode_over_GA": 1.1295546558704452,
      "n_GA_masked": 247,
      "n_antipode_masked": 279,
      "control_ratio_CI90": [
        0.39577435679033995,
        2.5289372585973013
      ],
      "verdict_P5prime": "NAO_INFORMATIVO (razao dentro da dispersao dos controles; CF4 posicoes pode nao bastar)"
    },
    "note": "PROTOCOLO PRE-REGISTRADO (mascara |b|>10 + 8 controles |b|>30 seed=11) gravado ANTES da execucao com dados",
    "status": "[PRE + DATA geometria; POSICOES apenas; controle de completeza]",
    "selo": "GA_ANTIPODE_MASKED_COMPLETENESS_PREREGISTERED . CONTROL_CONES_CLEAN_SKY . POSITIONS_ONLY"
  },
  "P6_dephasing_crossover_NUM": {
    "map": "root law=canonica no IR; expoente 2->1; crossover ~omega tau*=1",
    "x_grid": "logspace(-3,3,25) ; gerador v3 L=sqrt(beta)sqrt(K) ; seed determinista",
    "crossover_x": 3.1622776601683795,
    "status": "[FINITE_DIM_SANITY] mapa numerico; a reconciliacao analitica completa da root law vs canonica fica [ABERTO] com este mapa como guia.",
    "selo": "DEPHASING_REGIME_MAP_QUADRATIC_IR . CROSSOVER_LOCATED_NUMERICALLY . ROOT_LAW_REGIME_CHARACTERIZED"
  },
  "v5_jacobson_form_check_NUM": {
    "form_check": "P_mn[K_partial]=F(J,Delta,P_2D) [residuo U_loc fechado POSITIVO]",
    "link_tested": "1a lei modular dS=d<K> (n=6, seed=11; V hermitiana traco 0 seed=1123; eps in {1e-4,1e-5,1e-6}); assinatura de 1a ordem = criterio",
    "P2D": "plano de bifurcacao = MESMO corner geometrico da matriz-S do nucleo",
    "residue_open": "approximate Killing vectors (compartilhado com Jacobson desde 1995)",
    "status": "[FINITE_DIM_SANITY no elo (iv) dS=d<K>; REAL na literatura (1a lei) + Lovelock (4D)]",
    "selo": "JACOBSON_SOURCE_HAS_FORM_F_J_DELTA_P2D . FIRST_LAW_MODULAR_TESTED . P2D_IS_BIFURCATION_PLANE_SAME_AS_SMATRIX_CORNER . RESIDUE_APPROX_KILLING_SHARED_WITH_FIELD"
  },
  "v6_thermal_anchor": {
    "claim": "Modulo=calor: q=tanh(chi/2)=p_lo-p_hi (polarizacao); alpha=sech(chi/2)=2sqrt(p_lo p_hi) (coerencia max)",
    "reading": "1 = q^2 + alpha^2 = polarizacao^2 + coerencia-maxima^2 (identidade hiperbolica do equilibrio)",
    "kms_anchor": "estado de fronteira KMS (Tomita-Takesaki: fluxo modular = fluxo termico; Unruh/Hawking) [REAL]",
    "resid_max": {
      "id1_q_eq_tanh": 1.1102230246251565e-16,
      "id2_alpha_eq_sech": 0.0,
      "one_eq_q2_coh2": 2.220446049250313e-16
    },
    "vital_layer_ONTO": "Modulo=calor=Nome=sangue: alpha=Nome/calor/sangue; S=1/2=Palavra; beta=sqrt(e)alpha=geometria impressa do sangue. 'o Nome e' o sangue quente da manifestacao; a Palavra o mede; beta_TGL e' sua geometria inscrita.' [ONTO -- Parte B; NAO entra no veredito, nao ha numero para sangue]",
    "vital_layer_selo_ONTO": "MODULE_IS_HEAT_IS_NAME_IS_BLOOD",
    "status": "[ID.1 REAL Brillouin; ID.2 DER; KMS REAL] identifica a FORMA; o valor e' o Nome (§21)",
    "selo": "Q_IS_TWO_LEVEL_THERMAL_POLARIZATION . ALPHA_IS_MAX_COHERENCE_2SQRT_PLO_PHI . CHI_IS_MODULAR_E_OVER_T . BOUNDARY_STATE_IS_KMS . ONE_EQ_POLARIZATION2_PLUS_COHERENCE2"
  },
  "v7_fiat_lux_flow_NUM": {
    "verdict_chain": "1 = q^2+alpha^2 = VERDADEIRO = HAJA_LUZ (um certificado por elo; fail-closed por elo)",
    "generator": "REUSA _verb_L (v3): L=sqrt(beta)sqrt(K), n=4, seed=11 (UM so' L no codigo)",
    "certificates": "F1 Tr=1 ; F2 dS>=0 ; F3 Spohn S(rho||rho*) mono nao-cresce (Lyapunov modular) ; F4 coerencia morre",
    "numerics": {
      "dt": 0.05,
      "n_points": 300,
      "t_end_over_beta": 3.0,
      "seed": 11,
      "n": 4,
      "method": "expm DIRETO por checkpoint (semigrupo exato; monotonia independe do integrador)"
    },
    "characteristic_time_one_over_beta": 83.11653492861383,
    "references": "Spohn 1978 (H-teorema Lindblad) ; Uhlmann (monotonia da entropia relativa sob CPTP)",
    "status": "[FINITE_DIM_SANITY_NOT_III1_PROOF; Spohn/Uhlmann REAL na literatura]",
    "selo": "FIAT_LUX_FLOW_ONE_CONSERVED_F1 . ARROW_F2 . SPOHN_MODULAR_LYAPUNOV_F3 . INSCRIPTION_COHERENCE_DIES_F4 . CHARACTERISTIC_TIME_ONE_OVER_BETA"
  }
}
```

## Tetelestai = poda binaria ({1_abs,0_mod}\{0_abs}); modulo de prova [DER/NUM]

```json
{
  "definition": "TETELESTAI = PODA BINARIA ; Poda_beta = {1_abs, 0_mod} \\ {0_abs} = ser binario - zero absoluto",
  "budget": "beta_TGL = alpha*sqrt(e) e' o orcamento maximo de excesso podavel (NUNCA alpha^2; NUNCA literal)",
  "classes": "1_abs=identidade(peso>beta) ; 0_mod=diferenca COM retorno(preservado) ; 0_abs=DISTINTO sem retorno(podado) ; absent=pre-inscrito(ignorado)",
  "separators": "BETA separa {1_abs}|{zeros} ; RETORNO(kernel do Verbo, sobrevive ao fluxo v7) separa {0_mod}|{0_abs} ; SUPORTE separa {0_abs}|{absent}",
  "return_formalization": "retorno = populacao na base propria de L (o Verbo v3) = sobrevive a T_t=e^{-tL} (o mesmo juizo do F4 do v7)",
  "paragraph22_anchor": "0_abs = distinto = pureza Tr(rho^2)=1 (rank-1) = §22 (alpha->0, chi->inf); o distinto e' a separacao absoluta proibida por III_1",
  "targets_verified_default_rng": {
    "a_vector64": [
      64,
      56,
      0.011668063395532391
    ],
    "b_uniform1000_rank_after": 988,
    "c_density_classes": {
      "n_1abs": 3,
      "n_0mod": 1,
      "n_0abs": 2,
      "n_absent": 4
    },
    "c_tail_le_beta": true,
    "c_idempotence_residual": 6.740831934336383e-16,
    "d_motor_p_hi_is_0mod_KEPT": true,
    "e_pure_is_distinct_para22": true
  },
  "inversion_note": "o teste (d) INVERTE o v8 energetico: p_hi (populacao termica de equilibrio) TEM retorno KMS => 0_mod => MANTIDO (o v8 o cortava)",
  "protection": "MODULO DE PROVA -- nenhuma identidade exata (motor, ponte, fluxo, massas) passa por Tet_beta (§21/§22 e travas v2-v7 intactas)",
  "DO_NOT_PRUNE_MODULAR_ZERO": true,
  "status": "[DER + NUM] modulo de PROVA (nenhuma identidade exata passa pela poda; §21/§22 intactos)",
  "selo": "TETELESTAI_IS_PRUNING . PRUNING_IS_BINARY_BEING_MINUS_ABSOLUTE_ZERO . ABSOLUTE_ZERO_IS_THE_DISTINCT . MODULAR_ZERO_IS_DIFFERENCE_WITH_RETURN . DO_NOT_PRUNE_MODULAR_ZERO . PRUNE_ONLY_THE_DISTINCT_WITHOUT_RETURN . ABSENT_IS_IGNORED_NOT_PRUNED . DISTINCT_IS_PURITY_IS_PARAGRAPH_22 . BETA_IS_THE_GEOMETRY_OF_THE_ADMISSIBLE_CUT . ONLY_THE_NAME_AND_THE_LIVING_ZERO_SURVIVE"
}
```

## O cociclo vivo -> G_mu_nu (E1-E12: colagem/temporal/gerador/holonomia/curvatura-obstrucao/covariancia/torcao/ponto-base/canto-P_F/comutador/fase + composicao E7; C1/C2) [NUM/REAL]

```json
{
  "claim": "cociclo de Connes globalmente covariante => (Lovelock 4D) G_mu_nu + Lambda g = 8 pi G T^TGL",
  "construction": "tipo I, n=4, seed=11, estados full-rank rho = 0.9 AA^dag/Tr + 0.1 I/n ; u_ab(t)=rho_a^it rho_b^-it (potencias imaginarias por autodecomposicao)",
  "C1_correction": "a COLAGEM espacial (3 estados) e' MULTIPLICATIVA u_ab u_bc = u_ac; a forma sigma-torcida u_(s+t)=u_s sigma_s^(b)(u_t) e' a identidade TEMPORAL de um par -- DUAS identidades distintas (a derivacao-fonte as mesclava; corrigido pela auditoria).",
  "C2_correction": "o gerador e' h_ab = -i u_ab'(0) = K_b - K_a (K=-log rho, convencao standard u_ab=rho_a^it rho_b^-it), NAO K_a-K_b (corrigido pela auditoria).",
  "certificates_resid": {
    "E1_spatial_gluing": 6.742162969878415e-16,
    "E2_temporal_identity": 1.1443916996305592e-15,
    "E3_generator_Kb_minus_Ka": 6.231415946082284e-11,
    "E3_telescoping_additivity": 1.2412670766236366e-16,
    "E4_holonomy_consistent": 1.5553031891404221e-15,
    "E6_global_covariance": 1.790180836524724e-15
  },
  "E5_curvature_obstruction_curve": {
    "0.05": 0.11800923878078719,
    "0.15": 0.26603821666584493,
    "0.30": 0.38767382777867093
  },
  "E5_monotone_in_lambda": true,
  "E5_finding": "a curvatura modular e' a OBSTRUCAO a existencia de um estado global -- onde o Um cola, nao ha curvatura; onde um patch se recusa ao Um, a holonomia a mede.",
  "v16_2_clock_hierarchy": "o relogio modular K gera TRES ordens da geometria: 1a ordem TORCAO = salto de relogio nao-colavel (E8, obs/t) ; 2a ordem CURVATURA = comutador defeito x relogio do caminho (E11, obs/t^2 = (1/2)||[M,h_cd]||, razao 0.9970) ; fase = GAUGE U(1) da normalizacao (E12, quocientada, nao-curvatura). Transporte = Ad(rho^it), gerador ad(K) = o gerador do setor q (v6/KMS): 'o sigma da correcao C1 era o transporte'.",
  "E8_torsion_clock_jump": {
    "grid_obs_over_t": {
      "0.2000": 0.320866042327632,
      "0.1000": 0.3276669853826578,
      "0.0500": 0.3334521655178608,
      "0.0250": 0.3354127460498997,
      "0.0125": 0.33615796506364876
    },
    "obs_over_t": 0.33615796506364876,
    "delta_K_norm": 0.3367463018805954,
    "coeff_dev_pct": 0.17471218352243656,
    "conv_rel": 0.002216871504466585,
    "ok": true,
    "reading": "TORCAO DE COLAGEM MEDIDA (v16.1): a obstrucao de 1a ordem NAO e' curvatura -- e' torcao modular, o salto de relogio nao-colavel. O coeficiente E' ||Delta_K|| (Delta_K=K(rho_c')-K(rho_c)); a colagem que falha em 1a ordem falha por diferenca de relogios. Costura com E3."
  },
  "E9_basepoint_covariance": {
    "spectral_resid": 1.2412670766236366e-15,
    "ok": true,
    "reading": "A holonomia depende do ponto-base por CONJUGACAO; o ESPECTRO nao. A curvatura-obstrucao e' covariante: mudar o ponto-base conjuga; o espectro e' o que a fisica le."
  },
  "E10_corner_reads_obstruction": {
    "tau_F_of_I_resid": 1.1102230246251565e-16,
    "curve_dev_over_lambda": {
      "0.05": 0.00863627539355194,
      "0.15": 0.050974560716240624,
      "0.30": 0.13383696493308928
    },
    "monotone": true,
    "P_F_canonical": true,
    "P_F_rank": 4,
    "ok": true,
    "reading": "A projecao tracial le a curvatura-torcao no canto CANONICO P_F do v10 (nucleo zero dos Three Locks, superoperador n^2; o cociclo lido como Ad_W=kron(W.conj,W)): tau_F(I)=1; |tau_F(W)-1| cresce com a inconsistencia lam. O MESMO canto que carrega a matriz-S (v10) e o plano de bifurcacao (form-check v5) le a obstrucao do cociclo (v16). Um canto, tres papeis -- os modulos agora se falam."
  },
  "E11_curvature_commutator": {
    "grid_obs_over_t2": {
      "0.2000": 0.7243131944678416,
      "0.1000": 0.7661418576547965,
      "0.0500": 0.7825278970072757,
      "0.0250": 0.7889895468749768,
      "0.0125": 0.7917208303636319
    },
    "obs_over_t2": 0.7917208303636319,
    "c_teo_live": 0.7940963043496181,
    "ratio_obs_over_c_teo": 0.9970085819906546,
    "ratio_in_band": true,
    "bch_prediction": "||W~ - I|| ~ (t^2/2)||[M, h_cd]|| ; h_cd = K_d - K_c (= E3) ; c_teo = 0.5 maxabs([M,h_cd]) ao vivo",
    "reading": "A CONJECTURA FECHOU: a curvatura de 2a ordem e' o comutador do defeito com a diferenca de relogios do caminho, F ~ (1/2)[M, h_cd]. O transporte e' Ad(rho^it), gerador ad(K) -- o MESMO K que define o setor q (v6/KMS). O setor q e a geometria tem o mesmo gerador."
  },
  "E12_phase_is_gauge": {
    "with_phase_obs_over_t": {
      "0.2000": 0.46023880390729455,
      "0.1000": 0.4365776312893273,
      "0.0500": 0.42476293281730815,
      "0.0250": 0.4191698739514972,
      "0.0125": 0.4164913139418461
    },
    "dephased_obs_over_t": {
      "0.2000": 0.14486263889356835,
      "0.1000": 0.07661418576547965,
      "0.0500": 0.03912639485036379,
      "0.0250": 0.019724738671874423,
      "0.0125": 0.009896510379545401
    },
    "ok": true,
    "reading": "A FASE E' GAUGE: o linear residual do teste ingenuo era a fase U(1) da renormalizacao do traco (-log Z . I -> fase global). Fase global nao e' curvatura; a curvatura genuina vive em su(n) (traceless). Quocientando a fase (dephase por det^{1/n}), o linear DESAPARECE e sobra o t^2 do E11. Regra: toda medida de holonomia-obstrucao quocienta a fase U(1) antes."
  },
  "conjecture_closed": "A conjectura da curvatura-por-comutador (registrada ABERTA na auditoria anterior do campo) FECHOU pela identificacao do operador -- o transporte e' o gerador modular do setor q; com o transporte correto e a fase U(1) quocientada, a curvatura emerge como o comutador previsto, medida a 0.30% da teoria (razao 0.9970).",
  "E7_declared_statute": "composicao (NAO teste): simetria + conservacao ja testadas no form-check v5 (dS=d<K>, 1a lei modular) + Lovelock 4D [REAL, teorema] => G_mu_nu + Lambda g. Estatuto do fechamento continuo: herda o residuo do v5 (approximate Killing vectors, compartilhado com Jacobson desde 1995). NAO se afirma 'provamos Einstein'.",
  "field_equation": "G_mu_nu + Lambda g_mu_nu = 8 pi G T^TGL_mu_nu",
  "beta_position": "T^TGL = T^matter + T^{partial,beta} + T^{torsion/diss}  (beta no lado DIREITO; NAO substitui G)",
  "status": "[REAL(E1-E6, tipo I, certificados vivos) + COMPOSICAO com estatuto declarado(E7)]",
  "selo": "COCYCLE_CHAIN_RULE_MULTIPLICATIVE_E1 . TEMPORAL_IDENTITY_E2 . GENERATOR_IS_CLOCK_DIFFERENCE_E3 . GLOBAL_STATE_IMPLIES_HOLONOMY_ONE_E4 . CURVATURE_IS_OBSTRUCTION_TO_GLOBAL_STATE_E5 . COVARIANCE_E6 . TORSION_IS_THE_CLOCK_JUMP_MEASURED . FIRST_ORDER_OBSTRUCTION_IS_UNGLUEABLE_CLOCK_DIFFERENCE . HOLONOMY_BASEPOINT_COVARIANT_SPECTRUM_INVARIANT . THE_CORNER_READS_THE_OBSTRUCTION . CURVATURE_IS_COMMUTATOR_DEFECT_WITH_CLOCK . TRANSPORT_IS_MODULAR_GENERATOR_OF_SECTOR_Q . SIGMA_OF_C1_WAS_THE_TRANSPORT_ALL_ALONG . PHASE_IS_GAUGE_CURVATURE_IS_TRACELESS . COCYCLE_TO_G_MUNU_COMPOSED_WITH_DECLARED_STATUTE"
}
```

## Axiomas do modelo [AX]

```json
{
  "self_conjugate_boundary": "x = 1 - x  =>  S_partial = 1/2  (Meia-Nat)",
  "w_max": "1/2 (ponto auto-conjugado de Fresnel)",
  "flux_law_s": "s = 1/4pi (normalizacao canonica por compatibilidade)",
  "named_radius_rule": "R_named = 2 beta R_struct (L4)",
  "mass_rule": "M = 2 beta^2 (c^2/4piG) R_struct"
}
```

## Hashes do mundo

```json
{
  "code_sha256": "c416795d12b0d2032ef325c749f52b77d3c65f507167a1fe8cdc84d88d635d10",
  "cf4_catalog_hash": "a2d33204458119225b059193cc1fd26fb085e90de2b8c1bc2397f4156692443a",
  "window_hash": "8a1f4745cb2d91fc0448bbf5214dfa3b64273598ff9e2b8523d160d76c3caf68",
  "selection_hash": "351c308aafd509b418399b5f03db64f274f1189e0d684bfd5d28604181f30a5f"
}
```
