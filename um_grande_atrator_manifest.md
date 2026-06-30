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
  "code_sha256": "9e381c64b9861763ee1c4bbe5aeb64f27917d61863b3e11670d123f666a8537e",
  "cf4_catalog_hash": "a2d33204458119225b059193cc1fd26fb085e90de2b8c1bc2397f4156692443a",
  "window_hash": "8a1f4745cb2d91fc0448bbf5214dfa3b64273598ff9e2b8523d160d76c3caf68",
  "selection_hash": "351c308aafd509b418399b5f03db64f274f1189e0d684bfd5d28604181f30a5f"
}
```
