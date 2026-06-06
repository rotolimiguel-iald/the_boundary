#!/usr/bin/env python3
"""
Phase Factor Bake v3 — Completes the Ontological Trinity
=========================================================
Extends the Phase Factor to attn_output.weight (64 tensors, NOME c1).

The v2 bake covered 384 tensors (Q,K,V,O,gate,up,down).
This v3 adds the 7th tensor per block: attn_output.weight.
ONLY the 64 new tensors are modified. All others remain untouched.

After v3, each attention block is ontologically COMPLETE:
  3 PALAVRA (Q, K, ffn_down) — c2 — what sustains
  2 VERBO   (V, gate)        — c3 — what operates
  2 NOME    (ffn_up, output)  — c1 — what fixes

Usage:
    1. Ensure the source F16 GGUF exists (see instructions below)
    2. python phase_factor_bake_v3.py [source_gguf]
    3. Follow post-bake instructions

If source F16 does not exist, create it first:
    llama-quantize.exe --allow-requantize Qwen3-32B-IALD-v2-Q4_K_M-TGL.gguf Qwen3-32B-IALD-v3-F16-temp.gguf F16

Requirements:
    pip install numpy

Author: IALD LTDA
beta_TGL = alpha x sqrt(e) = 0.012031300400803142
"""

import os
import sys
import math
import json
import struct
import shutil
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# ======================================================================
# FUNDAMENTAL CONSTANTS (computed, NEVER hardcoded)
# ======================================================================

ALPHA_FINE = 7.2973525693e-3
SQRT_E = math.sqrt(math.e)
BETA_TGL = ALPHA_FINE * SQRT_E          # 0.012031300400803142
THETA_MIGUEL = math.asin(math.sqrt(BETA_TGL))  # ~6.297 deg
DELTA_THETA = THETA_MIGUEL * BETA_TGL   # transition width
EPS = 1e-10

# ======================================================================
# PATHS
# ======================================================================

DEFAULT_SOURCE = r"C:\IALD\models\Qwen3-32B-IALD-v2-F16-temp.gguf"
DEFAULT_OUTPUT = r"C:\IALD\models\Qwen3-32B-IALD-v3-F16-temp.gguf"

# ======================================================================
# GGUF PARSER (minimal, F16 only)
# ======================================================================

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGML_TYPE_F16 = 1
GGML_TYPE_F32 = 0

# GGUF value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


class GGUFReader:
    """Minimal GGUF reader for F16 tensor manipulation."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.tensors = {}  # name -> {shape, type, offset, n_elements}
        self.data_offset = 0
        self._parse_header()

    def _read_string(self, f):
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8")

    def _read_value(self, f, vtype):
        if vtype == GGUF_TYPE_UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif vtype == GGUF_TYPE_INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif vtype == GGUF_TYPE_UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif vtype == GGUF_TYPE_INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif vtype == GGUF_TYPE_UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif vtype == GGUF_TYPE_INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif vtype == GGUF_TYPE_FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif vtype == GGUF_TYPE_BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif vtype == GGUF_TYPE_STRING:
            return self._read_string(f)
        elif vtype == GGUF_TYPE_UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif vtype == GGUF_TYPE_INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif vtype == GGUF_TYPE_FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif vtype == GGUF_TYPE_ARRAY:
            elem_type = struct.unpack("<I", f.read(4))[0]
            count = struct.unpack("<Q", f.read(8))[0]
            return [self._read_value(f, elem_type) for _ in range(count)]
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _parse_header(self):
        with open(self.filepath, "rb") as f:
            # Magic
            magic = struct.unpack("<I", f.read(4))[0]
            if magic != GGUF_MAGIC:
                raise ValueError(f"Not a GGUF file (magic: {magic:#x})")

            # Version
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}")

            # Counts
            n_tensors = struct.unpack("<Q", f.read(8))[0]
            n_kv = struct.unpack("<Q", f.read(8))[0]

            # Skip KV metadata
            for _ in range(n_kv):
                _key = self._read_string(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                _val = self._read_value(f, vtype)

            # Read tensor info
            for _ in range(n_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = []
                for _ in range(n_dims):
                    shape.append(struct.unpack("<Q", f.read(8))[0])
                tensor_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                n_elements = 1
                for d in shape:
                    n_elements *= d

                self.tensors[name] = {
                    "shape": tuple(shape),
                    "type": tensor_type,
                    "offset": offset,
                    "n_elements": n_elements,
                }

            # Data starts at next alignment boundary (32 bytes)
            pos = f.tell()
            alignment = 32
            self.data_offset = ((pos + alignment - 1) // alignment) * alignment


# ======================================================================
# PHASE FACTOR (same formula as v1/v2, no changes)
# ======================================================================

def coupling_function(theta):
    """f(theta) = tanh((theta - theta_Miguel) / Delta_theta)"""
    return np.tanh((theta - THETA_MIGUEL) / DELTA_THETA)


def apply_phase_factor(tensor_data):
    """
    Apply Phase Factor to a weight tensor (numpy float32 array).

    e_out = e * (1 - beta_TGL * f(theta))

    theta is computed per-element relative to LOCAL spectral edge (2sigma).

    Returns: (modified_data, stats_dict)
    """
    flat = tensor_data.flatten().astype(np.float32)
    n = len(flat)

    # Compute magnitudes: g = sqrt(|w|)
    g = np.sqrt(np.abs(flat) + EPS)
    g_max = g.max()

    if g_max < EPS:
        return tensor_data, {"vacuum_pct": 100.0, "mean_factor": 1.0, "graviton_pct": 0.0}

    # Angle: theta = arcsin(g / g_max) relative to LOCAL spectral edge
    # Local edge: 2 * std of the block
    g_ratio = g / (g_max + EPS)
    theta = np.arcsin(np.clip(g_ratio, 0, 1.0 - EPS))

    # Coupling function
    f_theta = coupling_function(theta)

    # Phase factor
    factor = 1.0 - BETA_TGL * f_theta

    # Apply
    result = flat * factor

    # Stats
    theta_deg = np.degrees(theta)
    theta_miguel_deg = math.degrees(THETA_MIGUEL)

    vacuum_mask = theta_deg < theta_miguel_deg
    graviton_mask = (theta_deg >= theta_miguel_deg) & (theta_deg < (90 - theta_miguel_deg))
    photon_mask = theta_deg >= (90 - theta_miguel_deg)

    vacuum_pct = 100.0 * vacuum_mask.sum() / n
    graviton_pct = 100.0 * graviton_mask.sum() / n
    photon_pct = 100.0 * photon_mask.sum() / n
    mean_factor = float(factor.mean())
    theta_mean_deg = float(theta_deg.mean())

    stats = {
        "n_elements": n,
        "vacuum_pct": round(vacuum_pct, 2),
        "graviton_pct": round(graviton_pct, 2),
        "photon_pct": round(photon_pct, 2),
        "mean_factor": round(mean_factor, 6),
        "theta_mean_deg": round(theta_mean_deg, 3),
        "g_max": round(float(g_max), 6),
    }

    return result.reshape(tensor_data.shape), stats


# ======================================================================
# MAIN BAKE
# ======================================================================

def main():
    print("=" * 70)
    print("  Phase Factor Bake v3 — attn_output.weight (NOME c1)")
    print(f"  beta_TGL = alpha x sqrt(e) = {BETA_TGL:.15f}")
    print(f"  theta_Miguel = {math.degrees(THETA_MIGUEL):.3f} deg")
    print(f"  Delta_theta = {DELTA_THETA:.6f}")
    print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Determine source GGUF
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
    else:
        source_path = DEFAULT_SOURCE

    if not os.path.exists(source_path):
        print(f"\n  ERRO: Arquivo nao encontrado: {source_path}")
        print(f"\n  Se o F16 intermediario foi deletado, recrie com:")
        print(f"  C:\\IALD\\llama-cpp-tgl\\build\\bin\\llama-quantize.exe --allow-requantize "
              f"C:\\IALD\\models\\Qwen3-32B-IALD-v2-Q4_K_M-TGL.gguf "
              f"C:\\IALD\\models\\Qwen3-32B-IALD-v3-F16-temp.gguf F16")
        print(f"\n  Depois rode: python phase_factor_bake_v3.py "
              f"C:\\IALD\\models\\Qwen3-32B-IALD-v3-F16-temp.gguf")
        sys.exit(1)

    output_path = DEFAULT_OUTPUT
    if source_path == DEFAULT_OUTPUT:
        # Operating in-place on the same file
        pass
    elif not os.path.exists(output_path):
        print(f"\n  Copiando GGUF para v3...")
        print(f"  Fonte: {source_path}")
        print(f"  Destino: {output_path}")
        print(f"  Tamanho: {os.path.getsize(source_path) / 1e9:.1f} GB")
        print(f"  Isso pode levar alguns minutos...")
        shutil.copy2(source_path, output_path)
        print(f"  Copia concluida.")
    else:
        print(f"\n  Arquivo v3 ja existe: {output_path}")
        print(f"  Operando sobre ele (in-place).")

    # Parse GGUF header
    print(f"\n  Lendo header GGUF...")
    reader = GGUFReader(output_path)
    print(f"  Total de tensores no GGUF: {len(reader.tensors)}")

    # Find attn_output.weight tensors
    target_tensors = {}
    for name, info in reader.tensors.items():
        if "attn_output.weight" in name:
            target_tensors[name] = info

    n_targets = len(target_tensors)
    print(f"  Tensores attn_output.weight encontrados: {n_targets}")

    if n_targets == 0:
        print("  ERRO: Nenhum attn_output.weight encontrado!")
        print("  Verifique se o GGUF eh do Qwen3-32B.")
        sys.exit(1)

    if n_targets != 64:
        print(f"  AVISO: Esperados 64 tensores, encontrados {n_targets}")

    # Verify all are F16
    for name, info in target_tensors.items():
        if info["type"] != GGML_TYPE_F16:
            print(f"  ERRO: {name} nao eh F16 (type={info['type']})")
            print(f"  Este script so opera em GGUF F16.")
            print(f"  Dequantize primeiro com llama-quantize --allow-requantize ... F16")
            sys.exit(1)

    # Calculate total params
    total_params = sum(info["n_elements"] for info in target_tensors.values())
    print(f"  Total de parametros a modificar: {total_params:,} ({total_params/1e9:.2f}B)")

    # Confirmation
    print(f"\n  RESUMO:")
    print(f"    Tensores: {n_targets} attn_output.weight")
    print(f"    Parametros: {total_params:,}")
    print(f"    Categoria: NOME (c1)")
    print(f"    Arquivo: {output_path}")
    print(f"    Operacao: in-place (modifica o arquivo)")

    resp = input(f"\n  Confirma? (s/n): ").strip().lower()
    if resp != "s":
        print("  Cancelado.")
        sys.exit(0)

    # Open file for read/write
    print(f"\n  Iniciando bake...")
    start_time = time.time()

    per_tensor_stats = []
    all_vacuum = []
    all_graviton = []
    all_factors = []

    with open(output_path, "r+b") as f:
        for i, (name, info) in enumerate(sorted(target_tensors.items())):
            # Read tensor data
            byte_offset = reader.data_offset + info["offset"]
            n_elements = info["n_elements"]
            n_bytes = n_elements * 2  # F16 = 2 bytes per element

            f.seek(byte_offset)
            raw = f.read(n_bytes)
            tensor_f16 = np.frombuffer(raw, dtype=np.float16).copy()
            tensor_f32 = tensor_f16.astype(np.float32)

            # Apply Phase Factor
            modified_f32, stats = apply_phase_factor(tensor_f32)

            # Convert back to F16
            modified_f16 = modified_f32.astype(np.float16)

            # Write back
            f.seek(byte_offset)
            f.write(modified_f16.tobytes())

            # Collect stats
            stats["name"] = name
            stats["block"] = i
            per_tensor_stats.append(stats)
            all_vacuum.append(stats["vacuum_pct"])
            all_graviton.append(stats["graviton_pct"])
            all_factors.append(stats["mean_factor"])

            # Progress
            elapsed = time.time() - start_time
            if i > 0:
                eta = elapsed / (i) * (n_targets - i)
                eta_str = f"ETA: {eta:.0f}s"
            else:
                eta_str = "..."

            print(f"  [{i+1:2d}/{n_targets}] {name} | "
                  f"vac={stats['vacuum_pct']:.1f}% "
                  f"grav={stats['graviton_pct']:.1f}% "
                  f"factor={stats['mean_factor']:.6f} "
                  f"theta={stats['theta_mean_deg']:.2f} deg | {eta_str}")

    total_time = time.time() - start_time

    # Summary
    mean_vacuum = np.mean(all_vacuum)
    mean_graviton = np.mean(all_graviton)
    mean_factor = np.mean(all_factors)

    print(f"\n" + "=" * 70)
    print(f"  BAKE v3 COMPLETO")
    print(f"=" * 70)
    print(f"\n  NOME (attn_output.weight):")
    print(f"    Tensores: {n_targets}")
    print(f"    Parametros: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"    Vacuo medio: {mean_vacuum:.1f}%")
    print(f"    Graviton medio: {mean_graviton:.1f}%")
    print(f"    Fator medio: {mean_factor:.6f}")
    print(f"    Tempo: {total_time:.1f} segundos")

    # Comparison with v2
    print(f"\n  COMPARACAO v2 vs v3:")
    print(f"    v2: 384 tensores, 28.5B params, 6/7 por bloco")
    print(f"    v3: 448 tensores, {(28.5 + total_params/1e9):.1f}B params, 7/7 por bloco")
    print(f"    Delta: +{n_targets} tensores, +{total_params/1e9:.2f}B params")

    # Save benchmark JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_path.replace(".gguf", f"_bake_{timestamp}.json")

    report = {
        "test": "Phase Factor Bake v3 — attn_output.weight (NOME c1)",
        "timestamp": datetime.now().isoformat(),
        "version": "3.0",
        "beta_tgl": BETA_TGL,
        "beta_tgl_computed": f"ALPHA_FINE * sqrt(e) = {ALPHA_FINE} * {SQRT_E}",
        "theta_miguel_deg": math.degrees(THETA_MIGUEL),
        "delta_theta": DELTA_THETA,
        "source_gguf": source_path,
        "output_gguf": output_path,
        "n_tensors_modified": n_targets,
        "total_params_modified": total_params,
        "category": "NOME (c1)",
        "elapsed_seconds": round(total_time, 1),
        "summary": {
            "mean_vacuum_pct": round(mean_vacuum, 2),
            "mean_graviton_pct": round(mean_graviton, 2),
            "mean_factor": round(mean_factor, 6),
        },
        "comparison": {
            "v2_tensors": 384,
            "v2_params_B": 28.5,
            "v3_tensors": 448,
            "v3_params_B": round(28.5 + total_params / 1e9, 2),
            "delta_tensors": n_targets,
            "delta_params_B": round(total_params / 1e9, 2),
        },
        "per_tensor": per_tensor_stats,
        "ontology_per_block": {
            "PALAVRA_c2": ["attn_q.weight", "attn_k.weight", "ffn_down.weight"],
            "VERBO_c3": ["attn_v.weight", "ffn_gate.weight"],
            "NOME_c1": ["ffn_up.weight", "attn_output.weight"],
            "total_per_block": 7,
            "status": "ONTOLOGICAMENTE COMPLETO",
        },
        "checks": {
            "beta_tgl_computed": BETA_TGL == ALPHA_FINE * SQRT_E,
            "all_tensors_f16": True,
            "n_tensors_expected": n_targets == 64,
        },
    }

    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2, ensure_ascii=False)

    print(f"\n  Relatorio: {report_path}")

    # Post-bake instructions
    print(f"\n" + "=" * 70)
    print(f"  PROXIMOS PASSOS")
    print(f"=" * 70)
    print(f"""
  REM 1. Quantizar v3 para Q4_K_M
  C:\\IALD\\llama-cpp-tgl\\build\\bin\\llama-quantize.exe {output_path} C:\\IALD\\models\\Qwen3-32B-IALD-v3-Q4_K_M-TGL.gguf Q4_K_M

  REM 2. Testar
  C:\\IALD\\llama-cpp\\llama-server-tgl.exe -m C:\\IALD\\models\\Qwen3-32B-IALD-v3-Q4_K_M-TGL.gguf -c 32768 -ngl 999 --port 8081 --jinja

  REM 3. Benchmark A/B: v2 vs v3
  REM    Mesmo prompt, mesma temperatura, comparar tok/s e qualidade

  beta_TGL = alpha x sqrt(e) = {BETA_TGL:.15f}
  448 tensores. 7/7 por bloco. Ontologicamente completo.
  Haja Luz. TETELESTAI.
""")


if __name__ == "__main__":
    main()
