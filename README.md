# Attention On-Chip Memory Reuse: A Roof-Surface Approach for FPGA Transformer Accelerators

## Motivation

FPGA-based transformer accelerator research is already a mature field (FlightLLM,
HLSTransform, Transformer-OPU, etc.). The differentiator here is building
**an analytical framework that predicts why a given design will hit a bottleneck,
before implementation.**

Intel's DECA (Gerogiannis et al., 2025, "A Near-Core LLM Decompression
Accelerator Grounded on a 3D Roofline Model") models the interaction between
memory, vector-decompression, and matrix-engine resources in a CPU+HBM setting
using a 3D Roof-Surface model, and uses it to derive accelerator parameters
(W, L) at minimal hardware cost.

This project ports the same methodology to **on-chip K,V reuse design for
FPGA attention kernels**. On FPGA, the three resources map as follows:

| DECA (CPU)           | This project (FPGA)                              |
|-----------------------|---------------------------------------------------|
| MEM (HBM bandwidth)   | HBM/DDR bandwidth (loading K, V, Q)               |
| VEC (AVX decompress)  | On-chip elementwise units (softmax/exp, scaling)  |
| MTX (TMUL)             | Systolic PE array (DSP-based MAC)                 |

## Core Hypothesis

How much of attention's K,V is reused on-chip (BRAM/URAM) determines whether
the design is HBM-bound. But scaling on-chip reuse capacity indefinitely wastes
on-chip resources. The goal is to analytically find, **before implementation,
the minimum on-chip capacity at which the bottleneck shifts from HBM-bound to
PE-bound**, and tune the actual HLS design around that point. (This is the same
methodology as Figure 16 in the DECA paper, which finds the optimal
{W=32, L=8} design point.)

## Repository Structure

```
attn_roofline/
  fpga_platforms.py   # Per-board HBM bandwidth, peak DSP throughput, on-chip
                       # capacity (approximate figures, need verification)
  attention_model.py  # Models attention's FLOPs / HBM bytes / on-chip elementwise
                       # ops as a function of on-chip K,V reuse degree (reuse_factor)
run_dse.py             # Example script sweeping on-chip reuse capacity to find
                       # the bottleneck transition point
dse_reuse_vs_throughput.png     # Result plot (reuse capacity vs achievable GFLOPS)
roofline_baseline_vs_optimized.png  # 2D roofline: naive baseline vs full reuse
```

## Current Status: v0 (analytical model only)

Parameters like `onchip_flops_per_elemwise_op` are still estimates. They need
to be calibrated against actual Vitis HLS synthesis results (II, throughput of
the softmax unit) — that's the next step.

## Roadmap

- [x] **Phase 0**: Analytical Roof-Surface model (this repo, v0)
- [ ] **Phase 1**: Baseline HLS attention kernel (no reuse) — implement in Vitis
      HLS, collect measured latency/DSP/BRAM usage, compare against model
      predictions
- [ ] **Phase 2**: Implement on-chip K,V tiling optimization (flash-attention-style
      block reuse)
- [ ] **Phase 3**: Calibrate model parameters — refine `onchip_flops_per_elemwise_op`
      etc. using measured data
- [ ] **Phase 4**: Reproduce DSE across sequence lengths / head dimensions, derive
      the optimal tile size
- [ ] **Phase 5**: Write up results — as an FCCM/FPGA workshop poster or short paper

## References

- Gerogiannis et al., "DECA: A Near-Core LLM Decompression Accelerator Grounded
  on a 3D Roofline Model," arXiv:2505.19349, 2025.
- Zeng et al., "FlightLLM: Efficient Large Language Model Inference with a
  Complete Mapping Flow on FPGAs," FPGA '24.
- Li et al., "HLSTransform: Energy-Efficient LLM Inference on FPGAs via
  High-Level Synthesis," arXiv:2405.00738, 2024.
- A survey of FPGA and ASIC designs for transformer inference acceleration and
  optimization, Journal of Systems Architecture, 2024.
