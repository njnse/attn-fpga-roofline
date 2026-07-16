# Attention On-Chip Memory Reuse: A Roof-Surface Approach for FPGA Transformer Accelerators

## Motivation

FPGA 기반 트랜스포머 가속기 연구는 이미 성숙한 분야다 (FlightLLM, HLSTransform,
Transformer-OPU 등). 차별화 지점은 "왜 특정 설계가 병목에 걸리는지를 구현 전에
예측할 수 있는 분석 프레임워크"를 만드는 것이다.

Intel의 DECA (Gerogiannis et al., 2025, "A Near-Core LLM Decompression
Accelerator Grounded on a 3D Roofline Model")는 CPU+HBM 환경에서 memory /
vector-decompress / matrix-engine 세 자원의 상호작용을 3D Roof-Surface 모델로
표현하고, 이를 근거로 accelerator 파라미터(W, L)를 최소 비용으로 결정했다.

이 프로젝트는 같은 방법론을 **FPGA attention 커널의 온칩 K,V 재사용 설계**에
이식한다. FPGA에서는 세 자원이 다음과 같이 대응된다:

| DECA (CPU)         | 이 프로젝트 (FPGA)                          |
|---------------------|----------------------------------------------|
| MEM (HBM bandwidth) | HBM/DDR bandwidth (K,V,Q 로드)               |
| VEC (AVX 압축해제)  | 온칩 elementwise 유닛 (softmax/exp, scaling) |
| MTX (TMUL)          | Systolic PE array (DSP 기반 MAC)             |

## 핵심 가설

Attention의 K,V를 온칩(BRAM/URAM)에 얼마나 재사용하는지가 HBM 병목 여부를
결정한다. 그러나 무한정 재사용 용량을 늘리는 것은 온칩 자원 낭비다.
**"병목이 HBM-bound에서 PE-bound로 전환되는 최소 온칩 용량"**을 실측 이전에
분석적으로 찾아내고, 그 지점 근처로 실제 HLS 설계를 튜닝하는 것이 목표다.
(DECA 논문의 Figure 16, {W=32,L=8} 최적점을 찾는 과정과 동일한 방법론)

## 이 저장소의 구성

```
attn_roofline/
  fpga_platforms.py   # 보드별 HBM 대역폭, DSP 피크 연산량, 온칩 용량 (근사치, 검증 필요)
  attention_model.py  # attention 연산의 FLOPs/HBM bytes/온칩 elemwise ops를
                       # 온칩 K,V 재사용 정도(reuse_factor)의 함수로 모델링
run_dse.py             # 온칩 재사용 용량을 스윕하며 병목 전환점을 찾는 예시 스크립트
dse_reuse_vs_throughput.png     # 결과 그래프 (재사용 용량 vs achievable GFLOPS)
roofline_baseline_vs_optimized.png  # 2D roofline: naive baseline vs 완전 재사용
```

## 현재 상태: v0 (분석 모델만)

`onchip_flops_per_elemwise_op` 같은 파라미터는 아직 추정치다. 실제 Vitis HLS
합성 결과(softmax 유닛의 II, throughput)로 캘리브레이션이 필요하다 — 이게
다음 단계다.

## 로드맵

- [x] **Phase 0**: 분석적 Roof-Surface 모델 (이 리포, v0)
- [ ] **Phase 1**: 베이스라인 HLS attention 커널 (재사용 없음) — Vitis HLS로 구현,
      실측 latency/DSP/BRAM 사용량 수집, 모델 예측과 비교
- [ ] **Phase 2**: 온칩 K,V 타일링 최적화 버전 구현 (flash-attention 스타일 블록 재사용)
- [ ] **Phase 3**: 모델 파라미터 캘리브레이션 — 실측치로 `onchip_flops_per_elemwise_op` 등 보정
- [ ] **Phase 4**: 여러 시퀀스 길이/헤드 차원에서 DSE 재현, 최적 타일 크기 도출
- [ ] **Phase 5**: 결과 정리 — FCCM/FPGA 워크숍 poster 또는 short paper 형태로 작성

## 참고문헌

- Gerogiannis et al., "DECA: A Near-Core LLM Decompression Accelerator Grounded
  on a 3D Roofline Model," arXiv:2505.19349, 2025.
- Zeng et al., "FlightLLM: Efficient Large Language Model Inference with a
  Complete Mapping Flow on FPGAs," FPGA '24.
- Li et al., "HLSTransform: Energy-Efficient LLM Inference on FPGAs via
  High-Level Synthesis," arXiv:2405.00738, 2024.
- A survey of FPGA and ASIC designs for transformer inference acceleration and
  optimization, Journal of Systems Architecture, 2024.
