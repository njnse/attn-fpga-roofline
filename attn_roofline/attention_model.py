"""
Self-attention 한 층(single head 기준, 필요시 num_heads로 스케일)의
FLOPs / HBM 접근 바이트 / 온칩(softmax 등) 접근량을,
"K,V를 몇 번이나 다시 HBM에서 읽어오는가(reuse degree)"의 함수로 모델링한다.

핵심 아이디어 (DECA의 Roof-Surface을 FPGA로 이식):
- Query를 블록 크기 B_q로 타일링해서 처리한다고 가정.
- K, V 전체(N x d)가 온칩에 다 들어가면 -> HBM에서 K,V는 딱 1번만 로드.
- 온칩 용량이 부족해서 K,V를 다시 로드해야 하면 -> query 블록마다 K,V를 재로드
  (reuse_factor = N / B_q 번 다시 읽음).

reuse_factor = 1  -> 완전 온칩 재사용 (best case, 우리가 만들려는 최적화)
reuse_factor = N/B_q -> 재사용 없음 (naive baseline)
"""

from dataclasses import dataclass
from .fpga_platforms import FPGAPlatform


@dataclass
class AttentionShape:
    seq_len: int          # N
    head_dim: int          # d
    num_heads: int = 1
    precision_bytes: float = 2.0  # BF16=2, INT8=1, FP32=4


@dataclass
class AttentionCost:
    flops: float             # 총 matmul FLOPs (QK^T + AV, 전체 heads)
    hbm_bytes: float          # HBM에서 읽어와야 하는 총 바이트 (Q, K, V, reload 포함)
    onchip_elemwise_ops: float  # softmax(exp)/scale 등 온칩 elementwise 연산 수
    reuse_factor: float


def compute_cost(shape: AttentionShape, kv_onchip_capacity_kb: float) -> AttentionCost:
    N, d, h, bytes_ = shape.seq_len, shape.head_dim, shape.num_heads, shape.precision_bytes

    # matmul FLOPs: QK^T (2*N*N*d) + AV (2*N*N*d), heads만큼 스케일
    flops = 2 * (2 * N * N * d) * h

    # K,V 전체를 온칩에 올리는 데 필요한 용량 (heads 합산, KB)
    kv_full_kb = 2 * N * d * h * bytes_ / 1024.0

    if kv_full_kb <= kv_onchip_capacity_kb:
        # 온칩에 K,V가 통째로 들어감 -> HBM에서 1번만 로드
        reuse_factor = 1.0
    else:
        # 온칩 용량 제약으로 인해 몇 번이나 재로드해야 하는지 근사
        # (query 블록 크기를 온칩 용량에 맞춰 역산)
        reuse_factor = kv_full_kb / kv_onchip_capacity_kb

    q_bytes = N * d * h * bytes_
    out_bytes = N * d * h * bytes_
    kv_bytes = 2 * N * d * h * bytes_ * reuse_factor

    hbm_bytes = q_bytes + out_bytes + kv_bytes

    # softmax: N x N 원소마다 exp 1회씩, heads만큼
    onchip_elemwise_ops = N * N * h

    return AttentionCost(
        flops=flops,
        hbm_bytes=hbm_bytes,
        onchip_elemwise_ops=onchip_elemwise_ops,
        reuse_factor=reuse_factor,
    )


def achievable_gflops(shape: AttentionShape, platform: FPGAPlatform,
                        kv_onchip_capacity_kb: float,
                        onchip_flops_per_elemwise_op: float = 8.0) -> dict:
    """
    DECA의 Roof-Surface 식과 동일한 구조:
        rate = min(HBM_BW * AI_hbm, OnChipBW * AI_onchip, PE_peak)
    세 후보 중 가장 낮은 값이 실제 달성 가능한 성능(병목)이 된다.
    """
    cost = compute_cost(shape, kv_onchip_capacity_kb)

    ai_hbm = cost.flops / cost.hbm_bytes  # FLOPs per HBM byte
    mem_bound_flops = platform.hbm_bw_gbps * 1e9 * ai_hbm

    # onchip_flops_per_elemwise_op: exp 1회 처리하는 데 필요한 "실효 온칩 대역폭 단위" 가정치.
    # 실측(HLS synth report)으로 나중에 캘리브레이션 필요.
    onchip_bound_flops = platform.onchip_bw_gbps * 1e9 * (
        cost.flops / (cost.onchip_elemwise_ops * onchip_flops_per_elemwise_op)
    )

    pe_bound_flops = platform.peak_flops

    bottleneck = min(
        [
            ("HBM_BOUND", mem_bound_flops),
            ("ONCHIP_BOUND", onchip_bound_flops),
            ("PE_BOUND", pe_bound_flops),
        ],
        key=lambda x: x[1],
    )

    return {
        "ai_hbm": ai_hbm,
        "reuse_factor": cost.reuse_factor,
        "mem_bound_gflops": mem_bound_flops / 1e9,
        "onchip_bound_gflops": onchip_bound_flops / 1e9,
        "pe_bound_gflops": pe_bound_flops / 1e9,
        "achievable_gflops": bottleneck[1] / 1e9,
        "bottleneck": bottleneck[0],
    }
