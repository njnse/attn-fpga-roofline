"""
DSE 예시: Alveo U280에서 seq_len=2048, head_dim=64, num_heads=16 (예: BERT-large급)
attention을 온칩 K,V 캐시 용량을 0%~100%로 늘려가며 병목이 어떻게 이동하는지 관찰.

이건 DECA 논문의 Figure 16 (BORD: W,L 스윕에 따른 bound 영역 변화)과
동일한 역할을 하는 실험이다 -> "언제부터 메모리 재사용 최적화가
수확체감에 들어가는지"를 합성/구현 전에 먼저 짚어보는 것.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attn_roofline.attention_model import AttentionShape, achievable_gflops, compute_cost
from attn_roofline.fpga_platforms import PLATFORMS

shape = AttentionShape(seq_len=4096, head_dim=64, num_heads=16, precision_bytes=2.0)
platform = PLATFORMS["zcu104"]  # HBM 없는 DDR-only 보드라야 memory-bound 구간이 보인다

kv_full_kb = compute_cost(shape, kv_onchip_capacity_kb=1e12).hbm_bytes  # dummy call just to reuse helper
from attn_roofline.attention_model import compute_cost as _cc
_full_kv_kb = 2 * shape.seq_len * shape.head_dim * shape.num_heads * shape.precision_bytes / 1024.0

capacities_pct = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
rows = []
for pct in capacities_pct:
    cap_kb = _full_kv_kb * pct / 100.0
    r = achievable_gflops(shape, platform, kv_onchip_capacity_kb=cap_kb)
    rows.append((pct, r))

print(f"{'OnChip%':>8} {'ReuseFactor':>12} {'Bottleneck':>14} {'Achievable GFLOPS':>18}")
for pct, r in rows:
    print(f"{pct:>7}% {r['reuse_factor']:>12.2f} {r['bottleneck']:>14} {r['achievable_gflops']:>18.1f}")

# ---- Plot: achievable throughput vs on-chip K,V reuse capacity ----
xs = [pct for pct, _ in rows]
ys = [r["achievable_gflops"] for _, r in rows]
colors = {"HBM_BOUND": "tab:red", "ONCHIP_BOUND": "tab:orange", "PE_BOUND": "tab:green"}
bar_colors = [colors[r["bottleneck"]] for _, r in rows]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar([str(p) for p in xs], ys, color=bar_colors)
ax.set_xlabel("On-chip K,V cache capacity (% of full K,V)")
ax.set_ylabel("Achievable GFLOPS")
ax.set_title(f"Attention throughput vs on-chip reuse\n{platform.name}, N={shape.seq_len}, d={shape.head_dim}, h={shape.num_heads}")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
ax.legend(handles, colors.keys(), title="Bottleneck")
fig.tight_layout()
fig.savefig("/home/claude/attn-fpga-roofline/dse_reuse_vs_throughput.png", dpi=150)
print("\nSaved: dse_reuse_vs_throughput.png")

# ---- Classic 2D roofline for the two extreme design points ----
fig2, ax2 = plt.subplots(figsize=(7, 5))
ai_range = [10 ** (x / 10) for x in range(-10, 40)]
ridge_ai = platform.peak_flops / (platform.hbm_bw_gbps * 1e9)
mem_line = [min(platform.hbm_bw_gbps * 1e9 * ai, platform.peak_flops) / 1e9 for ai in ai_range]
ax2.loglog(ai_range, mem_line, "k-", label="Roofline (HBM vs PE peak)")

for pct in [1, 100]:
    cap_kb = _full_kv_kb * pct / 100.0
    r = achievable_gflops(shape, platform, kv_onchip_capacity_kb=cap_kb)
    ax2.scatter([r["ai_hbm"]], [r["achievable_gflops"]], s=80,
                label=f"{pct}% on-chip reuse (bound: {r['bottleneck']})")

ax2.axvline(ridge_ai, linestyle="--", color="gray", alpha=0.5)
ax2.set_xlabel("Arithmetic Intensity (FLOPs / HBM byte)")
ax2.set_ylabel("GFLOPS")
ax2.set_title("Naive baseline vs full on-chip K,V reuse")
ax2.legend()
fig2.tight_layout()
fig2.savefig("/home/claude/attn-fpga-roofline/roofline_baseline_vs_optimized.png", dpi=150)
print("Saved: roofline_baseline_vs_optimized.png")
