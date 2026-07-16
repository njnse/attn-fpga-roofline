"""
FPGA 플랫폼별 대략적인 스펙.

주의: 아래 수치는 공개 데이터시트 기준 근사치입니다 (정확한 설계 시
Xilinx/AMD 공식 데이터시트 및 실측치로 반드시 재검증할 것).
SHREC 랩에서 실제로 사용하는 보드로 값을 교체해서 쓰세요.
"""

from dataclasses import dataclass


@dataclass
class FPGAPlatform:
    name: str
    hbm_bw_gbps: float          # 오프칩 메모리(HBM or DDR) 대역폭, GB/s
    dsp_count: int              # DSP 슬라이스 개수
    clock_mhz: float            # 연산 클록
    macs_per_dsp_per_cycle: float  # DSP당 사이클당 MAC 수 (INT8 packing 등 고려)
    onchip_mem_kb: int          # BRAM+URAM 총 용량, KB
    onchip_bw_gbps: float       # 온칩 메모리(softmax/activation 유닛 등) 유효 대역폭 근사치, GB/s

    @property
    def peak_flops(self) -> float:
        # FMA = 2 FLOPs
        return self.dsp_count * self.macs_per_dsp_per_cycle * self.clock_mhz * 1e6 * 2


PLATFORMS = {
    "alveo_u280": FPGAPlatform(
        name="Alveo U280 (HBM2, 2x460GB/s stacks)",
        hbm_bw_gbps=460.0,
        dsp_count=9024,
        clock_mhz=300.0,
        macs_per_dsp_per_cycle=1.0,
        onchip_mem_kb=(4032 + 2160) * 36 // 8,  # BRAM36+URAM 대략치, KB 단위 근사
        onchip_bw_gbps=8000.0,
    ),
    "alveo_u50": FPGAPlatform(
        name="Alveo U50 (HBM2, 316GB/s)",
        hbm_bw_gbps=316.0,
        dsp_count=5952,
        clock_mhz=300.0,
        macs_per_dsp_per_cycle=1.0,
        onchip_mem_kb=(1344 + 1352) * 36 // 8,
        onchip_bw_gbps=6000.0,
    ),
    "zcu104": FPGAPlatform(
        name="ZCU104 (Zynq UltraScale+, DDR4 only, no HBM)",
        hbm_bw_gbps=19.2,  # 단일 DDR4-2400 채널 근사치
        dsp_count=1728,
        clock_mhz=300.0,
        macs_per_dsp_per_cycle=1.0,
        onchip_mem_kb=312 * 36 // 8,
        onchip_bw_gbps=2000.0,
    ),
}
