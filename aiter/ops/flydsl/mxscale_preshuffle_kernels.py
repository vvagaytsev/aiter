# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatcher for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA).

Operand convention (the weight is preshuffled caller-side):
    A       : [M, K]   row-major, NOT preshuffled  (fp8/fp6 = 1 byte/code, fp4 = 2 codes/byte)
    B       : preshuffled via aiter.ops.shuffle.shuffle_weight(., (16, 16))  (fp4 or fp8 weight)
    a_scale / b_scale : E8M0, each a block and a layout (flydsl_mxscale_preshuffle_gemm).
              "row" -- raw, as quantized, staged by the kernel itself (a8w8): x_scale
              1x32 [M, K//32] or the blockscale 1x128 column-major bytes of [K//128, M];
              w_scale 1x32 [N, K//32], 32x32 [N//32, K//32] or 128x128 [N//128, K//128].
              "mfma_tile" -- pre-shuffled into the MFMA operand order by the caller
              (per-1x32 shuffle_scale_a16w4'd, served by
              gemm_a8w8_mxscale_preshuffle_1x32_flydsl; the coarse 1x128 / 128x128
              blocks in the compact broadcast layout the kernel reads, which no
              aiter op passes).
    Out     : [M, N]   bf16 / fp16

``run_gemm_a8w8_mxfp8_gfx950`` is the entry point that gemm_a8w8_mxfp8_bpreshuffle,
gemm_a8w8_blockscale_bpreshuffle's "mxfp8" rows and the tuner share: raw scales,
a tuned kernelName (mxpsh or 8-wave) or the heuristic.
"""

from __future__ import annotations

import functools

import torch

from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

# gemm_a8w8_mxfp8_bpreshuffle's gfx950 scales per w_scale block (row-major 1x32
# x_scale; 1x32 or 32x32 w_scale), defined next to its tune pipelines;
# re-exported for the runner and AOT.
from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK,
)

_OUT_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}
# 256 CUs over 8 XCDs: the fused split-K l2_reduce grid's per-XCD budget.
_GFX950_CUS_PER_XCD = 32


@functools.cache
def _gemm_exe(fused):
    import flydsl.compiler as flyc

    from .kernels.gemm.mxscale_preshuffle import launch_gemm, launch_gemm_fused

    launcher = launch_gemm_fused if fused else launch_gemm
    return flyc.jit(launcher.func)


@functools.cache
def _reduce_exe():
    import flydsl.compiler as flyc

    from .kernels.gemm.mxscale_preshuffle import launch_splitk_reduce
    from .kernels.kernels_common import set_slp_vectorize_once_used

    # flyc.jit(.func) starts with empty compile_hints, so the hint on
    # launch_splitk_reduce itself never reaches this reduction.
    return set_slp_vectorize_once_used(flyc.jit(launch_splitk_reduce.func), False)


@functools.cache
def _fused_splitk_workspace(
    device: torch.device,
    stream_id: int,
    split_k: int,
    M: int,
    N: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Every split stores its whole partial before arriving, so this needs no
    # initialization and is shared with captures. Never evicted: a captured
    # graph may point at it.
    del stream_id
    with persistent_alloc(device):
        return torch.empty((split_k, M, N), dtype=dtype, device=device)


@capture_scoped_cache
def _fused_splitk_semaphore(
    device: torch.device, stream_id: int, num_tiles: int
) -> torch.Tensor:
    # Safe to reuse: launches on a stream are ordered and each tile's last
    # arrival hands its counter back zeroed.
    del stream_id
    with persistent_alloc(device):
        return torch.zeros((num_tiles,), dtype=torch.int32, device=device)


def _fused_splitk_buffers(
    device: torch.device,
    stream_id: int,
    split_k: int,
    M: int,
    N: int,
    num_tiles: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stream-private partials (BF16 for the coarse small-M path, FP32 for 1x32
    row scales) and per-output-tile arrival counters."""
    return (
        _fused_splitk_workspace(device, stream_id, split_k, M, N, dtype),
        _fused_splitk_semaphore(device, stream_id, num_tiles),
    )


def flydsl_mxscale_preshuffle_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    Out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str = "fp4",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
    split_k: int = 1,
    splitk_workspace: torch.Tensor | None = None,
    stream=None,
    a_scale_block: str,
    b_scale_block: str,
    a_scale_layout: str,
    b_scale_layout: str,
    fused_split_k: bool = False,
    prefetch: bool | int = False,
    b_nt: bool = False,
    l2_reduce: bool | None = None,
) -> torch.Tensor:
    """Run the gfx950 MXFP4/6/8 preshuffle GEMM. a8w8 = a_dtype="fp8", b_dtype="fp8".

    A is [M, K]; N is taken from Out ([M, N]); K from A. Returns Out.

    split_k>1 splits the K reduction across grid.z: fp32 partial slabs and a
    separate reduce kernel, or, with fused_split_k (row layouts, bf16 Out), an
    in-kernel reduce by each output tile's last-arriving block -- the same FP32
    numbers, a tunable (kernelName ``_fsk``). A mfma_tile coarse-block M<=16 tile
    reduces BF16 partials in-kernel instead.

    Each operand's scale is a block and a layout, always named by the caller (see
    flydsl_gemm_mxscale_preshuffle_common.SCALE_LAYOUTS); a default would read one
    contract's bytes as the other's:

    * a_scale_block "1x32" | "1x128", b_scale_block "1x32" | "128x128" | "32x32".
      The coarse blocks are a8w8-only; a 128x128 b_scale needs N%128==0.
    * "row" (a8w8): the UNSHUFFLED E8M0 as quantized, staged by the kernel
      itself: a 1x32 A [M, K//32] (no row padding) / B [N, K//32]; a 32x32 B
      [N//32, K//32]; a 1x128 A the blockscale column-major bytes [K//128, M]
      (a contiguous (M, K//128) tensor holding them, as bmm's transposed
      x_scale); a 128x128 B [N//128, K//128].
    * "mfma_tile": pre-shuffled into the MFMA operand order by the caller -- the
      per-1x32 scales through ``shuffle_scale_a16w4``, the coarse 1x128 / 128x128
      blocks in the compact broadcast layout the kernel reads (no aiter helper
      builds those; no aiter op passes them). The op does NOT repack them.

    splitk_workspace, when set, is the caller-owned contiguous float32 buffer of
    shape ``(split_k, M, N)`` on A's device. The kernel owns it exclusively for
    the     launch duration; the same buffer may be reused on the same stream after
    the GEMM+reduce finish (including across CUDA-graph replay of that launch).
    It is ignored when split_k == 1 or the fused split-K path runs.
    """
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        SCALE_BLOCKS_A,
        SCALE_BLOCKS_B,
        SCALE_LAYOUTS,
    )
    from .kernels.tensor_shim import _run_compiled, ptr_arg

    # Logical K: fp4 A packs 2 codes/byte (A last dim = K//2); fp6/fp8 A = 1 byte/code.
    if a_dtype not in ("fp4", "fp6", "fp8"):
        raise ValueError(
            f"unsupported a_dtype {a_dtype!r}; expected 'fp4', 'fp6', or 'fp8'"
        )
    if b_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported b_dtype {b_dtype!r}; expected 'fp4' or 'fp8'")

    M = int(A.shape[0])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    N = int(Out.shape[-1])
    if N % int(tile_n) != 0:
        raise ValueError(f"N ({N}) is not a multiple of tile_n ({tile_n})")
    if K % int(tile_k) != 0:
        raise ValueError(f"K ({K}) is not a multiple of tile_k ({tile_k})")
    if K % 128 != 0:
        raise ValueError(
            f"K ({K}) must be a multiple of 128 for MXFP microscale; got {K}"
        )
    out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
    if out_dtype is None:
        raise ValueError(
            f"unsupported Out dtype {Out.dtype}; expected bfloat16 or float16"
        )

    if a_scale_block not in SCALE_BLOCKS_A or b_scale_block not in SCALE_BLOCKS_B:
        raise ValueError(
            f"unsupported scale blocks a={a_scale_block!r} b={b_scale_block!r}; "
            f"expected a in {SCALE_BLOCKS_A}, b in {SCALE_BLOCKS_B}"
        )
    for layout in (a_scale_layout, b_scale_layout):
        if layout not in SCALE_LAYOUTS:
            raise ValueError(f"unsupported scale layout {layout!r}; {SCALE_LAYOUTS}")
    # The mfma_tile 1x128 / 128x128 blocks broadcast to the MFMA through the
    # load address; a row (unshuffled) scale of any block is not coarse (nor
    # does it take the kernel's fused small-M blockscale path).
    coarse = (a_scale_block == "1x128" and a_scale_layout != "row") or (
        b_scale_block == "128x128" and b_scale_layout != "row"
    )
    # An unsupported shape is an error rather than a silent downgrade.
    if coarse and (a_dtype != "fp8" or b_dtype != "fp8"):
        raise ValueError(
            f"a coarse scale block is a8w8-only; got a_dtype={a_dtype!r} "
            f"b_dtype={b_dtype!r}"
        )
    if b_scale_block == "128x128" and N % 128 != 0:
        raise ValueError(f"a 128x128 b_scale needs N ({N}) a multiple of 128")
    for name, t, block, layout, rows in (
        ("a_scale", a_scale, a_scale_block, a_scale_layout, M),
        ("b_scale", b_scale, b_scale_block, b_scale_layout, N),
    ):
        if layout != "row":
            if block == "32x32":
                raise ValueError(f"a 32x32 {name} is row layout only")
            continue
        if (a_dtype, b_dtype) != ("fp8", "fp8"):
            raise ValueError(f"a row {name} is a8w8 only")
        # (rows per scale row, K per scale column) of each row block.
        rows_per, k_per = {"1x32": (1, 32), "32x32": (32, 32)}.get(block, (128, 128))
        if block == "1x128":
            rows_per = 1  # column-major bytes of a (rows, K // 128) tensor
        if rows % rows_per != 0:
            raise ValueError(
                f"a {block} {name} needs {rows} rows a multiple of {rows_per}"
            )
        want = (rows // rows_per, K // k_per)
        if tuple(t.shape) != want or not t.is_contiguous():
            raise ValueError(
                f"row {name} must be a contiguous {want}, got "
                f"{tuple(t.shape)} stride {tuple(t.stride())}"
            )
    split_k = int(split_k)
    # For the latency-sensitive M<=16 path, keep the split partials in BF16 and
    # let the last arriving GEMM block reduce them. This avoids a second launch
    # without introducing output atomics or a grid-wide spin wait.
    splitk_fused = (
        coarse
        and split_k > 1
        and M <= 16
        and int(tile_m) == 16
        and out_dtype == "bf16"
        and int(tile_n) <= 128
    )
    # fused_split_k: row-layout scales reduce in-kernel at any tile, from FP32
    # partials (the kernel's fused_f32; it must agree, as the fused launch ABI
    # differs).
    fused_f32 = (
        bool(fused_split_k)
        and split_k > 1
        and (a_scale_layout, b_scale_layout) == ("row", "row")
        and out_dtype == "bf16"
    )
    splitk_fused = splitk_fused or fused_f32

    st = stream if stream is not None else torch.cuda.current_stream()

    if split_k > 1:
        # split-K legality (same constraints the tuner enforces in fits_shape):
        # per-split K must be a whole number of tile_k tiles AND 256-K scale chunks.
        k_per_split = K // split_k
        if K % split_k != 0 or k_per_split % int(tile_k) != 0 or k_per_split % 256 != 0:
            raise ValueError(
                f"illegal split_k={split_k} for K={K}, tile_k={tile_k}: "
                f"K/split_k ({k_per_split}) must be a multiple of tile_k and 256"
            )

    # Constexpr tail of launch_gemm -- the same for the direct and the split-K launch
    # apart from k_batch, and it doubles as the per-config compiled-kernel cache key.
    cfg = (
        N,
        K,
        int(tile_m),
        int(tile_n),
        int(tile_k),
        a_dtype,
        out_dtype,
        b_dtype,
        1,  # batch
        -1,  # a_row_stride     ) each <0 keeps the contiguous
        -1,  # a_batch_stride   ) [B,M,*] default; this op never
        -1,  # sca_row_stride   ) overrides them, the batched
        -1,  # sca_batch_stride ) callers do
        -1,  # c_row_stride
        -1,  # c_batch_stride
        int(waves_per_eu),
        int(xcd_swizzle),
        split_k,  # k_batch
        a_scale_block,
        b_scale_block,
    )
    multi_row = int(tile_m) == 16 and M > 1
    layouts = (a_scale_layout, b_scale_layout)  # launch_gemm's last Constexprs
    gemm_exe = _gemm_exe(splitk_fused)
    # launch_gemm's Constexpr tail but fused_split_k, which each launch appends.
    spec = (*cfg, multi_row, *layouts)
    if prefetch is True:
        from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import prefetch_depth

        N = int(Out.shape[-1])
        depth = prefetch_depth(
            int(tile_m), int(tile_n), int(tile_k), N, K, split_k, bool(b_nt)
        )
        if depth < 2:
            raise ValueError(
                f"no prefetch depth for tile {tile_m}x{tile_n}x{tile_k}, K={K}, "
                f"split_k={split_k}"
            )
    else:
        depth = max(1, int(prefetch))
    if b_nt and depth < 2:
        raise ValueError("b_nt streams the prefetch pipeline's B: needs prefetch")
    pf_tail = (depth, bool(b_nt)) if depth > 1 else ()
    # Build each runtime pointer wrapper once per op.  The fused ABI carries two
    # additional pointer slots; re-wrapping Out (or the split workspace) for
    # those aliases measurably increases launch gaps on very short decode GEMMs.
    a_ptr = ptr_arg(A)
    b_ptr = ptr_arg(B)
    a_scale_ptr = ptr_arg(a_scale)
    b_scale_ptr = ptr_arg(b_scale)
    out_ptr = ptr_arg(Out)

    if split_k == 1:
        _run_compiled(
            gemm_exe,
            out_ptr,
            a_ptr,
            b_ptr,
            a_scale_ptr,
            b_scale_ptr,
            M,
            N,
            st,
            *spec,
            False,  # fused_split_k
            *pf_tail,
            specialization_key=(*spec, False, *pf_tail),
        )
        return Out

    if splitk_fused:
        num_tiles = ((M + int(tile_m) - 1) // int(tile_m)) * (N // int(tile_n))
        if isinstance(st, torch.cuda.Stream):
            with torch.cuda.stream(st):
                workspace, semaphore = _fused_splitk_buffers(
                    A.device,
                    int(st.cuda_stream),
                    split_k,
                    M,
                    N,
                    num_tiles,
                    torch.float32 if fused_f32 else torch.bfloat16,
                )
        else:
            # Compile-only/AOT uses an fx.Stream placeholder and CPU/Fake tensors:
            # nothing launches, so skip the HIP pool and capture lookups.
            workspace = torch.empty(
                (split_k, M, N),
                dtype=torch.float32 if fused_f32 else torch.bfloat16,
                device=A.device,
            )
            semaphore = torch.zeros((num_tiles,), dtype=torch.int32, device=A.device)

        if l2_reduce is None:
            l2_reduce = -(-num_tiles // 8) * split_k <= _GFX950_CUS_PER_XCD
        fused_tail = (depth, bool(b_nt), True) if fused_f32 and l2_reduce else pf_tail
        _run_compiled(
            gemm_exe,
            ptr_arg(workspace),
            a_ptr,
            b_ptr,
            a_scale_ptr,
            b_scale_ptr,
            out_ptr,
            ptr_arg(semaphore),
            M,
            N,
            st,
            *spec,
            fused_f32,  # fused_split_k
            *fused_tail,
            specialization_key=(*spec, fused_f32, *fused_tail),
        )
        return Out

    # split-K: GEMM -> fp32 partial slabs tmp[split_k, M, N] -> fused fp32 reduce -> Out.
    # A caller-owned slab keeps the address stable for CUDA-graph replay.
    if splitk_workspace is None:
        tmp = torch.empty((split_k, M, N), dtype=torch.float32, device=A.device)
    else:
        if (
            splitk_workspace.shape != (split_k, M, N)
            or splitk_workspace.dtype != torch.float32
            or splitk_workspace.device != A.device
            or not splitk_workspace.is_contiguous()
        ):
            raise ValueError(
                "splitk_workspace must be contiguous float32 on A's device "
                f"with shape {(split_k, M, N)}"
            )
        tmp = splitk_workspace
    if isinstance(st, torch.cuda.Stream):
        tmp.record_stream(st)
    tmp_ptr = ptr_arg(tmp)
    _run_compiled(
        gemm_exe,
        tmp_ptr,
        a_ptr,
        b_ptr,
        a_scale_ptr,
        b_scale_ptr,
        M,
        N,
        st,
        *spec,
        False,  # fused_split_k
        *pf_tail,
        specialization_key=(*spec, False, *pf_tail),
    )
    _run_compiled(
        _reduce_exe(),
        tmp_ptr,
        out_ptr,
        (M * N) // 2,  # n_out_dw (2 out elems per dword)
        M * N,  # slab_stride_dw (fp32: 1 dword/elem)
        st,
        split_k,
        out_dtype,
        specialization_key=(split_k, out_dtype),
    )
    return Out


_HEURISTIC_MIN_TILE_N = 32


@functools.lru_cache(maxsize=1024)
def _warn_untuned(M, N, K, a_dtype, b_dtype, kernel_name):
    from aiter import logger

    logger.warning(
        f"[flydsl mxpsh] no tuned row for M={M} N={N} K={K} {a_dtype}/{b_dtype}; "
        f"falling back to the heuristic tile '{kernel_name}'. It picks a grid that "
        f"fills the CUs, but nothing about this shape was measured -- tune it with "
        f"csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py."
    )


@functools.lru_cache(maxsize=1024)
def _heuristic_tile(a_dtype, b_dtype, M, N, K, scales):
    from aiter.jit.utils.chip_info import get_cu_num

    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import candidates_for

    # scales: candidates_for's scale kwargs as sorted (name, value) pairs, which
    # keep the call hashable for the cache.
    # tile_k=512, _fsk and _pf are tuned-only options: the heuristic keeps its
    # untuned picks.
    cands = [
        ki
        for _, ki in candidates_for(a_dtype, b_dtype, M, N, K, **dict(scales))
        if ki.tile_k <= 256 and not ki.fused_split_k and not ki.prefetch
    ]
    if not cands:
        return None

    def _wg(ki):
        return -(-M // ki.tile_m) * (N // ki.tile_n) * ki.split_k

    pool = [ki for ki in cands if ki.tile_n >= _HEURISTIC_MIN_TILE_N] or cands
    filled = [ki for ki in pool if _wg(ki) >= get_cu_num()]
    if filled:
        pool = filled
    else:  # N too narrow, or split-K legality leaves nothing that fills the CUs
        best_wg = max(_wg(ki) for ki in pool)
        pool = [ki for ki in pool if _wg(ki) == best_wg]

    target_m = min(max((M + 31) // 32 * 32, 32), 128)
    return max(
        pool,
        key=lambda ki: (
            -ki.split_k,
            ki.tile_k,
            ki.tile_n,
            -abs(ki.tile_m - target_m),
            -ki.waves_per_eu,
        ),
    )


def run_gemm_a8w8_mxfp8_gfx950(
    XQ, WQ, x_scale, w_scale, Out, kernel_name=None, w_scale_block=None
):
    """gfx950 FP8 GEMM with unshuffled e8m0 scales, into ``Out``.

    XQ [M, K] fp8 row-major; WQ [N, K] fp8, shuffle_weight(., (16, 16)).
    x_scale [M, K // 32] and w_scale e8m0, row-major and NOT shuffled (no row
    padding): the kernel stages them through LDS itself. w_scale is
    [N, K // 32] for a 1x32 ``w_scale_block`` or [N // 32, K // 32] for 32x32;
    None reads the block off its shape. ``kernel_name`` is a tuned
    row's mxpsh or 8-wave kernelName, else the heuristic picks an mxpsh tile.
    """
    from ..gemm_op_common import mxscale_w_scale_block
    from .gemm_a8w8_mxfp8_8wave import NAME_PREFIX as MXFP8_8WAVE_PREFIX
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import parse_kernel_name

    if w_scale_block is None:
        w_scale_block = mxscale_w_scale_block(
            tuple(w_scale.shape), int(WQ.shape[0]), int(WQ.shape[-1])
        )
    scales = MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK.get(w_scale_block)
    if scales is None:
        raise NotImplementedError(
            f"[FlyDSL gfx950 mxfp8] no kernel reads a {w_scale_block} w_scale; "
            f"supported: {sorted(MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK)}"
        )

    if kernel_name is not None and kernel_name.startswith(f"{MXFP8_8WAVE_PREFIX}_"):
        if Out.dtype == torch.bfloat16:
            from .gemm_a8w8_mxfp8_8wave import run_gemm_a8w8_mxfp8_8wave

            return run_gemm_a8w8_mxfp8_8wave(
                XQ, WQ, x_scale, w_scale, Out, kernel_name, w_scale_block
            )
        # The 8-wave kernel writes bf16 only; any other Out runs the mxpsh
        # heuristic tile for this (tuned) shape instead.
        kernel_name = None
        warn_untuned = False
    else:
        warn_untuned = kernel_name is None

    for name, t in (("XQ", XQ), ("WQ", WQ), ("Out", Out)):
        if not t.is_contiguous():
            raise RuntimeError(
                f"[FlyDSL gfx950 mxfp8] {name} must be contiguous, got shape "
                f"{tuple(t.shape)} stride {tuple(t.stride())}"
            )
    M, K = int(XQ.shape[0]), int(XQ.shape[-1])
    N = int(Out.shape[-1])
    if kernel_name is None:
        ki = _heuristic_tile("fp8", "fp8", M, N, K, scales)
        if ki is None:
            raise NotImplementedError(
                f"[FlyDSL gfx950 mxfp8] no kernel for M={M}, N={N}, K={K} with "
                f"a {w_scale_block} w_scale (needs K%128==0 and N a multiple of a "
                f"tile_n)"
            )
        kernel_name = ki.name
        if warn_untuned:
            _warn_untuned(M, N, K, "fp8", "fp8", kernel_name)
    p = parse_kernel_name(kernel_name)
    if p is None or (p["a_dtype"], p["b_dtype"]) != ("fp8", "fp8"):
        raise ValueError(
            f"[FlyDSL gfx950 mxfp8] {kernel_name!r} is not an a8w8 mxpsh kernel"
        )
    # flydsl_mxscale_preshuffle_gemm checks the row scale shapes.
    return flydsl_mxscale_preshuffle_gemm(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        a_dtype="fp8",
        b_dtype="fp8",
        tile_m=p["tile_m"],
        tile_n=p["tile_n"],
        tile_k=p["tile_k"],
        waves_per_eu=p["waves_per_eu"],
        xcd_swizzle=p["xcd_swizzle"],
        split_k=p["split_k"],
        fused_split_k=p["fused_split_k"],
        prefetch=p["prefetch"],
        b_nt=p["b_nt"],
        **dict(scales),
    )


def pick_mxfp8_kernel_name_gfx950(M, N, K, w_scale_block="1x32"):
    """The kernelName run_gemm_a8w8_mxfp8_gfx950 runs for an untuned shape."""
    ki = _heuristic_tile(
        "fp8", "fp8", M, N, K, MXFP8_ROW_SCALES_BY_W_SCALE_BLOCK[w_scale_block]
    )
    return None if ki is None else ki.name


def _mx_mfma_tile_1x32_scale_shape(rows: int, k: int) -> tuple[int, int]:
    """``shuffle_scale_a16w4`` / mfma_tile 1x32 scale shape for ``rows`` of K."""
    from aiter.ops.inverse_rope_group_quant import scale_shape

    return scale_shape(rows, 1, k // 32, "mfma_tile", quant_group_size=32)[1:]


def _mx_scale_nbytes(rows: int, k: int) -> int:
    r, c = _mx_mfma_tile_1x32_scale_shape(rows, k)
    return r * c


_TUNED_CACHE: dict = {}


def get_mxscale_preshuffle_config(
    M: int,
    N: int,
    K: int,
    *,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    tuned_file=None,
):
    """Exact (gfx, CU, shape, operand dtype) per-1x32 MX tune row, or None."""
    import pandas as pd

    from aiter.jit.core import AITER_CONFIGS
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    tune_file = tuned_file or AITER_CONFIGS.AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE_FILE
    if tune_file not in _TUNED_CACHE:
        try:
            frame = pd.read_csv(tune_file).drop_duplicates()
            _TUNED_CACHE[tune_file] = frame.set_index(
                ["gfx", "cu_num", "M", "N", "K", "a_dtype", "b_dtype"]
            ).to_dict("index")
        except FileNotFoundError:
            if tuned_file is not None:
                raise
            _TUNED_CACHE[tune_file] = None
    table = _TUNED_CACHE[tune_file]
    if not table:
        return None
    return table.get(
        (get_gfx_runtime(), get_cu_num(), int(M), int(N), int(K), a_dtype, b_dtype)
    )


# shuffle_scale_a16w4 scales: one E8M0 byte per 32 K, in MFMA operand order.
_MXFP8_MFMA_1X32 = (
    ("a_scale_block", "1x32"),
    ("a_scale_layout", "mfma_tile"),
    ("b_scale_block", "1x32"),
    ("b_scale_layout", "mfma_tile"),
)


def gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
    A,
    B,
    a_scale,
    b_scale,
    Out,
    *,
    a_dtype,
    b_dtype,
    tile_m=None,
    tile_n=None,
    tile_k=None,
    waves_per_eu=None,
    xcd_swizzle=None,
    split_k=None,
    config=None,
    require_tuned=False,
    splitk_workspace=None,
    stream=None,
):
    """Dispatch a per-1x32 pre-shuffled MX GEMM.

    A and B have packed shape [M, K/pack] and [N, K/pack], respectively, where
    pack is 2 for fp4 and 1 otherwise. B is shuffled, and both scales use
    ``shuffle_scale_a16w4``. N must be a multiple of 32 and K a multiple of 128.
    Tiles must satisfy the MFMA-tile launch constraints; split-K must be positive.
    Exact CSV rows win; ``require_tuned`` refuses a heuristic on a missing or
    incompatible row.
    """
    from aiter.utility import dtypes

    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        mxscale_preshuffle_1x32_params,
        validate_mxscale_preshuffle_1x32_launch,
    )
    from .kernels.tensor_shim import check_e8m0

    for name, t in (("A", A), ("B", B), ("Out", Out)):
        if not t.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous, got shape {tuple(t.shape)} "
                f"stride {tuple(t.stride())}"
            )
        if t.ndim != 2:
            raise ValueError(f"{name} must be a 2D tensor, got {tuple(t.shape)}")
    for name, t in (("B", B), ("a_scale", a_scale), ("b_scale", b_scale), ("Out", Out)):
        if t.device != A.device:
            raise ValueError(f"{name} must be on {A.device}, got {t.device}")
    operand_dtypes = {
        "fp4": (torch.uint8, dtypes.fp4x2),
        "fp6": (torch.uint8,),
        "fp8": (torch.uint8, dtypes.fp8),
    }
    if a_dtype not in operand_dtypes:
        raise ValueError(f"unsupported a_dtype {a_dtype!r}")
    if b_dtype not in ("fp4", "fp8"):
        raise ValueError(f"unsupported b_dtype {b_dtype!r}")
    for name, t, dtype in (("A", A, a_dtype), ("B", B, b_dtype)):
        if t.dtype not in operand_dtypes[dtype]:
            raise ValueError(
                f"{name} has incompatible storage dtype {t.dtype} for {dtype}"
            )
    if Out.dtype not in _OUT_DTYPE_STR:
        raise ValueError(
            f"unsupported Out dtype {Out.dtype}; expected bfloat16 or float16"
        )
    if Out.shape[0] != A.shape[0]:
        raise ValueError(
            f"Out must be [M, N] with M=A.shape[0]={A.shape[0]}; got {tuple(Out.shape)}"
        )

    M = int(A.shape[0])
    N = int(Out.shape[-1])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    if min(M, N, K) <= 0:
        raise ValueError(f"M, N, and K must be positive, got {M}, {N}, {K}")
    if N % 32 != 0:
        raise ValueError(f"N ({N}) must be a multiple of 32 for MFMA-tile scales")
    if K % 128 != 0:
        raise ValueError(f"K ({K}) must be a multiple of 128 for MX microscale")
    expected_b_shape = (N, K // 2 if b_dtype == "fp4" else K)
    if tuple(B.shape) != expected_b_shape:
        raise ValueError(f"B must have shape {expected_b_shape}, got {tuple(B.shape)}")
    check_e8m0(a_scale, "a_scale", "FlyDSL mxscale preshuffle")
    check_e8m0(b_scale, "b_scale", "FlyDSL mxscale preshuffle")

    explicit_tiles = (tile_m, tile_n, tile_k)
    if any(value is not None for value in explicit_tiles) and not all(
        value is not None for value in explicit_tiles
    ):
        raise ValueError("tile_m, tile_n, and tile_k must be provided together")

    if tile_m is None:
        cfg = (
            config
            if config is not None
            else get_mxscale_preshuffle_config(
                M, N, K, a_dtype=a_dtype, b_dtype=b_dtype
            )
        )
        if cfg is not None:
            libtype = cfg.get("libtype")
            if isinstance(libtype, str):
                libtype = libtype.strip()
            parsed = mxscale_preshuffle_1x32_params(
                cfg.get("kernelName"),
                a_dtype=a_dtype,
                b_dtype=b_dtype,
                out_dtype=_OUT_DTYPE_STR[Out.dtype],
                libtype=libtype,
            )
            if parsed is not None:
                tile_m = parsed["tile_m"]
                tile_n = parsed["tile_n"]
                tile_k = parsed["tile_k"]
                if waves_per_eu is None:
                    waves_per_eu = parsed["waves_per_eu"]
                if xcd_swizzle is None:
                    xcd_swizzle = parsed["xcd_swizzle"]
                if split_k is None:
                    split_k = parsed["split_k"]
        if tile_m is None and require_tuned:
            raise RuntimeError(
                "no exact mxscale_preshuffle tune for "
                f"M={M}, N={N}, K={K}, a_dtype={a_dtype}, b_dtype={b_dtype}, "
                f"out_dtype={_OUT_DTYPE_STR[Out.dtype]}"
            )
        if tile_m is None:
            instance = _heuristic_tile(a_dtype, b_dtype, M, N, K, _MXFP8_MFMA_1X32)
            if instance is None:
                raise ValueError(
                    f"no legal tile for M={M} N={N} K={K} "
                    f"{a_dtype}/{b_dtype}; pass tile_m/n/k explicitly"
                )
            tile_m = instance.tile_m
            tile_n = instance.tile_n
            tile_k = instance.tile_k
            if waves_per_eu is None:
                waves_per_eu = instance.waves_per_eu
            if xcd_swizzle is None:
                xcd_swizzle = instance.xcd_swizzle
            if split_k is None:
                split_k = instance.split_k

    validate_mxscale_preshuffle_1x32_launch(
        int(tile_m),
        int(tile_n),
        int(tile_k),
        1 if split_k is None else int(split_k),
        a_dtype,
    )

    expected_a_scale = _mx_scale_nbytes(M, K)
    expected_b_scale = _mx_scale_nbytes(N, K)
    if a_scale.numel() * a_scale.element_size() != expected_a_scale:
        raise ValueError(
            f"a_scale nbytes {a_scale.numel() * a_scale.element_size()} != "
            f"{_mx_mfma_tile_1x32_scale_shape(M, K)} ({expected_a_scale} bytes)"
        )
    if b_scale.numel() * b_scale.element_size() != expected_b_scale:
        raise ValueError(
            f"b_scale nbytes {b_scale.numel() * b_scale.element_size()} != "
            f"{_mx_mfma_tile_1x32_scale_shape(N, K)} ({expected_b_scale} bytes)"
        )

    return flydsl_mxscale_preshuffle_gemm(
        A,
        B,
        a_scale,
        b_scale,
        Out,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        waves_per_eu=0 if waves_per_eu is None else waves_per_eu,
        xcd_swizzle=0 if xcd_swizzle is None else xcd_swizzle,
        split_k=1 if split_k is None else split_k,
        splitk_workspace=splitk_workspace,
        stream=stream,
        **dict(_MXFP8_MFMA_1X32),
    )


def run_gemm_a8w8_mxscale_preshuffle_1x32_gfx950(
    A,
    B,
    a_scale,
    b_scale,
    Out,
    kernel_name,
    *,
    splitk_workspace=None,
    stream=None,
):
    """gfx950 FP8 GEMM with pre-shuffled per-1x32 E8M0 scales, into ``Out``.

    A is row-major fp8. B is ``shuffle_weight(..., (16, 16))``. Both scales are
    ``shuffle_scale_a16w4`` buffers. ``kernel_name`` is a tuned mxpsh name.
    """
    from .gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        mxscale_preshuffle_1x32_params,
    )

    out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
    if out_dtype is None:
        raise ValueError(f"unsupported Out dtype {Out.dtype}")
    parsed = mxscale_preshuffle_1x32_params(
        kernel_name, a_dtype="fp8", b_dtype="fp8", out_dtype=out_dtype
    )
    if parsed is None:
        raise ValueError(f"{kernel_name!r} is incompatible with Out dtype {Out.dtype}")
    return gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
        A,
        B,
        a_scale,
        b_scale,
        Out,
        a_dtype=parsed["a_dtype"],
        b_dtype=parsed["b_dtype"],
        tile_m=parsed["tile_m"],
        tile_n=parsed["tile_n"],
        tile_k=parsed["tile_k"],
        waves_per_eu=parsed["waves_per_eu"],
        xcd_swizzle=parsed["xcd_swizzle"],
        split_k=parsed["split_k"],
        splitk_workspace=splitk_workspace,
        stream=stream,
    )
