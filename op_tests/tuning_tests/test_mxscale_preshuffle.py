# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch

from aiter.aot.flydsl import gemm as gemm_aot
from aiter.aot.flydsl.dry_run import dry_run
from aiter.aot.flydsl.replay import identified_kernels
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    instance_valid,
    kernelInstance,
    mxscale_preshuffle_1x32_params,
    parse_kernel_name,
)
from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
    _TUNED_CACHE,
    _heuristic_tile,
    _mx_mfma_tile_1x32_scale_shape,
    _mx_scale_nbytes,
    flydsl_mxscale_preshuffle_gemm,
    gemm_a8w8_mxscale_preshuffle_1x32_flydsl,
    get_mxscale_preshuffle_config,
    run_gemm_a8w8_mxscale_preshuffle_1x32_gfx950,
)
from aiter.ops.inverse_rope_group_quant import scale_shape
from aiter.ops.shuffle import shuffle_scale_a16w4
from aiter.utility import dtypes

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = (
    ROOT / "aiter/configs/model_configs/"
    "minimax_m3_dense_mxfp8_mxscale_preshuffle_tuned_gemm.csv"
)


def test_minimax_m3_dense_manifest_is_structurally_legal():
    frame = pd.read_csv(MODEL_CONFIG)
    assert len(frame) == 147
    assert set(frame["M"]) == {
        1,
        2,
        4,
        8,
        12,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        8320,
        16384,
    }
    assert len(frame.drop_duplicates(["M", "N", "K", "a_dtype", "b_dtype"])) == 147

    for row in frame.itertuples():
        parsed = parse_kernel_name(row.kernelName)
        assert parsed is not None
        assert not parsed["fused_split_k"]
        instance = kernelInstance(
            tile_m=parsed["tile_m"],
            tile_n=parsed["tile_n"],
            tile_k=parsed["tile_k"],
            a_dtype=parsed["a_dtype"],
            b_dtype=parsed["b_dtype"],
            out_dtype=parsed["out_dtype"],
            waves_per_eu=parsed["waves_per_eu"],
            xcd_swizzle=parsed["xcd_swizzle"],
            split_k=parsed["split_k"],
        )
        assert instance_valid(instance)
        assert int(row.splitK) == instance.split_k
        assert int(row.N) % instance.tile_n == 0
        assert int(row.K) % instance.tile_k == 0
        if instance.split_k > 1:
            k_per_split = int(row.K) // instance.split_k
            assert int(row.K) % instance.split_k == 0
            assert k_per_split % instance.tile_k == 0
            assert k_per_split % 256 == 0
            assert int(row.M) * int(row.N) * instance.split_k * 4 < (1 << 32)


def test_aot_parser_preserves_every_exact_signature():
    frame = pd.read_csv(MODEL_CONFIG)
    jobs = gemm_aot.parse_csv(str(MODEL_CONFIG))

    assert len(jobs) == len(frame) == 147
    assert all(job["m_low"] == job["m"] for job in jobs)
    assert {
        (job["m"], job["n"], job["k"], job["a_dtype"], job["b_dtype"]) for job in jobs
    } == {
        (int(row.M), int(row.N), int(row.K), row.a_dtype, row.b_dtype)
        for row in frame.itertuples()
    }


def test_aot_replays_the_public_entry():
    job = gemm_aot.parse_csv(str(MODEL_CONFIG))[0]
    with (
        mock.patch(
            "aiter.aot.flydsl.gemm._empty",
            side_effect=lambda *shape, dtype: torch.empty(
                shape, dtype=dtype, device="meta"
            ),
        ),
        mock.patch(
            "aiter.ops.flydsl.mxscale_preshuffle_kernels."
            "gemm_a8w8_mxscale_preshuffle_1x32_flydsl"
        ) as gemm,
    ):
        gemm_aot.replay(job)
    args, kwargs = gemm.call_args
    assert args[0].shape == (job["m"], job["k"])
    assert args[1].shape == (job["n"], job["k"])
    assert args[4].shape == (job["m"], job["n"])
    assert kwargs["require_tuned"] is True


@pytest.mark.parametrize(
    ("m", "n", "k", "expected_kernels"), [(1, 1536, 6144, 2), (1, 6144, 768, 1)]
)
def test_aot_replay_resolves_the_tuned_row_on_fake_tensors(m, n, k, expected_kernels):
    job = next(
        job
        for job in gemm_aot.parse_csv(str(MODEL_CONFIG))
        if (job["m"], job["n"], job["k"]) == (m, n, k)
    )
    _TUNED_CACHE.clear()
    try:
        with (
            dry_run("gfx950", 256, python_ops=True),
            identified_kernels() as kernels,
        ):
            gemm_aot.replay(job)
        assert len(kernels) == expected_kernels
    finally:
        _TUNED_CACHE.clear()


def test_shared_aot_driver_registers_mxscale_table():
    table = gemm_aot.table_of(str(MODEL_CONFIG))
    assert table is not None
    assert table.env == "AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE"
    assert gemm_aot._runtime_file(table) in gemm_aot.DEFAULT_CSVS


def test_runtime_lookup_is_exact_signature_only(tmp_path):
    row = pd.read_csv(MODEL_CONFIG).iloc[[0]]
    tune_file = tmp_path / "mxscale.csv"
    row.to_csv(tune_file, index=False)
    _TUNED_CACHE.clear()
    try:
        with (
            mock.patch(
                "aiter.jit.utils.chip_info.get_gfx_runtime",
                return_value="gfx950",
            ),
            mock.patch("aiter.jit.utils.chip_info.get_cu_num", return_value=256),
        ):
            exact = get_mxscale_preshuffle_config(
                int(row.iloc[0]["M"]),
                int(row.iloc[0]["N"]),
                int(row.iloc[0]["K"]),
                a_dtype="fp8",
                b_dtype="fp8",
                tuned_file=str(tune_file),
            )
            missing = get_mxscale_preshuffle_config(
                int(row.iloc[0]["M"]) + 1,
                int(row.iloc[0]["N"]),
                int(row.iloc[0]["K"]),
                a_dtype="fp8",
                b_dtype="fp8",
                tuned_file=str(tune_file),
            )
        assert exact is not None
        assert exact["kernelName"] == row.iloc[0]["kernelName"]
        assert missing is None
    finally:
        _TUNED_CACHE.clear()


def test_heuristic_keeps_split_k_one_when_the_grid_fills_the_cus():
    scales = (
        ("a_scale_block", "1x32"),
        ("a_scale_layout", "mfma_tile"),
        ("b_scale_block", "1x32"),
        ("b_scale_layout", "mfma_tile"),
    )
    instance = _heuristic_tile("fp8", "fp8", 8192, 6144, 6144, scales)
    assert instance is not None
    assert instance.split_k == 1


def test_unknown_kernel_name_is_not_parsed():
    assert parse_kernel_name("flydsl_mxpsh_not_a_tile") is None
    parsed = parse_kernel_name("flydsl_mxpsh_32x32x256_F6_F4_B16_w0_x0_sk1")
    assert parsed is not None
    assert parsed["a_dtype"] == "fp6"


def test_rejects_unknown_dtype():
    dummy = torch.empty((32, 256), dtype=torch.uint8)
    out = torch.empty((32, 64), dtype=torch.bfloat16)
    try:
        flydsl_mxscale_preshuffle_gemm(
            dummy,
            dummy,
            dummy,
            dummy,
            out,
            a_dtype="fp7",
            b_dtype="fp8",
            tile_m=32,
            tile_n=16,
            tile_k=256,
            a_scale_block="1x32",
            b_scale_block="1x32",
            a_scale_layout="mfma_tile",
            b_scale_layout="mfma_tile",
        )
    except ValueError as err:
        assert "fp7" in str(err)
        return
    raise AssertionError("unknown a_dtype must be rejected")


def test_per_1x32_rejects_blockscale_tile_m():
    M, N, K = 16, 128, 256
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty(_mx_scale_nbytes(M, K), dtype=torch.uint8)
    b_scale = torch.empty(_mx_scale_nbytes(N, K), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=torch.bfloat16)
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=16,
            tile_n=128,
            tile_k=256,
        )
    except ValueError as err:
        assert "multiple of 32" in str(err)
        return
    raise AssertionError("tile_m=16 must be rejected on the per-1x32 path")


def test_mx_scale_nbytes_matches_shuffle_scale_a16w4():
    for rows, k in ((1, 128), (17, 256), (31, 384), (32, 128), (33, 384), (64, 768)):
        padded_rows = (rows + 31) // 32 * 32
        raw = torch.empty((padded_rows, k // 32), dtype=torch.uint8)
        assert shuffle_scale_a16w4(raw, 1, False).numel() == _mx_scale_nbytes(rows, k)
        assert (
            _mx_mfma_tile_1x32_scale_shape(rows, k)
            == scale_shape(rows, 1, k // 32, "mfma_tile", quant_group_size=32)[1:]
        )


def test_rejects_unshuffled_scale_nbytes():
    M, N, K = 32, 64, 256
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty((1, K // 32), dtype=torch.uint8)
    b_scale = torch.empty((32, K // 32), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=torch.bfloat16)
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=32,
            tile_n=16,
            tile_k=256,
        )
    except ValueError as err:
        assert "a_scale nbytes" in str(err)
        return
    raise AssertionError("mismatched a_scale must be rejected")


def _uint8_mnk(M, N, K, out_dtype=torch.bfloat16):
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty(_mx_scale_nbytes(M, K), dtype=torch.uint8)
    b_scale = torch.empty(_mx_scale_nbytes(N, K), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=out_dtype)
    return A, B, a_scale, b_scale, out


def _run_explicit(tensors, **kwargs):
    return gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
        *tensors,
        a_dtype=kwargs.pop("a_dtype", "fp8"),
        b_dtype=kwargs.pop("b_dtype", "fp8"),
        tile_m=32,
        tile_n=16,
        tile_k=256,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tile_m": 0}, "tile_m must be positive"),
        ({"tile_n": 0}, "tile_n must be positive"),
        ({"tile_k": 0}, "tile_k must be positive"),
        ({"split_k": 0}, "split_k must be positive"),
        ({"tile_n": 80}, "whole 16-column tiles"),
        ({"tile_n": 192}, "row-layout scales"),
        ({"tile_n": 48, "tile_k": 128}, "incomplete A cooperative load"),
        ({"tile_k": 512}, "tile_k 128 or 256"),
    ],
)
def test_rejects_invalid_explicit_launch(override, message):
    params = {"tile_m": 32, "tile_n": 16, "tile_k": 256}
    params.update(override)
    with pytest.raises(ValueError, match=message):
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            *_uint8_mnk(32, 960, 512), a_dtype="fp8", b_dtype="fp8", **params
        )


@pytest.mark.parametrize("n", [16, 48])
def test_per_1x32_rejects_incomplete_scale_rows(n):
    with pytest.raises(ValueError, match="N .*multiple of 32"):
        _run_explicit(_uint8_mnk(32, n, 256))


@pytest.mark.parametrize(
    ("index", "shape", "message"),
    [
        (0, (32, 1, 256), "A must be a 2D tensor"),
        (1, (16, 256), "B must have shape"),
        (1, (64, 128), "B must have shape"),
        (4, (32, 0, 64), "Out must be a 2D tensor"),
        (4, (0, 64), "Out must be"),
    ],
)
def test_rejects_incompatible_operand_shapes(index, shape, message):
    tensors = list(_uint8_mnk(32, 64, 256))
    tensors[index] = torch.empty(shape, dtype=tensors[index].dtype)
    with pytest.raises(ValueError, match=message):
        _run_explicit(tensors)


@pytest.mark.parametrize(("index", "name"), [(0, "A"), (1, "B")])
def test_rejects_incompatible_operand_storage(index, name):
    tensors = list(_uint8_mnk(32, 64, 256))
    tensors[index] = torch.empty(tensors[index].shape, dtype=torch.float16)
    with pytest.raises(ValueError, match=f"{name} has incompatible storage dtype"):
        _run_explicit(tensors)


@pytest.mark.parametrize(
    ("index", "name"), [(1, "B"), (2, "a_scale"), (3, "b_scale"), (4, "Out")]
)
def test_rejects_mismatched_device(index, name):
    tensors = list(_uint8_mnk(32, 64, 256))
    tensors[index] = torch.empty_like(tensors[index], device="meta")
    with pytest.raises(ValueError, match=f"{name} must be on"):
        _run_explicit(tensors)


@pytest.mark.parametrize(("index", "name"), [(2, "a_scale"), (3, "b_scale")])
@pytest.mark.parametrize("bad_storage", ["strided", "wrong_dtype"])
def test_rejects_invalid_scale_storage(index, name, bad_storage):
    tensors = list(_uint8_mnk(32, 64, 256))
    size = tensors[index].numel()
    if bad_storage == "strided":
        tensors[index] = torch.empty(size * 2, dtype=torch.uint8)[::2]
        message = f"{name} must be contiguous"
    else:
        tensors[index] = torch.empty(size // 2, dtype=torch.int16)
        message = f"{name} must be e8m0/uint8"
    with pytest.raises(RuntimeError, match=message):
        _run_explicit(tensors)


@pytest.mark.parametrize("scale_dtype", [torch.uint8, dtypes.fp8_e8m0])
def test_accepts_valid_mfma_scale_storage(scale_dtype):
    tensors = list(_uint8_mnk(32, 64, 256))
    tensors[2] = torch.empty(tensors[2].shape, dtype=scale_dtype)
    tensors[3] = torch.empty(tensors[3].shape, dtype=scale_dtype)
    with mock.patch(
        "aiter.ops.flydsl.mxscale_preshuffle_kernels.flydsl_mxscale_preshuffle_gemm",
        return_value=tensors[4],
    ) as launch:
        assert _run_explicit(tensors) is tensors[4]
    launch.assert_called_once()


def test_accepts_valid_packed_fp4_operands():
    tensors = list(_uint8_mnk(32, 64, 256))
    tensors[0] = torch.empty((32, 128), dtype=torch.uint8)
    tensors[1] = torch.empty((64, 128), dtype=torch.uint8)
    with mock.patch(
        "aiter.ops.flydsl.mxscale_preshuffle_kernels.flydsl_mxscale_preshuffle_gemm",
        return_value=tensors[4],
    ):
        assert _run_explicit(tensors, a_dtype="fp4", b_dtype="fp4") is tensors[4]


def test_named_kernel_rejects_output_dtype_mismatch():
    tensors = _uint8_mnk(32, 64, 256, out_dtype=torch.float16)
    with pytest.raises(ValueError, match="incompatible with Out dtype"):
        run_gemm_a8w8_mxscale_preshuffle_1x32_gfx950(
            *tensors, "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk1"
        )


def test_runtime_and_aot_skip_non_flydsl_rows(tmp_path):
    row = pd.read_csv(MODEL_CONFIG).iloc[0].to_dict()
    row["libtype"] = "other"
    csv_path = tmp_path / "mxscale_preshuffle_tuned_gemm.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    assert gemm_aot.parse_csv(str(csv_path)) == []
    with pytest.raises(RuntimeError, match="no exact mxscale_preshuffle tune"):
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            *_uint8_mnk(32, 64, 256),
            a_dtype="fp8",
            b_dtype="fp8",
            config=row,
            require_tuned=True,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kernelName": "unknown"}, "not an mxpsh"),
        (
            {"kernelName": "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk2_fsk"},
            "fused split-K",
        ),
        ({"a_dtype": "fp4"}, "encodes"),
        ({"kernelName": ""}, "kernel name is required"),
        (
            {"kernelName": "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk1_pf"},
            "prefetch requires row-layout scales",
        ),
        (
            {"kernelName": "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk1_pf_nt"},
            "prefetch requires row-layout scales",
        ),
        (
            {"kernelName": "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk1_nt"},
            "prefetch requires row-layout scales",
        ),
        (
            {"kernelName": "flydsl_mxpsh_0x16x256_F8_F8_B16_w0_x0_sk1"},
            "tile_m must be positive",
        ),
        (
            {"kernelName": "flydsl_mxpsh_32x80x256_F8_F8_B16_w0_x0_sk1"},
            "whole 16-column tiles",
        ),
        (
            {"kernelName": "flydsl_mxpsh_32x192x256_F8_F8_B16_w0_x0_sk1"},
            "row-layout scales",
        ),
        (
            {"kernelName": "flydsl_mxpsh_32x48x128_F8_F8_B16_w0_x0_sk1"},
            "incomplete A cooperative load",
        ),
        (
            {"kernelName": "flydsl_mxpsh_32x16x512_F8_F8_B16_w0_x0_sk1"},
            "tile_k 128 or 256",
        ),
        (
            {"kernelName": "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk0"},
            "split_k must be positive",
        ),
        (
            {
                "a_dtype": "bf16",
                "kernelName": "flydsl_mxpsh_32x16x256_B16_F8_B16_w0_x0_sk1",
            },
            "unsupported a_dtype",
        ),
        (
            {
                "b_dtype": "bf16",
                "kernelName": "flydsl_mxpsh_32x16x256_F8_B16_B16_w0_x0_sk1",
            },
            "unsupported b_dtype",
        ),
    ],
)
def test_runtime_and_aot_reject_invalid_rows(tmp_path, change, message):
    row = pd.read_csv(MODEL_CONFIG).iloc[0].to_dict()
    row.update(change)
    csv_path = tmp_path / "mxscale_preshuffle_tuned_gemm.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match=message):
        gemm_aot.parse_csv(str(csv_path))
    tensors = list(_uint8_mnk(32, 64, 256))
    if row["a_dtype"] == "fp4":
        tensors[0] = torch.empty((32, 128), dtype=torch.uint8)
    with pytest.raises(ValueError, match=message):
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            *tensors,
            a_dtype=row["a_dtype"],
            b_dtype=row["b_dtype"],
            config=row,
            require_tuned=True,
        )


def test_runtime_lookup_surfaces_malformed_config(tmp_path):
    tune_file = tmp_path / "bad.csv"
    tune_file.write_text("invalid\n1\n")
    _TUNED_CACHE.clear()
    try:
        with pytest.raises(KeyError):
            get_mxscale_preshuffle_config(32, 64, 256, tuned_file=str(tune_file))
    finally:
        _TUNED_CACHE.clear()


def test_require_tuned_miss_raises():
    A, B, a_scale, b_scale, out = _uint8_mnk(32, 64, 256)
    with mock.patch(
        "aiter.ops.flydsl.mxscale_preshuffle_kernels.get_mxscale_preshuffle_config",
        return_value=None,
    ):
        try:
            gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
                A,
                B,
                a_scale,
                b_scale,
                out,
                a_dtype="fp8",
                b_dtype="fp8",
                require_tuned=True,
            )
        except RuntimeError as err:
            assert "no exact mxscale_preshuffle tune" in str(err)
            return
    raise AssertionError("require_tuned miss must raise")


def test_out_dtype_mismatch_is_tuned_miss():
    A, B, a_scale, b_scale, out = _uint8_mnk(32, 64, 256, out_dtype=torch.float16)
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            config={"kernelName": "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk1"},
            require_tuned=True,
        )
    except RuntimeError as err:
        assert "no exact mxscale_preshuffle tune" in str(err)
        assert "fp16" in str(err)
        return
    raise AssertionError("bf16 kernelName vs fp16 Out must be a tuned miss")


def test_rejects_noncontiguous_out():
    M, N, K = 32, 64, 256
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty(_mx_scale_nbytes(M, K), dtype=torch.uint8)
    b_scale = torch.empty(_mx_scale_nbytes(N, K), dtype=torch.uint8)
    out = torch.empty((M, N * 2), dtype=torch.bfloat16)[:, ::2]
    try:
        gemm_a8w8_mxscale_preshuffle_1x32_flydsl(
            A,
            B,
            a_scale,
            b_scale,
            out,
            a_dtype="fp8",
            b_dtype="fp8",
            tile_m=32,
            tile_n=16,
            tile_k=256,
        )
    except ValueError as err:
        assert "Out must be contiguous" in str(err)
        return
    raise AssertionError("noncontiguous Out must be rejected")


def test_rejects_bad_splitk_workspace():
    M, N, K = 32, 64, 512
    A = torch.empty((M, K), dtype=torch.uint8)
    B = torch.empty((N, K), dtype=torch.uint8)
    a_scale = torch.empty((1,), dtype=torch.uint8)
    b_scale = torch.empty((1,), dtype=torch.uint8)
    out = torch.empty((M, N), dtype=torch.bfloat16)
    bad = torch.empty((1, M, N), dtype=torch.float32)
    with (
        mock.patch(
            "aiter.ops.flydsl.mxscale_preshuffle_kernels._gemm_exe",
            return_value=object(),
        ),
        mock.patch(
            "aiter.ops.flydsl.kernels.tensor_shim.ptr_arg",
            side_effect=lambda t: t,
        ),
    ):
        try:
            flydsl_mxscale_preshuffle_gemm(
                A,
                B,
                a_scale,
                b_scale,
                out,
                a_dtype="fp8",
                b_dtype="fp8",
                tile_m=32,
                tile_n=16,
                tile_k=256,
                split_k=2,
                splitk_workspace=bad,
                stream=mock.Mock(),
                a_scale_block="1x32",
                b_scale_block="1x32",
                a_scale_layout="mfma_tile",
                b_scale_layout="mfma_tile",
            )
        except ValueError as err:
            assert "splitk_workspace" in str(err)
            return
    raise AssertionError("bad splitk_workspace must be rejected")


def test_shared_row_helper_rejects_fused_split_k():
    try:
        mxscale_preshuffle_1x32_params(
            "flydsl_mxpsh_32x16x256_F8_F8_B16_w0_x0_sk2_fsk",
            a_dtype="fp8",
            b_dtype="fp8",
        )
    except ValueError as err:
        assert "fused split-K" in str(err)
        return
    raise AssertionError("_fsk names must be rejected for mfma_tile 1x32")
