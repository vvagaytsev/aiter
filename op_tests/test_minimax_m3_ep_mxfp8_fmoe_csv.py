# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EP4_TUNED = ROOT / "aiter/configs/model_configs/minimax_m3_ep4_mxfp8_tuned_fmoe.csv"
EP4_UNTUNED = ROOT / "aiter/configs/model_configs/minimax_m3_ep4_mxfp8_untuned_fmoe.csv"
EP1_TUNED = ROOT / "aiter/configs/model_configs/minimax_m3_ep1_mxfp8_tuned_fmoe.csv"
EP1_UNTUNED = ROOT / "aiter/configs/model_configs/minimax_m3_ep1_mxfp8_untuned_fmoe.csv"
EP1_TOKENS = {1 << exponent for exponent in range(18)}


def test_ep4_post2_rows_are_tuned_and_cover_requested_tokens():
    tuned = pd.read_csv(EP4_TUNED)
    untuned = pd.read_csv(EP4_UNTUNED)

    assert len(tuned) == 9
    assert set(tuned["_tag"]) == {"post2_tuned"}
    assert set(tuned["token"]) == {1, 2, 4, 8, 16, 32, 64, 8192, 16384}
    assert set(tuned["topk"]) == {4}
    assert tuned["kernelName1"].str.endswith("_fp8").all()
    assert set(untuned["token"]) == {1, 2, 4, 8, 16, 32, 64, 8192, 16384}
    assert set(untuned["topk"]) == {4}


def test_ep1_rows_are_tuned_and_cover_every_padded_bucket():
    tuned = pd.read_csv(EP1_TUNED)
    untuned = pd.read_csv(EP1_UNTUNED)

    assert len(tuned) == len(untuned) == 18
    assert set(tuned["token"]) == set(untuned["token"]) == EP1_TOKENS
    for frame in (tuned, untuned):
        assert set(frame["model_dim"]) == {6144}
        assert set(frame["inter_dim"]) == {768}
        assert set(frame["expert"]) == {128}
        assert set(frame["topk"]) == {4}
    assert set(tuned["_tag"]) == {"ep1_paired_validated_20260813"}
    assert tuned["kernelName1"].str.endswith("_fp8").all()
