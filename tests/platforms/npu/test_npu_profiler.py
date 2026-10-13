# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import ModuleType, SimpleNamespace

import pytest

import vllm_omni.profiler.omni_torch_profiler as profiler_mod
from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("with_stack", [True, False])
def test_npu_profiler_honors_torch_profiler_with_stack(monkeypatch, tmp_path, with_stack):
    def fake_init(self, profiler_config):
        self.profiler_config = profiler_config

    monkeypatch.setattr(profiler_mod.WorkerProfiler, "__init__", fake_init)

    cpu_activity = object()
    npu_activity = object()
    experimental_config = object()
    captured = {}

    def fake_experimental_config(**kwargs):
        captured["experimental"] = kwargs
        return experimental_config

    def fake_profile(**kwargs):
        captured["profile"] = kwargs
        return SimpleNamespace()

    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU=cpu_activity, NPU=npu_activity),
        ExportType=SimpleNamespace(Text=object()),
        ProfilerLevel=SimpleNamespace(Level1=object()),
        AiCMetrics=SimpleNamespace(PipeUtilization=object()),
        _ExperimentalConfig=fake_experimental_config,
        profile=fake_profile,
        tensorboard_trace_handler=lambda path: path,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    NPUTorchProfilerWrapper(
        profiler_config=SimpleNamespace(
            torch_profiler_dir=str(tmp_path),
            torch_profiler_use_gzip=False,
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=False,
            torch_profiler_with_stack=with_stack,
        ),
        worker_name="worker0",
        local_rank=0,
    )

    assert captured["experimental"]["data_simplification"] is (not with_stack)
    profile_kwargs = captured["profile"]
    assert profile_kwargs["experimental_config"] is experimental_config
    assert profile_kwargs["with_stack"] is with_stack
    assert profile_kwargs["with_modules"] is with_stack
    assert profile_kwargs["record_shapes"] is True
    assert profile_kwargs["profile_memory"] is False
    assert profile_kwargs["activities"] == [cpu_activity, npu_activity]
