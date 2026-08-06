from firstflight import config


def test_models_load_and_smoke_default():
    models = config.load_models()
    assert models.default_smoke == "qwen2.5-0.5b-instruct"
    spec, variant = models.smoke()
    assert spec.id == "qwen2.5-0.5b-instruct"
    assert variant.file.endswith(".gguf")
    assert "huggingface.co" in variant.url


def test_instances_load_default():
    instances = config.load_instances()
    inst = instances.get()  # default
    assert inst.arch == "arm64"
    assert inst.usd_per_hour >= 0.0  # schema sanity; real dated prices live in instances.yaml


def test_workloads_prefill_scaling():
    workloads = config.load_workloads()
    wl = workloads.get("prefill-scaling")
    assert wl.prompt_lengths == sorted(wl.prompt_lengths)
    assert len(wl.prompt_lengths) >= 3  # a real sweep
    assert wl.repeats >= 1


def test_unknown_keys_raise():
    models = config.load_models()
    import pytest

    with pytest.raises(KeyError):
        models.get("does-not-exist")


def test_experiments_load():
    exps = config.load_experiments()
    assert exps.default_experiment == "quant-sweep"
    spec = exps.get("quant-sweep")
    assert spec.model
    assert "q4_0" in [c.label for c in spec.configs]


def test_experiment_bin_env_expansion(monkeypatch):
    monkeypatch.setenv("LLAMA_KLEIDIAI_BIN", "/opt/kleidiai-build")
    exps = config.load_experiments()
    bins = [c.bin for c in exps.get("kleidiai").configs]
    assert "/opt/kleidiai-build" in bins


def test_kv_cache_experiment_loads():
    exps = config.load_experiments()
    spec = exps.get("kv-cache")
    quantized = next(c for c in spec.configs if c.label == "kv-q8_0")
    assert quantized.cache_type_k == "q8_0" and quantized.cache_type_v == "q8_0"
    baseline = next(c for c in spec.configs if c.label == "kv-f16")
    assert baseline.cache_type_k == "" and baseline.cache_type_v == ""
    # FA must be pinned ON for both (quantized KV-cache without FA can be slower than f16)
    assert quantized.flash_attn == "on" and baseline.flash_attn == "on"


def test_prefill_batch_and_fa_experiments_load():
    exps = config.load_experiments()
    ubs = [c.ubatch_size for c in exps.get("prefill-batch").configs]
    assert ubs == [256, 512, 1024, 2048]
    fa = {c.label: c.flash_attn for c in exps.get("flash-attn").configs}
    assert fa == {"fa-off": "off", "fa-on": "on"}
