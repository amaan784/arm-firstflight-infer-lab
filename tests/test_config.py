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
