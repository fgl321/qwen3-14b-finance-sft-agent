from types import SimpleNamespace

from app.rag import embedding_factory


def _fake_settings(
    dense_size: int = 1024,
):
    return SimpleNamespace(
        embedding_provider="fake",
        rag_dense_vector_size=dense_size,
    )


def _bge_settings(
    model_name: str = "fake-bge-model",
):
    return SimpleNamespace(
        embedding_provider="bge-m3",
        rag_dense_vector_size=1024,
        bge_m3_model_name=model_name,
        bge_m3_batch_size=4,
        bge_m3_max_length=8192,
        bge_m3_device=None,
        bge_m3_use_fp16=True,
    )


def test_fake_embedding_provider_should_be_cached():
    embedding_factory.clear_embedding_provider_cache()

    provider_1 = embedding_factory.build_embedding_provider(
        _fake_settings()
    )

    provider_2 = embedding_factory.build_embedding_provider(
        _fake_settings()
    )

    assert provider_1 is provider_2

    embedding_factory.clear_embedding_provider_cache()


def test_different_fake_embedding_config_should_not_share_cache():
    embedding_factory.clear_embedding_provider_cache()

    provider_1 = embedding_factory.build_embedding_provider(
        _fake_settings(
            dense_size=128,
        )
    )

    provider_2 = embedding_factory.build_embedding_provider(
        _fake_settings(
            dense_size=256,
        )
    )

    assert provider_1 is not provider_2

    embedding_factory.clear_embedding_provider_cache()


def test_bge_embedding_provider_should_only_be_constructed_once(
    monkeypatch,
):
    embedding_factory.clear_embedding_provider_cache()

    created_count = {
        "value": 0,
    }

    class FakeBgeM3EmbeddingProvider:
        def __init__(
            self,
            *,
            model_name_or_path,
            use_fp16,
            batch_size,
            max_length,
            device,
        ) -> None:
            created_count["value"] += 1
            self.model_name_or_path = model_name_or_path
            self.use_fp16 = use_fp16
            self.batch_size = batch_size
            self.max_length = max_length
            self.device = device

    monkeypatch.setattr(
        embedding_factory,
        "BgeM3EmbeddingProvider",
        FakeBgeM3EmbeddingProvider,
    )

    provider_1 = embedding_factory.build_embedding_provider(
        _bge_settings()
    )

    provider_2 = embedding_factory.build_embedding_provider(
        _bge_settings()
    )

    assert provider_1 is provider_2
    assert created_count["value"] == 1

    embedding_factory.clear_embedding_provider_cache()
