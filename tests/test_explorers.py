"""Smoke tests for explorer web UIs."""

import pytest


@pytest.fixture
def gemma_model():
    from mlux import HookedModel
    return HookedModel.from_pretrained("mlx-community/gemma-2-2b-it-4bit")


class TestLogitLensExplorer:
    def test_app_creates(self):
        from mlux.tools.logit_lens_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        assert app is not None

    def test_index_loads(self):
        from mlux.tools.logit_lens_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        with app.test_client() as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert b"logit lens" in resp.data

    def test_model_dropdown_has_defaults(self):
        from mlux.tools.logit_lens_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        with app.test_client() as client:
            resp = client.get("/")
            assert b"gemma-2-2b-it-4bit" in resp.data
            assert b"Llama-3.1-8B-Instruct-4bit" in resp.data
            assert b"Qwen2.5-7B-Instruct-4bit" in resp.data


class TestSteeringExplorer:
    def test_app_creates(self):
        from mlux.tools.contrastive_steering_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        assert app is not None

    def test_index_loads(self):
        from mlux.tools.contrastive_steering_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        with app.test_client() as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert b"steering explorer" in resp.data


class TestAblationExplorer:
    def test_app_creates(self):
        from mlux.tools.ablation_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        assert app is not None

    def test_index_loads(self):
        from mlux.tools.ablation_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        with app.test_client() as client:
            resp = client.get("/")
            assert resp.status_code == 200


class TestPatchingExplorer:
    def test_app_creates(self):
        from mlux.tools.patching_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        assert app is not None

    def test_index_loads(self):
        from mlux.tools.patching_explorer import create_app
        app = create_app("mlx-community/gemma-2-2b-it-4bit")
        with app.test_client() as client:
            resp = client.get("/")
            assert resp.status_code == 200
            assert b"activation patching" in resp.data
