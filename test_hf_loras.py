"""A Hugging Face org as a LoRA source: lane from the repo name, the pretty
name, the catalog from the API (mocked), and the install with its sidecar."""
import json
from pathlib import Path

import mlx_ltx_panel as p


def test_lane_from_repo_name():
    assert p.hf_lora_lane_of_repo("Playtime-AI/Minimax_H3-Sydney_Sweeney") == "h3"
    assert p.hf_lora_lane_of_repo("Playtime-AI/Minimac_H3-Ana_de_Armas") == "h3"   # the org's typo
    assert p.hf_lora_lane_of_repo("Playtime-AI/LTX-2.3-Ted") == "ltx"
    assert p.hf_lora_lane_of_repo("Playtime-AI/Krea_2-Emma_Watson") is None


def test_pretty_names():
    assert p.hf_lora_pretty_name("Playtime-AI/Minimax_H3-The_Dude-Jeff_Bridges") == "The Dude — Jeff Bridges"
    assert p.hf_lora_pretty_name("Playtime-AI/LTX-2.3-DEV_AND_SULPHUR-Sofia_Vergara") == "Sofia Vergara"
    assert p.hf_lora_pretty_name("Playtime-AI/Minimax_H3-Megan_Fox") == "Megan Fox"


def test_query_kinds():
    assert p.hf_lora_query_params("h3", "") == ("search", "MiniMax H3 LoRA")
    assert p.hf_lora_query_params("h3", "megan") == ("search", "megan")
    assert p.hf_lora_query_params("h3", "author:Someone") == ("author", "Someone")
    assert p.hf_lora_query_params("h3", "https://huggingface.co/Someone") == ("author", "Someone")
    assert p.hf_lora_query_params("h3", "Someone/Minimax_H3-Thing") == ("repo", "Someone/Minimax_H3-Thing")


def test_catalog_lists_the_lane_with_previews(monkeypatch):
    def fake_get(path, timeout=20.0):
        if path.startswith("/api/models?author=") or path.startswith("/api/models?search="):
            return [{"id": "Playtime-AI/Minimax_H3-Megan_Fox", "likes": 37, "downloads": 0, "lastModified": "2026-08-29T00:00:00"},
                    {"id": "Playtime-AI/LTX-2.3-Ted", "likes": 7},
                    {"id": "Playtime-AI/Krea_2-Emma_Watson", "likes": 6},
                    {"id": "Playtime-AI/Minimax-H3_Showcase", "likes": 5}]
        if "Megan_Fox" in path:
            return {"siblings": [{"rfilename": "MM-H3 - Megan Fox.safetensors", "size": 155000000},
                                 {"rfilename": "example.mp4", "size": 20000000}, {"rfilename": "README.md", "size": 200}]}
        if "Showcase" in path:
            return {"siblings": [{"rfilename": "a.mp4", "size": 1}]}
        return {"siblings": []}
    monkeypatch.setattr(p, "_hf_api_get", fake_get)
    p._hf_lora_catalog_cache.clear()
    items = p.hf_lora_catalog("h3", "author:Playtime-AI", force=True)
    assert [i["name"] for i in items] == ["Megan Fox"]                # showcase has no weights; LTX/Krea are other lanes
    it = items[0]
    assert it["source"] == "huggingface" and it["lane"] == "h3" and it["base_model"] == "MiniMax H3"
    assert it["download_url"].endswith("/resolve/main/MM-H3%20-%20Megan%20Fox.safetensors")
    assert it["preview_url"].endswith("/resolve/main/example.mp4") and it["preview_type"] == "video"
    assert it["size_kb"] == 155000000 // 1024 and it["hf_url"] == "https://huggingface.co/Playtime-AI/Minimax_H3-Megan_Fox"


def test_download_lands_in_the_lane_with_a_sidecar(tmp_path, monkeypatch):
    h3dir = tmp_path / "h3loras"; h3dir.mkdir()
    monkeypatch.setattr(p, "_safe_h3_loras_dir", lambda: h3dir)
    monkeypatch.setattr(p, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(p, "_h3_lora_prepare", lambda t: {"layout": "bare", "converted": False, "prefix": ""})
    import huggingface_hub
    def fake_dl(repo_id, filename, token=None, local_dir=None):
        d = Path(local_dir); d.mkdir(parents=True, exist_ok=True)
        f = d / filename; f.write_bytes(b"0" * 4096); return str(f)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    meta = {"name": "Megan Fox", "lane": "h3", "preview_url": "https://huggingface.co/x/resolve/main/example.mp4",
            "preview_type": "video", "base_model": "MiniMax H3", "hf_url": "https://huggingface.co/Playtime-AI/Minimax_H3-Megan_Fox"}
    r = p._hf_lora_download("Playtime-AI/Minimax_H3-Megan_Fox", "MM-H3 - Megan Fox.safetensors", meta)
    assert r["ok"] and r["lane"] == "h3" and Path(r["path"]).parent == h3dir
    assert Path(r["path"]).name == "MM-H3_-_Megan_Fox.safetensors"
    side = json.loads(Path(r["sidecar_path"]).read_text())
    assert side["name"] == "Megan Fox" and side["source"] == "huggingface"
    assert side["hf_repo"] == "Playtime-AI/Minimax_H3-Megan_Fox" and side["preview_type"] == "video"
    assert side["base_model"] == "MiniMax H3" and side["lora_layout"] == "bare"
    # a second install of the same file is a skip, not a re-download
    r2 = p._hf_lora_download("Playtime-AI/Minimax_H3-Megan_Fox", "MM-H3 - Megan Fox.safetensors", meta)
    assert r2["skipped"] is True


def test_routes_registered():
    from panel import routes_loras  # noqa: F401
    from panel.routes import GET_ROUTES, POST_ROUTES
    assert "/hf/loras" in GET_ROUTES and "/hf/loras/download" in POST_ROUTES
