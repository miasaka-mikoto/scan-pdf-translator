import rich_renderer as r
from unittest.mock import patch


def test_translation_json_fallback():
    calls = []
    bad = {
        "choices": [{"message": {"content": '{"translations":[{"id":"r01","text":"bad " quote"}]}'}}],
        "usage": {"total_tokens": 1},
    }
    good = {
        "choices": [{"message": {"content": "测试译文"}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
    }


    def fake_post(*_args, **_kwargs):
        calls.append(1)
        return bad if len(calls) == 1 else good


    with patch.object(r, "_post_json", fake_post):
        regions, usage = r.translate_regions("not-used", [r.Region("r01", "text", (0, 0, 1, 1), "source")], "Qwen/Qwen3-8B")
    assert regions[0].translated == "测试译文"
    assert usage["translation_fallback"] == "per_block_plain_text"
    assert usage["total_tokens"] == 5
    assert len(calls) == 2
    print("TRANSLATION_JSON_FALLBACK_OK")
