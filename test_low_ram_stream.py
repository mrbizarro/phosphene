"""v4.9.9: low-RAM block streaming — the helper streams transformer blocks
from disk on small-memory Macs. Pinned here: the per-job decision (LoRA jobs
keep the exact unfused branch and never stream), the pipeline cache key
carrying that decision, every t2v/i2v/extend constructor receiving it, and
the cache policy keeping the Metal cache off while streaming.

The helper is a script whose module body ends in a blocking stdin read, so
the functions under test are extracted with `ast` and exec'd, never
imported (same discipline as test_mlx_cache_policy.py)."""
import ast
import unittest
from pathlib import Path

HELPER = Path(__file__).with_name("mlx_warm_helper.py")
SRC = HELPER.read_text(encoding="utf-8")


def _funcs(names: set[str], low_ram_stream: bool) -> dict:
    tree = ast.parse(SRC)
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in picked} == names, f"missing {names - {n.name for n in picked}}"
    ns: dict = {"LOW_RAM_STREAM": low_ram_stream, "os": __import__("os")}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(HELPER), "exec"), ns)
    return ns


class StreamingIsPerJob(unittest.TestCase):
    def test_lora_free_jobs_stream_only_when_enabled(self):
        on = _funcs({"_stream_for"}, True)
        off = _funcs({"_stream_for"}, False)
        self.assertTrue(on["_stream_for"]([]))
        self.assertTrue(on["_stream_for"](None))
        self.assertFalse(off["_stream_for"]([]))

    def test_lora_jobs_never_stream(self):
        on = _funcs({"_stream_for"}, True)
        self.assertFalse(on["_stream_for"]([{"path": "/x/a.safetensors", "strength": 1.0}]))

    def test_cache_key_carries_the_decision(self):
        ns = _funcs({"_stream_for", "_lora_fingerprint", "_lora_fingerprint_base"}, True)
        self.assertIn("stream", ns["_lora_fingerprint"]([]))
        self.assertNotIn("stream", ns["_lora_fingerprint"]([{"path": "/x/a.safetensors", "strength": 1.0}]))
        ns_off = _funcs({"_stream_for", "_lora_fingerprint", "_lora_fingerprint_base"}, False)
        self.assertNotIn("stream", ns_off["_lora_fingerprint"]([]))


class EveryPipelineGetsTheFlag(unittest.TestCase):
    def test_t2v_i2v_extend_constructors_pass_low_ram_streaming(self):
        self.assertGreaterEqual(SRC.count("low_ram_streaming=_stream_for(loras)"), 3)
        for ctor in ("ImageToVideoPipeline(", "ExtendPipeline(", "TextToVideoPipeline("):
            i = SRC.index("pipe = " + ctor)
            self.assertIn("low_ram_streaming=_stream_for(loras)", SRC[i:i + 400], ctor)

    def test_cache_policy_keeps_the_cache_off_while_streaming(self):
        i = SRC.index("def apply_mlx_cache_policy")
        body = SRC[i:i + 2500]
        self.assertIn("if LOW_RAM_STREAM:", body)
        self.assertIn("mx.set_cache_limit(0)", body)


if __name__ == "__main__":
    unittest.main()
