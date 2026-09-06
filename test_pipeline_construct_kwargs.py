"""A pipeline is constructed with the kwargs ITS class accepts — no more.

4.10.2 passed `low_ram_streaming=` to every pipeline. The vendored
RetakePipeline (Extend) has its own __init__ without it, so every Extend
render on 4.10.2–4.10.5 failed at construction. The fleet showed it as
"RetakePipeline.__init__() got an unexpected keyword argument
'low_ram_streaming'". Constructors go through `_construct_pipeline`, which
introspects the class the same way generate kwargs are already filtered.

The helper module is not imported (it talks to MLX at import); the two
functions are lifted from its source, the same discipline as
test_low_ram_stream.py.
"""
import ast, unittest
from pathlib import Path

SRC = Path(__file__).with_name("mlx_warm_helper.py").read_text(encoding="utf-8")


def _funcs(names):
    tree = ast.parse(SRC)
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in picked} == set(names), "helper renamed a function this test lifts"
    ns = {"emit": lambda e: ns.setdefault("_log", []).append(e)}
    exec(compile(ast.Module(body=picked, type_ignores=[]), "helper-extract", "exec"), ns)
    return ns


class _Strict:                      # RetakePipeline's shape: no low_ram_streaming, no **kwargs
    def __init__(self, model_dir, gemma_model_id="g", low_memory=True, dev_transformer="d"):
        self.args = dict(model_dir=model_dir, gemma_model_id=gemma_model_id,
                         low_memory=low_memory, dev_transformer=dev_transformer)


class _Streaming:                   # BasePipeline's shape
    def __init__(self, model_dir, gemma_model_id="g", low_memory=True, low_ram_streaming=False):
        self.args = dict(model_dir=model_dir, low_ram_streaming=low_ram_streaming)


class ConstructPipeline(unittest.TestCase):
    def setUp(self):
        self.ns = _funcs({"_construct_pipeline", "_filter_unsupported_kwargs"})
        self.build = self.ns["_construct_pipeline"]

    def test_extend_shape_drops_streaming_and_logs(self):
        p = self.build(_Strict, model_dir="m", gemma_model_id="g", low_memory=True,
                       low_ram_streaming=True, dev_transformer="dev.safetensors")
        self.assertEqual(p.args["dev_transformer"], "dev.safetensors")
        self.assertNotIn("low_ram_streaming", p.args)
        self.assertTrue(any("low_ram_streaming" in e.get("line", "") for e in self.ns["_log"]))

    def test_streaming_shape_keeps_the_flag(self):
        p = self.build(_Streaming, model_dir="m", gemma_model_id="g", low_memory=True,
                       low_ram_streaming=True)
        self.assertTrue(p.args["low_ram_streaming"])
        self.assertFalse(self.ns.get("_log"))

    def test_every_pipeline_construction_goes_through_the_filter(self):
        """No bare `SomethingPipeline(` construction left in get_pipe."""
        import re
        bare = [m.group(1) for m in re.finditer(r"^\s*pipe = (\w+Pipeline)\(", SRC, flags=re.M)
                if "low_ram_streaming" in SRC[m.start():m.start() + 400]]
        self.assertEqual(bare, [], f"passes low_ram_streaming without the kwargs filter: {bare}")
        for ctor in ("ImageToVideoPipeline", "ExtendPipeline", "TextToVideoPipeline"):
            m = re.search(r"_construct_pipeline\(\s*" + ctor + ",", SRC)
            self.assertIsNotNone(m, ctor)
            self.assertIn("low_ram_streaming=_stream_for(loras)", SRC[m.start():m.start() + 400], ctor)


if __name__ == "__main__":
    unittest.main()
