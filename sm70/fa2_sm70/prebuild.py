"""Compile the fa2_sm70 JIT extensions ahead of time, with or without a GPU.

    TORCH_CUDA_ARCH_LIST=7.0 FA2SM70_PREBUILD=1 FA2SM70_BUILD_DIR=/opt/vllm-sm70/build \\
        python -m fa2_sm70.prebuild                 # the production set
    python -m fa2_sm70.prebuild decode_ext clef_ext  # a subset

The default set is exactly what the production lever set loads (configs/qwen38-27b-int8-clef.env);
other extensions are compiled lazily on first use. torch rebuilds an extension whenever its sources,
flags, compiler or paths differ, so a prebuilt directory is reused only by the same image layout and
lever values -- otherwise it is recompiled in place, which is slower but correct.
"""
import os
import sys
import time

PRODUCTION = ["decode_ext", "prefill_ext", "gdn_ext", "w8a16_ext", "w12a16_ext", "tm8_ext", "exch_ext",
              "conv_ext", "i4_ext", "clef_ext"]


def main(names):
    # exch_ext does not go through _common(), which is what sets the arch list for the others
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    from fa2_sm70 import _ext

    bad = 0
    for n in names:
        t = time.time()
        try:
            getattr(_ext, n)()
            print(f"[prebuild] ok {n} ({time.time() - t:.0f} s)", flush=True)
        except Exception as ex:  # noqa: BLE001
            bad += 1
            print(f"[prebuild] FAILED {n}: {str(ex)[-2000:]}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in PRODUCTION if n != "clef_ext" or os.environ.get("FA2SM70_CLEF") != "0"]
    sys.exit(main(names))
