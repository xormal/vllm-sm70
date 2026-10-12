# Third-party code used by the tm8 extension

`sm70_turbomind/lmdeploy/` and `sm70_tile_runtime_signal.cuh` are the TurboMind GEMM headers that the
`tm8` extension (`sm70/fa2_sm70/csrc/tm8_gemm.cu`) is compiled against, in exactly the revision the
deployed binary was built from:

* TurboMind is part of [InternLM/LMDeploy](https://github.com/InternLM/lmdeploy) (Apache-2.0, see
  `sm70_turbomind/lmdeploy/LICENSE`).
* The sm_70 (s884) GEMM configuration, operand/mainloop/epilogue changes and
  `sm70_tile_runtime_signal.cuh` come from the [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 1.2.x
  line (`csrc/sm70_turbomind`, `csrc/sm70_tile_runtime_signal.cuh`, Apache-2.0). They were further modified in
  the xormal/1Cat-vLLM-fa2 fork (commit `25e1eb2fc`) and copied here unchanged from that tree.

The relative layout is preserved on purpose: `lmdeploy/src/turbomind/kernels/gemm/epilogue.h` includes
`../../../../../../sm70_tile_runtime_signal.cuh`. `fa2_sm70/_ext.py` uses
`sm70_turbomind/lmdeploy` as the include root unless `FA2SM70_TM_ROOT` is set.
