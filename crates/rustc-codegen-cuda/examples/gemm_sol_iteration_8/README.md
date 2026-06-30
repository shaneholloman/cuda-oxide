# gemm_sol_iteration_8

Frozen, runnable snapshot of the kernel accepted as **trial 0008** in the
`gemm-sol-autocuda` optimization loop. The source comes from commit
`6e91da250d6d3285968ee016562e51b3039c308e`.

The kernel is `gemm_sol_clc_multicast_4_stage_pipeline`, despite the historical
symbol name. This iteration widens each cooperative output tile to M256xN256
and combines:

- CLC work scheduling with a two-CTA cluster;
- `cta_group::2` pair-UMMA;
- four 32 KiB TMA/MMA pipeline stages;
- two 256-column TMEM accumulator stages;
- compiler-directed K-loop unrolling;
- size-adaptive L2 tile swizzling (`G=2` at 4096, `G=8` at larger sizes).

`src/kernels.rs` is copied byte-for-byte from the accepted iteration. The host
harness performs a full 16,777,216-element exact-BF16 validation at 4096³ and
benchmarks the fixed 4096³, 8192³, and 16384³ workloads.

## Run

From the cuda-oxide repository root:

```bash
GEMM_SOL_MODE=validate cargo oxide run gemm_sol_iteration_8
GEMM_SOL_MODE=bench cargo oxide run gemm_sol_iteration_8
```

`GEMM_SOL_MODE=both` is the default. The example requires Blackwell
`cta_group::2`/CLC support; on the B300 development system cargo-oxide detects
`sm_103a` automatically.

The live cuBLASLt comparison is optional. Build it once with:

```bash
cd crates/rustc-codegen-cuda/examples/gemm_sol_iteration_8/bench
bash build.sh
```

The accepted B300 checkpoint measured 1494.165, 1862.979, and 1949.843
TFLOPS, for a 1757.392 TFLOPS geomean. These numbers document provenance;
the example always reports fresh measurements on the host GPU.
