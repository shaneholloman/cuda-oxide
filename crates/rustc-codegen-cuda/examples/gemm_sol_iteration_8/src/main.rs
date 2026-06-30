/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(
    clippy::not_unsafe_ptr_arg_deref,
    clippy::needless_range_loop,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]

//! Frozen cuda-oxide example of gemm-sol-autocuda iteration 8.
//!
//! The example contains one device kernel:
//! `gemm_sol_clc_multicast_4_stage_pipeline`. It uses CLC work scheduling,
//! `cta_group::2` UMMA, a four-stage shared-memory pipeline, compiler-directed
//! loop unrolling, and L2-aware output-tile ordering.
//!
//! Data layout:
//! - A: M×K f16, row-major (K contiguous)
//! - B: N×K f16, row-major (transposed storage, K contiguous)
//! - C: M×N bf16 output, row-major (packed as u32 pairs)

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::barrier::{
    Barrier, fence_proxy_async_shared_cta, mbarrier_arrive, mbarrier_arrive_cluster,
    mbarrier_arrive_expect_tx, mbarrier_init, mbarrier_inval, mbarrier_try_wait_parity,
};
use cuda_device::clc::{
    clc_query_get_first_ctaid_x, clc_query_is_canceled, clc_try_cancel_multicast,
};
use cuda_device::cluster;
use cuda_device::shared::SharedArray;
use cuda_device::tcgen05::{
    Tcgen05AccumulatorType, Tcgen05ElementType, Tcgen05InstructionDescriptor, Tcgen05MmaShape,
    cvt_f32x2_bf16x2, stmatrix_m8n8_x2, tcgen05_alloc_cg2, tcgen05_commit_multicast_cg2,
    tcgen05_dealloc_cg2, tcgen05_ld_16x256b_pure, tcgen05_load_wait, tcgen05_mma_f16_cg2,
    tcgen05_relinquish_alloc_permit_cg2,
};
use cuda_device::tma::{TmaDescriptor, cp_async_bulk_tensor_2d_g2s_multicast_cg2};
use cuda_device::{DisjointSlice, cluster_launch, kernel, thread, warp};
use cuda_host::cuda_module;
use half::{bf16, f16};
use std::mem::MaybeUninit;
use std::sync::Arc;

// =============================================================================
// LIVE cuBLASLt BASELINE (replaces the previous hardcoded B200 constants)
// =============================================================================

/// Live cuBLASLt FP16-input/FP16-output reference used to compute "% of SoL"
/// in benchmark reports.
///
/// The target kernel writes BF16, but the pinned cuBLASLt stack does not
/// expose an FP16-input/BF16-output matrix combination. Its FP16-output path is the
/// closest supported reference: it uses the same input type, FP32 compute, and
/// two-byte output width, while differing in the final 16-bit conversion. The
/// benchmark is always measured live on the same GPU.
mod cublas_baseline {
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::OnceLock;

    static BASELINE: OnceLock<Option<HashMap<usize, f64>>> = OnceLock::new();

    /// FP16-input / FP16-output / FP32-compute cuBLASLt TFLOPS for an M×M×M
    /// GEMM on the host GPU, or `None` if the bench could not be measured.
    pub fn reference_tflops(m: usize) -> Option<f64> {
        BASELINE.get_or_init(load).as_ref()?.get(&m).copied()
    }

    /// Pre-warm the baseline so the one-time measurement runs at startup, not in
    /// the middle of a benchmark print.
    pub fn warmup() {
        let _ = BASELINE.get_or_init(load);
    }

    fn bench_binary() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bench")
            .join("cublaslt_bench")
    }

    fn load() -> Option<HashMap<usize, f64>> {
        let bin = bench_binary();
        if !bin.exists() {
            eprintln!(
                "ℹ️  No live cublasLt baseline at {} — % of SoL column will be omitted.",
                bin.display()
            );
            eprintln!(
                "    Build it once with: cd {} && bash build.sh",
                bin.parent().unwrap_or(Path::new(".")).display(),
            );
            return None;
        }

        eprintln!("ℹ️  Measuring live cuBLASLt FP16 reference on host GPU...");
        let out = match std::process::Command::new(&bin).output() {
            Ok(o) => o,
            Err(e) => {
                eprintln!("⚠️  Failed to run {}: {e}", bin.display());
                return None;
            }
        };
        if !out.status.success() {
            eprintln!(
                "⚠️  {} exited with status {}: {}",
                bin.display(),
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            );
            return None;
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let map = parse_fp16_reference(&stdout);
        if map.is_empty() {
            eprintln!("⚠️  Could not parse FP16 rows from cublaslt_bench output:\n{stdout}");
            None
        } else {
            let mut sizes: Vec<(usize, f64)> = map.iter().map(|(m, t)| (*m, *t)).collect();
            sizes.sort_by_key(|(m, _)| *m);
            let pretty = sizes
                .iter()
                .map(|(m, t)| format!("{m}={:.1}", t))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!("✓ live cuBLASLt FP16 reference (TFLOPS): {pretty}");
            Some(map)
        }
    }

    /// Extract `(M, TFLOPS)` pairs from the `--- FP16 ---` section of
    /// `cublaslt_bench` output. Lines look like:
    ///
    /// ```text
    /// FP16 FP32 compute 16384x16384x16384 3.9000 ms 2255.0 TFLOPS
    /// ```
    fn parse_fp16_reference(s: &str) -> HashMap<usize, f64> {
        let mut map = HashMap::new();
        let mut in_fp16 = false;
        for line in s.lines() {
            let l = line.trim_start();
            if l.starts_with("--- FP16") {
                in_fp16 = true;
                continue;
            }
            if l.starts_with("--- BF16") {
                in_fp16 = false;
                continue;
            }
            if !in_fp16 || !l.starts_with("FP16 ") {
                continue;
            }
            // ["FP16", "FP32", "compute", "MxNxK", "X.XXXX", "ms",
            //  "Y.Y", "TFLOPS"]
            let toks: Vec<&str> = l.split_whitespace().collect();
            let size = toks.get(3).copied().unwrap_or("");
            let m: Option<usize> = size.split('x').next().and_then(|s| s.trim().parse().ok());
            // TFLOPS value is the second-to-last token (last is the literal "TFLOPS")
            let tf: Option<f64> = toks.iter().rev().nth(1).and_then(|s| s.parse().ok());
            if let (Some(m), Some(tf)) = (m, tf) {
                map.insert(m, tf);
            }
        }
        map
    }
}

/// Print the "vs cuBLAS" line for a benchmark result, using the live baseline
/// from `bench/cublaslt_bench` if available, otherwise an explanatory
/// placeholder. Replaces the previous hardcoded `match m { ... }` blocks
/// that compared every host GPU against B200's cublasLt SoL.
fn print_cublas_comparison(tflops: f64, m: usize) {
    match cublas_baseline::reference_tflops(m) {
        Some(sol) => {
            let pct = (tflops / sol) * 100.0;
            println!(
                "  vs cuBLAS:   {:.2}% of live cublasLt FP16 reference ({:.0} TFLOPS)",
                pct, sol
            );
        }
        None => {
            println!("  vs cuBLAS:   (no live cublasLt baseline; see bench/build.sh)");
        }
    }
}

fn print_benchmark_summary(measurements: &[(usize, f64)]) {
    let count = measurements.len() as f64;
    let kernel_geomean = (measurements
        .iter()
        .map(|(_, value)| value.ln())
        .sum::<f64>()
        / count)
        .exp();

    println!("── Fixed-size geomean summary ───────────────────────");
    println!("  Kernel:      {:.3} TFLOPS", kernel_geomean);

    let reference_values: Option<Vec<f64>> = measurements
        .iter()
        .map(|(m, _)| cublas_baseline::reference_tflops(*m))
        .collect();
    if let Some(reference_values) = reference_values {
        let reference_geomean =
            (reference_values.iter().map(|value| value.ln()).sum::<f64>() / count).exp();
        println!("  cuBLAS ref:  {:.3} TFLOPS", reference_geomean);
        println!(
            "  Ratio:       {:.2}% of live FP16 reference",
            kernel_geomean / reference_geomean * 100.0
        );
    } else {
        println!("  cuBLAS ref:  unavailable");
    }
    println!("─────────────────────────────────────────────────────");
}

// =============================================================================
// KERNEL
// =============================================================================

/// Build a tcgen05 SMEM descriptor from components.
///
/// Bit layout of the 64-bit descriptor:
///   [0:13]  base_addr >> 4
///   [16:29] LBO >> 4 (leading byte offset — stride to next core matrix RIGHT)
///   [32:45] SBO >> 4 (stride byte offset — stride to next core matrix DOWN)
///   [46]    fixed 0b1
///   [61:63] swizzle mode
#[inline(always)]
fn build_smem_descriptor(
    smem_addr: u64,
    leading_dim_bytes: u32,
    stride_bytes: u32,
    swizzle: u8,
) -> u64 {
    let addr_enc = (smem_addr >> 4) & 0x3FFF;
    let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
    let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
    let fixed_bit = 1u64 << 46;
    let swizzle_bits = (swizzle as u64) << 61;

    addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit | swizzle_bits
}

// Device kernels live in kernels.rs (the autocuda-editable surface), include!d
// here so #[cuda_module] sees an inline module with a brace body.
include!("kernels.rs");

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("═══════════════════════════════════════════════════════");
    println!("  GEMM SoL — gemm_sol_clc_multicast_4_stage_pipeline");
    println!("═══════════════════════════════════════════════════════\n");

    let mode = std::env::var("GEMM_SOL_MODE").unwrap_or_else(|_| "both".to_string());
    let (do_validate, do_bench) = match mode.as_str() {
        "validate" => (true, false),
        "bench" => (false, true),
        "both" => (true, true),
        other => {
            return Err(format!(
                "invalid GEMM_SOL_MODE={other:?}; expected validate, bench, or both"
            )
            .into());
        }
    };

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU: sm_{}{}", major, minor);

    if major < 10 {
        println!("\nWARNING: tcgen05 requires a compatible datacenter Blackwell GPU");
        return verify_ptx_only();
    }

    let ptx_path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("gemm_sol_iteration_8.ptx");
    println!("Loading PTX: {}", ptx_path.display());
    let ptx_str = ptx_path.to_str().ok_or("PTX path must be valid UTF-8")?;
    let module = match ctx.load_module_from_file(ptx_str) {
        Ok(module) => module,
        Err(error) if error.0 == cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_PTX => {
            println!("\nThe GPU/driver could not load the tcgen05 PTX.");
            println!(
                "PTX generation succeeded; execution requires a compatible sm_100+ datacenter Blackwell GPU."
            );
            return verify_ptx_only();
        }
        Err(error) => return Err(error.into()),
    };
    let module = kernels::from_module(module).expect("failed to initialize typed CUDA module");
    println!("PTX loaded\n");

    if do_validate {
        println!("── Full-output correctness test ─────────────────────\n");
        run_correctness_test_clc_multicast_4_stage_pipeline(&stream, &module, 4096, 4096, 4096)?;
    }

    if do_bench {
        cublas_baseline::warmup();
        println!("\n── Benchmarks ───────────────────────────────────────\n");
        let sizes = [
            (4096, 4096, 4096),
            (8192, 8192, 8192),
            (16384, 16384, 16384),
        ];
        let mut measurements = Vec::with_capacity(sizes.len());
        for (m, n, k) in sizes {
            let tflops = run_benchmark_clc_multicast_4_stage_pipeline(&stream, &module, m, n, k)?;
            measurements.push((m, tflops));
        }
        print_benchmark_summary(&measurements);
    }

    println!("\n═══════════════════════════════════════════════════════");
    println!("  GEMM SoL target complete");
    println!("═══════════════════════════════════════════════════════");
    Ok(())
}

// A 12-bit Walsh fingerprint gives every row and column in the fixed 4096²
// validator a unique whole-vector signature while retaining an O(MK + NK + MN)
// analytic reference. The odd affine multipliers are permutations mod 4096.
const VALIDATION_CODE_COUNT: usize = 1 << 12;
const VALIDATION_CODE_MASK: u32 = VALIDATION_CODE_COUNT as u32 - 1;

#[inline]
fn validation_affine12(x: usize, multiplier: u32, addend: u32) -> u32 {
    ((x as u32).wrapping_mul(multiplier).wrapping_add(addend)) & VALIDATION_CODE_MASK
}

#[inline]
fn validation_k_code(kk: usize) -> u32 {
    validation_affine12(kk, 251, 17)
}

#[inline]
fn validation_row_code(row: usize) -> u32 {
    validation_affine12(row, 197, 101)
}

#[inline]
fn validation_col_code(col: usize) -> u32 {
    validation_affine12(col, 109, 1021)
}

#[inline]
fn validation_fingerprint(position_code: u32, kk: usize) -> f32 {
    let distance = (position_code ^ validation_k_code(kk)).count_ones() as i32;
    (13 - 2 * distance) as f32
}

#[inline]
fn validation_a_value(row: usize, kk: usize) -> f32 {
    validation_fingerprint(validation_row_code(row), kk)
}

#[inline]
fn validation_b_value(col: usize, kk: usize) -> f32 {
    validation_fingerprint(validation_col_code(col), kk)
}

fn run_correctness_test_clc_multicast_4_stage_pipeline(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(
        [m, n, k],
        [VALIDATION_CODE_COUNT; 3],
        "the bit-exact Walsh validator is defined for the fixed 4096³ contract"
    );

    println!("Matrix: {}x{}x{} (f16 -> bf16)", m, n, k);
    println!("CLC + cta_group::2 + 4-stage SMEM pipeline.");
    println!("Warps: 4=TMA, 5=MMA (leader only), 0-3=epilogue (both CTAs).");

    let mut host_a: Vec<u16> = vec![0u16; m * k];
    for i in 0..m {
        for kk in 0..k {
            host_a[i * k + kk] = f16::from_f32(validation_a_value(i, kk)).to_bits();
        }
    }
    let mut host_b: Vec<u16> = vec![0u16; n * k];
    for j in 0..n {
        for kk in 0..k {
            host_b[j * k + kk] = f16::from_f32(validation_b_value(j, kk)).to_bits();
        }
    }

    // Walsh orthogonality gives C[row,col] = K * (13 - 2 * HammingDistance)
    // for each row/column code pair. Cache 4096 XOR-indexed expected entries.
    let expected_by_xor: Vec<u16> = (0..VALIDATION_CODE_COUNT)
        .map(|difference| {
            let score = 13 - 2 * (difference as u32).count_ones() as i64;
            let fp32 = (k as i64 * score) as f32;
            bf16::from_f32(fp32).to_bits()
        })
        .collect();

    println!(
        "Validation data: 12-bit Walsh fingerprints varying along K and covering \
         each K code once, with unique signatures for all {} rows and columns",
        VALIDATION_CODE_COUNT
    );
    println!("Expected: analytic exact FP32 dot product rounded once to BF16\n");

    let dev_a = DeviceBuffer::from_host(stream, &host_a)?;
    let dev_b = DeviceBuffer::from_host(stream, &host_b)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    // B TMA: 64 rows per CTA (each CTA loads half the N tile, split by rank)
    let b_tma = create_tma_descriptor_f16_swizzled_box(
        b_ptr as *mut std::ffi::c_void,
        k as u64,
        n as u64,
        64,
        64,
    )?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles * 2, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    println!(
        "Grid: {} CTAs (1D, CLC-managed), cluster: 2x1x1 (cg2), block: 192 threads (6 warps)",
        total_tiles * 2
    );
    println!(
        "Total tiles: {} ({}x{}), SWIZZLE_G=8 L2 cache-blocked ordering",
        total_tiles, tiles_m, tiles_n
    );
    println!("K-loop: {} outer iters (BK=64, 4 MMAs each)", k / 64);

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    println!("\nLaunching gemm_sol_clc_multicast_4_stage_pipeline (cg2)...");

    unsafe {
        module.gemm_sol_clc_multicast_4_stage_pipeline(
            (stream).as_ref(),
            cfg,
            a_tma_ptr,
            b_tma_ptr,
            &mut dev_output,
            n_arg,
            k_arg,
            tiles_m,
            tiles_n,
        )
    }?;

    stream.synchronize()?;

    let host_output: Vec<u32> = dev_output.to_host_vec(stream)?;

    let expected_bits = |row: usize, col: usize| -> u16 {
        expected_by_xor[(validation_row_code(row) ^ validation_col_code(col)) as usize]
    };
    let read_c_bits = |row: usize, col: usize| -> u16 {
        let packed = host_output[row * (n / 2) + col / 2];
        if col.is_multiple_of(2) {
            packed as u16
        } else {
            (packed >> 16) as u16
        }
    };

    // Full-output correctness check (anti reward-hack): compare EVERY BF16
    // element against the exact rounded reference, not a sampled subset. The
    // signed integer inputs make every FP32 product and partial sum exact, so a
    // correct kernel must match the reference BF16 bits.
    let mut mismatches: u64 = 0;
    let mut first_bad: Option<(usize, usize, f32, f32)> = None;
    let mut max_rel_err: f32 = 0.0;
    for row in 0..m {
        for col in 0..n {
            let val_bits = read_c_bits(row, col);
            let exp_bits = expected_bits(row, col);
            let val = bf16_to_f32(val_bits);
            let exp = bf16_to_f32(exp_bits);
            let diff = (val - exp).abs();
            let rel = diff / (exp.abs() + 1.0);
            if rel > max_rel_err {
                max_rel_err = rel;
            }
            if val_bits != exp_bits {
                mismatches += 1;
                if first_bad.is_none() {
                    first_bad = Some((row, col, val, exp));
                }
            }
        }
    }
    let total = (m * n) as u64;
    let all_ok = mismatches == 0;
    println!(
        "\nFull check: {} / {} exact BF16 matches (max rel err {:.4})",
        total - mismatches,
        total,
        max_rel_err
    );
    if let Some((r, c, val, exp)) = first_bad {
        println!(
            "  first mismatch: C[{},{}] = {} (expected {})",
            r, c, val, exp
        );
    }

    println!("\n═══════════════════════════════════════════════════════");
    if all_ok {
        println!(
            "PASSED: gemm_sol_clc_multicast_4_stage_pipeline {}x{}x{} (full {}-element check)",
            m, n, k, total
        );
    } else {
        println!(
            "FAILED: gemm_sol_clc_multicast_4_stage_pipeline {}x{}x{}: {} / {} elements wrong",
            m, n, k, mismatches, total
        );
        return Err(format!(
            "Correctness check failed: {} / {} elements wrong",
            mismatches, total
        )
        .into());
    }
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

fn run_benchmark_clc_multicast_4_stage_pipeline(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    m: usize,
    n: usize,
    k: usize,
) -> Result<f64, Box<dyn std::error::Error>> {
    const WARMUP: usize = 10;
    const ITERS: usize = 100;

    assert!(m.is_multiple_of(256) && n.is_multiple_of(128) && k.is_multiple_of(256));

    let dev_a = DeviceBuffer::<u16>::zeroed(stream, m * k)?;
    let dev_b = DeviceBuffer::<u16>::zeroed(stream, n * k)?;

    let a_ptr = dev_a.cu_deviceptr();
    let b_ptr = dev_b.cu_deviceptr();

    let a_tma =
        create_tma_descriptor_f16_swizzled(a_ptr as *mut std::ffi::c_void, k as u64, m as u64)?;
    let b_tma = create_tma_descriptor_f16_swizzled_box(
        b_ptr as *mut std::ffi::c_void,
        k as u64,
        n as u64,
        64,
        64,
    )?;

    let dev_a_tma = DeviceBuffer::from_host(stream, &a_tma.opaque)?;
    let dev_b_tma = DeviceBuffer::from_host(stream, &b_tma.opaque)?;
    let a_tma_ptr = dev_a_tma.cu_deviceptr();
    let b_tma_ptr = dev_b_tma.cu_deviceptr();
    let a_tma_ptr = a_tma_ptr as *const TmaDescriptor;
    let b_tma_ptr = b_tma_ptr as *const TmaDescriptor;
    let n_arg = n as i32;
    let k_arg = k as i32;

    let tiles_m = (m / 256) as u32;
    let tiles_n = (n / 128) as u32;
    let total_tiles = tiles_m * tiles_n;

    let cfg = LaunchConfig {
        grid_dim: (total_tiles * 2, 1, 1),
        block_dim: (192, 1, 1),
        shared_mem_bytes: 0,
    };

    let output_u32_count = m * n / 2;
    let mut dev_output = DeviceBuffer::<u32>::zeroed(stream, output_u32_count)?;

    for _ in 0..WARMUP {
        unsafe {
            module.gemm_sol_clc_multicast_4_stage_pipeline(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
                tiles_m,
                tiles_n,
            )
        }?;
    }
    stream.synchronize()?;

    let start_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;

    for _ in 0..ITERS {
        unsafe {
            module.gemm_sol_clc_multicast_4_stage_pipeline(
                (stream).as_ref(),
                cfg,
                a_tma_ptr,
                b_tma_ptr,
                &mut dev_output,
                n_arg,
                k_arg,
                tiles_m,
                tiles_n,
            )
        }?;
    }

    let end_event =
        stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let elapsed_ms = start_event.elapsed_ms(&end_event)?;

    let avg_ms = elapsed_ms as f64 / ITERS as f64;
    let avg_us = avg_ms * 1000.0;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = (flops / (avg_ms / 1000.0)) / 1e12;

    // cuBLAS comparison is printed by print_cublas_comparison(tflops, m) below.

    println!("═══════════════════════════════════════════════════════");
    println!(
        "  BENCHMARK: gemm_sol_clc_multicast_4_stage_pipeline (cg2) {}x{}x{} f16 -> bf16",
        m, n, k
    );
    println!("═══════════════════════════════════════════════════════");
    println!(
        "  Grid:        {} CTAs (1D, CLC-managed, cluster=2, cg2)",
        total_tiles * 2
    );
    println!("  K-loop:      {} outer iters (BK=64, 4 MMAs each)", k / 64);
    println!("  Pipeline:    CLC + cta_group::2 + 4-stage SMEM + unroll(4)");
    println!("  Iterations:  {} (after {} warmup)", ITERS, WARMUP);
    println!("  Total time:  {:.3} ms", elapsed_ms);
    println!("  Average:     {:.3} us / kernel", avg_us);
    println!("  FLOPS/kern:  {:.3e}", flops);
    println!("  Throughput:  {:.3} TFLOPS", tflops);
    print_cublas_comparison(tflops, m);
    println!("═══════════════════════════════════════════════════════\n");

    Ok(tflops)
}

fn verify_ptx_only() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("gemm_sol_iteration_8.ptx");

    if !ptx_path.exists() {
        return Err("PTX file not found".into());
    }

    println!("\nPTX Verification:");
    println!("   PTX file generated at: {}", ptx_path.display());
    println!("\n   To inspect generated PTX:");
    println!("   cat {}", ptx_path.display());

    Ok(())
}

/// Create a 2D TMA descriptor for f16 data with SWIZZLE_128B.
///
/// Single copy of 64 K-elements × 128 M/N rows per TMA instruction.
/// The TMA hardware applies a 128-byte XOR swizzle during the transfer.
fn create_tma_descriptor_f16_swizzled(
    global_address: *mut std::ffi::c_void,
    global_width: u64,
    global_height: u64,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2];
    let box_dim: [u32; 2] = [64, 128]; // 64 K-elements × 128 M/N rows
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(format!("cuTensorMapEncodeTiled (SWIZZLE_128B) failed: {:?}", result).into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

fn create_tma_descriptor_f16_swizzled_box(
    global_address: *mut std::ffi::c_void,
    global_width: u64,
    global_height: u64,
    box_k: u32,
    box_mn: u32,
) -> Result<cuda_core::sys::CUtensorMap, Box<dyn std::error::Error>> {
    use cuda_core::sys::{
        CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B, cuTensorMapEncodeTiled,
        cudaError_enum_CUDA_SUCCESS,
    };

    let mut tensor_map = MaybeUninit::<cuda_core::sys::CUtensorMap>::uninit();
    let global_dim: [u64; 2] = [global_width, global_height];
    let global_strides: [u64; 1] = [global_width * 2];
    let box_dim: [u32; 2] = [box_k, box_mn];
    let element_strides: [u32; 2] = [1, 1];

    let result = unsafe {
        cuTensorMapEncodeTiled(
            tensor_map.as_mut_ptr(),
            CUtensorMapDataType_enum_CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            2,
            global_address,
            global_dim.as_ptr(),
            global_strides.as_ptr(),
            box_dim.as_ptr(),
            element_strides.as_ptr(),
            CUtensorMapInterleave_enum_CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle_enum_CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion_enum_CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill_enum_CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    };

    if result != cudaError_enum_CUDA_SUCCESS {
        return Err(format!(
            "cuTensorMapEncodeTiled (SWIZZLE_128B, box {}x{}) failed: {:?}",
            box_k, box_mn, result
        )
        .into());
    }

    Ok(unsafe { tensor_map.assume_init() })
}

fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}
