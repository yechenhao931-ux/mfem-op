// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_CPU_KERNELS_HPP
#define MFEM_CPU_KERNELS_HPP

// -----------------------------------------------------------------------------
// Host-only kernels optimized for serial / shared-memory CPU execution.
//
// The MFEM device-portable kernels (in linalg/kernels.hpp and the
// mfem::forall machinery) target both CPU and GPU and therefore avoid
// host-specific pragmas. The kernels in this header are dedicated CPU paths
// that combine three optimization layers:
//
//   1. Thread-level parallelism via OpenMP `parallel for`
//      (only emitted when the library is built with MFEM_USE_OPENMP).
//   2. Data-level parallelism via `omp simd` reductions and aligned
//      contiguous-stride loops, which let modern auto-vectorizers emit
//      AVX2 / AVX-512 / NEON / SVE FMA instructions.
//   3. Cache-friendly access patterns: row-static partitioning for CSR
//      SpMV and column-major streaming for dense GEMV.
//
// They are entry points used by SparseMatrix and DenseMatrix when the
// data lives in host memory and the active backend is the CPU. The
// device path is left untouched.
// -----------------------------------------------------------------------------

#include "../config/config.hpp"
#include <cmath>
#include <cstddef>

#ifdef MFEM_USE_OPENMP
#include <omp.h>
#endif

// Portable "restrict" qualifier for the host kernels. Avoid colliding with
// MFEM_CPU_RESTRICT used elsewhere — these are internal to this header.
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define MFEM_CPU_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define MFEM_CPU_RESTRICT __restrict
#else
#define MFEM_CPU_RESTRICT
#endif

namespace mfem
{
namespace cpu_kernels
{

// Threshold below which it is not worth spinning up OpenMP threads. Using
// threads on tiny problems is dominated by fork/join overhead.
inline constexpr int omp_threshold = 512;

// ---------------------------------------------------------------------------
// CSR SpMV: y += a * A * x
// ---------------------------------------------------------------------------
inline void CSRAddMult(const int height,
                       const int *MFEM_CPU_RESTRICT Ip,
                       const int *MFEM_CPU_RESTRICT Jp,
                       const real_t *MFEM_CPU_RESTRICT Ap,
                       const real_t *MFEM_CPU_RESTRICT xp,
                       real_t *MFEM_CPU_RESTRICT yp,
                       const real_t a)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for schedule(static) if (height >= omp_threshold)
#endif
   for (int i = 0; i < height; i++)
   {
      const int rb = Ip[i];
      const int re = Ip[i+1];
      real_t d = 0.0;
      // Inner loop has an indirect gather A[j]*x[J[j]]; modern compilers
      // emit vgather (AVX-512) or scalarize cleanly. The simd hint also
      // unrolls and exposes FMA pipelines.
      #if defined(MFEM_USE_OPENMP)
      #pragma omp simd reduction(+:d)
      #endif
      for (int j = rb; j < re; j++)
      {
         d += Ap[j] * xp[Jp[j]];
      }
      yp[i] += a * d;
   }
}

// ---------------------------------------------------------------------------
// CSR transpose SpMV: y += a * A^T * x
// Race-free per-thread accumulation buffers, reduced at the end.
// ---------------------------------------------------------------------------
inline void CSRAddMultTranspose(const int height,
                                const int width,
                                const int *MFEM_CPU_RESTRICT Ip,
                                const int *MFEM_CPU_RESTRICT Jp,
                                const real_t *MFEM_CPU_RESTRICT Ap,
                                const real_t *MFEM_CPU_RESTRICT xp,
                                real_t *MFEM_CPU_RESTRICT yp,
                                const real_t a)
{
#if defined(MFEM_USE_OPENMP)
   if (height >= omp_threshold)
   {
      const int nthreads = omp_get_max_threads();
      // Per-thread scratch in a single contiguous buffer. We avoid std::vector
      // here so the dependency stays on stack-friendly allocation.
      real_t *scratch = new real_t[(std::size_t)nthreads * width]();
      #pragma omp parallel
      {
         const int tid = omp_get_thread_num();
         real_t *yloc = scratch + (std::size_t)tid * width;
         #pragma omp for schedule(static) nowait
         for (int i = 0; i < height; i++)
         {
            const real_t xi = a * xp[i];
            const int rb = Ip[i];
            const int re = Ip[i+1];
            for (int j = rb; j < re; j++)
            {
               yloc[Jp[j]] += Ap[j] * xi;
            }
         }
         // Reduce all thread-local buffers into yp. Parallelize across the
         // output dimension; each thread owns a disjoint column range.
         #pragma omp barrier
         #pragma omp for schedule(static)
         for (int k = 0; k < width; k++)
         {
            real_t s = 0.0;
            for (int t = 0; t < nthreads; t++)
            {
               s += scratch[(std::size_t)t * width + k];
            }
            yp[k] += s;
         }
      }
      delete [] scratch;
      return;
   }
#endif
   // Serial fallback. The inner loop has a write dependency on yp[Jp[j]],
   // so we cannot use omp simd here, but auto-vectorization for the
   // gather-scatter is still worthwhile on hot caches.
   for (int i = 0; i < height; i++)
   {
      const real_t xi = a * xp[i];
      const int rb = Ip[i];
      const int re = Ip[i+1];
      for (int j = rb; j < re; j++)
      {
         yp[Jp[j]] += Ap[j] * xi;
      }
   }
}

// ---------------------------------------------------------------------------
// CSR |A| * x: y += sum_j |A_ij| * x_j
// ---------------------------------------------------------------------------
inline void CSRAddAbsMult(const int height,
                          const int *MFEM_CPU_RESTRICT Ip,
                          const int *MFEM_CPU_RESTRICT Jp,
                          const real_t *MFEM_CPU_RESTRICT Ap,
                          const real_t *MFEM_CPU_RESTRICT xp,
                          real_t *MFEM_CPU_RESTRICT yp)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for schedule(static) if (height >= omp_threshold)
#endif
   for (int i = 0; i < height; i++)
   {
      const int rb = Ip[i];
      const int re = Ip[i+1];
      real_t d = 0.0;
      #if defined(MFEM_USE_OPENMP)
      #pragma omp simd reduction(+:d)
      #endif
      for (int j = rb; j < re; j++)
      {
         d += std::fabs(Ap[j]) * xp[Jp[j]];
      }
      yp[i] += d;
   }
}

// ---------------------------------------------------------------------------
// Vector AXPY: y[i] += a * x[i]
// ---------------------------------------------------------------------------
inline void Axpy(const int N, const real_t a,
                 const real_t *MFEM_CPU_RESTRICT x,
                 real_t *MFEM_CPU_RESTRICT y)
{
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd schedule(static) if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      y[i] += a * x[i];
   }
}

// ---------------------------------------------------------------------------
// Dot product: returns sum_i x[i]*y[i]
// ---------------------------------------------------------------------------
inline real_t Dot(const int N,
                  const real_t *MFEM_CPU_RESTRICT x,
                  const real_t *MFEM_CPU_RESTRICT y)
{
   real_t s = 0.0;
#if defined(MFEM_USE_OPENMP)
   #pragma omp parallel for simd reduction(+:s) schedule(static) \
      if (N >= omp_threshold)
#endif
   for (int i = 0; i < N; i++)
   {
      s += x[i] * y[i];
   }
   return s;
}

} // namespace cpu_kernels
} // namespace mfem

#endif // MFEM_CPU_KERNELS_HPP
