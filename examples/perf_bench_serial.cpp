//                       MFEM Serial Performance Benchmark
//
// Compile with: make perf_bench_serial
//
// Sample runs:
//    ./perf_bench_serial -m ../data/inline-quad.mesh -r 6
//    ./perf_bench_serial -m ../data/star.mesh -r 4 -reps 200
//
// Description:  Builds a Poisson stiffness matrix and exercises the optimized
//               host CPU kernels (SpMV, transpose SpMV, AXPY, dot product).
//               Reports timings to demonstrate the benefit of the SIMD +
//               OpenMP CPU kernels added in linalg/cpu_kernels.hpp. When
//               built without OpenMP, the kernels still benefit from the
//               restrict qualifier, simd hints and tighter loop bodies.

#include "mfem.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace mfem;

namespace
{
struct Timer
{
   using clock = std::chrono::steady_clock;
   clock::time_point t0;
   void start() { t0 = clock::now(); }
   double stop()
   {
      auto t1 = clock::now();
      return std::chrono::duration<double>(t1 - t0).count();
   }
};
}

int main(int argc, char *argv[])
{
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_levels = 5;
   int order = 1;
   int reps = 100;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of uniform refinement steps.");
   args.AddOption(&order, "-o", "--order", "Polynomial order.");
   args.AddOption(&reps, "-reps", "--repetitions",
                  "Repetitions for timing each kernel.");
   args.Parse();
   if (!args.Good()) { args.PrintUsage(cout); return 1; }
   args.PrintOptions(cout);

   // Build mesh and FE space
   Mesh mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_levels; l++) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();

   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);
   const int N = fes.GetTrueVSize();
   cout << "DOFs: " << N << endl;

   // Assemble Poisson stiffness matrix in CSR form
   ConstantCoefficient one(1.0);
   BilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   a.Assemble();
   a.Finalize();
   const SparseMatrix &A = a.SpMat();
   cout << "nnz: " << A.NumNonZeroElems() << endl;

   // Random RHS vectors
   Vector x(N), y(N), z(N);
   for (int i = 0; i < N; i++)
   {
      x(i) = std::sin(0.001 * i) + 0.3;
      y(i) = 0.0;
      z(i) = std::cos(0.0007 * i);
   }
   x.UseDevice(false);
   y.UseDevice(false);
   z.UseDevice(false);

   Timer t;

   // Warm up
   A.Mult(x, y);

   // SpMV
   t.start();
   for (int r = 0; r < reps; r++) { A.Mult(x, y); }
   double t_mult = t.stop();

   // Transpose SpMV (force the explicit fallback by skipping pre-built At)
   t.start();
   for (int r = 0; r < reps; r++) { A.AddMultTranspose(x, z); }
   double t_multT = t.stop();

   // AXPY: y += a * z
   t.start();
   for (int r = 0; r < reps; r++) { y.Add(1.234e-3, z); }
   double t_axpy = t.stop();

   // Dot
   real_t s = 0.0;
   t.start();
   for (int r = 0; r < reps; r++) { s += (x * z); }
   double t_dot = t.stop();

   const double nnz = A.NumNonZeroElems();
   const double gflops_spmv = 2.0 * nnz * reps / t_mult / 1e9;
   const double gflops_axpy = 2.0 * N * reps / t_axpy / 1e9;
   const double gflops_dot  = 2.0 * N * reps / t_dot / 1e9;

   cout << fixed << setprecision(4);
   cout << "------------------------------------------------------------\n";
   cout << "Operation         time (s)    GFLOP/s    note\n";
   cout << "------------------------------------------------------------\n";
   cout << "SpMV   (y=Ax)     " << t_mult  << "    " << gflops_spmv
        << "    reps=" << reps << "\n";
   cout << "SpMV^T (z+=A^T x) " << t_multT << "\n";
   cout << "AXPY   (y+=a*z)   " << t_axpy  << "    " << gflops_axpy << "\n";
   cout << "Dot    (x . z)    " << t_dot   << "    " << gflops_dot
        << "    accum=" << s << "\n";
   cout << "------------------------------------------------------------\n";

   return 0;
}
