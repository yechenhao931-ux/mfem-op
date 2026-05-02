//                  MFEM Parallel (MPI) Performance Benchmark
//
// Build requirements:
//    MFEM_USE_MPI=YES, hypre, metis. From the build root:
//        make config MFEM_USE_MPI=YES MFEM_USE_OPENMP=YES \
//                    HYPRE_DIR=... METIS_DIR=...
//        make -j
//        cd examples && make perf_bench_par
//
// Sample runs:
//    mpirun -np 4 ./perf_bench_par -m ../data/inline-quad.mesh -r 6
//    mpirun -np 8 ./perf_bench_par -m ../data/inline-hex.mesh  -r 4 -o 2 -reps 50
//
// Description:  Distributed-memory upgrade of perf_bench_serial. Builds a
//               Poisson stiffness matrix on a partitioned ParMesh, exercises
//               the hot host-CPU paths (HypreParMatrix SpMV, AXPY, dot
//               product, conjugate-gradient solve) and reports per-rank and
//               aggregate timings. Combined with the SIMD/OpenMP host
//               kernels added in linalg/cpu_kernels.hpp this gives the
//               classic two-level (MPI x OpenMP) parallel decomposition.

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
   double stop() const
   {
      return std::chrono::duration<double>(clock::now() - t0).count();
   }
};
}

int main(int argc, char *argv[])
{
   // 1. MPI + HYPRE init.
   Mpi::Init(argc, argv);
   const int rank = Mpi::WorldRank();
   const int nranks = Mpi::WorldSize();
   const bool root = (rank == 0);
   Hypre::Init();

   // 2. Options.
   const char *mesh_file = "../data/inline-quad.mesh";
   int ref_serial = 2;
   int ref_parallel = 4;
   int order = 1;
   int reps = 100;
   bool run_solve = true;
   const char *device_config = "cpu";

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file.");
   args.AddOption(&ref_serial, "-rs", "--ref-serial",
                  "Serial uniform refinements before partitioning.");
   args.AddOption(&ref_parallel, "-rp", "--ref-parallel",
                  "Parallel uniform refinements after partitioning.");
   args.AddOption(&order, "-o", "--order", "Polynomial order.");
   args.AddOption(&reps, "-reps", "--repetitions",
                  "Repetitions for each timed kernel.");
   args.AddOption(&run_solve, "-solve", "--run-solve",
                  "-no-solve", "--no-run-solve",
                  "Run a CG + BoomerAMG solve and time it.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string (cpu, omp, cuda, ...).");
   args.Parse();
   if (!args.Good()) { if (root) { args.PrintUsage(cout); } return 1; }
   if (root) { args.PrintOptions(cout); }

   Device device(device_config);
   if (root) { device.Print(); }

   // 3. Build mesh: serial refinement -> partition -> parallel refinement.
   Mesh serial_mesh(mesh_file, 1, 1);
   for (int l = 0; l < ref_serial; l++) { serial_mesh.UniformRefinement(); }
   ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();
   for (int l = 0; l < ref_parallel; l++) { pmesh.UniformRefinement(); }

   const int dim = pmesh.Dimension();
   H1_FECollection fec(order, dim);
   ParFiniteElementSpace fes(&pmesh, &fec);
   const HYPRE_BigInt gdofs = fes.GlobalTrueVSize();
   const int ldofs = fes.GetTrueVSize();
   if (root)
   {
      cout << "Global DOFs: " << gdofs
           << "   ranks: " << nranks << endl;
   }

   // 4. Assemble parallel Poisson stiffness as a HypreParMatrix.
   ConstantCoefficient one(1.0);
   ParBilinearForm a(&fes);
   a.AddDomainIntegrator(new DiffusionIntegrator(one));
   {
      Timer t; t.start();
      a.Assemble();
      a.Finalize();
      const double ta = t.stop();
      if (root) { cout << "Assemble: " << fixed << setprecision(4)
                       << ta << " s" << endl; }
   }

   // True-DOF essential boundary DOFs (homogeneous Dirichlet on the boundary
   // attribute set, mimicking ex1p). If the mesh has no boundary, the array
   // stays empty and the system is purely Neumann.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   ParGridFunction x(&fes); x = 0.0;
   ParLinearForm b(&fes);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   HypreParMatrix &Ah = *A.As<HypreParMatrix>();

   // 5. Microbenchmarks.
   Vector y(X.Size()); y.UseDevice(false);
   Vector z(X.Size()); z.UseDevice(false);
   for (int i = 0; i < z.Size(); i++) { z(i) = std::sin(0.001*(i+rank+1)); }
   X.UseDevice(false);
   B.UseDevice(false);

   // Warm up.
   Ah.Mult(X, y);

   Timer t;

   t.start();
   for (int r = 0; r < reps; r++) { Ah.Mult(X, y); }
   const double t_spmv = t.stop();

   t.start();
   for (int r = 0; r < reps; r++) { y.Add(1.234e-3, z); }
   const double t_axpy = t.stop();

   real_t local_dot = 0.0, global_dot = 0.0;
   t.start();
   for (int r = 0; r < reps; r++) { local_dot += (X * z); }
   MPI_Allreduce(&local_dot, &global_dot, 1, MPITypeMap<real_t>::mpi_type,
                 MPI_SUM, MPI_COMM_WORLD);
   const double t_dot = t.stop();

   // Aggregate timings: max across ranks (the slowest rank dictates wall time).
   double t_spmv_max, t_axpy_max, t_dot_max;
   MPI_Reduce(&t_spmv, &t_spmv_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&t_axpy, &t_axpy_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
   MPI_Reduce(&t_dot,  &t_dot_max,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   const HYPRE_BigInt gnnz = Ah.NNZ();
   if (root)
   {
      const double gflops_spmv = 2.0 * (double)gnnz * reps / t_spmv_max / 1e9;
      const double gflops_axpy = 2.0 * (double)gdofs * reps / t_axpy_max / 1e9;
      const double gflops_dot  = 2.0 * (double)gdofs * reps / t_dot_max  / 1e9;
      cout << fixed << setprecision(4);
      cout << "------------------------------------------------------------\n";
      cout << "Operation         max(t)/rank (s)  aggregate GFLOP/s\n";
      cout << "------------------------------------------------------------\n";
      cout << "SpMV   y=A*x      " << t_spmv_max << "          " << gflops_spmv << "\n";
      cout << "AXPY   y+=a*z     " << t_axpy_max << "          " << gflops_axpy << "\n";
      cout << "Dot    x.z (red.) " << t_dot_max  << "          " << gflops_dot  << "\n";
      cout << "global nnz: " << gnnz << "  global DOFs: " << gdofs
           << "  reps: " << reps << "\n";
      cout << "------------------------------------------------------------\n";
   }

   // 6. Optional CG + AMG solve, just to sanity-check the assembled system.
   if (run_solve)
   {
      HypreBoomerAMG amg;
      amg.SetPrintLevel(0);
      HyprePCG cg(MPI_COMM_WORLD);
      cg.SetTol(1e-10);
      cg.SetMaxIter(200);
      cg.SetPrintLevel(root ? 1 : 0);
      cg.SetPreconditioner(amg);
      cg.SetOperator(Ah);
      Timer ts; ts.start();
      cg.Mult(B, X);
      const double t_solve = ts.stop();
      double t_solve_max = 0.0;
      MPI_Reduce(&t_solve, &t_solve_max, 1, MPI_DOUBLE, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      if (root) { cout << "CG+AMG solve : " << t_solve_max << " s\n"; }
   }

   return 0;
}
