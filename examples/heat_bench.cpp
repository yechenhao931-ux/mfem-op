/**
 * heat_bench.cpp
 * ==============
 * MFEM 热传导问题求解器自动选择 —— 训练数据采集
 *
 * 功能：
 *   - 构造 6 类典型热传导问题：
 *       1. 稳态各向同性                  −∇·(κ∇u)=f
 *       2. 稳态各向异性                  −∇·(K∇u)=f, K=diag(kx,ky,kz)
 *       3. 稳态空间变系数                −∇·(κ(x)∇u)=f
 *       4. 反应-扩散（带吸收项）         −∇·(κ∇u)+r·u=f
 *       5. 瞬态隐式（后向欧拉一步）       (M+dt·K) u = f
 *       6. 对流-扩散（带流场）            −∇·(κ∇u)+β·∇u=f
 *   - 多种网格（QUAD/TRI/HEX/TET），多级精炼，多阶元
 *   - 9 种迭代求解器 + 1 种直接法
 *   - 对每个 (问题, 网格, 求解器) 计时，提取矩阵特征
 *   - 输出 CSV，可直接喂给 train_solver_selector.py
 *
 * 编译（需先构建 MFEM 串行版库）：
 *   make heat_bench               # 在 examples/ 目录下
 *
 * 运行：
 *   ./heat_bench                                     # 完整跑一遍
 *   ./heat_bench --output heat_data.csv
 *   ./heat_bench --max-ref 5 --max-ref-3d 3 --no-direct
 *   ./heat_bench --problem trans                     # 仅瞬态
 *
 * 输出：heat_solver_data.csv — 每行一个 (问题, 求解器) 试验
 */

#include "mfem.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

// ====================================================================
//  配置
// ====================================================================
struct BenchConfig {
   int    min_ref          = 1;
   int    max_ref          = 5;       // 2D
   int    max_ref_3d       = 3;       // 3D（DOF 增长更快）
   int    order_min        = 1;
   int    order_max        = 2;
   double rtol             = 1e-8;
   double atol             = 1e-12;
   int    max_iter         = 3000;
   int    gmres_kdim       = 50;
   int    direct_max_dof   = 80000;
   double dt_transient     = 1e-3;     // 隐式时间步长
   bool   run_2d           = true;
   bool   run_3d           = true;
   bool   run_iso          = true;
   bool   run_aniso        = true;
   bool   run_var          = true;
   bool   run_react        = true;
   bool   run_trans        = true;
   bool   run_conv         = true;
   bool   run_direct       = true;
   bool   verbose          = false;
   std::string output_csv  = "heat_solver_data.csv";
};

BenchConfig ParseArgs(int argc, char** argv) {
   BenchConfig cfg;
   for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      auto next = [&]() { return std::string(argv[++i]); };
      if      (a == "--max-ref"     && i+1<argc) cfg.max_ref       = std::stoi(next());
      else if (a == "--max-ref-3d"  && i+1<argc) cfg.max_ref_3d    = std::stoi(next());
      else if (a == "--min-ref"     && i+1<argc) cfg.min_ref       = std::stoi(next());
      else if (a == "--order-min"   && i+1<argc) cfg.order_min     = std::stoi(next());
      else if (a == "--order-max"   && i+1<argc) cfg.order_max     = std::stoi(next());
      else if (a == "--output"      && i+1<argc) cfg.output_csv    = next();
      else if (a == "--rtol"        && i+1<argc) cfg.rtol          = std::stod(next());
      else if (a == "--max-iter"    && i+1<argc) cfg.max_iter      = std::stoi(next());
      else if (a == "--dt"          && i+1<argc) cfg.dt_transient  = std::stod(next());
      else if (a == "--direct-max"  && i+1<argc) cfg.direct_max_dof= std::stoi(next());
      else if (a == "--no-direct")  cfg.run_direct = false;
      else if (a == "--verbose")    cfg.verbose    = true;
      else if (a == "--dim") {
         std::string d = next();
         if (d == "2") cfg.run_3d = false;
         else          cfg.run_2d = false;
      }
      else if (a == "--problem") {
         std::string p = next();
         cfg.run_iso = cfg.run_aniso = cfg.run_var = false;
         cfg.run_react = cfg.run_trans = cfg.run_conv = false;
         if (p == "iso"   || p == "all") cfg.run_iso   = true;
         if (p == "aniso" || p == "all") cfg.run_aniso = true;
         if (p == "var"   || p == "all") cfg.run_var   = true;
         if (p == "react" || p == "all") cfg.run_react = true;
         if (p == "trans" || p == "all") cfg.run_trans = true;
         if (p == "conv"  || p == "all") cfg.run_conv  = true;
      }
   }
   return cfg;
}

// ====================================================================
//  单条记录（CSV 一行）
// ====================================================================
struct BenchRecord {
   // 物理/几何输入特征
   std::string problem;       // iso/aniso/var/react/trans/conv
   std::string mesh_type;     // quad/tri/hex/tet
   int    dim;
   int    ref_level;
   int    poly_order;
   int    n_elements;

   // 物理参数（输入特征）
   double kappa_iso;          // 平均导热系数
   double aniso_ratio;        // max(kx,ky,kz)/min — 1 表示各向同性
   double reaction_coef;      // r（反应项系数；非反应问题为 0）
   double dt_step;            // dt（瞬态；非瞬态为 0）
   double peclet;             // Pe（对流；非对流为 0）

   // 矩阵特征
   int    n_dof;
   long   nnz;
   double nnz_per_row;
   double sparsity;
   double symmetry_ratio;
   double diag_dominance;
   int    is_spd;             // 0/1
   double frob_norm_est;      // ||A||_F 近似（采样）

   // 求解器与结果
   std::string solver;
   double assemble_ms;
   double setup_ms;
   double solve_ms;
   double total_ms;
   int    converged;          // 0/1
   int    iterations;
   double final_residual;
   double relative_residual;
};

// ====================================================================
//  矩阵特征统计
// ====================================================================
struct MatStats {
   long   nnz       = 0;
   double per_row   = 0.0;
   double sparsity  = 0.0;
   double sym_ratio = 0.0;
   double diag_dom  = 0.0;
   bool   is_spd    = false;
   double frob_est  = 0.0;
};

MatStats ComputeMatStats(const SparseMatrix& A) {
   MatStats s;
   const int    n = A.Height();
   s.nnz          = A.NumNonZeroElems();
   s.per_row      = (double) s.nnz / std::max(n, 1);
   s.sparsity     = 1.0 - (double) s.nnz / (double(n) * double(n));

   const int*    I = A.GetI();
   const int*    J = A.GetJ();
   const double* V = A.GetData();

   // 对称偏差（最多采样 4000 个非零元）
   double asym = 0.0, norm = 0.0, frob_sq = 0.0;
   int taken = 0, max_take = 4000;
   for (int row = 0; row < n; ++row) {
      for (int k = I[row]; k < I[row+1]; ++k) {
         double v = V[k];
         frob_sq += v * v;
         if (taken < max_take && J[k] != row) {
            double aji = A.Elem(J[k], row);
            asym += std::abs(v - aji);
            norm += std::abs(v) + std::abs(aji) + 1e-30;
            ++taken;
         }
      }
   }
   s.sym_ratio = (norm > 1e-30) ? asym / norm : 0.0;
   s.frob_est  = std::sqrt(frob_sq);

   // 对角优势 + SPD 启发判定
   double sum_dd = 0.0;
   int spd_rows = 0;
   for (int row = 0; row < n; ++row) {
      double diag = 0.0, off = 0.0;
      for (int k = I[row]; k < I[row+1]; ++k) {
         if (J[k] == row) diag = V[k];
         else             off += std::abs(V[k]);
      }
      if (diag > off) ++spd_rows;
      sum_dd += (off > 1e-30) ? diag / off : 10.0;
   }
   s.diag_dom = sum_dd / std::max(n, 1);
   s.is_spd   = (spd_rows == n) && (s.sym_ratio < 1e-2);
   return s;
}

// ====================================================================
//  时间工具
// ====================================================================
using Clock = std::chrono::high_resolution_clock;
inline double ms_since(Clock::time_point t) {
   return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ====================================================================
//  各向异性矩阵系数
// ====================================================================
class AnisoTensor : public MatrixCoefficient {
   double kx_, ky_, kz_;
public:
   AnisoTensor(int dim, double kx, double ky, double kz = 1.0)
      : MatrixCoefficient(dim), kx_(kx), ky_(ky), kz_(kz) {}
   void Eval(DenseMatrix& K, ElementTransformation&,
             const IntegrationPoint&) override {
      const int d = GetWidth();
      K.SetSize(d); K = 0.0;
      K(0, 0) = kx_;
      if (d > 1) K(1, 1) = ky_;
      if (d > 2) K(2, 2) = kz_;
   }
};

// 空间变系数 κ(x) = 1 + 9·exp(-||x-c||²/0.05) ∈ [1, 10]
double VarKappa(const Vector& x) {
   const double cx = 0.5, cy = 0.5, cz = 0.5;
   double r2 = (x(0) - cx) * (x(0) - cx);
   if (x.Size() > 1) r2 += (x(1) - cy) * (x(1) - cy);
   if (x.Size() > 2) r2 += (x(2) - cz) * (x(2) - cz);
   return 1.0 + 9.0 * std::exp(-r2 / 0.05);
}

// ====================================================================
//  统一施加 Dirichlet 边界并组装线性系统
// ====================================================================
void FormSystem(BilinearForm& a, LinearForm& b, FiniteElementSpace& fes,
                Mesh& mesh, OperatorPtr& A, Vector& X, Vector& B,
                GridFunction& x_sol)
{
   Array<int> ess_tdof_list;
   if (mesh.bdr_attributes.Size()) {
      Array<int> ess_bdr(mesh.bdr_attributes.Max());
      ess_bdr = 1;
      fes.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
   x_sol = 0.0;
   a.FormLinearSystem(ess_tdof_list, x_sol, b, A, X, B);
}

// ====================================================================
//  通用求解器调度
//  返回 (converged, iterations, final_residual, setup_ms, solve_ms)
// ====================================================================
struct SolveOutcome {
   bool   converged = false;
   int    iters = 0;
   double final_res = 0.0;
   double setup_ms = 0.0;
   double solve_ms = 0.0;
};

SolveOutcome RunSolver(const std::string& name, SparseMatrix& A_sp,
                       const Vector& B, Vector& x, const BenchConfig& cfg)
{
   SolveOutcome out;
   const int print_lvl = cfg.verbose ? 1 : 0;

   auto setup_start = Clock::now();
   std::unique_ptr<Solver> prec;

   if (name == "PCG_Jacobi" || name == "GMRES_Jacobi" ||
       name == "BiCGSTAB_Jacobi") {
      prec.reset(new DSmoother(A_sp, 0));
   }
   else if (name == "PCG_GS") {
      prec.reset(new GSSmoother(A_sp));
   }
   else if (name == "PCG_Cheby") {
      prec.reset(new DSmoother(A_sp, 2));
   }
   out.setup_ms = ms_since(setup_start);

   auto solve_start = Clock::now();

   if (name == "CG" || name == "PCG_Jacobi" || name == "PCG_GS" ||
       name == "PCG_Cheby") {
      CGSolver cg;
      cg.SetOperator(A_sp);
      cg.SetRelTol(cfg.rtol);
      cg.SetAbsTol(cfg.atol);
      cg.SetMaxIter(cfg.max_iter);
      cg.SetPrintLevel(print_lvl);
      if (prec) cg.SetPreconditioner(*prec);
      cg.Mult(B, x);
      out.converged = cg.GetConverged();
      out.iters     = cg.GetNumIterations();
      out.final_res = cg.GetFinalNorm();
   }
   else if (name == "MINRES") {
      MINRESSolver mr;
      mr.SetOperator(A_sp);
      mr.SetRelTol(cfg.rtol);
      mr.SetAbsTol(cfg.atol);
      mr.SetMaxIter(cfg.max_iter);
      mr.SetPrintLevel(print_lvl);
      mr.Mult(B, x);
      out.converged = mr.GetConverged();
      out.iters     = mr.GetNumIterations();
      out.final_res = mr.GetFinalNorm();
   }
   else if (name == "GMRES" || name == "GMRES_Jacobi") {
      GMRESSolver gm;
      gm.SetOperator(A_sp);
      gm.SetRelTol(cfg.rtol);
      gm.SetAbsTol(cfg.atol);
      gm.SetMaxIter(cfg.max_iter);
      gm.SetKDim(cfg.gmres_kdim);
      gm.SetPrintLevel(print_lvl);
      if (prec) gm.SetPreconditioner(*prec);
      gm.Mult(B, x);
      out.converged = gm.GetConverged();
      out.iters     = gm.GetNumIterations();
      out.final_res = gm.GetFinalNorm();
   }
   else if (name == "BiCGSTAB" || name == "BiCGSTAB_Jacobi") {
      BiCGSTABSolver bs;
      bs.SetOperator(A_sp);
      bs.SetRelTol(cfg.rtol);
      bs.SetAbsTol(cfg.atol);
      bs.SetMaxIter(cfg.max_iter);
      bs.SetPrintLevel(print_lvl);
      if (prec) bs.SetPreconditioner(*prec);
      bs.Mult(B, x);
      out.converged = bs.GetConverged();
      out.iters     = bs.GetNumIterations();
      out.final_res = bs.GetFinalNorm();
   }
#ifdef MFEM_USE_SUITESPARSE
   else if (name == "DIRECT_UMF") {
      auto t = Clock::now();
      UMFPackSolver direct;
      direct.SetOperator(A_sp);
      out.setup_ms += ms_since(t);
      direct.Mult(B, x);
      out.converged = true;
      out.iters     = 1;
      out.final_res = 0.0;
   }
#endif
   else {
      out.converged = false;
   }

   out.solve_ms = ms_since(solve_start) - out.setup_ms;
   return out;
}

// ====================================================================
//  组装一个热传导问题，返回 SparseMatrix 与 RHS
//  problem ∈ {iso, aniso, var, react, trans, conv}
// ====================================================================
struct AssembledSystem {
   std::unique_ptr<BilinearForm> a;
   std::unique_ptr<LinearForm>   b;
   std::unique_ptr<GridFunction> x_sol;
   OperatorPtr A_form;
   Vector      X, B;
   double      assemble_ms = 0.0;

   // 系数所有权容器（保持其生命周期超过组装过程）
   std::vector<std::unique_ptr<Coefficient>>       scalar_coefs;
   std::vector<std::unique_ptr<VectorCoefficient>> vec_coefs;
   std::vector<std::unique_ptr<MatrixCoefficient>> mat_coefs;

   // 输入特征
   double kappa_iso     = 1.0;
   double aniso_ratio   = 1.0;
   double reaction_coef = 0.0;
   double dt_step       = 0.0;
   double peclet        = 0.0;
};

AssembledSystem AssembleHeat(const std::string& problem, Mesh& mesh,
                             FiniteElementSpace& fes,
                             const BenchConfig& cfg)
{
   AssembledSystem s;
   const int dim = mesh.Dimension();
   auto t0 = Clock::now();

   s.a.reset(new BilinearForm(&fes));
   s.b.reset(new LinearForm(&fes));
   auto* one = new ConstantCoefficient(1.0);
   s.scalar_coefs.emplace_back(one);
   s.b->AddDomainIntegrator(new DomainLFIntegrator(*one));

   if (problem == "iso") {
      s.kappa_iso = 1.0;
      auto* kappa = new ConstantCoefficient(s.kappa_iso);
      s.scalar_coefs.emplace_back(kappa);
      s.a->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
   }
   else if (problem == "aniso") {
      // 各向异性：x 方向比 y/z 强 100 倍
      double kx = 100.0, ky = 1.0, kz = 1.0;
      s.kappa_iso   = (kx + ky + kz) / 3.0;
      s.aniso_ratio = std::max({kx, ky, kz}) / std::min({kx, ky, kz});
      auto* K = new AnisoTensor(dim, kx, ky, kz);
      s.mat_coefs.emplace_back(K);
      s.a->AddDomainIntegrator(new DiffusionIntegrator(*K));
   }
   else if (problem == "var") {
      s.kappa_iso = 5.5;
      auto* kappa = new FunctionCoefficient(VarKappa);
      s.scalar_coefs.emplace_back(kappa);
      s.a->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
   }
   else if (problem == "react") {
      s.kappa_iso     = 1.0;
      s.reaction_coef = 10.0;
      auto* kappa = new ConstantCoefficient(s.kappa_iso);
      auto* rc    = new ConstantCoefficient(s.reaction_coef);
      s.scalar_coefs.emplace_back(kappa);
      s.scalar_coefs.emplace_back(rc);
      s.a->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
      s.a->AddDomainIntegrator(new MassIntegrator(*rc));
   }
   else if (problem == "trans") {
      // 隐式一步：(M + dt·K) u = M·u_prev + dt·b
      s.kappa_iso = 1.0;
      s.dt_step   = cfg.dt_transient;
      auto* dtcf = new ConstantCoefficient(s.dt_step);
      s.scalar_coefs.emplace_back(dtcf);
      s.a->AddDomainIntegrator(new MassIntegrator());
      s.a->AddDomainIntegrator(new DiffusionIntegrator(*dtcf));
   }
   else if (problem == "conv") {
      // 对流-扩散：β 沿 x 方向，强对流 → 非对称
      s.kappa_iso = 0.01;
      Vector beta(dim); beta = 0.0; beta(0) = 1.0;
      if (dim > 1) beta(1) = 0.5;
      double h = mesh.GetElementSize(0);
      s.peclet  = beta.Norml2() * h / (2.0 * s.kappa_iso);
      auto* kappa = new ConstantCoefficient(s.kappa_iso);
      auto* bc    = new VectorConstantCoefficient(beta);
      s.scalar_coefs.emplace_back(kappa);
      s.vec_coefs.emplace_back(bc);
      s.a->AddDomainIntegrator(new DiffusionIntegrator(*kappa));
      s.a->AddDomainIntegrator(new ConvectionIntegrator(*bc, -1.0));
   }

   s.a->Assemble(); s.a->Finalize();
   s.b->Assemble();

   s.x_sol.reset(new GridFunction(&fes));
   FormSystem(*s.a, *s.b, fes, mesh, s.A_form, s.X, s.B, *s.x_sol);
   s.assemble_ms = ms_since(t0);
   return s;
}

// ====================================================================
//  跑一个 (problem, mesh, order) × 全部求解器
// ====================================================================
void RunCase(const std::string& problem, const std::string& mesh_type,
             Mesh& mesh, int order, int ref_level,
             const std::vector<std::string>& solvers,
             const BenchConfig& cfg, std::ofstream& csv,
             std::vector<BenchRecord>& records)
{
   const int dim = mesh.Dimension();
   H1_FECollection fec(order, dim);
   FiniteElementSpace fes(&mesh, &fec);

   auto sys = AssembleHeat(problem, mesh, fes, cfg);
   if (!sys.A_form.Ptr()) return;
   SparseMatrix& A_sp = static_cast<SparseMatrix&>(*sys.A_form.Ptr());
   MatStats stats = ComputeMatStats(A_sp);

   for (const auto& solver : solvers) {
#ifdef MFEM_USE_SUITESPARSE
      if (solver == "DIRECT_UMF" &&
         (sys.X.Size() > cfg.direct_max_dof || !cfg.run_direct))
         continue;
#else
      if (solver == "DIRECT_UMF") continue;
#endif

      Vector x(sys.X.Size()); x = 0.0;
      auto t_total = Clock::now();
      auto out = RunSolver(solver, A_sp, sys.B, x, cfg);
      double total_ms = ms_since(t_total);
      double b_norm = sys.B.Norml2();

      BenchRecord r;
      r.problem        = problem;
      r.mesh_type      = mesh_type;
      r.dim            = dim;
      r.ref_level      = ref_level;
      r.poly_order     = order;
      r.n_elements     = mesh.GetNE();
      r.kappa_iso      = sys.kappa_iso;
      r.aniso_ratio    = sys.aniso_ratio;
      r.reaction_coef  = sys.reaction_coef;
      r.dt_step        = sys.dt_step;
      r.peclet         = sys.peclet;
      r.n_dof          = fes.GetTrueVSize();
      r.nnz            = stats.nnz;
      r.nnz_per_row    = stats.per_row;
      r.sparsity       = stats.sparsity;
      r.symmetry_ratio = stats.sym_ratio;
      r.diag_dominance = stats.diag_dom;
      r.is_spd         = stats.is_spd ? 1 : 0;
      r.frob_norm_est  = stats.frob_est;
      r.solver         = solver;
      r.assemble_ms    = sys.assemble_ms;
      r.setup_ms       = out.setup_ms;
      r.solve_ms       = out.solve_ms;
      r.total_ms       = total_ms;
      r.converged      = out.converged ? 1 : 0;
      r.iterations     = out.iters;
      r.final_residual = out.final_res;
      r.relative_residual = (b_norm > 1e-30) ? out.final_res / b_norm
                                             : out.final_res;

      // CSV 行
      csv << r.problem << ',' << r.mesh_type << ',' << r.dim << ','
          << r.ref_level << ',' << r.poly_order << ',' << r.n_elements << ','
          << std::fixed << std::setprecision(6)
          << r.kappa_iso << ',' << r.aniso_ratio << ','
          << r.reaction_coef << ',' << r.dt_step << ',' << r.peclet << ','
          << r.n_dof << ',' << r.nnz << ','
          << std::setprecision(3) << r.nnz_per_row << ','
          << std::setprecision(6) << r.sparsity << ','
          << r.symmetry_ratio << ',' << r.diag_dominance << ','
          << r.is_spd << ','
          << std::scientific << std::setprecision(4) << r.frob_norm_est << ','
          << r.solver << ','
          << std::fixed << std::setprecision(3)
          << r.assemble_ms << ',' << r.setup_ms << ','
          << r.solve_ms << ',' << r.total_ms << ','
          << r.converged << ',' << r.iterations << ','
          << std::scientific << std::setprecision(4)
          << r.final_residual << ',' << r.relative_residual << '\n';

      std::printf("  [%s|%s|%dD|p%d|ref%d|n=%d] %-18s "
                  "iters=%4d  total=%8.2f ms  %s\n",
                  problem.c_str(), mesh_type.c_str(), dim, order, ref_level,
                  r.n_dof, solver.c_str(), out.iters, total_ms,
                  out.converged ? "✓" : "✗");

      records.push_back(r);
   }
}

// ====================================================================
//  主循环
// ====================================================================
int main(int argc, char** argv)
{
   BenchConfig cfg = ParseArgs(argc, argv);

   std::printf("\n=== MFEM 热传导求解器训练数据采集 ===\n");
   std::printf("    rtol=%.0e  atol=%.0e  max_iter=%d\n",
               cfg.rtol, cfg.atol, cfg.max_iter);
   std::printf("    输出: %s\n\n", cfg.output_csv.c_str());

   std::vector<std::string> solvers = {
      "CG", "PCG_Jacobi", "PCG_GS", "PCG_Cheby",
      "MINRES", "GMRES", "GMRES_Jacobi",
      "BiCGSTAB", "BiCGSTAB_Jacobi"
   };
#ifdef MFEM_USE_SUITESPARSE
   if (cfg.run_direct) solvers.push_back("DIRECT_UMF");
#endif

   std::ofstream csv(cfg.output_csv);
   if (!csv) {
      std::fprintf(stderr, "无法打开输出文件 %s\n", cfg.output_csv.c_str());
      return 1;
   }
   csv << "problem,mesh_type,dim,ref_level,poly_order,n_elements,"
          "kappa_iso,aniso_ratio,reaction_coef,dt_step,peclet,"
          "n_dof,nnz,nnz_per_row,sparsity,symmetry_ratio,diag_dominance,"
          "is_spd,frob_norm_est,"
          "solver,assemble_ms,setup_ms,solve_ms,total_ms,"
          "converged,iterations,final_residual,relative_residual\n";

   std::vector<BenchRecord> records;
   records.reserve(2048);

   std::vector<std::string> problems;
   if (cfg.run_iso)   problems.push_back("iso");
   if (cfg.run_aniso) problems.push_back("aniso");
   if (cfg.run_var)   problems.push_back("var");
   if (cfg.run_react) problems.push_back("react");
   if (cfg.run_trans) problems.push_back("trans");
   if (cfg.run_conv)  problems.push_back("conv");

   // ------------------------------------------------------------------
   // 2D 网格
   // ------------------------------------------------------------------
   if (cfg.run_2d) {
      for (const auto& mt : std::vector<std::string>{"quad", "tri"}) {
         Element::Type etype = (mt == "quad") ? Element::QUADRILATERAL
                                              : Element::TRIANGLE;
         for (int order = cfg.order_min; order <= cfg.order_max; ++order) {
            for (int ref = cfg.min_ref; ref <= cfg.max_ref; ++ref) {
               Mesh mesh = Mesh::MakeCartesian2D(2, 2, etype, true);
               for (int i = 0; i < ref; ++i) mesh.UniformRefinement();
               for (const auto& p : problems) {
                  RunCase(p, mt, mesh, order, ref, solvers, cfg, csv, records);
               }
            }
         }
      }
   }

   // ------------------------------------------------------------------
   // 3D 网格
   // ------------------------------------------------------------------
   if (cfg.run_3d) {
      for (const auto& mt : std::vector<std::string>{"hex", "tet"}) {
         Element::Type etype = (mt == "hex") ? Element::HEXAHEDRON
                                             : Element::TETRAHEDRON;
         for (int order = cfg.order_min; order <= cfg.order_max; ++order) {
            for (int ref = cfg.min_ref; ref <= cfg.max_ref_3d; ++ref) {
               Mesh mesh = Mesh::MakeCartesian3D(2, 2, 2, etype);
               for (int i = 0; i < ref; ++i) mesh.UniformRefinement();
               for (const auto& p : problems) {
                  RunCase(p, mt, mesh, order, ref, solvers, cfg, csv, records);
               }
            }
         }
      }
   }

   csv.close();

   // ------------------------------------------------------------------
   // 汇总
   // ------------------------------------------------------------------
   std::printf("\n=== 汇总 ===\n");
   std::printf("    样本总数: %zu\n", records.size());

   // 每个 (problem, mesh, dim, order, ref) 组合中收敛最快的求解器 → 胜场
   std::map<std::string, int> wins, attempts;
   std::map<std::string, std::vector<const BenchRecord*>> groups;
   for (const auto& r : records) {
      std::ostringstream key;
      key << r.problem << '|' << r.mesh_type << '|' << r.dim << '|'
          << r.poly_order << '|' << r.ref_level;
      groups[key.str()].push_back(&r);
      attempts[r.solver]++;
   }
   for (auto& [k, v] : groups) {
      const BenchRecord* best = nullptr;
      for (const auto* r : v) {
         if (!r->converged) continue;
         if (!best || r->total_ms < best->total_ms) best = r;
      }
      if (best) wins[best->solver]++;
   }
   std::printf("\n    %-18s %8s %8s\n", "求解器", "胜场", "尝试");
   for (const auto& s : solvers) {
      std::printf("    %-18s %8d %8d\n", s.c_str(),
                  wins[s], attempts[s]);
   }
   std::printf("\n  ✓ CSV 已写入: %s\n", cfg.output_csv.c_str());
   std::printf("    下一步: python3 heat_solver_ai/train_solver_selector.py "
               "--csv %s\n\n", cfg.output_csv.c_str());

   return 0;
}
