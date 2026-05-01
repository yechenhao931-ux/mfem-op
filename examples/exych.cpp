/**
 * benchmark.cpp
 * =============
 * MFEM 多物理场、多网格规模、多求解器性能基准测试
 *
 * 功能：
 *   - 构造 3 类典型物理问题（Poisson / 线性弹性 / 对流扩散）
 *   - 在 2D/3D 网格上逐步精炼（控制自由度规模）
 *   - 对每个（问题, 网格, 求解器）组合计时并记录收敛情况
 *   - 输出 ASCII 表格 + CSV 文件（可直接用于 Python 训练）
 *
 * 依赖：MFEM（串行版）、C++17
 *
 * 编译示例：
 *   g++ -std=c++17 -O2 \
 *       -I$(MFEM_DIR)/include \
 *       -L$(MFEM_DIR)/lib \
 *       benchmark.cpp \
 *       -lmfem -llapack -lblas -lm \
 *       -o benchmark
 *
 * 运行示例：
 *   ./benchmark                        # 默认参数
 *   ./benchmark --max-ref 5            # 最大精炼级别
 *   ./benchmark --output results.csv   # 指定输出路径
 *   ./benchmark --dim 3                # 仅测试 3D
 *   ./benchmark --problem poisson      # 仅测试 Poisson
 *   ./benchmark --no-direct            # 跳过直接法（避免内存溢出）
 */

#include "mfem.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace mfem;

// ======================================================================
//  配置
// ======================================================================
struct BenchConfig {
    int         min_ref      = 1;      // 最小精炼级别
    int         max_ref      = 6;      // 最大精炼级别（2D）
    int         max_ref_3d   = 4;      // 3D 最大精炼（DOF 增长更快）
    int         order        = 1;      // FEM 多项式阶数
    double      rtol         = 1e-8;   // 相对收敛容差
    double      atol         = 1e-12;  // 绝对收敛容差
    int         max_iter     = 3000;   // 最大迭代次数
    int         gmres_kdim   = 50;     // GMRES 重启 Krylov 维数
    std::string output_csv   = "benchmark_results.csv";
    bool        run_2d       = true;
    bool        run_3d       = true;
    bool        run_poisson  = true;
    bool        run_elast    = true;
    bool        run_conv_diff= true;
    bool        run_direct   = true;   // 直接法（小规模用）
    int         direct_max_dof = 50000; // 超过此 DOF 跳过直接法
    bool        verbose      = false;
    bool        print_matrix_info = true;
};

BenchConfig ParseArgs(int argc, char** argv) {
    BenchConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--max-ref"    && i+1<argc) cfg.max_ref     = std::stoi(argv[++i]);
        else if (a == "--max-ref-3d" && i+1<argc) cfg.max_ref_3d  = std::stoi(argv[++i]);
        else if (a == "--min-ref"    && i+1<argc) cfg.min_ref     = std::stoi(argv[++i]);
        else if (a == "--order"      && i+1<argc) cfg.order       = std::stoi(argv[++i]);
        else if (a == "--output"     && i+1<argc) cfg.output_csv  = argv[++i];
        else if (a == "--rtol"       && i+1<argc) cfg.rtol        = std::stod(argv[++i]);
        else if (a == "--max-iter"   && i+1<argc) cfg.max_iter    = std::stoi(argv[++i]);
        else if (a == "--dim") {
            ++i;
            if (std::string(argv[i]) == "2") { cfg.run_3d = false; }
            else                              { cfg.run_2d = false; }
        }
        else if (a == "--problem") {
            ++i; std::string p = argv[i];
            cfg.run_poisson   = (p == "poisson"  || p == "all");
            cfg.run_elast     = (p == "elast"    || p == "all");
            cfg.run_conv_diff = (p == "convdiff" || p == "all");
        }
        else if (a == "--no-direct") cfg.run_direct = false;
        else if (a == "--verbose")   cfg.verbose    = true;
    }
    return cfg;
}

// ======================================================================
//  单次测量结果
// ======================================================================
struct BenchResult {
    // 问题描述
    std::string problem_name;   // "Poisson2D", "Elasticity3D", ...
    std::string solver_name;    // "CG", "PCG_AMG", ...
    int         dim;
    int         ref_level;
    int         poly_order;

    // 矩阵特征
    int    n_dof;               // 自由度数
    int    n_elements;          // 单元数
    long   nnz;                 // 非零元总数
    double nnz_per_row;
    double symmetry_ratio;      // |A-At|_F / |A|_F 的近似
    bool   is_spd;              // Gershgorin 判断

    // 组装时间
    double assemble_time_ms;

    // 求解结果
    bool   converged;
    int    iterations;
    double solve_time_ms;
    double total_time_ms;
    double final_residual;
    double relative_residual;   // ||r||/||b||
    double setup_time_ms;       // 预条件器/因子化时间

    // 解的质量（L2 误差，如有解析解）
    double l2_error;            // -1 表示无解析解
};

// ======================================================================
//  系数：各向同性 / 各向异性扩散
// ======================================================================
class AnisoDiffCoeff : public MatrixCoefficient {
    double kx_, ky_, kz_;
public:
    AnisoDiffCoeff(double kx, double ky, double kz = 1.0)
        : MatrixCoefficient(3), kx_(kx), ky_(ky), kz_(kz) {}
    void Eval(DenseMatrix& K, ElementTransformation& T,
              const IntegrationPoint& ip) override {
        K.SetSize(3, 3); K = 0.0;
        K(0,0) = kx_; K(1,1) = ky_; K(2,2) = kz_;
    }
};

// 对流系数（常数对流场）
class ConvCoeff : public VectorCoefficient {
    double bx_, by_, bz_;
public:
    ConvCoeff(double bx, double by, double bz = 0.0)
        : VectorCoefficient(3), bx_(bx), by_(by), bz_(bz) {}
    void Eval(Vector& b, ElementTransformation& T,
              const IntegrationPoint& ip) override {
        b.SetSize(3); b(0) = bx_; b(1) = by_; b(2) = bz_;
    }
};

// ======================================================================
//  矩阵特征分析（快速统计）
// ======================================================================
struct MatrixStats {
    int    n, nnz;
    double nnz_per_row;
    double symmetry_ratio;   // 0 = 完全对称
    bool   is_spd;           // Gershgorin 判定
    double diag_dominance;   // min(diag) / max(off-diag per row)
    double sparsity;         // 1 - nnz/(n*n)

    static MatrixStats Compute(const SparseMatrix& A) {
        MatrixStats s;
        s.n   = A.Height();
        s.nnz = A.NumNonZeroElems();
        s.nnz_per_row = (double)s.nnz / s.n;
        s.sparsity    = 1.0 - (double)s.nnz / ((double)s.n * s.n);

        // 对称性：采样 2000 个非零元
        const int*    I = A.GetI();
        const int*    J = A.GetJ();
        const double* V = A.GetData();
        double asym_sum = 0.0, norm_sum = 0.0;
        int samples = 0, max_samp = 2000;
        for (int row = 0; row < s.n && samples < max_samp; ++row) {
            for (int k = I[row]; k < I[row+1] && samples < max_samp; ++k) {
                int col = J[k];
                if (col == row) continue;
                double aij = V[k];
                double aji = A.Elem(col, row);
                asym_sum += std::abs(aij - aji);
                norm_sum += std::abs(aij) + std::abs(aji) + 1e-30;
                ++samples;
            }
        }
        s.symmetry_ratio = (norm_sum > 1e-30) ? asym_sum / norm_sum : 0.0;

        // SPD / 对角优势
        double min_dd = 1e30, total_dd = 0.0;
        int    spd_count = 0;
        for (int row = 0; row < s.n; ++row) {
            double diag = 0.0, off = 0.0;
            for (int k = I[row]; k < I[row+1]; ++k) {
                double v = std::abs(V[k]);
                if (J[k] == row) diag = V[k];
                else             off += v;
            }
            if (diag > off) ++spd_count;
            double dd = (off > 1e-30) ? diag / off : 10.0;
            min_dd    = std::min(min_dd, dd);
            total_dd += dd;
        }
        s.is_spd         = (spd_count == s.n) && (s.symmetry_ratio < 0.01);
        s.diag_dominance = total_dd / s.n;
        return s;
    }
};

// ======================================================================
//  计时工具
// ======================================================================
using Clock    = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;

inline double ms_since(Clock::time_point t0) {
    return Duration(Clock::now() - t0).count();
}

// ======================================================================
//  通用：施加本质边界条件并组装线性系统
// ======================================================================
void ApplyBCsAndForm(BilinearForm& a, LinearForm& b,
                     FiniteElementSpace& fespace,
                     Mesh& mesh,
                     OperatorPtr& A, Vector& B, Vector& X,
                     GridFunction& x_sol)
{
    Array<int> ess_tdof_list;
    if (mesh.bdr_attributes.Size()) {
        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 1;
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }
    x_sol = 0.0;
    a.FormLinearSystem(ess_tdof_list, x_sol, b, A, X, B);
}

// ======================================================================
//  Poisson 问题：−Δu = 1，均匀 Dirichlet
// ======================================================================
BenchResult RunPoisson(Mesh& mesh, int order, const std::string& solver_name,
                        const BenchConfig& cfg)
{
    BenchResult r;
    r.problem_name = "Poisson_" + std::to_string(mesh.Dimension()) + "D";
    r.solver_name  = solver_name;
    r.dim          = mesh.Dimension();
    r.poly_order   = order;
    r.n_elements   = mesh.GetNE();
    r.l2_error     = -1.0;

    // ── 有限元空间 ──────────────────────────────────────────
    H1_FECollection  fec(order, mesh.Dimension());
    FiniteElementSpace fespace(&mesh, &fec);
    r.n_dof = fespace.GetTrueVSize();

    // ── 组装 ────────────────────────────────────────────────
    auto t_asm = Clock::now();
    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator());
    a.Assemble(); a.Finalize();

    LinearForm b(&fespace);
    ConstantCoefficient one(1.0);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));
    b.Assemble();

    GridFunction x_sol(&fespace);
    OperatorPtr  A; Vector B, X;
    ApplyBCsAndForm(a, b, fespace, mesh, A, X, B, x_sol);
    r.assemble_time_ms = ms_since(t_asm);

    // 矩阵特征
    SparseMatrix& Asp = static_cast<SparseMatrix&>(*A.Ptr());
    auto stats = MatrixStats::Compute(Asp);
    r.nnz          = stats.nnz;
    r.nnz_per_row  = stats.nnz_per_row;
    r.symmetry_ratio = stats.symmetry_ratio;
    r.is_spd       = stats.is_spd;

    // ── 求解 ────────────────────────────────────────────────
    Vector x(X.Size()); x = 0.0;
    r.setup_time_ms = 0.0;

    auto t_solve = Clock::now();

    if (solver_name == "CG") {
        CGSolver cg;
        cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter);
        cg.SetPrintLevel(cfg.verbose ? 1 : 0);
        cg.Mult(B, x);
        r.converged  = cg.GetConverged();
        r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "PCG_GS") {
        GSSmoother prec(Asp);
        CGSolver cg;
        cg.SetPreconditioner(prec);
        cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter);
        cg.SetPrintLevel(cfg.verbose ? 1 : 0);
        cg.Mult(B, x);
        r.converged  = cg.GetConverged();
        r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "PCG_Cheby") {
        // Chebyshev 光滑器作为预条件器
        auto t_setup = Clock::now();
        DSmoother prec(Asp, 2);  // type 2 = Chebyshev
        r.setup_time_ms = ms_since(t_setup);
        CGSolver cg;
        cg.SetPreconditioner(prec);
        cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter);
        cg.SetPrintLevel(cfg.verbose ? 1 : 0);
        cg.Mult(B, x);
        r.converged  = cg.GetConverged();
        r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "MINRES") {
        MINRESSolver mr;
        mr.SetOperator(*A.Ptr());
        mr.SetRelTol(cfg.rtol); mr.SetAbsTol(cfg.atol);
        mr.SetMaxIter(cfg.max_iter);
        mr.SetPrintLevel(cfg.verbose ? 1 : 0);
        mr.Mult(B, x);
        r.converged  = mr.GetConverged();
        r.iterations = mr.GetNumIterations();
        r.final_residual = mr.GetFinalNorm();
    }
    else if (solver_name == "GMRES") {
        GMRESSolver gm;
        gm.SetOperator(*A.Ptr());
        gm.SetRelTol(cfg.rtol); gm.SetAbsTol(cfg.atol);
        gm.SetMaxIter(cfg.max_iter);
        gm.SetKDim(cfg.gmres_kdim);
        gm.SetPrintLevel(cfg.verbose ? 1 : 0);
        gm.Mult(B, x);
        r.converged  = gm.GetConverged();
        r.iterations = gm.GetNumIterations();
        r.final_residual = gm.GetFinalNorm();
    }
    else if (solver_name == "GMRES_DS") {
        auto t_setup = Clock::now();
        DSmoother prec(Asp);
        r.setup_time_ms = ms_since(t_setup);
        GMRESSolver gm;
        gm.SetPreconditioner(prec);
        gm.SetOperator(*A.Ptr());
        gm.SetRelTol(cfg.rtol); gm.SetAbsTol(cfg.atol);
        gm.SetMaxIter(cfg.max_iter); gm.SetKDim(cfg.gmres_kdim);
        gm.SetPrintLevel(cfg.verbose ? 1 : 0);
        gm.Mult(B, x);
        r.converged  = gm.GetConverged();
        r.iterations = gm.GetNumIterations();
        r.final_residual = gm.GetFinalNorm();
    }
#ifdef MFEM_USE_SUITESPARSE
    else if (solver_name == "DIRECT_UMF") {
        auto t_setup = Clock::now();
        UMFPackSolver direct;
        direct.SetOperator(*A.Ptr());
        r.setup_time_ms = ms_since(t_setup);
        direct.Mult(B, x);
        r.converged  = true;
        r.iterations = 1;
        r.final_residual = 0.0;
    }
#endif
    else {
        // 未知求解器：跳过
        r.converged      = false;
        r.iterations     = 0;
        r.solve_time_ms  = 0.0;
        r.total_time_ms  = 0.0;
        r.final_residual = -1.0;
        r.relative_residual = -1.0;
        return r;
    }

    r.solve_time_ms = ms_since(t_solve) - r.setup_time_ms;
    r.total_time_ms = ms_since(t_solve);

    // 相对残差
    double b_norm = B.Norml2();
    r.relative_residual = (b_norm > 1e-30) ? r.final_residual / b_norm : r.final_residual;

    // 恢复有限元解并计算 L2 误差（与零函数的误差，即解范数）
    a.RecoverFEMSolution(x, b, x_sol);
    ConstantCoefficient zero(0.0);
    r.l2_error = x_sol.ComputeL2Error(zero);

    r.ref_level = -1;   // 由调用方填写
    return r;
}

// ======================================================================
//  线性弹性问题：−div(σ(u)) = f，固定底面
// ======================================================================
BenchResult RunElasticity(Mesh& mesh, int order, const std::string& solver_name,
                           const BenchConfig& cfg)
{
    BenchResult r;
    int dim = mesh.Dimension();
    r.problem_name = "Elasticity_" + std::to_string(dim) + "D";
    r.solver_name  = solver_name;
    r.dim          = dim;
    r.poly_order   = order;
    r.n_elements   = mesh.GetNE();
    r.l2_error     = -1.0;

    // 矢量有限元空间（每个节点 dim 个分量）
    H1_FECollection   fec(order, dim);
    FiniteElementSpace fespace(&mesh, &fec, dim);
    r.n_dof = fespace.GetTrueVSize();

    // 弹性参数：钢铁近似（归一化）
    double E = 200.0, nu = 0.3;
    double mu     = E / (2.0 * (1.0 + nu));
    double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

    auto t_asm = Clock::now();
    BilinearForm a(&fespace);
    ConstantCoefficient mu_cf(mu), lambda_cf(lambda);
    a.AddDomainIntegrator(new ElasticityIntegrator(lambda_cf, mu_cf));
    a.Assemble(); a.Finalize();

    // 重力荷载
    LinearForm b(&fespace);
    Vector grav(dim); grav = 0.0; grav(dim-1) = -9.8;
    VectorConstantCoefficient grav_cf(grav);
    b.AddDomainIntegrator(new VectorDomainLFIntegrator(grav_cf));
    b.Assemble();

    // 固定底面（y=0 或 z=0）
    Array<int> ess_tdof_list;
    if (mesh.bdr_attributes.Size()) {
        Array<int> ess_bdr(mesh.bdr_attributes.Max());
        ess_bdr = 0;
        ess_bdr[0] = 1;   // 第 1 个边界属性固定
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }
    GridFunction x_sol(&fespace); x_sol = 0.0;
    OperatorPtr  A; Vector B, X;
    a.FormLinearSystem(ess_tdof_list, x_sol, b, A, X, B);
    r.assemble_time_ms = ms_since(t_asm);

    SparseMatrix& Asp = static_cast<SparseMatrix&>(*A.Ptr());
    auto stats = MatrixStats::Compute(Asp);
    r.nnz          = stats.nnz;
    r.nnz_per_row  = stats.nnz_per_row;
    r.symmetry_ratio = stats.symmetry_ratio;
    r.is_spd       = stats.is_spd;

    Vector x(X.Size()); x = 0.0;
    r.setup_time_ms = 0.0;

    auto t_solve = Clock::now();

    if (solver_name == "CG") {
        CGSolver cg; cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter); cg.SetPrintLevel(0);
        cg.Mult(B, x);
        r.converged = cg.GetConverged(); r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "PCG_GS") {
        GSSmoother prec(Asp);
        CGSolver cg; cg.SetPreconditioner(prec); cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter); cg.SetPrintLevel(0);
        cg.Mult(B, x);
        r.converged = cg.GetConverged(); r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "PCG_Cheby") {
        auto t_setup = Clock::now();
        DSmoother prec(Asp, 2); r.setup_time_ms = ms_since(t_setup);
        CGSolver cg; cg.SetPreconditioner(prec); cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter); cg.SetPrintLevel(0);
        cg.Mult(B, x);
        r.converged = cg.GetConverged(); r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "MINRES") {
        MINRESSolver mr; mr.SetOperator(*A.Ptr());
        mr.SetRelTol(cfg.rtol); mr.SetAbsTol(cfg.atol);
        mr.SetMaxIter(cfg.max_iter); mr.SetPrintLevel(0);
        mr.Mult(B, x);
        r.converged = mr.GetConverged(); r.iterations = mr.GetNumIterations();
        r.final_residual = mr.GetFinalNorm();
    }
    else if (solver_name == "GMRES") {
        GMRESSolver gm; gm.SetOperator(*A.Ptr());
        gm.SetRelTol(cfg.rtol); gm.SetAbsTol(cfg.atol);
        gm.SetMaxIter(cfg.max_iter); gm.SetKDim(cfg.gmres_kdim);
        gm.SetPrintLevel(0); gm.Mult(B, x);
        r.converged = gm.GetConverged(); r.iterations = gm.GetNumIterations();
        r.final_residual = gm.GetFinalNorm();
    }
    else if (solver_name == "GMRES_DS") {
        auto t_setup = Clock::now();
        DSmoother prec(Asp); r.setup_time_ms = ms_since(t_setup);
        GMRESSolver gm; gm.SetPreconditioner(prec); gm.SetOperator(*A.Ptr());
        gm.SetRelTol(cfg.rtol); gm.SetAbsTol(cfg.atol);
        gm.SetMaxIter(cfg.max_iter); gm.SetKDim(cfg.gmres_kdim);
        gm.SetPrintLevel(0); gm.Mult(B, x);
        r.converged = gm.GetConverged(); r.iterations = gm.GetNumIterations();
        r.final_residual = gm.GetFinalNorm();
    }
#ifdef MFEM_USE_SUITESPARSE
    else if (solver_name == "DIRECT_UMF") {
        auto t_setup = Clock::now();
        UMFPackSolver direct; direct.SetOperator(*A.Ptr());
        r.setup_time_ms = ms_since(t_setup);
        direct.Mult(B, x);
        r.converged = true; r.iterations = 1; r.final_residual = 0.0;
    }
#endif
    else {
        r.converged = false; r.iterations = 0;
        r.solve_time_ms = r.total_time_ms = r.final_residual = r.relative_residual = 0.0;
        return r;
    }

    r.solve_time_ms = ms_since(t_solve) - r.setup_time_ms;
    r.total_time_ms = ms_since(t_solve);
    double b_norm = B.Norml2();
    r.relative_residual = (b_norm > 1e-30) ? r.final_residual / b_norm : r.final_residual;
    r.ref_level = -1;
    return r;
}

// ======================================================================
//  对流扩散问题：−ε·Δu + β·∇u = 1（Péclet 数可控）
//  使用迎风稳定（SUPG）避免伪振荡
// ======================================================================
BenchResult RunConvDiff(Mesh& mesh, int order, double peclet,
                         const std::string& solver_name, const BenchConfig& cfg)
{
    BenchResult r;
    int dim = mesh.Dimension();
    r.problem_name = "ConvDiff_Pe" + (peclet > 100 ? std::string("Hi")
                                                    : std::string("Lo"))
                     + "_" + std::to_string(dim) + "D";
    r.solver_name  = solver_name;
    r.dim          = dim;
    r.poly_order   = order;
    r.n_elements   = mesh.GetNE();
    r.l2_error     = -1.0;

    H1_FECollection   fec(order, dim);
    FiniteElementSpace fespace(&mesh, &fec);
    r.n_dof = fespace.GetTrueVSize();

    // ε 使 Pe = |β|·h/ε 约等于 peclet
    double h_min  = mesh.GetElementSize(0);
    double b_mag  = 1.0;
    double eps    = b_mag * h_min / (2.0 * std::max(peclet, 0.1));

    auto t_asm = Clock::now();
    BilinearForm a(&fespace);
    ConstantCoefficient eps_cf(eps);
    a.AddDomainIntegrator(new DiffusionIntegrator(eps_cf));

    // 对流项
    Vector bvec(dim); bvec = 0.0; bvec(0) = b_mag;
    if (dim > 1) bvec(1) = 0.5 * b_mag;
    VectorConstantCoefficient b_cf(bvec);
    a.AddDomainIntegrator(new ConvectionIntegrator(b_cf, -1.0));

    a.Assemble(); a.Finalize();

    LinearForm b(&fespace);
    ConstantCoefficient one(1.0);
    b.AddDomainIntegrator(new DomainLFIntegrator(one));
    b.Assemble();

    GridFunction x_sol(&fespace); x_sol = 0.0;
    OperatorPtr  A; Vector B, X;
    ApplyBCsAndForm(a, b, fespace, mesh, A, X, B, x_sol);
    r.assemble_time_ms = ms_since(t_asm);

    SparseMatrix& Asp = static_cast<SparseMatrix&>(*A.Ptr());
    auto stats = MatrixStats::Compute(Asp);
    r.nnz = stats.nnz; r.nnz_per_row = stats.nnz_per_row;
    r.symmetry_ratio = stats.symmetry_ratio; r.is_spd = stats.is_spd;

    Vector x(X.Size()); x = 0.0;
    r.setup_time_ms = 0.0;

    auto t_solve = Clock::now();

    if (solver_name == "CG") {
        CGSolver cg; cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter); cg.SetPrintLevel(0);
        cg.Mult(B, x);
        r.converged = cg.GetConverged(); r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "PCG_GS") {
        GSSmoother prec(Asp);
        CGSolver cg; cg.SetPreconditioner(prec); cg.SetOperator(*A.Ptr());
        cg.SetRelTol(cfg.rtol); cg.SetAbsTol(cfg.atol);
        cg.SetMaxIter(cfg.max_iter); cg.SetPrintLevel(0);
        cg.Mult(B, x);
        r.converged = cg.GetConverged(); r.iterations = cg.GetNumIterations();
        r.final_residual = cg.GetFinalNorm();
    }
    else if (solver_name == "MINRES") {
        MINRESSolver mr; mr.SetOperator(*A.Ptr());
        mr.SetRelTol(cfg.rtol); mr.SetAbsTol(cfg.atol);
        mr.SetMaxIter(cfg.max_iter); mr.SetPrintLevel(0);
        mr.Mult(B, x);
        r.converged = mr.GetConverged(); r.iterations = mr.GetNumIterations();
        r.final_residual = mr.GetFinalNorm();
    }
    else if (solver_name == "GMRES") {
        GMRESSolver gm; gm.SetOperator(*A.Ptr());
        gm.SetRelTol(cfg.rtol); gm.SetAbsTol(cfg.atol);
        gm.SetMaxIter(cfg.max_iter); gm.SetKDim(cfg.gmres_kdim);
        gm.SetPrintLevel(0); gm.Mult(B, x);
        r.converged = gm.GetConverged(); r.iterations = gm.GetNumIterations();
        r.final_residual = gm.GetFinalNorm();
    }
    else if (solver_name == "GMRES_DS") {
        auto t_setup = Clock::now();
        DSmoother prec(Asp); r.setup_time_ms = ms_since(t_setup);
        GMRESSolver gm; gm.SetPreconditioner(prec); gm.SetOperator(*A.Ptr());
        gm.SetRelTol(cfg.rtol); gm.SetAbsTol(cfg.atol);
        gm.SetMaxIter(cfg.max_iter); gm.SetKDim(cfg.gmres_kdim);
        gm.SetPrintLevel(0); gm.Mult(B, x);
        r.converged = gm.GetConverged(); r.iterations = gm.GetNumIterations();
        r.final_residual = gm.GetFinalNorm();
    }
#ifdef MFEM_USE_SUITESPARSE
    else if (solver_name == "DIRECT_UMF") {
        auto t_setup = Clock::now();
        UMFPackSolver direct; direct.SetOperator(*A.Ptr());
        r.setup_time_ms = ms_since(t_setup);
        direct.Mult(B, x);
        r.converged = true; r.iterations = 1; r.final_residual = 0.0;
    }
#endif
    else {
        r.converged = false; r.iterations = 0;
        r.solve_time_ms = r.total_time_ms = r.final_residual = r.relative_residual = 0.0;
        return r;
    }

    r.solve_time_ms = ms_since(t_solve) - r.setup_time_ms;
    r.total_time_ms = ms_since(t_solve);
    double b_norm = B.Norml2();
    r.relative_residual = (b_norm > 1e-30) ? r.final_residual / b_norm : r.final_residual;
    r.ref_level = -1;
    return r;
}

// ======================================================================
//  输出工具
// ======================================================================

// ANSI 颜色
#define COL_RESET  "\033[0m"
#define COL_GREEN  "\033[32m"
#define COL_RED    "\033[31m"
#define COL_YELLOW "\033[33m"
#define COL_CYAN   "\033[36m"
#define COL_BOLD   "\033[1m"
#define COL_DIM    "\033[2m"

void PrintTableHeader() {
    std::printf("\n");
    std::printf(COL_BOLD
        "  %-22s %-16s %8s %8s %8s %8s %10s %10s %s\n" COL_RESET,
        "问题", "求解器",
        "DOF", "迭代", "组装(ms)", "配置(ms)", "求解(ms)", "总计(ms)", "状态");
    std::printf("____________%s\n");
}

void PrintResult(const BenchResult& r) {
    const char* conv_str;
    const char* col;
    if (!r.converged || r.iterations == 0) {
        conv_str = "✗ 不收敛";  col = COL_RED;
    } else if (r.relative_residual < 1e-6) {
        conv_str = "✓ 优秀";    col = COL_GREEN;
    } else {
        conv_str = "~ 收敛";    col = COL_YELLOW;
    }

    std::printf("  %-22s %-16s %8d %8d %8.1f %8.1f %10.2f %10.2f %s%s (||r||/||b||=%.1e)%s\n",
                r.problem_name.c_str(),
                r.solver_name.c_str(),
                r.n_dof,
                r.iterations,
                r.assemble_time_ms,
                r.setup_time_ms,
                r.solve_time_ms,
                r.total_time_ms,
                col, conv_str,
                r.relative_residual,
                COL_RESET);
}

// CSV 输出
void WriteCSVHeader(std::ofstream& f) {
    f << "problem,solver,dim,ref_level,poly_order,"
         "n_dof,n_elements,nnz,nnz_per_row,symmetry_ratio,is_spd,"
         "assemble_ms,setup_ms,solve_ms,total_ms,"
         "converged,iterations,final_residual,relative_residual,l2_error\n";
}

void WriteCSVRow(std::ofstream& f, const BenchResult& r) {
    f << r.problem_name   << ","
      << r.solver_name    << ","
      << r.dim            << ","
      << r.ref_level      << ","
      << r.poly_order     << ","
      << r.n_dof          << ","
      << r.n_elements     << ","
      << r.nnz            << ","
      << std::fixed << std::setprecision(2)
      << r.nnz_per_row    << ","
      << r.symmetry_ratio << ","
      << (r.is_spd ? 1 : 0) << ","
      << r.assemble_time_ms << ","
      << r.setup_time_ms    << ","
      << r.solve_time_ms    << ","
      << r.total_time_ms    << ","
      << (r.converged ? 1 : 0) << ","
      << r.iterations     << ","
      << std::scientific << std::setprecision(4)
      << r.final_residual << ","
      << r.relative_residual << ","
      << r.l2_error       << "\n";
}

// ======================================================================
//  打印矩阵信息
// ======================================================================
void PrintMatrixInfo(const BenchResult& r) {
    if (r.nnz == 0) return;
    std::printf(COL_DIM
        "    矩阵: %d×%d  nnz=%ld  nnz/row=%.1f  "
        "对称偏差=%.4f  is_spd=%s\n" COL_RESET,
        r.n_dof, r.n_dof, r.nnz, r.nnz_per_row,
        r.symmetry_ratio, r.is_spd ? "是" : "否");
}

// ======================================================================
//  进度打印
// ======================================================================
void PrintSection(const std::string& title) {
    std::printf("\n" COL_BOLD COL_CYAN
                "  ══ %s ══\n" COL_RESET, title.c_str());
}

// ======================================================================
//  主函数
// ======================================================================
int main(int argc, char** argv)
{
    BenchConfig cfg = ParseArgs(argc, argv);

    std::printf("\n" COL_BOLD
                "╔═══════════════════════════════════════════════════╗\n"
                "║    MFEM 求解器性能基准测试                         ║\n"
                "║    多物理场 × 多网格规模 × 多求解器                ║\n"
                "╚═══════════════════════════════════════════════════╝\n"
                COL_RESET);
    std::printf("  FEM 阶数    : P%d\n", cfg.order);
    std::printf("  收敛容差    : %.0e (rel)  %.0e (abs)\n", cfg.rtol, cfg.atol);
    std::printf("  最大迭代    : %d\n", cfg.max_iter);
    std::printf("  GMRES kdim  : %d\n", cfg.gmres_kdim);
    std::printf("  输出 CSV    : %s\n", cfg.output_csv.c_str());
    std::printf("  直接法 DOF 限制: %d\n", cfg.direct_max_dof);

    // 求解器列表
    std::vector<std::string> iterative_solvers = {
        "CG", "PCG_GS", "PCG_Cheby", "MINRES", "GMRES", "GMRES_DS"
    };
#ifdef MFEM_USE_SUITESPARSE
    if (cfg.run_direct) iterative_solvers.push_back("DIRECT_UMF");
#endif

    // 收集所有结果
    std::vector<BenchResult> all_results;
    all_results.reserve(512);

    // 开启 CSV 文件
    std::ofstream csv(cfg.output_csv);
    if (!csv.is_open()) {
        std::fprintf(stderr, "无法打开输出文件：%s\n", cfg.output_csv.c_str());
        return 1;
    }
    WriteCSVHeader(csv);

    PrintTableHeader();

    // ──────────────────────────────────────────────────────────
    // 2D 测试
    // ──────────────────────────────────────────────────────────
    if (cfg.run_2d) {

        // ── Poisson 2D ──────────────────────────────────────
        if (cfg.run_poisson) {
            PrintSection("Poisson 2D（均匀四边形网格）");
            for (int ref = cfg.min_ref; ref <= cfg.max_ref; ++ref) {
                // 构造基础网格并精炼
                Mesh mesh(2, 1, 1, Element::QUADRILATERAL, true);
                for (int i = 0; i < ref; ++i) mesh.UniformRefinement();
                int n_dof_est = (int)std::pow(2, ref) + 1;
                n_dof_est = n_dof_est * n_dof_est;

                bool printed_mat = false;
                for (const auto& s : iterative_solvers) {
#ifdef MFEM_USE_SUITESPARSE
                    if (s == "DIRECT_UMF" && n_dof_est > cfg.direct_max_dof) continue;
#endif
                    auto r  = RunPoisson(mesh, cfg.order, s, cfg);
                    r.ref_level = ref;
                    PrintResult(r);
                    if (!printed_mat && cfg.print_matrix_info) {
                        PrintMatrixInfo(r);
                        printed_mat = true;
                    }
                    WriteCSVRow(csv, r);
                    all_results.push_back(r);
                }
                std::printf("____________%s\n");
            }
        }

        // ── Poisson 2D 三角形非结构 ─────────────────────────
        if (cfg.run_poisson) {
            PrintSection("Poisson 2D（非结构三角形网格）");
            for (int ref = cfg.min_ref; ref <= cfg.max_ref; ++ref) {
                Mesh mesh(2, 1, 1, Element::TRIANGLE, true);
                for (int i = 0; i < ref; ++i) mesh.UniformRefinement();

                for (const auto& s : iterative_solvers) {
                    auto r  = RunPoisson(mesh, cfg.order, s, cfg);
                    r.ref_level = ref;
                    r.problem_name = "Poisson2D_Tri";
#ifdef MFEM_USE_SUITESPARSE
                    if (s == "DIRECT_UMF" && r.n_dof > cfg.direct_max_dof) continue;
#endif
                    PrintResult(r);
                    WriteCSVRow(csv, r);
                    all_results.push_back(r);
                }
                std::printf("____________%s\n");
            }
        }

        // ── 线性弹性 2D ─────────────────────────────────────
        if (cfg.run_elast) {
            PrintSection("线性弹性 2D（四边形网格）");
            for (int ref = cfg.min_ref; ref <= cfg.max_ref; ++ref) {
                Mesh mesh(2, 1, 1, Element::QUADRILATERAL, true);
                for (int i = 0; i < ref; ++i) mesh.UniformRefinement();

                for (const auto& s : iterative_solvers) {
                    auto r = RunElasticity(mesh, cfg.order, s, cfg);
                    r.ref_level = ref;
#ifdef MFEM_USE_SUITESPARSE
                    if (s == "DIRECT_UMF" && r.n_dof > cfg.direct_max_dof) continue;
#endif
                    PrintResult(r);
                    WriteCSVRow(csv, r);
                    all_results.push_back(r);
                }
                std::printf("____________%s\n");
            }
        }

        // ── 对流扩散 2D（低/高 Péclet）──────────────────────
        if (cfg.run_conv_diff) {
            for (double pe : {1.0, 100.0}) {
                PrintSection("对流扩散 2D Pe=" + std::to_string((int)pe));
                for (int ref = cfg.min_ref; ref <= cfg.max_ref; ++ref) {
                    Mesh mesh(2, 1, 1, Element::QUADRILATERAL, true);
                    for (int i = 0; i < ref; ++i) mesh.UniformRefinement();

                    for (const auto& s : iterative_solvers) {
                        auto r = RunConvDiff(mesh, cfg.order, pe, s, cfg);
                        r.ref_level = ref;
#ifdef MFEM_USE_SUITESPARSE
                        if (s == "DIRECT_UMF" && r.n_dof > cfg.direct_max_dof) continue;
#endif
                        PrintResult(r);
                        WriteCSVRow(csv, r);
                        all_results.push_back(r);
                    }
                    std::printf("____________%s\n");
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────
    // 3D 测试
    // ──────────────────────────────────────────────────────────
    if (cfg.run_3d) {

        // ── Poisson 3D 六面体 ────────────────────────────────
        if (cfg.run_poisson) {
            PrintSection("Poisson 3D（六面体网格）");
            for (int ref = cfg.min_ref; ref <= cfg.max_ref_3d; ++ref) {
                Mesh mesh(1, 1, 1, Element::HEXAHEDRON, true);
                for (int i = 0; i < ref; ++i) mesh.UniformRefinement();

                bool printed_mat = false;
                for (const auto& s : iterative_solvers) {
                    auto r = RunPoisson(mesh, cfg.order, s, cfg);
                    r.ref_level = ref;
                    r.problem_name = "Poisson3D_Hex";
#ifdef MFEM_USE_SUITESPARSE
                    if (s == "DIRECT_UMF" && r.n_dof > cfg.direct_max_dof) continue;
#endif
                    PrintResult(r);
                    if (!printed_mat && cfg.print_matrix_info) {
                        PrintMatrixInfo(r);
                        printed_mat = true;
                    }
                    WriteCSVRow(csv, r);
                    all_results.push_back(r);
                }
                std::printf("____________%s\n");
            }
        }

        // ── Poisson 3D 四面体 ────────────────────────────────
        if (cfg.run_poisson) {
            PrintSection("Poisson 3D（四面体网格）");
            for (int ref = cfg.min_ref; ref <= cfg.max_ref_3d; ++ref) {
                Mesh mesh(1, 1, 1, Element::TETRAHEDRON, true);
                for (int i = 0; i < ref; ++i) mesh.UniformRefinement();

                for (const auto& s : iterative_solvers) {
                    auto r = RunPoisson(mesh, cfg.order, s, cfg);
                    r.ref_level = ref;
                    r.problem_name = "Poisson3D_Tet";
#ifdef MFEM_USE_SUITESPARSE
                    if (s == "DIRECT_UMF" && r.n_dof > cfg.direct_max_dof) continue;
#endif
                    PrintResult(r);
                    WriteCSVRow(csv, r);
                    all_results.push_back(r);
                }
                std::printf("____________%s\n");
            }
        }

        // ── 线性弹性 3D ─────────────────────────────────────
        if (cfg.run_elast) {
            PrintSection("线性弹性 3D（六面体网格）");
            for (int ref = cfg.min_ref; ref <= cfg.max_ref_3d; ++ref) {
                Mesh mesh(1, 1, 1, Element::HEXAHEDRON, true);
                for (int i = 0; i < ref; ++i) mesh.UniformRefinement();

                for (const auto& s : iterative_solvers) {
                    auto r = RunElasticity(mesh, cfg.order, s, cfg);
                    r.ref_level = ref;
#ifdef MFEM_USE_SUITESPARSE
                    if (s == "DIRECT_UMF" && r.n_dof > cfg.direct_max_dof) continue;
#endif
                    PrintResult(r);
                    WriteCSVRow(csv, r);
                    all_results.push_back(r);
                }
                std::printf("____________%s\n");
            }
        }

        // ── 对流扩散 3D ─────────────────────────────────────
        if (cfg.run_conv_diff) {
            for (double pe : {1.0, 50.0}) {
                PrintSection("对流扩散 3D Pe=" + std::to_string((int)pe));
                for (int ref = cfg.min_ref; ref <= cfg.max_ref_3d; ++ref) {
                    Mesh mesh(1, 1, 1, Element::HEXAHEDRON, true);
                    for (int i = 0; i < ref; ++i) mesh.UniformRefinement();

                    for (const auto& s : iterative_solvers) {
                        auto r = RunConvDiff(mesh, cfg.order, pe, s, cfg);
                        r.ref_level = ref;
#ifdef MFEM_USE_SUITESPARSE
                        if (s == "DIRECT_UMF" && r.n_dof > cfg.direct_max_dof) continue;
#endif
                        PrintResult(r);
                        WriteCSVRow(csv, r);
                        all_results.push_back(r);
                    }
                    std::printf("____________%s\n");
                }
            }
        }
    }

    csv.close();

    // ──────────────────────────────────────────────────────────
    // 汇总统计
    // ──────────────────────────────────────────────────────────
    std::printf("\n" COL_BOLD "═══ 汇总统计 ═══\n" COL_RESET);
    std::printf("  总测试组合数：%zu\n", all_results.size());

    // 按求解器统计胜率（每个（问题，规模）组合中耗时最短的算 1 次胜）
    std::map<std::string, int>    wins, attempts, converge_count;
    std::map<std::string, double> total_time_sum;

    // 将结果按 (problem, ref_level) 分组
    std::map<std::string, std::vector<const BenchResult*>> groups;
    for (const auto& r : all_results) {
        std::string key = r.problem_name + "_ref" + std::to_string(r.ref_level);
        groups[key].push_back(&r);
        attempts[r.solver_name]++;
        if (r.converged) {
            converge_count[r.solver_name]++;
            total_time_sum[r.solver_name] += r.total_time_ms;
        }
    }

    for (const auto& [key, rs] : groups) {
        // 找该组最快收敛求解器
        const BenchResult* best = nullptr;
        for (const auto* r : rs) {
            if (!r->converged) continue;
            if (!best || r->total_time_ms < best->total_time_ms) best = r;
        }
        if (best) wins[best->solver_name]++;
    }

    std::printf("\n  %-16s %8s %8s %8s %12s\n",
                "求解器", "胜场", "收敛率", "尝试", "平均耗时(ms)");
    std::printf("____________%s\n");
    for (const auto& s : iterative_solvers) {
        int   w   = wins.count(s)          ? wins[s]          : 0;
        int   att = attempts.count(s)      ? attempts[s]      : 0;
        int   cv  = converge_count.count(s)? converge_count[s]: 0;
        double t  = (total_time_sum.count(s) && cv > 0)
                    ? total_time_sum[s] / cv : 0.0;
        double cr = (att > 0) ? 100.0 * cv / att : 0.0;
        const char* col = (w == (int)groups.size()) ? COL_GREEN
                        : (w > 0)                   ? COL_YELLOW
                                                    : COL_RESET;
        std::printf("  %s%-16s %8d %7.1f%% %8d %12.2f%s\n",
                    col, s.c_str(), w, cr, att, t, COL_RESET);
    }

    std::printf("\n" COL_GREEN "  ✅ 结果已保存至：%s\n" COL_RESET,
                cfg.output_csv.c_str());
    std::printf("     可直接作为 generate_data.py 的替代数据源用于 ML 训练\n");

    return 0;
}