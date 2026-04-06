// ============================================================
//  用 MFEM 在 4×4 均匀网格上求解 Poisson 方程
//
//    -Δu = f,   Ω = [0,1]²
//    u = 0  (Dirichlet 边界)
//    f = 2π² sin(πx) sin(πy)
//    精确解: u = sin(πx) sin(πy)
//
//  编译示例 (假设 MFEM 已安装到 /usr/local/mfem):
//    g++ -std=c++14 -O2 poisson_4x4.cpp \
//        -I/usr/local/mfem/include \
//        -L/usr/local/mfem/lib -lmfem -lm -o poisson_4x4
// ============================================================

#include "mfem.hpp"
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace mfem;

// 右端项 f = 2π² sin(πx)sin(πy)
double rhs_func(const Vector &x)
{
    return 2.0 * M_PI * M_PI * sin(M_PI * x[0]) * sin(M_PI * x[1]);
}

// 精确解 u = sin(πx)sin(πy)
double exact_sol(const Vector &x)
{
    return sin(M_PI * x[0]) * sin(M_PI * x[1]);
}

int main(int argc, char *argv[])
{
    // ── 1. 参数 ──────────────────────────────────────────────
    int nx = 4;          // x 方向单元数
    int ny = 4;          // y 方向单元数
    int order = 1;       // FEM 多项式阶次

    cout << "========================================\n";
    cout << "  MFEM Poisson 求解器  (4×4 网格)\n";
    cout << "========================================\n";

    // ── 2. 创建正方形网格，划分为 4×4 四边形单元 ────────────
    Mesh mesh = Mesh::MakeCartesian2D(
        nx, ny,
        Element::QUADRILATERAL,   // 单元类型：四边形
        true,                     // sfc_ordering
        1.0, 1.0                  // 区域大小 [0,1]²
    );

    int dim = mesh.Dimension();
    int nelem = mesh.GetNE();
    int nvert = mesh.GetNV();
    int nedge = mesh.GetNEdges();

    cout << "\n[网格信息]\n";
    cout << "  空间维度   : " << dim    << "\n";
    cout << "  单元总数   : " << nelem  << " (= " << nx << "×" << ny << ")\n";
    cout << "  节点总数   : " << nvert  << "\n";
    cout << "  边总数     : " << nedge  << "\n";
    cout << "  最小 h     : " << mesh.GetElementSize(0) << "\n";

    // ── 3. 打印每个单元的顶点坐标 ────────────────────────────
    cout << "\n[各单元顶点坐标]\n";
    cout << setw(6) << "单元" << "  顶点 (x, y)\n";
    Array<int> elem_verts;
    for (int e = 0; e < nelem; e++)
    {
        mesh.GetElementVertices(e, elem_verts);
        cout << "  [" << setw(2) << e << "] ";
        for (int v = 0; v < elem_verts.Size(); v++)
        {
            double *coord = mesh.GetVertex(elem_verts[v]);
            cout << "(" << fixed << setprecision(3)
                 << coord[0] << "," << coord[1] << ") ";
        }
        cout << "\n";
    }

    // ── 4. 定义有限元空间 (H1, 阶次 order) ───────────────────
    H1_FECollection fec(order, dim);
    FiniteElementSpace fespace(&mesh, &fec);

    cout << "\n[有限元空间]\n";
    cout << "  FE 集合    : " << fec.Name() << "，阶次=" << order << "\n";
    cout << "  总自由度   : " << fespace.GetTrueVSize() << "\n";

    // ── 5. 设置 Dirichlet 边界 ────────────────────────────────
    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mesh.bdr_attributes.Max());
    ess_bdr = 1;   // 所有边界均施加 Dirichlet
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // ── 6. 组装线性系统 ───────────────────────────────────────
    FunctionCoefficient f_coeff(rhs_func);

    LinearForm b(&fespace);
    b.AddDomainIntegrator(new DomainLFIntegrator(f_coeff));
    b.Assemble();

    ConstantCoefficient one(1.0);
    BilinearForm a(&fespace);
    a.AddDomainIntegrator(new DiffusionIntegrator(one));
    a.Assemble();

    GridFunction u(&fespace);
    u = 0.0;   // 初值 (满足零 Dirichlet 条件)

    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

    cout << "\n[线性系统]\n";
    cout << "  矩阵大小   : " << X.Size() << " × " << X.Size() << "\n";

    // ── 7. 求解 (PCG + GS 预条件) ────────────────────────────
    GSSmoother M((SparseMatrix &)(*A));
    PCG(*A, M, B, X, 1, 500, 1e-12, 0.0);

    a.RecoverFEMSolution(X, b, u);
    cout << "  PCG 求解完毕\n";

    // ── 8. 计算 L2 误差 ───────────────────────────────────────
    FunctionCoefficient exact_coeff(exact_sol);
    double err = u.ComputeL2Error(exact_coeff);

    cout << "\n[精度分析]\n";
    cout << "  ||u_h - u_exact||_L2 = " << scientific << setprecision(4) << err << "\n";

    // ── 9. 打印解在各节点的值 ─────────────────────────────────
    cout << "\n[节点解值 (x, y, u_h, u_exact, 误差)]\n";
    cout << fixed << setprecision(5);
    cout << setw(8) << "x"
         << setw(10) << "y"
         << setw(12) << "u_h"
         << setw(12) << "u_exact"
         << setw(12) << "误差\n";

    const FiniteElementSpace *vfes = &fespace;
    for (int i = 0; i < vfes->GetNDofs(); i++)
    {
        // 获取第 i 个 DOF 的坐标
        IntegrationPoint ip;
        ip.Set2(0.0, 0.0);
        // 通过网格顶点坐标遍历 (order=1 时 DOF 与节点一一对应)
    }

    // // 遍历网格节点
    // for (int v = 0; v < mesh.GetNV(); v++)
    // {
    //     double *coord = mesh.GetVertex(v);
    //     Vector pt(coord, 2);

    //     // 在该点求解
    //     double uh  = u.GetValue(v,);   // order=1 时节点值即 DOF 值
    //     double uex = exact_sol(pt);
    //     cout << setw(8)  << coord[0]
    //          << setw(10) << coord[1]
    //          << setw(12) << uh
    //          << setw(12) << uex
    //          << setw(12) << fabs(uh - uex) << "\n";
    // }

    // ── 10. 保存结果文件 (可用 VisIt / GLVis 可视化) ─────────
    {
        ofstream mesh_ofs("square_4x4.mesh");
        mesh_ofs.precision(8);
        mesh.Print(mesh_ofs);

        ofstream sol_ofs("solution.gf");
        sol_ofs.precision(8);
        u.Save(sol_ofs);
    }
    cout << "\n[文件输出]\n";
    cout << "  网格 → square_4x4.mesh\n";
    cout << "  解   → solution.gf\n";
    cout << "  (可用 'glvis -m square_4x4.mesh -g solution.gf' 可视化)\n";

    cout << "\n========================================\n";
    cout << "  完成！\n";
    cout << "========================================\n";
    return 0;
}