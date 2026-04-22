/**
 * @file solver.hpp
 * @author Maria Teresa Cazzola
 * @brief Field solver class for the PIC simulator
 * @date 2026-04-22
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef SOLVER
#define SOLVER

#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>  
#include <deal.II/numerics/data_out.h>
#include <deal.II/dofs/dof_handler.h> 

#include "simulator.hpp"

namespace PICsimulator{
    using namespace dealii;
        
    template <int dim>
    class Solver{
    public:
        explicit Solver(DoFHandler<dim>& dof_h_):
            dof_handler(dof_h_),
            fe(1)
            {};

        void run_solver_test(){

            // setup system
            dof_handler.distribute_dofs(fe);

            DynamicSparsityPattern dsp(dof_handler.n_dofs());
            DoFTools::make_sparsity_pattern(dof_handler, dsp);
            sparsity_pattern.copy_from(dsp);

            system_matrix.reinit(sparsity_pattern);
            solution.reinit(dof_handler.n_dofs());
            system_rhs.reinit(dof_handler.n_dofs());
            //
  
            // assemble system
            const QGauss<2> quadrature_formula(fe.degree+1);
            FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);
            
            const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
 
            FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
            Vector<double>     cell_rhs(dofs_per_cell);
            
            std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

             
            for (const auto &cell : dof_handler.active_cell_iterators())
            {
                fe_values.reinit(cell);
            
                cell_matrix = 0;
                cell_rhs    = 0;
            
                for (const unsigned int q_index : fe_values.quadrature_point_indices())
                    {
                    for (const unsigned int i : fe_values.dof_indices())
                        for (const unsigned int j : fe_values.dof_indices())
                        cell_matrix(i, j) +=
                            (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                            fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                            fe_values.JxW(q_index));           // dx
            
                    for (const unsigned int i : fe_values.dof_indices())
                        cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                        1. *                                // f(x_q)
                                        fe_values.JxW(q_index));            // dx
                    }
                cell->get_dof_indices(local_dof_indices);
            
                for (const unsigned int i : fe_values.dof_indices())
                    for (const unsigned int j : fe_values.dof_indices())
                    system_matrix.add(local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i, j));
            
                for (const unsigned int i : fe_values.dof_indices())
                    system_rhs(local_dof_indices[i]) += cell_rhs(i);
            }

            std::map<types::global_dof_index, double> boundary_values;
            VectorTools::interpolate_boundary_values(dof_handler,
                                                    types::boundary_id(0),
                                                    Functions::ZeroFunction<2>(),
                                                    boundary_values);
            MatrixTools::apply_boundary_values(boundary_values,
                                                system_matrix,
                                                solution,
                                                system_rhs);

            //

            // SOLVER
            SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
            SolverCG<Vector<double>> solver(solver_control);
            solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
            
            std::cout << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
            //


            // OUTPUT

            DataOut<2> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(solution, "solution");
            data_out.build_patches();
            
            const std::string filename = "outputs/solution.vtk";
            std::ofstream     output(filename);
            data_out.write_vtk(output);
            std::cout << "Output written to " << filename << std::endl;

            //
        };
        
    private:
        //////////// methods
        // void setup_system();
        // void assemble_system();
        // void solve();
        // void print();

        //////////// members
        DoFHandler<dim>& dof_handler;
        const dealii::FE_Q<dim> fe;
        dealii::SparsityPattern sparsity_pattern;
        dealii::SparseMatrix<double> system_matrix;
        dealii::Vector<double> solution;
        dealii::Vector<double> system_rhs;

    }; // end Solver class

} //end PICsimulator namespace


#endif // SOLVER    