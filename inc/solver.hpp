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

    // forward declaration of Simulator
    template <int dim>
    class Simulator;

    /// @brief solves the electrostatic potential field.
    /// @details Assembles and solves the Poisson problem, then provides the 
    /// electric potential from which the electric field is computed.
    /// @tparam dim
    template <int dim>
    class Solver{

        using species_type = Species::MultiSpecies;
        using ph_type      = dealii::Particles::ParticleHandler<dim>;
        using vec_ph       = std::vector<ph_type>;

    public:
        /// @brief constructs the solver.
        /// @param dof_h_ DOF handler associated with the simulation mesh
        /// @param mapping_ mapping used for real-cell transformations
        /// @param phs_ particle handler vector
        /// @param ms_ multispecies data structure
        explicit Solver(DoFHandler<dim>& dof_h_,
                        const MappingQ<dim>& mapping_,
                        vec_ph& phs_,
                        species_type& ms_)
            :
            dof_handler(dof_h_),
            mapping(mapping_),
            phs(phs_),
            ms(ms_),
            fe(1),
            quadrature_formula(fe.degree+1),
            fe_values(fe, quadrature_formula,
                    update_values | update_gradients | update_JxW_values)
            {};

        /// @brief initializes linear algebra structures and DOF data
        void setup_system();

        /// @brief assembles the stiffness matrix of the Poisson problem
        /// @details builds the matrix associated with -Laplace(phi)
        void assemble_matrix();

        /// @brief computes and assembles the right-hand side vector 
        /// @details evaluates the charge density contribution
        /// integral(rho(x)/eps_0 * phi_i(x) dx) 
        void assemble_rhs();

        /// @brief applies boundary conditions to matrix and right-hand side 
        void add_bc();

        /// @brief solves the linear system for the electric potential 
        void solve_potential();

        /// @brief writes solution data to output files 
        void save_output();

        /// @brief runs a standalone test of the solver 
        void run_solver_test();

    private:
        //////////// members

        /// @brief DOF handler of the simulation mesh 
        DoFHandler<dim>& dof_handler;

        /// @brief mapping between reference and real cells 
        const dealii::MappingQ<dim>& mapping;

        /// @brief particle handler vector
        vec_ph& phs;

        /// @brief multispecies container
        const species_type& ms;

        /// @brief scalar Lagrange finite element used for the potential
        const dealii::FE_Q<dim> fe;

        /// @brief Gauss quadrature rule used for cell integration
        const QGauss<dim> quadrature_formula;

        /// @brief finite-element evaluator on quadrature points
        FEValues<dim> fe_values;

        /// @brief sparsity pattern describing the nonzero structure of the system matrix
        dealii::SparsityPattern sparsity_pattern;

        /// @brief global sparse matrix of the linear system 
        dealii::SparseMatrix<double> system_matrix;

        /// @brief solution vector containing the nodal potential 
        dealii::Vector<double> solution;

        /// @brief right-hand side vector of the linear system 
        dealii::Vector<double> system_rhs;

    }; // end Solver class

    /* DEFINITIONS */

    template <int dim>
    inline void Solver<dim>::setup_system() {
        dof_handler.distribute_dofs(fe);

        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sparsity_pattern.copy_from(dsp);

        system_matrix.reinit(sparsity_pattern);
        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());

        LOGVAR("ndof", dof_handler.n_dofs());
    } // end setup system

    template <int dim>
    inline void Solver<dim>::assemble_matrix()  {
        LOG("ASSEMBLE_MATRIX");
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

        FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
            
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);
        
            cell_matrix = 0;

            // * compute matrix *
            for (const unsigned int q_index : fe_values.quadrature_point_indices())
            {
                for (const unsigned int i : fe_values.dof_indices())
                {
                    for (const unsigned int j : fe_values.dof_indices())
                        cell_matrix(i, j) += 
                            (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                            fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                            fe_values.JxW(q_index));
                }
            }
            cell->get_dof_indices(local_dof_indices);
        
            for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                system_matrix.add(local_dof_indices[i],
                                    local_dof_indices[j],
                                    cell_matrix(i, j));
        }
    } // end assemble_matrix

    template <int dim>
    inline void Solver<dim>::assemble_rhs() {
        LOG("ASSEMBLE_RHS");
        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

        Vector<double>     cell_rhs(dofs_per_cell);
        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
            
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            fe_values.reinit(cell);

            cell_rhs    = 0;

            // * compute rhs *
            for(unsigned int a = 0; a<ms.n_species(); ++a){
                unsigned int n_part = phs[a].n_particles_in_cell(cell);
                
                double q = ms[a].get_charge()/EPS0;

                if(q!=0 && n_part>0){

                    auto pic = phs[a].particles_in_cell(cell);
                    for(auto& it_p : pic) {
                        Point<dim> part_loc(it_p.get_location());
                        Point<dim> ref_loc = mapping.transform_real_to_unit_cell(cell, part_loc);

                        for(unsigned int i : fe_values.dof_indices()){
                            cell_rhs(i) += q* fe.shape_value(i,ref_loc);
                        }
                    // LOGVEC("particle_loc", part_loc, 2);
                    // LOGVEC("ref_loc", ref_loc, 2);
                    // LOGVAR("charge", q);
                    // LOGVAR("phi0", fe.shape_value(0,ref_loc));
                    // LOGVAR("phi1", fe.shape_value(1,ref_loc));
                    // LOGVAR("phi2", fe.shape_value(2,ref_loc));
                    // LOGVAR("phi3", fe.shape_value(3,ref_loc));

                    // LOGVAR("sum", fe.shape_value(0,ref_loc)+fe.shape_value(1,ref_loc)+fe.shape_value(2,ref_loc)+fe.shape_value(3,ref_loc));
                    // LOG(" -- ");
                    }

                }
            }
        
            cell->get_dof_indices(local_dof_indices);
        
            for (const unsigned int i : fe_values.dof_indices())
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }; // end assemble rhs

    template <int dim>
    inline void Solver<dim>::add_bc()
    {
        // // Neumann + 1 fixed point
        // std::map<types::global_dof_index, double> boundary_values;
        // boundary_values[0] = 0.0; 
        // MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

        // // DIRICHLET
        // std::map<types::global_dof_index, double> boundary_values;
        // VectorTools::interpolate_boundary_values(dof_handler,
        //                                         types::boundary_id(0),
        //                                         Functions::ZeroFunction<dim>(),
        //                                         boundary_values);
        // MatrixTools::apply_boundary_values(boundary_values,
        //                                     system_matrix,
        //                                     solution,
        //                                     system_rhs);

    }

    template <int dim>
    inline void Solver<dim>::solve_potential() {
        SolverControl            solver_control(1000, 1e-6 * system_rhs.l2_norm());
        SolverCG<Vector<double>> solver(solver_control);
        solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
        
        std::cout << solver_control.last_step()
        << " CG iterations needed to obtain convergence." << std::endl;

        // Normalise the solution 
        QGauss<dim> quadrature(fe.degree + 1);
        double mean_value = VectorTools::compute_mean_value(dof_handler, quadrature, solution, 0);
        solution.add(-mean_value);

    } // end solve_potential

    template <int dim>
    inline void Solver<dim>::save_output() {

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        
        const std::string filename = "outputs/solution.vtk";
        std::ofstream     output(filename);
        data_out.write_vtk(output);
        std::cout << "Output written to " << filename << std::endl;
    } // end save output

    template <int dim>
    inline void Solver<dim>::run_solver_test() {

        // setup system
        setup_system();
        //

        // assemble system
        assemble_matrix();
        assemble_rhs();
        //

        // boundary conditions
        add_bc();
        //

        // SOLVER
        solve_potential();
        //

        // OUTPUT
        save_output();
        //
    };

} // end PICsimulator namespace

#endif // SOLVER    