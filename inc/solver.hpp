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
#include <deal.II/lac/affine_constraints.h>

#include "species.hpp"
#include "simulator.hpp"
#include "domain.hpp"
#include "config.hpp"

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
        /// @param domain_ domain data structure
        explicit Solver(DoFHandler<dim>& dof_h_,
                        const MappingQ<dim>& mapping_,
                        vec_ph& phs_,
                        species_type& ms_,
                        const Domain::DomainData<dim>& domain_)
            :
            dof_handler(dof_h_),
            mapping(mapping_),
            phs(phs_),
            ms(ms_),
            domain(domain_),
            fe(1),
            quadrature_formula(fe.degree+1),
            fe_values(fe, quadrature_formula,
                    update_values | update_gradients | update_JxW_values),
            is_charged(phs.size(), 0)
            {};

        /// @brief initializes solver
        void initialize();
        
        /// @brief runs a standalone test of the solver 
        void initialize_accelerations();

        /// @brief solves potential field and updates accelerations
        void run_solver(const unsigned int);

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

        /// @brief domain data structure
        const Domain::DomainData<dim>& domain;

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

        /// @brief constraints for periodic domain
        dealii::AffineConstraints<double> constraints;

        /// @brief
        std::vector<char> is_charged;

        //////// mthods

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
        void save_output(const unsigned int i=0) const;

        /// @brief computes and stores the acceleration for each particle
        void compute_accelerations();


    }; // end Solver class

    /* DEFINITIONS */

    template <int dim>
    inline void Solver<dim>::initialize() {
        // setup system
        setup_system();
        
        // assemble system
        assemble_matrix();
        assemble_rhs();
        
        // boundary conditions
        add_bc();
    }

    template <int dim>
    inline void Solver<dim>::initialize_accelerations() {
        // SOLVER
        solve_potential();
        
        // OUTPUT
        save_output();
        
        // COMPUTE ACCELERATIONS
        compute_accelerations();
    }

    template <int dim>
    inline void Solver<dim>::run_solver(const unsigned int index_timestep) {
        assemble_rhs();
        solve_potential();
        save_output(index_timestep);
        compute_accelerations();
    };

    template <int dim>
    inline void Solver<dim>::setup_system() {

        dof_handler.distribute_dofs(fe);

        // apply constraints
        constraints.clear();
        domain.apply_constraints(dof_handler, constraints);
        constraints.close();

        // setup matrices 
        DynamicSparsityPattern dsp(dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
        sparsity_pattern.copy_from(dsp);

        system_matrix.reinit(sparsity_pattern);
        solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());

        // initialise is_charged
        for(unsigned int i = 0; i<phs.size();++i){
            if(abs(ms[i].get_charge())>1e-40){
                is_charged[i] = 1;
            }
        }
    } // end setup system

    template <int dim>
    inline void Solver<dim>::assemble_matrix()  {
        // LOG("ASSEMBLE_MATRIX");
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
        constraints.condense(system_matrix);
    } // end assemble_matrix

    template <int dim>
    inline void Solver<dim>::assemble_rhs() {
        // LOG("ASSEMBLE_RHS");

        // reinitialize the rhs system
        system_rhs.reinit(dof_handler.n_dofs());
        system_rhs = 0;

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        Vector<double> cell_rhs(dofs_per_cell);        
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        // calculate total charge in the domain
        double total_charge = 0.0;
        for (unsigned int idx_species = 0; idx_species < ms.n_species(); ++idx_species) {
            if (is_charged[idx_species]) {
                total_charge += ms[idx_species].get_charge() * phs[idx_species].n_global_particles();
            }
        }

        // compute background density offset: rho_bg = total_charge / domain_Volume
        const double domain_volume = domain.volume(); 
        const double bg_density = total_charge / domain_volume;
             
        // iteration on cells
        for (const auto &cell : dof_handler.active_cell_iterators())
        {
            // reinitialize fe values for the given cell
            fe_values.reinit(cell);
            cell_rhs    = 0;

            // * compute rhs *
            for(unsigned int idx_species = 0; idx_species<ms.n_species(); ++idx_species){

                unsigned int n_part = phs[idx_species].n_particles_in_cell(cell);

                // skip if the species has null charge or there are not any particles in the cell
                if(is_charged[idx_species] && n_part>0){

                    double f = ms[idx_species].get_charge()/EPS0;

                    auto pic = phs[idx_species].particles_in_cell(cell);
                    for(auto& it_p : pic) {
                        // Point<dim> particle_loc(it_p.get_location());
                        // Point<dim> particle_ref_loc = 
                        //     mapping.transform_real_to_unit_cell(cell, particle_loc);

                        Point<dim> particle_ref_loc = 
                            mapping.transform_real_to_unit_cell(cell, it_p.get_location());

                        for(unsigned int i : fe_values.dof_indices()){
                            // since the charge is punctual the integral over 
                            // the cell is computed as charge/eps0*shape_fun(reference_location)
                            cell_rhs(i) += f * fe.shape_value(i,particle_ref_loc);
                        }
                    }
                }
            }

            // continuous background subtraction for charge neutrality
            if (std::abs(bg_density) > 1e-40) {
                const double bg_factor = bg_density / EPS0;
                for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
                    for (unsigned int i : fe_values.dof_indices()) {
                        cell_rhs(i) -= bg_factor * fe_values.shape_value(i, q_index) * fe_values.JxW(q_index);
                    }
                }
            }
        
            cell->get_dof_indices(local_dof_indices);
        
            for (const unsigned int i : fe_values.dof_indices())
                system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
        constraints.condense(system_rhs);
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
    } // end solve_potential

    template <int dim>
    inline void Solver<dim>::save_output(const unsigned int index_time) const {

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();
        
        const std::string filename = "outputs/phi_"+
            Utilities::int_to_string(index_time, 5)+".vtk";
        std::ofstream     output(filename);
        data_out.write_vtk(output);
        // std::cout << "Output written to " << filename << std::endl;
    } // end save output

    template <int dim>
    inline void Solver<dim>::compute_accelerations()
    {
        Vector<double> solution_values(fe.n_dofs_per_cell());
        FEPointEvaluation<1, dim> evaluator(mapping, fe, update_gradients);

        for (const auto &cell : dof_handler.active_cell_iterators()){
            cell->get_dof_values(solution, solution_values);
            for (unsigned int s = 0; s<ms.n_species(); ++s){
                if (is_charged[s] && phs[s].n_particles_in_cell(cell) > 0){
                    double f = - ms[s].get_charge()/ms[s].get_mass();
                    auto particles_in_cell(phs[s].particles_in_cell(cell));
                
                    std::vector<Point<dim>> particle_positions;
                    for (const auto &particle : particles_in_cell)
                        particle_positions.push_back(particle.get_reference_location());

                    evaluator.reinit(cell, particle_positions);
                    evaluator.evaluate(make_array_view(solution_values), EvaluationFlags::gradients);

                    unsigned int index_particle = 0;
                    for(auto particle = particles_in_cell.begin();
                        particle!=particles_in_cell.end(); 
                        ++particle, ++index_particle)
                    {
                        const Tensor<1,dim>& E = evaluator.get_gradient(index_particle);
                        const Tensor<1,dim> acc =  f*E;

                        set_acceleration(particle, acc);
                        // LOGVEC("acceleration", acc, dim);
                    }
                }
            }
        }
    }// compute accelerations

} // end PICsimulator namespace

#endif // SOLVER    