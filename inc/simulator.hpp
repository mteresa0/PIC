/**
 * @file simulator.hpp
 * @author Maria Teresa Cazzola
 * @brief declarations of class simulator
 * @date 2025-04-11
 *
 * @copyright Copyright (c) 2025
 *
 */
#ifndef SIMULATOR
#define SIMULATOR

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/mapping_q.h> 
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h> 
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/particle_accessor.h>
#include <deal.II/base/point.h>
#include <deal.II/particles/data_out.h>
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

#include <fstream>
#include <type_traits>

#include "utilities.hpp"
#include "domain.hpp"
#include "streaming.hpp"
#include "collisions.hpp"
#include "species.hpp"

namespace PICsimulator
{

    // using namespace Timers;
    using namespace dealii;

    /// @brief Simulator class.
    /// @tparam dim spatial dimension
    /// @details This class is the main class of the simulation.
    /// It handles the initialization of the grid and of the particle handler,
    /// the simulation loop, the integration of the particle state
    /// and the output of the data.
    template <int dim>
    class Simulator
    {

        // only 2D and 3D are allowed
        static_assert(dim == 2 || dim == 3, "The dimension must be 2 or 3.");

    public:
        using velocity_type = Tensor<1, dim>;
        using location_type = Point<dim>;
        using species_type = Species::MultiSpecies;
        using ph_type = dealii::Particles::ParticleHandler<dim>;

        /// @brief Constructor of the Simulator class
        /// @param dt_ time step
        /// @param T_ total simulation time
        /// @param species_ species
        /// @param domain_data_ domain data object
        Simulator(
            const double &dt_ = 1e-8,
            const double &T_ = 1e-5,
            const species_type &species_ = species_type(),
            const Domain::DomainData<dim> &domain_data_ =
                Domain::ReflectiveDomain<dim>());

        /// @brief run the simulation
        /// This is the method in which the simulatotion loop is defined.
        void run();

        ///////////
        void solve_test(){
            initialize();

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
        ///////////

        /// @brief Destructor of the Simulator class
        ~Simulator() = default;

        /// no copy
        Simulator(const Simulator &) = delete;
        /// no move
        Simulator(Simulator &&) = delete;
        /// no assignment
        Simulator &operator=(const Simulator &) = delete;
        /// no move assignment
        Simulator &operator=(Simulator &&) = delete;

        friend class Streaming<dim>;
        friend class Collisions<dim, HardSphere<dim>>;

        /// @brief get the simulated spicies
        const species_type &get_species() const
        {
            return species;
        };

    private:
        /// @brief It handles lost particles and reinsert them in the domain.
        /// This function is called when a particle leaves the domain and it
        /// is passed to the particle handler in the constructor definition.
        /// @param particle particle iterator object. This is the particle that
        /// left the domain.
        /// @param cell cell iterator object.
        void boundary_wall(
            const typename Particles::ParticleIterator<dim> &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell,
            const unsigned int &index_species);

        //// INITIALIZATION PARAMETERS

        /// @brief flag to check if the simulator is initialized
        bool initialized = false;

        // simulation parameters

        // /// @brief number of particles in the simulation
        // const unsigned int n_particles;

        /// @brief time step
        const double dt;

        /// @brief total simulation time
        const double T;

        /// species
        species_type species;

        const unsigned int n_species;

        /// @brief cell volume
        double cell_volume;

        // /// @brief number density (# particles / tot. volume)
        // double number_density;

        //// INIZIALIZATION FUNCTIONS

        /// @brief initialize grid and particles
        void initialize();

        /// @brief create the grid according to the domain data
        void make_grid();

        //// SIMULATION FUNCTIONS

        /// @brief evaluate acceleration
        // TODO: implement this function
        location_type evaluate_acceleration() { return location_type(); };

        //// PRINTING FUNCTIONS

        /// @brief prints at a given timestep i the properties of each particle
        /// @param i timestep index
        void print_data_to_files(const int &);

        //// DATA MEMBERS

        // dealii objects
        /// @brief triangulation of the domain
        dealii::Triangulation<dim> triangulation;

        /// @brief finite element mapping
        const dealii::MappingQ<dim> mapping;

        /// @brief finite element object
        DoFHandler<dim> dof_handler;

        //////
        const dealii::FE_Q<dim> fe;
        dealii::SparsityPattern sparsity_pattern;
        dealii::SparseMatrix<double> system_matrix;
        dealii::Vector<double> solution;
        dealii::Vector<double> system_rhs;
        //////

        /// @brief particle handler object. It handles the particles and their properties.
        std::vector<ph_type> particle_handlers;

        /// @brief domain data object. It stores the domain information and
        /// how the particles behave when they leave the domain.
        const Domain::DomainData<dim> &domain_data;

        /// @brief streaming algorithm object. It handles the streaming of the
        /// particles and computes new force field.
        Streaming<dim> streaming_algorithm;

        /// @brief collision object. It handles the collisions of the particles.
        Collisions<dim, HardSphere<dim>> collision_handler;
        // TODO: generalize for different kernels

#ifdef EXTRA_OUTPUTS
        CSVoutput out_kinetic_energy{"outputs/total_kinetic_energy.csv",
                                     {"time", "total_kinetic_energy", "sum_particles_kinetic_energy"}};
#endif

        /// @brief insert particles (dummy, it will be replaced)
        void insert_particles_dummy();

        /// @brief set initial velocity of each particles according to
        /// maxwell-boltzmann distribution
        void set_initial_velocity();
    };

};

#include "simulator.tpp"

#endif // SIMULATOR