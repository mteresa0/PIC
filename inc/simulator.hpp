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
#include <deal.II/dofs/dof_handler.h> 
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/particle_accessor.h>
#include <deal.II/base/point.h>

#include <fstream>
#include <type_traits>

#include "utilities.hpp"
#include "domain.hpp"
#include "streaming.hpp"
#include "collisions.hpp"
#include "species.hpp"
#include "solver.hpp"

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

        /////////// test
        void solve_test(){
            initialize();
            field_solver.run_solver_test();
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

        Solver<dim> field_solver;

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