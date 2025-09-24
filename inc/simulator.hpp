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
#include <deal.II/particles/data_out.h>

#include <fstream>
#include <random>
#include <type_traits>

#include "utilities.hpp"
#include "domain.hpp"
#include "streaming.hpp"
#include "collisions.hpp"

namespace PICsimulator{

    // using namespace Timers;    
    using namespace dealii;

    /// @brief default domain for the simulation. Lower and upper vertices are
    /// of the unit square (2D) or unit cube (3D).
    /// @tparam dim spatial dimension
    template<int dim>
    Domain::ReflectiveDomain<dim> default_domain;

    /// @brief get the default domain for the simulation
    /// @tparam dim spatial dimension
    /// @return ReflectiveDomain<dim> with UnitSquare lower and upper vertices
    template<int dim>
    const Domain::DomainData<dim>& get_default_domain() {
        return default_domain<dim>;  // Return reference to the instantiated object
    }

    /// @brief Simulator class. 
    /// @tparam dim spatial dimension
    /// @details This class is the main class of the simulation. 
    /// It handles the initialization of the grid and of the particle handler,
    /// the simulation loop, the integration of the particle state 
    /// and the output of the data.
    template<int dim>
    class Simulator{

        // only 2D and 3D are allowed
        static_assert(dim==2||dim==3, "The dimension must must 2 or 3.");

    public:
        using velocity_type = Tensor<1, dim>;
        using location_type = Point<dim>;

        /// @brief Constructor of the Simulator class
        /// @param dt_ time step
        /// @param T_ total simulation time
        /// @param n_particles_ number of particles
        /// @param domain_data_ domain data object
        Simulator(
            const double &dt_ = 1e-8,
            const double &T_ = 1e-5,
            const unsigned int &n_particles_ = 100,
            const Domain::DomainData<dim> & domain_data_ = get_default_domain<dim>());
        
        /// @brief run the simulation
        void run();   

        /// @brief Destructor of the Simulator class
        ~Simulator() = default;

        /// no copy
        Simulator(const Simulator &) = delete;
        /// no move
        Simulator(Simulator &&) = delete;
        /// no assignment
        Simulator & operator=(const Simulator &) = delete;
        /// no move assignment
        Simulator & operator=(Simulator &&) = delete;

        /// @brief get the particle handler pointer
        /// @return particle handler pointer
        typename Particles::ParticleHandler<dim> & get_particle_handler() {
            return particle_handler;
        };

        /// @brief get dof handler pointer
        /// @return dof handler pointer
        const DoFHandler<dim> & get_dof_handler() const {
            return dof_handler;
        };

        friend class Streaming<dim, StreamingAlgorithm::VERLET>;

        double get_cell_volume() const {
            return cell_volume;
        };

        double get_number_density() const {
            return number_density;
        };

        double get_time_step() const {
            return dt;
        };

        double get_cell_diameter() const {
            double d_cell = dof_handler.begin()->diameter();
            return d_cell;
        };

        const Species::ChemicalSpecies & get_species() const {
            return species;
        };

    private:
        Species::N2 species;
       
        /// @brief It handles lost particles and reinsert them in the domain.
        /// This function is called when a particle leaves the domain and it 
        /// is passed to the particle handler in the constructor definition.
        /// @param particle particle iterator object. This is the particle that
        /// left the domain.
        /// @param cell cell iterator object.
        void boundary_wall(
            const typename Particles::ParticleIterator<dim>         &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell);

        //// INITIALIZATION VARIABLES

        /// @brief flag to check if the simulator is initialized
        bool initialized = false;

        // simulation parameters
        /// @brief number of particles in the simulation
        const unsigned int n_particles;
        /// @brief time step
        const double dt;
        /// @brief total simulation time
        const double T;

        /// @brief cell volume
        double cell_volume;
        double number_density;

        //// INIZIALIZATION FUNCTIONS

        /// @brief initialize grid and particles
        void initialize();
        
        /// @brief create the grid according to the domain data
        void make_grid();

        //// SIMULATION FUNCTIONS

        /// @brief evaluate acceleration (TODO)
        location_type evaluate_acceleration(){return location_type();};

        //// PRINTING FUNCTIONS

        /// @brief prints at a given timestep i the properties of each particle
        /// @param i timestep index
        void print_data_to_files(const int &);

        //// DATA MEMBERS

        // dealii objects
        /// @brief triangulation of the domain
        dealii::Triangulation<dim>              triangulation;
        /// @brief finite element mapping
        const dealii::MappingQ<dim>             mapping;
        /// @brief finite element object
        DoFHandler<dim>                         dof_handler;
        /// @brief particle handler object. It handles the particles and their properties.
        dealii::Particles::ParticleHandler<dim> particle_handler;
        
        /// @brief domain data object. It stores the domain information and 
        /// how the particles behave when they leave the domain.
        const Domain::DomainData<dim> & domain_data;

        /// @brief streaming algorithm object. It handles the streaming of the
        /// particles and computes new force field.
        Streaming<dim, StreamingAlgorithm::VERLET> streaming_algorithm;

        /// @brief collision object. It handles the collisions of the particles.
        Collisions<dim, HardSphere<dim>> collision_handler;

        /// @brief insert particles (dummy, it will be replaced)
        void insert_particles_dummy();

        /// @brief set initial velocity of each particles according to 
        /// maxwell-boltzmann distribution
        void set_initial_velocity();

        // random generators for maxwell distribution of velocity
        /// @brief random device
        std::random_device rd;
        /// @brief random number generator
        std::mt19937 gen; 
        /// @brief normal distribution for velocity
        std::normal_distribution<double> normal_distr;

    };

};

#include "simulator.tpp"

#endif // SIMULATOR