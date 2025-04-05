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

#include "domain.hpp"
#include <chrono>

static std::chrono::time_point<std::chrono::high_resolution_clock> start_chrono, starting_time_sorting;
static double DT, DT_sort;

#define tic() start_chrono = std::chrono::high_resolution_clock::now();
#define toc()                                                               \
{                                                                           \
    auto now = std::chrono::high_resolution_clock::now();                   \
    std::chrono::duration<double> diff = now - start_chrono;                \
    DT = diff.count();                                                      \
}
#define start_chrono_sort() {starting_time_sorting = std::chrono::high_resolution_clock::now();}
#define restart_DT_sort() {DT_sort = 0;}
#define add_time() {                                                        \
    auto now = std::chrono::high_resolution_clock::now();                   \
    std::chrono::duration<double> diff = now - starting_time_sorting;       \
    DT_sort += diff.count();                                                \
}


namespace PICsimulator{

    const double BOLTZMANN_K = 1.380649e-23;  // J/K
    const double TEMPERATURE = 10000;         // K
    const double ELECTR_MASS = 1.9093837e-31; // kg
    
    using namespace dealii;

    template<int dim, Domain::boundary_type btype>
    class Simulator{
        static_assert(dim==2||dim==3, "The dimension must must 2 or 3.");

        using velocity_type = Tensor<1, dim>;

    public:

        /// @brief constructor
        /// @param domain_data_ 
        Simulator(
            const Domain::DomainData<dim, btype> &domain_data_ = Domain::DomainData<dim, btype>(),
            const double &dt_ = 1e-8,
            const double &T_ = 1e-5);
        
        /// @brief run simulation
        void run();        
    
    private:

        //// INITIALIZATION VARIABLES
        bool initialized = false;
        unsigned int n_particles = 0;

        //// INIZIALIZATION FUNCTIONS

        /// @brief initialize grid and particles
        void initialize();

        /// @brief create the grid
        void make_grid();

        /// @brief insert particles (dummy, it will be replaced)
        void insert_particles_dummy();

        /// @brief set initial velocity of each particles according to maxwell-boltzmann distribution
        void set_initial_velocity();

        /// @brief called by the particle lost signal. 
        /// It handles lost particles and reinsert them in the domain according to btype 
        /// @param particle 
        /// @param cell 
        void boundary_wall(
            const typename Particles::ParticleIterator<dim>         &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell);

        //// SIMULATION FUNCTIONS

        /// @brief update the position according to Verlet algorithm
        void update_position();

        /// @brief update the velocity according to Verlet algorithm
        void update_velocity();

        //// SETTERS AND GETTERS

        /// @brief set the velocity of a particle
        /// @param particle
        /// @param velocity
        void set_velocity(typename Particles::ParticleIterator<dim> &particle, const velocity_type &velocity);

        /// @brief get the velocity of a particle
        /// @param particle
        /// @return 
        velocity_type get_velocity(const typename Particles::ParticleIterator<dim> &particle);

        //// PRINTING FUNCTIONS

        /// @brief prints at a given timestep i the properties of each particle
        /// @param i timestep index
        void print_data_to_files(const int &);

        //// DATA MEMBERS

        // dealii objects
        dealii::Triangulation<dim>              triangulation;
        const dealii::MappingQ<dim>             mapping;
        DoFHandler<dim>                         dof_handler;
        dealii::Particles::ParticleHandler<dim> particle_handler;

        // random generators for maxwell distribution of velocity
        std::random_device rd;
        std::mt19937 gen; 
        std::normal_distribution<double> normal_distr;

        // simulation parameters
        const double dt;
        const double T;
        const double particle_mass = ELECTR_MASS;

        // storing domain data
        const Domain::DomainData<dim, btype> domain_data;

    };

};

#include "simulator_def.inc"

#endif // SIMULATOR