#ifndef SIMULATOR
#define SIMULATOR

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/particle_accessor.h>
#include <deal.II/base/point.h>
#include <deal.II/particles/data_out.h>

#include <fstream>
#include <random>
#include <type_traits>

namespace PICsimulator{

    const double BOLTZMANN_K = 1.380649e-23;  // J/K
    const double TEMPERATURE = 10000;         // K
    const double ELECTR_MASS = 1.9093837e-31; // kg
    
    using namespace dealii;

    template<int dim>
    class Simulator{
        static_assert(dim==2||dim==3, "The dimension must must 2 or 3.");

        using velocity_type = Tensor<1, dim>;

    public:

        Simulator();

        void run();        
    
    private:

        void make_grid();
        void insert_particles_dummy();
        void set_initial_velocity();

        void boundary_wall(
            const typename Particles::ParticleIterator<dim>         &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell);

        // setter and getter for velocity
        void set_velocity(typename Particles::ParticleIterator<dim> &particle, const velocity_type &velocity);
        velocity_type get_velocity(const typename Particles::ParticleIterator<dim> &particle);

        // print data to files
        void print_data_to_files(const int &);

        // dealii objects
        dealii::Triangulation<dim>              triangulation;
        const dealii::MappingQ<dim>             mapping;
        dealii::Particles::ParticleHandler<dim> particle_handler;

        // random generators for maxwell distribution of velocity
        std::random_device rd;
        std::mt19937 gen; 
        std::normal_distribution<double> normal_distr;

        // simulation parameters
        double dt = 1e-8;
        double T = 1e-5;
        double particle_mass = ELECTR_MASS;

    };

};

#include "simulator_def.inc"

#endif // SIMULATOR