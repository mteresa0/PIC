#ifndef SIMULATOR
#define SIMULATOR


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/base/point.h>

#include <fstream>
#include <random>

namespace PICsimulator{
    
    using namespace dealii;

    template<int dim>
    class Simulator{
    public:
        Simulator();

        void run();
    
    private:

        void make_grid();
        void insert_particles_dummy();
        double compute_particle_kinetic_energy(const Tensor<1, dim> & vel);

        void boundary_wall(
            const typename Particles::ParticleIterator<dim>         &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell);

        dealii::Triangulation<dim>              triangulation;
        const dealii::MappingQ<dim>             mapping;
        dealii::Particles::ParticleHandler<dim> particle_handler;

        double dt = 0.1;
        double T = 10;
        double particle_mass = 1;

    };

};

template<int dim>
PICsimulator::Simulator<dim>::Simulator() :
mapping(1), 
particle_handler(triangulation, mapping, dim) 
{
        particle_handler.signals.particle_lost.connect(
      [this](const typename Particles::ParticleIterator<dim>         &particle,
             const typename Triangulation<dim>::active_cell_iterator &cell) {
        this->boundary_wall(particle, cell);
      });
};


template<int dim>
void PICsimulator::Simulator<dim>::run() {
    make_grid();
    insert_particles_dummy();

    unsigned int NT = T/dt;
    std::cout << NT << std::endl;

    std::vector<double> total_kinetic_energy(NT, 0);
    std::vector<double> sum_particles_kinetic_energy(NT, 0);
     

    for(unsigned int i = 0; i<NT; ++i){
        int j = 0;
        Tensor<1, dim> sum_vels;
        for (auto it = particle_handler.begin(); it!=particle_handler.end(); ++it)
        {
            Point<dim> position(it->get_location());
            const Tensor<1, dim> velocity(it->get_properties());
            
            // update position
            it->set_location(position + dt*velocity);

            // std::cout <<"t = " << i*dt <<" ) position         velocity           # particles = "<< particle_handler.n_global_particles()<< "\n";
            // for (auto d = 0; d<dim; ++d){
            //     std::cout << "   "<< position[d] << "       " << velocity[d] << std::endl;
            // }
            // std::cout << std::endl;
            
            // save kinetic energy per particle
            double ke = particle_mass * velocity.norm();
            sum_particles_kinetic_energy[i] += ke;
            sum_vels += velocity;

            ++j;
        }
        total_kinetic_energy[i] = particle_handler.n_global_particles() * particle_mass * sum_vels.norm();
        particle_handler.sort_particles_into_subdomains_and_cells();
    }

    // print kinetic energies
    std::cout  << "kin energy of the system\n";
    for (unsigned int t = 0; t<NT; ++t)
    {
        std::cout << total_kinetic_energy[t] << "   ";
    }
    std::cout << std::endl;

    std::cout  << "sum of the particles kin energy\n";
    for (unsigned int t = 0; t<NT; ++t)
    {
        std::cout << sum_particles_kinetic_energy[t] << "   ";
    }
    std::cout << std::endl;

    
}

template<int dim>
double PICsimulator::Simulator<dim>::compute_particle_kinetic_energy(const dealii::Tensor<1, dim> & vel)
{
    return particle_mass * vel.norm();
}

template<int dim>
void PICsimulator::Simulator<dim>::make_grid() {

    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);

    std::cout << "Grid created" << std::endl;
};

template<int dim>
void PICsimulator::Simulator<dim>::insert_particles_dummy()
{
    int N = 100;

    Point<dim> initial_pos_vector;

    for(int n = 0; n<dim; ++n){
        initial_pos_vector[n] = 0.5;
    }
    
    std::vector<Point<dim>> positions(N, initial_pos_vector);

    particle_handler.insert_particles(positions);

    // set velocities
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> velocity_distribution(-0.5, 0.5);

    {
        Point<dim> velocity;
        for(auto it = particle_handler.begin(); it!= particle_handler.end(); ++it){
            for(int d = 0; d<dim; ++d){
                velocity[d] = velocity_distribution(gen);
            }
            it->set_properties(velocity);
        }
    }

    std::cout << "particles inserted!\n" << particle_handler.n_global_particles() << "\n";
};



template <int dim>
void PICsimulator::Simulator<dim>::boundary_wall( 
    const typename Particles::ParticleIterator<dim>         &particle,
    const typename Triangulation<dim>::active_cell_iterator &cell) 
{
    Point<dim> location(particle->get_location());
    Tensor<1, dim> velocity(particle->get_properties());
    std::vector<double> new_velocity(dim);

    for (int i = 0; i<dim; ++i){
        if (location[i]<0){
            new_velocity[i] = -velocity[i];
            location[i] = -location[i];
            // std::cout << "particle has been lost (<0)\n";
        } else if (location[i]>1){
            new_velocity[i] = -velocity[i];
            location[i] = 2-location[i];
            // std::cout << "particle has been lost (>1)\n";
        } else {
            new_velocity[i] = velocity[i];
        }
    }

    // add new particle
    Particles::Particle new_particle(location, particle->get_reference_location(), particle->get_id());

    particle_handler.insert_particle(location,
    particle->get_reference_location(),
    particle->get_id(),
    cell,
    new_velocity);    

}


#endif // SIMULATOR