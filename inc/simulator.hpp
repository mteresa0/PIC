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

        void boundary_wall(
            const typename Particles::ParticleIterator<dim>         &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell);

        dealii::Triangulation<dim>              triangulation;
        const dealii::MappingQ<dim>             mapping;
        dealii::Particles::ParticleHandler<dim> particle_handler;

        double dt = 0.1;
        double T = 2;

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

    for(unsigned int i = 0; i<NT; ++i){
        for (auto it = particle_handler.begin(); it!=particle_handler.end(); ++it)
        {
            Point<dim> position(it->get_location());
            const Tensor<1, dim> velocity(it->get_properties());
            
            // update position
            it->set_location(position + dt*velocity);

            std::cout <<"t = " << i*dt <<" ) position         velocity\n";
            for (auto d = 0; d<dim; ++d){
                std::cout << "   "<< position[d] << "       " << velocity[d] << std::endl;
            }
            std::cout << std::endl;
        }
        particle_handler.sort_particles_into_subdomains_and_cells();
    }
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
    int N = 1;

    Point<dim> initial_pos_vector({0.5, 0.5});
    
    if(dim!=2)
        std::cerr << "non si può fare ora";
    
    std::vector<Point<dim>> positions(N, initial_pos_vector);

    particle_handler.insert_particles(positions);

    // set velocities
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<> velocity_distribution(-2., 2.);

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

    for (int i = 0; i<dim; ++i){
        if (location[i]<0){
            velocity[i] = -velocity[i];
            location[i] = -location[i];
            std::cout << "particle has been lost (<0)\n";
        } else if (location[i]>1){
            velocity[i] = -velocity[i];
            location[i] = 1-location[i];
            std::cout << "particle has been lost (>1)\n";
        }
    }
    // add new particle
    

}


#endif // SIMULATOR