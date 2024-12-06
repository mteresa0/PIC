#ifndef SIMULATOR
#define SIMULATOR


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/base/point.h>

#include <fstream>

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

        dealii::Triangulation<dim>              triangulation;
        const dealii::MappingQ<dim>             mapping;
        dealii::Particles::ParticleHandler<dim> particle_handler;

    };

};

template<int dim>
PICsimulator::Simulator<dim>::Simulator() :
mapping(1), 
particle_handler(triangulation, mapping, dim) {};


template<int dim>
void PICsimulator::Simulator<dim>::run() {
    make_grid();
    insert_particles_dummy();
}

template<int dim>
void PICsimulator::Simulator<dim>::make_grid() {


    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);

    // std::ofstream out("grid.svg");
    // dealii::GridOut     gridout;
    // gridout.write_vtk(triangulation, out);
    // std::cout << "grid written in vtk" << std::endl;

    std::cout << "Grid created" << std::endl;
};

template<int dim>
void PICsimulator::Simulator<dim>::insert_particles_dummy()
{
    std::vector<Point<dim>> positions(10, {0.9,0.9});


    particle_handler.insert_particles(positions);

    std::cout << "particle inserted!\n" << particle_handler.n_global_particles() << "\n";
}


#endif // SIMULATOR