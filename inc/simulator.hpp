#ifndef SIMULATOR
#define SIMULATOR


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>

namespace PICsimulator{
    
    using namespace dealii;

    template<int dim>
    class Simulator{
    public:
        Simulator() = default;

        void make_grid();
    
    private:

    };

};

template<int dim>
void PICsimulator::Simulator<dim>::make_grid() {

    dealii::Triangulation<dim> triangulation;

    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);

    std::ofstream out("grid.svg");
    dealii::GridOut     gridout;
    gridout.write_vtk(triangulation, out);
    std::cout << "grid written in vtk" << std::endl;
};


#endif // SIMULATOR