#include "simulator.hpp"
#include "domain.hpp"
#include "utilities.hpp"

#include "test_simulators.hpp"
#include "deal.II/base/tensor.h"

int main() {

    using namespace PICsimulator;

    constexpr int dim = 2;
    // std::vector<double> dts = {0.1, 0.05, 0.025, 0.0125, 0.00625};
    // double total_time = 5;

    // PICsimulator::TestSimulators::PeriodicMotion<dim> motion;


    // PICsimulator::TestSimulators::convergence_test<dim>(
    //     motion, dts, total_time);

    Point<dim> lower_vertex(0, 0);
    Point<dim> upper_vertex(5e-7, 5e-7);
    Domain::PeriodicDomain<dim> domain_data(lower_vertex, upper_vertex);

    std::vector<Species::ChemicalSpecies> species({Species::N2(), Species::O2()});

    Species::MultiSpecies ms(species, {100, 100});

    double dt = species[0].get_free_path() / species[0].get_mean_velocity();

    PICsimulator::Simulator<dim> simulator(dt, 100*dt, ms, domain_data);

    simulator.run();
    
    
}

