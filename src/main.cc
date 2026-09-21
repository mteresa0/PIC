#include "simulator.hpp"
// #include "domain.hpp"
// #include "utilities.hpp"
#include "species.hpp"
#include "config.hpp"

// #include "test_simulators.hpp"
#include "deal.II/base/tensor.h"
#include <vector>

#include <iostream>

int main(int argc, char *argv[]) {

    using namespace PICsimulator;

    std::string config_file;

    if (argc>1){
        config_file = argv[1];
    } else {
        config_file = "data/input.json";
    }

    Config c(config_file);

    if (c.dim==2) {

        constexpr int dim = 2;

        Domain::PeriodicDomain<dim> domain_data(c.lower_vertex, c.upper_vertex);

        PICsimulator::Simulator<dim> simulator(c, domain_data);

        simulator.run();
    } else {
        constexpr int dim = 3;

        Domain::PeriodicDomain<dim> domain_data(c.lower_vertex, c.upper_vertex);

        PICsimulator::Simulator<dim> simulator(c, domain_data);

        simulator.run();
    }



    // std::vector<double> dts({1e-1, 1e-2, 1e-3});
    // TestSimulators::PeriodicMotion<3> motion;

    // TestSimulators::convergence_test(motion, dts, 1.0);
   
    
}

