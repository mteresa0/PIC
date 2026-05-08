#include "simulator.hpp"
// #include "domain.hpp"
// #include "utilities.hpp"
#include "species.hpp"
#include "config.hpp"

// #include "test_simulators.hpp"
#include "deal.II/base/tensor.h"

int main() {

    using namespace PICsimulator;

    constexpr int dim = 2;
    Config<dim> c;

    Domain::ReflectiveDomain<dim> domain_data(c.lower_vertex, c.upper_vertex);

    PICsimulator::Simulator<dim> simulator(c, domain_data);

    simulator.run();
    
    
}

