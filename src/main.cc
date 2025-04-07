#include "simulator.hpp"
#include "domain.hpp"

// #include "deal.II/base/point.h"
// #include "deal.II/base/tensor.h"

int main() {

    Domain::PeriodicDomain<2> domain_data;

    PICsimulator::Simulator<2> sim(domain_data);

    sim.run();

}
