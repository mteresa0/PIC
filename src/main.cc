#include "simulator.hpp"
#include "domain.hpp"

#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

int main() {

    dealii::Point<2> pos(1.5, 1);
    dealii::Tensor<1, 2> vel ( {0., -0.2});

    Domain::PeriodicDomain<2> domain_data;

    PICsimulator::Simulator<2> sim(domain_data);

    sim.run();

}
