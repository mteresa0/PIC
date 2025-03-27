#include "simulator.hpp"
#include "domain.hpp"

#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

int main() {

    // dealii::Point<2> upper_v(1.5, 1);
    // dealii::Point<2> lower_v(0, -0.2);

    // Domain::DomainData<2, Domain::PERIODIC> domain_data(lower_v, upper_v);

    PICsimulator::Simulator<2, Domain::PERIODIC> sim;

    sim.run();

}
