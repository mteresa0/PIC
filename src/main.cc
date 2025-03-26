#include "simulator.hpp"
#include "domain.hpp"

#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

int main() {

    dealii::Point<2> upper_v(1, 1);
    dealii::Point<2> lower_v(0, 0);
    dealii::Tensor<1, 2> vel;
    Domain::DomainData<2, Domain::REFLECTIVE> domain_data_r(lower_v, upper_v);
    domain_data_r.out_of_boundaries(upper_v, vel);

    Domain::DomainData<2, Domain::PERIODIC> domain_data_p(lower_v, upper_v);
    domain_data_p.out_of_boundaries(lower_v, vel);
            
    // PICsimulator::Simulator<2> sim;

    // sim.run();

}
