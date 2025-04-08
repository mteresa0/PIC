#include "simulator.hpp"
#include "domain.hpp"
#include "test_simulators.hpp"

// #include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

int main() {

    constexpr int dim = 2;

    Domain::PeriodicDomain<dim> domain_data;
    dealii::Tensor<1, dim> vel;
    vel[0] = 10; vel[1] = 10.9;

    TestSimulators::TestConstantVelocity<dim> sim(domain_data, 1e-2, 1, vel);

    sim.run_test();

}
