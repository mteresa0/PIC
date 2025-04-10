#include "simulator.hpp"
#include "domain.hpp"
#include "test_simulators.hpp"

// #include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

int main() {

    constexpr int dim = 2;
    Domain::PeriodicDomain<dim> domain_data;

    /// constant velocity
    std::cout << "\n\n*************** Constant Velocity Test ***************\n";
    dealii::Tensor<1, dim> vel({1, 0});
    
    TestSimulators::TestConstantVelocity<dim> test_const_vel (
        domain_data, 1e-2, 1, vel
    );

    test_const_vel.run_test();

    /// constant acceleration
    std::cout << "\n\n*************** Constant Acceleration Test ***************\n";
    dealii::Point<dim> acc(0.1, 0.1);
    dealii::Tensor<1, dim> initial_vel({-1, 0});

    TestSimulators::TestConstantAcceleration<dim> test_sim(
        domain_data, 1e-2, 1, acc, initial_vel
    );

    test_sim.run_test();

}
