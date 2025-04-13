#include "simulator.hpp"
#include "domain.hpp"
#include "test_simulators.hpp"

// #include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

int main() {

    constexpr int dim = 2;
    Domain::PeriodicDomain<dim> domain_data;

    // /// constant velocity
    // std::cout << "\n\n*************** Constant Velocity Test ***************\n";
    // dealii::Tensor<1, dim> vel({1, 0});
    
    // TestSimulators::TestConstantVelocity<dim> test_const_vel (
    //     domain_data, 1e-2, 1, vel
    // );

    // test_const_vel.run_test();

    /// constant acceleration
    std::cout << "\n\n*************** Constant Acceleration Test ***************\n";
    dealii::Point<dim> acc(0.1, 0.1);
    dealii::Tensor<1, dim> initial_vel({-1, 0});

    TestSimulators::TestConstantAcceleration<dim> test_const_acc(
        domain_data, 1e-2, 1, acc, initial_vel
    );

    // test_const_acc.run_test();

    /// linear acceleration 
    std::cout << "\n\n*************** Linear Acceleration Test ***************\n";
    // dealii::Point<dim> acc(0.1, 0.1);
    dealii::Point<dim> dacc0(-0.0, -0.0);
    // dealii::Tensor<1, dim> initial_vel({-1, 0});

    TestSimulators::TestLinearAcceleration<dim> test_linear_acc0(
        domain_data, 1e-2, 1e-1, acc, dacc0, initial_vel);

    // test_linear_acc0.run_test();

    dealii::Point<dim> dacc1(-0.01, -0.01);

    std::vector<double> dts = {1e-2, 1e-3, 1e-4, 1e-5};

    bool overwrtie_file = true;
    for (auto dt : dts) {
        TestSimulators::TestLinearAcceleration<dim> test_linear_acc1(
            domain_data, dt, 1e-1, acc, dacc1, initial_vel);
        test_linear_acc1.run_test();
        test_linear_acc1.print_errors_to_file(
            "linear_acceleration_test.txt", overwrtie_file);
        overwrtie_file = false;
    }

}
