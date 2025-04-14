#include "simulator.hpp"
#include "domain.hpp"
#include "test_simulators.hpp"

// #include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> ud;
double RND(const double & mult = 1.)
{
    double a = ud(gen); // from 0 to 1
    a = a-0.5;          // from -0.5 to 0.5
    a *= 2*mult;        // from -mutl to mult
    return a;
};

int main() {

    constexpr int dim = 2;
    Domain::PeriodicDomain<dim> domain_data;

    double random(const double & mult);

    /// constant velocity
    std::cout << "\n\n*************** Constant Velocity Test ***************\n";
    dealii::Tensor<1, dim> vel({RND(), RND()});
    
    TestSimulators::TestConstantVelocity<dim> test_const_vel (
        domain_data, 1e-2, 1, vel
    );

    test_const_vel.run_test();

    /// constant acceleration
    std::cout << "\n\n*************** Constant Acceleration Test ***************\n";
    dealii::Point<dim> acc(RND(1.0), RND(0.01));
    dealii::Tensor<1, dim> initial_vel({RND(0), RND(0)});

    TestSimulators::TestConstantAcceleration<dim> test_const_acc(
        domain_data, 1e-2, 1, acc, initial_vel
    );

    test_const_acc.run_test();

    /// linear acceleration 
    std::cout << "\n\n*************** Linear Acceleration Test ***************\n";
    std::cout << "*************** (constant acceleration) ***************\n";
    dealii::Point<dim> dacc0(0.1, 0.1);

    TestSimulators::TestLinearAcceleration<dim> test_linear_acc0(
        domain_data, 0.5, 10, acc, dacc0, initial_vel);

    test_linear_acc0.run_test();

    std::cout << "\n*************** convergence test ***************\n";

    dealii::Point<dim> dacc1(RND(0.1), RND(0.1));

    // std::vector<double> dts = {1e-2, 5e-3, 25e-4};
    std::vector<double> dts = {5e-1, 1e-1, 1e-2, 5e-3};
    std::vector<double> err_vels(dts.size());
    std::vector<double> err_locs(dts.size());

    double total_time = 5;

    bool overwrite_file = true;
    for (unsigned int i = 0; i<dts.size(); ++i) {
        TestSimulators::TestLinearAcceleration<dim> test_linear_acc1(
            domain_data, dts[i], total_time, acc, dacc1, initial_vel);
        test_linear_acc1.run_test();
        test_linear_acc1.print_errors_to_file(
            "outputs/linear_acceleration_test.txt", overwrite_file);
        overwrite_file = false;

        err_locs[i] = test_linear_acc1.get_loc_error();
        err_vels[i] = test_linear_acc1.get_vel_error();
        
    };

    std::cout << "---- convergence rates ----\n";
    for(unsigned int i = 1; i<dts.size(); ++i){
        std::cout << "p loc = " << std::log(err_locs[i-1] / err_locs[i]) / std::log(dts[i-1] / dts[i]) << std::endl;
        std::cout << "p vel = " << std::log(err_vels[i-1] / err_vels[i]) / std::log(dts[i-1] / dts[i]) << std::endl;
    }

}

