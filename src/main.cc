#include "simulator.hpp"
#include "domain.hpp"
#include "test_simulators.hpp"

// #include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

int main() {

    constexpr int dim = 2;
    std::vector<double> dts = {0.1, 0.05, 0.025, 0.0125, 0.00625};
    double total_time = 5;

    PICsimulator::TestSimulators::PeriodicMotion<dim> motion;


    PICsimulator::TestSimulators::convergence_test<dim>(
        motion, dts, total_time);


    
}

