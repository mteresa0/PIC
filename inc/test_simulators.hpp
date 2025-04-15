/**
 * @file test_simulators.hpp
 * @author Maria Teresa Cazzola
 * @brief declaration and definition of tests cases for the PIC simulator
 * @date 2025-04-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef TEST_SIMULATORS_HPP
#define TEST_SIMULATORS_HPP

#include "simulator.hpp"
#include "domain.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>

namespace PICsimulator::TestSimulators
{
    using namespace dealii;

    template<int dim>
    using true_val_fun = Point<dim>(*)(const double &);

    namespace ConstantVelocity2D {

        Point<2> get_loc(const double &t)
        {
            return Point<2>(t+0.1, 0.1);
        }

        /// @brief constant velocity function
        /// @return velocity vector
        Point<2> get_vel(const double &)
        {
            return Point<2>(1, 0);
        }

        /// @brief constant acceleration function
        /// @return acceleration vector
        Point<2> get_acc(const double &)
        {
            return Point<2>(0, 0);
        }

    }

    namespace PeriodicMotion2D {
        /// @brief periodic motion function
        /// @param t time
        /// @return location vector
        Point<2> get_loc(const double &t)
        {
            return Point<2>(std::cos(t), std::sin(t));
        }

        /// @brief periodic velocity function
        /// @param t time
        /// @return velocity vector
        Point<2> get_vel(const double &t)
        {
            return Point<2>(-std::sin(t), std::cos(t));
        }

        /// @brief periodic acceleration function
        /// @param t time
        /// @return acceleration vector
        Point<2> get_acc(const double & t)
        {
            return Point<2>(-std::cos(t), -std::sin(t));
        }
    }

    template<int dim>
    void convergence_test(  const true_val_fun<dim> &true_loc,
                            const true_val_fun<dim> &true_vel,
                            const true_val_fun<dim> &true_acc,
                            const std::vector<double> &dts,
                            const double &total_time);

    /// @brief Test simulator base class. 
    /// @brief Test different scenario described in the derived classes.
    /// @tparam dim dimension of the domain
    template<int dim>
    class TestSimulator : public PICsimulator::Simulator<dim>
    {
    public:
        using typename PICsimulator::Simulator<dim>::location_type;
        using typename PICsimulator::Simulator<dim>::velocity_type;

        using fun = true_val_fun<dim>;

        /// @brief Constructor
        /// @param dt_ timestep time
        /// @param T_ total simulation time
        /// @param true_loc_ function to get the true location
        /// @param true_vel_ function to get the true velocity
        /// @param true_acc_ function to get the acceleration
        TestSimulator(  const double &dt_,
                        const double &T_, 
                        const fun &true_loc_=ConstantVelocity2D::get_loc, 
                        const fun &true_vel_=ConstantVelocity2D::get_vel, 
                        const fun &true_acc_=ConstantVelocity2D::get_acc):
            PICsimulator::Simulator<dim>(dt_, T_),
            get_true_location(true_loc_),
            get_true_velocity(true_vel_), 
            get_acceleration(true_acc_)
            {};

        /// @brief Destructor
        ~TestSimulator() = default;
        
        /// @brief run test
        void run_test();

        /// @brief print errors to file
        /// @param filename name of the file
        /// @param overwrite file
        void print_errors_to_file(
            const std::string &filename, 
            const bool &overwrite = false)
        {
            std::ofstream file;
            if (overwrite) {
                file.open(filename);
                file << "location_error velocity_error dt\n";
            } 
            else
                file.open(filename, std::ios::app);

            file << this->location_error << " " << this->velocity_error << " " 
                << this->dt << "\n";
            file.close();
        };

        double get_loc_error() const {
            return location_error;
        };

        double get_vel_error() const {
            return velocity_error;
        };

    private:
        /// @brief initialize grid and particle handler
        void initialize() override;

        /// @brief initialize particle
        void initialize_particle();

        /// @brief initialize grid
        void initialize_grid(){ this->make_grid(); };

        /// @brief updates position and velocity errors and 
        /// updates true location and position
        /// @param particle particle iterator
        void compute_errors(
            typename Particles::ParticleIterator<dim> &particle);

        /// @brief compute L2 norm 
        /// @param v velocity_type vector (dealii::Tensor<1, dim>)
        /// @return L2 norm of the input Tensor<1, dim> v
        double norm(const velocity_type & v){
            double r = 0;
            for (int d = 0; d<dim; ++d){
                r += v[d]*v[d];
            }
            return std::sqrt(r);
        };

        /// @brief compute L2 norm 
        /// @param l location_type vector (dealii::Point<dim>)
        /// @return L2 norm of the input Point<dim> l
        double norm(const location_type & l){
            double r = 0;
            for (int d = 0; d<dim; ++d){
                r += l[d]*l[d];
            }
            return std::sqrt(r);
        };

        double location_error = 0; /// location error
        double velocity_error = 0; /// velocity error

        double current_time = 0.;
        unsigned int count = 0;

        fun get_true_location;
        fun get_true_velocity;
        fun get_acceleration;

        location_type evaluate_acceleration() override{
            if(count%2!=0){
                current_time += this->dt;
            }
            ++count;
            return get_acceleration(current_time);
        }

    }; // end TestSimulator class

    
    /****************************
            DEFINITIONS   
    ****************************/

    template <int dim>
    void convergence_test(
        const true_val_fun<dim> &true_loc, 
        const true_val_fun<dim> &true_vel, 
        const true_val_fun<dim> &true_acc, 
        const std::vector<double> &dts, 
        const double &total_time)
    {
        bool overwrite = true;

        std::vector<double> err_loc(dts.size());
        std::vector<double> err_vel(dts.size());

        for(unsigned int i = 0; i<dts.size(); ++i)
        {
            TestSimulator<dim> test_simulator(
                dts[i], total_time, true_loc, true_vel, true_acc);
            
            test_simulator.run_test();
            test_simulator.print_errors_to_file("outputs/errors.csv", overwrite);
            overwrite = false;

            err_loc[i] = test_simulator.get_loc_error();
            err_vel[i] = test_simulator.get_vel_error();

        }

        // compute convergence rate
        std::vector<double> loc_rate(dts.size()-1);
        std::vector<double> vel_rate(dts.size()-1);
        for(unsigned int i = 0; i<dts.size()-1; ++i)
        {
            loc_rate[i] = std::log(err_loc[i]/err_loc[i+1]) / std::log(dts[i]/dts[i+1]);
            vel_rate[i] = std::log(err_vel[i]/err_vel[i+1]) / std::log(dts[i]/dts[i+1]);
        }

        // print errors and convergence rates
        std::cout << "\nErrors:\n";
        std::cout << std::scientific;
        std::cout << std::setw(20) << "location error" << std::setw(20) << "velocity error" << "\n";
        for (unsigned int i = 0; i < dts.size(); ++i)
        {
            std::cout << std::setw(20) << err_loc[i]
                    << std::setw(20) << err_vel[i] << "\n";
        }

        std::cout << "\nConvergence rates:\n";
        std::cout << std::fixed << std::setprecision(5);
        std::cout << std::setw(20) << "location rate" << std::setw(20) << "velocity rate" << "\n";

        for (unsigned int i = 0; i < dts.size() - 1; ++i)
        {
            std::cout << std::setw(20) << loc_rate[i]
                    << std::setw(20) << vel_rate[i] << "\n";
        }

    };


    /*
     * TestSimulator Definitions
     */

    template<int dim>
    inline void TestSimulator<dim>::initialize()
    {
        if (this->initialized)
        {
            std::cout << "Already initialized with:" << std::endl;
            std::cout << "n# particles: " << this->particle_handler.n_global_particles() << std::endl;
            // std::cout << "n# cells: "     << this->particle_handler.n_global_cells() << std::endl;
            return;
        }


        std::cout << "\n------- TEST SIMULATOR INITIALIZATION -------\n";

        // initialize grid and particles
        initialize_grid();
        initialize_particle();

        std::cout << "dt: " << this->dt << "\nT: " << this->T << std::endl;
        std::cout << "# time steps: " << this->T/this->dt << std::endl;
        std::cout << "# cells: " << this->triangulation.n_active_cells()<< std::endl;
        std::cout << "# particles: " << this->particle_handler.n_global_particles() << std::endl;
        std::cout << "---------------------------------------------\n";
        this->initialized = true;
        return;
    }

    template <int dim>
    inline void TestSimulators::TestSimulator<dim>::initialize_particle()
    {
        location_type initial_location = get_true_location(0.);
        velocity_type initial_velocity(get_true_velocity(0.));

        std::cout << "\ninitial location: " << initial_location;
        std::cout << "\ninitial velocity: " << initial_velocity;

        std::vector<location_type> position(1, initial_location);

        this->particle_handler.insert_particles(position);

        auto it = this->particle_handler.begin();
        this->set_velocity(it, initial_velocity);

        std::cout << "\n * particles inserted * \n";
    }

    template <int dim>
    void TestSimulator<dim>::run_test()
    {
        // initialize the grid and the particle_handler
        this->initialize();

        // get particle iterator and starting data
        auto it = this->particle_handler.begin();

        // make sure to reset DT variable
        REINIT_CUMULATIVE_CHRONO();

        unsigned int NT = this->T/this->dt;
        for (unsigned int i = 0; i<NT; ++i)
        {
            // one-step iteration update using the implemented algorithm (verlet)
            // also timing it
            START_CUMULATIVE_CHRONO();
            this->update_simulation_state();
            ADD_TIME_CUMULATIVE_CHRONO();

            // particle handler reinitialization is skipped because we don't 
            // need it to have the particle reassigned to the correct cell

            // computing the error:
            compute_errors(it);
        }

        // printing the errors and the elipsed time of the simulation
        std::cout << "\ntotal location error: " << location_error << std::endl;
        std::cout << "total velocity error: " << velocity_error << std::endl;
        std::cout << "\nstreaming time:       " << DT_CUMULATIVE_CHRONO << "\n\n"<< std::endl;
    
    }

    template <int dim>
    inline void TestSimulator<dim>::compute_errors(
        typename Particles::ParticleIterator<dim> &particle)
    {
        // get location and velocity of the particle
        location_type location(particle->get_location());
        velocity_type velocity(this->get_velocity_from_properties(particle));

        // get the true location and velocity
        location_type true_location(get_true_location(current_time));
        velocity_type true_velocity(get_true_velocity(current_time));

        // compute the errors
        location_error += norm(location - true_location) * this->dt;
        velocity_error += norm(velocity - true_velocity) * this->dt;

    }

} // namespace TestSimulators

#endif // TEST_SIMULATORS_HPP