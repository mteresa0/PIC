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
#include "streaming.hpp"
#include "domain.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <iomanip>

namespace PICsimulator::TestSimulators
{
    using namespace dealii;

    /// @brief Base class for testing different types of motions.
    /// All the methods are virtual.
    /// @tparam dim 
    template<int dim>
    class BaseMotion {
    public:
        /// @brief Constructor
        BaseMotion() = default;

        /// @brief Destructor
        virtual ~BaseMotion() = default;

        /// @brief get location function
        /// @param t time
        /// @return location vector
        virtual Point<dim> get_loc(const double &t) const = 0;

        /// @brief get velocity function
        /// @param t time
        /// @return velocity vector
        virtual Point<dim> get_vel(const double &t) const = 0;

        /// @brief get acceleration function
        /// @param t time
        /// @return acceleration vector
        virtual Point<dim> get_acc(const double &t) const = 0;
    };

    /// @brief represents a constant velocity particle motion
    template<int dim>
    class ConstantVelocity : public BaseMotion<dim> {
    public:
        /// @brief ConstantVelocity constructor
        /// @param velocity_ constant velocity of the particle
        /// @param initial_location_ initial location of the particle
        ConstantVelocity(
            const Point<dim> &velocity_,
            const Point<dim> &initial_location_):
            velocity(velocity_),
            initial_location(initial_location_)
        {};

        /// @brief get location
        /// @param t 
        Point<dim> get_loc(const double &t) const override;
        /// @brief get velocity
        /// @param t 
        Point<dim> get_vel(const double &t) const override;
        /// @brief get acceleration
        /// @param t 
        Point<dim> get_acc(const double &t) const override;

    private:
        const Point<dim> velocity;
        const Point<dim> initial_location;
    };

    /// @brief represents a periodic particle motion
    template<int dim>
    class PeriodicMotion : public BaseMotion<dim> {  
    public: 
        /// @brief PeriodicMotion constructor. 
        /// Initial location and angular velocity are hardcoded.
        PeriodicMotion(){};     
        /// @brief get location
        /// @param t 
        Point<dim> get_loc(const double &t) const override;
        /// @brief get velocity
        /// @param t 
        Point<dim> get_vel(const double &t) const override;
        /// @brief get acceleration
        /// @param t 
        Point<dim> get_acc(const double &t) const override;
    };

    /// @brief computes convergence rates for the location and velocity errors
    /// @details It runs multiple TestStreaming with difference timesteps. It 
    /// collects the errors and then computes the convergence rate. Errors and 
    /// convergence rates are printed and saved in "outputs/errors.csv"
    /// @tparam dim template parameter for the dimension
    /// @param motion motion object
    /// @param dts vector of time steps
    /// @param total_time final time of the simulation
    template<int dim>
    void convergence_test(  const BaseMotion<dim> &motion,
                            const std::vector<double> &dts,
                            const double &total_time);

    /// @brief TestStreaming class. This class is created as a proto simulator
    /// adapted to test the correctness and the convergence of the streaming
    /// algorithm. It adds 1 particles and computes the difference between the 
    /// true location and velocity of the particle to the computed ones by the 
    /// streaming algorithm.
    template<int dim>
    class TestStreaming{
    public:
        // getting types from Simulator class
        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;

        /// @brief Constructor
        /// @param dt_ timestep
        /// @param T_ total simulation time
        /// @param motion_ particle motion data
        TestStreaming(
            const double &dt_, 
            const double &T_,
            const BaseMotion<dim> &motion_):
            dt(dt_), 
            T(T_),
            streaming_algorithm(
                dt_, 
                std::bind(&TestStreaming::get_acceleration, this)),
            motion(&motion_),
            mapping(1),
            particle_handler(triangulation, mapping, dim)
        {
            dealii::GridGenerator::hyper_cube(triangulation);

            triangulation.refine_global(4);
        };

        /// @brief Destructor
        ~TestStreaming() = default;

        /// @brief run test
        void run_test();

        /// @brief print errors to file
        /// @param filename name of the file
        /// @param overwrite if true, overwrite the file
        void print_errors_to_file( const std::string &filename, 
                                    const bool &overwrite = false);

        /// @brief get location error
        /// @return location error
        double get_loc_error() const { return location_error; }

        /// @brief get velocity error
        /// @return velocity error
        double get_vel_error() const { return velocity_error; }

    private:
        // TODO: move this outside so you dont have to create multiple istances 
        // of TestStreaming when running `test_convergence`
        const double dt; /// time step 
        const double T; /// total simulation time

        double location_error = 0; /// location error
        double velocity_error = 0; /// velocity error
        double current_time = 0.; /// current time

        unsigned int count = 0; /// counts how many times acceleration is called

        Streaming<dim> streaming_algorithm; /// streaming algorithm object

        /// @brief initialize grid and particle handler
        void initialize();

        /// @brief compute errors
        /// @param particle particle iterator
        void compute_errors(
            typename Particles::ParticleIterator<dim> &particle);
            
        /// get acceleration function
        Point<dim> get_acceleration();

        const BaseMotion<dim> *motion; /// motion object

        // dealii objects
        /// @brief triangulation of the domain
        dealii::Triangulation<dim>              triangulation;
        /// @brief finite element mapping
        const dealii::MappingQ<dim>             mapping;

        Particles::ParticleHandler<dim> particle_handler; /// particle handler object

    }; // end TestStreaming class
   
    /****************************
            DEFINITIONS   
    ****************************/

    template <int dim>
    void convergence_test(
        const BaseMotion<dim> &motion, 
        const std::vector<double> &dts, 
        const double &total_time)
    {
        bool overwrite = true;

        std::vector<double> err_loc(dts.size());
        std::vector<double> err_vel(dts.size());

        for(unsigned int i = 0; i<dts.size(); ++i)
        {
            TestStreaming<dim> test_streaming(
                dts[i], total_time, motion);
            
            test_streaming.run_test();
            test_streaming.print_errors_to_file("outputs/errors.csv", overwrite);
            overwrite = false;

            err_loc[i] = test_streaming.get_loc_error();
            err_vel[i] = test_streaming.get_vel_error();

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

    }

    template<int dim>
    Point<dim> ConstantVelocity<dim>::get_loc(const double &t) const { 
        return initial_location + velocity*t;
    }
    template<int dim>
    Point<dim> ConstantVelocity<dim>::get_vel(const double &)const {
        return velocity;
    }
    template<>
    Point<2> ConstantVelocity<2>::get_acc(const double &) const {
        return Point<2>(0, 0);
    }
    template<>
    Point<3> ConstantVelocity<3>::get_acc(const double &) const {
        return Point<3>(0, 0, 0);
    }

    template<>
    Point<2> PeriodicMotion<2>::get_loc(const double &t) const {
        return Point<2>(std::cos(t), std::sin(t));
    }
    template<>
    inline Point<2> PeriodicMotion<2>::get_vel(const double &t) const {
        return Point<2>(-std::sin(t), std::cos(t));
    }
    template<>
    inline Point<2> PeriodicMotion<2>::get_acc(const double &t) const {
        return Point<2>(-std::cos(t), -std::sin(t));
    }

    template<>
    inline Point<3> PeriodicMotion<3>::get_loc(const double &t) const {
        return Point<3>(std::cos(t), std::sin(t), 0);
    }
    template<>
    inline Point<3> PeriodicMotion<3>::get_vel(const double &t) const {
        return Point<3>(-std::sin(t), std::cos(t), 0);
    }
    template<>
    inline Point<3> PeriodicMotion<3>::get_acc(const double &t) const {
        return Point<3>(-std::cos(t), -std::sin(t), 0);
    }

    template <int dim>
    inline void TestStreaming<dim>::print_errors_to_file(
        const std::string &filename, 
        const bool &overwrite)
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

    template <int dim>
    inline void TestSimulators::TestStreaming<dim>::run_test()
    {
        // initialize the grid and the particle_handler
        initialize();

        // get particle iterator and starting data
        auto it = particle_handler.begin();

        // initialize timers
        Timers::Cumulative_Timer timer_streaming;

        unsigned int NT = this->T/this->dt;
        for (unsigned int i = 0; i<NT; ++i)
        {
            // one-step iteration update using the implemented algorithm (verlet)
            // also timing it
            timer_streaming.tic();
            streaming_algorithm.advance(particle_handler);
            timer_streaming.toc();

            current_time += this->dt;

            // particle handler reinitialization is skipped because we don't 
            // need it to have the particle reassigned to the correct cell

            // computing the error:
            compute_errors(it);
        }

        // printing the errors and the elipsed time of the simulation
        std::cout << "\ntotal location error: " << location_error << std::endl;
        std::cout << "total velocity error: " << velocity_error << std::endl;
        std::cout << "\nstreaming time:       " << timer_streaming.get_dt() << "\n\n"<< std::endl;
    
    }

    template <int dim>
    inline void TestSimulators::TestStreaming<dim>::initialize()
    {
        // initialize particle handler
        location_type initial_location = motion->get_loc(0.);
        velocity_type initial_velocity;

        initial_velocity = (motion->get_vel(0.));

        std::cout << "\ninitial location: " << initial_location;
        std::cout << "\ninitial velocity: " << initial_velocity;

        std::vector<location_type> position(1, initial_location);

        particle_handler.insert_particles(position);

        auto it = particle_handler.begin();
        set_velocity(it, initial_velocity);

        std::cout << "\n * particles inserted * \n";
        
    }

    template <int dim>
    inline void TestStreaming<dim>::compute_errors(
        typename Particles::ParticleIterator<dim> &particle)
    {
        // get location and velocity of the particle
        location_type location(particle->get_location());
        velocity_type velocity(get_velocity_from_properties(particle));

        // get the true location and velocity
        location_type true_location(motion->get_loc(current_time));
        velocity_type true_velocity;

        true_velocity = motion->get_vel(current_time);

        // compute the errors
        location_error += norm(location - true_location) * dt;
        velocity_error += norm(velocity - true_velocity) * dt;

        // std::cout << "\nit: " << current_time << "  vel_error    pos_error" << 
        // std::endl  << "          " << norm(velocity - true_velocity) << 
        // "        " << norm(location - true_location)  << std::endl;

        // std::cout << "sim location: " << location << std::endl;
        // std::cout << "ref location: " << true_location << std::endl;
        // std::cout << "sim velocity: " << velocity  << std::endl;
        // std::cout << "ref velocity: " << true_velocity << std::endl;
    }

    template <int dim>
    inline Point<dim> TestSimulators::TestStreaming<dim>::get_acceleration()
    {
        double time = current_time;
        if(count%2!=0){
            time += this->dt;
        }
        ++count;
        return motion->get_acc(time);
        
    }

} // namespace TestSimulators

#endif // TEST_SIMULATORS_HPP