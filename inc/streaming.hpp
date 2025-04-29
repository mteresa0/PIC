/**
 * @file streaming.hpp
 * @author Maria Teresa Cazzola
 * @brief Streaming class for the PIC simulator
 * @version 0.1
 * @date 2025-04-19
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef STREAMING_HPP
#define STREAMING_HPP

#include "simulator.hpp"
#include "utilities.hpp"
#include "deal.II/particles/particle_handler.h"
#include <functional>

namespace PICsimulator{

    using namespace dealii;

    enum class StreamingAlgorithm {
        VERLET,
        LEAPFROG
    };
    
    template<int dim>
    class Simulator;


    template<int dim, StreamingAlgorithm sa>
    class Streaming;

    template<int dim>
    class Streaming<dim, StreamingAlgorithm::VERLET> {

        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;

    public:
        /// constructor
        /// @param simulator_ simulator pointer
        /// @param dt time step
        explicit Streaming(Simulator<dim> *simulator_) : 
            sim(simulator_) ,
            dt(simulator_->dt), 
            acc_fun(std::bind(&Simulator<dim>::evaluate_acceleration, simulator_))
        {};

        /// constructor for testing
        explicit Streaming(const double &dt_, 
            const std::function<Point<dim>()> &acc_fun_) : 
            sim(nullptr) ,
            dt(dt_),
            acc_fun(acc_fun_)
        {};

        /// advance the particles
        /// @param ph particle handler
        void advance(typename Particles::ParticleHandler<dim> & ph);

    private:
        /// simulator pointer
        Simulator<dim> *sim;

        /// time step
        const double dt;

        /// update location loop according to the Verlet algorithm
        /// @param ph particle handler
        void update_location(
            typename Particles::ParticleHandler<dim> & ph) const;

        /// update velocity loop according to the Verlet algorithm
        /// @param ph particle handler
        void update_velocity(
            typename Particles::ParticleHandler<dim> & ph) const;

        /// acceleration function
        std::function<location_type()> acc_fun;
    };

    template<int dim>
    class Streaming<dim, StreamingAlgorithm::LEAPFROG> {

        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;

    public:
        /// constructor
        /// @param simulator_ simulator pointer
        /// @param dt time step
        explicit Streaming(Simulator<dim> *simulator_) : 
            sim(simulator_) ,
            dt(simulator_->dt), 
            acc_fun(std::bind(&Simulator<dim>::evaluate_acceleration, simulator_))
        {};

        /// constructor for testing
        explicit Streaming(const double &dt_, 
            const std::function<Point<dim>()> &acc_fun_) : 
            sim(nullptr) ,
            dt(dt_),
            acc_fun(acc_fun_)
        {};

        /// print the streaming algorithm
        void print() const {
            sim->domain_data.print_data_domain();
            std::cout << "Streaming algorithm: LEAPFROG" << std::endl;
        };

        /// advance the particles according to the Leapfrog algorithm
        /// @param ph particle handler
        void advance(typename Particles::ParticleHandler<dim> & ph);

    private:
        /// simulator pointer
        Simulator<dim> *sim;

        /// time step
        const double dt;

        /// acceleration function
        std::function<location_type()> acc_fun;
    };

    /************
     DEFINITIONS
    *************/

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::VERLET>::advance(
        typename Particles::ParticleHandler<dim> & ph)
    {
        // update location
        update_location(ph);

        // // update force field
        // sim->update_force_field();

        // update velocity
        update_velocity(ph);
    }

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::LEAPFROG>::advance(
        typename Particles::ParticleHandler<dim> & ph)
    {
        // initialize acceleration
        location_type acceleration;

        // iterate over particles
        for (auto it = ph.begin(); it!=ph.end(); ++it)
        {
            // get location and velocity
            location_type location(it->get_location());
            velocity_type velocity(get_velocity_from_properties(it));

            // get acceleration
            acceleration = acc_fun();

            // update velocity
            velocity += dt*acceleration;
            set_velocity(it, velocity);
            // update locations
            it->set_location(location + dt*velocity);
        }
    }

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::VERLET>::update_location(
        typename Particles::ParticleHandler<dim> & ph) const
    {
        // initialize acceleration
        location_type acceleration;

        // iterate over particles
        for (auto it = ph.begin(); it!=ph.end(); ++it)
        {
            // get location and velocity
            location_type location(it->get_location());
            velocity_type velocity(get_velocity_from_properties(it));

            // get acceleration
            acceleration = acc_fun();

            // update location
            velocity += 0.5*dt*acceleration;
            it->set_location(location + dt*velocity);

            // update velocity
            set_velocity(it, velocity);
        }
    }

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::VERLET>::update_velocity(
        typename Particles::ParticleHandler<dim> & ph) const
    {
        // initialize acceleration
        location_type acceleration;

        // iterate over particles
        for (auto it = ph.begin(); it!=ph.end(); ++it)
        {
            // get location and velocity
            location_type location(it->get_location());
            velocity_type velocity(get_velocity_from_properties(it));

            // get acceleration
            acceleration = acc_fun();

            // update velocity
            set_velocity(it, velocity + 0.5*dt*acceleration);
        }
    }

}


#endif // STREAMING_HPP