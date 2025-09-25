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
    
    // forward declaration of Simulator
    template<int dim>
    class Simulator;

    /// @brief manges the streaming algorithm.
    /// @details the developed streaming algorithm is the Verlet algorithm.
    /// At each timestep (n), the simulator call for `advance` method which 
    /// updates the particle locations and velocities at the next timestep  
    /// according to the Verlet algorithm.
    template<int dim>
    class Streaming {

        // retireve type from simulator class
        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;

    public:
        /// constructor
        /// @param simulator_ simulator pointer
        explicit Streaming(Simulator<dim> *simulator_) : 
            sim(simulator_) ,
            dt(simulator_->dt), 
            acc_fun(std::bind(&Simulator<dim>::evaluate_acceleration, simulator_))
        {};

        /// constructor for testing
        /// @param dt_ timestep
        /// @param acc_fun_ acceleration function to test
        /// the simulator pointer is not needed since this construct should 
        /// be used only for testing the streaming algorithm.
        explicit Streaming(const double &dt_, 
            const std::function<Point<dim>()> &acc_fun_) : 
            sim(nullptr) ,
            dt(dt_),
            acc_fun(acc_fun_)
        {};

        /// advance the particles according to the Verlet algorithm.
        /// @param ph particle handler
        void advance(typename Particles::ParticleHandler<dim> & ph);

    private:
        /// simulator pointer
        Simulator<dim> *sim;

        /// time step
        const double dt;

        /// update location loop according to the Verlet algorithm
        /// @param ph particle handler
        /// actually it updates the velocities too to the n+1/2 timestep
        void update_location(
            typename Particles::ParticleHandler<dim> & ph) const;

        /// update velocity loop according to the Verlet algorithm
        /// @param ph particle handler
        void update_velocity(
            typename Particles::ParticleHandler<dim> & ph) const;

        /// acceleration function
        std::function<location_type()> acc_fun;
    };

    /************
     DEFINITIONS
    *************/

    template<int dim>
    void Streaming<dim>::advance(
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
    inline void Streaming<dim>::update_location(
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
    inline void Streaming<dim>::update_velocity(
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