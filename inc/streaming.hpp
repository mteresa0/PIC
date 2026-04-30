/**
 * @file streaming.hpp
 * @author Maria Teresa Cazzola
 * @brief Streaming class for the PIC simulator
 * @date 2025-04-19
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef STREAMING_HPP
#define STREAMING_HPP

#include "simulator.hpp"
#include "utilities.hpp"
#include "solver.hpp"
#include "deal.II/particles/particle_handler.h"
#include <functional>

namespace PICsimulator{

    using namespace dealii;
    
    // forward declaration of Simulator
    template<int dim>
    class Simulator;

    /// @brief manages the streaming algorithm.
    /// @details the developed streaming algorithm is the Verlet algorithm.
    /// At each timestep (n), the simulator call for `advance` method which 
    /// updates the particle locations and velocities at the next timestep  
    /// according to the Verlet algorithm.
    template<int dim>
    class Streaming {

        // retireve type from simulator class
        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;
        using ph_type = dealii::Particles::ParticleHandler<dim>;

    public:
        /// constructor
        /// @param dt_ timestep
        /// @param solver_ solves potential field and computes accelerations
        explicit Streaming( const double & dt_, 
                            Solver<dim>& solver_,
                            std::vector<ph_type>& phs_) : 
            dt(dt_),
            solver(solver_),
            phs(phs_)
        {};

        // /// constructor for testing
        // /// @param dt_ timestep
        // /// @param acc_fun_ acceleration function to test
        // /// the simulator pointer is not needed since this construct should 
        // /// be used only for testing the streaming algorithm.
        // explicit Streaming(const double &dt_, 
        //     const std::function<Point<dim>()> &acc_fun_) : 
        //     dt(dt_),
        //     acc_fun(acc_fun_)
        // {};

        /// advance the particles according to the Verlet algorithm.
        /// @param ph particle handler
        void advance();

    private:

        /// @brief time step
        const double dt;

        /// @brief update potential field and computes accelerations
        Solver<dim>& solver;

        /// @brief vector of particle handlers
        std::vector<ph_type>& phs;

        /// @brief update location loop according to the Verlet algorithm
        /// @details actually it updates the velocities too to the n+1/2 timestep
        void update_location() const;

        /// @brief update velocity loop according to the Verlet algorithm
        void update_velocity() const;

        // /// acceleration function
        // std::function<location_type()> acc_fun;
    };

    /************
     DEFINITIONS
    *************/

    template<int dim>
    void Streaming<dim>::advance()
    {
        // update location
        update_location();

        // // update force field
        // solver.run_solver();

        // update velocity
        update_velocity();
    }

    template<int dim>
    inline void Streaming<dim>::update_location() const
    {
        // initialize acceleration
        location_type acceleration;

        for(auto& ph : phs)
            // iterate over particles
            for (auto it = ph.begin(); it!=ph.end(); ++it)
            {
                // get location and velocity
                location_type location(it->get_location());
                velocity_type velocity(get_velocity_from_properties(it));

                // get acceleration
                acceleration = get_acceleration(it);

                // update location
                velocity += 0.5*dt*acceleration;
                it->set_location(location + dt*velocity);

                // update velocity
                set_velocity(it, velocity);
            }
    }

    template<int dim>
    inline void Streaming<dim>::update_velocity() const
    {
        // initialize acceleration
        location_type acceleration;

        for(auto& ph : phs)
            // iterate over particles
            for (auto it = ph.begin(); it!=ph.end(); ++it)
            {
                // get velocity
                velocity_type velocity(get_velocity_from_properties(it));

                // get acceleration
                acceleration = get_acceleration(it);

                // update velocity
                set_velocity(it, velocity + 0.5*dt*acceleration);
            }
    }

}


#endif // STREAMING_HPP