
#ifndef STREAMING_HPP
#define STREAMING_HPP

#include "simulator.hpp"
#include "deal.II/particles/particle_handler.h"

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
        Streaming(Simulator<dim> *simulator_) : sim(simulator_) {
            std::cout << "Streaming algorithm: VERLET" << std::endl;
        };

        /// advance the particles
        /// @param ph particle handler
        /// @param dt time step
        void advance(   
            typename Particles::ParticleHandler<dim> & ph, 
            const double & dt);
    private:
        /// simulator pointer
        Simulator<dim> *sim;

        void update_location(
            typename Particles::ParticleHandler<dim> & ph,
            const double & dt) const;

        void update_velocity(
            typename Particles::ParticleHandler<dim> & ph,
            const double & dt) const;
    };

    template<int dim>
    class Streaming<dim, StreamingAlgorithm::LEAPFROG> {

        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;
    public:
        explicit Streaming(Simulator<dim> *simulator_) : sim(simulator_) {
            std::cout << "Streaming algorithm: LEAPFROG" << std::endl;
        };

        void print() const {
            sim->domain_data.print_data_domain();
            std::cout << "Streaming algorithm: LEAPFROG" << std::endl;
        };

        void advance(   typename Particles::ParticleHandler<dim> & ph, 
                        const double & dt);
    private:
        /// simulator pointer
        Simulator<dim> *sim;
    };

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::VERLET>::advance(
        typename Particles::ParticleHandler<dim> & ph,
        const double & dt)
    {
        // update location
        update_location(dt);

        // // update force field
        // sim->update_force_field();

        // update velocity
        update_velocity(dt);
    }

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::LEAPFROG>::advance(
        typename Particles::ParticleHandler<dim> & ph,
        const double & dt)
    {
        // initialize acceleration
        location_type acceleration;

        // iterate over particles
        for (auto it = ph.begin(); it!=ph.end(); ++it)
        {
            // get location and velocity
            location_type location(it->get_location());
            velocity_type velocity(sim->get_velocity_from_properties(it));

            // get acceleration
            acceleration = sim->evaluate_acceleration();

            // update velocity
            velocity += dt*acceleration;
            sim->set_velocity(it, velocity);
            // update location
            it->set_location(location + dt*velocity);
        }
    }

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::VERLET>::update_location(
        typename Particles::ParticleHandler<dim> & ph,
        const double & dt) const
    {
        // initialize acceleration
        location_type acceleration;

        // iterate over particles
        for (auto it = ph.begin(); it!=ph.end(); ++it)
        {
            // get location and velocity
            location_type location(it->get_location());
            velocity_type velocity(sim->get_velocity_from_properties(it));

            // get acceleration
            acceleration = sim->evaluate_acceleration();

            // update location
            velocity += 0.5*dt*acceleration;
            it->set_location(location + dt*velocity);

            // update velocity
            sim->set_velocity(it, velocity);
        }
    }

    template<int dim>
    inline void Streaming<dim, StreamingAlgorithm::VERLET>::update_velocity(
        typename Particles::ParticleHandler<dim> & ph,
        const double & dt) const
    {
        // initialize acceleration
        location_type acceleration;

        // iterate over particles
        for (auto it = ph.begin(); it!=ph.end(); ++it)
        {
            // get location and velocity
            location_type location(it->get_location());
            velocity_type velocity(sim->get_velocity_from_properties(it));

            // get acceleration
            acceleration = sim->evaluate_acceleration();

            // update velocity
            sim->set_velocity(velocity + 0.5*dt*acceleration);
        }
    }

}


#endif // STREAMING_HPP