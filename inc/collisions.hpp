/**
 * @file collisions.hpp
 * @author Maria Teresa Cazzola (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2025-04-29
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef COLLISIONS_HPP
#define COLLISIONS_HPP

#include "simulator.hpp"
#include "utilities.hpp"
#include "deal.II/particles/particle_handler.h"

enum class CollisionType {
    HS,     // hard sphere
    VHS,    // variable hard sphere
    VSS,    // variable soft sphere
    INVALID
};

template<int dim, CollisionType CT>
class Collisions {
    using namespace dealii;
    using velocity_type = typename Simulator<dim>::velocity_type;
    using location_type = typename Simulator<dim>::location_type;
    using particle_index = typename Particles::ParticleHandler<dim>::particle_index;

public:
    explicit Collisions(Simulator & sim_):
        sim(&sim_),
        gen(rd())
    {};
    ~Collisions() = default;

    void apply_collisions();

private:
    ///
    Simulator<dim> *sim;

    //// COLLISION PARAMENTERS
    /// maximum value between the cross section and the relative velocity
    double sigma_cr_max;

    /// weight of computational particles
    unsigned int weight = 10;

    //// random number generator
    /// @brief random device
    std::random_device rd;
    /// @brief random number generator
    std::mt19937 gen;
    /// @brief uniform distribution for collision probability
    std::uniform_real_distribution<double> uniform_distr{0., 1.};
    
    /// compute cross section
    const double compute_cross_section(const double & sigma) const;

    /// compute post-collisional relative velocity
    Point<dim> & compute_post_collisional_velocity(const Point<dim> & cr) const;
};

template <int dim, CollisionType CT>
inline void Collisions<dim, CT>::apply_collisions()
{
    // get the particle handler
    auto &ph = sim->get_particle_handler();
    const auto &dof_handler = sim->get_dof_handler();

    for (const auto & cell: dof_handler->active_cell_iterators())
    {
        particle_index N = ph->n_particles_in_cell(cell);
        auto pic = ph->particles_in_cell(cell);

        auto cell_volume = cell->measure();

        // compute number of collisions
        double N_coll = 0.5*N*N*weight*sigma_cr_max/cell_volume;

        for (unsigned int k = 0; k<N_coll; ++k)
        {
            // select two particles
            unsigned int i = std::rand() % N;
            unsigned int j = std::rand() % N;

            // get the two particles
            auto it_i = pic[i];
            auto it_j = pic[j];

            // get the velocities of the two particles
            velocity_type v_i(get_velocity_from_properties(it_i));
            velocity_type v_j(get_velocity_from_properties(it_j));

            // compute the relative velocity
            Point<dim> cr = v_i - v_j;

            // compute cross section
            double sigma = compute_cross_section();
    
            // compute collision probability
            double P_coll = sigma * cr.norm() / sigma_cr_max;

            // pick random double [0, 1)
            double U = uniform_distr(gen);

            if (U<P_coll)
            {
                // compute new velocities and assigned them to the particles
                // TODO

            }
        }
    }
    
}


#endif // COLLISIONS_HPP
