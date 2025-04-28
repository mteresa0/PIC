#ifndef UTILITIES_PIC_HPP
#define UTILITIES_PIC_HPP

#include <deal.II/base/point.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/particle_accessor.h>

namespace PICsimulator {
    using namespace dealii;
    
    /// @brief get the velocity of a particle
    /// @param particle particle pointer (particle iterator type).
    /// @return velocity of the particle. Tensor<1, dim> type.
    template<int dim>
    Tensor<1, dim> get_velocity_from_properties(
        const typename Particles::ParticleIterator<dim> &particle)
    {
        return Tensor<1, dim>(particle->get_properties());
    }

    /// @brief set the velocity of a particle
    /// @param particle particle pointer (particle iterator type).
    /// @param velocity velocity of the particle to set.
    template<int dim>
    void set_velocity(
        typename Particles::ParticleIterator<dim> &particle, 
        const Tensor<1, dim> &velocity)
    {
        particle->set_properties(velocity);
        return;
    }



} // namespace PICsimulator

#endif // UTILITIES_PIC_HPP