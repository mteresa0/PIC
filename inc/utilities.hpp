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

    /// @brief compute L2 norm
    /// @param v dealii::Tensor<1, dim> velocity vector
    /// @return L2 norm of the input Tensor<1, dim> v
    template<int dim>
    double norm(const Tensor<1,dim> & l){
        double r = 0;
        for (int d = 0; d<dim; ++d){
            r += l[d]*l[d];
        }
        return std::sqrt(r);
    };

    /// @brief compute L2 norm
    /// @param l dealii::Point<dim> location vector
    /// @return L2 norm of the input Point<dim> l
    template<int dim>
    double norm(const Point<dim> & l){
        double r = 0;
        for (int d = 0; d<dim; ++d){
            r += l[d]*l[d];
        }
        return std::sqrt(r);
    };

} // namespace PICsimulator

#endif // UTILITIES_PIC_HPP