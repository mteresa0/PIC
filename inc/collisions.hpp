/**
 * @file collisions.hpp
 * @author Maria Teresa Cazzola 
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

namespace PICsimulator {

    using namespace dealii;

    enum class CollisionType {
        HS,     // hard sphere
        VHS,    // variable hard sphere
        VSS,    // variable soft sphere
        INVALID
    };

    template<int dim>
    class CollisionKernel {
    public:
        CollisionKernel() = default;
        ~CollisionKernel() = default;

        virtual double compute_cross_section(/*const Point<dim> & */) const = 0;
        virtual double get_chi(/*const Point<dim> & */) const = 0;
        virtual double get_epsilon() const = 0;
    }; // end CollisionKernel

    template<int dim>
    class HardSphere: public CollisionKernel<dim> {
    public:
        /// HardSphere constructor
        HardSphere(const double & d12_=1e-12):
            d12(d12_) {};
        /// HardSphere destructor
        ~HardSphere() = default;

        /// compute cross section
        double compute_cross_section(/*const Point<dim> & relative velocity*/) const override {
            return M_PI * d12*d12;
        };

        /// compute cosine of the deflection angle chi
        double get_chi(/*const Point<dim> & relative velocity*/) const override{
            return std::acos(Random::RND()*2.0 - 1.0);
        }

        /// compute the deflection angle epsilon
        double get_epsilon() const override{
            return Random::RND()*2.0*M_PI-1.0;
        }

    private:
        /// diameter of the particles
        const double d12;
    }; // end HardSphere

    /// @brief Manages particle collisions within a simulation.
    /// @details The central function, `apply_collisions()`, iterates through each cell, 
    /// estimates the number of potential collisions, and randomly selects particle pairs. 
    /// For each pair, a collision probability is calculated. If this probability exceeds 
    /// a randomly generated value, the particles' velocities are updated based on the 
    /// provided collision kernel.
    /// @tparam dim spatial dimension
    /// @tparam T collision kernel class used to compute cross section and deflection angle
    template<int dim, class T>
    class Collisions {
        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;
        using particle_index = types::particle_index;
        using particle_iterator = typename Particles::ParticleHandler<dim>::particle_iterator;

    public:
        /// @brief Collisions Constructor
        explicit Collisions(Simulator<dim> * sim_):
            sim(sim_)
        {};
        /// @brief Collisions Destructor
        ~Collisions() = default;

        /// @brief apply collisions
        /// @details This function iterates through each cell in the simulation,
        /// estimates the number of potential collisions, and randomly selects
        /// particle pairs. For each pair, a collision probability is calculated.
        /// If this probability exceeds a randomly generated value, the particles'
        /// velocities are updated based on the provided collision kernel.
        void apply_collisions();

    private:
        //// DATA MEMBERS

        /// simulator pointer
        Simulator<dim> *sim;

        /// collision kernel object
        T collision_kernel;

        //// COLLISION PARAMENTERS

        /// maximum value between the cross section and the relative velocity
        double sigma_cr_max= ;

        /// weight of computational particles
        double weight = 10;
        
        //// COLLISION FUNCTIONS

        /// compute post-collisional relative velocity
        Tensor<1,dim> compute_post_collisional_velocity(const velocity_type & cr) const;

        /// compute and update new velocities of the particles
        void update_velocities(
            particle_iterator & it_i, 
            particle_iterator & it_j,
            const velocity_type & cm,
            const velocity_type & cr_);
    }; // end class Collisions

    template <int dim, class T>
    inline void Collisions<dim, T>::apply_collisions()
    {
        // get the particle handler and dof handler
        Particles::ParticleHandler<dim>& ph = sim->get_particle_handler();
        const DoFHandler<dim>& dof_handler = sim->get_dof_handler();

        for (const auto & cell: dof_handler.active_cell_iterators())
        {
            particle_index N = ph.n_particles_in_cell(cell);
            if (N < 2){
                continue;
            }
            std::cout << "# particles in cell "<< cell->index() <<" : " << N << std::endl;
            
            auto pic = ph.particles_in_cell(cell);
            const auto & pic_begin = pic.begin();            

            auto cell_volume = cell->measure();

            // compute number of collisions
            double N_coll = 0.5*N*N*weight*sigma_cr_max/cell_volume;
            unsigned int int_N_coll = static_cast<unsigned int>(N_coll);
            
            // if (N_coll>0)
            std::cout << "# collisions in cell " << cell->index() << " : " << N_coll << " "<< int_N_coll << std::endl;

            for (unsigned int k = 0; k<int_N_coll; )
            {
                // select two particles
                unsigned int i = Random::RND() * N;
                unsigned int j = Random::RND() * N;
                while (i==j) {
                    j = Random::RND() * N;
                }

                // get the two particles
                auto it_i = std::next(pic_begin, i);
                auto it_j = std::next(pic_begin, j);

                // get the velocities of the two particles
                velocity_type v_i(get_velocity_from_properties(it_i));
                velocity_type v_j(get_velocity_from_properties(it_j));

                // compute the relative velocity
                velocity_type cr(v_i - v_j);

                // compute system velocity
                velocity_type cm((v_i + v_j) / 2.0);

                // compute cross section
                double sigma = collision_kernel.compute_cross_section();
        
                // compute collision probability
                double P_coll = sigma * cr.norm() / sigma_cr_max;

                // pick random double [0, 1)
                double U = Random::RND();

                if (U<P_coll)
                {
                    // compute post collisional relative velocity
                    velocity_type cr_ = compute_post_collisional_velocity(cr);

                    // compute and update new velocities
                    update_velocities(it_i, it_j, cm, cr_);

                    ++k;
                }
                // std::cout <<" sigma*cr " << sigma * cr.norm() << "  probability" << P_coll << std::endl;
                // if (sigma * cr.norm() > sigma_cr_max)
                // {
                //     sigma_cr_max = sigma * cr.norm();
                //     std::cout << "    New max: " << sigma_cr_max << std::endl;
                // }
            }
        }
        
    } // end apply_collisions

    template <int dim, class T>
    Tensor<1, dim> Collisions<dim, T>::compute_post_collisional_velocity(const velocity_type & cr) const
    {

        // initialize post-collisional relative velocity
        velocity_type cr_;

        // compute the deflection angle chi
        double chi = collision_kernel.get_chi();

        if constexpr (dim == 2)
        {
            // in 2D, the deflection angle epsilon is not needed
            cr_[0] = norm(cr) * std::cos(chi);
            cr_[1] = norm(cr) * std::sin(chi);
        } else {
        
            // compute the deflection angle epsilon
            double epsilon = collision_kernel.get_epsilon();

            cr_[0] = norm(cr) * std::cos(chi);
            cr_[1] = norm(cr) * std::sin(chi) * std::cos(epsilon);
            cr_[2] = norm(cr) * std::sin(chi) * std::sin(epsilon);

        }
        return cr_;
    } // end compute_post_collisional_velocity

    template <int dim, class T>
    inline void Collisions<dim, T>::update_velocities(
        particle_iterator & it_i, 
        particle_iterator & it_j,
        const velocity_type & cm,
        const velocity_type & cr_)
    {
        // compute the new velocities
        velocity_type v_i_new(cm + cr_ / 2.0);
        velocity_type v_j_new(cm - cr_ / 2.0);

        // set the new velocities
        set_velocity(it_i, v_i_new);
        set_velocity(it_j, v_j_new);
    }


} // namespace PICsimulator


#endif // COLLISIONS_HPP
