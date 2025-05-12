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
        HardSphere(const double & d12_=1e-8):
            d12(d12_) {};
        /// HardSphere destructor
        ~HardSphere() = default;

        /// compute cross section
        double compute_cross_section(/*const Point<dim> & relative velocity*/) const override;

        /// setter for the diameter
        void set_diameter(const double & d12_){
            d12 = d12_;
        }

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
        double d12;
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
            sim(sim_),
            N2(sim_->get_species())
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

        void initialize() {;
            collision_kernel.set_diameter(N2.get_diameter());
            sigma_cr_max = collision_kernel.compute_cross_section()*(N2.get_mean_velocity() + 3.0*N2.get_std_velocity());


            std::cout << "---------- COLLISIONS INFO ------------\n";
            std::cout << "sigma_cr_max: " << sigma_cr_max << std::endl;
            std::cout << "sigma cr " << collision_kernel.compute_cross_section()*N2.get_mean_velocity() << std::endl;
        }

    private:
        //// DATA MEMBERS

        /// simulator pointer
        Simulator<dim> *sim;

        /// collision kernel object
        T collision_kernel;

        //// COLLISION PARAMENTERS
        Species::N2 N2;

        /// maximum value between the cross section and the relative velocity
        double sigma_cr_max;

        /// weight of computational particles
        double weight = 100;
        
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

        const double number_density = sim->get_number_density();
        const double dt = sim->get_time_step();

        for (const auto & cell: dof_handler.active_cell_iterators())
        {
            particle_index N = ph.n_particles_in_cell(cell);

            // skip cell containing less than 2 particles
            if (N < 2){
                continue;
            }
            // std::cout << "# particles in cell "<< cell->index() <<" : " << N << std::endl;
            
            // get the particles in the cell
            auto pic = ph.particles_in_cell(cell);
            // get the begin iterator of the particles in the cell
            const auto & pic_begin = pic.begin();            

            // get the volume of the cell
            auto cell_volume = cell->measure();
            // double n = number_density*cell_volume;

            // compute number of collisions
            double N_coll_max_double = 0.5*N*(N-1)*weight*sigma_cr_max*dt/cell_volume;
            unsigned int N_coll_max = static_cast<unsigned int>(std::floor(N_coll_max_double+0.5));
            unsigned int N_pair_tested;
            if (N_coll_max > std::floor(N/2)) {
                N_pair_tested = std::floor(N/2);
            } else {
                N_pair_tested = N_coll_max;
            }
            double F_correction = N_coll_max/N_pair_tested;
            
            // std::cout << "# collisions in cell " << cell->index() << " : " << N_coll_max << " "<< N_pair_tested  << "  F_corr: "  << F_correction<< std::endl;

            for (unsigned int k = 0; k<N_pair_tested; )
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
                double P_coll = F_correction * sigma * cr.norm() / sigma_cr_max;

                // pick random double [0, 1)
                double U = Random::RND();

                if(P_coll>1.0){
                    std::cout << "P>1.0: " << P_coll << std::endl;
                }

                if (U<P_coll)
                {
                    // compute post collisional relative velocity
                    velocity_type cr_ = compute_post_collisional_velocity(cr);

                    // compute and update new 
                    // std::cout << "\nv_i: " << norm(v_i) ;
                    // std::cout << "         v_j: " << norm(v_j) << std::endl;
                    update_velocities(it_i, it_j, cm, cr_);

                    ++k;
                    if ((sigma*cr.norm())/sigma_cr_max > 1.0)
                    {
                        sigma_cr_max = sigma * cr.norm();
                        std::cout << "sigma_cr_max: " << sigma_cr_max << std::endl;
                    }
                }
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

        // std::cout << "v_i_new: " << norm(v_i_new);
        // std::cout << "       v_j_new: " << norm(v_j_new) << std::endl;

        // set the new velocities
        set_velocity(it_i, v_i_new);
        set_velocity(it_j, v_j_new);
    }

    template<>
    inline double HardSphere<2>::compute_cross_section() const
    {
        return d12;
    }

    template<>
    inline double HardSphere<3>::compute_cross_section() const
    {
        return M_PI * d12*d12;
    }

} // namespace PICsimulator

#endif // COLLISIONS_HPP
