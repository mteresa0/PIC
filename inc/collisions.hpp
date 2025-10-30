/**
 * @file collisions.hpp
 * @author Maria Teresa Cazzola 
 * @brief Collisions kernels (HardShpere, ) and the collision algorithm are implemented here.
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
#include "species.hpp"

namespace PICsimulator {

    using namespace dealii;

   /// base class for different collisions algorithms.
    template<int dim>
    class CollisionKernel {
    public:
        /// CollisionKernel construtor
        CollisionKernel() = default;
        /// CollisionKernel destructor
        ~CollisionKernel() = default;

        /// compute cross section method (pure virtual)
        virtual double compute_cross_section(/*const Point<dim> & */) const = 0;

        /// compute deflaction angle chi
        virtual double get_chi(/*const Point<dim> & */) const = 0;

        /// compute deflation angle epsilon
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

        /// @brief Compute the cross section
        /// @details Different implementation for 2d and 3d
        double compute_cross_section(/*const Point<dim> & relative velocity*/) const override;

        /// @brief Setter for the diameter
        void set_diameter(const double & d12_){
            d12 = d12_;
        }

        /// @brief Compute deflection angle chi.
        /// @details the cosine is randomly picked for a uniform distribution from -1 to 1 and then the angle is computed through the acos operator.
        double get_chi(/*const Point<dim> & relative velocity*/) const override{
            return std::acos(Random::rnd_uniform(-1., 1));
        }

        /// @brief Compute the deflection angle epsilon
        /// @details The angle is randomly pick for an uniform distribution between -PI and PI.
        double get_epsilon() const override{
            return (Random::rnd_uniform(-M_PI, M_PI));
        }

    private:

        /// Diameter of the particles
        double d12;

    }; // end HardSphere

    /// @brief Manages particle collisions within a simulation.
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
            data_species(sim_->get_species())
        {};
        /// @brief Collisions Destructor
        ~Collisions() = default;

        /// @brief Run the collision algorithm.
        /// @details This function iterates through each cell in the simulation,
        /// estimates the number of potential collisions and randomly selects
        /// particle pairs. For each pair the collision probability is calculated.
        /// According to this probability the particle velocities are updated based
        /// on the proveded `collision_kernel`
        void apply_collisions();

        /// @brief initialize collision framework
        /// @details setup collision kernel and inititialize `sigma_cr_max`. Printing initial collision info.
        void initialize() {;
            // setup kernel
            collision_kernel.set_diameter(data_species.get_diameter());
            // initialize `sigma_cr_max`
            sigma_cr_max = collision_kernel.compute_cross_section()*(data_species.get_mean_velocity() + 3.0*data_species.get_std_velocity());
            /* 
            max velocity is computed as:
                v_max = v_mean + 3*v_std
            This value represents the 99.87th percentile for a velocity drawn from a normal distribution (which is not)
            */

            std::cout << "---------- COLLISIONS INFO ------------\n";
            std::cout << "sigma_cr_max:    " << sigma_cr_max << std::endl;
            std::cout << "sigma cr mean:   " << collision_kernel.compute_cross_section()*data_species.get_mean_velocity() << std::endl;
        }

    private:
        //// DATA MEMBERS

        /// simulator pointer
        Simulator<dim> *sim;

        /// collision kernel object
        T collision_kernel;

        //// COLLISION PARAMENTERS
        
        /// species
        Species::ChemicalSpecies data_species;

        /// maximum value between the cross section and the relative velocity
        double sigma_cr_max;

        // /// weight of computational particles
        // double weight = 100;
        
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

        const double weight = data_species.get_weight();

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
            double N_coll_max_double = 0.5*N*(N-1)*weight*sigma_cr_max*dt / cell_volume;
            // double N_coll_max_double_ = 0.5*N*number_density*weight*sigma_cr_max*dt;
            // std::cout << N_coll_max_double <<  "       " << N_coll_max_double_ << std::endl;
            unsigned int N_coll_max = static_cast<unsigned int>(std::floor(N_coll_max_double+0.5));
            unsigned int N_pair_tested = std::floor(N/2);
            if (N_pair_tested > N_coll_max) {
                N_pair_tested = N_coll_max;
            }
            double F_correction = N_coll_max/N_pair_tested;
            
            // std::cout << "# collisions in cell " << cell->index() << " : " << N_coll_max << "    "<< N_pair_tested  << "    F_corr: "  << F_correction<< std::endl;

            for (unsigned int k = 0; k<N_pair_tested; )
            {
                // select two particles
                unsigned int i = static_cast<unsigned int>(std::floor(Random::rnd_uniform() * N));
                unsigned int j = static_cast<unsigned int>(std::floor(Random::rnd_uniform() * N));
                while (i==j) {
                    j = static_cast<unsigned int>(std::floor(Random::rnd_uniform() * N));
                }

                // get the two particles
                auto it_i = std::next(pic_begin, i);
                auto it_j = std::next(pic_begin, j);

                // get the velocities of the two particles
                velocity_type v_i(get_velocity_from_properties(it_i));
                velocity_type v_j(get_velocity_from_properties(it_j));

                // compute the relative velocity
                velocity_type cr(v_i - v_j);
                const double cr_norm(cr.norm());

                // compute system velocity
                velocity_type cm((v_i + v_j) / 2.0);

                // compute cross section
                double sigma = collision_kernel.compute_cross_section();
        
                // compute collision probability
                double P_coll = (F_correction * sigma * cr_norm )/ sigma_cr_max;

                // pick random double [0, 1)
                double U = Random::rnd_uniform();

                // if(P_coll>1.0){
                //     std::cout << "P>1.0: " << P_coll << "      " << sigma * cr_norm << "    " << F_correction << "      "  << sigma_cr_max << std::endl;
                // }

                if (U < P_coll)
                {
                    // compute post collisional relative velocity
                    velocity_type cr_ = compute_post_collisional_velocity(cr);

                    // compute and update new velocities
                    update_velocities(it_i, it_j, cm, cr_);

                    ++k;

                    if (sigma * cr_norm > sigma_cr_max)
                    {
                        sigma_cr_max = sigma * cr_norm;
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

    /// compute the cross section.
    /// @details in 2D the cross section is the diameter
    template<>
    inline double HardSphere<2>::compute_cross_section() const
    {
        return d12;
    }

    /// compute the cross section.
    /// @details in 3D the cross section is the circle area of the given diameter
    template<>
    inline double HardSphere<3>::compute_cross_section() const
    {
        return M_PI * d12*d12;
    }

} // namespace PICsimulator

#endif // COLLISIONS_HPP
