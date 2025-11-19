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

    using species_type = Species::MultiSpecie;

   /// base class for different collisions algorithms.
    template<int dim>
    class CollisionKernel {
    public:
        /// CollisionKernel construtor
        CollisionKernel() = default;
        /// CollisionKernel destructor
        ~CollisionKernel() = default;

        /// compute cross section method (pure virtual)
        virtual double compute_cross_section(
            const unsigned int & /*, const Point<dim> & */) const = 0;

        /// compute deflaction angle chi
        virtual double get_chi(/*const Point<dim> & */) const = 0;

        /// compute deflation angle epsilon
        virtual double get_epsilon() const = 0;

    }; // end CollisionKernel

    template<int dim>
    class HardSphere: public CollisionKernel<dim> {
    public:
        /// HardSphere constructor
        HardSphere() = default;
        /// HardSphere destructor
        ~HardSphere() = default;

        /// @brief Compute the cross section
        /// @details Different implementation for 2d and 3d
        double compute_cross_section(const unsigned int &k
                                    /*, const Point<dim> & relative_velocity*/
                                    ) const override;

        /// @brief Setter for the diameter
        void set_diameter(const species_type& ms);

        /// @brief Compute deflection angle chi.
        /// @details the cosine is randomly picked for a uniform distribution 
        /// from -1 to 1 and then the angle is computed through the acos operator.
        double get_chi(/*const Point<dim> & relative velocity*/) const override{
            return std::acos(Random::rnd_uniform(-1., 1));
        }

        /// @brief Compute the deflection angle epsilon
        /// @details The angle is randomly pick for an uniform distribution 
        /// between -PI and PI.
        double get_epsilon() const override{
            return (Random::rnd_uniform(-M_PI, M_PI));
        }

    private:

        /// Diameter of the particles
        std::vector<double> d12;

        // TODO: make setter/initializer for temperature and simulated viscosity
        /// @brief reference temperature
        double temp_ref = 300; /*K*/
        /// @brief simulated viscosity
        double viscosity = 1.8e-5; /*Pa s*/

    }; // end HardSphere

    /// @brief Manages particle collisions within a simulation.
    /// @tparam dim spatial dimension
    /// @tparam T collision kernel class used to compute cross section and deflection angle
    template<int dim, class T>
    class Collisions {
        using velocity_type = typename Simulator<dim>::velocity_type;
        using location_type = typename Simulator<dim>::location_type;
        using particle_index = types::particle_index; // unsigned int
        using particle_iterator = typename Particles::ParticleHandler<dim>::particle_iterator;
        using ph_type = typename Simulator<dim>::ph_type;

        friend Species::MultiSpecie;

    public:
        // TODO: do you actually need ms?
        /// @brief Collisions Constructor
        explicit Collisions(Simulator<dim> * sim_):
            sim(sim_),
            ms(sim_->get_species())
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
            collision_kernel.set_diameter(ms);

            double sigma_cr_max = collision_kernel.compute_cross_section(0) * 2*
                            ms[0].get_max_velocity();

            unsigned int k = 0;
            for(unsigned int i = 0; i<ms.get_n_species(); ++i){
                double max_v_i = ms[i].get_max_velocity();
                for (unsigned int j = i; i<ms.get_n_species(); ++j){
                    double max_v = std::max(max_v_i, ms[j].get_max_velocity());
                    sigma_cr_max = std::max(sigma_cr_max, 
                        max_v*2*collision_kernel.compute_cross_section(k)
                    );
                    ++k;
                }
            }

            std::cout << "---------- COLLISIONS INFO ------------\n";
            std::cout << "sigma_cr_max:    " << sigma_cr_max << std::endl;
        }

    private:
        //// DATA MEMBERS

        /// simulator pointer
        Simulator<dim> *sim;

        /// collision kernel object
        T collision_kernel;

        //// COLLISION PARAMENTERS
        
        /// species
        species_type ms;

        /// maximum value between the cross section and the relative velocity
        double sigma_cr_max;
        
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
        std::vector<ph_type>& particle_handlers(sim->particle_handlers);
        const DoFHandler<dim>& dof_handler(sim->dof_handler);

        // const double number_density = sim->number_density;
        const double dt = sim->dt;

        for (const auto & cell: dof_handler.active_cell_iterators())
        {
            // compute n particles in cell for each species and the total number
            // of particles in cell 
            std::vector<particle_index> n_pic(ms.get_n_species());
            particle_index N_pic_tot = 0;            
            for( unsigned int i = 0; i<ms.get_n_species(); ++i){
                n_pic[i] = particle_handlers[i].n_particles_in_cell(cell);
                N_pic_tot += n_pic[i];
            }

            // skip cell containing less than 2 particles
            if (N_pic_tot < 2){
                continue;
            }

            // compute fractions of collision for each combination
            std::vector<particle_index> coll_comb_frac(ms.n_combinations);
            {
            unsigned int k = 0;
            for (unsigned int i = 0; i<ms.get_n_species(); ++i) {
                coll_comb_frac[k] = (
                    0.5*n_pic[i]*(n_pic[i]-1))/(0.5*N_pic_tot*(N_pic_tot-1));
                    ++k;
                for (unsigned int j = i+1; j<ms.get_n_species(); ++j){
                    coll_comb_frac[k] = 
                        (n_pic[i]*n_pic[j])/(0.5*N_pic_tot*(N_pic_tot-1));
                    ++k;
                }
            }
            }

            // get the volume of the cell
            auto cell_volume = cell->measure();

            // double n = number_density*cell_volume;

            const double weight = ms[0].get_weight();

            // compute number of collisions
            double N_coll_max_double = 
                0.5*N_pic_tot*(N_pic_tot-1)*weight*sigma_cr_max*dt / cell_volume;
            // double N_coll_max_double_ = 0.5*N*number_density*weight*sigma_cr_max*dt;
            // std::cout << N_coll_max_double <<  "       " << N_coll_max_double_ << std::endl;
            unsigned int N_coll_max = static_cast<unsigned int>(std::floor(N_coll_max_double+0.5));
            unsigned int N_pair_tested = std::floor(N_pic_tot/2);
            if (N_pair_tested > N_coll_max) {
                N_pair_tested = N_coll_max;
            }
            double F_correction = N_coll_max/N_pair_tested;

            // std::cout << "# collisions in cell " << cell->index() << " : " << N_coll_max << "    "<< N_pair_tested  << "    F_corr: "  << F_correction<< std::endl;

            for (unsigned int ind1 = 0; ind1<ms.get_n_species(); ++ind1) {
                auto pic1 = particle_handlers[ind1].particles_in_cell(cell);
                // get the begin iterator of the particles in the cell
                const auto & pic_begin1 = pic1.begin();  
                for (unsigned int ind2=ind1+1; ind2<ms.get_n_species(); ++ind2){
                
                // combination index
                unsigned int ind_comb = ms.combination_index(ind1, ind2);

                // get the particles in the cell
                auto pic2 = particle_handlers[ind2].particles_in_cell(cell);

                // get the begin iterator of the particles in the cell
                const auto & pic_begin2 = pic2.begin(); 

                unsigned int N_pair_tested_comb = N_pair_tested*coll_comb_frac[ind_comb];           
                
                for (unsigned int k = 0; k<N_pair_tested_comb; )
                {
                    // select two particles
                    unsigned int i = static_cast<unsigned int>(
                        std::floor(Random::rnd_uniform() * n_pic[ind1]));
                    unsigned int j = static_cast<unsigned int>(
                        std::floor(Random::rnd_uniform() * n_pic[ind2]));
                    if(ind1==ind2){
                        while (i==j) {
                            j = static_cast<unsigned int>(
                                std::floor(Random::rnd_uniform() * n_pic[ind1]));
                        }
                    }

                    // get the two particles
                    auto it_i = std::next(pic_begin1, i);
                    auto it_j = std::next(pic_begin2, j);

                    // get the velocities of the two particles
                    velocity_type v_i(get_velocity_from_properties(it_i));
                    velocity_type v_j(get_velocity_from_properties(it_j));

                    // compute the relative velocity
                    velocity_type cr(v_i - v_j);
                    const double cr_norm(cr.norm());

                    // compute system velocity
                    velocity_type cm((v_i + v_j) / 2.0);

                    // compute cross section
                    double sigma = collision_kernel.compute_cross_section(ind_comb);
            
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
            }}
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

        template <int dim>
    inline void HardSphere<dim>::set_diameter(const species_type &ms)
    {
        unsigned int L = ms.get_n_species();
        unsigned int M = int(1/2 * L * (L-1));

        d12.reserve(M);

        for(unsigned int i = 0; i<M; ++M){
            double mr = ms.get_reduced_mass(i);
            double d2 = 5/16 * std::sqrt(BOLTZMANN_K*mr*temp_ref/
                (M_PI))/viscosity;
            d12.push_back(std::sqrt(d2));
        }

        std::cout << "-------- set ref diameters ----------\n";
        std::cout << "d_ref = [ " << d12[0];
        for (unsigned int i = 1; i<M; ++i){
            std::cout << "\n         " << d12[i];
        }
        std::cout << " ]\n\n";
    }

    /// compute the cross section.
    /// @details in 2D the cross section is the diameter
    template<>
    inline double HardSphere<2>::compute_cross_section(
        const unsigned int & k
    ) const
    {
        return d12[k];
    }

    /// compute the cross section.
    /// @details in 3D the cross section is the circle area of the given diameter
    template<>
    inline double HardSphere<3>::compute_cross_section(
        const unsigned int & k
    ) const
    {
        return M_PI * d12[k]*d12[k];
    }

} // namespace PICsimulator

#endif // COLLISIONS_HPP
