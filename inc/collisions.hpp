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

            sigma_cr_max = collision_kernel.compute_cross_section(0) * 2*
                            ms[0].get_max_velocity();

            unsigned int k = 0;
            for(unsigned int i = 0; i<ms.get_n_species(); ++i){
                double max_v_i = ms[i].get_max_velocity();
                for (unsigned int j = i; j<ms.get_n_species(); ++j){
                    double max_v = std::max(max_v_i, ms[j].get_max_velocity());
                    sigma_cr_max = std::max(sigma_cr_max, 
                        max_v*2*collision_kernel.compute_cross_section(k)
                    );
                    ++k;
                }
            }

            LOGTITLE("COLLISIONS INFO");
            LOGVAR("sigma_cr_max", sigma_cr_max);
        }

    private:
        //// DATA MEMBERS

        /// simulator pointer
        Simulator<dim> *sim;

        /// collision kernel object
        T collision_kernel;

        //// COLLISION PARAMENTERS
        
        /// species
        const species_type& ms;

        /// maximum value between the cross section and the relative velocity
        double sigma_cr_max;

        #ifdef EXTRA_OUTPUTS
            CSVoutput out_iterations{"outputs/collisions_it.csv", {"P_coll", "F_correction", "sigma", "sigma/sigma_cr_max"}};
            CSVoutput out_n_coll{"outputs/n_coll.csv", {"N_coll_max", "N_pait_tested", "N_pic_tot"}};    
        #endif
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
            std::vector<double> coll_comb_frac(ms.n_combinations);
            {
            unsigned int k = 0;
            double total_collisions = (0.5*N_pic_tot*(N_pic_tot-1));
            for (unsigned int i = 0; i<ms.get_n_species(); ++i) {
                coll_comb_frac[k] = 
                    ((0.5*n_pic[i]*(n_pic[i]-1))/total_collisions);
                    ++k;
                for (unsigned int j = i+1; j<ms.get_n_species(); ++j){
                    coll_comb_frac[k] = 
                        ((n_pic[i]*n_pic[j])/total_collisions);
                    ++k;
                }
            }
            }

            // get the volume of the cell
            auto cell_volume = cell->measure();

            const double weight = ms[0].get_weight();

            // compute number of collisions
            double N_coll_max_double = 
                0.5*N_pic_tot*(N_pic_tot-1)*weight*sigma_cr_max*dt / cell_volume;
            // double N_coll_max_double_ = 0.5*N*number_density*weight*sigma_cr_max*dt;
            unsigned int N_coll_max = static_cast<unsigned int>(std::floor(N_coll_max_double+0.5));
            unsigned int N_pair_tested = static_cast<unsigned int>(std::floor(N_pic_tot/2.0));
            if (N_pair_tested > N_coll_max) {
                N_pair_tested = N_coll_max;
            }

            #ifdef EXTRA_OUTPUTS
            out_n_coll.write_row({
                static_cast<double>(N_coll_max), 
                static_cast<double>(N_pair_tested), 
                static_cast<double>(N_pic_tot)});
            #endif
        
            // double F_correction;
            if (N_coll_max==0 && N_pair_tested==0){
                continue;
            }
            double F_correction = N_coll_max/N_pair_tested;

            for (unsigned int ind1 = 0; ind1<ms.get_n_species(); ++ind1) {
                if (n_pic[ind1] == 0){
                    continue;
                }
                auto pic1 = particle_handlers[ind1].particles_in_cell(cell);
                // get the begin iterator of the particles in the cell
                const auto & pic_begin1 = pic1.begin();  
                
            for (unsigned int ind2=ind1; ind2<ms.get_n_species(); ++ind2){
                if (n_pic[ind2] == 0){
                    continue;
                }
                // combination index
                unsigned int ind_comb = ms.combination_index(ind1, ind2);

                // get the particles in the cell
                auto pic2 = particle_handlers[ind2].particles_in_cell(cell);

                // get the begin iterator of the particles in the cell
                const auto & pic_begin2 = pic2.begin(); 

                // get masses
                const double mass1 = ms[ind1].get_mass();
                const double mass2 = ms[ind2].get_mass();

                unsigned int N_pair_tested_comb = 
                    std::floor(N_pair_tested*coll_comb_frac[ind_comb] + 0.5);  
                
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
                    velocity_type cm((mass1*v_i + mass2*v_j) / (mass1+mass2));

                    // compute cross section
                    double sigma = collision_kernel.compute_cross_section(ind_comb);
            
                    // compute collision probability
                    double P_coll = (F_correction * sigma * cr_norm )/ sigma_cr_max;

                    #ifdef EXTRA_OUTPUTS
                    out_iterations.write_row({P_coll, F_correction, 
                        sigma*cr_norm, sigma*cr_norm/sigma_cr_max});
                    #endif

                    // pick random double [0, 1)
                    double U = Random::rnd_uniform();

                    if (U < P_coll)
                    {
                        // compute post collisional relative velocity
                        velocity_type cr_post = 
                            compute_post_collisional_velocity(cr);

                        // compute and update new velocities
                        update_velocities(it_i, it_j, cm, cr_post);

                        ++k;

                        if (sigma * cr_norm > sigma_cr_max)
                        {
                            sigma_cr_max = sigma * cr_norm;
                        }
                    }
                }
            }}
        }
    } // end apply_collisions

    template <int dim, class T>
    Tensor<1, dim> Collisions<dim, T>::compute_post_collisional_velocity(
        const velocity_type & cr) const
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

        template <int dim>
    inline void HardSphere<dim>::set_diameter(const species_type &ms)
    {
        // unsigned int L = ms.get_n_species();
        unsigned int M = ms.n_combinations;

        d12.reserve(M);

        for(unsigned int i = 0; i<M; ++i){
            double mr = ms.get_reduced_mass(i);
            double d2 = 5.0/16.0 * std::sqrt(BOLTZMANN_K*mr*temp_ref/
                (M_PI))/viscosity;
            d12.push_back(std::sqrt(d2));
        }

        LOG("\n-------- set ref diameters ----------\n");
        LOGVEC("d_ref", d12, M);

        LOG("\n-------- diameters species ----------\n"
         << "d_ref =  " << ms[0].get_diameter());
        for (unsigned int i = 1; i<ms.get_n_species(); ++i){
            LOG("          " << ms[i].get_diameter());
        }

        LOG("\n-------- free path species ----------\n"
        << "free path =  " << ms[0].get_free_path());
        for (unsigned int i = 1; i<ms.get_n_species(); ++i){
            LOG("              " << ms[i].get_free_path());
        }

        LOG("\n-------- dt species ----------\n"
        << "dt =  " << ms[0].get_free_path()/ms[0].get_max_velocity());
        for (unsigned int i = 1; i<ms.get_n_species(); ++i){
            LOG("      " << ms[i].get_free_path()/ms[0].get_max_velocity());
        }

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
