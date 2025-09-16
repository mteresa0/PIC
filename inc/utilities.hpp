/**
 * @file utilities.hpp
 * @author Maria Teresa Cazzola
 * @brief 
 * @version 0.1
 * @date 2025-04-28
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef UTILITIES_PIC_HPP
#define UTILITIES_PIC_HPP

#include <deal.II/base/point.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/particle_iterator.h>
#include <deal.II/particles/particle_accessor.h>


/// Boltzmann constant in J/K
static const double BOLTZMANN_K = 1.380649e-23;  // J/K
/// Temperature in K
static const double TEMPERATURE = 100;         // K
/// Electron mass in kg
static const double ELECTR_MASS = 1.9093837e-31; // kg

/*static const double DEV_VELOCITY = std::sqrt(
    BOLTZMANN_K * TEMPERATURE / ELECTR_MASS); // m/s

static const double MEAN_VELOCITY = 2*DEV_VELOCITY / std::sqrt(2.0/M_PI); // m/s
*/

namespace PICsimulator {
    using namespace dealii;

    namespace Species {

        class ChemicalSpecies {
        protected: 
            /// @brief ChemicalSpecies constructor
            /// @param 'c_real_mass' mass in (kg) of one particle
            /// @param 'c_charge' in (C)
            /// @param 'c_diameter' in (m)
            /// @param 'c_temperature' in (K)
            /// @param 'c_pressure' in (Pa)
            /// @param 'c_weight' [unsigned int] computatational weight (how many real particles are represented by one computational particle)
            ChemicalSpecies(const double & c_real_mass,
                      const double & c_charge,
                      const double & c_diameter,
                      const double & c_temperature,
                      const double & c_pressure,
                      const unsigned int & c_weight):
                    real_mass(c_real_mass), 
                    charge(c_charge),
                    diameter(c_diameter),
                    temperature(c_temperature),
                    pressure(c_pressure),
                    weight(c_weight),
                    computed_mass(c_weight*c_real_mass) {};

        public:
            /// @brief ChemicalSpecies destructor virtual
            // virtual ~ChemicalSpecies() = 0;0
            ~ChemicalSpecies() = default;

            /// @brief get computed mass
            double get_mass() const {
                return computed_mass;
            }

            /// @brief get computational weight
            unsigned int get_weight() const {
                return weight;
            }

            /// @brief get diameter
            double get_diameter() const {
                return diameter;
            }

            /// @brief get std velocity according to (?) #todo
            double get_std_velocity() const {
                return BOLTZMANN_K * temperature / real_mass;
            }

            /// @brief get mean velocity according to (?) #todo
            double get_mean_velocity() const {
                return 2 * get_std_velocity() / std::sqrt(2.0 / M_PI);
            }

            /// @brief get free path length
            double get_free_path() const {
                return (BOLTZMANN_K * temperature) / (std::sqrt(2) * M_PI * diameter * diameter * pressure);
            }

        private:
            const double real_mass; // kg
            const double charge; // C
            const double diameter; // m
            const double temperature; // K
            const double pressure; // Pa
            const unsigned int weight;
            const double computed_mass; // kg
        };

        // ChemicalSpecies::~ChemicalSpecies() {};

        class N2 : public ChemicalSpecies {
        public:
            N2() : ChemicalSpecies(
                /*real mass*/ 4.26e-26, /*kg*/
                /*charge*/ 0.0,
                /*diameter*/ 3.0e-10,
                /*temperature*/ 300.0,
                /*pressure*/ 1.0e5,
                /*weight*/ 30) {};
            
            ~N2() = default;
            N2(const N2 &) = default;
            N2(N2 &&) = delete;
            N2 & operator=(const N2 &) = delete;
            N2 & operator=(N2 &&) = delete;

        /*
        private:
            const double real_mass = 4.65e-26; // kg
            const double charge = 0.0; // C
            const double diameter = 3.0e-10; // m
            const double temperature = 300; // K
            const double pressure = 1e5; // Pa
            const unsigned int weight = 30;
            const double computed_mass = real_mass*weight; // kg
        */
        };
    }

    
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

    namespace Random {

        /// @brief random device
        std::random_device rd;
        /// @brief random number generator
        std::mt19937 gen(rd());
        /// @brief uniform distribution: [0, 1)
        std::uniform_real_distribution<double> uniform_distribution{0., 1.};

        double RND(){
            return uniform_distribution(gen);
        };
    }

} // namespace PICsimulator

#endif // UTILITIES_PIC_HPP