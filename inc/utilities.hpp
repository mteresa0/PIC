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
#include <chrono>
#include <iomanip> 

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
namespace Timers {

    // static std::chrono::time_point<std::chrono::high_resolution_clock> STARTING_TIME;
    // static std::chrono::time_point<std::chrono::high_resolution_clock> STARTING_TIME_CUMULATIVE_CHRONO;
    // static double DT, DT_CUMULATIVE_CHRONO;

    /*
    // #define TIC() STARTING_TIME = std::chrono::high_resolution_clock::now();
    // #define TOC(){                                                                  \
    //     auto now = std::chrono::high_resolution_clock::now();                       \
    //     std::chrono::duration<double> diff = now - STARTING_TIME;                   \
    //     DT = diff.count();                                                          \
    // }

    // #define START_CUMULATIVE_CHRONO() {STARTING_TIME_CUMULATIVE_CHRONO = std::chrono::high_resolution_clock::now();}
    // #define REINIT_CUMULATIVE_CHRONO() {DT_CUMULATIVE_CHRONO = 0;}
    // #define ADD_TIME_CUMULATIVE_CHRONO() {                                          \
    //     auto now = std::chrono::high_resolution_clock::now();                       \
    //     std::chrono::duration<double> diff = now - STARTING_TIME_CUMULATIVE_CHRONO; \
    //     DT_CUMULATIVE_CHRONO += diff.count();                                       \
    // }
    */

    /// @brief Classic Timer class.
    /// There are implemented the classic tic toc functions
    class Timer {
    public:
        /// @brief Timer constructor
        Timer(): dt(0.0), is_started(false) {};

        /// @brief starting timer
        /// @details call this function before the code snippet you want to time.
        /// `start` stores the time when `tic` have been run.
        void tic() {
            is_started = true;
            start = std::chrono::high_resolution_clock::now();
        }


        /// @brief returns the time difference from the last tic in seconds.
        /// @details place this function after the code snippet you want to time.
        /// toc returns the duration of time between the calls of `tic' and 'toc`.
        double toc() {
            auto now = std::chrono::high_resolution_clock::now();

            if(!is_started) {
                throw std::logic_error("Call toc() before get_dt()");
            }
            
            std::chrono::duration<double> diff = now - start;
            dt = diff.count();
            return dt;
        }

        /// @brief gets the last time difference measured.
        double get_dt() {
            return dt;
        }

        /// @brief print time elapsed
        void print() {
            std::cout << dt << " seconds\n";
        }

    private:
        /// stores the time in which tic has been called
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        /// stores the time difference (in seconds) between the calls of tic and toc
        double dt;

        /// bool for state timer
        bool is_started;

    };
 
    /// @brief Cumultative_Timer class. It can time the elasped time of a code snippet in a loop 
    /// @details for example, you may want to time the function 'function_to_time' but it is in a for loop:
    /// for(...) {
    ///    /* code */
    ///    function_to time();
    ///    /* code */
    /// }
    /// If you place tic and toc before and after 'function_to_time' you will the total elapsed time
    class Cumulative_Timer {

    public:
        Cumulative_Timer(): dt(0.0) {};

        /// @brief Initializes the time difference variable `dt` to 0. 
        /// if `tic` and `toc` are used in a loop, the timer has to be initialized before.
        void clean_timer() {
            dt = 0.0;
        }

        /// @brief starting timer
        /// @details call this function before the code snippet you want to time.
        /// `start` stores the time when `tic` have been run.
        void tic() {
            start = std::chrono::high_resolution_clock::now();
        }


        /// @brief returns the time difference from the last tic in seconds.
        /// @details place this function after the code snippet you want to time.
        /// toc returns the duration of time between the calls of `tic' and 'toc`.
        void toc() {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = now - start;
            dt += diff.count();
        }

        /// @brief returns the cumulative elapsed time of the timed code snippet
        double get_dt() {
            return dt;
        }

        /// @brief print dt
        void print() {
            std::cout << std::setprecision(6) << dt << " seconds\n";
        }

        /// @brief print dt and percentage wrt to total simulation time.
        void print(const double &sim_time) {
            std::cout << std::fixed;
            std::cout << std::setprecision(6) << dt << " seconds    ["
            << std::setprecision(3) << dt/sim_time*100 << "%]" << std::endl;
        }

    private:

        /// stores the time in which tic has been called
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        /// stores the time difference (in seconds) between the calls of tic and toc
        double dt;

    };

}


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