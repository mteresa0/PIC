#ifndef SPECIES_PIC_HPP
#define SPECIES_PIC_HPP

#include "utilities.hpp"

namespace PICsimulator::Species {

    class ChemicalSpecies {
    protected: 
        /// @brief ChemicalSpecies constructor
        /// @param 'c_name' name specie
        /// @param 'c_real_mass' mass in (kg) of one particle
        /// @param 'c_charge' in (C)
        /// @param 'c_diameter' in (m)
        /// @param 'c_temperature' in (K)
        /// @param 'c_pressure' in (Pa)
        /// @param 'c_weight' [unsigned int] computatational weight 
        /// (how many real particles are represented by one comp particle)
        ChemicalSpecies(const std::string& c_name,
                    const double & c_real_mass,
                    const double & c_charge,
                    const double & c_diameter,
                    const double & c_temperature,
                    const double & c_pressure,
                    const unsigned int & c_weight):
                name(c_name),
                real_mass(c_real_mass), 
                charge(c_charge),
                diameter(c_diameter),
                temperature(c_temperature),
                pressure(c_pressure),
                weight(c_weight),
                computed_mass(c_weight*c_real_mass) {};

    public:
        /// @brief ChemicalSpecies destructor 
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

        /// @brief get std velocity according to (?) TODO
        double get_std_velocity() const {
            return BOLTZMANN_K * temperature / real_mass;
        }

        /// @brief get mean velocity according to (?) 
        // TODO: get reference
        double get_mean_velocity() const {
            return 2 * get_std_velocity() / std::sqrt(2.0 / M_PI);
        }
        
        /// @brief get free path length
        double get_free_path() const {
            return (BOLTZMANN_K * temperature) / (std::sqrt(2) * M_PI * diameter * diameter * pressure);
        }

        /// @brief get specie name
        const std::string get_name() const {return name;}

    private:
        const std::string name;
        const double real_mass; // kg
        const double charge; // C
        const double diameter; // m
        const double temperature; // K
        const double pressure; // Pa
        const unsigned int weight;
        const double computed_mass; // kg
    };

    class N2 : public ChemicalSpecies {
    public:
        N2() : ChemicalSpecies(
            /*name*/ "N2",
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

    };

    class O2 : public ChemicalSpecies {
    public:
        O2() : ChemicalSpecies(
            /*name*/ "O2",
            /*real mass*/ 5.31339290530668e-26, /*kg*/
            /*charge*/ 0.0, /*C*/
            /*diameter*/ 3.0e-10, /*m*/
            /*temperature*/ 300.0, /*K*/
            /*pressure*/ 1.0e5, /*Pa*/
            /*weight*/ 30) {};
        
        ~O2() = default;
        O2(const O2 &) = default;
        O2(O2 &&) = delete;
        O2 & operator=(const O2 &) = delete;
        O2 & operator=(O2 &&) = delete;
    };


    class MultiSpecie {
    public:
        MultiSpecie(
            const std::vector<ChemicalSpecies> & sp_vec_,
            const std::vector<unsigned int> n_particles_
        ):  species_vec(sp_vec_),
            n_particles(n_particles_),
            n_species(species_vec.size())
        {
            std::cout << n_species << "   " << n_particles.size() << std::endl;
            if (n_species!=n_particles.size()){
                std::string error_msg = 
                    "Invalid argument: The 'species' vector size (" +
                    std::to_string(n_species) + 
                    ") must match the 'n_particle' vector size (" + 
                    std::to_string(n_particles.size()) + ").";
                                            
                    throw std::invalid_argument(error_msg);
            }

        };
    
    unsigned int get_n_particles(const unsigned int i ) const {
        return n_particles[i];
    };

    unsigned int get_n_species() const {return n_species;};

    const ChemicalSpecies& get_specie(const unsigned int i ) const {
        return species_vec[i];
    }

    const std::vector<ChemicalSpecies>& get_species_vector() const {
        return species_vec;
    };

    private:
        const std::vector<ChemicalSpecies> species_vec;
        const std::vector<unsigned int> n_particles;

        const unsigned int n_species;

    };
}


#endif // SPECIES_PIC_HPP
