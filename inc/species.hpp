/**
 * @file species.hpp
 * @author Maria Teresa Cazzola 
 * @brief Defines the **ChemicalSpecies** and **MultiSpecies** classes for 
 * particle simulation. 
 * @details This header contains definitions for:
 * 1. The fundamental properties of a single particle type 
 * (@ref ChemicalSpecies).
 * 2. The container class for managing a complete mixture of species 
 * (@ref MultiSpecies).
 * 
 * @date 2025-10-30
 */

#ifndef SPECIES_PIC_HPP
#define SPECIES_PIC_HPP

#include "utilities.hpp"
#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace PICsimulator::Species {

    /// @brief base class to species
    class ChemicalSpecies {
    public: 

        /// @brief ChemicalSpecies constructor
        /// @param `c_name` name specie
        /// @param `c_real_mass` mass in (kg) of one particle
        /// @param `c_charge` in (C)
        /// @param `c_diameter` in (m)
        /// @param `c_temperature` in (K)
        /// @param `c_pressure` in (Pa)
        /// @param `c_weight` [unsigned int] computatational weight 
        /// (how many real particles are represented by one comp particle)
        ChemicalSpecies(const std::string& c_name,
                    const double & c_real_mass,
                    const double & c_charge,
                    const double & c_diameter,
                    const double & c_temperature,
                    const unsigned int & c_weight);

        ChemicalSpecies(const std::string& species_file,
                        std::string& name);
    
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

        /// @brief get scale parameter of the Maxwell distribution
        double get_a() const {
            return BOLTZMANN_K * temperature / real_mass;
        }

        /// @brief get std of the Maxwell distribution
        double get_std() const {
            return (std::sqrt(3-8/M_PI))*scale_parameter;
        }

        /// @brief get mean velocity of the Maxwell distribution
        double get_mean_velocity() const {
            return 2 * scale_parameter / std::sqrt(2.0 / M_PI);
        }
        
        /// @brief get free path length
        double get_free_path() const {
            return 1;
            // TODO: fix get_free_path (no more pressure in chemicalspecies)
            // return (BOLTZMANN_K * temperature) / (std::sqrt(2) * M_PI * 
            // diameter * diameter * pressure);
        }

        /// @brief get specie name
        const std::string get_name() const {return name;}

        /// @brief get the 3 std from mean of the Maxwell distribution as the 
        /// max velocity
        double get_max_velocity() const {
            return get_mean_velocity() + 3*get_std();
        }

    private:
        std::string name;
        double real_mass; // kg
        double charge; // C
        double diameter; // m
        double temperature; // K
        // const double pressure; // Pa
        unsigned int weight;
        double computed_mass; // kg
        /// @brief scale parameter of the Boltzmann-Maxwell distribution
        /// @details usually denoted as 'a'
        double scale_parameter;
    };

    // TODO: add constructor from json file
    /// @brief class containg all the info about the gas mixture. 
    class MultiSpecies {
    public:
        /// @brief constructor
        /// @param sp_vec_ vector of species
        /// @param vec_n_part_ number of particles of each species
        /// @param temp_ neutral temperature (K)
        /// @param p_tot_ total pressure (Pa)
        MultiSpecies(
            const std::vector<ChemicalSpecies> & sp_vec_,
            const std::vector<unsigned int> vec_n_part_,
            const double & temp_,
            const double & p_tot_);
            
        // /// @brief default constructor
        // MultiSpecies() : MultiSpecies({N2()}, {1000}, 300.0, 1e5) {};

        //// indexes functions
        /// @brief compute the combination index of two species
        /// @param i index of the first species
        /// @param j index of the second species
        /// @details the combined index is computed such that `i` is always greater 
        /// or equal to `j`
        unsigned int combination_index(unsigned int , unsigned int ) const;

        /// @brief compute the indexes of the species given the combination index
        /// @param k combination index of species `i` and `j`
        /// @param i index of the first species
        /// @param j index of the second species
        /// @details the combined index is computed such that `i` is always greater 
        /// or equal to `j`
        void species_indexes(
            const unsigned int & k, unsigned int & i, unsigned int & j) const;

    private:
        /// @brief vector of species
        const std::vector<ChemicalSpecies> species_vec_;
        /// @brief vecotr of particles numbers (corresponding to each specie)
        const std::vector<unsigned int> n_particles_;
        /// @brief number of species
        const unsigned int n_species_;    

        const unsigned int n_combinations_;  

        std::vector<double> reduced_mass_;
        /// neutrals temperature
        const double temperature_;
        /// total pressure
        const double p_total_;

    public:

        ////// getters //////

        /// @brief provides read-only array-like access to a specie
        /// @param i specie index
        const ChemicalSpecies& operator[](const unsigned int i) const {
            if (i >= species_vec_.size()) { 
                throw std::out_of_range("Index out of bounds for species_vec_");
            }
            return species_vec_[i];
        }
        
        /// @brief get the number of particles of the i-specie
        /// @param i 
        unsigned int n_particles(const unsigned int i ) const {  
            if (i >= n_particles_.size()) { 
                throw std::out_of_range("Index out of bounds for n_particles_");
            }
            return n_particles_[i];
        };

        /// @brief get the number of species in this gas mixture
        /// @return 
        unsigned int n_species() const {return n_species_;};

        /// @brief get the i-specie
        /// @param i
        const ChemicalSpecies& get_chem_species(const unsigned int i ) const {
            if (i >= species_vec_.size()) { 
                throw std::out_of_range("Index out of bounds for species_vec_");
            }
            return species_vec_[i];
        }

        /// @brief get the species vector
        /// @return reference of the species vector
        const std::vector<ChemicalSpecies>& species_vec() const {
            return species_vec_;
        };

        /// @brief get reduced mass of the i-combination
        const double & reduced_mass(const unsigned int & i) const {
            if (i >= n_combinations_) { 
                throw std::out_of_range(
                "Index out of bounds: i >= number of collisions combinations");
            }
            return reduced_mass_[i];
        };

        /// @brief get number of combinations
        const unsigned int & n_combinations() const {return n_combinations_;}
    };
    // TODO: add printing info method

    ChemicalSpecies::ChemicalSpecies(
        const std::string& c_name,
        const double & c_real_mass,
        const double & c_charge,
        const double & c_diameter,
        const double & c_temperature,
        const unsigned int & c_weight):
            name(c_name),
            real_mass(c_real_mass), 
            charge(c_charge),
            diameter(c_diameter),
            temperature(c_temperature),
            weight(c_weight),
            computed_mass(c_weight*c_real_mass),
            scale_parameter(BOLTZMANN_K*temperature/ real_mass) 
            {}

    inline ChemicalSpecies::ChemicalSpecies(
        const std::string & species_file, std::string & species_name)
    {
        // Load species.json
        std::ifstream fs(species_file);
        if (!fs)
            throw std::runtime_error("Cannot open species file: " + species_file);

        json data = json::parse(fs);

        // Search the species by name
        bool found(false);
        for (auto& s : data["species"]) {
            if (s["name"] == species_name) {
                found = true;
                name = std::move(species_name);
                real_mass = s["real_mass"];
                charge = s["charge"];
                diameter = s["diameter"];
                temperature = s["temperature"];
                weight = s["weight"];
                computed_mass = weight*real_mass;
                scale_parameter = BOLTZMANN_K*temperature/ real_mass;
            };
        }

        if(not(found)){
            throw std::runtime_error("Cannot find species " + species_name);
        }
    };

    MultiSpecies::MultiSpecies(
        const std::vector<ChemicalSpecies> & sp_vec_,
        const std::vector<unsigned int> vec_n_part_,
        const double & temp_=300.0,
        const double & p_tot_=1e5):
            species_vec_(sp_vec_),
            n_particles_(vec_n_part_),
            n_species_(species_vec_.size()),
            n_combinations_(int(0.5*n_species_*(n_species_+1))),
            reduced_mass_(n_combinations_),
            temperature_(temp_),
            p_total_(p_tot_)
    {
        if (n_species_!=n_particles_.size()){
            std::string error_msg = 
                "Invalid argument: The 'species' vector size (" +
                std::to_string(n_species_) + 
                ") must match the 'n_particle' vector size (" + 
                std::to_string(n_particles_.size()) + ").";
                                        
                throw std::invalid_argument(error_msg);
        }

        // initialize reduced_mass_ vector
        unsigned int k = 0;
        for (unsigned int i = 0; i<n_species_; ++i) {
            double m_i(species_vec_[i].get_mass());
            for (unsigned int j = i; j<n_species_; ++j){
                double m_j(species_vec_[j].get_mass());
                reduced_mass_[k] = m_i*m_j/(m_i+m_j);
                ++k;
            }
        }
    }

    inline unsigned int MultiSpecies::combination_index(
        unsigned int i,
        unsigned int j) const
    {
        if(i>j) {
            std::swap(i, j);
        }

        unsigned int k = j;
        for (unsigned int ii = 0; ii<i; ++ii)
        {
            k += n_species_-1-ii;
        }

        return k;
    };

    inline void MultiSpecies::species_indexes(
        const unsigned int & k, unsigned int & i, unsigned int & j) const
    {
        unsigned int k1 = k;
        unsigned int l1 = n_species_;

        for (i = 0 ; i<n_species_; ++i){
            
            if(k1>=l1){
                k1 -=l1;
                l1 -=1;
            } else  {
                j = k1 +i;
                break;
            }
        }
        return;
    };

}


#endif // SPECIES_PIC_HPP
