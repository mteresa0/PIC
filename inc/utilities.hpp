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
#include <random>
#include <fstream>

// LOGGING OPTION for printing simulation info
#if LOG_ENABLED
    #define LOG(x) do { std::cout  << " " << x << std::endl; } while (0)    
    #define LOGVAR(name, value) do { \
        std::cout << std::left << std::setw(25) << (" " +std::string(name) + ":") \
                  << std::right << std::setw(15) << value << std::endl; \
    } while(0)
    #define LOGVEC(name, vector, dim) do { \
        std::cout << std::left << std::setw(25-2) << (" " + std::string(name) + ":")\
        << "[ " << std::right << std::setw(15)  << vector[0];\
        for(unsigned int i = 1; i<dim; ++i){ \
            std::cout << ", \n" << std::setw(25) <<" "<< std::right<<  std::setw(15) << vector[i];\
        } \
        std::cout << " ]"<<std::endl;\
    } while(0)

    inline std::string make_centered_title(
        const std::string &title, 
        unsigned int width = 40, 
        char fill = '=')
    {
        if (title.size() >= width)
            return title;  // too long, just return as-is

        int padding = (width - title.size()) / 2;
        return std::string(padding, fill) + " " + title + " " + 
            std::string(width - title.size() - padding, fill);
    }
    #define LOGTITLE(x) do {std::cout << "\n" << make_centered_title(x) \
        << std::endl;} while(0)
#else
    #define LOG(x) do {} while (0)
    #define LOGVAR(x, y) do {} while(0)
    #define LOGVEC(x, y, z) do {} while(0)
    #define LOGTITLE(x) do {} while(0)
#endif

// EXTRA OUTPUT OPTION to get detailed info of the simulation; this is important 
// if you would like to check if the simulation make sense. For example, you 
// would like to have P_coll (collision probability) that do not exceed 20%, 
// according to Boyd (chapter 6)
#ifdef EXTRA_OUTPUTS
    class CSVoutput{
    public:
        // filename: CSV file path
        // headers: column names
        // buffer_size: number of rows to keep in memory before writing to disk
        CSVoutput(const std::string &filename, 
                        const std::vector<std::string> &headers,
                        std::size_t buffer_size = 1000)
            : file(filename), buffer_size(buffer_size)
        {
            if(file.is_open()){
                for(std::size_t i = 0; i < headers.size(); ++i){
                    file << headers[i];
                    if(i != headers.size() - 1) file << ",";
                }
                file << "\n";
            }
        }

        ~CSVoutput() {
            flush(); // write any remaining rows
            if(file.is_open()) file.close();
        }

        // write a row of data
        void write_row(const std::vector<double> &row) {
            buffer.push_back(row);
            if(buffer.size() >= buffer_size){
                flush();
            }
        }

            // flush the buffer to the file
        void flush() {
            for(const auto &row : buffer){
                for(std::size_t i = 0; i < row.size(); ++i){
                    file << row[i];
                    if(i != row.size()-1) file << ",";
                }
                file << "\n";
            }
            buffer.clear();
            file.flush(); // force flush to disk
        }

    private:
        std::ofstream file;
        std::vector<std::vector<double>> buffer; // buffer for rows
        std::size_t buffer_size;

    };

#endif

/// Boltzmann constant in J/K
static const double BOLTZMANN_K = 1.380649e-23;  // J/K
/// Temperature in K
static const double TEMPERATURE = 100;         // K
/// Electron mass in kg
static const double ELECTR_MASS = 1.9093837e-31; // kg

namespace Timers {

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
                throw std::logic_error("Call tic() before get_dt()");
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
            LOG(dt << " seconds\n");
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

        /// @brief uniform distribution: [0, 1)
        std::normal_distribution<double> normal_distribution{0., 1.};

        /// @brief returns a random double drawn for a uniform distribution 
        /// from 0 to 1.
        double rnd_uniform(){
            return uniform_distribution(gen);
        };

        /// @brief returns a random double drawn for a uniform distribution 
        /// from a to b
        /// @param a left extreme
        /// @param b right extreme
        double rnd_uniform(const double& a, const double& b) {
            return rnd_uniform()*(b-a) + a;
        }

        /// @brief returns a random double drawn from a normal distibution 
        /// with 0 mean and std of 1.
        double rnd_gaussian(){
            return normal_distribution(gen);
        }

        /// @brief returns a random double drawn from a normal distibution 
        /// with mu mean and std of sigma.
        /// @param mu mean
        /// @param sigma standard deviation
        double rnd_gaussian(const double & mu, const double & sigma){
            return rnd_gaussian()*sigma + mu;
        }


    }

} // namespace PICsimulator

#endif // UTILITIES_PIC_HPP