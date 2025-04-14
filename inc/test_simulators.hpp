/**
 * @file test_simulators.hpp
 * @author Maria Teresa Cazzola
 * @brief declaration and definition of tests cases for the PIC simulator
 * @date 2025-04-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef TEST_SIMULATORS_HPP
#define TEST_SIMULATORS_HPP

#include "simulator.hpp"
#include "domain.hpp"
#include <cmath>

namespace TestSimulators
{
    using namespace dealii;

    /// @brief Test simulator base class. 
    /// @brief Test different scenario described in the derived classes.
    /// @tparam dim dimension of the domain
    template<int dim>
    class TestSimulator : public PICsimulator::Simulator<dim>
    {
    public:
        using typename PICsimulator::Simulator<dim>::location_type;
        using typename PICsimulator::Simulator<dim>::velocity_type;

        /// @brief Constructor
        /// @param domain_data_ object describing the domain
        /// @param dt_ timestep time
        /// @param T_ total simulation time
        TestSimulator(
            const Domain::DomainData<dim> &domain_data_, 
            const double &dt_,
            const double &T_):
            PICsimulator::Simulator<dim>(domain_data_, dt_, T_) {};

        /// @brief Destructor
        ~TestSimulator() = default;
        
        /// @brief run test
        void run_test();

        /// @brief print errors to file
        /// @param filename name of the file
        /// @param overwrite file
        void print_errors_to_file(
            const std::string &filename, 
            const bool &overwrite = false)
        {
            std::ofstream file;
            if (overwrite) {
                file.open(filename);
                file << "location_error velocity_error dt\n";
            } 
            else
                file.open(filename, std::ios::app);

            file << this->location_error << " " << this->velocity_error << " " 
                << this->dt << "\n";
            file.close();
        };

        double get_loc_error() const {
            return location_error;
        };
        double get_vel_error() const {
            return velocity_error;
        };

    protected:
        /// @brief initialize grid and particle handler
        void initialize() override;

        /// @brief initialize particle
        virtual void initialize_particles() = 0;

        /// @brief initialize grid
        virtual void initialize_grid() = 0;

        /// @brief updates position and velocity errors and 
        /// updates true location and position
        /// @param particle particle iterator
        /// @param loc_ref current true location
        /// @param vel_ref current true velocity
        /// @param loc_error error location
        /// @param vel_error error velocity
        virtual void compute_errors(
            typename Particles::ParticleIterator<dim> &particle,
            location_type & loc_ref, 
            velocity_type & vel_ref, 
            double & loc_error, 
            double & vel_error) = 0;

        /// @brief compute L2 norm 
        /// @param v velocity_type vector (dealii::Tensor<1, dim>)
        /// @return L2 norm of the input Tensor<1, dim> v
        double norm(const velocity_type & v){
            double r = 0;
            for (int d = 0; d<dim; ++d){
                r += v[d]*v[d];
            }
            return std::sqrt(r);
        };

        /// @brief compute L2 norm 
        /// @param l location_type vector (dealii::Point<dim>)
        /// @return L2 norm of the input Point<dim> l
        double norm(const location_type & l){
            double r = 0;
            for (int d = 0; d<dim; ++d){
                r += l[d]*l[d];
            }
            return std::sqrt(r);
        };

        double location_error = 0; /// location error
        double velocity_error = 0; /// velocity error
    };

    /// @brief Testing the linear motion of a particle
    /// @tparam dim dimension of the domain
    template<int dim>
    class TestConstantVelocity : public TestSimulator<dim>
    {
    public:
        using typename TestSimulator<dim>::location_type;
        using typename TestSimulator<dim>::velocity_type;

        /// @brief Constructor of TestConstantVelocity
        /// @param domain_data_ object containing domain data
        /// @param dt_ timestep time
        /// @param T_ total simulation time
        /// @param vel_ particle velocity
        TestConstantVelocity(
            const Domain::DomainData<dim> &domain_data_, 
            const double &dt_,
            const double &T_, 
            const velocity_type &vel_):
            TestSimulator<dim>(domain_data_, dt_, T_), 
            velocity_ref(vel_) 
            {};

    private:
        /// @brief initialize one particle randomly chosing the location 
        /// in the domain and setting the initial velocity as the given one
        void initialize_particles() override;

        /// @brief initialize grid with the standard `make_grid`
        void initialize_grid() override{
            this->make_grid(); return; 
        };

        /// @brief updates position and velocity errors and 
        /// updates true location and position
        /// @param particle particle iterator
        /// @param loc_ref current true location
        /// @param vel_ref current true velocity
        /// @param loc_error error location
        /// @param vel_error error velocity
        void compute_errors(
            typename Particles::ParticleIterator<dim> &particle,
            location_type & loc_ref, 
            velocity_type & vel_ref, 
            double & loc_error, 
            double & vel_error) override;

        const velocity_type velocity_ref; /// velocity of the particle
    };

    /// @brief Testing uniformly accelerated motion of a particle
    /// @tparam dim spatial dimension of the domain
    template<int dim>
    class TestConstantAcceleration : public TestSimulator<dim>
    {
    public:
        using typename TestSimulator<dim>::location_type;
        using typename TestSimulator<dim>::velocity_type;

        /// @brief Constructor of TestConstantAcceleration
        /// @param domain_data_ object describing the domain
        /// @param dt_ timestep time
        /// @param T_ total simulation time
        /// @param acc_ given constant acceleration
        /// @param vel_in_ initial velocity of the particle
        TestConstantAcceleration(
            const Domain::DomainData<dim> &domain_data_, 
            const double &dt_,
            const double &T_, 
            const location_type &acc_,
            const velocity_type &vel_in_):
            TestSimulator<dim>(domain_data_, dt_, T_), 
            acceleration_ref(acc_),
            initial_velocity(vel_in_)
            {};
    protected:
        /// @brief updates position and velocity errors and 
        /// updates true location and position
        /// @param particle particle iterator
        /// @param loc_ref current true location
        /// @param vel_ref current true velocity
        /// @param loc_error error location
        /// @param vel_error error velocity
        void compute_errors(
            typename Particles::ParticleIterator<dim> &particle,
            location_type & loc_ref, 
            velocity_type & vel_ref, 
            double & loc_error, 
            double & vel_error) override;

        
        /// @brief override evaluate_acceleration with returning constant acceleration
        /// @return constant acceleration `acceleration_ref`
        location_type evaluate_acceleration() override{
            return acceleration_ref;
        };

        const location_type acceleration_ref; /// given constant acceleration
        const velocity_type initial_velocity; /// initial velocity of the particle

    private:
        /// @brief initialize one particle randomly chosing the location 
        /// in the domain and setting the initial velocity as the given one
        void initialize_particles() override;

        /// @brief initialize grid with the standard `make_grid`
        void initialize_grid() override{
            this->make_grid(); return; 
        };

    };


    /// @brief Testing 
    /// @tparam dim spatial dimension of the domain
    template<int dim>
    class TestLinearAcceleration : public TestConstantAcceleration<dim>
    {
    public:
        using typename TestConstantAcceleration<dim>::location_type;
        using typename TestConstantAcceleration<dim>::velocity_type;

        /// @brief Constructor of TestConstantAcceleration
        /// @param domain_data_ object describing the domain
        /// @param dt_ timestep time
        /// @param T_ total simulation time
        /// @param acc_in_ initial acceleration
        /// @param vel_in__ initial velocity of the particle
        /// @param dacc_ acceleration slope
        TestLinearAcceleration(
            const Domain::DomainData<dim> &domain_data_, 
            const double &dt_,
            const double &T_, 
            const location_type &acc_in_,
            const location_type &dacc_,
            const velocity_type &vel_in__):
            TestConstantAcceleration<dim>(domain_data_, dt_, T_, acc_in_, vel_in__), 
            slope_acc(dacc_)
            {};
    
    private:
        /// @brief updates position and velocity errors and 
        /// updates true location and position
        /// @param particle particle iterator
        /// @param loc_ref current true location
        /// @param vel_ref current true velocity
        /// @param loc_error error location
        /// @param vel_error error velocity
        void compute_errors(
            typename Particles::ParticleIterator<dim> &particle,
            location_type & loc_ref, 
            velocity_type & vel_ref, 
            double & loc_error, 
            double & vel_error) override;

        /// @brief override evaluate_acceleration with returning constant acceleration
        /// @return constant acceleration `acceleration_ref`
        location_type evaluate_acceleration() override{
            // first call in position update (still using acc^n)
            if(count%2==0) {
                ++count;
            } else {
            // second call in velocity_update when acc^{n+1} is needed
                ++count;
                current_time += this->dt;
            }
            return this->current_time * slope_acc + this->acceleration_ref;
        };

        unsigned int count = 0;
        const location_type slope_acc;
        double current_time = 0;               /// current time of the test simulation
    };

    /****************************
            DEFINITIONS   
    ****************************/

    /*
     * TestSimulator Definitions
     */

    template<int dim>
    void TestSimulator<dim>::initialize()
    {
        if (this->initialized)
        {
            std::cout << "Already initialized with:" << std::endl;
            std::cout << "n# particles: " << this->particle_handler.n_global_particles() << std::endl;
            // std::cout << "n# cells: "     << this->particle_handler.n_global_cells() << std::endl;
            return;
        }

        // initialize grid and particles
        initialize_grid();
        initialize_particles();

        std::cout << "\n------- TEST SIMULATOR INITIALIZED -------";
        std::cout << "\ndt: " << this->dt << "\nT: " << this->T;
        std::cout << "\n# time steps: " << this->T/this->dt;
        std::cout << "\n# cells: " << this->triangulation.n_active_cells();
        std::cout << "\n# particles: " << this->particle_handler.n_global_particles() << std::endl;
        std::cout << "------------------------------------------\n";
        this->initialized = true;
        return;
    }

    template<int dim>
    void TestSimulator<dim>::run_test()
    {
        // initialize the grid and the particle_handler
        this->initialize();

        // variables storing the errors
        double error_vel(0.), error_pos(0.);

        // get particle iterator and starting data
        auto it = this->particle_handler.begin();
        location_type ref_location(it->get_location());
        velocity_type ref_velocity(this->get_velocity_from_properties(it));

        // make sure to reset DT variable
        REINIT_CUMULATIVE_CHRONO();

        unsigned int NT = this->T/this->dt;
        for (unsigned int i = 0; i<NT; ++i)
        {
            // one-step iteration update using the implemented algorithm (verlet)
            // also timing it
            START_CUMULATIVE_CHRONO();
            this->update_simulation_state();
            ADD_TIME_CUMULATIVE_CHRONO();

            // particle handler reinitialization is skipped because we don't 
            // need it to have the particle reassigned to the correct cell

            // computing the error:
            compute_errors(it, ref_location, ref_velocity, error_pos, error_vel);
        }

        error_vel *= this->dt;
        error_pos *= this->dt;

        // printing the errors and the elipsed time of the simulation
        std::cout << "\ntotal velocity error: " << error_vel << std::endl;
        std::cout << "total position error: " << error_pos << std::endl;
        std::cout << "streaming time:       " << DT_CUMULATIVE_CHRONO << "\n\n"<< std::endl;
    
        location_error = error_pos;
        velocity_error = error_vel;
    }

    /*
    * Constant Velocity Test Definitions
    */

    template<int dim>
    void TestConstantVelocity<dim>::initialize_particles()
    {
        // number of particles
        unsigned int N = 1;

        // random
        std::random_device rd_pos;
        std::mt19937 gen_pos;
        std::uniform_real_distribution<double> pick_position(0, 1);
        
        // get particle location

        std::vector<location_type> positions(N);

        auto lower_vertex(this->domain_data.get_lower_vertex());
        auto difference(this->domain_data.get_upper_vertex() - lower_vertex);

        for(auto & pos : positions) {
            for(unsigned int d = 0; d<dim; ++d){
                pos[d] = lower_vertex[d] + 
                pick_position(gen_pos)*(difference[d]);
            }
        } 

        // insert particle

        this->particle_handler.insert_particles(positions);

        // assigned same constant velocity to each particle (there should be only one particle)

        for(auto it = this->particle_handler.begin();
            it!= this->particle_handler.end(); ++it){
                this->set_velocity(it, velocity_ref);
            }
        
        return;
    }

    template<int dim>
    inline void TestConstantVelocity<dim>::compute_errors(
        typename Particles::ParticleIterator<dim> &particle,
        location_type & loc_ref, 
        velocity_type & , 
        double & loc_error, 
        double & vel_error)
    {
        // get new position from simulation
        location_type loc = (particle->get_location());

        // update location from theory
        loc_ref = loc_ref + 
                velocity_ref * this->dt;

        // compute position error
        loc_error += this->norm(loc - loc_ref);

        // get new velocity from simulation
        velocity_type vel = (this->get_velocity_from_properties(particle));

        // compute velocity error
        vel_error += this->norm(vel - this->velocity_ref);
    }



    /*
    * Constant Acceleration Test Definitions
    */

    template <int dim>
    void TestConstantAcceleration<dim>::initialize_particles()
    {
        // number of particles
        unsigned int N = 1;

        // random
        std::random_device rd_pos;
        std::mt19937 gen_pos;
        std::uniform_real_distribution<double> pick_position(0, 1);
        
        // get particle location

        std::vector<location_type> positions(N);

        auto lower_vertex(this->domain_data.get_lower_vertex());
        auto difference(this->domain_data.get_upper_vertex() - lower_vertex);

        for(auto & pos : positions) {
            for(unsigned int d = 0; d<dim; ++d){
                pos[d] = lower_vertex[d] + 
                pick_position(gen_pos)*(difference[d]);
            }
        } 

        // insert particle

        this->particle_handler.insert_particles(positions);

        // assigned same constant velocity to each particle (there should be only one particle)

        for(auto it = this->particle_handler.begin();
            it!= this->particle_handler.end(); ++it){
                this->set_velocity(it, this->initial_velocity);
            }
        
        return;
    }

    template<int dim>
    inline void TestConstantAcceleration<dim>::compute_errors(
        typename Particles::ParticleIterator<dim> &particle,
        location_type & loc_ref, 
        velocity_type & vel_ref, 
        double & loc_error, 
        double & vel_error)
    {
        // get new position from simulation
        location_type loc = (particle->get_location());

        // update location from theory
        loc_ref = loc_ref + 
                vel_ref * this->dt + 
                0.5*this->acceleration_ref * this->dt * this->dt;

        // compute position error
        loc_error += this->norm(loc - loc_ref);

        // get new velocity from simulation
        velocity_type vel = (this->get_velocity_from_properties(particle));

        // compute new velocity from theory and update
        vel_ref = (vel_ref + this->acceleration_ref * this->dt);

        // compute velocity error
        vel_error += this->norm(vel - vel_ref);
    }

    template <int dim>
    inline void TestLinearAcceleration<dim>::compute_errors(
        typename Particles::ParticleIterator<dim> &particle, 
        location_type &loc_ref, 
        velocity_type &vel_ref, 
        double &loc_error, 
        double &vel_error)
    {
        location_type current_acceleration = this->acceleration_ref + 
                                            (current_time - this->dt) * slope_acc;

        // get new position from simulation
        location_type loc = (particle->get_location());

        // get new velocity from simulation
        velocity_type vel = (this->get_velocity_from_properties(particle));

        // update location from theory
        loc_ref = loc_ref + 
                vel_ref * this->dt + 
                0.5*current_acceleration * this->dt * this->dt +
                1./6. * slope_acc * this->dt * this->dt * this->dt;

        // compute position error
        loc_error += this->norm(loc - loc_ref);

        // compute new velocity from theory and update
        vel_ref = (vel_ref + 
                current_acceleration * this->dt + 
                0.5 * this->slope_acc * this->dt * this->dt
                );

        // compute velocity error
        vel_error += this->norm(vel - vel_ref);

        // std::cout << "\nit: " << current_time << "  vel_error    pos_error" << 
        // std::endl  << "          " << this->norm(vel - vel_ref) << 
        // "        " << this->norm(loc - loc_ref)  << std::endl;

        // std::cout << "sim location: " << loc << std::endl;
        // std::cout << "ref location: " << loc_ref << std::endl;
        // std::cout << "sim velocity: " << vel  << std::endl;
        // std::cout << "ref velocity: " << vel_ref << std::endl;
        // std::cout << "acceleration: " << current_acceleration << std::endl;

    }

} // namespace TestSimulators

#endif // TEST_SIMULATORS_HPP