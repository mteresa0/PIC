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
    /// @tparam dim 
    template<int dim>
    class TestSimulator : public PICsimulator::Simulator<dim>
    {
    public:
        using typename PICsimulator::Simulator<dim>::location_type;
        using typename PICsimulator::Simulator<dim>::velocity_type;

        TestSimulator(
            const Domain::DomainData<dim> &domain_data_, 
            const double &dt_,
            const double &T_):
            PICsimulator::Simulator<dim>(domain_data_, dt_, T_) {};
        
        /// @brief run test
        virtual void run_test() = 0;

    protected:
        /// @brief initialize grid and particle handler
        void initialize() override;

        /// @brief initialize particle
        virtual void initialize_particles() = 0;

        /// @brief initialize grid
        virtual void initialize_grid() = 0;

        /// @brief compute L2 norm 
        /// @param v tensor
        /// @return L2 norm of the input Tensor<1, dim> v
        double norm(const velocity_type & v){
            double r = 0;
            for (int d = 0; d<dim; ++d){
                r += v[d]*v[d];
            }
            return std::sqrt(r);
        };

        /// @brief compute L2 norm 
        /// @param l 
        /// @return L2 norm of the input Point<dim> l
        double norm(const location_type & l){
            double r = 0;
            for (int d = 0; d<dim; ++d){
                r += l[d]*l[d];
            }
            return std::sqrt(r);
        };

    };

    template<int dim>
    class TestConstantVelocity : public TestSimulator<dim>
    {
    public:
        using typename TestSimulator<dim>::location_type;
        using typename TestSimulator<dim>::velocity_type;

        TestConstantVelocity(
            const Domain::DomainData<dim> &domain_data_, 
            const double &dt_,
            const double &T_, 
            const velocity_type &vel_):
            TestSimulator<dim>(domain_data_, dt_, T_), 
            velocity_ref(vel_) 
            {};

        void run_test() override;
    
    private:
        void initialize_particles() override;
        void initialize_grid() override{
            this->make_grid(); return; 
        };

        const velocity_type velocity_ref;
    };

    template<int dim>
    class TestConstantAcceleration : public TestSimulator<dim>
    {
    public:
        using typename TestSimulator<dim>::location_type;
        using typename TestSimulator<dim>::velocity_type;

        TestConstantAcceleration(
            const Domain::DomainData<dim> &domain_data_, 
            const double &dt_,
            const double &T_, 
            const location_type &acc_,
            const velocity_type &i_vel_):
            TestSimulator<dim>(domain_data_, dt_, T_), 
            acceleration_ref(acc_),
            initial_velocity(i_vel_)
            {};

        void run_test() override;
    
    private:
        void initialize_particles() override;
        void initialize_grid() override {
            this->make_grid();
            return;
        };

        location_type evaluate_acceleration() override{
            return acceleration_ref;
        };

        const location_type acceleration_ref;
        const velocity_type initial_velocity;
    };

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

        this->initialized = true;
        return;
    }

    /****************************
            DEFINITIONS   
    ****************************/

    template<int dim>
    void TestConstantVelocity<dim>::run_test()
    {
        /// initialize the grid and the particle_handler
        this->initialize();

        /// variables storing the errors
        double error_vel(0.), error_pos(0.);

        // get particle iterator and starting data
        auto it = this->particle_handler.begin();
        location_type prev_location(it->get_location());

        /// make sure to reset DT variable
        REINIT_CUMULATIVE_CHRONO();

        unsigned int NT = this->T/this->dt;
        for (unsigned int i = 0; i<NT; ++i)
        {
            /// one-step iteration update using the implemented algorithm (verlet)
            /// also timing it
            START_CUMULATIVE_CHRONO();
            this->update_simulation_state();
            ADD_TIME_CUMULATIVE_CHRONO();

            /// particle handler reinitialization is skipped because we don't 
            /// need it to have the particle reassigned to the correct cell

            /// computing the error:
            {
                velocity_type vel(this->get_velocity_from_properties(it));
                error_vel += this->norm(vel-velocity_ref);
                location_type loc(it->get_location());
                error_pos += this->norm(loc - (prev_location + velocity_ref*this->dt));

                // update the previous location
                prev_location = loc;
            }   
        }

        error_vel /= NT;
        error_pos /= NT;

        /// printing the errors and the elipsed time of the simulation
        std::cout << "total velocity error: " << error_vel << std::endl;
        std::cout << "total position error: " << error_pos << std::endl;
        std::cout << "streaming time:       " << DT_CUMULATIVE_CHRONO << std::endl;
    }

    template<int dim>
    void TestConstantVelocity<dim>::initialize_particles()
    {
        // number of particles
        unsigned int N = 1;

        // random
        std::random_device rd_pos;
        std::mt19937 gen_pos;
        std::uniform_real_distribution<double> pick_position(0, 1);
        
        /// get particle location

        std::vector<location_type> positions(N);

        auto lower_vertex(this->domain_data.get_lower_vertex());
        auto difference(this->domain_data.get_upper_vertex() - lower_vertex);

        for(auto & pos : positions) {
            for(unsigned int d = 0; d<dim; ++d){
                pos[d] = lower_vertex[d] + 
                pick_position(gen_pos)*(difference[d]);
            }
        } 

        /// insert particle

        this->particle_handler.insert_particles(positions);

        /// assigned same constant velocity to each particle (there should be only one particle)

        for(auto it = this->particle_handler.begin();
            it!= this->particle_handler.end(); ++it){
                this->set_velocity(it, velocity_ref);
            }
        
        return;
    }

    template <int dim>
    void TestConstantAcceleration<dim>::run_test()
    {
        /// initialize the grid and the particle_handler
        this->initialize();

        /// variables storing the errors
        double error_vel(0.), error_pos(0.);

        // get particle iterator and starting data
        auto it = this->particle_handler.begin();
        location_type loc_ref(it->get_location());
        velocity_type vel_ref(initial_velocity);
        location_type loc;
        velocity_type vel;

        /// make sure to reset DT variable
        REINIT_CUMULATIVE_CHRONO();

        unsigned int NT = this->T/this->dt;
        for (unsigned int i = 0; i<NT; ++i)
        {
            /// one-step iteration update using the implemented algorithm (verlet)
            /// also timing it
            START_CUMULATIVE_CHRONO();
            this->update_simulation_state();
            ADD_TIME_CUMULATIVE_CHRONO();

            /// particle handler reinitialization is skipped because we don't 
            /// need it to have the particle reassigned to the correct cell

            /// computing the error:
            {
                // get new position from simulation
                loc = (it->get_location());

                // update location from theory
                loc_ref = loc_ref + 
                        vel_ref * this->dt + 
                        0.5*this->acceleration_ref * this->dt * this->dt;

                // compute position error
                error_pos += this->norm(loc - loc_ref);

                // get new velocity from simulation
                vel = (this->get_velocity_from_properties(it));

                // compute new velocity from theory and update
                vel_ref = (vel_ref + acceleration_ref * this->dt);

                // compute velocity error
                error_vel += this->norm(vel - vel_ref);

                // std::cout << "\nit: " << i << "  vel_error    pos_error" << 
                // std::endl  << "          " << this->norm(vel - vel_ref) << 
                // "        " << this->norm(loc - loc_ref)  << std::endl;

                // std::cout << "sim location: " << loc << std::endl;
                // std::cout << "ref location: " << loc_ref << std::endl;
                // std::cout << "sim velocity: " << vel << std::endl;
                // std::cout << "ref velocity: " << vel_ref << std::endl;
            }    
        }

        error_vel /= NT;
        error_pos /= NT;

        /// printing the errors and the elipsed time of the simulation
        std::cout << "\ntotal velocity error: " << error_vel << std::endl;
        std::cout << "total position error: " << error_pos << std::endl;
        std::cout << "streaming time:       " << DT_CUMULATIVE_CHRONO << std::endl;
    };

    template <int dim>
    void TestConstantAcceleration<dim>::initialize_particles()
    {
        // number of particles
        unsigned int N = 1;

        // random
        std::random_device rd_pos;
        std::mt19937 gen_pos;
        std::uniform_real_distribution<double> pick_position(0, 1);
        
        /// get particle location

        std::vector<location_type> positions(N);

        auto lower_vertex(this->domain_data.get_lower_vertex());
        auto difference(this->domain_data.get_upper_vertex() - lower_vertex);

        for(auto & pos : positions) {
            for(unsigned int d = 0; d<dim; ++d){
                pos[d] = lower_vertex[d] + 
                pick_position(gen_pos)*(difference[d]);
            }
        } 

        /// insert particle

        this->particle_handler.insert_particles(positions);

        /// assigned same constant velocity to each particle (there should be only one particle)
        
        for(auto it = this->particle_handler.begin();
            it!= this->particle_handler.end(); ++it){
                this->set_velocity(it, this->initial_velocity);
            }
        
        return;
    }

} // namespace TestSimulators


#endif // TEST_SIMULATORS_HPP