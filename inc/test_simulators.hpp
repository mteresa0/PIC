#ifndef TEST_SIMULATORS_HPP
#define TEST_SIMULATORS_HPP

#include "simulator.hpp"
#include "domain.hpp"
#include <cmath>

namespace TestSimulators
{
    using namespace dealii;

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
        void initialize() override;

        virtual void initialize_particles() = 0;
        virtual void initialize_grid() = 0;

        double norm(const velocity_type & v){
            double r = 0;
            for (int d = 0; d<dim; ++d){
                r += v[d]*v[d];
            }
            return std::sqrt(r);
        }

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
        void initialize_grid() override;

        const velocity_type velocity_ref;
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

    template<int dim>
    void TestConstantVelocity<dim>::run_test()
    {
        this->initialize();

        unsigned int NT = this->T/this->dt;

        double error;

        for (unsigned int i = 0; i<NT; ++i)
        {
            this->update_simulation_state();
            
            for(auto it = this->particle_handler.begin();
            it!= this->particle_handler.end(); ++it){
                velocity_type vel(this->get_velocity_from_properties(it));
                error += this->norm(vel-velocity_ref);
            }
        }

        std::cout << "total error: " << error << std::endl;
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
        
        std::vector<location_type> positions(N);

        auto lower_vertex(this->domain_data.get_lower_vertex());
        auto difference(this->domain_data.get_upper_vertex() - lower_vertex);

        std::cout << "picking location particle\n";
        for(auto & pos : positions) {
            for(unsigned int d = 0; d<dim; ++d){
                pos[d] = lower_vertex[d] + 
                pick_position(gen_pos)*(difference[d]);
            }
        } 
        // location_type position(lower_vertex + 0.5*difference);

        std::cout << "inserting particle\n";
        this->particle_handler.insert_particles(positions);
        // this->particle_handler.insert_particle(position);

        std::cout << "setting velocity particle\n";
        for(auto it = this->particle_handler.begin();
            it!= this->particle_handler.end(); ++it){
                this->set_velocity(it, velocity_ref);
            }
        
        return;
    }

    template<int dim>
    void TestConstantVelocity<dim>::initialize_grid()
    {
        this->make_grid();
        return;
    }

} // namespace TestSimulators


#endif // TEST_SIMULATORS_HPP