#include "simulator.hpp"

template<int dim>
PICsimulator::Simulator<dim>::Simulator(
    const Domain::DomainData<dim> &domain_data_, 
    const double &dt_,
    const double &T_):
mapping(1), 
dof_handler(triangulation),
particle_handler(triangulation, mapping, dim), 
dt(dt_),
T(T_),
domain_data(domain_data_),
gen(rd()),
normal_distr(0., std::sqrt(BOLTZMANN_K * TEMPERATURE / ELECTR_MASS))
{
    particle_handler.signals.particle_lost.connect(
    [this]( const typename Particles::ParticleIterator<dim>         &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell) {
        this->boundary_wall(particle, cell);
        }
    );
};

template<int dim>
void PICsimulator::Simulator<dim>::run() {
    std::cout << "-------- starting initialization --------" << std::endl;
    initialize();

    unsigned int NT = T/dt;
    std::cout << "# timepsteps: " << NT;
    std::cout << " ( dt: " << dt;
    std::cout << "; T: "  << T << " )" << std::endl;

    std::vector<double> total_kinetic_energy(NT, 0);
    std::vector<double> sum_particles_kinetic_energy(NT, 0);
     
    std::cout << "---------- starting simulation ----------" << std::endl;
    TIC();
    REINIT_CUMULATIVE_CHRONO();

    for(unsigned int i = 0; i<NT; ++i){
        velocity_type sum_vels;

        update_simulation_state();

        {
            for (auto it = particle_handler.begin(); it!=particle_handler.end(); ++it)
            {
                // Point<dim> position(it->get_location());
                const Tensor<1, dim> &velocity(get_velocity_from_properties(it));
                
                // // update position
                // it->set_location(position + dt*velocity);

                // update velocity
                // set_velocity(it, velocity);
                
                // save kinetic energy per particle
                double ke = particle_mass * velocity.norm();
                sum_particles_kinetic_energy[i] += ke;
                sum_vels += velocity;

            }
        }

        total_kinetic_energy[i] = particle_handler.n_global_particles() * particle_mass * sum_vels.norm();

        // sort particles into cells
        START_CUMULATIVE_CHRONO();
        particle_handler.sort_particles_into_subdomains_and_cells();
        ADD_TIME_CUMULATIVE_CHRONO();

        // print data to files
        print_data_to_files(i);
    }

    TOC();
    std::cout << "Simulation completed in " << DT      << " seconds" << std::endl; 
    std::cout << "sort particles time:    " << DT_CUMULATIVE_CHRONO << " seconds" << std::endl;

    // print kinetic energies into csv
    std::ofstream output("outputs/total_kinetic_energy.csv");
    output << "time, total_kinetic_energy, sum_particles_kinetic_energy\n";
    for (unsigned int i = 0; i<NT; ++i){
        output << i*dt << "," << total_kinetic_energy[i] << "," << sum_particles_kinetic_energy[i] << "\n";
    }
};

template<int dim>
void PICsimulator::Simulator<dim>::make_grid() {

    dealii::GridGenerator::hyper_rectangle(
        triangulation, 
        domain_data.get_lower_vertex(), 
        domain_data.get_upper_vertex());

    triangulation.refine_global(4);

    std::cout << "Grid created" << std::endl;
    domain_data.print_data_domain();

    std::ofstream out("outputs/grid.vtk");
    dealii::GridOut     gridout;
    gridout.write_vtk(triangulation, out);

};

template<int dim>
void PICsimulator::Simulator<dim>::insert_particles_dummy()
{
    int N = 3; // move it from here please!!!

    Point<dim> initial_pos_vector;

    auto upper_vertex = domain_data.get_upper_vertex();
    auto lower_vertex = domain_data.get_lower_vertex();

    for(int n = 0; n<dim; ++n){
        initial_pos_vector[n] = lower_vertex[n] + 
            (upper_vertex[n] - lower_vertex[n]) * 0.5;
    }
    
    std::vector<Point<dim>> positions(N, initial_pos_vector);

    particle_handler.insert_particles(positions);

    // set velocities
    set_initial_velocity();

    std::cout << "Inserted " << particle_handler.n_global_particles() << " particles \n";
};

template<int dim>
void PICsimulator::Simulator<dim>::set_initial_velocity()
{
    Tensor<1, dim> initial_velocity;
    for(auto it = particle_handler.begin(); it!=particle_handler.end(); ++it){
        for(int d = 0; d<dim; ++d){
            initial_velocity[d] = normal_distr(gen);
        }
        set_velocity(it, initial_velocity);
    }
    return;
};

template <int dim>
void PICsimulator::Simulator<dim>::boundary_wall( 
    const typename Particles::ParticleIterator<dim>         &particle,
    const typename Triangulation<dim>::active_cell_iterator &cell) 
{
    Point<dim> location(particle->get_location());
    Tensor<1, dim> velocity(get_velocity_from_properties(particle));
    std::vector<double> new_velocity(dim);

    domain_data.out_of_boundaries(location, velocity);

    for(int d = 0; d<dim; ++d){
        new_velocity[d] = velocity[d];
    }

    particle_handler.insert_particle(
        location,
        mapping.transform_real_to_unit_cell(cell, location),
        particle->get_id(),
        cell,
        new_velocity);   
}

template <int dim>
void PICsimulator::Simulator<dim>::initialize()
{
    if (initialized)
    {
        std::cout << "Already initialized with:" << std::endl;
        std::cout << "n# particles: " << particle_handler.n_global_particles() << std::endl;
        std::cout << "n# cells: "     << triangulation.n_active_cells() << std::endl;
        return;
    }

    // initialize grid and particles
    make_grid();
    insert_particles_dummy();

    initialized = true;
    return;
}

template <int dim>
void PICsimulator::Simulator<dim>::update_simulation_state()
{
    // update position
    update_position();

    // // evaluate forces
    // evaluate_forces();

    // update velocity
    update_velocity();

    return;
}

template <int dim>
typename PICsimulator::Simulator<dim>::location_type
 PICsimulator::Simulator<dim>::evaluate_acceleration()
{
    return location_type();
}

template <int dim>
void PICsimulator::Simulator<dim>::update_position()
{
    // initialize acceleration
    location_type acceleration;

    {
        for (auto it = particle_handler.begin(); it!=particle_handler.end(); ++it)
        {
            Point<dim> position(it->get_location());
            const Tensor<1, dim> &velocity(get_velocity_from_properties(it));

            acceleration = this->evaluate_acceleration();
            
            // update position
            it->set_location(position + dt*velocity + 0.5*dt*dt*acceleration);
            set_velocity(it, velocity + 0.5 *dt * acceleration);
        }
    }
}

template <int dim>
void PICsimulator::Simulator<dim>::update_velocity()
{

    // initialize acceleration
    location_type new_acceleration;

    {
        for (auto it = particle_handler.begin(); it!=particle_handler.end(); ++it)
        {
            const Tensor<1, dim> &velocity(get_velocity_from_properties(it));

            new_acceleration = this->evaluate_acceleration();

            // update position
            set_velocity(it, velocity + 0.5*dt*new_acceleration);
        }
    }
}

template <int dim>
void PICsimulator::Simulator<dim>::set_velocity(
    typename Particles::ParticleIterator<dim> &particle, 
    const velocity_type &velocity)
{
    particle->set_properties(velocity);
    return;
}

template <int dim>
typename PICsimulator::Simulator<dim>::velocity_type
PICsimulator::Simulator<dim>::get_velocity_from_properties(
    const typename Particles::ParticleIterator<dim> &particle)
{
    return velocity_type(particle->get_properties());
}

template<int dim>
void PICsimulator::Simulator<dim>::print_data_to_files(
    const int  & i)
{

    Particles::DataOut<dim> particle_out;
    
    particle_out.build_patches( 
        particle_handler,
        std::vector<std::string>(dim, "velocity"),
        std::vector<DataComponentInterpretation::DataComponentInterpretation>(
            dim, 
            DataComponentInterpretation::component_is_part_of_vector));

    DataOutBase::VtkFlags output_flags;
    output_flags.time                       = dt*i;
    output_flags.cycle                      = i;
    output_flags.physical_units["velocity"] = "m/s";

    particle_out.set_flags(output_flags);

    std::ofstream output("outputs/particles-" +
                        Utilities::int_to_string(i, 5) +
                        ".vtu");
    particle_out.write_vtu(output);
}
