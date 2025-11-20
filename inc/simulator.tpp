/**
 * @file simulator.tpp
 * @author Maria Teresa Cazzola 
 * @brief definitions of class simulator methods
 * @date 2025-04-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "simulator.hpp"

template<int dim>
PICsimulator::Simulator<dim>::Simulator(
    const double &dt_,
    const double &T_,
    const species_type &species_,
    const Domain::DomainData<dim> &domain_data_):
dt(dt_),
T(T_),
species(species_),
n_species(species.get_n_species()),
mapping(1), 
dof_handler(triangulation),
particle_handlers(n_species),
domain_data(domain_data_),
streaming_algorithm(this),
collision_handler(this)
{};

template<int dim>
void PICsimulator::Simulator<dim>::run() {
    LOGTITLE("starting initialization");
    initialize();

    unsigned int NT = T/dt;

    // TODO: add option to not run timers
    Timers::Timer timer;
    Timers::Cumulative_Timer sorting_timer;
    Timers::Cumulative_Timer streaming_timer;
    Timers::Cumulative_Timer collision_timer;
     
    LOGTITLE("starting simulation");
    timer.tic();

    for(unsigned int i = 0; i<NT; ++i){
        velocity_type sum_vels;

        streaming_timer.tic();
        for (auto & ph : particle_handlers){
            streaming_algorithm.advance(ph);
        }
        streaming_timer.toc();

        LOG("time step: " << i );

        // sort particles into cells
        sorting_timer.tic();
        for(auto & ph: particle_handlers){
            ph.sort_particles_into_subdomains_and_cells();
        }
        sorting_timer.toc();

        // apply collisions
        collision_timer.tic();
        collision_handler.apply_collisions();
        collision_timer.toc();

        #ifdef EXTRA_OUTPUTS
        // to check the kinetic enegies
        double total_kinetic_energy(0.);
        double sum_particles_kinetic_energy(0.);
        {
            unsigned int it_ph = 0;
            for (auto & ph : particle_handlers) {
                for (auto it = ph.begin(); it!=ph.end(); ++it)
                {
                    // Point<dim> position(it->get_location());
                    Tensor<1, dim> velocity(get_velocity_from_properties(it));
                    
                    // save kinetic energy per particle
                    double ke = species[it_ph].get_mass() * velocity.norm();
                    sum_particles_kinetic_energy += ke;
                    sum_vels += velocity;

                }
                total_kinetic_energy = ph.n_global_particles() * 
                    species[it_ph].get_mass() * sum_vels.norm();
                ++it_ph;
            }
        }
        out_kinetic_energy.write_row({i*dt, total_kinetic_energy, sum_particles_kinetic_energy});

        // print data to files
        print_data_to_files(i);
        #endif

    } // end of time loop

    // printing elapsed times
    {
    double sim_time = timer.toc();

    std::cout << "Simulation completed in    " << sim_time    << " seconds" << std::endl; 
    std::cout << "streaming particles time:  ";
    streaming_timer.print(sim_time);
    std::cout << "sorting particles time:    ";
    sorting_timer.print(sim_time);
    std::cout << "collisions time:           ";
    collision_timer.print(sim_time);
    }

};

template<int dim>
void PICsimulator::Simulator<dim>::make_grid() {

    dealii::GridGenerator::hyper_rectangle(
        triangulation, 
        domain_data.get_lower_vertex(), 
        domain_data.get_upper_vertex());

    triangulation.refine_global(4);

    LOG("\n * Grid created * ");
    domain_data.print_data_domain();

    std::ofstream out("outputs/grid.vtk");
    dealii::GridOut     gridout;
    gridout.write_vtk(triangulation, out);

};

template<int dim>
void PICsimulator::Simulator<dim>::insert_particles_dummy()
{
    // get upper and lower vertex
    auto upper_vertex = domain_data.get_upper_vertex();
    auto lower_vertex = domain_data.get_lower_vertex();

    for (unsigned int i = 0; i<n_species; ++i) {
        // generate random position for each particle. store positions in a vector
        std::vector<Point<dim>> positions(species.get_n_particles(i));    
        for(auto & position : positions){
            for(int n = 0; n<dim; ++n){
                position[n] = Random::rnd_uniform(lower_vertex[n], upper_vertex[n]);
            }
        }

        // insert particles by location
        particle_handlers[i].insert_particles(positions);
    }

    // set velocities
    set_initial_velocity();

    // printing message
    LOG("Inserted particles");
};

template<int dim>
void PICsimulator::Simulator<dim>::set_initial_velocity()
{
    // initialize velocity 
    Tensor<1, dim> initial_velocity;

    for(unsigned int i = 0; i<n_species; ++i) {
        // for each particle draw all the velocity components and set the velocity of the particle
        for(auto it = particle_handlers[i].begin(); it!=particle_handlers[i].end(); ++it){
            for(int d = 0; d<dim; ++d){
                initial_velocity[d] = 
                    Random::rnd_gaussian(0., species[i].get_a());
            }
            set_velocity(it, initial_velocity);
        }
    }
    return;
};

template <int dim>
void PICsimulator::Simulator<dim>::boundary_wall( 
    const typename Particles::ParticleIterator<dim>         &particle,
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int & index_species) 
{
    // get position and velocity of the particle. initialize new velocity variable
    Point<dim> location(particle->get_location());
    Tensor<1, dim> velocity(get_velocity_from_properties(particle));
    std::vector<double> new_velocity(dim);

    // run out_of_boundaries algorithm. it depends on the type of domain.
    // it will update the location and velocity given.
    domain_data.out_of_boundaries(location, velocity);

    // move the new velocity in the vector form bc the particle handler wants it this way.
    for(int d = 0; d<dim; ++d){
        new_velocity[d] = velocity[d];
    }

    // insert the lost particle 
    particle_handlers[index_species].insert_particle(
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
        std::cerr << "\n============================== WARNING ==============================\n"
        << " Simulation has already been initialized.\n"
        << " The initialization step has been skipped.\n"
        << " Please verify if running the simulation multiple times is intended.\n"
        << "=====================================================================\n";
        return;  
    }

    // initialize particles handlers
    for (unsigned int i = 0; i<n_species; ++i){
        particle_handlers[i].initialize(triangulation, mapping, dim);
        particle_handlers[i].signals.particle_lost.connect(
        [this, i]( const typename Particles::ParticleIterator<dim>         &particle,
                const typename Triangulation<dim>::active_cell_iterator &cell) {
            this->boundary_wall(particle, cell, i);
            }
        );
    }

    // initialize grid and particles
    make_grid();
    insert_particles_dummy();

    {
    // compute and store data for grid and particles
    auto n_cells = triangulation.n_active_cells();
    double volume_domain, sum_d, mean_d_cell, mean_cell_volume, cell_volume_;

    for(auto & cell: dof_handler.active_cell_iterators()){
        cell_volume_ = cell->measure();
        volume_domain += cell_volume_;
        sum_d += cell->diameter();
    }

    mean_d_cell = sum_d/n_cells;
    mean_cell_volume = volume_domain/n_cells;

    // printing data
    LOGVAR("Number of cells", n_cells);
    LOGVAR("Mean diameter of cells", mean_d_cell);
    LOGVAR("Mean cell volume", mean_cell_volume);
    LOGVAR("Domain volume", volume_domain);
    LOGVAR("Timestep", dt);
    LOGVAR("Number of timesteps", T/dt);
    }

    // initialize collisions
    collision_handler.initialize();

    initialized = true;
    return;
}

template<int dim>
void PICsimulator::Simulator<dim>::print_data_to_files(
    const int  & i)
{
    // for (const auto ph : particle_handlers){
    // Particles::DataOut<dim> particle_out;
    
    // particle_out.build_patches( 
    //     ph,
    //     std::vector<std::string>(dim, "velocity"),
    //     std::vector<DataComponentInterpretation::DataComponentInterpretation>(
    //         dim, 
    //         DataComponentInterpretation::component_is_part_of_vector));

    // DataOutBase::VtkFlags output_flags;
    // output_flags.time                       = dt*i;
    // output_flags.cycle                      = i;
    // output_flags.physical_units["velocity"] = "m/s";

    // particle_out.set_flags(output_flags);

    // std::ofstream output("outputs/particles-" +
    //                     Utilities::int_to_string(i, 5) +
    //                     ".vtu");
    // particle_out.write_vtu(output);
    // }
}
