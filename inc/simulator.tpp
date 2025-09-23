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
    const unsigned int &n_particles_,
    const Domain::DomainData<dim> &domain_data_):
n_particles(n_particles_),
dt(dt_),
T(T_),
mapping(1), 
dof_handler(triangulation),
particle_handler(triangulation, mapping, dim), 
domain_data(domain_data_),
streaming_algorithm(this),
collision_handler(this),
gen(rd()),
normal_distr(0., species.get_std_velocity())
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

    Timers::Timer timer;
    Timers::Cumulative_Timer sorting_timer;
    Timers::Cumulative_Timer collision_timer;
     
    std::cout << "---------- starting simulation ----------" << std::endl;
    timer.tic();

    for(unsigned int i = 0; i<NT; ++i){
        velocity_type sum_vels;

        streaming_algorithm.advance(particle_handler);

        std::cout << "\ntime step: " << i << std::endl;
        // std::cout << "# number of particles: " 
        //     << particle_handler.n_global_particles() << std::endl;
        collision_timer.tic();
        collision_handler.apply_collisions();
        collision_timer.toc();
        {
            for (auto it = particle_handler.begin(); it!=particle_handler.end(); ++it)
            {
                // Point<dim> position(it->get_location());
                Tensor<1, dim> velocity(get_velocity_from_properties(it));
                
                // save kinetic energy per particle
                double ke = particle_mass * velocity.norm();
                sum_particles_kinetic_energy[i] += ke;
                sum_vels += velocity;

            }
        }

        total_kinetic_energy[i] = particle_handler.n_global_particles() * particle_mass * sum_vels.norm();

        // sort particles into cells
        sorting_timer.tic();
        particle_handler.sort_particles_into_subdomains_and_cells();
        sorting_timer.toc();

        // print data to files
        print_data_to_files(i);
    } // end of time loop

    {
    double sim_time = timer.toc();

    std::cout << "Simulation completed in    " << sim_time    << " seconds" << std::endl; 
    std::cout << "sorting particles time:    ";
    sorting_timer.print(sim_time);
    std::cout << "collisions time:           ";
    collision_timer.print(sim_time);
    }
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

    triangulation.refine_global(2);

    std::cout << " * Grid created * " << std::endl;
    domain_data.print_data_domain();

    std::ofstream out("outputs/grid.vtk");
    dealii::GridOut     gridout;
    gridout.write_vtk(triangulation, out);

};

template<int dim>
void PICsimulator::Simulator<dim>::insert_particles_dummy()
{
    // int N = n_particles; // move it from here please!!!

    auto upper_vertex = domain_data.get_upper_vertex();
    auto lower_vertex = domain_data.get_lower_vertex();

   
    std::vector<Point<dim>> positions(n_particles);    
    for(auto & position : positions){
        for(int n = 0; n<dim; ++n){
            position[n] = lower_vertex[n] + 
                (upper_vertex[n] - lower_vertex[n]) * Random::RND();
        }
    }

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

        std::cout << " * Initialization skipped * " << std::endl;
        std::cout << "!!! YOU ARE PROBABLY RUNNING AGAIN THE SIMULATION !!!" << std::endl;
        return;
    }

    // initialize grid and particles
    make_grid();
    insert_particles_dummy();

    auto n_cells = triangulation.n_active_cells();
    double cell_volume_, volume_domain, particles_per_cell;
    double d_cell;

    for(auto & cell: dof_handler.active_cell_iterators()){
        cell_volume_ = cell->measure();
        volume_domain += cell_volume_;
        d_cell = cell->diameter();
    }
    number_density = n_particles/volume_domain;
    particles_per_cell = static_cast<double>(n_particles)/n_cells;

    cell_volume = volume_domain/n_cells;

    std::cout << "Number density: " << number_density << " #/m^dim"<< std::endl;
    std::cout << "Number of particles: " << n_particles << std::endl;
    std::cout << "Number of cells: " << n_cells << std::endl;
    std::cout << "Number of particles per cell: " 
        << particles_per_cell << std::endl;
    std::cout << "Diameter of a cell: " << d_cell << std::endl;
    std::cout << "Volume of a cell: " 
        << cell_volume << std::endl;
    std::cout << "Volume of the domain: " << volume_domain << std::endl;

    std::cout << "--------- species data ---------" << std::endl;
    std::cout << "mean velocity: " 
        << species.get_mean_velocity() << std::endl;
    std::cout << "mean free path : " 
        << species.get_free_path() << std::endl;

    std::cout << "time to exit a cell: " 
        << d_cell / species.get_mean_velocity() << std::endl;
    std::cout << "time step: " 
        << dt << std::endl;

    collision_handler.initialize();

    initialized = true;
    return;
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
