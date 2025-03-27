#include "simulator.hpp"

template<int dim, Domain::boundary_type btype>
PICsimulator::Simulator<dim, btype>::Simulator(
    const Domain::DomainData<dim, btype> &domain_data_
) :
mapping(1), 
particle_handler(triangulation, mapping, dim), 
gen(rd()),
normal_distr(0., std::sqrt(BOLTZMANN_K * TEMPERATURE / ELECTR_MASS)),
domain_data(domain_data_)
{
    particle_handler.signals.particle_lost.connect(
    [this]( const typename Particles::ParticleIterator<dim>         &particle,
            const typename Triangulation<dim>::active_cell_iterator &cell) {
        this->boundary_wall(particle, cell);
        }
    );
};

template<int dim, Domain::boundary_type btype>
void PICsimulator::Simulator<dim, btype>::run() {
    make_grid();
    insert_particles_dummy();

    unsigned int NT = T/dt;
    std::cout << NT << std::endl;

    std::vector<double> total_kinetic_energy(NT, 0);
    std::vector<double> sum_particles_kinetic_energy(NT, 0);
     

    for(unsigned int i = 0; i<NT; ++i){
        velocity_type sum_vels;
        for (auto it = particle_handler.begin(); it!=particle_handler.end(); ++it)
        {
            Point<dim> position(it->get_location());
            const Tensor<1, dim> velocity(get_velocity(it));
            
            // update position
            it->set_location(position + dt*velocity);

            // update velocity
            // set_velocity(it, velocity);
            
            // save kinetic energy per particle
            double ke = particle_mass * velocity.norm();
            sum_particles_kinetic_energy[i] += ke;
            sum_vels += velocity;

        }
        total_kinetic_energy[i] = particle_handler.n_global_particles() * particle_mass * sum_vels.norm();

        // sort particles into cells
        particle_handler.sort_particles_into_subdomains_and_cells();

        // print data to files
        print_data_to_files(i);
    }

    // print kinetic energies into csv
    std::ofstream output("outputs/total_kinetic_energy.csv");
    output << "time, total_kinetic_energy, sum_particles_kinetic_energy\n";
    for (unsigned int i = 0; i<NT; ++i){
        output << i*dt << "," << total_kinetic_energy[i] << "," << sum_particles_kinetic_energy[i] << "\n";
    }
};

template<int dim, Domain::boundary_type btype>
void PICsimulator::Simulator<dim, btype>::make_grid() {

    dealii::GridGenerator::hyper_rectangle(
        triangulation, 
        domain_data.lower_vertex, 
        domain_data.upper_vertex);

    triangulation.refine_global(4);

    std::cout << "Grid created" << std::endl;

    std::ofstream out("outputs/grid.vtk");
    dealii::GridOut     gridout;
    gridout.write_vtk(triangulation, out);
    std::cout << "grid written in vtk" << std::endl;

};

template<int dim, Domain::boundary_type btype>
void PICsimulator::Simulator<dim, btype>::insert_particles_dummy()
{
    int N = 5;

    Point<dim> initial_pos_vector;

    for(int n = 0; n<dim; ++n){
        initial_pos_vector[n] = 0.5;
    }
    
    std::vector<Point<dim>> positions(N, initial_pos_vector);

    particle_handler.insert_particles(positions);

    // set velocities
    set_initial_velocity();

    std::cout << "particles inserted!\n" << particle_handler.n_global_particles() << "\n";
};

template<int dim, Domain::boundary_type btype>
void PICsimulator::Simulator<dim, btype>::set_initial_velocity()
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

template <int dim, Domain::boundary_type btype>
void PICsimulator::Simulator<dim, btype>::boundary_wall( 
    const typename Particles::ParticleIterator<dim>         &particle,
    const typename Triangulation<dim>::active_cell_iterator &cell) 
{
    Point<dim> location(particle->get_location());
    Tensor<1, dim> velocity(get_velocity(particle));
    std::vector<double> new_velocity(dim);

    domain_data.out_of_boundaries(location, velocity);

    for(int d = 0; d<dim; ++d){
        new_velocity[d] = velocity[d];
    }

    particle_handler.insert_particle(
        location,
        particle->get_reference_location(),
        particle->get_id(),
        cell,
        new_velocity);    

}

template <int dim, Domain::boundary_type btype>
void PICsimulator::Simulator<dim, btype>::set_velocity(
    typename Particles::ParticleIterator<dim> &particle, 
    const velocity_type &velocity)
{
    particle->set_properties(velocity);
    return;
}

template <int dim, Domain::boundary_type btype>
typename PICsimulator::Simulator<dim, btype>::velocity_type 
PICsimulator::Simulator<dim, btype>::get_velocity(
    const typename Particles::ParticleIterator<dim> &particle)
{
    return velocity_type(particle->get_properties());
}

template<int dim, Domain::boundary_type btype>
void PICsimulator::Simulator<dim, btype>::print_data_to_files(
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
