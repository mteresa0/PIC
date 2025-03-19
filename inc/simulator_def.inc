#include "simulator.hpp"

/// @brief constructor
/// @tparam dim 
template<int dim>
PICsimulator::Simulator<dim>::Simulator() :
mapping(1), 
particle_handler(triangulation, mapping, dim), 
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

/// @brief run simulation
/// @tparam dim 
template<int dim>
void PICsimulator::Simulator<dim>::run() {
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

/// @brief creates the mesh
/// @tparam dim 
template<int dim>
void PICsimulator::Simulator<dim>::make_grid() {

    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(4);

    std::cout << "Grid created" << std::endl;

    std::ofstream out("outputs/grid.vtk");
    dealii::GridOut     gridout;
    gridout.write_vtk(triangulation, out);
    std::cout << "grid written in vtk" << std::endl;

};

/// @brief dummy function that creates particles (it will be replaced)
/// @tparam dim 
template<int dim>
void PICsimulator::Simulator<dim>::insert_particles_dummy()
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

/// @brief set the initial velocity of each particle in the particle handler according to maxwell-boltzmann distribution (see documentation)
/// @tparam dim 
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

/// @brief called by the particle lost signal.
/// @brief Instead of losing the particle when exiting the domain, the new position is computed and a new particle is inserted
/// @tparam dim 
/// @param particle 
/// @param cell 
template <int dim>
void PICsimulator::Simulator<dim>::boundary_wall( 
    const typename Particles::ParticleIterator<dim>         &particle,
    const typename Triangulation<dim>::active_cell_iterator &cell) 
{
    Point<dim> location(particle->get_location());
    Tensor<1, dim> velocity(get_velocity(particle));
    std::vector<double> new_velocity(dim);

    double upper_boundary = 1;
    double lower_boundary = 0;

    for (int i = 0; i<dim; ++i){
        if (location[i]<lower_boundary){
            new_velocity[i] = 2*lower_boundary-velocity[i];
            location[i] = -location[i];
            // std::cout << "particle has been lost (<0)\n";
        } else if (location[i]>upper_boundary){
            new_velocity[i] = -velocity[i];
            location[i] = 2*upper_boundary-location[i];
            // std::cout << "particle has been lost (>1)\n";
        } else {
            new_velocity[i] = velocity[i];
        }
    }

    // add new particle
    Particles::Particle new_particle(location, particle->get_reference_location(), particle->get_id());

    particle_handler.insert_particle(location,
    particle->get_reference_location(),
    particle->get_id(),
    cell,
    new_velocity);    

}

/// @brief set the velocity of a particle
/// @tparam dim 
/// @param particle 
/// @param velocity 
template <int dim>
void PICsimulator::Simulator<dim>::set_velocity(typename Particles::ParticleIterator<dim> &particle, const velocity_type &velocity)
{
    particle->set_properties(velocity);
    return;
}

/// @brief get the velocity of a particle
/// @tparam dim 
/// @param particle 
/// @return 
template <int dim>
typename PICsimulator::Simulator<dim>::velocity_type PICsimulator::Simulator<dim>::get_velocity(const typename Particles::ParticleIterator<dim> &particle)
{
    return velocity_type(particle->get_properties());
}

/// @brief prints at a given timestep i the properties of each particle
/// @tparam dim 
/// @param i 
template<int dim>
void PICsimulator::Simulator<dim>::print_data_to_files(const int  & i){

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
