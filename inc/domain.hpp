/**
 * @file domain.hpp
 * @author Maria Teresa Cazzola
 * @brief Declaration and definition of structures storing domain properties
 * @date 2025-04-11
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h> 
#include <deal.II/lac/affine_constraints.h>
#include "utilities.hpp"
#include <cmath>
#include <vector>

namespace Domain{    
    using namespace dealii;

    namespace UnitSquare{
        /// @brief return the lower vertex of the unit square (2D) or unit cube (3D) 
        /// @tparam dim 
        /// @return 
        template<int dim>
        Point<dim> get_lower_vertex();

        /// @brief return the upper vertex of the unit square (2D) or unit cube (3D)
        /// @tparam dim 
        /// @return 
        template<int dim>
        Point<dim> get_upper_vertex();
    };

    /// @brief store the vertex of the domain (shaped like a rectangle (2D) or a parallelepiped (3D)) 
    /// and the type of boundary conditions 
    /// @tparam dim 
    template<int dim>
    class DomainData{
        static_assert(dim==2||dim==3, "Only 2D and 3D domain are allowed");

    public:
        using location_type = Point<dim>;
        using velocity_type = Tensor<1,dim>;

        /// @brief constructor
        /// @param lower_v_ lower vertex
        /// @param upper_v_ upper vertex
        DomainData(
            const location_type & lower_v_ = UnitSquare::get_lower_vertex<dim>(), 
            const location_type & upper_v_ = UnitSquare::get_upper_vertex<dim>()):  
            lower_vertex(lower_v_), upper_vertex(upper_v_) {};

        /// descructor
        virtual ~DomainData() = default;

        /// @brief updates location and velocity of a particle when it leaves the domain
        /// @param loc location of the particle (to be updated)
        /// @param vel velocity of the particle (to be updated)
        virtual void out_of_boundaries (
            location_type &, 
            velocity_type &) const {};

        /// @brief returns the lower vertex of the domain
        /// @return
        const location_type & get_lower_vertex() const {
            return lower_vertex;
        }

        /// @brief returns the upper vertex of the domain
        /// @return
        const location_type & get_upper_vertex() const {
            return upper_vertex;
        }

        /// @brief prints domain informations
        void print_data_domain() const {
            LOGTITLE("DOMAIN INFO");
            LOGVAR("Domain type", this->domainType());
            LOGVEC("Lower vertex", lower_vertex, dim);
            LOGVEC("Upper vertex", upper_vertex, dim);
        }

        /// @brief allows further customization on the triangulation
        virtual void modify_triangulation(Triangulation<dim>& ) const {};

        /// @brief allows to apply constraints
        virtual void apply_constraints(
            DoFHandler<dim>& , AffineConstraints<double>& ) const {};

        /// @brief return domain volume
        double volume() const{
            double vol=upper_vertex[0] - lower_vertex[0];
            for (unsigned int i = 1; i<dim; ++i){
                vol *= upper_vertex[i] - lower_vertex[i];
            }
            return vol;
        }

    protected:

        // lower and upper vertex of the domain (rectangle or parallelepiped)
        const location_type lower_vertex; /// lower vertex point
        const location_type upper_vertex; /// upper vertex point

        virtual const char* domainType() const {return "Base Class Domain";}; /// domain type string
    };

    /// @brief domain with elastic collisions against the "walls"
    /// @tparam dim 
    template<int dim>
    class ReflectiveDomain : public DomainData<dim>{
    public:
        using location_type = typename DomainData<dim>::location_type;
        using velocity_type = typename DomainData<dim>::velocity_type;

        /// @brief constructor from dealii points
        /// @param lower_v_ dealii::Point<dim> lower vertex
        /// @param upper_v_ dealii::Point<dim> upper vertex
        ReflectiveDomain(
            const location_type & lower_v_ = UnitSquare::get_lower_vertex<dim>(), 
            const location_type & upper_v_ = UnitSquare::get_upper_vertex<dim>()): 
            DomainData<dim>(lower_v_, upper_v_) {};

        /// @brief constructor from std::vectors
        /// @param vec_lv vector<double> lower vertex
        /// @param vec_uv vector<double> upper vertex
        ReflectiveDomain(
            const std::vector<double> & vec_lv, 
            const std::vector<double> & vec_uv
        ) : ReflectiveDomain(
            PICsimulator::vectorToPoint<dim>(vec_lv), 
            PICsimulator::vectorToPoint<dim>(vec_uv)
        ) {};

        /// @brief this method is called when a particle is out of the domain.
        /// According to the domain it reinsertes the particle.
        /// @param loc particle lost location
        /// @param vel particle lost velocity
        void out_of_boundaries(
            location_type & loc, 
            velocity_type & vel) const override;
    private:
        /// @brief get domain type string
        const char* domainType() const override {return "Reflective walls domain";};
    };

    /// @brief periodic box domain
    /// @tparam dim 
    template<int dim>
    class PeriodicDomain : public DomainData<dim>{
    public:
        using location_type = typename DomainData<dim>::location_type;
        using velocity_type = typename DomainData<dim>::velocity_type;

        /// @brief constructor from dealii points
        /// @param lower_v_ dealii::Point<dim> lower vertex
        /// @param upper_v_ dealii::Point<dim> upper vertex
        explicit PeriodicDomain(
            const location_type & lower_v_ = UnitSquare::get_lower_vertex<dim>(), 
            const location_type & upper_v_ = UnitSquare::get_upper_vertex<dim>()): 
            DomainData<dim>(lower_v_, upper_v_) {};
        
        /// @brief constructor from std::vectors
        /// @param vec_lv vector<double> lower vertex
        /// @param vec_uv vector<double> upper vertex
        explicit PeriodicDomain(
            const std::vector<double> & vec_lv, 
            const std::vector<double> & vec_uv
        ) : PeriodicDomain(
            PICsimulator::vectorToPoint<dim>(vec_lv), 
            PICsimulator::vectorToPoint<dim>(vec_uv)
        ) {};

        /// @brief this method is called when a particle is out of the domain.
        /// According to the domain it reinsertes the particle.
        /// @param loc particle lost location
        /// @param vel particle lost velocity
        void out_of_boundaries(
            location_type & loc, 
            velocity_type & vel) const override;

        /// @brief adds periodicity on the triangulation
        void modify_triangulation(Triangulation<dim>& ) const override;

        /// @brief setup constraints and dof_handler for periodic boundaries
        void apply_constraints(DoFHandler<dim>&, 
                            AffineConstraints<double>& ) const override;
    private: 
        /// @brief get domain type string
        const char* domainType() const override {return "Periodic domain";}
    };


    // IMPLEMENTATION

    template<>
    inline Point<2> UnitSquare::get_lower_vertex(){
        return Point<2>(0,0);
    }

    template<>
    inline Point<2> UnitSquare::get_upper_vertex(){
        return Point<2>(1,1);
    }

    template<>
    inline Point<3> UnitSquare::get_lower_vertex(){
        return Point<3>(0,0,0);
    }

    template<>
    inline Point<3> UnitSquare::get_upper_vertex(){
        return Point<3>(1,1,1);
    }

    template<int dim>
    inline void ReflectiveDomain<dim>::out_of_boundaries(
        location_type & location, velocity_type & velocity) const
    {
        location_type difference(this->upper_vertex - this->lower_vertex);
        for (int i = 0; i<dim; ++i){
            double loc_norm_i = location[i]-this->lower_vertex[i];
            int n_lengths = std::floor(std::abs(loc_norm_i)/
                            (difference[i]));

            if (loc_norm_i<0) {++n_lengths;}
            if (n_lengths==0) {continue;}

            if (n_lengths%2==1){
                double vertex = loc_norm_i>0 ? 
                this->upper_vertex[i] : 
                this->lower_vertex[i];
                location[i] = -std::fmod(loc_norm_i, difference[i]) 
                    + vertex;
                velocity[i] = -velocity[i];
            } else {
                double vertex = loc_norm_i>0 ? 
                this->lower_vertex[i] : 
                this->upper_vertex[i];
                location[i] = std::fmod(location[i]-this->lower_vertex[i], 
                    difference[i]) + vertex;
            }
        }
        return;
    }

    template<int dim>
    inline void PeriodicDomain<dim>::out_of_boundaries(location_type & location, 
                                                       velocity_type & ) const
    {
        location_type difference(this->upper_vertex - this->lower_vertex);

        for (int i = 0; i<dim; ++i){
            double vertex = location[i]>this->lower_vertex[i] ? 
                this->lower_vertex[i] : 
                this->upper_vertex[i];
            location[i] = std::fmod(location[i]-this->lower_vertex[i], difference[i]) 
                        + vertex;
        }
        return;
    }

    template <int dim>
    inline void PeriodicDomain<dim>::modify_triangulation(
        Triangulation<dim>& triangulation) const
    {
        for (auto &cell : triangulation.active_cell_iterators())
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary()) {
            const auto center = cell->face(f)->center();
            if      (std::abs(center[0] - this->lower_vertex[0]) < 1e-18)
                cell->face(f)->set_boundary_id(0); // left
            else if (std::abs(center[0] - this->upper_vertex[0]) < 1e-18)
                cell->face(f)->set_boundary_id(1); // right
            else if (std::abs(center[1] - this->lower_vertex[1]) < 1e-18)
                cell->face(f)->set_boundary_id(2); // bottom
            else if (std::abs(center[1] - this->upper_vertex[1]) < 1e-18)
                cell->face(f)->set_boundary_id(3); // up
            else if constexpr (dim > 2) {
                if      (std::abs(center[2] - this->lower_vertex[2]) < 1e-18)
                    cell->face(f)->set_boundary_id(4); // 
                else if (std::abs(center[2] - this->upper_vertex[2]) < 1e-18)
                    cell->face(f)->set_boundary_id(5); //
            }
        }

        std::vector<GridTools::PeriodicFacePair<
            typename Triangulation<dim>::cell_iterator>> matched_pairs;

        GridTools::collect_periodic_faces(triangulation,
                                    0,
                                    1,
                                    0,
                                    matched_pairs);

        GridTools::collect_periodic_faces(triangulation,
                                    2,
                                    3,
                                    1,
                                    matched_pairs);

        if constexpr (dim > 2) {
            GridTools::collect_periodic_faces(triangulation,
                                        4,
                                        5,
                                        2,
                                        matched_pairs);
        }

        triangulation.add_periodicity(matched_pairs);

        return;
    }

    template <int dim>
    inline void PeriodicDomain<dim>::apply_constraints(
        DoFHandler<dim>& dof_handler, 
        AffineConstraints<double>& constraints) const 
    {
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        std::vector<GridTools::PeriodicFacePair
        <typename DoFHandler<dim>::cell_iterator>> periodicity_vector;
        
        GridTools::collect_periodic_faces(dof_handler,
                                0,
                                1,
                                0,
                                periodicity_vector);

        GridTools::collect_periodic_faces(dof_handler,
                        2,
                        3,
                        1,
                        periodicity_vector);
        
        DoFTools::make_periodicity_constraints<dim,dim>(periodicity_vector,constraints);
        return;
    }

} // end namespace Domain
#endif // DOMAIN_HPP