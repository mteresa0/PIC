#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

namespace Domain{    
    using namespace dealii;

    enum boundary_type {
        REFLECTIVE,
        PERIODIC, 
        INVALID // to be deleted
    };

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
    /// @tparam BTYPE 
    template<int dim>
    class DomainData{
        static_assert(dim==2||dim==3, "Only 2D and 3D domain are allowed");

    public:

        using location_type = Point<dim>;
        using velocity_type = Tensor<1,dim>;

        /// @brief constructor
        /// @param lower_v_ 
        /// @param upper_v_ 
        DomainData(
            const location_type & lower_v_ = UnitSquare::get_lower_vertex<dim>(), 
            const location_type & upper_v_ = UnitSquare::get_upper_vertex<dim>()): 
            lower_vertex(lower_v_), upper_vertex(upper_v_) {
                std::cout << "DomainData constructor" << std::endl;
            };
        
        virtual ~DomainData() = default;

        /// @brief updates location and velocity of a particle when it leaves the domain
        /// @param loc 
        /// @param vel 
        virtual void out_of_boundaries (
            location_type & loc, 
            velocity_type & vel) const;

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

        void print_data_domain() const {
            std::cout << "---------- DOMAIN INFO ------------\n";
            std::cout << "DOMAIN TYPE:      " << this->domainType() << std::endl;
            std::cout << "lower vertex:      {";
            for(unsigned d = 0; d<dim; ++d){
                std::cout << "   " << lower_vertex[d];
            }
            std::cout << "   }\n";
            std::cout << "upper vertex:      {";
            for(unsigned d = 0; d<dim; ++d){
                std::cout << "   " << upper_vertex[d];
            }
            std::cout << "   }\n";
            std::cout << "-----------------------------------\n";
        }

    protected:

        // lower and upper vertex of the domain (rectangle or parallelepiped)
        const location_type lower_vertex;
        const location_type upper_vertex;

        virtual const char* domainType() const {return "Base Class Domain";};
    };

    /// @brief domain with elastic collisions against the "walls"
    /// @tparam dim 
    template<int dim>
    class ReflectiveDomain : public DomainData<dim>{
    public:
        using location_type = typename DomainData<dim>::location_type;
        using velocity_type = typename DomainData<dim>::velocity_type;

        ReflectiveDomain(
            const location_type & lower_v_ = UnitSquare::get_lower_vertex<dim>(), 
            const location_type & upper_v_ = UnitSquare::get_upper_vertex<dim>()): 
            DomainData<dim>(lower_v_, upper_v_) {};

        void out_of_boundaries(
            location_type & loc, 
            velocity_type & vel) const override;
    private:
        const char* domainType() const override {return "Reflective walls domain";};
    };

    /// @brief periodic box domain
    /// @tparam dim 
    template<int dim>
    class PeriodicDomain : public DomainData<dim>{
    public:
        using location_type = typename DomainData<dim>::location_type;
        using velocity_type = typename DomainData<dim>::velocity_type;

        PeriodicDomain(
            const location_type & lower_v_ = UnitSquare::get_lower_vertex<dim>(), 
            const location_type & upper_v_ = UnitSquare::get_upper_vertex<dim>()): 
            DomainData<dim>(lower_v_, upper_v_) {};

        void out_of_boundaries(
            location_type & loc, 
            velocity_type & vel) const override;
    private: 
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

    template <int dim>
    inline void DomainData<dim>::out_of_boundaries(location_type &, velocity_type &) const
    {
        std::cout << "base datadomain" << std::endl;
    }

    template<int dim>
    inline void ReflectiveDomain<dim>::out_of_boundaries(location_type & location, velocity_type & velocity) const
    {
        for (int i = 0; i<dim; ++i){
            if (location[i]<this->lower_vertex[i]){
                location[i] = 2*this->lower_vertex[i]-location[i];
                velocity[i] = -velocity[i];
            } else if (location[i]>this->upper_vertex[i]){
                location[i] = 2*this->upper_vertex[i]-location[i];
                velocity[i] = -velocity[i];
            } else {
                velocity[i] = velocity[i];
                location[i] = location[i];
            }
        };   
        return;
    }

    template<int dim>
    inline void PeriodicDomain<dim>::out_of_boundaries(location_type & location, velocity_type & ) const
    {
        location_type difference(this->upper_vertex - this->lower_vertex);

        for (int i = 0; i<dim; ++i){
            if (location[i]<this->lower_vertex[i]){
                location[i] += difference[i];
            }  
            else if (location[i]>this->upper_vertex[i]){
                location[i] -= difference[i];
            }
        }
        return;
    }

} // end namespace Domain
#endif // DOMAIN_HPP