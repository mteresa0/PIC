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
    public:
        static_assert(dim==2||dim==3, "Only 2D and 3D domain are allowed");

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
    public:
        // lower and upper vertex of the domain (rectangle or parallelepiped)
        const location_type lower_vertex;
        const location_type upper_vertex;

        /// @brief updates location and velocity of a particle when it leaves the domain
        /// @param loc 
        /// @param vel 
        virtual void out_of_boundaries (
            location_type & loc, 
            velocity_type & vel) const;

    };

    template<int dim>
    class ReflectiveDomainData : public DomainData<dim>{
    public:
        using location_type = typename DomainData<dim>::location_type;
        using velocity_type = typename DomainData<dim>::velocity_type;

        ReflectiveDomainData(
            const location_type & lower_v_ = UnitSquare::get_lower_vertex<dim>(), 
            const location_type & upper_v_ = UnitSquare::get_upper_vertex<dim>()): 
            DomainData<dim>(lower_v_, upper_v_) {};

        void out_of_boundaries(
            location_type & loc, 
            velocity_type & vel) const override;
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

    // template <int dim, boundary_type BTYPE>
    // inline void DomainData<dim, BTYPE>::out_of_boundaries(location_type &location, velocity_type &velocity) const 
    // {
    //     // if REFLECTIVE WALLS
    //     if constexpr (BTYPE==REFLECTIVE) 
    //     {
    //         for (int i = 0; i<dim; ++i){
    //             if (location[i]<lower_vertex[i]){
    //                 location[i] = 2*lower_vertex[i]-location[i];
    //                 velocity[i] = -velocity[i];
    //             } else if (location[i]>upper_vertex[i]){
    //                 location[i] = 2*upper_vertex[i]-location[i];
    //                 velocity[i] = -velocity[i];
    //             } else {
    //                 velocity[i] = velocity[i];
    //                 location[i] = location[i];
    //             }
    //         };        
    //     } 
    //     // if PERIODIC BOX
    //     else if constexpr (BTYPE==PERIODIC)
    //     {
    //         // location_type new_location(location);
    //         location_type difference(upper_vertex - lower_vertex);

    //         for (int i = 0; i<dim; ++i){
    //             if (location[i]<lower_vertex[i]){
    //                 location[i] += difference[i];
    //             }  
    //             else if (location[i]>upper_vertex[i]){
    //                 location[i] -= difference[i];
    //             }
    //         }
    //     } 
    //     // with an invalid type, in compile time i should have an error given by the static_assert (only for the invalid type)
    //     else 
    //     {
    //         static_assert(false, "invalid boundary type");
    //     }
        
    // }

    template <int dim>
    inline void DomainData<dim>::out_of_boundaries(location_type &, velocity_type &) const
    {
        std::cout << "base datadomain" << std::endl;
    }



template<int dim>
inline void ReflectiveDomainData<dim>::out_of_boundaries(location_type & location, velocity_type & velocity) const
{
    // std::cout << "Reflective domain" << std::endl;
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
}

}
#endif // DOMAIN_HPP