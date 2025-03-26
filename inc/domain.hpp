#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

namespace Domain{    
    using namespace dealii;

    enum boundary_type {
        REFLECTIVE,
        PERIODIC
    };

    template<int dim, boundary_type BTYPE>
    class DomainData{

        static_assert(dim==2||dim==3, "Only 2D and 3D domain are allowed");

        using location_type = Point<dim>;
        using velocity_type = Tensor<1,dim>;

    public: 
        // constructor
        DomainData(
            const location_type & lower_v_, 
            const location_type & upper_v_) : 
            lower_vertex(lower_v_), upper_vertex(upper_v_) {};
            
        static constexpr boundary_type btype = BTYPE;

        const location_type lower_vertex;
        const location_type upper_vertex;

        void out_of_boundaries(
            location_type & loc, 
            velocity_type & vel);

    };

    template <int dim, boundary_type BTYPE>
    inline void DomainData<dim, BTYPE>::out_of_boundaries(location_type &location, velocity_type &velocity)
    {
        // if REFLECTIVE WALLS
        if constexpr (btype==REFLECTIVE) 
        {
            std::cout << "reflective\n";

            for (int i = 0; i<dim; ++i){
                if (location[i]<lower_vertex[i]){
                    location[i] = 2*lower_vertex[i]-location[i];
                    velocity[i] = -velocity[i];
                } else if (location[i]>upper_vertex[i]){
                    location[i] = 2*upper_vertex[i]-location[i];
                    velocity[i] = -velocity[i];
                } else {
                    velocity[i] = velocity[i];
                    location[i] = location[i];
                }
            };
            return;            
        } 
        // if PERIODIC BOX
        else if constexpr (btype==PERIODIC)
        {
            std::cout << "periodic\n";

            // location_type new_location(location);
            location_type difference(upper_vertex - lower_vertex);

            for (int i = 0; i<dim; ++i){
                if (location[i]<lower_vertex[i]){
                    location[i] += difference[i];
                }  
                else if (location[i]>upper_vertex[i]){
                    location[i] -= difference[i];
                }
            }
            return;
        }

        std::cerr << "boundary type chosen is invalid\n";
        return;
        
    }

}
#endif // DOMAIN_HPP