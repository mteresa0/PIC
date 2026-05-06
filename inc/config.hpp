#ifndef CONFIG
#define CONFIG

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <deal.II/base/point.h>
using json = nlohmann::json;

template<int dim>
struct Config {
    double dt;
    int n_timesteps;
    dealii::Point<dim> lower_vertex;
    dealii::Point<dim> upper_vertex;
    std::string multispecies_file;
    std::string multispecies_name;
    unsigned int grid_refinement;

    Config();
};

template<int dim>
inline Config<dim>::Config()
{
    const std::string filename = "data/input.json";

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    json j;
    file >> j;

    // dim = j.at("dim").get<int>();
    dt = j.at("dt").get<double>();
    n_timesteps = j.at("n_timesteps").get<int>();

    auto lv = j.at("lower_vertex").get<std::vector<double>>();
    auto uv = j.at("upper_vertex").get<std::vector<double>>();

    for(unsigned int i=0; i<dim; ++i){
        upper_vertex[i] = uv[i];
        lower_vertex[i] = lv[i];
    }

    multispecies_file = j.at("multispecies_file").get<std::string>();
    multispecies_name = j.at("multispecies_name").get<std::string>();

    grid_refinement = j.at("grid_refinement").get<unsigned int>();
}

#endif // CONFIG
