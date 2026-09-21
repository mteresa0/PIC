#ifndef CONFIG
#define CONFIG

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <deal.II/base/point.h>
using json = nlohmann::json;

struct Config {
    int dim;
    double dt;
    int n_timesteps;
    std::vector<double> lower_vertex;
    std::vector<double> upper_vertex;
    std::string multispecies_file;
    std::string multispecies_name;
    unsigned int grid_refinement;

    Config(const std::string & );
};

inline Config::Config(const std::string & filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    json j;
    file >> j;

    // // check if the input dimention in the same give as templare param
    // int dim_from_json = j.at("dim").get<int>();
    // if(dim!=dim_from_json){
    //     throw std::runtime_error(
    //         "template paramenter dimention and config dimension are different");
    // }
    dim = j.at("dim").get<int>();

    // continue 
    dt = j.at("dt").get<double>();
    n_timesteps = j.at("n_timesteps").get<int>();

    lower_vertex = j.at("lower_vertex").get<std::vector<double>>();
    upper_vertex = j.at("upper_vertex").get<std::vector<double>>();

    multispecies_file = j.at("multispecies_file").get<std::string>();
    multispecies_name = j.at("multispecies_name").get<std::string>();

    grid_refinement = j.at("grid_refinement").get<unsigned int>();
}

#endif // CONFIG
