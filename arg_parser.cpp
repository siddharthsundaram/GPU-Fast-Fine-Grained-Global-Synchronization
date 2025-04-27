#include "arg_parser.h"

std::string input_file;
bool seq;
bool basic;
bool fg;

void parse_args(int argc, char **argv) {
    
    bpo::options_description od("Options");
    od.add_options()
        // ("-i", bpo::value<std::string>()->required(), "Input filename")
        ("-s", "Enable sequential implementation")
        ("-g", "Enable basic GPU implementation")
        ("-f", "Enable fine-grain buffer GPU implementation");

    bpo::variables_map var_map;
    try {
        bpo::store(bpo::parse_command_line(argc, argv, od), var_map);
        bpo::notify(var_map);
    } catch (const bpo::error &e) {
        std::cout << "Error: " << e.what() << "\n" << std::endl;
        return;
    }

    if (var_map.empty()) {
        std::cout << "No options were parsed!" << std::endl;
    }

    // input_file = var_map["-i"].as<std::string>();
    seq = var_map.count("-s") > 0;
    basic = var_map.count("-g") > 0;
    fg = var_map.count("-f") > 0;
}

void print_args() {
    std::cout << "Input File: " << input_file << "\n";
}