#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <fstream>

namespace bpo = boost::program_options;

extern std::string input_file;

void parse_args(int argc, char **argv);
void print_args();
