#pragma once

#include <string>

void *load_image(std::string filename,
                 int &width, int &height, int &channels);
