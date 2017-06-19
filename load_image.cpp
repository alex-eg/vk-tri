#include <iostream>
#include <stdexcept>
#include <string>

#include <SDL.h>
#include <SDL_image.h>

void *load_image(std::string filename,
                 int &width, int &height, int &channels)
{
    int flags = IMG_INIT_JPG | IMG_INIT_PNG;
    int inited = IMG_Init(flags);
    if ((inited & flags) != flags) {
        std::cerr << IMG_GetError() << '\n';
        throw std::runtime_error("failed to init sdl2_image");
    }
    SDL_Surface *image;
    image = IMG_Load(filename.c_str());
    if (!image) {
        throw std::runtime_error("failed to load image");
    }
    IMG_Quit();
    width = image->w;
    height = image->h;
    channels = image->format->BytesPerPixel;
    std::cout << "pixel format: " << SDL_GetPixelFormatName(image->format->format) << '\n';
    return image->pixels;
}
