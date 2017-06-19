VK_SDK_PATH = /home/ex/dev/vulkan-lunarg-sdk/VulkanSDK/1.0.46.0/x86_64

CFLAGS = -std=c++11 -I$(VK_SDK_PATH)/include $(shell pkg-config --cflags sdl2 SDL2_image)

LDFLAGS = -L$(VK_SDK_PATH)/lib $(shell pkg-config --static --libs glfw3 sdl2 SDL2_image) -lvulkan

SHADER_CC = $(VK_SDK_PATH)/bin/glslangValidator

MODULES = main load_image
SRC = $(MODULES:%=%.cpp)
OBJ = $(MODULES:%=%.o)

all: vktest shaders

vktest: $(OBJ)
	g++ $(CFLAGS) -o $@ $^ $(LDFLAGS)

debug: CFLAGS += -ggdb -DDEBUG
debug: vktest

%.o: %.cpp
	g++ $(CFLAGS) -c -o $@ $<

shaders: shader.vert shader.frag
	$(SHADER_CC) -V shader.vert
	$(SHADER_CC) -V shader.frag

test: vktest shaders
	LD_LIBRARY_PATH=$(VK_SDK_PATH)/lib/ VK_LAYER_PATH=$(VK_SDK_PATH)/etc/explicit_layer.d ./$<

clean:
	rm -f ./vktest $(OBJ)

.PHONY: test debug clean
