VK_SDK_PATH = /home/ex/dev/vulkan-lunarg-sdk/VulkanSDK/1.0.46.0/x86_64

CFLAGS = -std=c++11 -I$(VK_SDK_PATH)/include

LDFLAGS = -L$(VK_SDK_PATH)/lib `pkg-config --static --libs glfw3` -lvulkan

SHADER_CC = $(VK_SDK_PATH)/bin/glslangValidator

all: vktest

vktest: main.cpp shaders
	g++ $(CFLAGS) -o $@ $< $(LDFLAGS)

debug: CFLAGS += -ggdb -DDEBUG
debug: vktest

shaders: shader.vert shader.frag
	$(SHADER_CC) -V shader.vert
	$(SHADER_CC) -V shader.frag

test: vktest
	LD_LIBRARY_PATH=$(VK_SDK_PATH)/lib/ VK_LAYER_PATH=$(VK_SDK_PATH)/etc/explicit_layer.d ./$<

clean:
	rm -f ./vktest

.PHONY: test debug clean
