#pragma once
#include <glad/glad.h>
#include <fstream>
#include <vector>

std::vector<GLchar> LoadShader(const std::string &file) {

    std::ifstream shaderFile;
    long shaderFileLength;

    shaderFile.open(file, std::ios::binary);

    if (shaderFile.fail()) {
        throw std::runtime_error("COULD NOT FIND SHADER FILE " + file);
    }

    shaderFile.seekg(0, shaderFile.end);
    shaderFileLength = shaderFile.tellg();
    shaderFile.seekg(0, shaderFile.beg);

    std::vector<GLchar> shaderCode;
    shaderCode.resize(shaderFileLength + 1);
    shaderFile.read(shaderCode.data(), shaderFileLength);

    shaderFile.close();

    shaderCode[shaderFileLength] = '\0';

    return shaderCode;
}
