#include "gpu_depthmap_fusion/gl_compute_test.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>

#include "radix_sort.h"

/**
* The number of particles we want to simulate. Since compute shaders are limited in their
* work group size, we'll also need to know the work group size so we can find out how many
* work groups to dispatch
*/
#define MIO 1000000
#define KILO 1000
//#define NUM_PARTICLES (10*KILO)
//#define NUM_PARTICLES (10*MIO)
//#define NUM_PARTICLES (500*KILO)
//#define NUM_PARTICLES 1000
//#define NUM_PARTICLES 10000
// #define NUM_PARTICLES 100000
#define NUM_PARTICLES (1*MIO)
//#define NUM_PARTICLES (2*MIO)
// #define NUM_PARTICLES (5*MIO)
//#define NUM_PARTICLES 1000
//#define NUM_PARTICLES (10*MIO)
//#define NUM_COMPUTE_UNITS 1
//#define NUM_COMPUTE_UNITS 10
 //#define NUM_COMPUTE_UNITS 28
//#define NUM_COMPUTE_UNITS 280
//#define NUM_COMPUTE_UNITS 1000
//#define NUM_COMPUTE_UNITS 10000
//#define RADIX_SORT_GROUP_SIZE (1024*2)
//#define RADIX_SORT_GROUP_SIZE 512
//#define RADIX_SORT_GROUP_SIZE 1024*128
// #define RADIX_SORT_GROUP_SIZE (1024*256)
// #define RADIX_SORT_GROUP_SIZE (1024)
#define RADIX_SORT_GROUP_SIZE (1024*8)
// #define RADIX_SORT_THREADS 64
#define RADIX_SORT_THREADS 8
// #define RADIX_SORT_THREADS 2
//#define RADIX_SORT_THREADS 32
//#define RADIX_SORT_GROUP_SIZE 20


// This MUST match with local_size_x inside cursor.glsl
#define WORK_GROUP_SIZE 1000

#define SCREENX 1920
#define SCREENY 1080

glm::vec2 defaultCursor = glm::vec2(0.0f, 0.0f);
typedef uint32_t uint;



GLchar* LoadShaderAllocate(const std::string &file) {

    std::ifstream shaderFile;
    long shaderFileLength;

    shaderFile.open(file, std::ios::binary);

    if (shaderFile.fail()) {
        throw std::runtime_error("COULD NOT FIND SHADER FILE " + file);
    }

    shaderFile.seekg(0, shaderFile.end);
    shaderFileLength = shaderFile.tellg();
    shaderFile.seekg(0, shaderFile.beg);

    GLchar *shaderCode = new GLchar[shaderFileLength + 1];
    shaderFile.read(shaderCode, shaderFileLength);

    shaderFile.close();

    shaderCode[shaderFileLength] = '\0';

    return shaderCode;
}

void printGpuInfo()
{
    std::vector<std::string> paramNames{ 
        "GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS",
        "GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS",
        "GL_MAX_COMPUTE_UNIFORM_BLOCKS",
        "GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS",
        "GL_MAX_COMPUTE_UNIFORM_COMPONENTS",
        "GL_MAX_COMPUTE_ATOMIC_COUNTERS",
        "GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS",
        "GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS",
        "GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS",
        "GL_MAX_COMPUTE_WORK_GROUP_COUNT",
        "GL_MAX_COMPUTE_WORK_GROUP_SIZE"
    };
    std::vector<GLenum> params{ 
        GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS,
        GL_MAX_COMBINED_SHADER_STORAGE_BLOCKS,
        GL_MAX_COMPUTE_UNIFORM_BLOCKS,
        GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS,
        GL_MAX_COMPUTE_UNIFORM_COMPONENTS,
        GL_MAX_COMPUTE_ATOMIC_COUNTERS,
        GL_MAX_COMPUTE_ATOMIC_COUNTER_BUFFERS,
        GL_MAX_COMBINED_COMPUTE_UNIFORM_COMPONENTS,
        GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS,
        GL_MAX_COMPUTE_WORK_GROUP_COUNT,
        GL_MAX_COMPUTE_WORK_GROUP_SIZE
    };
    std::vector<GLint> intValues;
    intValues.resize(params.size());
    for (int i = 0; i < params.size(); ++i)
    {
        glGetIntegerv(params[i], &intValues[i]);
        std::cout << paramNames[i] << " " << intValues[i] << std::endl;
    }
}

struct GridCell
{
    glm::uint32 seqNum;
    glm::uint32 start;
    glm::uint32 size;
    //glm::uint32 seqNum;
};

int glComputeTest(
    const std::string& filename_compute,
    const std::string& filename_compute2,
    const std::string& filename_vertex,
    const std::string& filename_fragment
) {
    srand(time(0));

    const GLfloat delta_time = 0.1f;

    //Window Setup
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

    GLFWwindow* window = glfwCreateWindow(SCREENX, SCREENY, "particles", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    gladLoadGL();

    glEnable(GL_DEBUG_OUTPUT);


    //Setup Initial Positions/Velocities
    std::vector<glm::vec4> positions;
    std::vector<glm::vec4> velocities;
    std::vector<glm::uint32> cellNumbers;
    glm::uint32 numUniqueCells;
    std::vector<glm::uint32> cellStarts;
    std::vector<glm::uint32> cellSizes;
    
    std::vector<GridCell> grid;
    glm::uint32 seqNum=1;

    std::vector<glm::uint32> sortedIndices;
    std::vector<glm::uint32> sortedCellNumbers;
    std::vector<glm::uint32> tmpSortedIndices;
    std::vector<glm::uint32> tmpSortedCellNumbers;

    std::vector<glm::uint32> cellContents5;
    positions.resize(NUM_PARTICLES);
    velocities.resize(NUM_PARTICLES);
    cellNumbers.resize(NUM_PARTICLES);
    sortedIndices.resize(NUM_PARTICLES);
    sortedCellNumbers.resize(NUM_PARTICLES);
    tmpSortedIndices.resize(NUM_PARTICLES);
    tmpSortedCellNumbers.resize(NUM_PARTICLES);

    cellContents5.resize(NUM_PARTICLES);
    for(int i = 0; i < NUM_PARTICLES; i++) {
        float r = (float)rand() / RAND_MAX;
        r *= 8;
        float velx = r * sin(i) / 10;
        float vely = r * cos(i) / 10;
        positions[i] = glm::vec4(velx, vely, 0.0f, 0.0f);
        velocities[i] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    glm::vec2 lowerBound(-1, -1);
    glm::vec2 upperBound(+1, +1);
    glm::vec2 worldSize = upperBound - lowerBound;
    //glm::vec2 cellSize(0.001, 0.001);
    // glm::vec2 cellSize(0.005, 0.005);
    glm::vec2 cellSize(0.01, 0.01);
    //glm::vec2 cellSize(0.05, 0.05);
    //glm::vec2 cellSize(0.1, 0.1);
    //glm::vec2 cellSize(0.5, 0.5);
    glm::ivec2 gridSize((int)ceil(worldSize.x / cellSize.x), (int)ceil(worldSize.y / cellSize.y));
    //cellSizes.resize(gridSize.x*gridSize.y);

    grid.resize(gridSize.x * gridSize.y);
    cellStarts.resize(std::min(gridSize.x * gridSize.y, NUM_PARTICLES));
    cellSizes.resize(std::min(gridSize.x * gridSize.y, NUM_PARTICLES));

    GLuint vao, pos, timebuffer, vel, cursor, cellNumbersBuf, cellSizesBuf, cellContentsBuf, cellContents2Buf;
    GLuint gridBuf, sortedIndicesBuf;


    /**
    * VAO Setup - Even though we'll be sourcing all of our positions from a shader storage buffer,
    * a VAO must be bound for drawing, so let's just create an empty one. We also don't need a VBO
    * since all our rendering will be points
    */
// TODO: replace by glGenVertexArrays    glCreateVertexArrays(1, &vao);
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    /**
    * Buffer setup - We'll need 4 SSBOs, one for the position of each point, one for velocity,
    * one for cursor position, and one for delta_time. As cursor and delta_time are small, these
    * could have been implemented using regular uniforms, but I opted for consistency instead
    */
    glGenBuffers(1, &pos);
    glGenBuffers(1, &timebuffer);
    glGenBuffers(1, &vel);
    glGenBuffers(1, &cursor);
    glGenBuffers(1, &cellNumbersBuf);
    glGenBuffers(1, &gridBuf);
    glGenBuffers(1, &sortedIndicesBuf);

    //glCreateBuffers(1, &cellSizesBuf);
    //glCreateBuffers(1, &cellContentsBuf);
    //glCreateBuffers(1, &cellContents2Buf);
    
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, vel);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * velocities.size(), velocities.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, pos);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * positions.size(), positions.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, timebuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GLfloat), &delta_time, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cursor);
    glBufferData(GL_SHADER_STORAGE_BUFFER, 2 * sizeof(GLfloat), glm::value_ptr(defaultCursor), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellNumbersBuf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::uint32) * cellNumbers.size(), cellNumbers.data(), GL_STATIC_READ);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortedIndicesBuf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::uint32) * sortedIndices.size(), sortedIndices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridBuf);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(GridCell) * grid.size(), grid.data(), GL_DYNAMIC_DRAW);


// TODO: replace by glBufferData    glNamedBufferData(vel, sizeof(glm::vec4) * velocities.size(), velocities.data(), GL_STATIC_DRAW);
// TODO: replace by glBufferData    glNamedBufferData(pos, sizeof(glm::vec4) * positions.size(), positions.data(), GL_STATIC_DRAW);
// TODO: replace by glBufferData    glNamedBufferData(timebuffer, sizeof(GLfloat), &delta_time, GL_DYNAMIC_DRAW);
// TODO: replace by glBufferData    glNamedBufferData(cursor, 2 * sizeof(GLfloat), glm::value_ptr(defaultCursor), GL_DYNAMIC_DRAW);
// TODO: replace by glBufferData    glNamedBufferData(cellNumbersBuf, sizeof(glm::uint32) * cellNumbers.size(), cellNumbers.data(), GL_STATIC_READ);
// TODO: replace by glBufferData    glNamedBufferData(sortedIndicesBuf, sizeof(glm::uint32) * sortedIndices.size(), sortedIndices.data(), GL_DYNAMIC_DRAW);
// TODO: replace by glBufferData    glNamedBufferData(gridBuf, sizeof(GridCell) * grid.size(), grid.data(), GL_DYNAMIC_DRAW);
    //glNamedBufferData(cellNumbersBuf, sizeof(glm::uint32) * cellNumbers.size(), cellNumbers.data(), GL_STATIC_DRAW);
    //glNamedBufferData(cellSizesBuf, sizeof(glm::uint32) * cellSizes.size(), cellSizes.data(), GL_STATIC_DRAW);
    //glNamedBufferData(cellContentsBuf, sizeof(glm::uint32) * cellContents.size(), cellContents.data(), GL_STATIC_DRAW);
    //glNamedBufferData(cellContents2Buf, sizeof(glm::uint32) * cellContents2.size(), cellContents2.data(), GL_STATIC_DRAW);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, cursor);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, timebuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, cellNumbersBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, sortedIndicesBuf);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, gridBuf);
    //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, cellSizesBuf);
    //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, cellContentsBuf);
    //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, cellContents2Buf);

    //printGpuInfo();

    /*
    * Basic Shader Setup - This is standard shader loading and compilation. The only difference is that the compute
    * shader must be linked by itself in a separate program
    */
    const GLchar* computeCode =  LoadShaderAllocate(filename_compute);
    const GLchar* compute2Code = LoadShaderAllocate(filename_compute2);
    const GLchar* vertCode =     LoadShaderAllocate(filename_vertex);
    const GLchar* fragCode =     LoadShaderAllocate(filename_fragment);

    // const GLchar* vertCode = LoadShaderAllocate("/home/huenermu/ros/ws/current/build/gpu_depthmap_fusion/shader.vert");
    // const GLchar* fragCode = LoadShaderAllocate("/home/huenermu/ros/ws/current/build/gpu_depthmap_fusion/shader.frag");
    // const GLchar* computeCode = LoadShaderAllocate("/home/huenermu/ros/ws/current/build/gpu_depthmap_fusion/compute.glsl");
    // const GLchar* compute2Code = LoadShaderAllocate("/home/huenermu/ros/ws/current/build/gpu_depthmap_fusion/compute2.glsl");

    GLuint vertShader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShader = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    GLuint compute2Shader = glCreateShader(GL_COMPUTE_SHADER);

    glShaderSource(vertShader, 1, &vertCode, nullptr);
    glShaderSource(fragShader, 1, &fragCode, nullptr);
    glShaderSource(computeShader, 1, &computeCode, nullptr);
    glShaderSource(compute2Shader, 1, &compute2Code, nullptr);
    GLchar infolog[512];
    glCompileShader(vertShader);
    glGetShaderInfoLog(vertShader, 512, nullptr, infolog);

    std::cout << infolog << std::endl;

    glCompileShader(fragShader);
    glGetShaderInfoLog(fragShader, 512, nullptr, infolog);
    std::cout << infolog << std::endl;

    glCompileShader(computeShader);
    glGetShaderInfoLog(computeShader, 512, nullptr, infolog);
    std::cout << infolog << std::endl;

    glCompileShader(compute2Shader);
    glGetShaderInfoLog(compute2Shader, 512, nullptr, infolog);
    std::cout << infolog << std::endl;

    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertShader);
    glAttachShader(shaderProgram, fragShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    glGetProgramInfoLog(shaderProgram, 512, nullptr, infolog);

    std::cout << infolog << std::endl;

    GLuint computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    glDeleteShader(computeShader);

    glGetProgramInfoLog(computeProgram, 512, nullptr, infolog);
    std::cout << infolog << std::endl;

    //GLuint compute2Program = glCreateProgram();
    //glAttachShader(compute2Program, compute2Shader);
    //glLinkProgram(compute2Program);
    //glDeleteShader(compute2Shader);

    //glGetProgramInfoLog(compute2Program, 512, nullptr, infolog);
    //std::cout << infolog << std::endl;

    GLint loc_screenSize = glGetUniformLocation(computeProgram, "screenSize");
    GLint loc_lowerBound = glGetUniformLocation(computeProgram, "lowerBound");
    GLint loc_cellSize = glGetUniformLocation(computeProgram, "cellSize");
    GLint loc_gridSize = glGetUniformLocation(computeProgram, "gridSize");
    GLint loc_seqNum = glGetUniformLocation(computeProgram, "seqNum");
    


    glUniform2f(loc_lowerBound, lowerBound.x, lowerBound.y);
    glUniform2f(loc_cellSize, cellSize.x, cellSize.y);
    glUniform2i(loc_gridSize, gridSize.x, gridSize.y);
    glUniform1ui(loc_seqNum, seqNum);

    //GLint loc_numItems = glGetUniformLocation(compute2Program, "numItems");
    //glUniform1ui(loc_numItems, RADIX_SORT_GROUP_SIZE);
    //std::cout << "num items per sort core " << RADIX_SORT_GROUP_SIZE << std::endl;
            
    /**
    * Setup some basic properties for screen size, background color and point size
    * 2.0 Point Size was used to make it easier to see
    */
    glViewport(0, 0, SCREENX, SCREENY);
    //glClearColor(0.05f, 0.05, 0.05f, 1.0f);
    glPointSize(2.0f);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); 
    glEnable(GL_BLEND); 
    glClearColor(0.0, 0.0, 0.0, 0.0);

    GLuint timeQueries[2];
    glGenQueries(2,timeQueries);
    
    std::vector<glm::uvec4> digitCounts;
    std::vector<glm::uvec4> digitStarts;
    std::vector<glm::uvec4> digitPointers;

    int lastW = SCREENX, lastH = SCREENY;
    // Draw Loop
    while(glfwWindowShouldClose(window) == 0) {
        glClear(GL_COLOR_BUFFER_BIT);
        /**
        * Get our cursor coordinates in normalized device coordinates
        */
        glfwPollEvents();
        double cursorx;
        double cursory;
        glfwGetCursorPos(window, &cursorx, &cursory);
        int w, h;
        glfwGetWindowSize(window, &w, &h);
        if (w != lastW || h != lastH)
        {
                glViewport(0, 0, w, h);
                lastW = w;
                lastH = h;
        }

        cursorx = cursorx - (w / 2);
        cursorx /= w;
        cursorx *= 2;

        cursory = cursory - (h / 2);
        cursory /= h;
        cursory *= -2;

        if (loc_screenSize != -1)
        {
            glProgramUniform2f(computeProgram, loc_screenSize, w, h);
        }
        if (loc_lowerBound != -1) glProgramUniform2f(computeProgram, loc_lowerBound, lowerBound.x, lowerBound.y);
        if (loc_cellSize != -1) glProgramUniform2f(computeProgram, loc_cellSize, cellSize.x, cellSize.y);
        if (loc_gridSize != -1) glProgramUniform2i(computeProgram, loc_gridSize, gridSize.x, gridSize.y);
        if (loc_seqNum != -1) glProgramUniform1ui(computeProgram, loc_seqNum, seqNum);
        //if (loc_numItems != -1) glProgramUniform1ui(compute2Program, loc_numItems, RADIX_SORT_GROUP_SIZE);
        /**
        * Copy over the cursor position into the buffer
        */
        glm::vec2 current_cursor = glm::vec2(cursorx, cursory);

// TODO: replace by glBufferSubData        glNamedBufferSubData(cursor, 0, 2 * sizeof(GLfloat), glm::value_ptr(current_cursor));
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cursor);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 2 * sizeof(GLfloat), glm::value_ptr(current_cursor));
        
        auto t0 = std::chrono::high_resolution_clock::now();

        glBeginQuery(GL_TIME_ELAPSED, timeQueries[0]);
        /**
        * Fire off our compute shader run. Since we want to simulate 100000 particles, and our work group size is 1000
        * we need to dispatch 100 work groups
        */
        glUseProgram(computeProgram);
        glDispatchCompute(NUM_PARTICLES / WORK_GROUP_SIZE, 1, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        //glMemoryBarrier(GL_ALL_BARRIER_BITS);

        glEndQuery(GL_TIME_ELAPSED);
        GLuint64 elapsed_time0;
        glGetQueryObjectui64v(timeQueries[0], GL_QUERY_RESULT, &elapsed_time0);

        //for (int i = 0; i < 128; ++i)
        //{
        //    glDispatchCompute(NUM_PARTICLES / WORK_GROUP_SIZE, 1, 1);
        //    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        //}


        auto t1 = std::chrono::high_resolution_clock::now();

        glBeginQuery(GL_TIME_ELAPSED, timeQueries[1]);

       /**
        * Fire off our compute shader run. Since we want to simulate 100000 particles, and our work group size is 1000
        * we need to dispatch 100 work groups
        */
        //glUseProgram(compute2Program);
        //glDispatchCompute((int)ceil(NUM_PARTICLES / RADIX_SORT_GROUP_SIZE), 1, 1);
        //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        //glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glEndQuery(GL_TIME_ELAPSED);

        //glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        auto t2 = std::chrono::high_resolution_clock::now();

        /**
        * Draw the particles
        */
        glUseProgram(shaderProgram);
        glDrawArraysInstanced(GL_POINTS, 0, 1, NUM_PARTICLES);

        glfwSwapBuffers(window);
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        auto t3 = std::chrono::high_resolution_clock::now();
        //glGetNamedBufferSubData(cellContentsBuf, 0, sizeof(glm::uint32) * cellContents.size(), cellContents.data());
// TODO: replace by glGetBufferSubData      glGetNamedBufferSubData(cellNumbersBuf, 0, sizeof(glm::uint32) * cellNumbers.size(), cellNumbers.data());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, cellNumbersBuf);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::uint32) * cellNumbers.size(), cellNumbers.data());

        //glGetNamedBufferSubData(vel, 0, sizeof(glm::vec4) * velocities.size(), velocities.data());
        //glm::uint32* cellNumbersPtr = static_cast<glm::uint32*>(glMapNamedBuffer(cellNumbersBuf, GL_READ_ONLY));
        auto t4 = std::chrono::high_resolution_clock::now();
        //radixSort(cellNumbers, cellContents, cellContents2, cellContents3, cellContents4);
        //radixSortInParts(RADIX_SORT_GROUP_SIZE, cellNumbers, cellContents, cellContents2, cellContents3, cellContents4);
        omp_set_num_threads(RADIX_SORT_THREADS);
        radixWithRedistribution(
            RADIX_SORT_GROUP_SIZE, 
            //cellNumbersPtr, cellNumbers.size(),
            cellNumbers,
            sortedIndices, sortedCellNumbers,
            tmpSortedIndices, tmpSortedCellNumbers,
            digitCounts, digitStarts, digitPointers
        );
        auto t5 = std::chrono::high_resolution_clock::now();
        
        uint lastCellNumber = sortedCellNumbers[0];
        uint lastCellStart = 0;
        //uint currentCellSize = 0;
        numUniqueCells = 0;
        ++seqNum;
        for (int i = 1; i < NUM_PARTICLES; ++i)
        {
            uint cellNumber = sortedCellNumbers[i];
            if (cellNumber != lastCellNumber)
            {
                cellStarts[numUniqueCells] = lastCellStart;
                cellSizes[numUniqueCells] = i - lastCellStart;
                grid[lastCellNumber].seqNum = seqNum;
                grid[lastCellNumber].start = cellStarts[numUniqueCells];
                grid[lastCellNumber].size = cellSizes[numUniqueCells];
                lastCellNumber = cellNumber;
                ++numUniqueCells;
                lastCellStart = i;
            }
        }
        cellStarts[numUniqueCells] = lastCellStart;
        cellSizes[numUniqueCells] = NUM_PARTICLES - lastCellStart;
        grid[lastCellNumber].seqNum = seqNum;
        grid[lastCellNumber].start = cellStarts[numUniqueCells];
        grid[lastCellNumber].size = cellSizes[numUniqueCells];
        ++numUniqueCells;

        //lastCellNumber = cellNumber;
        //radixSortInParts(1024, cellNumbers, cellContents, cellContents2, cellContents3, cellContents4);
        //radixSortInParts(512*32, cellNumbers, cellContents, cellContents2, cellContents3, cellContents4);
        //radixSortInParts(512*1024, cellNumbers, cellContents, cellContents2, cellContents3, cellContents4);
        //radixSortInParts(1024*1024, cellNumbers, cellContents, cellContents2, cellContents3, cellContents4);
        //radixSortInParts(1024*1024*512, cellNumbers, cellContents, cellContents2, cellContents3, cellContents4);
        auto t6 = std::chrono::high_resolution_clock::now();
        //glUnmapBuffer(cellNumbersBuf);
//         glBindBuffer(GL_SHADER_STORAGE_BUFFER, sortedIndicesBuf);
//         glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(glm::uint32) * sortedIndices.size(), sortedIndices.data());
//         glBindBuffer(GL_SHADER_STORAGE_BUFFER, gridBuf);
//         glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GridCell) * grid.size(), grid.data());
// // TODO: replace by glBufferSubData        glNamedBufferSubData(sortedIndicesBuf, 0, sizeof(glm::uint32) * sortedIndices.size(), sortedIndices.data());
// // TODO: replace by glBufferSubData        glNamedBufferSubData(gridBuf, 0, sizeof(GridCell) * grid.size(), grid.data());
//         glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        auto t7 = std::chrono::high_resolution_clock::now();

        //glNamedBufferData(cellContentsBuf, sizeof(glm::uint32) * cellContents.size(), cellContents.data(), GL_STATIC_DRAW);
        //glNamedBufferData(cellContents2Buf, sizeof(glm::uint32) * cellContents2.size(), cellContents2.data(), GL_STATIC_DRAW);
        //glGetBufferglGetNamedBufferSubData

        // http://www.lighthouse3d.com/tutorials/opengl-timer-query/
        for (int k = 0; k < 2; k++)
        {
            GLint done = 0;
            while (!done) {
                glGetQueryObjectiv(timeQueries[0],
                    GL_QUERY_RESULT_AVAILABLE,
                    &done);
            }
        }
        GLuint64 elapsed_time1;
        glGetQueryObjectui64v(timeQueries[1], GL_QUERY_RESULT, &elapsed_time1);
        auto d0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        auto d1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        auto d2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
        auto d3 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        auto d4 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
        auto d5 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count();
        auto d6 = std::chrono::duration_cast<std::chrono::microseconds>(t7 - t6).count();
        auto td = std::chrono::duration_cast<std::chrono::microseconds>(t7 - t0).count();

        //std::cout << "elapsed0 " << elapsed_time0 << " compute" << std::endl;
        //std::cout << "elapsed1 " << elapsed_time1 << " compute2" << std::endl;

        std::cout << "duration0 " << d0 << " compute" << std::endl;
        std::cout << "duration1 " << d1 << " compute2" << std::endl;
        std::cout << "duration2 " << d2 << " draw" << std::endl;
        std::cout << "duration3 " << d3 << " read mem" << std::endl;
        std::cout << "duration4 " << d4 << " cpu radix sort" << std::endl;
        std::cout << "duration5 " << d5 << " count cells " << std::endl;
        std::cout << "duration6 " << d6 << " copy grid data to gpu  " << std::endl;
        std::cout << "duration total " << td << " " << std::endl;
        std::cout << "numUniqueCells " << numUniqueCells << " " << std::endl;
    }

    /**
    * We're done now, just need to free up our resources
    */

    //OpenGL Shutdown
    glDeleteProgram(shaderProgram);
    glDeleteProgram(computeProgram);
    //glDeleteProgram(compute2Program);
    delete[] vertCode;
    delete[] fragCode;
    delete[] computeCode;
    delete[] compute2Code;

    //Window Shutdown
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
