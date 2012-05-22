/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef PARSER_H
#define PARSER_H
#include <vector>
#include <stdio.h>
#include <string>

#include "../Objects/Object.h"
#include "../Objects/Camera.h"
#include "../Objects/LightSource.h"
#include "../Objects/Plane.h"
#include "../Objects/Sphere.h"
#include "../Objects/Triangle.h"

#include "BVH.h"

#include "Vector.h"

#ifdef CUDA_ENABLED
#include "../Cuda/CudaDefs.h"
#endif

extern Camera *camera;
extern LightSource **lights;
extern Object **spheres;
extern Object **planes;
extern Object **triangles;
extern BVH *bvh;
extern int numLights;
extern int numTriangles;
extern int numSpheres;
extern int numPlanes;
extern int maxSpheres;
extern int maxPlanes;
extern int maxTriangles;
extern int maxLights;

class Parser
{
    public:
        Parser( std::string filename );
};
#endif
