/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef SCENE_H
#define SCENE_H
#include "../Objects/Objects.h"

typedef struct Scene {
   Sphere *spheres;
   Plane *planes;
   Triangle *triangles;
   PointLight *pointLights;
   int numSpheres;
   int numTriangles;
   int numPlanes;
   int numPointLights;

   Camera camera;
} Scene;
#endif
