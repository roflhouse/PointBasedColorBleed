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
#include "../Objects/Camera.h"
#include "../Objects/Objects.h"

typedef struct Scene {
   struct Sphere *spheres;
   struct Plane *planes;
   struct Triangle *triangles;
   struct PointLight *pointLights;
   struct Camera camera;

   int numSpheres;
   int numTriangles;
   int numPlanes;
   int numPointLights;
} Scene;
#endif
