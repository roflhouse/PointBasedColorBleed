/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#include "../Objects/Objects.h"

typedef struct Scene {
   Sphere *spheres;
   Plane *planes;
   Triangle *triangles;
   int numSpheres;
   int numTriangles;
   int numPlanes;
} Scene;
