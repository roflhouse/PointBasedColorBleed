/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#include "Sphere.h"
#include "Plane.h"
#include "Triangle.h"

typedef struct Scene {
   Sphere *spheres;
   Plane *planes;
   Triangle *triangles;
} Scene;
