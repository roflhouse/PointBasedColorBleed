/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef RAY_H
#define RAY_H
#include "vec3.h"

typedef struct Ray {
   vec3 pos;
   vec3 dir;
   int i, j;
} Ray;

#include "Scene.h"
#include "../Objects/Objects.h"
#include "Intersection.h"
int createInitRays( Ray **rays, int width, int height, Camera cam );
void castRays( Scene scene, Ray *rays, int numRays, int width, int height, Color **buffer);
Color raytrace( Scene scene, Ray ray );
#endif
