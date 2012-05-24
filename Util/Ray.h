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
#include "Color.h"

typedef struct Ray {
   vec3 pos;
   vec3 dir;
   int i, j;
} Ray;

#include "../Objects/Objects.h"
#include "Scene.h"
#include "Intersection.h"
int createInitRays( struct Ray **rays, int width, int height, struct Camera cam );
void castRays( struct Scene scene, struct Ray *rays, int numRays, int width, int height, Color **buffer);
Color raytrace( struct Scene scene, Ray ray );
#endif
