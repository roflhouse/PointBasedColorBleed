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
int createInitRays( struct Ray **rays, int width, int height, float growth, struct Camera cam );
int createDrawingRays( struct Ray **rays, int width, int height, struct Camera cam );
struct SurfelArray createSurfels( const struct Scene &scene, Ray *rays, int numRays );
struct Scene createSurfelSpheres( const struct Scene &scene, Ray *rays, int numRays );
void castRays( const struct Scene &scene, struct Ray *rays, int numRays, Color **buffer);
void castRaysSphere( const struct Scene &scene, struct Ray *rays, int numRays, Color **buffer);
void castRays( const struct SurfelArray &scene, struct Ray *rays, int numRays, Color **buffer);
void collectIntersections( const Scene &scene, const Ray &ray, IntersectionArray &IA );
Color raytrace( const struct Scene &scene, const Ray &ray );
Color raytrace2( const struct Scene &scene, const Ray &ray );
#include "../Objects/Surfel.h"
Color raytrace( const struct SurfelArray &scene, const Ray &ray );
#endif
