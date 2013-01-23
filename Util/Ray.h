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

#include "../Objects/Objects.h"
#include "RayType.h"
typedef struct TreeHitMark {
   float t;
   Color color;
} TreeHitMark;
void evaluateSphereicalHermonics();
void evaluateSphereicalHermonicsPower();

#include "Scene.h"
#include "Intersection.h"
struct SurfelArray createSurfels( const struct Scene &scene, Ray *rays, int numRays );
struct Scene createSurfelSpheres( const struct Scene &scene, Ray *rays, int numRays );
void castRays( const struct Scene &scene, struct Ray *rays, int numRays, Color *buffer, int width);
void castRaysSphere( const struct Scene &scene, struct Ray *rays, int numRays, Color *buffer, int width);
void castRays( const struct SurfelArray &scene, struct Ray *rays, int numRays, Color *buffer, int width);
void collectIntersections( const Scene &scene, const Ray &ray, IntersectionArray &IA );
Color raytrace( const struct Scene &scene, const Ray &ray );
Color raytrace2( const struct Scene &scene, const Ray &ray );
#include "Octree.h"
struct TreeNode createSurfelTree( const struct Scene &scene, Ray *rays, int numRays );
void castRays( const struct TreeNode &scene, struct Ray *rays, int numRays, Color *buffer, int width );
Color raytrace( const struct TreeNode &Tree, const Ray &ray );
#include "../Objects/Surfel.h"
Color raytrace( const struct SurfelArray &scene, const Ray &ray );

TreeHitMark transTree( TreeNode root, const Ray &ray );

void castRays( const struct ArrayNode *scene, int size, SurfelArray &SA, struct Ray *rays, 
      int numRays, Color *buffer, int width );
struct ArrayNode *createSurfelsCuda( const struct Scene &scene, Ray *rays, int numRays,
      SurfelArray &SA, int &size );
Color raytrace( const struct ArrayNode *Tree, int size, SurfelArray &SA, const Ray &ray );
struct ArrayNode *createSurfelsCuda( const struct Scene &scene, Ray *rays, int numRays, int &size );

void traverseOctreeCPU( TreeNode &node, float maxangle );
#endif
