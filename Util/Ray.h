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
#include "ColorType.h"
#include "RasterCube.h"
#include "../Objects/Objects.h"
#include "RayType.h"
#include "Scene.h"
#include "Intersection.h"
#include "UtilTypes.h"
#include "../Objects/Surfel.h"
#include "Octree.h"
#include "OctreeType.h"
#include "BoundingBox.h"

int createInitRays( struct Ray **rays, int width, int height, float growth, struct Camera cam );
int createDrawingRays( struct Ray **rays, int width, int height, struct Camera cam );
struct SurfelArray createSurfels( const struct Scene &scene, Ray *rays, int numRays );
struct Scene createSurfelSpheres( const struct Scene &scene, Ray *rays, int numRays );
void collectIntersections( const Scene &scene, const Ray &ray, IntersectionArray &IA );
struct TreeNode createSurfelTree( const struct Scene &scene, Ray *rays, int numRays );
void castRays( const struct TreeNode &scene, struct Ray *rays, int numRays, Color *buffer, int width );
Color raytrace( const struct TreeNode &tree, const Ray &ray, vec3 ***cuberay, glm::mat4 *cubtrans );

Color raytrace( const struct SurfelArray &scene, const Ray &ray );

TreeHitMark transTree( TreeNode root, const Ray &ray );

void pollTest( const TreeNode &tree, float angle, vec3 ***cuberay, glm::mat4 *cubetrans );

void traverseOctreeCPU( RasterCube &cube, const TreeNode &node, float maxangle, vec3 &position,
      vec3 &normal, vec3 ***cuberays, glm::mat4 *cubetrans);
void rasterizeClusterToCube( RasterCube &cube, Color &c, float area, vec3 nodePosition,
      glm::mat4 *cubetransforms, vec3 ***cuberays, vec3 &position, vec3 &normal);
void rasterizeSurfelToCube( RasterCube &cube, Surfel &surfel, glm::mat4 *cubetransforms,
      vec3 ***cuberays, vec3 &position, vec3 &normal );
void raytraceSurfelToCube( RasterCube &cube, Surfel &surfel, vec3 ***cuberays, vec3 &position,
      vec3 &normal );
float evaluateSphericalHermonicsArea( const TreeNode &node, vec3 &centerToEye );
Color evaluateSphericalHermonicsPower(const TreeNode &node, vec3 &centerToEye);
vec3 ***initCuberays( );
void initCubeTransforms( glm::mat4 **cubetrans );
glm::mat4 getViewPixelMatrix();
glm::mat4 getOrthMatrix();
glm::mat4 getProjectMatrix();
void tester( const struct TreeNode &tree, vec3 ***cuberay, glm::mat4 *cubetrans );
#endif
