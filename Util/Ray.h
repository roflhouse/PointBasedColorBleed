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
#include "UtilTypes.h"
#include "../Objects/Objects.h"
#include "RayType.h"
#include "Scene.h"
#include "Intersection.h"
#include "UtilTypes.h"
#include "../Objects/Surfel.h"
#include "Octree.h"
#include "OctreeType.h"
#include "BoundingBox.h"

glm::vec4 *getWVecs( );
Intersection raytraceVisionRay( const Scene &scene, const Ray &ray );
int createInitRays( struct Ray **rays, int width, int height, float growth, struct Camera cam );
int createDrawingRays( struct Ray **rays, int width, int height, struct Camera cam );
struct SurfelArray createSurfels( const struct Scene &scene, Ray *rays, int numRays );
struct Scene createSurfelSpheres( const struct Scene &scene, Ray *rays, int numRays );
void collectIntersections( const Scene &scene, const Ray &ray, IntersectionArray &IA, const Ray &sec, const Ray &third1 );
struct TreeNode createSurfelTree( const struct Scene &scene, Ray *rays, int numRays, int width );
void createCudaSurfelTree( const Scene &scene, Ray *rays, int numRays, int width, CudaNode* &gpu_root,
      int &nodes, SurfelArray &gpu_array );
void castRays( const struct TreeNode &scene, struct Ray *rays, int numRays, Color *buffer, int width );
void castRays( CudaNode *cpu_root, int nodes, SurfelArray cpu_array, Ray *rays, int number,
      Color *buffer, int width);
void castRaysCPU( CudaNode *cpu_root, int nodes, SurfelArray cpu_array, Ray *rays, int number,
      Color *buffer, int width);
Color raytrace( struct CudaNode *root, SurfelArray surfels, const Ray &ray,
      vec3 ***cuberay, glm::mat4 *cubetrans );
Color raytrace( const struct TreeNode &tree, const Ray &ray, vec3 ***cuberay, glm::mat4 *cubtrans );

Color raytrace( const struct SurfelArray &scene, const Ray &ray );

TreeHitMark transTree( TreeNode root, const Ray &ray );
TreeHitMark transTreeCPU( CudaNode *cpu_root, int current, SurfelArray cpu_array, const Ray &ray );

void pollTest( const TreeNode &tree, float angle, vec3 ***cuberay, glm::mat4 *cubetrans );

void traverseOctreeCPU( RasterCube &cube, const TreeNode &node, float maxangle, vec3 &position,
      vec3 &normal, vec3 ***cuberays, glm::mat4 *cubetrans);
void traverseOctreeCPU( RasterCube &cube, CudaNode *cpu_root, int current, SurfelArray &cpu_array,
      float maxangle, vec3 &position, vec3 &normal, vec3 ***cuberays, glm::mat4 *cubetransforms, bool above);
void rasterizeClusterToCube( RasterCube &cube, Color &c, float area, vec3 nodePosition,
      glm::mat4 *cubetransforms, vec3 ***cuberays, vec3 &position, vec3 &normal, float dis);
void rasterizeSurfelToCube( RasterCube &cube, Surfel &surfel, glm::mat4 *cubetransforms,
      vec3 ***cuberays, vec3 &position, vec3 &normal);
void raytraceSurfelToCube( RasterCube &cube, Surfel &surfel, vec3 ***cuberays, vec3 &position,
      vec3 &normal );
float evaluateSphericalHermonicsArea( const TreeNode &node, vec3 &centerToEye );
Color evaluateSphericalHermonicsPower(const TreeNode &node, vec3 &centerToEye);
float evaluateSphericalHermonicsArea( const CudaNode &node, vec3 &centerToEye );
Color evaluateSphericalHermonicsPower(const CudaNode &node, vec3 &centerToEye);
vec3 ***initCuberays( );
void initCubeTransforms( glm::mat4 **cubetrans );
glm::mat4 getViewPixelMatrix();
glm::mat4 getOrthMatrix();
glm::mat4 getProjectMatrix();
void castRays( Scene scene, CudaNode *cpu_root, int nodes, SurfelArray cpu_array,
      Ray *rays, int number,
      Color *buffer, int width);
float evaluateSphericalHermonicsAreaAll( const CudaNode &node, const BoundingBox &box, 
      const vec3 &position );
void castRaysGPUSurfels( Scene scene, CudaNode *cpu_root, int nodes, SurfelArray cpu_array,
      Ray *rays, int number, Color *buffer, int width);
#endif
