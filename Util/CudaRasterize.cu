/**
 *  CPE 2013
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "vec3.h"
#include "RasterCube.h"
#include "BoundingBoxType.h"
#include "../Objects/SurfelType.h"
#include "cutil.h"
#include <cuda.h>
#include <curand_kernel.h>
#include "OctreeType.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define CUDASAFECALL( call )  CUDA_SAFE_CALL( call )
#define CUDAERRORCHECK() {                   \
   cudaError err = cudaGetLastError();        \
   if( cudaSuccess != err){ \
      printf("CudaErrorCheck %d\n", err);           \
      exit(1); \
   } }
#define FAR_PLANE -100.0
#define NEAR_PLANE -1.0
#define RIGHT 1.0
#define LEFT -1.0
#define TOP 1.0
#define BOTTOM -1.0
#define NPIXELS 8
#define PI 3.141592
#define THREADS 512
#define MAX_OCTREE_DEPTH 30
#define MAX_ANGLE 0.05

__device__ vec3 gpuCudaUnit(vec3 &in)
{
   float temp;
   vec3 newVector;
   newVector.x = 0;
   newVector.y = 0;
   newVector.z = 0;
   temp = sqrt( in.x*in.x + in.y*in.y + in.z*in.z );

   if(temp > 0)
   {
      newVector.x = in.x/temp;
      newVector.y = in.y/temp;
      newVector.z = in.z/temp;
   }
   return newVector;
}

__device__ vec3 gpuGetCenter( const BoundingBox &box )
{
   vec3 c;
   c.x = (box.max.x -box.min.x)/2 + box.min.x;
   c.y = (box.max.y -box.min.y)/2 + box.min.y;
   c.x = (box.max.z -box.min.z)/2 + box.min.z;
   return c;
}
__device__ bool gpuBBInTest( const BoundingBox &box, const vec3 &pos )
{
   if (pos.x >= box.max.x || pos.x < box.min.x )
      return false;
   if (pos.y >= box.max.y || pos.y < box.min.y )
      return false;
   if (pos.z >= box.max.z || pos.z < box.min.z )
      return false;
   return true;
}
__device__ float gpuCudaDot(const vec3 &one, const vec3 &two)
{
   return one.x*two.x + one.y*two.y + one.z*two.z;
}
__device__ float gpuDistance(const vec3 &one, const vec3 &two )
{
   return sqrt((one.x-two.x)*(one.x-two.x) + (one.y-two.y)*(one.y-two.y) +
         (one.z-two.z)*(one.z-two.z));
}
__device__ int gpuBelowHorizon( const BoundingBox &box, vec3 &position, vec3 &normal )
{
   vec3 points[8];
   points[0] = box.min;
   points[1] = box.min;
   points[1].z = box.max.z;
   points[2] = box.min;
   points[2].y = box.max.y;
   points[3] = box.min;
   points[3].y = box.max.y;
   points[3].z = box.max.z;
   points[4] = box.min;
   points[4].x = box.max.x;
   points[5] = box.min;
   points[5].x = box.max.x;
   points[5].z = box.max.z;
   points[6] = box.min;
   points[6].x = box.max.x;
   points[6].y = box.max.y;
   points[7] = box.max;
   int below = 0;
   for( int i = 0; i < 8; i++ )
   {
      vec3 temp;
      temp.x = points[i].x - position.x;
      temp.y = points[i].y - position.y;
      temp.z = points[i].z - position.z;
      temp = gpuCudaUnit( temp );
      if( gpuCudaDot( normal, temp ) <= 0.01 )
         below++;
   }
   return below;
}
__device__ void gpuGetWVecs( glm::vec4 ret[] )
{
   ret[0] = glm::vec4( 0.0, 0.0, -1.0, 0.0 );
   ret[1] = glm::vec4( -1.0, 0.0, 0.0, 0.0 );
   ret[2] = glm::vec4( 0.0, 0.0, 1.0, 0.0 );
   ret[3] = glm::vec4( 1.0, 0.0, 0.0, 0.0 );
   ret[4] = glm::vec4( 0.0, 1.0, 0.0, 0.0 );
   ret[5] = glm::vec4( 0.0, -1.0, 0.0, 0.0 );
}
__device__ glm::mat4 gpuGetM( )
{
   glm::mat4 pro = glm::mat4(1.0);
   pro[0] = glm::vec4( NEAR_PLANE, 0, 0, 0 );
   pro[1] = glm::vec4( 0, NEAR_PLANE, 0, 0 );
   pro[2] = glm::vec4( 0, 0, NEAR_PLANE + FAR_PLANE, - NEAR_PLANE * FAR_PLANE );
   pro[3] = glm::vec4( 0, 0, 1, 0 );
   pro = glm::transpose(pro);

   glm::mat4 orth = glm::mat4( 1.0 );
   orth[0] = glm::vec4( 2.0/(RIGHT - LEFT), 0, 0, -(RIGHT +LEFT)/(RIGHT -LEFT) );
   orth[1] = glm::vec4( 0, 2.0/(TOP-BOTTOM), 0, -(TOP + BOTTOM)/(TOP-BOTTOM) );
   orth[2] = glm::vec4( 0, 0, 2.0/(NEAR_PLANE - FAR_PLANE),
         -(NEAR_PLANE + FAR_PLANE)/(NEAR_PLANE - FAR_PLANE) );
   orth[3] = glm::vec4( 0, 0, 0, 1.0 );
   orth = glm::transpose(orth);

   glm::mat4 vp = glm::mat4(1.0);
   vp[0] = glm::vec4( NPIXELS/2.0, 0 ,0, (NPIXELS)/2.0 );
   vp[1] = glm::vec4( 0, -NPIXELS/2.0, 0, (NPIXELS)/2.0 );
   vp = glm::transpose(vp);
   return vp * orth * pro;
}

__device__ void gpuInitCubetrans( glm::mat4 cubetrans[] )
{
   glm::vec4 x = glm::vec4( 1.0, 0.0, 0.0, 0.0 );
   glm::vec4 y = glm::vec4( 0.0, 1.0, 0.0, 0.0 );
   glm::vec4 z = glm::vec4( 0.0, 0.0, 1.0, 0.0 );

   //front w = neg Z, u= neg x, v = pos y
   glm::mat4 *front = cubetrans;
   *front = glm::mat4(1.0); //build ident
   (*front)[0] = -x;
   (*front)[1] = y;
   (*front)[2] = -z;

   //right w = neg x, u = pos z, v pos y
   glm::mat4 *right = &(cubetrans[1]);
   *right = glm::mat4(1.0); //build ident
   (*right)[0] =z;
   (*right)[1] =y;
   (*right)[2] =-x;

   //back w = pos z, u=  pos x, v = pos y
   glm::mat4 *back = &(cubetrans[2]);
   *back = glm::mat4(1.0); //build ident
   (*back)[0] =x;
   (*back)[1] =y;
   (*back)[2] =z;

   //left w = pos x, u neg z, v pos y
   glm::mat4 *left = &(cubetrans[3]);
   *left = glm::mat4(1.0); //build ident
   (*left)[0] = -z;
   (*left)[1] = y;
   (*left)[2] = x;

   glm::mat4 *bottom = &(cubetrans[4]);
   *bottom = glm::mat4(1.0); //build ident
   (*bottom)[0] = -x;
   (*bottom)[1] = z;
   (*bottom)[2] = y;

   glm::mat4 *top = &(cubetrans[5]);
   *top = glm::mat4(1.0); //build ident
   (*top)[0] = -x;
   (*top)[1] = -z;
   (*top)[2] = -y;

   //Transpose so in col major
   for( int i =0 ; i < 6; i++ )
      cubetrans[i] = glm::transpose(cubetrans[i]);
}
__device__ void gpuInitCuberays( vec3 cuberays[] )
{
   float half = (2.0/8.0) / 2;
   //Front
   for( int i = 0; i < 8; i++ )
   {
      for( int j =0; j < 8; j++ )
      {
         vec3 ray;
         //front
         ray.x = 1 - (2.0/8.0) * j - half;
         ray.y = 1 - (2.0/8.0) * i - half;
         ray.z = 1;
         cuberays[i*8 +j] = gpuCudaUnit(ray);
         //right
         ray.x = 1;
         ray.y = 1 - (2.0/8.0) * i - half;
         ray.z = -1 + (2.0/8.0) * j + half;
         cuberays[1*8*8+i*8 + j] = gpuCudaUnit(ray);
         //back
         ray.x = -1 + (2.0/8.0) * j +half;
         ray.y = 1 - (2.0/8.0) * i - half;
         ray.z = -1;
         cuberays[2*8*8 + i*8 +j] = gpuCudaUnit(ray);
         //left
         ray.x = -1;
         ray.y = 1 - (2.0/8.0) *i - half;
         ray.z = 1 - (2.0/8.0) * j - half;
         cuberays[3*8*8+i*8+j] = gpuCudaUnit(ray);
         //bottom
         ray.x = 1 - (2.0/8.0)  * j - half;
         ray.y = -1;
         ray.z = 1 - (2.0/8.0) * i - half;
         cuberays[4*8*8+i*8+j] = gpuCudaUnit(ray);
         //top
         ray.x = 1 - (2.0/8.0) * j -half;
         ray.y = 1;
         ray.z = -1 + (2.0/8.0) * i +half;
         cuberays[5*8*8+i*8+j] = gpuCudaUnit(ray);
      }
   }
}
__device__ void gpuGetAxisAlinedPoints(glm::vec4 ret[], vec3 position, float len, int k )
{
   if( k == 0 || k == 2 ) // front and back: x,y pin z
   {
      ret[0] = glm::vec4( position.x - len, position.y + len, position.z, 1.0 );
      ret[1] = glm::vec4( position.x + len, position.y + len, position.z, 1.0 );
      ret[2] = glm::vec4( position.x + len, position.y - len, position.z, 1.0 );
      ret[3] = glm::vec4( position.x - len, position.y - len, position.z, 1.0 );
   }
   else if ( k == 1 || k == 3 ) //right and left: y,z pin x
   {
      ret[0] = glm::vec4( position.x, position.y + len, position.z - len, 1.0 );
      ret[1] = glm::vec4( position.x, position.y + len, position.z + len, 1.0 );
      ret[2] = glm::vec4( position.x, position.y - len, position.z + len, 1.0 );
      ret[3] = glm::vec4( position.x, position.y - len, position.z - len, 1.0 );
   }
   else //top and bottom: x,z pin y
   {
      ret[0] = glm::vec4( position.x - len, position.y, position.z + len, 1.0 );
      ret[1] = glm::vec4( position.x + len, position.y, position.z + len, 1.0 );
      ret[2] = glm::vec4( position.x + len, position.y, position.z - len, 1.0 );
      ret[3] = glm::vec4( position.x - len, position.y, position.z - len, 1.0 );
   }
}
__device__ float squareDistanceGPU( vec3 &one, vec3 &two )
{
   return ((one.x-two.x)*(one.x-two.x) + (one.y-two.y)*(one.y-two.y) + (one.z-two.z)*(one.z-two.z));
}

__device__ float surfelHitTestGPU( Surfel s, Ray &ray )
{
   vec3 direction = gpuCudaUnit(ray.dir);
   vec3 position;
   vec3 normal = gpuCudaUnit(s.normal);

   position.x = ray.pos.x;
   position.y = ray.pos.y;
   position.z = ray.pos.z;

   float vd = gpuCudaDot(normal, direction);
   if( vd > 0.001)
      return -1;
   float v0 = -(gpuCudaDot(position, normal) + s.distance );
   float t = v0/vd;
   if( t < 0.01)
      return -1;

   vec3 hitMark;
   hitMark.x = ray.pos.x + direction.x*t;
   hitMark.y = ray.pos.y + direction.y*t;
   hitMark.z = ray.pos.z + direction.z*t;
   float d = squareDistanceGPU( hitMark, s.pos );

   if( d < s.radius*s.radius )
      return t;
   else
      return -1;
}
__device__ float gpuEvaluateSphericalHermonicsArea( const CudaNode &node, vec3 &t )
{
   double TYlm[9];
   TYlm[0] = 0.282095; //0 0
   TYlm[1] = .488603 * -t.y;//1 -1
   TYlm[2] = .488603 * t.z;//1 0
   TYlm[3] = .488603 * -t.x; //1 1
   TYlm[4] = 1.092548 * t.x * t.y; // 2 -2
   TYlm[5] = 1.092548 * -t.y * t.z; //2 -1
   TYlm[6] = 0.315392 * (3*t.z*t.z - 1); //2 0
   TYlm[7] = 1.092548 * -t.x * t.z; //2 1
   TYlm[8] = .546274 * (t.x*t.x - t.y*t.y); //2 2

   float area = 0;

   for( int i =0; i < 9; i++ )
   {
      area += node.hermonics.area[i] * TYlm[i];
   }
   area = fmax(area, 0);
   return area;
}

__device__ float gpuEvaluateSphericalHermonicsAreaAll( const CudaNode &node, const BoundingBox &box,
      const vec3 &position )
{
   vec3 points[8];
   points[0] = box.min;
   points[1] = box.min;
   points[1].z = box.max.z;
   points[2] = box.min;
   points[2].y = box.max.y;
   points[3] = box.min;
   points[3].y = box.max.y;
   points[3].z = box.max.z;
   points[4] = box.min;
   points[4].x = box.max.x;
   points[5] = box.min;
   points[5].x = box.max.x;
   points[5].z = box.max.z;
   points[6] = box.min;
   points[6].x = box.max.x;
   points[6].y = box.max.y;
   points[7] = box.max;
   float area = 0;

   for(int i =0; i < 8; i++)
   {
      vec3 cte;
      cte.x = position.x - points[i].x;
      cte.y = position.y - points[i].y;
      cte.z = position.z - points[i].z;
      cte = gpuCudaUnit(cte);

      float t = gpuEvaluateSphericalHermonicsArea( node, cte );
      if (area < t )
         area = t;
   }
   return area;
}
__device__ Color gpuEvaluateSphericalHermonicsPower( const CudaNode &node, vec3 &t )
{
   double TYlm[9];
   TYlm[0] = 0.282095; //0 0
   TYlm[1] = .488603 * -t.y;//1 -1
   TYlm[2] = .488603 * t.z;//1 0
   TYlm[3] = .488603 * -t.x; //1 1
   TYlm[4] = 1.092548 * t.x * t.y; // 2 -2
   TYlm[5] = 1.092548 * -t.y * t.z; //2 -1
   TYlm[6] = 0.315392 * (3*t.z*t.z - 1); //2 0
   TYlm[7] = 1.092548 * -t.x * t.z; //2 1
   TYlm[8] = .546274 * (t.x*t.x - t.y*t.y); //2 2
   Color color;
   color.r = 0;
   color.g = 0;
   color.b = 0;

   for( int i =0; i < 9; i++ )
   {
      color.r += node.hermonics.red[i] * TYlm[i];
      color.g += node.hermonics.green[i] * TYlm[i];
      color.b += node.hermonics.blue[i] * TYlm[i];
   }
   color.r = fmax( color.r, 0);
   color.g = fmax( color.g, 0);
   color.b = fmax( color.b, 0);
   return color;
}


__device__ void gpuRaytraceSurfelToCube( RasterCube &cube, Surfel &surfel, vec3 ***cuberays, vec3 &position,
      vec3 &normal )
{
   glm::vec4 wVecs[4];
   gpuGetWVecs( wVecs );

   for( int i = 0; i < 6; i++ )
   {
      for( int j = 0; j < 8; j ++ )
      {
         for( int k = 0; k < 8; k++ )
         {
            if( cube.depth[i][j][k] > 0 )
            {
               Ray ray;
               ray.dir = cuberays[i][j][k];
               ray.pos = position;

               float t = surfelHitTestGPU( surfel, ray );
               if( t > 0 && t < cube.depth[i][j][k] )
               {
                  vec3 hit;
                  hit.x = ray.pos.x + ray.dir.x * t;
                  hit.y = ray.pos.y + ray.dir.y * t;
                  hit.z = ray.pos.z + ray.dir.z * t;
                  vec3 diff;
                  diff.x = position.x - hit.x;
                  diff.y = position.y - hit.y;
                  diff.z = position.z - hit.z;
                  diff = gpuCudaUnit(diff);
                  float dis = gpuDistance( hit, position );

                  cube.depth[i][j][k] = dis;
                  cube.sides[i][j][k].r = surfel.color.r;
                  cube.sides[i][j][k].g = surfel.color.g;
                  cube.sides[i][j][k].b = surfel.color.b;
               }
            }
         }
      }
   }
}
__device__ void gpuRasterizeSurfelToCube( RasterCube &cube, Surfel &surfel, glm::mat4 *cubetransforms,
      vec3 ***cuberays, vec3 &position, vec3 &normal, float dis )
{
   glm::vec4 wVecs[4];
   gpuGetWVecs( wVecs );

   vec3 diff;
   diff.x = position.x - surfel.pos.x;
   diff.y = position.y - surfel.pos.y;
   diff.z = position.z - surfel.pos.z;
   diff = gpuCudaUnit(diff);
   float dotPro = gpuCudaDot(normal, diff);
   if( dotPro > 0 )
      return;
   double area = surfel.radius *surfel.radius * PI;

   double areas[6];
   for( int i =0; i < 6; i++ )
   {
      vec3 w;
      w.x = wVecs[i][0];
      w.y = wVecs[i][1];
      w.z = wVecs[i][2];
      if(  gpuCudaDot( w, diff ) > 0 && gpuCudaDot( diff, surfel.normal ) > 0 )
         areas[i] = area * gpuCudaDot( diff, surfel.normal );// * dot(w, diff);
      else
         areas[i] = 0;

   }

   for( int k = 0; k< 6; k++ )
   {
      if( areas[k] < 0.00001 )
         continue;
      double length = sqrt( areas[k] );
      glm::vec4 points[4];
      gpuGetAxisAlinedPoints( points, surfel.pos, length/2.0, k );
      points[0] = cubetransforms[k] * points[0];
      points[1] = cubetransforms[k] * points[1];
      points[2] = cubetransforms[k] * points[2];
      points[3] = cubetransforms[k] * points[3];
      for( int i = 0; i < 4; i++ )
      {
         points[i][0] /= points[i][3];
         points[i][1] /= points[i][3];
         points[i][2] /= points[i][3];
         points[i][3] = 1;
      }
      int minX = 0;
      int minY = 0;
      int maxX = 0;
      int maxY = 0;
      minX = points[0][0];
      maxX = points[0][0];
      minY = points[0][1];
      maxY = points[0][1];
      for( int i = 1; i < 4; i++ )
      {
         minX = min( minX, (int)roundf(points[i][0] +0.5) );
         minY = min( minY, (int)roundf(points[i][1] +0.5) );
         maxX = max( maxX, (int)roundf(points[i][0] +0.5) );
         maxY = max( maxY, (int)roundf(points[i][1] +0.5) );
      }
      if( !(maxX < 0 || maxY < 0 || minY > 7 || minX > 7 ))
      {
         minX = max( minX, 0 );
         minY = max( minY, 0 );
         maxX = min( maxX, 7 );
         maxY = min( maxY, 7 );
         for( int i = minY; i <= maxY; i++ )
         {
            for( int j = minX; j <= maxX; j++ )
            {
               if (cube.depth[k][i][j] < 0)
                  continue;
               else if ( dis < cube.depth[k][i][j] )
               {
                  cube.sides[k][i][j].r = surfel.color.r;
                  cube.sides[k][i][j].g = surfel.color.g;
                  cube.sides[k][i][j].b = surfel.color.b;
                  cube.depth[k][i][j] = dis;
               }
            }
         }
      }
   }
}
__device__ void gpuRasterizeClusterToCube( RasterCube &cube, Color &c, float area, vec3 nodePosition,
      glm::mat4 *cubetransforms, vec3 ***cuberays, vec3 &position, vec3 &normal, float dis )
{
   glm::vec4 wVecs[4];
   gpuGetWVecs( wVecs );

   vec3 check;
   check.x = position.x - nodePosition.x;
   check.y = position.y - nodePosition.y;
   check.z = position.z - nodePosition.z;
   check = gpuCudaUnit(check);
   if( gpuCudaDot(check, normal) > -0.001 )
      return;
   float areas[6];
   for( int i =0; i < 6; i++ )
   {
      vec3 w;
      w.x = wVecs[i][0];
      w.y = wVecs[i][1];
      w.z = wVecs[i][2];
      if( gpuCudaDot( w, check ) > 0 )
         areas[i] = gpuCudaDot( w, check) * area;
      else
         areas[i] = 0;
   }

   for( int k = 0; k< 6; k++ )
   {
      if( areas[k] <= 0 )
         continue;
      float length = sqrtf(areas[k]);

      glm::vec4 points[4];
      gpuGetAxisAlinedPoints( points, nodePosition, length/2.0, k );

      points[0] = (cubetransforms[k]) * points[0];
      points[1] = (cubetransforms[k]) * points[1];
      points[2] = (cubetransforms[k]) * points[2];
      points[3] = (cubetransforms[k]) * points[3];

      for( int i = 0; i < 4; i++ )
      {
         points[i][0] /= points[i][3];
         points[i][1] /= points[i][3];
         points[i][2] /= points[i][3];
         points[i][3] = 1;
      }
      int minX = 0;
      int minY = 0;
      int maxX = 0;
      int maxY = 0;
      minX = points[0][0];
      maxX = points[0][0];
      minY = points[0][1];
      maxY = points[0][1];
      float fminx = points[0][0];
      float fmaxx = fminx;
      float fminy = points[0][1];
      float fmaxy = fminy;
      for( int i = 1; i < 4; i++ )
      {
         fminx = fmin( fminx, points[i][0] );
         fmaxx = fmax( fmaxx, points[i][0] );
         fminy = fmin( fminy, points[i][1] );
         fminy = fmin( fmaxy, points[i][1] );
         if( minX > points[i][0] )
            minX = roundf(points[i][0]+0.5);
         if( minY > points[i][1] )
            minY = roundf(points[i][1]+0.5);
         if( maxX < points[i][0] )
            maxX = roundf(points[i][0]+0.5);
         if( maxY < points[i][1] )
            maxY = roundf(points[i][1]+0.5);
      }
      if( fmaxx - fminx < 0.5 || fmaxy - fminy < 0.5 )
      {
         continue;
      }
      if( !(maxX < 0 || maxY < 0 || minY > 7 || minX > 7 ))
      {
         minX = max( minX, 0 );
         minY = max( minY, 0 );
         maxX = min( maxX, 7 );
         maxY = min( maxY, 7 );

         for( int i = minY; i <= maxY; i++ )
         {
            for( int j = minX; j <= maxX; j++ )
            {
               if (cube.depth[k][i][j] < 0)
                  continue;
               else if ( dis < cube.depth[k][i][j] )
               {
                  cube.sides[k][i][j].r = c.r;
                  cube.sides[k][i][j].g = c.g;
                  cube.sides[k][i][j].b = c.b;
                  cube.depth[k][i][j] = dis;
               }
            }
         }
      }
   }
}

__global__ void kernel_SharedSurfelBleed( Surfel *surfels, int num_surfels, vec3 *positions,
      vec3 *normals, Color *indirect, int num, int batch, int batch_size )
{
   int index = (blockIdx.x + batch*batch_size)*THREADS + threadIdx.x;

   vec3 position;
   vec3 normal;
   if( index < num || threadIdx.x < THREADS )
   {
      position = positions[index];
      normal = normals[index];
   }
   float ndotr;
   vec3 cr[6*8*8];
   glm::mat4 ct[6];
   gpuInitCuberays( cr );
   gpuInitCubetrans( ct );
   glm::mat4 eyeTrans = glm::mat4(1.0);
   eyeTrans[3][0] = -position.x;
   eyeTrans[3][1] = -position.y;
   eyeTrans[3][2] = -position.z;
   glm::mat4 M = gpuGetM();
   for( int i = 0; i < 6; i++ )
      ct[i] = M * ct[i] * eyeTrans;


   Color black;
   black.r = 0;
   black.g = 0;
   black.b = 0;
   RasterCube cube;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            ndotr = gpuCudaDot(normal, cr[i*8*8+j*8+k]);
            if( ndotr < 0.001 )
            {
               cube.sides[i][j][k] = black;
               cube.depth[i][j][k] = -1;
            }
            else {
               cube.sides[i][j][k] = black;
               cube.depth[i][j][k] = 100+1;
            }
         }

   int loop_count = ceilf( (float)num_surfels / THREADS );
   __shared__ Surfel s_block[THREADS];
   Surfel current;
   for( int i =0; i < loop_count; i++ )
   {
      if( i*THREADS + threadIdx.x < num_surfels )
         s_block[threadIdx.x] = surfels[i*THREADS + threadIdx.x];
      __syncthreads();

      for( int j =0; j < THREADS; j++ )
      {
         if( index < num && i*THREADS + j < num_surfels )
         {
            current = s_block[j];
            float dis = gpuDistance( position, current.pos );
            if( dis > current.radius )
               gpuRasterizeSurfelToCube( cube, current, ct, (vec3 ***)cr, position, normal, dis );
         }
      }
      __syncthreads();
   }
   Color color;
   color.r = 0;
   color.g =0;
   color.b =0;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            if( cube.depth[i][j][k] < 0 )
               continue;
            num++;
            if( cube.depth[i][j][k] < -FAR_PLANE +1 )
            {
               float dotProd = gpuCudaDot(normal, cr[i*8*8+j*8+k]);
               color.r += (cube.sides[i][j][k].r*dotProd
                     /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
               color.g += (cube.sides[i][j][k].g*dotProd
                     /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
               color.b += (cube.sides[i][j][k].b*dotProd
                     /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
            }
         }

   indirect[index] = color;
   return;
}

__global__ void kernel_SurfelBleed( Surfel *surfels, int num_surfels, vec3 *positions,
      vec3 *normals, Color *indirect, int num, int batch, int batch_size )
{
   int index = (blockIdx.x + batch*batch_size)*THREADS + threadIdx.x;

   if( index > num || threadIdx.x > THREADS )
      return;

   vec3 position = positions[index];
   vec3 normal = normals[index];
   float ndotr;
   vec3 cr[6*8*8];
   glm::mat4 ct[6];
   gpuInitCuberays( cr );
   gpuInitCubetrans( ct );
   glm::mat4 eyeTrans = glm::mat4(1.0);
   eyeTrans[3][0] = -position.x;
   eyeTrans[3][1] = -position.y;
   eyeTrans[3][2] = -position.z;
   glm::mat4 M = gpuGetM();
   for( int i = 0; i < 6; i++ )
      ct[i] = M * ct[i] * eyeTrans;


   Color black;
   black.r = 0;
   black.g = 0;
   black.b = 0;
   RasterCube cube;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            ndotr = gpuCudaDot(normal, cr[i*8*8+j*8+k]);
            if( ndotr < 0.001 )
            {
               cube.sides[i][j][k] = black;
               cube.depth[i][j][k] = -1;
            }
            else {
               cube.sides[i][j][k] = black;
               cube.depth[i][j][k] = 100+1;
            }
         }

   Surfel current;
   for( int i =0; i < num_surfels; i++ )
   {
      current = surfels[i];
      float dis = gpuDistance( position, current.pos );
      if( dis > current.radius )
         gpuRasterizeSurfelToCube( cube, current, ct, (vec3 ***)cr, position, normal, dis );
   }
   Color color;
   color.r = 0;
   color.g =0;
   color.b =0;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            if( cube.depth[i][j][k] < 0 )
               continue;
            num++;
            if( cube.depth[i][j][k] < -FAR_PLANE +1 )
            {
               float dotProd = gpuCudaDot(normal, cr[i*8*8+j*8+k]);
               color.r += (cube.sides[i][j][k].r*dotProd
                     /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
               color.g += (cube.sides[i][j][k].g*dotProd
                     /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
               color.b += (cube.sides[i][j][k].b*dotProd
                     /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
            }
         }

   indirect[index] = color;
   return;
}

extern "C" Color *gpuSurfelColorBleeding( SurfelArray cpu_array, vec3 *positions, vec3 *normals,
      int num )
{
   printf("GPU Surfel only Bleeding: Surfels %d, Points %d\n", cpu_array.num, num );
   float surfel_size = (float)(sizeof(Surfel) * cpu_array.num)/1048576.0;
   float positions_size = (float)(sizeof(vec3) * num)/1048576.0;
   float indirect_size = (float)(sizeof(Color)*num)/1048576.0;
   printf("Sizes Surfels: %f, Points: %f, indirect %f, Total: %f\n", surfel_size, positions_size*2,
         indirect_size, surfel_size + positions_size*2 );

   Surfel *d_surfels;
   vec3 *d_positions;
   vec3 *d_normals;
   Color *d_indirect;

   Color *indirect = (Color *)malloc( sizeof(Color) * num );
   CUDASAFECALL(cudaMalloc( (void **)&d_surfels, sizeof(Surfel) * cpu_array.num));
   CUDASAFECALL(cudaMalloc( (void **)&d_positions, sizeof(vec3) * num));
   CUDASAFECALL(cudaMalloc( (void **)&d_normals, sizeof(vec3) * num));
   CUDASAFECALL(cudaMalloc( (void **)&d_indirect, sizeof(Color) * num));

   CUDASAFECALL(cudaMemcpy( d_surfels, cpu_array.array, sizeof(Surfel) * cpu_array.num,
            cudaMemcpyHostToDevice));
   CUDASAFECALL(cudaMemcpy( d_positions, positions, sizeof(vec3) * num,
            cudaMemcpyHostToDevice));
   CUDASAFECALL(cudaMemcpy( d_normals, normals, sizeof(vec3) * num,
            cudaMemcpyHostToDevice));

   int num_blocks = ceilf( (float)num /THREADS );
   /*
      int batch_size = 1;
      int batches = ceilf( (float)num_blocks/batch_size );
    */
   dim3 dimBlock( THREADS );
   dim3 dimGrid( num_blocks );

   printf("GPU Surfel only Bleeding\n");
   kernel_SharedSurfelBleed<<<dimGrid, dimBlock>>>( d_surfels, cpu_array.num, d_positions,
         d_normals, d_indirect, num, 0, 0 );
   CUDAERRORCHECK();

   CUDASAFECALL(cudaMemcpy( indirect, d_indirect, sizeof(Color) * num,
            cudaMemcpyDeviceToHost ));
   cudaFree( d_surfels );
   cudaFree( d_positions );
   cudaFree( d_normals );
   cudaFree( d_indirect );
   return indirect;
}

__device__ void gpuTraverseOctreeStack( RasterCube &cube, CudaNode *gpu_root, Surfel *gpu_array,
      vec3 &position, vec3 normal, vec3 ***cuberays, glm::mat4 *cubetransforms )
{
   float dis = 0;
   int stack[MAX_OCTREE_DEPTH * 8];
   int pointer = 0;
   stack[pointer] = 0;
   pointer++;

   while( pointer )
   {
      pointer--;
      int current = stack[pointer];

      CudaNode node = gpu_root[current];
      if( node.leaf )
      {
         for( int i = node.children[0]; i < node.children[1]; i++ )
         {
            Surfel s = gpu_array[i];
            dis = gpuDistance( position, s.pos );
            if ( dis < s.radius)
            {
               gpuRaytraceSurfelToCube( cube, s, cuberays, position, normal );
            }
            else
            {
               gpuRasterizeSurfelToCube( cube, s, cubetransforms, cuberays,
                     position, normal, dis );
            }
         }
      }
      else
      {
         if( gpuBBInTest( node.box, position ) )
         {
            for(int i = 7; i <= 0; i-- )
            {
               stack[pointer] = node.children[i];
               pointer++;
            }
            continue;
         }
         int horizon = gpuBelowHorizon( node.box, position, normal );
         if( horizon == 8 )
            continue;

         vec3 center = gpuGetCenter(node.box);

         vec3 centerToEye;
         centerToEye.x = position.x - center.x;
         centerToEye.y = position.y - center.y;
         centerToEye.z = position.z - center.z;
         centerToEye = gpuCudaUnit(centerToEye);

         dis = gpuDistance( position, center );
         float area = gpuEvaluateSphericalHermonicsAreaAll( node, node.box, centerToEye );
         float solidangle = area / (dis * dis);
         if( solidangle < MAX_ANGLE )
         {
            Color c = gpuEvaluateSphericalHermonicsPower( node, centerToEye );
            gpuRasterizeClusterToCube( cube, c, area, center, cubetransforms, cuberays,
                  position, normal, dis );
            continue;
         }
         else
         {
            for(int i = 7; i <= 0; i-- )
            {
               stack[pointer] = node.children[i];
               pointer++;
            }
            continue;
         }
      }
   }
}
