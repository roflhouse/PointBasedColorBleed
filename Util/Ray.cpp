/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Ray.h"
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include "Tga.h"
#include "UtilTypes.h"

#define PI 3.141592
#define MAXDEPTH 15
#define MAX_ANGLE 0.03
#define FAR_PLANE -100.0
#define NEAR_PLANE -1.0
#define RIGHT 1.0
#define LEFT -1.0
#define TOP 1.0
#define BOTTOM -1.0
#define NPIXELS 8
#define NO_BLEED 0

void displayRasterCube( RasterCube &cube, int num );
extern "C" Surfel *gpuCastRays( CudaNode *cpu_root, int nodes, SurfelArray cpu_array,
      Ray *rays, int num_rays );

int createDrawingRaysTest( Ray **rays, int width, int height )
{
   float rightUnitX = 1;
   float rightUnitY = 0;
   float rightUnitZ = 0;
   float upUnitX = 0;
   float upUnitY = 1;
   float upUnitZ = 0;
   vec3 lookat;
   lookat.x = 0;
   lookat.y = 0;
   lookat.z = 0;
   vec3 pos;
   pos.x =0;
   pos.y =0;
   pos.z = 3;
   vec3 uv = unit( newDirection(lookat, pos) );

   float l = -.5;
   float r = .5;
   float t = .5;
   float b = -0.5;
   *rays = (Ray *) malloc( sizeof(Ray) *height*width );
   for( int i = 0; i < height; i++)
   {
      for( int j = 0; j < width ; j ++ )
      {
         float u = l + (r-l)*( (float)j)/(float)width;
         float v = b + (t-b)*( (float)i)/(float)height;
         float w = 1;
         int c = i*width + j;

         (*rays)[c].pos = pos;
         (*rays)[c].dir.x = u * rightUnitX + v * upUnitX + w * uv.x;
         (*rays)[c].dir.y = u * rightUnitY + v * upUnitY + w * uv.y;
         (*rays)[c].dir.z = u * rightUnitZ + v * upUnitZ + w * uv.z;
         (*rays)[c].dir = unit((*rays)[c].dir);
         (*rays)[c].i = i;
         (*rays)[c].j = j;
      }
   }
   return width * height;
}


int createInitRays( Ray **rays, int width, int height, float growth, Camera cam )
{
   width *= 1;
   height *= 1;
   vec3 right = unit(cam.right);
   vec3 up = unit(cam.up);
   float rightUnitX = right.x;
   float rightUnitY = right.y;
   float rightUnitZ = right.z;
   float upUnitX = up.x;
   float upUnitY = up.y;
   float upUnitZ = up.z;
   vec3 uv = unit(newDirection(cam.lookat, cam.pos));

   *rays = (Ray *) malloc( sizeof(Ray) *height*width );
   for( int i = 0; i < height; i++)
   {
      for( int j = 0; j < width ; j ++ )
      {
         float r = (float)rand() / (float)RAND_MAX;
         float u = cam.l + (cam.r-cam.l)*((float)j+r)/(float)width;
         r = (float)rand() / (float)RAND_MAX;
         float v = cam.b + (cam.t-cam.b)*((float)i+r)/(float)height;
         float w = 1;
         int c = i*width + j;

         (*rays)[c].pos = cam.pos;
         (*rays)[c].dir.x = growth*u * rightUnitX + growth * v * upUnitX + w * uv.x;
         (*rays)[c].dir.y = growth*u * rightUnitY + growth * v * upUnitY + w * uv.y;
         (*rays)[c].dir.z = growth*u * rightUnitZ + growth * v * upUnitZ + w * uv.z;
         (*rays)[c].i = i;
         (*rays)[c].j = j;
      }
   }
   return width * height;
}
int createDrawingRays( Ray **rays, int width, int height, Camera cam )
{
   vec3 right = unit(cam.right);
   vec3 up = unit(cam.up);
   float rightUnitX = right.x;
   float rightUnitY = right.y;
   float rightUnitZ = right.z;
   float upUnitX = up.x;
   float upUnitY = up.y;
   float upUnitZ = up.z;
   vec3 uv = unit(newDirection(cam.lookat, cam.pos));

   *rays = (Ray *) malloc( sizeof(Ray) *height*width );
   for( int i = 0; i < height; i++)
   {
      for( int j = 0; j < width ; j ++ )
      {
         float u = cam.l + (cam.r-cam.l)*( (float)j)/(float)width;
         float v = cam.b + (cam.t-cam.b)*( (float)i)/(float)height;
         float w = 1;
         int c = i*width + j;

         (*rays)[c].pos = cam.pos;
         (*rays)[c].dir.x = u * rightUnitX + v * upUnitX + w * uv.x;
         (*rays)[c].dir.y = u * rightUnitY + v * upUnitY + w * uv.y;
         (*rays)[c].dir.z = u * rightUnitZ + v * upUnitZ + w * uv.z;
         (*rays)[c].dir = unit((*rays)[c].dir);
         (*rays)[c].i = i;
         (*rays)[c].j = j;
      }
   }
   return width * height;
}
vec3 ***initCuberays( )
{
   vec3 ***cuberays = (vec3 ***)malloc( sizeof(vec3 **) * 6 );
   for(int i = 0; i < 6; i++ )
   {
      cuberays[i] = (vec3 **)malloc( sizeof(vec3 *) * 8 );
      for( int j = 0; j< 8; j++ )
      {
         cuberays[i][j] = (vec3 *)malloc( sizeof(vec3) * 8 );
      }
   }
   //cube size does not matter.

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
         cuberays[0][i][j] = unit(ray);
         //right
         ray.x = 1;
         ray.y = 1 - (2.0/8.0) * i - half;
         ray.z = -1 + (2.0/8.0) * j + half;
         cuberays[1][i][j] = unit(ray);
         //back
         ray.x = -1 + (2.0/8.0) * j +half;
         ray.y = 1 - (2.0/8.0) * i - half;
         ray.z = -1;
         cuberays[2][i][j] = unit(ray);
         //left
         ray.x = -1;
         ray.y = 1 - (2.0/8.0) *i - half;
         ray.z = 1 - (2.0/8.0) * j - half;
         cuberays[3][i][j] = unit(ray);
         //bottom
         ray.x = 1 - (2.0/8.0)  * j - half;
         ray.y = -1;
         ray.z = 1 - (2.0/8.0) * i - half;
         cuberays[4][i][j] = unit(ray);
         //top
         ray.x = 1 - (2.0/8.0) * j -half;
         ray.y = 1;
         ray.z = -1 + (2.0/8.0) * i +half;
         cuberays[5][i][j] = unit(ray);
      }
   }
   return cuberays;
}
void initCubeTransforms( glm::mat4 **cubetrans )
{
   glm::vec4 x = glm::vec4( 1.0, 0.0, 0.0, 0.0 );
   glm::vec4 y = glm::vec4( 0.0, 1.0, 0.0, 0.0 );
   glm::vec4 z = glm::vec4( 0.0, 0.0, 1.0, 0.0 );
   *cubetrans = new glm::mat4[6];

   //front w = neg Z, u= neg x, v = pos y
   glm::mat4 *front = (*cubetrans);
   *front = glm::mat4(1.0); //build ident
   (*front)[0] = -x;
   (*front)[1] = y;
   (*front)[2] = -z;

   //right w = neg x, u = pos z, v pos y
   glm::mat4 *right = &((*cubetrans)[1]);
   *right = glm::mat4(1.0); //build ident
   (*right)[0] =z;
   (*right)[1] =y;
   (*right)[2] =-x;

   //back w = pos z, u=  pos x, v = pos y
   glm::mat4 *back = &((*cubetrans)[2]);
   *back = glm::mat4(1.0); //build ident
   (*back)[0] =x;
   (*back)[1] =y;
   (*back)[2] =z;

   //left w = pos x, u neg z, v pos y
   glm::mat4 *left = &((*cubetrans)[3]);
   *left = glm::mat4(1.0); //build ident
   (*left)[0] = -z;
   (*left)[1] = y;
   (*left)[2] = x;

   glm::mat4 *bottom = &((*cubetrans)[4]);
   *bottom = glm::mat4(1.0); //build ident
   (*bottom)[0] = -x;
   (*bottom)[1] = z;
   (*bottom)[2] = y;

   glm::mat4 *top = &((*cubetrans)[5]);
   *top = glm::mat4(1.0); //build ident
   (*top)[0] = -x;
   (*top)[1] = -z;
   (*top)[2] = -y;

   //Transpose so in col major
   for( int i =0 ; i < 6; i++ )
      (*cubetrans)[i] = glm::transpose((*cubetrans)[i]);
}
void castRays( const TreeNode &tree, Ray *rays, int numRays, Color *buffer, int width )
{
   glm::mat4 *cubetrans;
   vec3 ***cuberays = initCuberays();
   initCubeTransforms( &cubetrans );
   printf("Casting Rays: \n");
   int last = 0;
   int cur =0;
   //pollTest( tree, MAX_ANGLE, cuberays, cubetrans );
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i*width + rays[i].j] = raytrace( tree, rays[i], (vec3 ***)cuberays, cubetrans );
      cur = (int)((float)i / (float)numRays * 100);
      if ( cur > last )
      {
         printf("Percent Complete: %d      \r", cur);
         fflush(stdout);
         last = cur;
      }
   }
   printf("Percent Complete: 100     \n");
}
void castRaysCPU( CudaNode *cpu_root, int nodes, SurfelArray cpu_array, Ray *rays, int number,
      Color *buffer, int width)
{
   glm::mat4 *cubetrans;
   vec3 ***cuberays = initCuberays();
   initCubeTransforms( &cubetrans );
   printf("Casting Rays: \n");
   int last = 0;
   int cur =0;
   //pollTest( tree, MAX_ANGLE, cuberays, cubetrans );
   for( int i = 0; i < number; i++ )
   {
      buffer[rays[i].i*width + rays[i].j] = raytrace( cpu_root, cpu_array, rays[i],
            (vec3 ***)cuberays, cubetrans );
      cur = (int)((float)i / (float)number * 100);
      if ( cur > last )
      {
         printf("Percent Complete: %d      \r", cur);
         fflush(stdout);
         last = cur;
      }
   }
   printf("Percent Complete: 100     \n");
}
extern "C" Color *gpuSurfelColorBleeding( SurfelArray cpu_array, vec3 *positions, vec3 *normals,
      int num );
bool first;
void castRaysGPUSurfels( Scene scene, CudaNode *cpu_root, int nodes, SurfelArray cpu_array,
      Ray *rays, int number, Color *buffer, int width)
{
   glm::mat4 *cubetransforms;
   vec3 ***cuberays = initCuberays();
   initCubeTransforms( &cubetransforms );
   printf("Casting Rays: \n");

   Color black;
   black.r = 0;
   black.g = 0;
   black.b = 0;

   int last = 0;
   int cur =0;
   int index = 0;
   bool *misses =(bool *)malloc(sizeof(bool) * number);
   vec3 *positions = (vec3 *)malloc( sizeof(vec3) * number );
   vec3 *normals = (vec3 *)malloc( sizeof(vec3) * number );
   int numHits = 0;

   printf("Gathering Hits for GPU\n");
   //#pragma omp parallel for
   for( int h = 0; h < number; h++ )
   {
      Color color;
      color.r =0;
      color.g = 0;
      color.b = 0;
      Intersection inter = raytraceVisionRay( scene, rays[h] );
      if ( inter.hit == false )
      {
         index++;
         misses[h] = true;
         buffer[rays[h].i*width + rays[h].j] = color;
         continue;
      }
      misses[h] = false;

      Surfel hit = intersectionToSurfel( inter, scene);
      hit.pos.x += hit.normal.x * 0.00001;
      hit.pos.y += hit.normal.y * 0.00001;
      hit.pos.z += hit.normal.z * 0.00001;

      buffer[rays[h].i*width + rays[h].j] = hit.color;
      positions[numHits] = hit.pos;
      normals[numHits] = hit.normal;
      numHits++;

      index++;
      cur = (int)((float)index / (float)number * 100);
      if ( cur > last )
      {
         printf("Percent Complete: %d/%d      \r", cur, 100);
         fflush(stdout);
         last = cur;
      }
   }
   printf("Percent Complete: 100/100\n\n");
   printf("Going to GPU\n");

   Color *indirect = gpuSurfelColorBleeding( cpu_array, positions, normals, numHits );
   int k = 0;
   for( int h = 0; h < number; h++ )
   {
      if( misses[h] )
         continue;

      /*
         buffer[rays[h].i*width + rays[h].j].r = fmax( 0.0, fmin( 1.0, indirect[k].r));
         buffer[rays[h].i*width + rays[h].j].g = fmax( 0.0, fmin( 1.0, indirect[k].g));
         buffer[rays[h].i*width + rays[h].j].b = fmax( 0.0, fmin( 1.0, indirect[k].b));
       */
      buffer[rays[h].i*width + rays[h].j].r = fmax( 0.0, fmin( 1.0, indirect[k].r+
               buffer[rays[h].i*width + rays[h].j].r));
      buffer[rays[h].i*width + rays[h].j].g = fmax( 0.0, fmin( 1.0, indirect[k].g+
               buffer[rays[h].i*width + rays[h].j].g));
      buffer[rays[h].i*width + rays[h].j].b = fmax( 0.0, fmin( 1.0, indirect[k].b+
               buffer[rays[h].i*width + rays[h].j].b));
      if( k > numHits )
      {
         printf("ERROR %d %d %d\n", k, numHits, h);
         exit(1);
      }
      k++;
   }

   free(indirect);
}
void castRays( Scene scene, CudaNode *cpu_root, int nodes, SurfelArray cpu_array,
      Ray *rays, int number,
      Color *buffer, int width)
{
   glm::mat4 *cubetransforms;
   vec3 ***cuberays = initCuberays();
   initCubeTransforms( &cubetransforms );
   glm::mat4 M = getViewPixelMatrix() * getOrthMatrix() * getProjectMatrix();

   glm::mat4 saveCT[6];
   for( int i = 0; i < 6; i++ )
      saveCT[i] = M * cubetransforms[i];
   /*
      glm::mat4 *cubetransforms;
      vec3 ***cuberays = initCuberays();
      initCubeTransforms( &cubetransforms );
    */
   printf("Casting Rays: \n");

   Color black;
   black.r = 0;
   black.g = 0;
   black.b = 0;

   int last = 0;
   int cur =0;
   int index = 0;
#pragma omp parallel for
   for( int h = 0; h < number; h++ )
   {
      Color color;
      color.r =0;
      color.g = 0;
      color.b = 0;
      Intersection inter = raytraceVisionRay( scene, rays[h] );
      if ( inter.hit == false )
      {
         buffer[rays[h].i*width + rays[h].j] = color;
         continue;
      }

      Surfel hits = intersectionToSurfel( inter, scene);
      hits.pos.x += hits.normal.x * 0.00001;
      hits.pos.y += hits.normal.y * 0.00001;
      hits.pos.z += hits.normal.z * 0.00001;
      //    buffer[rays[h].i*width + rays[h].j] = hits.color;
      //     continue;

      glm::mat4 eyeTrans = glm::mat4(1.0);
      eyeTrans[3][0] = -hits.pos.x;
      eyeTrans[3][1] = -hits.pos.y;
      eyeTrans[3][2] = -hits.pos.z;


      RasterCube cube;
      for( int i = 0; i <6; i++)
         for( int j = 0; j<8; j++)
            for( int k =0; k<8;k++)
            {
               float ndotr = dot(hits.normal, cuberays[i][j][k]);
               if( ndotr < 0.001 )
               {
                  cube.sides[i][j][k] = black;
                  cube.depth[i][j][k] = -1;
               }
               else {
                  cube.sides[i][j][k] = black;
                  cube.depth[i][j][k] = -FAR_PLANE+1;
               }
            }
      //printf("Ray: %f %f %f, %f %f %f\n", hits.pos.x, hits.pos.y, hits.pos.z, hits.normal.x, hits.normal.y, hits.normal.z );

      /*
         if((rays[h].i == 511-188 || rays[h].i == 511-187) && rays[h].j == 376 )
         {
         traverseOctreeCPU( cube, cpu_root, 0, cpu_array, MAX_ANGLE, hits.pos,
         hits.normal, cuberays, cubetransforms, false, true );
         displayRasterCube( cube, rays[h].i );
         printf("\n\n\n");
         }
       */
      glm::mat4 ct[6];
      for( int i = 0; i < 6; i++ )
         ct[i] = saveCT[i] * eyeTrans;
      traverseOctreeCPU( cube, cpu_root, 0, cpu_array, MAX_ANGLE, hits.pos,
            hits.normal, cuberays, ct, false );
      int num = 0;
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
                  float dotProd = dot( cuberays[i][j][k], (hits.normal) );
                  color.r += (cube.sides[i][j][k].r*dotProd
                        /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
                  color.g += (cube.sides[i][j][k].g*dotProd
                        /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
                  color.b += (cube.sides[i][j][k].b*dotProd
                        /(0.9+cube.depth[i][j][k]*cube.depth[i][j][k]))/80.0;
               }
            }

      color.r += hits.color.r;
      color.g += hits.color.g;
      color.b += hits.color.b;

      color.r = fmin( color.r, 1.0 );
      color.r = fmax( color.r, 0.0 );
      color.g = fmin( color.g, 1.0 );
      color.g = fmax( color.g, 0.0 );
      color.b = fmin( color.b, 1.0 );
      color.b = fmax( color.b, 0.0 );

      buffer[rays[h].i*width + rays[h].j] = color;
      index++;
      cur = (int)((float)index / (float)number * 100);
      if ( cur > last )
      {
         printf("Percent Complete: %d/%d      \r", cur, 100);
         fflush(stdout);
         last = cur;
      }
   }
   printf("Percent Complete: 100/100\n\n");
}
void castRays( CudaNode *cpu_root, int nodes, SurfelArray cpu_array, Ray *rays, int number,
      Color *buffer, int width)
{
   first = true;
   static int tt = 0;
   glm::mat4 *cubetransforms;
   vec3 ***cuberays = initCuberays();
   initCubeTransforms( &cubetransforms );
   printf("Casting Rays: \n");
   Surfel *gpu_hits = gpuCastRays( cpu_root, nodes, cpu_array, rays, number );

   Color black;
   black.r = 0;
   black.g = 0;
   black.b = 0;

   int last = 0;
   int cur =0;
   for( int h = 0; h < number; h++ )
   {
      Color color;
      color.r =0;
      color.g = 0;
      color.b = 0;
      if( gpu_hits[h].radius < 0 )
      {
         buffer[rays[h].i*width + rays[h].j] = color;
         continue;
      }
      //if( rays[h].i != 256-156 || rays[h].j > 82 || rays[h].j < 78  )
      //   continue;
      buffer[rays[h].i*width + rays[h].j] = gpu_hits[h].color;
      continue;
      RasterCube cube;
      for( int i = 0; i <6; i++)
         for( int j = 0; j<8; j++)
            for( int k =0; k<8;k++)
            {
               float ndotr = dot(gpu_hits[h].normal, cuberays[i][j][k]);
               if( ndotr < 0.001 )
               {
                  cube.sides[i][j][k] = black;
                  cube.depth[i][j][k] = -1;
               }
               else {
                  cube.sides[i][j][k] = black;
                  cube.depth[i][j][k] = -FAR_PLANE+1;
               }
            }
      /*
         if( !NO_BLEED )
         traverseOctreeCPU( cube, cpu_root, 0, cpu_array, MAX_ANGLE, gpu_hits[h].pos,
         gpu_hits[h].normal, cuberays, cubetransforms, false );
       */
      int num = 0;
      for( int i = 0; i <6; i++)
         for( int j = 0; j<8; j++)
            for( int k =0; k<8;k++)
            {
               if( cube.depth[i][j][k] < 0 )
                  continue;
               num++;
               if( cube.depth[i][j][k] < -FAR_PLANE +1 )
               {
                  float dotProd = dot( cuberays[i][j][k], gpu_hits[h].normal );
                  float atten = 1.0 / ( cube.depth[i][j][k] );
                  atten = fmin( atten, 10 );
                  atten = 1;
                  if(cube.sides[i][j][k].r > 0 )
                     color.r += cube.sides[i][j][k].r*dotProd*atten;
                  if(cube.sides[i][j][k].g > 0 )
                     color.g += cube.sides[i][j][k].g*dotProd*atten;
                  if(cube.sides[i][j][k].b > 0 )
                     color.b += cube.sides[i][j][k].b*dotProd*atten;
               }
            }
      //displayRasterCube( cube, rays[h].j );
      //printf("Color: %f %f %f\n", color.r, color.g, color.b );
      /*
         if( num > 0 )
         {
         color.r /= (float)num;
         color.g /= (float)num;
         color.b /= (float)num;
         }
       */
      /*
         color.r += gpu_hits[h].color.r;
         color.g += gpu_hits[h].color.g;
         color.b += gpu_hits[h].color.b;
       */

      color.r = fmin( color.r, 0.6 );
      color.r = fmax( color.r, 0.0 );
      color.g = fmin( color.g, 0.6 );
      color.g = fmax( color.g, 0.0 );
      color.b = fmin( color.b, 0.6 );
      color.b = fmax( color.b, 0.0 );

      buffer[rays[h].i*width + rays[h].j] = color;
      cur = (int)((float)h / (float)number * 100);
      //if ( cur > last )
      //{
      printf("Percent Complete: %d/%d      \r", h, number);
      fflush(stdout);
      last = cur;
      // }
   }
   printf("Percent Complete: 100/100\n\n");
}
TreeNode createSurfelTree( const Scene &scene, Ray *rays, int numRays, int width )
{
   vec3 min;
   vec3 max;
   IntersectionArray IA = createIntersectionArray();

   for( int i = 0; i < numRays; i++ )
   {
      //if( i%width < width-1 )
      //collectIntersections( scene, rays[i], IA, rays[i+1] );
      //else
      //collectIntersections( scene, rays[i], IA, rays[i-1] );
   }
   shrinkIA( IA );
   SurfelArray SA = createSurfelArray( IA.num );
   for( int i = 0; i < IA.num; i++ )
   {
      if( i == 0 )
      {
         min = IA.array[i].hitMark;
         max = min;
      }
      addToSA( SA, intersectionToSurfel( IA.array[i], scene ) );
      keepMin( min, IA.array[i].hitMark );
      keepMax( max, IA.array[i].hitMark );
   }
   shrinkSA( SA );

   TreeNode ret = createOctreeMark2( SA, min, max );
   freeSurfelArray( SA );
   return ret;
}
void createCudaSurfelTree( const Scene &scene, Ray *rays, int numRays, int width, CudaNode* &gpu_root,
      int &nodes, SurfelArray &gpu_array )
{
   vec3 min;
   vec3 max;
   IntersectionArray IA = createIntersectionArray();

   for( int i = 0; i < numRays; i++ )
   {
      if( i%width < width-1 )
         collectIntersections( scene, rays[i], IA, rays[i], rays[i+1] );
      else
         collectIntersections( scene, rays[i], IA, rays[i-1], rays[i] );
      //collectIntersections( scene, rays[i], IA, rays[numRays/2], rays[numRays/2+1] );
   }
   shrinkIA( IA );
   SurfelArray SA = createSurfelArray( IA.num );
   for( int i = 0; i < IA.num; i++ )
   {
      if( i == 0 )
      {
         min = IA.array[i].hitMark;
         max = min;
      }
      addToSA( SA, intersectionToSurfel( IA.array[i], scene ) );
      keepMin( min, IA.array[i].hitMark );
      keepMax( max, IA.array[i].hitMark );
   }
   shrinkSA( SA );

   gpu_root = NULL;

   createCudaTree( SA, min, max, gpu_root, nodes, gpu_array );
   freeSurfelArray( SA );
}
void collectIntersections( const Scene &scene, const Ray &ray, IntersectionArray &IA, const Ray &sec,
      const Ray &third)
{
   float t;
   int i = 0;
   for( int j = 0; j < scene.numTriangles; j++ )
   {
      t = triangleHitTest( scene.triangles[j], ray );
      if( t > 0 )
      {
         vec3 h;
         h.x = sec.pos.x + sec.dir.x * t;
         h.y = sec.pos.y + sec.dir.y * t;
         h.z = sec.pos.z + sec.dir.z * t;
         vec3 k;
         k.x = third.pos.x + third.dir.x * t;
         k.y = third.pos.y + third.dir.y * t;
         k.z = third.pos.z + third.dir.z * t;
         Intersection inter = triangleIntersection( scene.triangles[j], ray, t );
         inter.radius = sqrt(distance( h, k )/2);
         addToIA( IA,  inter);
         i++;
      }
   }
   for( int j = 0; j < scene.numSpheres; j++ )
   {
      float_2 sphereT = sphereHitTest( scene.spheres[j], ray );
      if( sphereT.t0 > 0 )
      {
         vec3 h;
         h.x = sec.pos.x + sec.dir.x * t;
         h.y = sec.pos.y + sec.dir.y * t;
         h.z = sec.pos.z + sec.dir.z * t;
         vec3 k;
         k.x = third.pos.x + third.dir.x * t;
         k.y = third.pos.y + third.dir.y * t;
         k.z = third.pos.z + third.dir.z * t;
         Intersection inter = sphereIntersection( scene.spheres[j], ray, sphereT.t0 );
         inter.radius = sqrt(distance( h, k )/2);
         addToIA( IA, inter );
         i++;
      }
      if( sphereT.t1 > 0 )
      {
         vec3 h;
         h.x = sec.pos.x + sec.dir.x * t;
         h.y = sec.pos.y + sec.dir.y * t;
         h.z = sec.pos.z + sec.dir.z * t;
         vec3 k;
         k.x = third.pos.x + third.dir.x * t;
         k.y = third.pos.y + third.dir.y * t;
         k.z = third.pos.z + third.dir.z * t;
         Intersection inter = sphereIntersection( scene.spheres[j], ray, sphereT.t1 );
         inter.radius = sqrt(distance( h, k )/2);
         addToIA( IA, inter );
         i++;
      }
   }
   for( int j = 0; j < scene.numPlanes; j++ )
   {
      t = planeHitTest( scene.planes[j], ray );
      if( t > 0 )
      {
         vec3 h;
         h.x = sec.pos.x + sec.dir.x * t;
         h.y = sec.pos.y + sec.dir.y * t;
         h.z = sec.pos.z + sec.dir.z * t;
         vec3 k;
         k.x = third.pos.x + third.dir.x * t;
         k.y = third.pos.y + third.dir.y * t;
         k.z = third.pos.z + third.dir.z * t;
         Intersection inter = planeIntersection( scene.planes[j], ray, t );
         inter.radius = sqrt(distance( h, k )/2);
         addToIA( IA, inter);
         i++;
      }
   }
}
Intersection raytraceVisionRay( const Scene &scene, const Ray &ray )
{
   int best = -1;
   Intersection bInter;
   bInter.hit = false;
   for( int j = 0; j < scene.numTriangles; j++ )
   {
      float t;
      t = triangleHitTest( scene.triangles[j], ray );
      if( t > 0 )
      {
         if( best < 0 || t < best )
         {
            best = t;
            bInter = triangleIntersection( scene.triangles[j], ray, t );
         }
      }
   }
   for( int j = 0; j < scene.numSpheres; j++ )
   {
      float_2 sphereT = sphereHitTest( scene.spheres[j], ray );
      if( sphereT.t0 > 0 )
      {
         if( best < 0 || sphereT.t0 < best )
         {
            best = sphereT.t0;
            bInter = sphereIntersection( scene.spheres[j], ray, sphereT.t0 );
         }
      }
   }
   for( int j = 0; j < scene.numPlanes; j++ )
   {
      float t;
      t = planeHitTest( scene.planes[j], ray );
      if( t > 0 )
      {
         if( best < 0 || t < best )
         {
            best = t;
            bInter = planeIntersection( scene.planes[j], ray, t );
         }
      }
   }
   return bInter;
}
void displayRasterCube( RasterCube &cube, int num )
{
   for( int i = 0; i < 6; i++ )
   {
      std::stringstream s;
      s << "Output/side-" << num << "-" <<i << ".tga";
      printf("%s\n",s.str().c_str());
      Tga outfile( 8, 8 );
      Color *buffer = outfile.getBuffer();
      for( int j = 0; j < 8; j++ )
      {
         for( int k =0; k < 8; k++ )
         {
            printf("%f/%f ", cube.sides[i][j][k].g, cube.depth[i][j][k]);
            buffer[(7-j)*8 + k].r = cube.sides[i][j][k].r;
            buffer[(7-j)*8 + k].g = cube.sides[i][j][k].g;
            buffer[(7-j)*8 + k].b = cube.sides[i][j][k].b;
         }
         printf("\n");
      }
      outfile.writeTga( s.str().c_str() );
   }
}
bool testForHitTEST( BoundingBox &boxIn, const Ray &ray )
{
   vec3 min = boxIn.min;
   min.x -= 0.01;
   min.y -= 0.01;
   min.z -= 0.01;

   vec3 max = boxIn.max;
   max.x += 0.01;
   max.y += 0.01;
   max.z += 0.01;
   BoundingBox box;
   box.min = min;
   box.max = max;

   if( ray.dir.x > -0.0001 && ray.dir.x < 0.0001 )
   {
      if( ray.pos.x < box.min.x || ray.pos.x > box.max.x )
         return false;
   }
   if( ray.dir.y > -0.0001 && ray.dir.y < 0.0001 )
   {
      if( ray.pos.y < box.min.y || ray.pos.y > box.max.y )
         return false;
   }
   if( ray.dir.z > -0.0001 && ray.dir.z < 0.0001 )
   {
      if( ray.pos.z < box.min.z || ray.pos.z > box.max.z )
         return false;
   }
   float txmin = (box.min.x - ray.pos.x) / ray.dir.x;
   float tymin = (box.min.y - ray.pos.y) / ray.dir.y;
   float tzmin = (box.min.z - ray.pos.z) / ray.dir.z;
   float txmax = (box.max.x - ray.pos.x) / ray.dir.x;
   float tymax = (box.max.y - ray.pos.y) / ray.dir.y;
   float tzmax = (box.max.z - ray.pos.z) / ray.dir.z;

   if( txmin > txmax )
   {
      float temp = txmax;
      txmax = txmin;
      txmin = temp;
   }
   if( tymin > tymax )
   {
      float temp = tymax;
      tymax = tymin;
      tymin = temp;
   }
   if( tzmin > tzmax )
   {
      float temp = tzmax;
      tzmax = tzmin;
      tzmin = temp;
   }

   float tgmin = txmin;
   float tgmax = txmax;
   //find largest min
   if( tgmin < tymin )
      tgmin = tymin;
   if( tgmin < tzmin )
      tgmin = tzmin;

   //find smallest max
   if( tgmax > tymax )
      tgmax = tymax;
   if( tgmax > tzmax )
      tgmax = tzmax;

   if( tgmin > tgmax )
      return false;
   return true;
}

float surfelHitTestTEST( Surfel s, const Ray &ray )
{
   vec3 direction = unit(ray.dir);
   vec3 position;
   vec3 normal = unit(s.normal);

   position.x = ray.pos.x;
   position.y = ray.pos.y;
   position.z = ray.pos.z;

   float vd = dot(normal, direction);
   if( vd > 0.001 )
      return -1;
   float v0 = -(dot(position, normal) + s.distance );
   float t = v0/vd;
   if( t < 0.01)
      return -1;

   vec3 hitMark;
   hitMark.x = ray.pos.x + direction.x*t;
   hitMark.y = ray.pos.y + direction.y*t;
   hitMark.z = ray.pos.z + direction.z*t;
   float d = squareDistance( hitMark, s.pos );

   if( d < s.radius*s.radius )
      return t;
   else
      return -1;
}


Color raytrace( struct CudaNode *root, SurfelArray surfels, const Ray &ray,
      vec3 ***cuberay, glm::mat4 *cubetrans )
{
   int stack[30*8+2];
   bool hit = false;
   float t = 0;
   float bestT = 100000;
   Surfel bestSurfel;
   int stack_current = 1;
   CudaNode cached;

   stack[0] = 0;
   while( stack_current )
   {
      stack_current--;

      cached = root[stack[stack_current]];
      if( testForHitTEST( cached.box, ray ) )
      {
         if( cached.leaf )
         {
            for( int i = cached.children[0]; i < cached.children[1]; i++ )
            {
               t = surfelHitTestTEST( surfels.array[i], ray );
               if( (t > 0 && t < bestT) || (hit == false && t > 0) )
               {
                  bestT = t;
                  bestSurfel = surfels.array[i];
                  hit = true;
               }
            }
         }
         else
         {
            for( int i = 0; i < 8; i++ )
            {
               if( cached.children[i] > 0 ){
                  stack[stack_current] = cached.children[i];
                  stack_current++;
               }
            }

         }
      }
   }
   if( hit )
   {
      vec3 hitMark;
      hitMark.x = ray.dir.x * bestT + ray.pos.x;
      hitMark.y = ray.dir.y * bestT + ray.pos.y;
      hitMark.z = ray.dir.z * bestT + ray.pos.z;
      bestSurfel.pos = hitMark;
      bestSurfel.radius = 1;
   }
   else
   {
      bestSurfel.radius = -1;
   }
   return bestSurfel.color;

}
Color raytrace( const struct TreeNode &tree, const Ray &ray, vec3 ***cuberay, glm::mat4 *cubetrans )
{
   static int it = 0;
   static int q = 0;
   Color color;
   color.r = 0;
   color.b = 0;
   color.g = 0;
   it++;

   TreeHitMark cur = transTree( tree, ray );
   if ( cur.t > 0 )
   {
      RasterCube cube;
      for( int i = 0; i <6; i++)
         for( int j = 0; j<8; j++)
            for( int k =0; k<8;k++)
            {
               float ndotr = dot(cur.surfel.normal, cuberay[i][j][k]);
               if( ndotr < 0.001 )
               {
                  cube.sides[i][j][k] = color;
                  cube.depth[i][j][k] = -1;
               }
               else {
                  cube.sides[i][j][k] = color;
                  cube.depth[i][j][k] = -FAR_PLANE+1;
               }
            }
      vec3 hit;
      hit.x = ray.pos.x + ray.dir.x * cur.t;
      hit.y = ray.pos.y + ray.dir.y * cur.t;
      hit.z = ray.pos.z + ray.dir.z * cur.t;

      traverseOctreeCPU( cube, tree, MAX_ANGLE, hit, cur.surfel.normal, cuberay, cubetrans );
      int num = 0;
      color.r = 0;
      color.g = 0;
      color.b = 0;
      for( int i = 0; i <6; i++)
         for( int j = 0; j<8; j++)
            for( int k =0; k<8;k++)
            {
               if( cube.depth[i][j][k] < 0 )
                  continue;
               num++;
               if( cube.depth[i][j][k] < -FAR_PLANE +1 )
               {
                  float dotProd = dot( cuberay[i][j][k], cur.surfel.normal );
                  if(cube.sides[i][j][k].r > 0 )
                     color.r += cube.sides[i][j][k].r*dotProd;
                  if(cube.sides[i][j][k].g > 0 )
                     color.g += cube.sides[i][j][k].g*dotProd;
                  if(cube.sides[i][j][k].b > 0 )
                     color.b += cube.sides[i][j][k].b*dotProd;
               }
            }

      if( num > 0 )
      {
         color.r /= (float)num;
         color.g /= (float)num;
         color.b /= (float)num;
      }
      color.r += cur.color.r;
      color.g += cur.color.g;
      color.b += cur.color.b;

      color.r = fmin( color.r, 1.0 );
      color.r = fmax( color.r, 0.0 );
      color.g = fmin( color.g, 1.0 );
      color.g = fmax( color.g, 0.0 );
      color.b = fmin( color.b, 1.0 );
      color.b = fmax( color.b, 0.0 );

      return color;
   }
   return color;
}
TreeHitMark transTreeCPU( CudaNode *cpu_root, int current, SurfelArray cpu_array, const Ray &ray )
{
   if( testForHit( cpu_root[current].box, ray ) )
   {
      if( cpu_root[current].leaf )
      {
         TreeHitMark best;
         best.color.r = 0;
         best.color.g = 0;
         best.color.b = 0;
         TreeHitMark cur;
         best.t = -1;
         for( int j = cpu_root[current].children[0]; j < cpu_root[current].children[1]; j++ )
         {
            cur.t = surfelHitTest( cpu_array.array[j], ray );
            if( cur.t > 0 )
            {
               if( cur.t < best.t || best.t < 0 )
               {
                  best.surfel = cpu_array.array[j];
                  best.color = cpu_array.array[j].color;
                  best.t = cur.t;
               }
            }
         }
         return best;
      }
      else
      {
         TreeHitMark best = transTreeCPU( cpu_root, cpu_root[current].children[0], cpu_array, ray );
         for( int i = 1; i < 8; i++ )
         {
            TreeHitMark cur = transTreeCPU( cpu_root, cpu_root[current].children[i], cpu_array, ray );
            if( cur.t > 0 )
            {
               if( cur.t < best.t || best.t < 0 )
               {
                  best = cur;
               }
            }
         }
         return best;
      }
   }
   TreeHitMark none;
   none.color.r = 0;
   none.color.g = 0;
   none.color.b = 0;
   none.t = -1;
   return none;
}
TreeHitMark transTree( TreeNode tree, const Ray &ray )
{
   if( testForHit( tree.box, ray ) )
   {
      if( tree.leaf )
      {
         TreeHitMark best;
         best.color.r = 0;
         best.color.g = 0;
         best.color.b = 0;
         TreeHitMark cur;
         best.t = -1;
         for( int j = 0; j < tree.SA.num; j++ )
         {
            cur.t = surfelHitTest( tree.SA.array[j], ray );
            if( cur.t > 0 )
            {
               if( cur.t < best.t || best.t < 0 )
               {
                  best.surfel = tree.SA.array[j];
                  best.color = tree.SA.array[j].color;
                  best.t = cur.t;
               }
            }
         }
         return best;
      }
      else
      {
         TreeHitMark best = transTree( *(tree.children[0]), ray );
         for( int i = 1; i < 8; i++ )
         {
            TreeHitMark cur = transTree( *(tree.children[i]), ray );
            if( cur.t > 0 )
            {
               if( cur.t < best.t || best.t < 0 )
               {
                  best = cur;
               }
            }
         }
         return best;
      }
   }
   TreeHitMark none;
   none.color.r = 0;
   none.color.g = 0;
   none.color.b = 0;
   none.t = -1;
   return none;
}

void traverseOctreeCPU( RasterCube &cube, const TreeNode &node, float maxangle,
      vec3 &position, vec3 &normal, vec3 ***cuberays, glm::mat4 *cubetransforms)
{
   //THIS IS WRONG FUNCTION;
   return;
   if( node.leaf == 1 )
   {
      float dis = 0;
      for( int i = 0; i < node.SA.num; i++ )
      {
         dis = distance( position, node.SA.array[i].pos );
         if ( dis < node.SA.array[i].radius )
         {
            raytraceSurfelToCube( cube, node.SA.array[i], cuberays, position, normal );
         }
         else
         {
            //rasterizeSurfelToCube( cube, node.SA.array[i], cubetransforms, cuberays,
            //      position, normal );
         }
      }
   }
   else
   {
      if( isIn( node.box, position ) )
      {
         for( int i = 0; i < 8; i++)
         {
            if( node.children[i] != NULL )
            {
               traverseOctreeCPU( cube, *node.children[i], maxangle, position,
                     normal, cuberays, cubetransforms );
            }
         }
         return;
      }

      int horizon = belowHorizon( node.box, position, normal );
      //if whole box is below skip
      if( horizon == 8 )
         return;
      //if some of box is below go finer
      /*
         else if ( horizon )
         {
         for( int i = 0; i < 8; i++)
         {
         if( node.children[i] != NULL )
         {
         traverseOctreeCPU( cube, *node.children[i], maxangle, position,
         normal, cuberays, cubetransforms );
         }
         }
         return;
         }
       */
      //Whole box is above horizon so go
      vec3 center;
      center = newDirection(node.box.max, node.box.min);
      center.x /= 2.0;
      center.y /= 2.0;
      center.z /= 2.0;
      center.x += node.box.min.x;
      center.y += node.box.min.y;
      center.z += node.box.min.z;

      vec3 centerToEye = newDirection( position, center );
      centerToEye = unit(centerToEye);

      float dis = distanceToBox( node.box, position );
      float area = evaluateSphericalHermonicsArea( node, centerToEye );
      float solidangle = area / (dis *dis);
      if( dis > 0.0001 && solidangle < maxangle )
      {
         Color c = evaluateSphericalHermonicsPower( node, centerToEye );
         //rasterizeClusterToCube( cube, c, area, center, cubetransforms,
         //      cuberays, position, normal, dis, false );
         //rasterize the cluster as a disk
      }
      else
      {
         for( int i = 0; i < 8; i++)
         {
            if( node.children[i] != NULL )
            {
               traverseOctreeCPU( cube, *node.children[i], maxangle, position, normal,
                     cuberays, cubetransforms );
            }
         }
      }
   }
}
void traverseOctreeCPU( RasterCube &cube, CudaNode *cpu_root, int current, SurfelArray &cpu_array,
      float maxangle, vec3 &position, vec3 &normal, vec3 ***cuberays, glm::mat4 *cubetransforms, bool above )
{
   if( cpu_root[current].leaf == 1 )
   {
      float dis = 0;
      for( int i = cpu_root[current].children[0]; i < cpu_root[current].children[1]; i++ )
      {
         dis = distance( position, cpu_array.array[i].pos );
         if ( dis < cpu_array.array[i].radius)
         {
            raytraceSurfelToCube( cube, cpu_array.array[i], cuberays, position, normal );
         }
         else
         {
            rasterizeSurfelToCube( cube, cpu_array.array[i], cubetransforms, cuberays,
                  position, normal );
         }
      }
      return;
   }
   else
   {
      if( !above && isIn( cpu_root[current].box, position ) )
      {
         for( int i = 0; i < 8; i++)
         {
            traverseOctreeCPU( cube, cpu_root, cpu_root[current].children[i], cpu_array,
                  maxangle, position, normal, cuberays, cubetransforms, above );
         }
         return;
      }
      int horizon = 0;
      if( !above )
         horizon= belowHorizon( cpu_root[current].box, position, normal );
      ////if whole box is below skip
      /*
         if( horizon == 8 )
         return;
       */
      /*
         if( horizon )
         {
         for( int i = 0; i < 8; i++)
         {
         traverseOctreeCPU( cube, cpu_root, cpu_root[current].children[i], cpu_array,
         maxangle, position, normal, cuberays, cubetransforms, false, debug );
         }
         return;
         }
       */
      vec3 center;
      center = newDirection(cpu_root[current].box.max, cpu_root[current].box.min);

      center.x /= 2.0;
      center.y /= 2.0;
      center.z /= 2.0;
      center.x += cpu_root[current].box.min.x;
      center.y += cpu_root[current].box.min.y;
      center.z += cpu_root[current].box.min.z;

      vec3 centerToEye = newDirection( position, center );
      centerToEye = unit(centerToEye);

      float dis = distance( center, position );
      float dis2B = distanceToBox( cpu_root[current].box, position );
      float area = evaluateSphericalHermonicsAreaAll( cpu_root[current], cpu_root[current].box,
            position );
      float solidangle = area / (dis *dis);
      if( solidangle < MAX_ANGLE )
      {
         Color c = evaluateSphericalHermonicsPower( cpu_root[current], centerToEye );
         rasterizeClusterToCube( cube, c, area, center, cubetransforms,
               cuberays, position, normal, dis );
         return;
      }
      else
      {
         for( int i = 0; i < 8; i++)
         {
            traverseOctreeCPU( cube, cpu_root, cpu_root[current].children[i], cpu_array,
                  maxangle, position, normal, cuberays, cubetransforms, horizon==0 );

         }
         return;
      }
   }
}
glm::vec4 *getWVecs( )
{
   glm::vec4 *ret = new glm::vec4[6];
   //front w = neg Z, u= neg x, v = pos y
   ret[0] = glm::vec4( 0.0, 0.0, -1.0, 0.0 );
   //right w = neg x, u = pos z, v pos y
   ret[1] = glm::vec4( -1.0, 0.0, 0.0, 0.0 );
   //back w = pos z, u=  pos x, v = pos y
   ret[2] = glm::vec4( 0.0, 0.0, 1.0, 0.0 );
   //left w = pos x, u neg z, v pos y
   ret[3] = glm::vec4( 1.0, 0.0, 0.0, 0.0 );
   //bottom w = pos y, pos x, v = neg z
   ret[4] = glm::vec4( 0.0, 1.0, 0.0, 0.0 );
   //top w = -y, u = pos x, v = pos z
   ret[5] = glm::vec4( 0.0, -1.0, 0.0, 0.0 );

   return ret;
}
glm::vec4 *getAxisAlinedPoints( vec3 position, float len, int k )
{
   glm::vec4 *ret = new glm::vec4[4];
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
   return ret;
}
glm::mat4 getProjectMatrix()
{
   glm::mat4 ret = glm::mat4(1.0);
   ret[0] = glm::vec4( NEAR_PLANE, 0, 0, 0 );
   ret[1] = glm::vec4( 0, NEAR_PLANE, 0, 0 );
   ret[2] = glm::vec4( 0, 0, NEAR_PLANE + FAR_PLANE, - NEAR_PLANE * FAR_PLANE );
   ret[3] = glm::vec4( 0, 0, 1, 0 );
   return glm::transpose(ret);
}
glm::mat4 getOrthMatrix()
{
   glm::mat4 ret = glm::mat4( 1.0 );
   ret[0] = glm::vec4( 2.0/(RIGHT - LEFT), 0, 0, -(RIGHT +LEFT)/(RIGHT -LEFT) );
   ret[1] = glm::vec4( 0, 2.0/(TOP-BOTTOM), 0, -(TOP + BOTTOM)/(TOP-BOTTOM) );
   ret[2] = glm::vec4( 0, 0, 2.0/(NEAR_PLANE - FAR_PLANE),
         -(NEAR_PLANE + FAR_PLANE)/(NEAR_PLANE - FAR_PLANE) );
   ret[3] = glm::vec4( 0, 0, 0, 1.0 );
   return glm::transpose(ret);
}
glm::mat4 getViewPixelMatrix()
{
   glm::mat4 ret = glm::mat4(1.0);
   ret[0] = glm::vec4( NPIXELS/2.0, 0 ,0, (NPIXELS)/2.0 );
   ret[1] = glm::vec4( 0, -NPIXELS/2.0, 0, (NPIXELS)/2.0 );
   return glm::transpose(ret);
}
void rasterizeClusterToCube( RasterCube &cube, Color &c, float area, vec3 nodePosition,
      glm::mat4 *cubetransforms, vec3 ***cuberays, vec3 &position, vec3 &normal, float dis )
{
   const static glm::vec4 *wVecs = getWVecs();
   vec3 check = newDirection( position, nodePosition );
   check = unit(check);
   if( dot(check, normal) > -0.001 )
      return;
   float areas[6];
   for( int i =0; i < 6; i++ )
   {
      vec3 w;
      w.x = wVecs[i][0];
      w.y = wVecs[i][1];
      w.z = wVecs[i][2];
      if( dot( w, check ) > 0 )
         areas[i] = dot( w, check) * area;
      else
         areas[i] = 0;
   }

   for( int k = 0; k< 6; k++ )
   {
      if( areas[k] <= 0 )
         continue;
      float length = sqrtf(areas[k]);

      glm::vec4 *points = getAxisAlinedPoints( nodePosition, length/2.0, k );
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
         delete []points;
         continue;
      }
      if( !(maxX < 0 || maxY < 0 || minY > 7 || minX > 7 ))
      {
         if( minX < 0 )
            minX = 0;
         if( minY < 0 )
            minY = 0;
         if( maxX > 7 )
            maxX = 7;
         if( maxY > 7 )
            maxY = 7;

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
      delete []points;

   }
}
void rasterizeSurfelToCube( RasterCube &cube, Surfel &surfel, glm::mat4 *cubetransforms,
      vec3 ***cuberays, vec3 &position, vec3 &normal )
{
   const static glm::vec4 *wVecs = getWVecs();
   vec3 diff = newDirection( position, surfel.pos );
   diff = unit(diff);
   float dotPro = dot(normal, diff);
   if( dotPro > 0 )
      return;
   double area = surfel.radius *surfel.radius * PI;

   float dis = distance( position, surfel.pos );
   double areas[6];
   for( int i =0; i < 6; i++ )
   {
      vec3 w;
      w.x = wVecs[i][0];
      w.y = wVecs[i][1];
      w.z = wVecs[i][2];
      if(  dot( w, diff ) > 0 && dot( diff, surfel.normal ) > 0 )
         areas[i] = area * dot( diff, surfel.normal );// * dot(w, diff);
      else
         areas[i] = 0;

   }

   for( int k = 0; k< 6; k++ )
   {
      if( areas[k] < 0.00001 )
         continue;
      double length = sqrt( areas[k] );
      glm::vec4 *points = getAxisAlinedPoints( surfel.pos, length/2.0, k );
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
      for( int i = 1; i < 4; i++ )
      {
         if( minX > points[i][0] )
            minX = roundf(points[i][0] +0.5);
         if( minY > points[i][1] )
            minY = roundf(points[i][1] + 0.5);
         if( maxX < points[i][0] )
            maxX = roundf(points[i][0] + 0.5);
         if( maxY < points[i][1] )
            maxY = roundf(points[i][1] +0.5);
      }
      if( !(maxX < 0 || maxY < 0 || minY > 7 || minX > 7 ))
      {
         if( minX < 0 )
            minX = 0;
         if( minY < 0 )
            minY = 0;
         if( maxX > 7 )
            maxX = 7;
         if( maxY > 7 )
            maxY = 7;
         for( int i = minY; i <= maxY; i++ )
         {
            for( int j = minX; j <= maxX; j++ )
            {
               if (cube.depth[k][i][j] < 0)
                  continue;
               else if ( dis < cube.depth[k][i][j] )
               {
                  float atten = (areas[k]/ (dis * dis));
                  cube.sides[k][i][j].r = surfel.color.r;
                  cube.sides[k][i][j].g = surfel.color.g;
                  cube.sides[k][i][j].b = surfel.color.b;
                  cube.depth[k][i][j] = dis;
               }
            }
         }
      }
      delete []points;
   }

}

void raytraceSurfelToCube( RasterCube &cube, Surfel &surfel, vec3 ***cuberays, vec3 &position,
      vec3 &normal )
{
   const static glm::vec4 *wVecs = getWVecs();
   vec3 to;
   to = newDirection( surfel.pos, position );
   to = unit( to );
   if( dot( to, normal ) < 0.1 )
      return;
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
               /*
                  ray.pos.x += ray.dir.x*.1;
                  ray.pos.y += ray.dir.y*.1;
                  ray.pos.z += ray.dir.z*.1;
                */

               float t = surfelHitTest( surfel, ray );
               if( t < 0 )
               {
                  Surfel s = surfel;
                  s.normal.x *= -1;
                  s.normal.y *= -1;
                  s.normal.z *= -1;
                  t = surfelHitTest( s, ray );
               }
               if( t > 0 && t < cube.depth[i][j][k] )
               {
                  vec3 hit;
                  hit.x = ray.pos.x + ray.dir.x * t;
                  hit.y = ray.pos.y + ray.dir.y * t;
                  hit.z = ray.pos.z + ray.dir.z * t;
                  vec3 w;
                  w.x = wVecs[i][0];
                  w.y = wVecs[i][1];
                  w.z = wVecs[i][2];
                  vec3 diff = newDirection( position, hit );
                  diff = unit(diff);
                  float dis = distance( hit, position );
                  float area = surfel.radius * surfel.radius * PI * dot( w, diff );
                  float atten = area/(1.0 + dis*dis);

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
float evaluateSphericalHermonicsArea( const TreeNode &node, vec3 &centerToEye )
{
   /*float theta = acosf( centerToEye.y );
   //centerToEye.x can be 0 but atanf can handle -inf and inf

   float phi = atanf( centerToEye.y/centerToEye.x );
   float * TYlm = getYLM( sinf(theta) *cosf(phi), sinf(theta) * sinf(phi), cosf(theta) );
    */
   double * TYlm = getYLM( centerToEye.x, centerToEye.y, centerToEye.z );

   float area = 0;

   for( int i =0; i < 9; i++ )
   {
      area += node.hermonics.area[i] * TYlm[i];
   }
   if( area < 0 )
      printf("area %f\n", area);
   area = fmax(area, 0);
   free( TYlm );
   return area;
}
Color evaluateSphericalHermonicsPower( const TreeNode &node, vec3 &centerToEye )
{
   /*float theta = acosf( centerToEye.y );
   //centerToEye.x can be 0 but atanf can handle -inf and inf
   floa/ct phi = atanf( centerToEye.y/centerToEye.x );
   float * TYlm = getYLM( sinf(theta) *cosf(phi), sinf(theta) * sinf(phi), cosf(theta) );
    */
   double * TYlm = getYLM( centerToEye.x, centerToEye.y, centerToEye.z );
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
   /*color.r = fmin( color.r, 1.0 );
     color.g = fmin( color.g, 1.0 );
     color.b = fmin( color.b, 1.0 );
    */
   color.r = fmax( color.r, 0);
   color.g = fmax( color.g, 0);
   color.b = fmax( color.b, 0);
   free( TYlm );
   return color;
}
float evaluateSphericalHermonicsArea( const CudaNode &node, vec3 &centerToEye )
{
   /*float theta = acosf( centerToEye.y );
   //centerToEye.x can be 0 but atanf can handle -inf and inf
   float phi = atanf( centerToEye.y/centerToEye.x );
   float * TYlm = getYLM( sinf(theta) *cosf(phi), sinf(theta) * sinf(phi), cosf(theta) );
    */

   double * TYlm = getYLM( centerToEye.x, centerToEye.y, centerToEye.z );

   float area = 0;

   for( int i =0; i < 9; i++ )
   {
      area += node.hermonics.area[i] * TYlm[i];
   }
   area = fmax(area, 0);
   free( TYlm );
   return area;
}
float evaluateSphericalHermonicsAreaAll( const CudaNode &node, const BoundingBox &box,
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
      vec3 cte = unit(newDirection( position, points[i] ));

      float t = evaluateSphericalHermonicsArea( node, cte );
      if (area < t )
         area = t;
   }
   return area;
}
Color evaluateSphericalHermonicsPower( const CudaNode &node, vec3 &centerToEye )
{
   double * TYlm = getYLM( centerToEye.x, centerToEye.y, centerToEye.z );
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
   free( TYlm );
   return color;
}
