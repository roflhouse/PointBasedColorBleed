/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include <stdio.h>
#include "cutil.h"
#include "Octree.h"
#define MAX_OCTREE_DEPTH 40
#define MAX_ANGLE 0.00
#define RADIUS 0.01
#include "UtilTypes.h"
#include "RasterCube.h"
#include "../Objects/SurfelType.h"
#define CUDASAFECALL( call )  CUDA_SAFE_CALL( call )
#define CUDAERRORCHECK() {                   \
   cudaError err = cudaGetLastError();        \
   if( cudaSuccess != err){ \
      printf("CudaErrorCheck %d\n", err);           \
      exit(1); \
   } }


__device__ float gpuDot(const vec3 &one, const vec3 &two);
__device__ float surfelHitTestCuda( Surfel s, Ray &ray );
__device__ float squareDistanceCuda( vec3 &one, vec3 &two );
__device__ vec3 gpuUnit(vec3 &in);
__device__ float magCuda(const vec3 &in);
__device__ bool testForHitCuda( BoundingBox &boxIn, Ray &ray );
__device__ Surfel gpu_raytrace( CudaNode *gpu_root, Surfel *gpu_array, Ray &ray );

__global__ void kernel_CastRays( CudaNode *gpu_root, Surfel *gpu_array,
      int surfels, Ray *gpu_rays, int num_rays, Surfel *output );

extern "C" Surfel *gpuCastRays( CudaNode *cpu_root, int nodes, SurfelArray cpu_array,
      Ray *rays, int num_rays )
{
   printf("Surfels: %d, CudaNodes: %d, Rays: %d\n", cpu_array.num, nodes, num_rays );
   float surfel_size = (float)(sizeof(Surfel) * cpu_array.num)/1048576.0;
   float cn_size = (float)(sizeof(CudaNode) * nodes)/1048576.0;
   float ray_size = (float)(sizeof(Ray) * num_rays)/1048576.0;
   float output_size = (float)(sizeof(Surfel) * num_rays )/1048576.0;
   printf("Sizes: Surfel %f CudaNodes %f Rays %f output: %f\n Total %f\n",
         surfel_size, cn_size, ray_size, output_size,
         surfel_size + cn_size + ray_size, output_size);

   CudaNode * d_root;
   Surfel *d_surfels;
   Ray *d_rays;
   Surfel *d_output;
   Surfel *cpu_output = (Surfel *)malloc( sizeof(Surfel)*num_rays );

   CUDASAFECALL(cudaMalloc( (void **)&d_surfels, sizeof(Surfel) * cpu_array.num));
   CUDASAFECALL(cudaMalloc( (void **)&d_root, sizeof(CudaNode) * nodes));
   CUDASAFECALL(cudaMalloc( (void **)&d_rays, sizeof(Ray) * num_rays));
   CUDASAFECALL(cudaMalloc( (void **)&d_output, sizeof(Surfel) * num_rays ));

   CUDASAFECALL(cudaMemcpy( d_surfels, cpu_array.array, sizeof(Surfel) * cpu_array.num,
            cudaMemcpyHostToDevice));
   CUDASAFECALL(cudaMemcpy( d_root, cpu_root, sizeof(CudaNode) * nodes,cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_rays, rays, sizeof(Ray) * num_rays,cudaMemcpyHostToDevice ));

   int num_blocks = ceilf( (float)num_rays / 32.0 );
   dim3 dimBlock( 32 );
   dim3 dimGrid( num_blocks );

   printf("GPU Casting Rays\n");
   kernel_CastRays<<<dimGrid, dimBlock>>>( d_root, d_surfels, cpu_array.num, d_rays,
         num_rays, d_output );
   printf("Done GPU casting\n");
   CUDAERRORCHECK();

   CUDASAFECALL(cudaMemcpy( cpu_output, d_output, sizeof(Surfel) * num_rays,
            cudaMemcpyDeviceToHost ));
   cudaFree( d_output );
   cudaFree( d_surfels );
   cudaFree( d_root );
   cudaFree( d_rays );

   return cpu_output;
}

__global__ void kernel_CastRays( CudaNode *gpu_root, Surfel *gpu_array,
      int surfels, Ray *gpu_rays, int num_rays, Surfel *output )
{
   int index = blockIdx.x * blockDim.x + threadIdx.x;

   if( index >= num_rays )
      return;
   Ray ray = gpu_rays[index];

   Surfel ret = gpu_raytrace( gpu_root, gpu_array, ray );

   output[index] = ret;
}
__device__ Surfel gpu_raytrace( CudaNode *gpu_root, Surfel *gpu_array, Ray &ray )
{
   int stack[MAX_OCTREE_DEPTH*8+2];
   bool hit = false;
   float t = 0;
   float bestT = 100000;
   Surfel bestSurfel;
   int stack_current = 1;
   CudaNode cached;

   //push root on stack;
   stack[0] = 0;
   while( stack_current )
   {
      stack_current--;

      cached = gpu_root[stack[stack_current]];
      if( testForHitCuda( cached.box, ray ) )
      {
         if( cached.leaf )
         {
            for( int i = cached.children[0]; i < cached.children[1]; i++ )
            {
               t = surfelHitTestCuda( gpu_array[i], ray );
               if( (t > 0 && t < bestT) || (hit == false && t > 0) )
               {
                  bestT = t;
                  bestSurfel = gpu_array[i];
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
   return bestSurfel;
}
__device__ float gpuDot(const vec3 &one, const vec3 &two)
{
      return one.x*two.x + one.y*two.y + one.z*two.z;
}

__device__ float surfelHitTestCuda( Surfel s, Ray &ray )
{
   vec3 direction = gpuUnit(ray.dir);
   vec3 position;
   vec3 normal = gpuUnit(s.normal);

   position.x = ray.pos.x;
   position.y = ray.pos.y;
   position.z = ray.pos.z;

   float vd = gpuDot(normal, direction);
   if( vd > 0.001)
      return -1;
   float v0 = -(gpuDot(position, normal) + s.distance );
   float t = v0/vd;
   if( t < 0.01)
      return -1;

   vec3 hitMark;
   hitMark.x = ray.pos.x + direction.x*t;
   hitMark.y = ray.pos.y + direction.y*t;
   hitMark.z = ray.pos.z + direction.z*t;
   float d = squareDistanceCuda( hitMark, s.pos );

   if( d < s.radius*s.radius )
      return t;
   else
      return -1;
}
__device__ float squareDistanceCuda( vec3 &one, vec3 &two )
{
   return ((one.x-two.x)*(one.x-two.x) + (one.y-two.y)*(one.y-two.y) + (one.z-two.z)*(one.z-two.z));
}
__device__ vec3 gpuUnit(vec3 &in)
{
   float temp;
   vec3 newVector;
   newVector.x = 0;
   newVector.y = 0;
   newVector.z = 0;
   temp = magCuda(in);

   if(temp > 0)
   {
      newVector.x = in.x/temp;
      newVector.y = in.y/temp;
      newVector.z = in.z/temp;
   }
   return newVector;
}
__device__ float magCuda(const vec3 &in)
{
   return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}
__device__ bool testForHitCuda( BoundingBox &boxIn, Ray &ray )
{
   vec3 min = boxIn.min;
   min.x -= RADIUS;
   min.y -= RADIUS;
   min.z -= RADIUS;

   vec3 max = boxIn.max;
   max.x += RADIUS;
   max.y += RADIUS;
   max.z += RADIUS;
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
