/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "CudaRay.h"
#include <stdio.h>
#include "cutil.h"
#define CUDASAFECALL( call )  CUDA_SAFE_CALL( call )
#define CUDAERRORCHECK() {                   \
   cudaError err = cudaGetLastError();        \
   if( cudaSuccess != err){ \
      printf("CudaErrorCheck %d\n", err);           \
      exit(1); \
   } }


__global__ void cast( Surfel *s, int numS, Ray *rays, int numRays, Color *buffer, int width );
__device__ Color raytrace( Surfel *s, int numS, Ray ray );
__device__ float surfelHitTestCuda( Surfel s, Ray ray );
__device__ float dotCuda( vec3 one, vec3 two );
__device__ float squareDistanceCuda( vec3 one, vec3 two );
__device__ vec3 unitCuda(vec3 in);
__device__ float magCuda(const vec3 &in);
__device__ Color limitColorCuda( Color color );

void castRaysCuda( const SurfelArray &s, Ray *rays, int numRays, Color *buffer, int width, int height )
{
   Surfel *d_s;
   Ray *d_r;
   Color *d_c;
   CUDASAFECALL(cudaMalloc( (void **)&(d_s), sizeof(Surfel) * s.num ));
   CUDASAFECALL(cudaMalloc( (void **)&(d_r), sizeof(Ray) * 10000 ));
   CUDASAFECALL(cudaMalloc( (void **)&(d_c), sizeof(Color) * width * height ));

   CUDASAFECALL(cudaMemcpy( d_s, s.array, sizeof(Surfel) * s.num, cudaMemcpyHostToDevice ));

   int x = numRays / 32;
   if( numRays%32 )
      x++;

   dim3 dimBlock(32);
   dim3 dimGrid( x );
   int curRay = 10000;
   int curser = 0;
   while( numRays > 0 )
   {
      if ( curRay > numRays )
         curRay = numRays;

      CUDASAFECALL(cudaMemcpy( d_r, rays+curser, sizeof(Ray) * curRay, cudaMemcpyHostToDevice ));
      cast<<<dimGrid, dimBlock>>>( d_s, s.num, d_r, curRay, d_c, width );
      numRays -= curRay;
      curser += curRay;
   }

   CUDASAFECALL(cudaMemcpy( buffer, d_c, sizeof(Color) * width * height, cudaMemcpyDeviceToHost ));

   cudaFree( d_c );
   cudaFree( d_s );
   cudaFree( d_r );
}

__global__ void cast( Surfel *s, int numS, Ray *rays, int numRays, Color *buffer, int width )
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   //int y = blockIdx.y * gridDim.y + threadIdx.y;
   if( x >= numRays )
      return;

   Ray ray = rays[x];
   buffer[ray.i*width + ray.j] = raytrace( s, numS, ray );
}

__device__ Color raytrace( Surfel *s, int numS, Ray ray )
{
   Color color;
   color.r = 0;
   color.g = 0;
   color.b = 0;

   bool hit = false;
   float bestT = 10000;
   float t;
   for( int j = 0; j < numS; j++ )
   {
      t = surfelHitTestCuda( s[j], ray );
      if( t > 0 )
      {
         if( !hit || t < bestT )
         {
            color = s[j].color;
            bestT = t;
            hit = true;
         }
      }
   }
   return limitColorCuda( color );
}
__device__ float surfelHitTestCuda( Surfel s, Ray ray )
{
   vec3 direction = unitCuda(ray.dir);
   vec3 position;
   vec3 normal = unitCuda(s.normal);

   direction.x = direction.x;
   direction.y = direction.y;
   direction.z = direction.z;
   position.x = ray.pos.x;
   position.y = ray.pos.y;
   position.z = ray.pos.z;

   float vd = dotCuda(normal, direction);
   if( vd < 0.001 && vd > -0.001 )
      return -1;
   float v0 = -(dotCuda(position, normal) - s.distance );
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
__device__ float dotCuda( vec3 one, vec3 two )
{
   return one.x*two.x + one.y*two.y + one.z*two.z;
}
__device__ float squareDistanceCuda( vec3 one, vec3 two )
{
   return ((one.x-two.x)*(one.x-two.x) + (one.y-two.y)*(one.y-two.y) + (one.z-two.z)*(one.z-two.z));
}
__device__ vec3 unitCuda(vec3 in)
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
__device__ Color limitColorCuda( Color in )
{
   Color ret;
   if( in.r > 1.0 )
      ret.r = 1.0;
   else if( in.r < 0.0 )
      ret.r = 0;
   else
      ret.r = in.r;

   if( in.g > 1.0 )
      ret.g = 1.0;
   else if( in.g < 0.0 )
      ret.g = 0;
   else
      ret.g = in.g;

   if( in.b > 1.0 )
      ret.b = 1.0;
   else if( in.b < 0.0 )
      ret.b = 0;
   else
      ret.b = in.b;

   return ret;
}
