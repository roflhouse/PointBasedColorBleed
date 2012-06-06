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
#include "Octree.h"
#define MAXDEPTH 15
#define RADIUS 0.01
#define CUDASAFECALL( call )  CUDA_SAFE_CALL( call )
#define CUDAERRORCHECK() {                   \
   cudaError err = cudaGetLastError();        \
   if( cudaSuccess != err){ \
      printf("CudaErrorCheck %d\n", err);           \
      exit(1); \
   } }


__global__ void cast( Surfel *s, int numS, Ray *rays, int numRays, Color *buffer, int width );
__global__ void castTree( ArrayNode *tree, int size, Surfel *s, int numS,
      Ray *rays, int numRays, Color *buffer, int width );
__device__ Color raytrace( Surfel *s, int numS, Ray ray );
__device__ float surfelHitTestCuda( Surfel s, Ray ray );
__device__ float dotCuda( vec3 one, vec3 two );
__device__ float squareDistanceCuda( vec3 one, vec3 two );
__device__ vec3 unitCuda(vec3 in);
__device__ float magCuda(const vec3 &in);
__device__ Color raytrace( struct ArrayNode *tree, int size, Surfel *s, Ray ray );
__device__ Color limitColorCuda( Color color );
__device__ bool testForHitCuda( BoundingBox boxIn, Ray ray );

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
void castRaysCuda( const struct ArrayNode *tree, int size, const SurfelArray &s, Ray *rays, int numRays,
      Color *buffer, int width, int height )
{
   Surfel *d_s;
   Ray *d_r;
   Color *d_c;
   ArrayNode *d_t;
   printf("size %lu\n", sizeof(Surfel) * s.num + sizeof(ArrayNode) * size +
         sizeof(Ray)* numRays + sizeof(Color) *width *height );// + sizeof(ArrayNode) * size );
   CUDASAFECALL(cudaMalloc( (void **)&(d_s), sizeof(Surfel) * s.num ));
   CUDASAFECALL(cudaMalloc( (void **)&(d_t), sizeof(ArrayNode) * size ) );
   CUDASAFECALL(cudaMalloc( (void **)&(d_r), sizeof(Ray) * numRays ));
   CUDASAFECALL(cudaMalloc( (void **)&(d_c), sizeof(Color) * width * height ));

   CUDASAFECALL(cudaMemcpy( d_r, rays, sizeof(Ray) * numRays, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_s, s.array, sizeof(Surfel) * s.num, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_t, tree, sizeof(ArrayNode) * size, cudaMemcpyHostToDevice ) );

   int x = sqrt( numRays /32 ) ;
   if( x < sqrt( numRays /32 ) )
      x++;

   dim3 dimBlock(32);
   dim3 dimGrid( x, x );
   castTree<<<dimGrid, dimBlock>>>( d_t, size, d_s, s.num, d_r, numRays, d_c, width );

   CUDASAFECALL(cudaMemcpy( buffer, d_c, sizeof(Color) * width * height, cudaMemcpyDeviceToHost ));

   cudaFree( d_c );
   cudaFree( d_s );
   cudaFree( d_r );
}
__global__ void castTree( ArrayNode *tree, int size, Surfel *s, int numS, Ray *rays, int numRays,
      Color *buffer, int width )
{
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = blockIdx.y * gridDim.x;
   int index = y * blockDim.x + x;
   //int y = blockIdx.y * gridDim.y + threadIdx.y;
   if( index >= numRays )
      return;

   Ray ray = rays[index];
   buffer[ray.i*width + ray.j] = raytrace( tree, size, s, ray );
}
__device__ Color raytrace( struct ArrayNode *tree, int size, Surfel *s, Ray ray )
{
   Color color;
   color.r = 0;
   color.b = 0;
   color.g = 0;

   bool hit = false;
   float bestT = 10000;
   float t = 0;

   int stack[MAXDEPTH*8+2];
   int curser = 1;
   stack[0] = 0;
   while( curser ){
      curser--;
      int now = stack[curser];
      //printf("Doing %d\n", now );
      if( testForHitCuda( tree[now].box, ray ) )
      {
         if( tree[now].leaf )
         {
            for( int j = tree[now].children[0]; j < tree[now].children[1]; j++ )
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
         }
         else
         {
            for( int i = 7; i >= 0; i-- )
            {
               if( tree[now].children[i] > 0 )
               {
                  stack[curser] = tree[now].children[i];
                  //printf("Push %d\n", stack[curser]);
                  if( curser > MAXDEPTH*8 )
                     printf("FUCK\n");
                  curser++;
               }
            }
         }
      }
   }

   return color;
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
__device__ bool testForHitCuda( BoundingBox boxIn, Ray ray )
{
   vec3 min = boxIn.min;
   min.x -= RADIUS;
   min.y -= RADIUS;
   min.z -= RADIUS;

   vec3 max = boxIn.max;
   max.x += RADIUS;
   max.y += RADIUS;
   max.z += RADIUS;
   BoundingBox box ;
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
