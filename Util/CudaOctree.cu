/**
 *  CPE 2013
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#define CUDASAFECALL( call )  CUDA_SAFE_CALL( call )
#include "cutil.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include "UtilTypes.h"
#include "Octree.h"
#define CUDAERRORCHECK() {                   \
   cudaError err = cudaGetLastError();        \
   if( cudaSuccess != err){ \
      printf("CudaErrorCheck %d\n", err);           \
      exit(1); \
   } }
#define PI 3.14159265359
#define MONTE_CARLO_N 256
#define MAX_DEPTH  20
#define MAX_OCTREE_SIZE 1000

__device__ Hermonics gpuCalculateSphericalHermonics( struct Surfel surfel );
__device__ Hermonics gpuCreateHermonics();
__device__ void gpuAddHermonics( Hermonics &save, Hermonics &gone );
void cpuAddHermonics( Hermonics &save, Hermonics &gone );
__global__ void fillLeafSphericalHermonics( CudaNode *d_root, int tree_total,
      Surfel *surfels, int surfel_total, int *d_leaf_addrs, int leaf_nodes );
__global__ void kernel_FirstPassSphericalHermonics( Surfel *d_surfels, Hermonics *d_hermonics,
      int num, int batch, int batch_size );
__global__ void kernel_SecondPassSphericalHermonics( CudaNode *d_root, int nodes,
      Hermonics *d_hermonics, int num_her, int *d_leaf_addrs, int leafs );
extern "C" int getTime( );
extern "C" float getDiffTime( int start, int end );
void FillOutHermonicsFromArray( int current, CudaNode *root, Hermonics *hermonics );
__device__ vec3 gpuUnit( vec3 in )
{
   float mag = sqrt(in.x*in.x + in.y *in.y + in.z*in.z);
   in.x /= mag;
   in.y /= mag;
   in.z /= mag;
   return in;
}
void checkCUDAError(const char *msg) {
   cudaError_t err = cudaGetLastError();
   if( cudaSuccess != err) {
      fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
      exit(EXIT_FAILURE);
   }
}
__device__ Hermonics gpuCalculateSphericalHermonics( struct Surfel surfel, int seed )
{
   double red[9];
   double green[9];
   double blue[9];
   double areas[9];
   for( int i = 0; i < 9; i++ )
   {
      red[i] = 0;
      green[i] = 0;
      blue[i] = 0;
      areas[i] = 0;
   }

   double area = PI * surfel.radius * surfel.radius;
   Color inColor;
   inColor.r = fmin(surfel.color.r, (float)0.999);
   inColor.g = fmin(surfel.color.g, (float)0.999);
   inColor.b = fmin(surfel.color.b, (float)0.999);
   surfel.normal = gpuUnit(surfel.normal);
   curandState s;
   curand_init(seed, 0, 0, &s);

   //Weighted Stocasically sample phi from 0 to 2pi

   double TYlm[9];
   double phi;
   double theta;

   double sin_theta;
   double cos_theta;
   double sin_phi;
   double cos_phi;
   double dx;
   double dy;
   double dz;
   double tx,ty,tz;

   double d_dot_n;
   double x,y;
   for( int j = 0; j < MONTE_CARLO_N; j++ )
   {
      for( int i = 0; i < MONTE_CARLO_N; i++ )
      {
         x = ((double)j + curand_uniform_double(&s) ) / MONTE_CARLO_N;
         y = ((double)i + curand_uniform_double(&s)) / MONTE_CARLO_N;

         phi = 2.0 * PI * y;
         theta = 2.0 * acos( sqrt( 1.0 - x ) );

         sin_theta = sin(theta);
         cos_theta = cos(theta);
         sin_phi = sin(phi);
         cos_phi = cos(phi);
         dx = sin_theta*cos_phi;
         dy = sin_theta*sin_phi;
         dz = cos_theta;

         d_dot_n = dx * (double)surfel.normal.x;
         d_dot_n += dy * (double)surfel.normal.y;
         d_dot_n += dz * (double)surfel.normal.z;

         tx = sin_theta * cos_phi;
         ty = sin_theta * sin_phi;
         tz = cos_theta;
         TYlm[0] = 0.282095; //0 0
         TYlm[1] = .488603 * -ty;//1 -1
         TYlm[2] = .488603 * tz;//1 0
         TYlm[3] = .488603 * -tx; //1 1
         TYlm[4] = 1.092548 * tx * ty; // 2 -2
         TYlm[5] = 1.092548 * -ty * tz; //2 -1
         TYlm[6] = 0.315392 * (3*tz*tz - 1); //2 0
         TYlm[7] = 1.092548 * -tx * tz; //2 1
         TYlm[8] = .546274 * (tx*tx - ty*ty); //2 2

         //now > 0
         if(d_dot_n > 0.001)
         {
            for( int k = 0; k < 9; k++ )
            {
               double a = (area * d_dot_n * TYlm[k] * sin_theta);
               /*
               //Red
               red[k] += (double)inColor.r * a;
               //Green
               green[k] += (double)inColor.g * a;
               //Blue
               blue[k] += (double)inColor.b * a;
               //area
               */
               areas[k] += a;
            }
         }
      }
   }
   for( int k = 0; k < 9; k++ )
   {
      red[k] = areas[k] * inColor.r;
      green[k] = areas[k] * inColor.g;
      blue[k] = areas[k] * inColor.b;
   }
   double factor = ((4.0*PI)/((double)MONTE_CARLO_N*(double)MONTE_CARLO_N));
   for( int j = 0; j < 9; j++ )
   {
      red[j] *= factor;
      green[j] *= factor;
      blue[j] *= factor;
      areas[j] *= factor;
   }
   Hermonics sh;
   for( int i =0; i < 9; i++ )
   {
      sh.red[i] = red[i];
      sh.green[i] = green[i];
      sh.blue[i] = blue[i];
      sh.area[i] = areas[i];
   }
   return sh;
}
__device__ Hermonics gpuCreateHermonics()
{
   Hermonics sh;
   for( int i =0; i< 9; i++ )
   {
      sh.red[i] = 0;
      sh.green[i] = 0;
      sh.blue[i] = 0;
      sh.area[i] = 0;
   }
   return sh;
}
void cpuAddHermonics( Hermonics &save, Hermonics &gone )
{
   for( int j= 0; j < 9; j++ )
   {
      save.red[j] += gone.red[j];
      save.green[j] += gone.green[j];
      save.blue[j] += gone.blue[j];
      save.area[j] += gone.area[j];
   }
}
__device__ void gpuAddHermonics( Hermonics &save, Hermonics &gone )
{
   for( int j= 0; j < 9; j++ )
   {
      save.red[j] += gone.red[j];
      save.green[j] += gone.green[j];
      save.blue[j] += gone.blue[j];
      save.area[j] += gone.area[j];
   }
}
__global__ void testGPU()
{
   printf("test\n");
}
extern "C" void testME( )
{
   printf("This\n");
   testGPU<<<1,2>>>();
   cudaDeviceSynchronize();
   CUDAERRORCHECK();
   printf("Happened\n");
}
__global__ void fillLeafSphericalHermonics( CudaNode *d_root, int tree_total,
      Surfel *surfels, int surfel_total, int *d_leaf_addrs, int leaf_nodes,
      int batch, int batch_size )
{
   int leaf_addr_index = blockIdx.x + batch * batch_size;
   int surfel_index = threadIdx.x;

   if( leaf_addr_index >= leaf_nodes)
      return;
   if( surfel_index > 32 || surfel_index < 0 )
      return;
   leaf_addr_index = 0;
   surfel_index = 0;
   __shared__ int leaf_node_index;
   __shared__ int surfel_start;
   __shared__ int surfel_end;
   leaf_node_index = d_leaf_addrs[leaf_addr_index];
   surfel_start = d_root[leaf_node_index].children[0];
   surfel_end = d_root[leaf_node_index].children[1];
   leaf_node_index = 0 ;
   surfel_start = 0;
   surfel_end = 20;

   if( leaf_node_index >= tree_total || surfel_start < 0 || surfel_end < 0 || surfel_start >= surfel_total || surfel_end >= surfel_total )
      return;
   if( surfel_start == surfel_end )
   {
      if( threadIdx.x == 0 )
         d_root[leaf_node_index].hermonics = gpuCreateHermonics();
      return;
   }
   __shared__ Hermonics hermonics[1];

   if( surfel_start + surfel_index < surfel_end )
      hermonics[surfel_index] = gpuCalculateSphericalHermonics(surfels[surfel_start + surfel_index],
            leaf_addr_index * surfel_index);

   return;
   __syncthreads();

   if( threadIdx.x == 0 )
   {
      for( int i = 1; i < surfel_end - surfel_start; i++ )
         gpuAddHermonics( hermonics[0], hermonics[i] );
      d_root[leaf_node_index].hermonics = hermonics[0];
   }
}
__global__ void kernel_FirstPassSphericalHermonics( Surfel *d_surfels, Hermonics *d_hermonics,
      int num, int batch, int batch_size )
{
   int index = (blockIdx.x + batch*batch_size) * 32 + threadIdx.x;

   if(index > num || threadIdx.x >= 32)
      return;

   Surfel s = d_surfels[index];
   Hermonics temp = gpuCalculateSphericalHermonics( s, index );

   d_hermonics[index] = temp;
}
extern "C" void gpuTestFirstPassSphericalHermonics( CudaNode *root, int nodes, SurfelArray &SA,
      int *leaf_addrs, int leaf_nodes )
{
   printf("leaf_addrs: %d, Surfels: %d, CudaNodes: %d\n", leaf_nodes, SA.num, nodes );
   float surfel_size = (float)(sizeof(Surfel) * SA.num)/1048576.0;
   float hermonics_size = (float)(sizeof(Hermonics) * SA.num)/1048576.0;
   printf("Sizes: Surfel %f Hermonics %f\n Total %f\n", surfel_size, hermonics_size,
         surfel_size + hermonics_size);
   CudaNode *d_root;
   Surfel *d_surfels;
   Hermonics *d_hermonics;
   Hermonics *hermonics = (Hermonics *) malloc( sizeof(Hermonics) * SA.num );
   int * d_leaf_addrs;
   int num_blocks = ceilf((float)SA.num / 32.0);
   int batch_size = 50;
   int batches = ceilf( (float)num_blocks/batch_size );

   dim3 dimBlock( 32 );
   dim3 dimGrid( batch_size );

   CUDASAFECALL(cudaMalloc( (void **)&d_surfels, sizeof(Surfel) * SA.num));
   CUDASAFECALL(cudaMalloc( (void **)&d_hermonics, sizeof(Hermonics) * SA.num));

   CUDASAFECALL(cudaMemcpy( d_surfels, SA.array, sizeof(Surfel) * SA.num,
            cudaMemcpyHostToDevice ));

   cudaEvent_t t0,t1;
   cudaEventCreate(&t0);
   cudaEventCreate(&t1);
   cudaEventRecord( t0, 0 );
   cudaDeviceSynchronize();
   for( int i = 0; i < batches; i++ )
   {
      kernel_FirstPassSphericalHermonics<<<dimGrid, dimBlock>>>( d_surfels, d_hermonics, SA.num,
            i, batch_size );
   }

   CUDAERRORCHECK();
   cudaEventRecord( t1, 0 );
   cudaEventSynchronize( t1 );
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, t0, t1);
   printf("Time for First Pass: %f\n", elapsedTime/1000 );
   cudaEventDestroy( t1 );
   cudaEventDestroy( t0 );

   CUDASAFECALL(cudaMemcpy( hermonics, d_hermonics, sizeof(Hermonics) * SA.num,
            cudaMemcpyDeviceToHost ));
   CUDASAFECALL(cudaFree( d_surfels));
   CUDASAFECALL(cudaFree( d_hermonics));
   cudaDeviceSynchronize();

   printf("Starting CPU FILL\n");
   FillOutHermonicsFromArray( 0, root, hermonics );
   free(hermonics);
   printf("Ending\n");
}
extern "C" void gpuTwoPassSphericalHermonics( CudaNode *root, int nodes, SurfelArray &SA,
      int *leaf_addrs, int leaf_nodes )
{
   printf("Leaf_nodes %d", leaf_nodes );
   CudaNode *d_root;
   Surfel *d_surfels;
   Hermonics *d_hermonics;
   int * d_leaf_addrs;
   int num_blocks = ceilf((float)SA.num / 32.0);
   int batch_size = 50;
   int batches = ceilf( (float)num_blocks/batch_size );

   dim3 dimBlock( 32 );
   dim3 dimGrid( batch_size );

   CUDASAFECALL(cudaMalloc( (void **)&d_surfels, sizeof(Surfel) * SA.num));
   CUDASAFECALL(cudaMalloc( (void **)&d_hermonics, sizeof(Hermonics) * SA.num));

   CUDASAFECALL(cudaMemcpy( d_surfels, SA.array, sizeof(Surfel) * SA.num,
            cudaMemcpyHostToDevice ));

   for( int i = 0; i < batches; i++ )
   {
      kernel_FirstPassSphericalHermonics<<<dimGrid, dimBlock>>>( d_surfels, d_hermonics, SA.num,
            i, batch_size );
   }

   CUDASAFECALL(cudaFree( d_surfels));

   CUDASAFECALL(cudaMalloc( (void **)&d_root, sizeof(CudaNode) * nodes));
   CUDASAFECALL(cudaMalloc( (void **)&d_leaf_addrs, sizeof(int) * leaf_nodes));

   CUDASAFECALL(cudaMemcpy( d_root, root, sizeof(CudaNode) * nodes, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_leaf_addrs, leaf_addrs, sizeof(int) * leaf_nodes,
            cudaMemcpyHostToDevice ));

   num_blocks = ceilf((float)leaf_nodes / 32.0);
   dim3 dimGrid2( num_blocks );
   kernel_SecondPassSphericalHermonics<<<dimGrid, dimBlock>>>( d_root, nodes, d_hermonics, SA.num,
         d_leaf_addrs, leaf_nodes );


   CUDASAFECALL(cudaMemcpy( root, d_root, sizeof(CudaNode) * nodes, cudaMemcpyDeviceToHost ));
   cudaFree( d_root );
   cudaFree( d_hermonics );
   cudaFree( d_leaf_addrs );
}
__global__ void  kernel_SecondPassSphericalHermonics( CudaNode *d_root, int nodes,
      Hermonics *d_hermonics, int num, int *d_leaf_addrs, int leafs )
{
}
extern "C" void gpuFilloutSphericalHermonics( CudaNode *root, int nodes, SurfelArray &SA,
      int *leaf_addrs, int leaf_nodes )
{
   int batch_size = 5;
   int batches = (leaf_nodes + batch_size - 1)/batch_size;
   dim3 dimBlock( 32 );
   dim3 dimGrid( batch_size );

   CudaNode *d_root;
   Surfel *d_surfels;
   int * d_leaf_addrs;

   printf("Leaf_nodes: %d\n", leaf_nodes );
   fflush(stdout);

   cudaEvent_t t0,t1,t2,t3;
   cudaEventCreate(&t0);
   cudaEventCreate(&t1);
   cudaEventCreate(&t2);
   cudaEventCreate(&t3);
   cudaEventRecord( t0, 0 );

   CUDASAFECALL(cudaMalloc( (void **)&d_root, sizeof(CudaNode) * nodes));
   CUDASAFECALL(cudaMalloc( (void **)&d_surfels, sizeof(Surfel) * SA.num));
   CUDASAFECALL(cudaMalloc( (void **)&d_leaf_addrs, sizeof(int) * leaf_nodes));

   CUDASAFECALL(cudaMemcpy( d_root, root, sizeof(CudaNode) * nodes, cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_surfels, SA.array, sizeof(Surfel) * SA.num,
            cudaMemcpyHostToDevice ));
   CUDASAFECALL(cudaMemcpy( d_leaf_addrs, leaf_addrs, sizeof(int) * leaf_nodes,
            cudaMemcpyHostToDevice ));

   cudaDeviceSynchronize();

   cudaEventRecord( t1, 0 );
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, t0, t1);
   printf("time %f\n", elapsedTime );

   CUDAERRORCHECK();
   for( int i = 0; i < batches; i++ )
   {
      fillLeafSphericalHermonics<<<dimGrid, dimBlock>>>( d_root, nodes, d_surfels, SA.num,
            d_leaf_addrs, leaf_nodes, 0, batch_size );
      cudaDeviceSynchronize();
      CUDAERRORCHECK();
      fprintf(stderr, "%d/%d\n", i, batches );
   }
   CUDAERRORCHECK();
   cudaEventRecord( t2, 0 );
   float elapsedTime2;
   cudaEventElapsedTime(&elapsedTime2, t1, t2);
   printf("Batched Time %f\n", elapsedTime );


   CUDASAFECALL(cudaMemcpy( root, d_root, sizeof(CudaNode) * nodes, cudaMemcpyDeviceToHost ));
   cudaFree( d_root );
   cudaFree( d_surfels );
   cudaFree( d_leaf_addrs );
}
cudaEvent_t times[100];
int current = 0;
extern "C" int getTime( )
{
   if( current >= 100 )
      return -1;
   cudaEventCreate(times+current);
   cudaEventRecord( times[current], 0 );
   current++;
   return current-1;
}
extern "C" float getDiffTime( int start, int end )
{
   if(start < 0 || end < 0)
      return -1;
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, times[start], times[end]);
   return elapsedTime;
}
/*
   extern "C" Hermonics gpuCalculateSphericalHermonics( Surfel &surfel, int sqrt_samples )
   {
   dim3 dimBlock( sqrt_samples );
   dim3 dimGrid( leaf_nodes );

   Hermonics *d_hermonics;
   CUDASAFECALL(cudaMalloc( (void **)&d_hermonics, sizeof(Hermonics)));

   gpu_kernel_calc_spherical_hermonics( surfel, sqrt_samples, d_hermoncis );
   Hermonics ret;

   CUDASAFECALL(cudaMemcpy( ret, d_hermonics, sizeof(Hermonics), cudaMemcpyDeviceToHost ));
   }
 */
void FillOutHermonicsFromArray( int current, CudaNode *root, Hermonics *hermonics )
{
   root[current].hermonics = createHermonics();
   if( root[current].leaf )
   {
      for( int i = root[current].children[0]; i < root[current].children[1]; i++ )
         cpuAddHermonics(root[current].hermonics, hermonics[i]);
      return;
   }
   for( int i = 0; i < 8; i++ )
   {
      FillOutHermonicsFromArray(root[current].children[i], root, hermonics );
      cpuAddHermonics(root[current].hermonics, root[root[current].children[i]].hermonics);
   }

}
