/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Octree.h"
#define PI 3.14159265359
#define MONTE_CARLO_N 256
#define MAX_DEPTH  30
#define MAX_OCTREE_SIZE 10

int glob;

extern "C" void gpuFilloutSphericalHermonics( CudaNode *root, int nodes, SurfelArray &SA, 
            int *gpu_leaf_addrs, int leaf_nodes );
extern "C" void gpuTestFirstPassSphericalHermonics( CudaNode *root, int nodes, SurfelArray &SA, 
            int *gpu_leaf_addrs, int leaf_nodes );

void addToNode( TreeNode *root, const Surfel &surfel, int depth )
{
   if( root->leaf && root->SA.num >= 31 && depth < MAX_DEPTH )
   {
      BoundingBox *boxes = getSubBoxes( root->box );
      for( int i = 0; i < 8; i++ )
      {
         root->children[i] = (TreeNode *)malloc( sizeof(TreeNode) );
         root->children[i]->leaf = true;
         root->children[i]->SA = createSurfelArray( 33 );
         root->children[i]->box = boxes[i];
         root->children[i]->numInNode = 0;
         clearHermonics( root->children[i]->hermonics );
      }
      free(boxes);

      for( int i = 0; i < root->SA.num ; i++ )
      {
         for( int k = 0; k < 8; k++ )
         {
            if( isIn( root->children[k]->box, root->SA.array[i].pos ) )
            {
               addToNode( root->children[k], root->SA.array[i], depth+1 );
               break;
            }
         }
      }
      root->leaf = false;
      freeSurfelArray( root->SA );
   }
   if( root->leaf )
   {
      addToSA( root->SA, surfel );
      root->numInNode++;
   }
   else
   {
      for( int k = 0; k < 8; k++ )
      {
         if( isIn( root->children[k]->box, surfel.pos ) )
         {
            addToNode( root->children[k], surfel, depth+1 );
            root->numInNode++;
            return;
         }
      }
   }
}
TreeNode createOctreeMark2( SurfelArray &SA, vec3 min, vec3 max )
{
   int num = SA.num;
   TreeNode *root = (TreeNode *) malloc( sizeof(TreeNode) );

   min.x -= .00001;
   min.y -= .00001;
   min.z -= .00001;
   max.x += 0.00001;
   max.y += 0.00001;
   max.z += 0.00001;
   min.x = fmax( min.x, -MAX_OCTREE_SIZE );
   min.y = fmax( min.y, -MAX_OCTREE_SIZE );
   min.z = fmax( min.z, -MAX_OCTREE_SIZE );
   max.x = fmin( max.x, MAX_OCTREE_SIZE );
   max.y = fmin( max.y, MAX_OCTREE_SIZE );
   max.z = fmin( max.z, MAX_OCTREE_SIZE );
   printf("%f %f %f, %f %f %f\n", min.x, min.y, min.z, max.x, max.y, max.z );

   root->box = createBoundingBox( min, max );
   root->leaf = true;
   root->SA = createSurfelArray( 33 );
   root->numInNode = 0;
   clearHermonics( root->hermonics );

   printf("Createing Octree\n");
   int curPercent = 0;
   int lastPercent = 0;
   for( int i =0; i < SA.num; i++ )
   {
      addToNode( root, SA.array[i], 0 );
      curPercent = (float)i / SA.num * 100;
      if( curPercent > lastPercent )
      {
         printf("Percent Complete: %d   \r", curPercent);
         lastPercent = curPercent;
      }
   }

   printf("Percent Complete: 100   \r\n" );
   printf("Filling Out Hermonics: num: %d\n", SA.num);
   printf("Percent Complete: 0    \r");
   filloutHermonics( root, num );
   printf("Percent Complete: 100   \n");
   return *root;
}
int octreeToCudaTree( TreeNode *cpu_root, CudaNode* gpu_root, int current_node,
      SurfelArray &gpu_array )
{
   gpu_root[current_node].leaf = cpu_root->leaf;
   gpu_root[current_node].box = cpu_root->box;
   if( cpu_root->leaf )
   {
      gpu_root[current_node].children[0] = gpu_array.num;
      gpu_root[current_node].children[1] = gpu_array.num + cpu_root->SA.num;
      for( int i =2; i < 8; i++ )
         gpu_root[current_node].children[i] = -1;
      for( int i = 0; i < cpu_root->SA.num; i++ )
         addToSA( gpu_array, cpu_root->SA.array[i] );
      return current_node+1;
   }
   else
   {
      int child_node = current_node+1;
      for( int i = 0; i < 8; i++ )
      {
         gpu_root[current_node].children[i] = child_node;
         child_node = octreeToCudaTree( cpu_root->children[i], gpu_root, child_node, gpu_array );
      }
      return child_node;
   }
}
void createCudaTree( SurfelArray cpu_array, vec3 min, vec3 max, CudaNode* &gpu_root, int &nodes,
      SurfelArray &gpu_array )
{
   int total;
   if( gpu_root != NULL )
   {
      printf("Improper use of createCudaTree\n");
      exit(1);
   }

   int num = cpu_array.num;
   TreeNode *cpu_root = (TreeNode *) malloc( sizeof(TreeNode) );

   min.x -= 0.00001;
   min.y -= 0.00001;
   min.z -= 0.00001;
   max.x += 0.00001;
   max.y += 0.00001;
   max.z += 0.00001;
   min.x = fmax( min.x, -MAX_OCTREE_SIZE );
   min.y = fmax( min.y, -MAX_OCTREE_SIZE );
   min.z = fmax( min.z, -MAX_OCTREE_SIZE );
   max.x = fmin( max.x, MAX_OCTREE_SIZE );
   max.y = fmin( max.y, MAX_OCTREE_SIZE );
   max.z = fmin( max.z, MAX_OCTREE_SIZE );

   cpu_root->box = createBoundingBox( min, max );
   cpu_root->leaf = true;
   cpu_root->SA = createSurfelArray( 33 );
   cpu_root->numInNode = 0;
   //clearHermonics( cpu_root->hermonics );

   printf("Createing Octree\n");
   int curPercent = 0;
   int lastPercent = 0;
   for( int i =0; i < cpu_array.num; i++ )
   {
      if( isIn( cpu_root->box, cpu_array.array[i].pos ) )
         addToNode( cpu_root, cpu_array.array[i], 0 );
      curPercent = (float)i / cpu_array.num * 100;
      if( curPercent > lastPercent )
      {
         printf("Percent Complete: %d   \r", curPercent);
         lastPercent = curPercent;
      }
   }
   printf("Coverting to CudaTree\n");

   int leaf_nodes = 0;
   int *gpu_leaf_addr;
   if( gpu_root == NULL )
   {
      leaf_nodes = countLeafNodes( cpu_root );
      total = countNodes(cpu_root);
      nodes = total;
      gpu_root = (CudaNode *)malloc( sizeof( CudaNode ) * total );
      gpu_array = createSurfelArray( cpu_root->numInNode );
      gpu_leaf_addr = (int *) malloc( sizeof( int ) * leaf_nodes );
   }

   octreeToCudaTree( cpu_root, gpu_root, 0, gpu_array );
   int retLeafs = getLeafAddrs( gpu_root, 0, gpu_leaf_addr, 0 );
   if(leaf_nodes != retLeafs )
   {
      printf("Mismatch leafs\n");
      exit(1);
   }

   printf("...Complete\n");

   printf("Generating Spherical Hermonics on GPU\n");
   //gpuFilloutSphericalHermonics( gpu_root, total, gpu_array, gpu_leaf_addr, leaf_nodes );
   gpuTestFirstPassSphericalHermonics( gpu_root, total, gpu_array, gpu_leaf_addr, leaf_nodes );
   printf("...Complete\n");
}
int getLeafAddrs( CudaNode *gpu_root, int node, int *leaf_addrs, int current )
{
   if( gpu_root[node].leaf )
   {
      leaf_addrs[current] = node;
      return current+1;
   }
   else
   {
      for( int i = 0; i < 8; i++ )
         current = getLeafAddrs( gpu_root, gpu_root[node].children[i], leaf_addrs, current );
      return current;
   }
}
int countNodes( TreeNode *root )
{
   if( root->leaf )
      return 1;
   else
   {
      int ret = 1;
      for( int i = 0; i< 8; i++ )
         ret += countNodes( root->children[i] );
      return ret;
   }
}
int countLeafNodes( TreeNode *root )
{
   if( root->leaf )
      return 1;
   else
   {
      int ret = 0;
      for( int i = 0; i< 8; i++ )
         ret += countLeafNodes( root->children[i] );
      return ret;
   }
}
Hermonics calculateSphericalHermonics( struct Surfel &surfel )
{
   double red[9];
   double green[9];
   double blue[9];
   double areas[9];
   Hermonics sh;
   for( int i = 0; i < 9; i++ )
   {
      red[i] = 0;
      green[i] = 0;
      blue[i] = 0;
      areas[i] = 0;
   }

   double area = PI * surfel.radius * surfel.radius;

   //Weighted Stocasically sample phi from 0 to 2pi

   //Sum
   for( int j = 0; j < MONTE_CARLO_N; j++ )
   {
      for( int i = 0; i < MONTE_CARLO_N; i++ )
      {
         double r = (double)rand() / (double)RAND_MAX;
         double x = ((double)j + r) / MONTE_CARLO_N;
         r = (double)rand() / (double)RAND_MAX;
         double y = ((double)i + r) / MONTE_CARLO_N;

         double phi = 2.0 * PI * y;
         double theta = 2.0 * acos( sqrt( 1.0 - x ) );

         double sin_theta = sin(theta);
         double cos_theta = cos(theta);
         double sin_phi = sin(phi);
         double cos_phi = cos(phi);
         double dx = sin_theta*cos_phi;
         double dy = sin_theta*sin_phi;
         double dz = cos_theta;

         double d_dot_n = dx * (double)surfel.normal.x;
         d_dot_n += dy * (double)surfel.normal.y;
         d_dot_n += dz * (double)surfel.normal.z;

         double *TYlm = getYLM( sin_theta * cos_phi, sin_theta* sin_phi, cos_theta );

         //now > 0
         if(d_dot_n > 0.0)
         {
            for( int k = 0; k < 9; k++ )
            {
               //Red
               red[k] += ((double)surfel.color.r * (double)TYlm[k] * (double)area *(double)d_dot_n);
               //Green
               green[k] += surfel.color.g * TYlm[k] * area * d_dot_n;
               //Blue
               blue[k] += surfel.color.b  *TYlm[k] * area * d_dot_n;
               //area
               areas[k] += (area * d_dot_n * TYlm[k]);
            }
         }
         free( TYlm );
      }
   }
   for( int i =0; i < 9; i++ )
   {
      sh.red[i] = red[i];
      sh.green[i] = green[i];
      sh.blue[i] = blue[i];
      sh.area[i] = areas[i];
   }
   //Average
   averageHermonics( sh, ((4*PI)/((float)MONTE_CARLO_N*(float)MONTE_CARLO_N)));
   return sh;
}

double *getYLM( double x, double y, double z )
{
   const static double Ylm[9] = { 0.282095, .488603,.488603,.488603,
      1.092548, 1.092548, 1.092548, 0.315392, .546274 };

   double *ret = (double *)malloc( sizeof(double) * 9 );
   ret[0] = Ylm[0]; //0 0
   ret[1] = Ylm[3] * -y;//1 -1
   ret[2] = Ylm[2] * z;//1 0
   ret[3] = Ylm[1] * -x; //1 1
   ret[4] = Ylm[6] * x * y; // 2 -2
   ret[5] = Ylm[5] * -y * z; //2 -1
   ret[6] = Ylm[7] * (3*z*z - 1); //2 0
   ret[7] = Ylm[4] * -x * z; //2 1
   ret[8] = Ylm[8] * (x*x - y*y); //2 2
   return ret;
}
void clearHermonics( Hermonics &hermonics )
{
   for( int i =0; i< 9; i++ )
   {
      hermonics.red[i] = 0;
      hermonics.green[i] = 0;
      hermonics.blue[i] = 0;
      hermonics.area[i] = 0;
   }
}
Hermonics createHermonics()
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
void addHermonics( Hermonics &save, Hermonics &gone )
{
   for( int j= 0; j < 9; j++ )
   {
      save.red[j] += gone.red[j];
      save.green[j] += gone.green[j];
      save.blue[j] += gone.blue[j];
      save.area[j] += gone.area[j];
   }
}
void averageHermonics( Hermonics &save, float factor )
{
   for( int j = 0; j < 9; j++ )
   {
      save.red[j] *= factor;
      save.green[j] *= factor;
      save.blue[j] *= factor;
      save.area[j] *= factor;
   }
}

void filloutHermonics( TreeNode *root, int total )
{
   static int s = 0;
   static int in = 0;
   if( root->leaf )
   {
      if( root->SA.num > 0 )
      {
         root->hermonics = calculateSphericalHermonics(root->SA.array[0]);
      }
      for(int i = 1; i < root->SA.num; i++ )
      {
         Hermonics temp = calculateSphericalHermonics( root->SA.array[0] );
         addHermonics( root->hermonics, temp );
      }
      s += root->SA.num;
      printf("Percent Complete: %d    \r", (int) ((float)s/(float)total * 100) );
   }
   else
   {
      in++;
      //printf("Doing children %d\n", in);
      for( int j = 0; j < 8; j++ )
      {
         filloutHermonics( root->children[j], total );
         addHermonics( root->hermonics, root->children[j]->hermonics );
      }
   }
}
