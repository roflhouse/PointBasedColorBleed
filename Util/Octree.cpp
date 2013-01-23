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
#define MONTE_CARLO_N 128

int glob;
int count( TreeNode *root );
int buildOctreeArray( TreeNode *tree, ArrayNode *octree, int &cur, SurfelArray &SA );
TreeNode createOctree( SurfelArray &SA, vec3 min, vec3 max )
{
   TreeNode *root = (TreeNode *) malloc( sizeof(TreeNode) );

   root->box = createBoundingBox( min, max );
   root->SA = SA;
   printf("first %d\n", SA.num );
   if( root->SA.num > 32 )
   {
      root->leaf = false;
      BoundingBox *boxes = getSubBoxes( root->box );
      for( int i = 0; i < 8; i++ )
         root->children[i] = createTreeNode( root, boxes[i], 1 );
      freeSurfelArray( SA );
   }
   else
      root->leaf = true;
   //printf("Octree finished %d\n", sizeof(Surfel) * SA.num);
   int numberNodes = count( root );
   printf("Octree constructed filling out SphericalHermonics for nodes\n");

   filloutHermonics( root );

   printf("Octree finished %d\n", numberNodes);
   return *root;
}
int count( TreeNode *root )
{
   if( root == NULL )
   {
      printf("NULL\n");
      return 0;
   }
   if( root->leaf )
   {
      glob += root->SA.num;
      if( root->SA.num )
         return 1;
      else
         return 0;
   }
   else
   {
      int ret = 0;
      for( int i = 0; i < 8; i++ )
      {
         ret += count( root->children[i] );
      }
      return ret + 1;
   }
}
ArrayNode *createOctreeForCuda( SurfelArray &SA, vec3 min, vec3 max, int &size )
{
   glob = 0;
   TreeNode *root = (TreeNode *) malloc( sizeof(TreeNode) );

   root->box = createBoundingBox( min, max );
   root->SA = SA;
   printf("first %d\n", SA.num );
   if( root->SA.num > 32 )
   {
      root->leaf = false;
      BoundingBox *boxes = getSubBoxes( root->box );
      for( int i = 0; i < 8; i++ )
         root->children[i] = createTreeNode( root, boxes[i], 1 );
      free( boxes );
   }
   else
      root->leaf = true;

   int numberNodes = count( root );
   shrinkSA(SA);

   printf("Octree finished %d, %d, %d\n", glob, SA.num, numberNodes);

   ArrayNode *octree = (ArrayNode *) malloc ( sizeof(ArrayNode) * numberNodes );
   int cur = 0;
   freeSurfelArray( SA );
   SA = createSurfelArray();
   buildOctreeArray( root, octree, cur, SA );
   shrinkSA( SA );
   size = numberNodes;
   /*for( int i = 0; i < size; i++ )
     {
   //printf("Octree %d\n", i);
   //printf("Leaf %d\n", octree[i].leaf );
   for( int j = 0; j< 8; j++ )
   //printf("\tChild: %d\n", octree[i].children[j] );
   }
    */
   return octree;
}
int buildOctreeArray( TreeNode *tree, ArrayNode *octree, int &cur, SurfelArray &SA )
{
   int mySpot = cur;
   ArrayNode temp;
   temp.leaf = tree->leaf;
   temp.box = tree->box;
   if( temp.leaf )
   {
      if( tree->SA.num )
      {
         int adding = tree->SA.num;
         temp.children[0] = SA.num;
         temp.children[1] = SA.num+adding;
         if( SA.num +adding > SA.max )
         {
            growSA( SA );
         }
         memcpy( &(SA.array[SA.num]), tree->SA.array, adding * sizeof(Surfel) );
         SA.num += adding;
         freeSurfelArray( tree->SA );
         cur++;
         octree[mySpot] = temp;
         free( tree );
         return mySpot;
      }
      else
      {
         free( tree );
         return -1;
      }
   }
   else
   {
      cur++;
      for( int i = 0; i < 8; i++ )
      {
         temp.children[i] = buildOctreeArray( tree->children[i], octree, cur, SA );
      }
      octree[mySpot] = temp;
      free( tree );
      return mySpot;
   }
}
TreeNode *createTreeNode( TreeNode *root, const BoundingBox &box, int depth )
{
   TreeNode *ret = (TreeNode *) malloc ( sizeof( TreeNode ) );
   ret->hermonics = createHermonics();
   ret->box = box;
   ret->SA = createSurfelArray();
   for( int i = 0; i < root->SA.num; i++ )
   {
      if( isIn( ret->box, root->SA.array[i].pos ) )
         addToSA( ret->SA, root->SA.array[i] );
   }
   shrinkSA( ret->SA );
   if( ret->SA.num > 32 && depth < 15 )
   {
      ret->leaf = false;
      BoundingBox *boxes = getSubBoxes( ret->box );
      for( int i = 0; i < 8; i++ )
         ret->children[i] = createTreeNode( ret, boxes[i], depth+1 );
      freeSurfelArray( ret->SA );
      free( boxes );
   }
   else
      ret->leaf = true;

   return ret;
}
Hermonics calculateSphericalHermonics( struct Surfel &surfel )
{
   Hermonics sh;
   for( int i = 0; i < 9; i++ )
   {
      sh.red[i] = 0;
      sh.green[i] = 0;
      sh.blue[i] = 0;
      sh.area[i] = 0;
   }

   float area = PI * surfel.radius * surfel.radius;

   //Weighted Stocasically sample phi from 0 to 2pi
   float simple_spacing = 1.0 / MONTE_CARLO_N;

   //Sum
   for( int j = 0; j < MONTE_CARLO_N-1; j++ )
   {
      float phi_j = j * simple_spacing;
      //Random Float 0->1
      float r = (float)rand() / (float)RAND_MAX;
      phi_j +=  r * simple_spacing;
      float phi = 2.0 * PI * phi_j;
      for( int i = 0; i < MONTE_CARLO_N-1; i++ )
      {
         float theta_i = i * simple_spacing;
         r = (float)rand() / (float)RAND_MAX;
         theta_i += r * simple_spacing;

         float theta = 2 * acosf( sqrt( 1 - theta_i ) );

         float sin_theta = sinf(theta);
         float cos_theta = cosf(theta);
         float sin_phi = sinf(phi);
         float cos_phi = cosf(phi);
         vec3 d = {sin_theta * cos_phi, sin_theta * sin_phi, cos_theta };
         float d_dot_n = dot( d, surfel.normal );

         float *TYlm = getYLM( sin_theta * cos_phi, sin_theta* sin_phi, cos_theta );

         for( int i = 0; i < 9; i++ )
         {
            //Red
            sh.red[i] += surfel.color.r * area * d_dot_n + TYlm[i] * sin_theta;
            //Green
            sh.green[i] += surfel.color.g * area * d_dot_n + TYlm[i] * sin_theta;
            //Blue
            sh.blue[i] += surfel.color.b * area * d_dot_n + TYlm[i] * sin_theta;
            //area
            sh.area[i] += area* d_dot_n + TYlm[i] *sin_theta;
         }
         free( TYlm );
      }
   }

   //Average
   averageHermonics( sh, ((4*PI)/(float)MONTE_CARLO_N));
   return sh;
}
float *getYLM( float x, float y, float z )
{
   const static float Ylm[9] = { 0.282095, .488603,.488603,.488603,
      1.092548, 1.092548, 1.092548, 0.315392, .546274 };

   float *ret = (float *)malloc( sizeof(float) * 9 );
   ret[0] = Ylm[0];
   ret[1] = Ylm[1] * x;
   ret[2] = Ylm[2] * z;
   ret[3] = Ylm[3] * y;
   ret[4] = Ylm[4] * x * z;
   ret[5] = Ylm[5] * y * z;
   ret[6] = Ylm[6] * x * y;
   ret[7] = Ylm[7] * (3*z*z - 1);
   ret[8] = Ylm[8] * (x*x - y*y);
   return ret;
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
      save.red[j] += factor;
      save.green[j] += factor;
      save.blue[j] += factor;
      save.area[j] += factor;
   }
}

void filloutHermonics( TreeNode *root )
{
   if( root->leaf )
   {
      if( root->SA.num > 0 )
         root->hermonics = calculateSphericalHermonics(root->SA.array[0]);
      for(int i = 1; i < root->SA.num; i++ )
      {
         Hermonics temp = calculateSphericalHermonics( root->SA.array[0] );
         addHermonics( root->hermonics, temp );
      }
   }
   else
   {
      for( int j = 0; j < 8; j++ )
      {
         filloutHermonics( root->children[j] );
         addHermonics( root->hermonics, root->children[j]->hermonics );
      }
   }
}
