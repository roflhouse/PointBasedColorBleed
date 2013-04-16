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
#define MAX_DEPTH  20
#define MAX_OCTREE_SIZE 1000

int glob;
int count( TreeNode *root );
int buildOctreeArray( TreeNode *tree, ArrayNode *octree, int &cur, SurfelArray &SA );
TreeNode createOctree( SurfelArray &SA, vec3 min, vec3 max )
{
   int num = SA.num;
   TreeNode *root = (TreeNode *) malloc( sizeof(TreeNode) );

   min.x -= .0001;
   min.y -= .0001;
   min.z -= .0001;
   max.x += 0.0001;
   max.y += 0.0001;
   max.z += 0.0001;
   min.x = fmax( min.x, -MAX_OCTREE_SIZE );
   min.y = fmax( min.y, -MAX_OCTREE_SIZE );
   min.z = fmax( min.z, -MAX_OCTREE_SIZE );
   max.x = fmin( max.x, MAX_OCTREE_SIZE );
   max.y = fmin( max.y, MAX_OCTREE_SIZE );
   max.z = fmin( max.z, MAX_OCTREE_SIZE );

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

   filloutHermonics( root, num );

   printf("Octree finished %d\n", numberNodes);
   return *root;
}
void addToNode( TreeNode *root, const Surfel &surfel )
{
   if( root->leaf && root->SA.num >= 31 )
   {
      BoundingBox *boxes = getSubBoxes( root->box );
      for( int i = 0; i < 8; i++ )
      {
         root->children[i] = (TreeNode *)malloc( sizeof(TreeNode) );
         root->children[i]->leaf = true;
         root->children[i]->SA = createSurfelArray( 33 );
         root->children[i]->box = boxes[i];
         clearHermonics( root->children[i]->hermonics );
      }
      free(boxes);

      for( int i = 0; i < root->SA.num ; i++ )
      {
         for( int k = 0; k < 8; k++ )
         {
            if( isIn( root->children[k]->box, root->SA.array[i].pos ) )
            {
               addToNode( root->children[k], root->SA.array[i] );
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
   }
   else
   {
      for( int k = 0; k < 8; k++ )
      {
         if( isIn( root->children[k]->box, surfel.pos ) )
         {
            addToNode( root->children[k], surfel );
            return;
         }
      }
   }
}
TreeNode createOctreeMark2( SurfelArray &SA, vec3 min, vec3 max )
{
   int num = SA.num;
   TreeNode *root = (TreeNode *) malloc( sizeof(TreeNode) );

   min.x -= .0001;
   min.y -= .0001;
   min.z -= .0001;
   max.x += 0.0001;
   max.y += 0.0001;
   max.z += 0.0001;
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
   clearHermonics( root->hermonics );

   printf("Createing Octree\n");
   int curPercent = 0;
   int lastPercent = 0;
   for( int i =0; i < SA.num; i++ )
   {
      addToNode( root, SA.array[i] );
      curPercent = (float)i / SA.num * 100;
      if( curPercent > lastPercent )
      {
         printf("Percent Complete: %d   \r", curPercent);
         lastPercent = curPercent;
      }
   }
   freeSurfelArray( SA );

   printf("Percent Complete: 100   \r\n" );
   printf("Filling Out Hermonics: num: %d\n", SA.num);
   printf("Percent Complete: 0    \r");
   filloutHermonics( root, num );
   printf("Percent Complete: 100   \n");
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
   static int p = 0;
   TreeNode *ret = (TreeNode *) malloc ( sizeof( TreeNode ) );
   ret->hermonics = createHermonics();
   ret->box = box;
   ret->SA = createSurfelArray( root->SA.num );
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
   {
      ret->leaf = true;
      p += ret->SA.num;
   }

   return ret;
}
/*int factorial( int x)
  {
  int ret = 1;
  for( int i = 1; i <= x; i++ )
  ret *= i;
  return ret;
  }
  double K(int l, int m)
  {
// renormalisation constant for SH function
double temp = ((2.0*l+1.0)*factorial(l-m)) / (4.0*PI*factorial(l+m));
return sqrt(temp);
}
double P(int l,int m,double x)
{
// evaluate an Associated Legendre Polynomial P(l,m,x) at x
double pmm = 1.0;
if(m>0) {
double somx2 = sqrt((1.0-x)*(1.0+x));
double fact = 1.0;
for(int i=1; i<=m; i++) {
pmm *= (-fact) * somx2;
fact += 2.0;
}
}
if(l==m) return pmm;
double pmmp1 = x * (2.0*m+1.0) * pmm;
if(l==m+1) return pmmp1;
double pll = 0.0;
for(int ll=m+2; ll<=l; ++ll) {
pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m);
pmm = pmmp1;
pmmp1 = pll;
}
return pll;
}
double SH(int l, int m, double theta, double phi)
{
// return a point sample of a Spherical Harmonic basis function
// l is the band, range [0..N]
// m in the range [-l..l]
// theta in the range [0..Pi]
// phi in the range [0..2*Pi]
const double sqrt2 = sqrt(2.0);
if(m==0) return K(l,0)*P(l,m,cos(theta));
else if(m>0) return sqrt2*K(l,m)*cos(m*phi)*P(l,m,cos(theta));
else return sqrt2*K(l,-m)*sin(-m*phi)*P(l,-m,cos(theta));
}
 */
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
