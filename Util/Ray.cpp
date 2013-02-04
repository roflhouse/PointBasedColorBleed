/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Ray.h"

#define PI 3.141592
#define MAXDEPTH 15
#define MAX_ANGLE 1.0

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
   printf("malloced\n");
   for( int i = 0; i < height; i++)
   {
      for( int j = 0; j < width ; j ++ )
      {
         float u = cam.l + (cam.r-cam.l)*((float)j)/(float)width;
         float v = cam.b + (cam.t-cam.b)*((float)i)/(float)height;
         float w = -1;
         int c = i*width + j;

         (*rays)[c].pos = cam.pos;
         (*rays)[c].dir.x = growth*u * rightUnitX + growth * v * upUnitX + -w * uv.x;
         (*rays)[c].dir.y = growth*u * rightUnitY + growth * v * upUnitY + -w * uv.y;
         (*rays)[c].dir.z = growth*u * rightUnitZ + growth * v * upUnitZ + -w * uv.z;
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
         float w = -1;
         int c = i*width + j;

         (*rays)[c].pos = cam.pos;
         (*rays)[c].dir.x = u * rightUnitX + v * upUnitX + -w * uv.x;
         (*rays)[c].dir.y = u * rightUnitY + v * upUnitY + -w * uv.y;
         (*rays)[c].dir.z = u * rightUnitZ + v * upUnitZ + -w * uv.z;
         (*rays)[c].i = i;
         (*rays)[c].j = j;
      }
   }
   return width * height;
}
void castRays( const Scene &scene, Ray *rays, int numRays, Color *buffer, int width )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i * width + rays[i].j] = raytrace( scene, rays[i] );
   }
}
void castRaysSphere( const Scene &scene, Ray *rays, int numRays, Color *buffer, int width )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i*width + rays[i].j] = raytrace2( scene, rays[i] );
   }
}
void castRays( const SurfelArray &surfels, Ray *rays, int numRays, Color *buffer, int width )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i*width + rays[i].j] = raytrace( surfels, rays[i] );
   }
}
void initCuberays( vec3 ***cuberays )
{
   //cube size does not matter.

   //Front
   for( int i = 0; i < 8; i++ )
   {
      for( int j =0; j < 8; j++ )
      {
         vec3 ray;
         //front 
         ray.x = -1 + (0.25/*2.0/8.0*/) * j;
         ray.y = 1 - (0.25) * i;
         ray.z = 1;
         cuberays[0][i][j] = unit(ray); 
         //right
         ray.x = 1;
         ray.y = 1 - .25 * j;
         ray.z = 1 - .25 * i;
         cuberays[1][i][j] = unit(ray);
         //back
         ray.x = 1 - .25 * j; 
         ray.y = 1 - .25 * i;
         ray.z = -1;
         cuberays[2][i][j] = unit(ray);
         //left
         ray.x = -1;
         ray.y = 1 - .25 * j;
         ray.z = -1 + .25 *i;
         cuberays[3][i][j] = unit(ray);
         //bottom
         ray.x = -1 + .25  * j;
         ray.y = -1;
         ray.z = 1 - .25 * i;
         cuberays[4][i][j] = unit(ray);
         //top
         ray.x = -1 + .25 * j;
         ray.y = -1;
         ray.z = -1 + .25 * i;
         cuberays[5][i][j] = unit(ray);
      }
   }
}
void castRays( const TreeNode &tree, Ray *rays, int numRays, Color *buffer, int width )
{
   vec3 cuberays[6][8][8];
   initCuberays( cuberays );
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i*width + rays[i].j] = raytrace( tree, rays[i], cuberays );
   }
}
void castRays( const ArrayNode *tree, int size, struct SurfelArray &SA, Ray *rays, int numRays, Color *buffer, int width )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i*width + rays[i].j] = raytrace( tree, size, SA, rays[i] );
   }
}
SurfelArray createSurfels( const Scene &scene, Ray *rays, int numRays )
{
   IntersectionArray IA = createIntersectionArray();

   for( int i = 0; i < numRays; i++ )
   {
      collectIntersections( scene, rays[i], IA );
   }
   shrinkIA( IA );
   SurfelArray SA = createSurfelArray();
   for( int i = 0; i < IA.num; i++ )
   {
      addToSA( SA, intersectionToSurfel( IA.array[i], scene ) );
   }
   shrinkSA( SA );
   return SA;
}
ArrayNode *createSurfelsCuda( const Scene &scene, Ray *rays, int numRays, SurfelArray &SA, int &size )
{
   vec3 min;
   vec3 max;
   IntersectionArray IA = createIntersectionArray();

   for( int i = 0; i < numRays; i++ )
   {
      collectIntersections( scene, rays[i], IA );
   }
   shrinkIA( IA );
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
   freeIntersectionArray( IA );
   shrinkSA( SA );

   return createOctreeForCuda( SA, min, max, size );
}
TreeNode createSurfelTree( const Scene &scene, Ray *rays, int numRays )
{
   vec3 min;
   vec3 max;
   IntersectionArray IA = createIntersectionArray();

   for( int i = 0; i < numRays; i++ )
   {
      collectIntersections( scene, rays[i], IA );
   }
   shrinkIA( IA );
   SurfelArray SA = createSurfelArray();
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

   return createOctree( SA, min, max );
}
Scene createSurfelSpheres( const Scene &scene, Ray *rays, int numRays )
{
   IntersectionArray IA = createIntersectionArray();

   for( int i = 0; i < numRays; i++ )
   {
      if( rays[i].i  == 10 && rays[i].j == 10 )
         collectIntersections( scene, rays[i], IA );
   }
   shrinkIA( IA );

   Scene scene2;
   scene2.spheres = (Sphere *) malloc(sizeof( Sphere ) * IA.num );
   scene2.numSpheres = IA.num;
   for( int i = 0; i < IA.num; i++ )
   {
      //addToSA( SA, intersectionToSurfel( IA.array[i], scene ) );
      scene2.spheres[i] = intersectionToSphere( IA.array[i], scene );
   }
   return scene2;
}
void collectIntersections( const Scene &scene, const Ray &ray, IntersectionArray &IA )
{
   float t;
   int i = 0;
   for( int j = 0; j < scene.numTriangles; j++ )
   {
      t = triangleHitTest( scene.triangles[j], ray );
      if( t > 0 )
      {
         addToIA( IA,  triangleIntersection( scene.triangles[j], ray, t ));
         i++;
      }
   }
   for( int j = 0; j < scene.numSpheres; j++ )
   {
      float_2 sphereT = sphereHitTest( scene.spheres[j], ray );
      if( sphereT.t0 > 0 )
      {
         addToIA( IA, sphereIntersection( scene.spheres[j], ray, sphereT.t0 ) );
         i++;
      }
      if( sphereT.t1 > 0 )
      {
         addToIA( IA, sphereIntersection( scene.spheres[j], ray, sphereT.t1 ) );
         i++;
      }
   }
   for( int j = 0; j < scene.numPlanes; j++ )
   {
      t = planeHitTest( scene.planes[j], ray );
      if( t > 0 )
      {
         addToIA( IA, planeIntersection( scene.planes[j], ray, t ));
         i++;
      }
   }
   //printf("Total: %d %d\n", IA.num, i );
}
Color raytrace( const struct Scene &scene, const Ray &ray )
{
   Color color;
   color.r = 0;
   color.b = 0;
   color.g = 0;

   Intersection best;
   best.hit = false;

   float bestT = 100000;
   float t;
   for( int j = 0; j < scene.numSpheres; j++ )
   {
      float_2 s = sphereHitTest( scene.spheres[j], ray );
      if( s.t0 > 0 )
      {
         if( !best.hit || s.t0 < bestT )
         {
            best = sphereIntersection( scene.spheres[j], ray, s.t0 );
            bestT = s.t0;
         }
      }
      else if( s.t1 > 0 )
      {
         if( !best.hit || s.t1 < bestT )
         {
            best = sphereIntersection( scene.spheres[j], ray, s.t1 );
            bestT = s.t0;
         }
      }
   }
   for( int j = 0; j < scene.numTriangles; j++ )
   {
      t = triangleHitTest( scene.triangles[j], ray );
      if( t > 0 )
      {
         if( !best.hit || t < bestT )
         {
            best = triangleIntersection( scene.triangles[j], ray, t );
            bestT = t;
         }
      }
   }
   for( int j = 0; j < scene.numPlanes; j++ )
   {
      t = planeHitTest( scene.planes[j], ray );
      if( t > 0 )
      {
         if( !best.hit || t < bestT )
         {
            best = planeIntersection( scene.planes[j], ray, t );
            bestT = t;
         }
      }
   }
   if( best.hit )
   {
      color = directIllumination( best, scene );
      //printf("color: %f, %f, %f\n", color.r, color.g, color.b);
   }
   return limitColor( color );
}
Color raytrace2( const struct Scene &SA, const Ray &ray )
{
   Color color;
   color.r = 0;
   color.b = 0;
   color.g = 0;

   bool hit = false;
   float bestT = 10000;
   float_2 s;
   for( int j = 0; j < SA.numSpheres; j++ )
   {
      s = sphereHitTest( SA.spheres[j], ray );
      if( s.t0 > 0 )
      {
         if( !hit || s.t0 < bestT )
         {
            color = SA.spheres[j].info.colorInfo.pigment;
            bestT = s.t0;
            hit = true;
         }
      }
      else if( s.t1 > 0 )
      {
         if( !hit || s.t1 < bestT )
         {
            color = SA.spheres[j].info.colorInfo.pigment;
            bestT = s.t1;
            hit = true;
         }
      }
   }
   return limitColor( color );
}
Color raytrace( const struct SurfelArray &SA, const Ray &ray )
{
   Color color;
   color.r = 0;
   color.b = 0;
   color.g = 0;

   bool hit = false;
   float bestT = 10000;
   float t;
   for( int j = 0; j < SA.num; j++ )
   {
      t = surfelHitTest( SA.array[j], ray );
      if( t > 0 )
      {
         if( !hit || t < bestT )
         {
            color = SA.array[j].color;
            bestT = t;
            hit = true;
         }
      }
   }
   return limitColor( color );
}
Color raytrace( const struct TreeNode &tree, const Ray &ray, vec3 ***cuberay )
{
   Color color;
   color.r = 0;
   color.b = 0;
   color.g = 0;

   TreeHitMark cur = transTree( tree, ray );
   if ( cur.t > 0 )
      RasterCube cube;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            float ndotr = dot(cur.surfel.normal, cuberay[i][j][k]);
            if( ndotr < 0.001 )
            { 
               cube.sides[i][j][k] = 0;
               cube.depth[i][j][k] = -1;
            }
            else {
               cube.sides[i][j][k] = 0;
               cube.depth[i][j][k] = 101;
            }
         }
   vec3 hit;
   hit.x = ray.pos.x + ray.dir.x * cur.t;
   hit.y = ray.pos.y + ray.dir.y * cur.t;
   hit.z = ray.pos.z + ray.dir.z * cur.t;

   traverseOctreeCPU( cube, tree, MAX_ANGLE, hit, cur.surfel.normal, cuberay );
   return cur.color;
   else
      return color;
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
Color raytrace( const struct ArrayNode *tree, int size, SurfelArray &SA, const Ray &ray )
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
      if( testForHit( tree[now].box, ray ) )
      {
         if( tree[now].leaf )
         {
            for( int j = tree[now].children[0]; j < tree[now].children[1]; j++ )
            {
               t = surfelHitTest( SA.array[j], ray );
               if( t > 0 )
               {
                  if( !hit || t < bestT )
                  {
                     color = SA.array[j].color;
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

void traverseOctreeCPU( RasterCube &cube, TreeNode &node, float maxangle,
      vec3 &position, vec3 normal, vec3 ***cuberays )
{
   if( node.leaf == 1 )
   {
      float distance = 0;
      for( int i = 0; i < sa.num; i++ )
      {
         distance = distance( position, sa.array[i].pos );
         if ( distance < sa.radius )
            raytraceSurfelToCube( cube, sa.array[i], cuberays, position );
         else
            rasterizeSufelToCube( cube, sa.array[i] );
      }
   }
   else
   {
      if( belowHorizon( node.box, position, normal ) )
         return;
      vec3 center;
      center = newDirection(node.box.max, node.box.min);
      center.x /= 2;
      center.y /= 2;
      center.z /= 2;

      vec3 centerToEye = newDirection( position,center );
      centerToEye = unit(centerToEye);

      float distance = distance( center, position ); 
      float area = evaluateSphericalHermonicsArea( node, eyeToNode );
      float solidangle = area / (distance *distance);
      if( solidangle < maxangle )
      {
         evaluateSphereicalHermonicsPower( );
         //rasterize the cluster as a disk
      }
      else
         for( int i = 0; i < 8; i++)
            if( node.children[i] != NULL )
               traverseOctreeCPU( cube, node, maxangle, position );
   }
}
void rasterizeSurfelToCube( RasterCube &cube, Surfel &surfel, vec3 &position )
{
}
void rayTraceSurfelToCube( RasterCube &cube, Surfel &surfel, vec3 ***cuberays, vec3 &position )
{
   for( int i = 0; i < 6; i++ )
   {
      for( int j = 0; j < 8; j ++ )
      {
         for( int k = 0; k < 8; k++ )
         {
            if( cube.depth[i][j][k] > 0.0001 )
            {
               Ray ray;
               ray.dir = cuberays[i][j][k];
               ray.pos = position;
               float t = surfelHitTest( surfel, ray );
               if( t > 0 && t < cube.depth[i][j][k] )
               {
                  cube.depth[i][j][k] = t;
                  cube.sides[i][j][k] = surfel.color;
               }
            }
         }
      }
   }
}
void evaluateSphereicalHermonicsArea( TreeNode &node, vec3 &centerToEye )
{
   theta = acosf( centerToEye.z );
   phi = atanf( centerToEye.y / centerToEye.x );
   float sin_theta = sinf(theta);
   float cos_theta = cosf(theta);
   float cos_phi = cosf(phi);
   float sin_phi = sinf(phi); 
   float * TYlm = getYLM( sin_theta *cos_phi, sin_theta * sin_phi, cos_theta ); 
   float area = 0;

   for( int i =0; i < 9; i++ )
   {
      area += node.hermonics.area[i] * TYlm[i];
   }
   return area;
}
void evaluateSphereicalHermonicsPower( TreeNode &node, vec3 &centerToEye )
{
   theta = acosf( centerToEye.z );
   phi = atanf( centerToEye.y / centerToEye.x );
   float sin_theta = sinf(theta);
   float cos_theta = cosf(theta);
   float cos_phi = cosf(phi);
   float sin_phi = sinf(phi); 
   float * TYlm = getYLM( sin_theta *cos_phi, sin_theta * sin_phi, cos_theta ); 
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
   return color;
}

/*
   void rasterizeSurfelVaribleVectors( RasterCube &cube, Intersection &position, Surfel &surfel ) {
//rasterizeSurfelToSide( SIDEOFCUBE, up vec, right vec, in vec, surfel )
vec3 up = cube.up;
vec3 in = cube.in;
vec3 right = cube.right;
vec3 down = neg(cube.up);
vec3 left = neg(cube.right);
vec3 out = neg(cube.in);
rasterizeSurfelToSide( cube.topface, in, right, down, surfel );

rasterizeSurfelToSide( cube.frontface, up, right, in,  surfel );
rasterizeSurfelToSide( cube.backface, up, left, out, surfel);

rasterizeSurfelToSide( cube.rightface, up, in, left, surfel);
rasterizeSurfelToSide( cube.leftface, up, out, right, surfel);

}
void createCubeVectors( RasterCube &cube, Intersection &position )
{
cube.up = position.normal;
//Find another vector for right
//find 2 smallest components
if(position.normal.x < position.normal.y)
{
if(position.normal.y < position.normal.z)
{
cube.right.x = position.hitMark.x + 1;
cube.right.y = position.hitMark.y + 1;

cube.right.z = (position.normal.x *(cube.right.x - position.hitMark.x) +
position.normal.y *(cube.right.y - position.hitMark.y)) / position.normal.z
+ position.hitMark.z;
}
else
{
cube.right.x = position.hitMark.x + 1;
cube.right.z = position.hitMark.z + 1;

cube.right.y = (position.normal.x *(cube.right.x - position.hitMark.x) +
position.normal.z *(cube.right.z - position.hitMark.z)) / position.normal.y
+ position.hitMark.y;
}
}
else
{
if(position.normal.x < position.normal.z)
{
cube.right.x = position.hitMark.x + 1;
cube.right.y = position.hitMark.y + 1;

cube.right.z = (position.normal.x *(cube.right.x - position.hitMark.x) +
position.normal.y *(cube.right.y - position.hitMark.y)) / position.normal.z
+ position.hitMark.z;
}
else
{
cube.right.y = position.hitMark.y + 1;
cube.right.z = position.hitMark.z + 1;

cube.right.x = (position.normal.y *(cube.right.y - position.hitMark.y) +
position.normal.z *(cube.right.z - position.hitMark.z)) / position.normal.x
+ position.hitMark.x;
}
}
cube.right = unit(cube.right);
cube.in = cross( cube.up, cube.right );
//Have camera vectors for sides of cubes.
}
 */
