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
#include "Tga.h"

#define PI 3.141592
#define MAXDEPTH 15
#define MAX_ANGLE 1
#define FAR_PLANE -100.0
#define NEAR_PLANE -1
#define RIGHT 1
#define LEFT -1
#define TOP 1
#define BOTTOM -1
#define NPIXELS 8

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
         ray.z = 1 - (2.0/8.0) * i -half;
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

   //bottom w = pos y, pos x, v = neg z
   glm::mat4 *bottom = &((*cubetrans)[4]);
   *bottom = glm::mat4(1.0); //build ident
   (*bottom)[0] = x;
   (*bottom)[1] = -z;
   (*bottom)[2] = y;

   //top w = -y, u = pos x, v = pos z
   glm::mat4 *top = &((*cubetrans)[5]);
   *top = glm::mat4(1.0); //build ident
   (*top)[0] = x;
   (*top)[1] = z;
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
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i*width + rays[i].j] = raytrace( tree, rays[i], (vec3 ***)cuberays, cubetrans );
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
            buffer[(7-j)*8 + k] = cube.sides[i][j][k];
         }
      }
      outfile.writeTga( s.str().c_str() );
   }
   for( int i = 0; i < 6; i++ )
   {
      std::stringstream s;
      s << "Output/side-" << num << "-" <<i << "depth.tga";
      printf("%s\n",s.str().c_str());
      Tga outfile( 8, 8 );
      Color *buffer = outfile.getBuffer();
      for( int j = 0; j < 8; j++ )
      {
         for( int k =0; k < 8; k++ )
         {
            float dep = cube.depth[i][j][k]/10;
            float b = 0;
            if( dep < 10 && dep > 0 )
               printf( "dep %f\n", dep*10 );
            if( dep < 0 )
               dep = 1;
            else if( dep < 10 )
            {
               dep = 0;
               b = 1;
            }
            Color c;
            c.r = 0;
            c.g = dep;
            c.b = b;
            buffer[(7-j)*8 + k] = c;
         }
      }
      outfile.writeTga( s.str().c_str() );
   }
}
Color raytrace( const struct TreeNode &tree, const Ray &ray, vec3 ***cuberay, glm::mat4 *cubetrans )
{
   static int it = 0;
   static int q = 0;
   static bool first = true;
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
      float dis = 0;
      int num = 0;
      color.r = 0;
      color.g = 0;
      color.b = 0;
      float mind = 100;
      for( int i = 0; i <6; i++)
         for( int j = 0; j<8; j++)
            for( int k =0; k<8;k++)
            {
               if( cube.depth[i][j][k] < -FAR_PLANE +1 && cube.depth[i][j][k] > -0.0001 )
               {
                  num++;
                  if(cube.sides[i][j][k].r > 0 )
                     color.r += cube.sides[i][j][k].r;
                  if(cube.sides[i][j][k].g > 0 )
                     color.g += cube.sides[i][j][k].g;
                  if(cube.sides[i][j][k].b > 0 )
                     color.b += cube.sides[i][j][k].b;
                  dis += cube.depth[i][j][k];
                  if (mind > cube.depth[i][j][k])
                     mind = cube.depth[i][j][k];
               }
            }
      dis /= num;

      if( num > 0 && !(it % 10)  )
         printf( "num: %d, %d, %f %f %f, %f\n", it, num, color.r, color.g, color.b, dis );

      if( color.r > 1 )
         color.r = 1;
      if( color.g > 1 )
         color.g = 1;
      if( color.b > 1 )
         color.b = 1;
      return color;
   }
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
                  curser++;
               }
            }
         }
      }
   }

   return color;
}

void traverseOctreeCPU( RasterCube &cube, const TreeNode &node, float maxangle,
      vec3 &position, vec3 &normal, vec3 ***cuberays, glm::mat4 *cubetransforms)
{
   if( node.leaf == 1 )
   {
      float dis = 0;
      for( int i = 0; i < node.SA.num; i++ )
      {
         dis = distance( position, node.SA.array[i].pos );
         if ( dis < node.SA.array[i].radius )
         {
            //raytraceSurfelToCube( cube, node.SA.array[i], cuberays, position, normal );
            //rasterizeSurfelToCube( cube, node.SA.array[i], cubetransforms, cuberays,
            //      position, normal );
         }
         else
         {
            //raytraceSurfelToCube( cube, node.SA.array[i], cuberays, position, normal );
            rasterizeSurfelToCube( cube, node.SA.array[i], cubetransforms, cuberays,
                  position, normal );
         }
      }
   }
   else
   {
      if( belowHorizon( node.box, position, normal ) )
         return;


      for( int i = 0; i < 8; i++)
      {
         if( node.children[i] != NULL )
         {
            traverseOctreeCPU( cube, *node.children[i], maxangle, position, normal, cuberays, cubetransforms );
         }
      }
      return;

      vec3 center;
      center = newDirection(node.box.max, node.box.min);
      center.x /= 2;
      center.y /= 2;
      center.z /= 2;

      vec3 centerToEye = newDirection( position,center );
      centerToEye = unit(centerToEye);

      float dis = distance( center, position );
      float area = evaluateSphericalHermonicsArea( node, centerToEye );
      float solidangle = area / (dis *dis);
      if( solidangle < maxangle )
      {
         Color c = evaluateSphericalHermonicsPower( node, centerToEye );
         rasterizeClusterToCube( cube, c, area, getCenter(node.box), cubetransforms,
               cuberays, position, normal );
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
      glm::mat4 *cubetransforms, vec3 ***cuberays, vec3 &position, vec3 &normal)
{
   const static glm::mat4 M = getViewPixelMatrix() * getOrthMatrix() * getProjectMatrix();
   glm::mat4 eyeTrans = glm::mat4(1.0);
   eyeTrans[0][3] = -position.x;
   eyeTrans[1][3] = -position.y;
   eyeTrans[2][3] = -position.z;
   float length = sqrtf(area);
   for( int k = 0; k< 6; k++ )
   {
      glm::mat4 cur = M * cubetransforms[k] * eyeTrans;
      glm::vec4 middle = glm::vec4( nodePosition.x, nodePosition.y, nodePosition.z, 1.0 );
      glm::vec4 temp = cur * middle;
      temp[0] /= temp[3];
      temp[1] /= temp[3];
      temp[2] /= temp[3];
      temp[3] = 1;
      if (temp[0] > -0.0001 && temp[0] < NPIXELS+0.001 && temp[1] > -0.001 && temp[1] < NPIXELS+.001 )
      {
         glm::vec4 *points = getAxisAlinedPoints( nodePosition, length/2.0, k );
         points[0] = cur * points[0];
         points[1] = cur * points[1];
         points[2] = cur * points[2];
         points[3] = cur * points[3];
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
         for( int i = 0; i < 4; i++ )
         {
            if( minX > points[i][0] )
               minX = roundf(points[i][0]);
            if( minY > points[i][1] )
               minY = roundf(points[i][1]);
            if( maxX < points[i][0] )
               maxX = roundf(points[i][0]);
            if( maxY < points[i][1] )
               maxY = roundf(points[i][1]);
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

            float dis = distance( position, nodePosition );
            for( int i = minY; i <= maxY; i++ )
            {
               for( int j = minX; j <= maxX; j++ )
               {
                  float d = dot( cuberays[k][i][j], normal );
                  if (d < 0.01 )// || cube.depth[k][i][j] < 0)
                     continue;
                  float atten = 1/ (1 +2*dis+ 3*dis * dis);
                  if( atten > 1 )
                     atten = 1;
                  cube.sides[k][i][j].r = c.r*d *atten;
                  cube.sides[k][i][j].g = c.g*d *atten;
                  cube.sides[k][i][j].b = c.b*d *atten;
                  cube.depth[k][i][j] = dis;

                  cube.sides[k][i][j] = c;
                  cube.depth[k][i][j] = dis;
               }
            }
         }
         delete []points;
      }
   }
}
void rasterizeSurfelToCube( RasterCube &cube, Surfel &surfel, glm::mat4 *cubetransforms,
      vec3 ***cuberays, vec3 &position, vec3 &normal )
{
   const static glm::mat4 M = getViewPixelMatrix() * getOrthMatrix() * getProjectMatrix();
   const static glm::vec4 *wVecs = getWVecs();
   vec3 diff = newDirection( surfel.pos, position );
   diff = unit(diff);
   float dotPro = dot(normal, diff);
   if( dotPro < 0 )
      return;
   //get projected area for each side
   double area = surfel.radius *surfel.radius * PI;
   double areas[6];
   glm::vec3 Snormal = glm::vec3( surfel.normal.x, surfel.normal.y, surfel.normal.z );
   for( int i =0; i < 6; i++ )
   {
      glm::vec3 t = glm::vec3(wVecs[i][0], wVecs[i][1], wVecs[i][2]);
      areas[i] = glm::dot( t, Snormal ) * area;
   }

   glm::mat4 eyeTrans = glm::mat4(1.0);
   eyeTrans[3][0] = -position.x;
   eyeTrans[3][1] = -position.y;
   eyeTrans[3][2] = -position.z;

   //For each face
   for( int k = 0; k< 6; k++ )
   {
      if( areas[k] < 0.0000001 )
      {
         continue;
      }
      glm::mat4 cur = M * cubetransforms[k] * eyeTrans;
      glm::vec4 middle = glm::vec4( surfel.pos.x, surfel.pos.y, surfel.pos.z, 1.0 );
      glm::vec4 temp = cur * middle;
      temp[0] /= temp[3];
      temp[1] /= temp[3];
      temp[2] /= temp[3];
      temp[3] = 1;
      if (temp[0] > -0.001 && temp[0] < NPIXELS+0.001 && temp[1] > -0.001
            && temp[1] < NPIXELS+.001 )
      {
         /*if( false &&surfel.pos.y < -3.5 && position.y > -3 )
           {
           printf("\n\nMID: %f %f %f\n", temp[0], temp[1], temp[2] );
           printf("surfel: %f %f %f\n", surfel.pos.x, surfel.pos.y, surfel.pos.z );
           printf("snorm: %f %f %f\n", surfel.normal.x, surfel.normal.y, surfel.normal.z );
           printf("pos: %f %f %f\n", position.x, position.y, position.z );
           printf("normal: %f %f %f\n", normal.x, normal.y, normal.z );
           printf("diff: %f %f %f\n", diff.x, diff.y, diff.z );
           printf("dot: %f\n", dot( normal, diff ) );
           printf("Z: %f\n", temp[2] );
           }
          */
         //printf("RASTER\n");
         double length = sqrt( areas[k] );
         glm::vec4 *points = getAxisAlinedPoints( surfel.pos, length/2.0, k );
         points[0] = cur * points[0];
         points[1] = cur * points[1];
         points[2] = cur * points[2];
         points[3] = cur * points[3];
         for( int i = 0; i < 4; i++ )
         {
            points[i][0] /= points[i][3];
            points[i][1] /= points[i][3];
            points[i][2] /= points[i][3];
            points[i][3] = 1;
            //printf("Point: %f %f %f\n", points[i][0], points[i][1], points[i][2] );
         }
         int minX = 0;
         int minY = 0;
         int maxX = 0;
         int maxY = 0;
         minX = points[0][0];
         maxX = points[0][0];
         minY = points[0][1];
         maxY = points[0][1];
         for( int i = 0; i < 4; i++ )
         {
            if( minX > points[i][0] )
               minX = roundf(points[i][0]);
            if( minY > points[i][1] )
               minY = roundf(points[i][1]);
            if( maxX < points[i][0] )
               maxX = roundf(points[i][0]);
            if( maxY < points[i][1] )
               maxY = roundf(points[i][1]);
         }
         //printf("minY: %d, maxY: %d, minX: %d, maxX: %d\n", minY, maxY, minX, maxX );
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
            float dis = distance( position, surfel.pos );
            for( int i = minY; i <= maxY; i++ )
            {
               for( int j = minX; j <= maxX; j++ )
               {
                  float d = dot( cuberays[k][j][i], normal );
                  if (d < 0.001 || cube.depth[k][j][i] < 0)
                     continue;
                  float atten = 1/ (1 +8*dis * dis);
                  if( atten > 1 )
                     atten = 1;
                  cube.sides[k][i][j].r = surfel.color.r*d *atten;
                  cube.sides[k][i][j].g = surfel.color.g*d *atten;
                  cube.sides[k][i][j].b = surfel.color.b*d *atten;
                  cube.depth[k][i][j] = dis;
               }
            }
         }
         delete []points;
      }
   }
}

void raytraceSurfelToCube( RasterCube &cube, Surfel &surfel, vec3 ***cuberays, vec3 &position,
      vec3 &normal )
{
   for( int i = 0; i < 6; i++ )
   {
      for( int j = 0; j < 8; j ++ )
      {
         for( int k = 0; k < 8; k++ )
         {
            float dotProd = dot(cuberays[i][j][k], normal);
            if( dotProd > 0.01 )
            {
               Ray ray;
               ray.dir = cuberays[i][j][k];
               ray.pos = position;
               ray.pos.x += ray.dir.x*.001;
               ray.pos.y += ray.dir.y*.001;
               ray.pos.z += ray.dir.z*.001;
               float t = surfelHitTest( surfel, ray );
               if( t > 0 && t < cube.depth[i][j][k] )
               {
                  cube.depth[i][j][k] = t;
                  cube.sides[i][j][k] = surfel.color;
                  cube.sides[i][j][k].r *= dotProd;
                  cube.sides[i][j][k].g *= dotProd;
                  cube.sides[i][j][k].b *= dotProd;
                  vec3 hit;
                  hit.x = ray.pos.x + ray.dir.x * t;
                  hit.y = ray.pos.y + ray.dir.y * t;
                  hit.z = ray.pos.z + ray.dir.z * t;
                  /*printf( "srt %f %f %f %f\n", t, position.x, position.y, position.z );
                    printf( "hit %d %f %f %f\n", i, hit.x, hit.y, hit.z );
                    printf( "dir %f %f %f %f\n", t, ray.dir.x, ray.dir.y, ray.dir.z );
                    printf( "nor %f %f %f %f\n", t, surfel.normal.x, surfel.normal.y, surfel.normal.z );
                    printf( "col %f %f %f %f\n", t, surfel.color.r, surfel.color.g, surfel.color.b );
                    printf( "pos %f %f %f %f\n", t, surfel.pos.x, surfel.pos.y, surfel.pos.z );
                   */
               }
            }
         }
      }
   }
}
float evaluateSphericalHermonicsArea( const TreeNode &node, vec3 &centerToEye )
{
   float theta = acosf( centerToEye.z );
   float phi = atanf( centerToEye.y / centerToEye.x );
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
Color evaluateSphericalHermonicsPower( const TreeNode &node, vec3 &centerToEye )
{
   float theta = acosf( centerToEye.z );
   float phi = atanf( centerToEye.y / centerToEye.x );
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
