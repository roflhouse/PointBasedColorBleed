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
void castRays( const Scene &scene, Ray *rays, int numRays, Color **buffer )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i][rays[i].j] = raytrace( scene, rays[i] );
   }
}
void castRaysSphere( const Scene &scene, Ray *rays, int numRays, Color **buffer )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i][rays[i].j] = raytrace2( scene, rays[i] );
   }
}
void castRays( const SurfelArray &surfels, Ray *rays, int numRays, Color **buffer )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i][rays[i].j] = raytrace( surfels, rays[i] );
   }
}
void castRays( const TreeNode &tree, Ray *rays, int numRays, Color **buffer )
{
   for( int i = 0; i < numRays; i++ )
   {
      buffer[rays[i].i][rays[i].j] = raytrace( tree, rays[i] );
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

   float bestT = 10000;
   float t;
   for( int j = 0; j < scene.numSpheres; j++ )
   {
      float_2 s = sphereHitTest( scene.spheres[j], ray );
      if( s.t0 > 0 )
      {
         if( !best.hit || s.t0 < bestT )
         {
            best = sphereIntersection( scene.spheres[j], ray, s.t0 );
            bestT = t;
         }
      }
      if( s.t1 > 0 )
      {
         if( !best.hit || s.t1 < bestT )
         {
            best = sphereIntersection( scene.spheres[j], ray, s.t1 );
            bestT = t;
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
      if( s.t1 > 0 )
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
Color raytrace( const struct TreeNode &tree, const Ray &ray )
{
   Color color;
   color.r = 0;
   color.b = 0;
   color.g = 0;

   TreeHitMark cur = transTree( tree, ray );
   if ( cur.t > 0 )
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
