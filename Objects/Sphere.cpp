/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Sphere.h"

float sphereHitTest( const Sphere &sphere, const Ray &ray )
{
   vec3 direction = unit( ray.dir );
   glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
   glm::vec4 pos = glm::vec4(ray.pos.x, ray.pos.y, ray.pos.z, 1.0f);
   dir = sphere.info.transforms*dir;
   pos = sphere.info.transforms*pos;
   float xc = sphere.pos.x;
   float yc = sphere.pos.y;
   float zc = sphere.pos.z;
   float x0 = pos[0];
   float y0 = pos[1];
   float z0 = pos[2];
   float xd = dir[0];
   float yd = dir[1];
   float zd = dir[2];

   float A = xd*xd + yd*yd + zd*zd;
   float B = 2*(xd *(x0-xc) + yd*(y0-yc) + zd*(z0-zc));
   float C = (x0-xc)*(x0-xc) + (y0-yc)*(y0-yc) + (z0-zc)*(z0-zc) - sphere.radius*sphere.radius;
   float disc = B*B -4*A*C;
   if(disc < .0001)
      return -1;

   float t0 = (-B - sqrt(disc))/2;
   if ( t0 < 0.001)
   {
      t0 = (-B + sqrt(disc))/2;
   }
   if( t0 <= 0.001 )// && t0 >= -.00001 )
      return -1;
   return t0;
}
Intersection sphereIntersection( const Sphere &sphere, const Ray &ray, float t0 )
{
   vec3 direction = unit( ray.dir );
   Intersection ret;
   glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
   glm::vec4 pos = glm::vec4(ray.pos.x, ray.pos.y, ray.pos.z, 1.0f);
   dir = sphere.info.transforms * dir;
   pos = sphere.info.transforms * pos;
   float xc = sphere.pos.x;
   float yc = sphere.pos.y;
   float zc = sphere.pos.z;
   float x0 = ray.pos.x;
   float y0 = ray.pos.y;
   float z0 = ray.pos.z;
   float xd = direction.x;
   float yd = direction.y;
   float zd = direction.z;
   ret.viewVector.x = -direction.x;
   ret.viewVector.y = -direction.y;
   ret.viewVector.z = -direction.z;
   ret.hitMark.x = x0 + xd*t0;
   ret.hitMark.y = y0 + yd*t0;
   ret.hitMark.z = z0 + zd*t0;

   vec3 objHit;
   objHit.x = pos[0] + dir[0]* t0;
   objHit.y = pos[1] + dir[1]* t0;
   objHit.z = pos[2] + dir[2]* t0;

   ret.normal.x = (objHit.x - xc)/sphere.radius;
   ret.normal.y = (objHit.y - yc)/sphere.radius;
   ret.normal.z = (objHit.z - zc)/sphere.radius;
   ret.normal = unit(ret.normal);
   glm::vec4 n = glm::vec4( ret.normal.x, ret.normal.y, ret.normal.z, 1 );
   n = sphere.info.transpose * n;
   ret.normal.x = n[0];
   ret.normal.y = n[1];
   ret.normal.z = n[2];
   ret.normal = unit(ret.normal);
   ret.colorInfo = sphere.info.colorInfo;

   ret.hit = true;
   return ret;
}

Sphere parseSphere( FILE *file )
{
   Sphere sphere;
   char cur = '\0';

   //location
   while(cur != '<')
   {
      if(fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error parsing sphere\n");
         exit(1);
      }
   }
   if( fscanf(file, " %f, %f, %f ", &(sphere.pos.x), &(sphere.pos.y), &(sphere.pos.z) ) == EOF )
   {
      printf("Error parsing sphere\n");
      exit(1);
   }

   printf( "location: %f %f %f\n", sphere.pos.x, sphere.pos.y, sphere.pos.z );
   cur = '\0';

   //radius
   //Read in everything until , so next item is radius
   while( cur != ',' )
   {
      if(fscanf( file, "%c", &cur) == EOF)
      {
         printf("Error parsing sphere\n");
         exit(1);
      }
   }
   if( fscanf(file, "%f", &sphere.radius) == EOF )
   {
      printf("Error parsing sphere\n");
      exit(1);
   }
   printf( "radius: %f \n", sphere.radius );


   sphere.info = createObjectInfo();
   parseObjectPigment( file, sphere.info );
   parseObjectFinish( file, sphere.info );
   parseObjectTransforms( file, sphere.info );

   return sphere;
}

