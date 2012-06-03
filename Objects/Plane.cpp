/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Plane.h"
float planeHitTest(const Plane &plane, const Ray &ray )
{
   vec3 direction = unit(ray.dir);
   vec3 position;
   vec3 normal = unit(plane.normal);
   glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
   glm::vec4 pos = glm::vec4(ray.pos.x, ray.pos.y, ray.pos.z, 1.0f);
   dir = plane.info.transforms*dir;
   pos = plane.info.transforms*pos;
   direction.x = dir[0];
   direction.y = dir[1];
   direction.z = dir[2];
   position.x = pos[0];
   position.y = pos[1];
   position.z = pos[2];

   //float above = dot( normal, position ) + plane.distance;
   //if( above < 0 )
   //{
     // normal.x = -normal.x;
      //normal.y = -normal.y;
      //normal.z = -normal.z;
   //}
   float vd = dot(normal, direction);
   if( vd < 0.0001 && vd > -0.0001 )
      return -1;
   float v0 = -(dot(position, plane.normal) - plane.distance );
   vec3 viewVector;
   vec3 hitMark;
   float t = v0/vd;
   viewVector.x = -ray.dir.x;
   viewVector.y = -ray.dir.y;
   viewVector.z = -ray.dir.z;
   hitMark.x = pos[0] + direction.x*t;
   hitMark.y = pos[1] + dir[1]*t;
   hitMark.z = pos[2] + dir[2]*t;
   if( ray.i == 10 && ray.j == 10 )
      printf("play ray dir:%f %f %f pos: %f %f %f  surf n: %f %f %f  t: %f dis: %f vd: %f v0: %f hit: %f %f %f \n", direction.x, direction.y, direction.z, ray.pos.x, ray.pos.y, ray.pos.z, plane.normal.x, plane.normal.y, plane.normal.z, t, plane.distance, vd, v0, hitMark.x, hitMark.y, hitMark.z );
   if( ray.i == 30 && ray.j == 30 )
   {
      printf("this happened %f\n", t );
   }
   if( t < 0.001)
      return -1;
   return t;
}
Intersection planeIntersection( const Plane &plane, const Ray &ray, float t )
{
   Intersection ret;
   ret.hit = true;
   vec3 direction = unit( ray.dir );
   glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
   glm::vec4 pos = glm::vec4(ray.pos.x, ray.pos.y, ray.pos.z, 1.0f);
   dir = plane.info.transforms * dir;
   pos = plane.info.transforms * pos;
   ret.viewVector.x = -ray.dir.x;
   ret.viewVector.y = -ray.dir.y;
   ret.viewVector.z = -ray.dir.z;

   ret.hitMark.x = pos[0] + dir[0]*t;
   ret.hitMark.y = pos[1] + dir[1]*t;
   ret.hitMark.z = pos[2] + dir[2]*t;
   ret.normal = plane.normal;
   ret.colorInfo = plane.info.colorInfo;
   return ret;
}
Plane parsePlane( FILE *file )
{
   Plane plane;
   float distance;
   vec3 point;
   vec3 normal;
   char cur = '\0';
   //location
   while(cur != '<')
   {
      if(fscanf(file, "%c", &cur) == EOF)
      {
         printf("Plane normal not valid\n");
         exit(1);
      }
   }
   if( fscanf(file, " %f , %f , %f ", &(normal.x), &(normal.y), &(normal.z) ) == EOF )
   {
      printf("Plane normal not valid\n");
      exit(1);
   }
   printf( "normal: %f %f %f\n", normal.x, normal.y, normal.z );

   cur = '\0';
   //distance
   //Read in everything until , so next item is distance
   while( cur != ',' )
   {
      if(fscanf( file, "%c", &cur) == EOF)
      {
         printf("Plane normal not valid\n");
         exit(1);
      }

   }
   if( fscanf(file, "%f", &distance) == EOF )
   {
      printf("Plane normal not valid\n");
      exit(1);
   }
   printf( "distance: %f \n", distance );
   distance = distance / mag(normal);
   normal = unit(normal);
   //A != 0
   if( normal.x > 0.0001 || normal.x < -0.0001 )
   {
      point.x = distance / normal.x;
      point.y = 0;
      point.z = 0;
   }
   //B != 0
   else if( normal.y > 0.0001 || normal.y < -0.0001 )
   {
      point.x = 0;
      point.y = distance / normal.y;
      point.z = 0;
   }
   //C != 0
   else if( normal.z > 0.0001 || normal.z < -0.0001 )
   {
      point.x = 0;
      point.y = 0;
      point.z = distance / normal.z;
   }
   else
   {
      printf("Plane normal not valid\n");
      exit(1);
   }
   plane.info = createObjectInfo();
   parseObjectPigment( file, plane.info );
   parseObjectFinish( file, plane.info );
   parseObjectTransforms( file, plane.info );

   glm::vec4 n = glm::vec4( normal.x, normal.y, normal.z, 1 );

   n = plane.info.transpose * n ;
   normal.x = n[0];
   normal.y = n[1];
   normal.z = n[2];
   normal = unit(normal);

   plane.point = point;
   plane.normal = normal;
   plane.distance = distance;
   return plane;

   //Parsing transforms uses up the ending bracket so no need to read to it
}


