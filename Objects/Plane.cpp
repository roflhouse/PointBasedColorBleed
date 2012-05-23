/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Plane.h"

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
   distance = distance / normal.mag();
   normal = normal.unit();
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
   parsePigment( file, plane.info );
   parseFinish( file, plane.info );
   parseTransforms( file, plane.info );

   glm::vec4 n = glm::vec4( normal.x, normal.y, normal.z, 1 );

   n = transpose * n ;
   normal.x = n[0];
   normal.y = n[1];
   normal.z = n[2];
   normal = normal.unit();

   plane.point = point;
   plane.normal = normal;
   plane.distance = distance;
   return plane;

   //Parsing transforms uses up the ending bracket so no need to read to it
}


