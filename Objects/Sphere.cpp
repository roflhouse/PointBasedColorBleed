/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Sphere.h"

Sphere parseSphere( FILE *file )
{
   Sphere sphere
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
   parsePigment( file, sphere.info );
   parseFinish( file, sphere.info );
   parseTransforms( file, sphere.info );

   return sphere;
}

