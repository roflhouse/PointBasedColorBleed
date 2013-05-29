/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "LightSource.h"

PointLight parsePointLight( FILE *file )
{
   PointLight light;
   char cur = '\0';
   //location
   while(cur != '<')
   {
      //read in until data
      if( fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error parsing PointLight\n");
         exit(1);
      }
   }
   if( fscanf(file, " %f , %f , %f ", &(light.pos.x), &(light.pos.y), &(light.pos.z) ) == EOF )
   {
      printf("Error parsing PointLight\n");
      exit(1);
   }

   printf( " location: %f, %f, %f\n", light.pos.x, light.pos.y, light.pos.z );
   cur = '\0';

   while(cur != '<')
   {
      //read in until rgb data
      if( fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error parsing PointLight\n");
         exit(1);
      }
   }

   if( fscanf(file, " %f , %f , %f ", &(light.color.r), &(light.color.g), &(light.color.b) ) == EOF )
   {
      printf("Error parsing PointLight\n");
      exit(1);
   }

   printf( " color: %f, %f, %f\n", light.color.r, light.color.g, light.color.b );
   while(cur != '}')
   {
      if( fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error parsing PointLight\n");
         exit(1);
      }
   }
   for( int i = 0; i < 100; i++ )
   {
      vec3 r;
      r.x = (2*(float)rand() / (float)RAND_MAX) - 1;
      r.y = (2*(float)rand() / (float)RAND_MAX) - 1;
      r.z = (2*(float)rand() / (float)RAND_MAX) - 1;
      light.points[i].x = light.pos.x + r.x;
      light.points[i].y = light.pos.y + r.y;
      light.points[i].z = light.pos.z + r.z;
   }
   return light;
}
