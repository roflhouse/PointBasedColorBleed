/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Parser.h"
#include <stdlib.h>

Scene parseFile( std::string filename )
{
   Scene scene;
   int maxSpheres = 1000;
   int maxTriangles = 1000;
   int maxPlanes = 1000;
   int maxPointLights = 10;
   scene.numPointLights = 0;
   scene.numSpheres = 0;
   scene.numTriangles = 0;
   scene.numPlanes = 0;
   scene.spheres = (Sphere *)malloc( sizeof(Sphere) * maxSpheres );
   scene.triangles = (Triangle *)malloc( sizeof(Triangle) * maxTriangles );
   scene.planes = (Plane *)malloc( sizeof(Plane) * maxSpheres );
   scene.pointLights = (PointLight *)malloc( sizeof(PointLight) * maxPointLights );

   //Open file for writing
   FILE *file = fopen(filename.c_str(), "r");
   if(file == NULL)
   {
      printf("Error Occured opening file\n");
      exit(EXIT_FAILURE);
   }

   while(1)
   {
      //starting off eating all whitespace
      char cur = ' ';
      while( isspace(cur) )
      {
         if( fscanf( file, "%c", &cur ) == EOF )
         {
            printf("End of File reached\n");
            return scene;
         }
      }

      //check for comment
      if(cur == '/')
      {
         while(cur != '\n')
         {
            if( fscanf( file, "%c", &cur ) == EOF )
            {
               printf("Error Occured reading file\n");
               exit(EXIT_FAILURE);
            }
         }
      }
      else if( cur == 'c' || cur == 'C' )
      {
         scene.camera = parseCamera( file );
      }
      else if( cur == 'l' || cur == 'L' )
      {
         PointLight light = parsePointLight( file );
         if( scene.numPointLights + 1 >= maxPointLights )
         {
            maxPointLights = maxPointLights*10;
            scene.pointLights = (PointLight *)realloc( scene.pointLights, sizeof(PointLight)*maxPointLights );
         }
         scene.pointLights[scene.numPointLights] = light;
         scene.numPointLights++;
      }
      else if( cur == 's' || cur == 'S' )
      {
         if( scene.numSpheres+1 >= maxSpheres )
         {
            maxSpheres = maxSpheres*1000;
            scene.spheres = (Sphere *) realloc( scene.spheres, sizeof(Sphere) * maxSpheres );
         }
         scene.spheres[scene.numSpheres] = parseSphere(file);
         scene.numSpheres++;
      }
      else if( cur == 'p' || cur == 'P' )
      {
         if( scene.numPlanes+1 >= maxPlanes )
         {
            maxPlanes = maxPlanes*1000;
            scene.planes = (Plane *) realloc( scene.planes, maxPlanes * sizeof(Plane) );
         }
         scene.planes[scene.numPlanes] = parsePlane(file);
         scene.numPlanes++;
      }
      else if( cur == 't' || cur == 'T' )
      {
         if( scene.numTriangles+1 >= maxTriangles )
         {
            maxTriangles = maxTriangles*1000;
            scene.triangles = (Triangle *) realloc( scene.triangles, maxTriangles * sizeof(Triangle) );
         }
         scene.triangles[scene.numTriangles] = parseTriangle(file);
         scene.numTriangles++;
      }
      else
      {
         printf("Unknown Keyword Failure char was |%c|\n", cur);
         exit(EXIT_FAILURE);
      }
   }
}
