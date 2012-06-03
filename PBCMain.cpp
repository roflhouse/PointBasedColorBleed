/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#define CUDA_ENABLED

#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <ctype.h>
#include <iostream>
#include <sstream>
#include <time.h>

#include "Util/Parser.h"
#include <vector>
#include "Objects/Objects.h"
#include "Util/Ray.h"
#include "Util/Tga.h"
#include "Util/Color.h"

int width_of_image;
int height_of_image;
char *parseCommandLine(int argc, char *argv[]);

int main(int argc, char *argv[])
{
   char *filename = parseCommandLine(argc, argv);
   std::string str(filename);

   Scene scene = parseFile( str );

   Ray *rays;

   int number = createInitRays( &rays, width_of_image, height_of_image, 1.0, scene.camera );
   Tga outfile( width_of_image, height_of_image );

   Color **buffer = outfile.getBuffer();

   SurfelArray surfels = createSurfels( scene, rays, number ); 
   free( rays );
   /*

   SurfelArray surfels = createSurfelArray();
   Surfel s;
   s.pos.x =0;
   s.pos.y =0;
   s.pos.z =1;

   s.normal.x = -1;
   s.normal.y = -1;
   s.normal.z = 1;
   s.normal = unit(s.normal);
   s.color.r = 1;
   s.color.g = 0;
   s.color.b = 0;
   s.radius = 1;
   s.distance = -dot( s.normal, s.pos );
   addToSA( surfels, s );
   */

   number = createDrawingRays( &rays, width_of_image, height_of_image, scene.camera );
   //Scene s2 = createSurfelSpheres( scene, rays, number );
   castRays( surfels, rays, number, buffer );
   //castRaysSphere( s2, rays, number, buffer );
   //castRays( scene, rays, number, buffer );

   outfile.writeTga( "outfile.tga" );
   return EXIT_SUCCESS;
}
char *parseCommandLine(int argc, char *argv[])
{
   if (argc >= 3 )
   {
      if( argv[1][0] == '+' && (argv[1][1] == 'W' || argv[1][1] == 'w') )
      {
         char *temp = &(argv[1][2]);
         std::string tempstring( temp );
         std::stringstream s( tempstring );
         s >> width_of_image;
         if (!s )
         {
            printf("Input Error width unknown\n");
            exit(1);
         }
      }
      if( argv[2][0] == '+' && (argv[2][1] == 'H' || argv[2][1] == 'h') )
      {
         char *temp = &(argv[2][2]);
         std::string tempstring( temp );
         std::stringstream s( tempstring );
         s >> height_of_image;
         if (!s)
         {
            printf("Input Error height unknown\n");
            exit(1);
         }
      }
      if (width_of_image <= 0 || height_of_image <= 0)
      {
         printf("Input Error invalid demenstions, width: %d, height: %d\n", width_of_image, height_of_image);
         exit(1);
      }
      if( argc > 3 )
      {
         return argv[3];
      }
   }
   printf("Error miss use of PBC: PBC +w#### +h#### filename.pov\n");
   exit(EXIT_FAILURE);
}
