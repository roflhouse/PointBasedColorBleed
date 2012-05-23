/**
 *  CPE 2011
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Camera.h"

int createInitRays( Ray ***rays, int width, int height, Camera cam )
{
   float rightUnitX = right.unit().x;
   float rightUnitY = right.unit().y;
   float rightUnitZ = right.unit().z;
   float upUnitX = up.unit().x;
   float upUnitY = up.unit().y;
   float upUnitZ = up.unit().z;
   vec3 uv = newDirection(cam.lookat, cam.pos).unit();

   *rays = (Ray **) malloc( sizeof(Ray *) *width );
   for( int j = 0; j < height; j++)
   {
      (*rays)[j] = (Ray **) malloc( sizeof(Ray) * height );
      for( int i = 0; i < width ; i ++ )
      {
         float rando = rand()/RAND_MAX;
         //First Ray
         float u = cam.l + (cam.r-cam.l)*( (float)i+(.5  * rando) )/(float)width;
         float v = cam.b + (cam.t-cam.b)*( (float)j+(.5  * rando) )/(float)height;
         float w = -1;

         (*ray)[j][i].x = u * rightUnitX + v * upUnitX + -w * uv.x;
         (*ray)[j][i].y = u * rightUnitY + v * upUnitY + -w * uv.y;
         (*ray)[j][i].z = u * rightUnitZ + v * upUnitZ + -w * uv.z;

         /*
         //Second Ray
         rando = rand()/RAND_MAX;
         u = l + (r-l)*((float)i+0.5*rando)/(float)width_of_image;
         v = b + (t-b)*((float)j+(rando*.5 +.5))/(float)height_of_image;

         rayDirection.x = u * rightUnitX + v * upUnitX + -w * uv.x;
         rayDirection.y = u * rightUnitY + v * upUnitY + -w * uv.y;
         rayDirection.z = u * rightUnitZ + v * upUnitZ + -w * uv.z;
         rays.push_back( new Ray( rayDirection, location, i, j, 1 ) );

         //third ray
         rando = rand()/RAND_MAX;
         u = l + (r-l)*((float)i+(rando*.5 +.5))/(float)width_of_image;
         v = b + (t-b)*((float)j+0.5*rando)/(float)height_of_image;

         rayDirection.x = u * rightUnitX + v * upUnitX + -w * uv.x;
         rayDirection.y = u * rightUnitY + v * upUnitY + -w * uv.y;
         rayDirection.z = u * rightUnitZ + v * upUnitZ + -w * uv.z;
         rays.push_back( new Ray( rayDirection, location, i, j, 1 ) );

         //Fourth Ray
         rando = rand()/RAND_MAX;
         u = l + (r-l)*((float)i+(rando*.5 +.5))/(float)width_of_image;
         v = b + (t-b)*((float)j+(rando*.5 +.5))/(float)height_of_image;

         rayDirection.x = u * rightUnitX + v * upUnitX + -w * uv.x;
         rayDirection.y = u * rightUnitY + v * upUnitY + -w * uv.y;
         rayDirection.z = u * rightUnitZ + v * upUnitZ + -w * uv.z;
         rays.push_back( new Ray( rayDirection, location, i, j, 1 ) );
          */
      }
   }
   return width * height;
}
Camera parseCamera( FILE *file )
{
   Camera cam;
   char cur = '\0';
   //location
   while (cur != '<')
   {
      //eat white space and the location tag
      if(fscanf(file, "%c", &cur) == EOF )
         return NULL;
   }
   //Read in location data
   if(fscanf(file, " %f , %f , %f ", &(cam.pos.x), &(cam.pos.y), &(cam.pos.z)) == EOF)
      return NULL;

   printf( "location: %f, %f, %f\n", cam.pos.x, cam.pos.y, cam.pos.z );
   cur = '\0';

   //up
   while (cur != '<')
   {
      //eat white space and the up tag
      if(fscanf(file, "%c", &cur) == EOF )
         return NULL;
   }
   //Read in up data
   if(fscanf(file, " %f , %f , %f ", &(cam.up.x), &(cam.up.y), &(cam.up.z)) == EOF)
      return NULL;

   printf( "up: %f, %f, %f\n", cam.up.x, cam.up.y, cam.up.z );
   cur = '\0';

   //right
   while (cur != '<')
   {
      //eat white space and the right tag
      if(fscanf(file, "%c", &cur) == EOF )
         return NULL;
   }
   //Read in right data
   if(fscanf(file, " %f , %f , %f ", &(cam.right.x), &(cam.right.y), &(cam.right.z)) == EOF)
      return NULL;

   printf( "right: %f, %f, %f\n", cam.right.x, cam.right.y, cam.right.z );
   cur = '\0';

   //look_at
   while (cur != '<')
   {
      //eat white space and the look_at tag
      if(fscanf(file, "%c", &cur) == EOF )
         return NULL;
   }
   //Read in look_at data
   if(fscanf(file, " %f , %f , %f ", &(cam.lookat.x), &(cam.lookat.y), &(cam.lookat.z)) == EOF)
      return NULL;

   printf( "look_at: %f, %f, %f\n", cam.lookat.x, cam.lookat.y, cam.lookat.z );
   cur = '\0';

   //Read in rest of the trash
   while( cur != '}' )
   {
      if(fscanf(file, "%c", &cur) == EOF )
         return NULL;
   }
   cam.l = -mag(cam.right)/2;
   cam.r = mag(cam.right)/2;
   cam.t = mag(cam.up)/2;
   cam.b = -mag(cam.up)/2;
   return cam;
}
