/**
 *  CPE 2011
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Camera.h"

Camera parseCamera( FILE *file )
{
   Camera cam;
   char cur = '\0';
   //location
   while (cur != '<')
   {
      //eat white space and the location tag
      if(fscanf(file, "%c", &cur) == EOF )
      {
         printf("Error Parseing camera\n");
         exit(1);
      }
   }
   //Read in location data
   if(fscanf(file, " %f , %f , %f ", &(cam.pos.x), &(cam.pos.y), &(cam.pos.z)) == EOF)
   {
      printf("Error Parseing camera\n");
      exit(1);
   }

   printf( "location: %f, %f, %f\n", cam.pos.x, cam.pos.y, cam.pos.z );
   cur = '\0';

   //up
   while (cur != '<')
   {
      //eat white space and the up tag
      if(fscanf(file, "%c", &cur) == EOF )
      {
         printf("Error Parseing camera\n");
         exit(1);
      }
   }
   //Read in up data
   if(fscanf(file, " %f , %f , %f ", &(cam.up.x), &(cam.up.y), &(cam.up.z)) == EOF)
   {
      printf("Error Parseing camera\n");
      exit(1);
   }

   printf( "up: %f, %f, %f\n", cam.up.x, cam.up.y, cam.up.z );
   cur = '\0';

   //right
   while (cur != '<')
   {
      //eat white space and the right tag
      if(fscanf(file, "%c", &cur) == EOF )
      {
         printf("Error Parseing camera\n");
         exit(1);
      }
   }
   //Read in right data
   if(fscanf(file, " %f , %f , %f ", &(cam.right.x), &(cam.right.y), &(cam.right.z)) == EOF)
   {
      printf("Error Parseing camera\n");
      exit(1);
   }

   printf( "right: %f, %f, %f\n", cam.right.x, cam.right.y, cam.right.z );
   cur = '\0';

   //look_at
   while (cur != '<')
   {
      //eat white space and the look_at tag
      if(fscanf(file, "%c", &cur) == EOF )
      {
         printf("Error Parseing camera\n");
         exit(1);
      }
   }
   //Read in look_at data
   if(fscanf(file, " %f , %f , %f ", &(cam.lookat.x), &(cam.lookat.y), &(cam.lookat.z)) == EOF)
   {
      printf("Error Parseing camera\n");
      exit(1);
   }

   printf( "look_at: %f, %f, %f\n", cam.lookat.x, cam.lookat.y, cam.lookat.z );
   cur = '\0';

   //Read in rest of the trash
   while( cur != '}' )
   {
      if(fscanf(file, "%c", &cur) == EOF )
      {
         printf("Error Parseing camera\n");
         exit(1);
      }
   }
   cam.l = -mag(cam.right)/2;
   cam.r = mag(cam.right)/2;
   cam.t = mag(cam.up)/2;
   cam.b = -mag(cam.up)/2;
   return cam;
}
