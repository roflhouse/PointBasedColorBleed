/**
 *  CPE 2011
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef CAMERA_H
#define CAMERA_H
#include "../Util/vec3.h"
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


typedef struct Camera {
   vec3 pos;
   vec3 up;
   vec3 right;
   vec3 lookat;
   float l,r,b,t;
} Camera;

//int createInitRays( Rays ***rays, int width, int height, Camera cam );
Camera parseCamera( FILE *file );
#endif
