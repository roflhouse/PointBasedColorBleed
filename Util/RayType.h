/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef RAYTYPE_H
#define RAYTYPE_H 

typedef struct Ray {
   vec3 pos;
   vec3 dir;
   int i, j;
} Ray;
int createInitRays( struct Ray **rays, int width, int height, float growth, struct Camera cam );
int createDrawingRays( struct Ray **rays, int width, int height, struct Camera cam );
#endif
