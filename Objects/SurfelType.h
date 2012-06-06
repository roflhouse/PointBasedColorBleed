/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef SURFELTYPE_H
#define SURFELTYPE_H
typedef struct Surfel {
   vec3 pos;
   vec3 normal;
   float distance;
   Color color;
   float radius;
} Surfel;

typedef struct SurfelArray {
   Surfel *array;
   int num;
   int max;
} SurfelArray;
#endif
