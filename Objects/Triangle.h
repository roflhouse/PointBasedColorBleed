/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef TRIANGLE_H 
#define TRIANGLE_H 
#include "ObjectInfo.h"
#include "../Util/vec3.h"
#include <stdio.h>

typedef struct Triangle {
   float distance;
   vec3 a;
   vec3 b;
   vec3 c;
   vec3 normal;
} Triangle;

Triangle parseTriangle( FILE *file );
#endif
