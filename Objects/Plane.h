/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef PLANE_H 
#define PLANE_H 
#include "ObjectInfo.h"
#include "../Util/vec3.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Plane {
   float distance;
   vec3 normal;
   vec3 point;
   ObjectInfo info;
} Plane;

Plane parsePlane( FILE *file );
#endif
