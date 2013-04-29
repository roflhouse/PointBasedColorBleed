/**
 *  CPE 2013
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef INTERSECTIONTYPE_H
#define INTERSECTIONTYPE_H
#include "vec3.h"
#include "../Objects/ObjectInfo.h"
typedef struct Intersection {
   vec3 hitMark;
   vec3 normal;
   vec3 viewVector;
   float hit;
   ColorInfo colorInfo;
} Intersection;

typedef struct IntersectionArray {
   Intersection *array;
   int num;
   int max;
} IntersectionArray;

#endif
