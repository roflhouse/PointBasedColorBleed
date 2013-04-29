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
#include "ColorType.h"
#include "../Objects/SurfelType.h"

typedef struct Ray {
   vec3 pos;
   vec3 dir;
   int i, j;
} Ray;
typedef struct TreeHitMark {
   float t;
   Surfel surfel;
   Color color;
} TreeHitMark;
#endif
