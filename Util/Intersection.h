/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef INTERSECTION_H
#define INTERSECTION_H
#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "vec3.h"
#include "Color.h"
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
#include "../Objects/Surfel.h"
#include "Scene.h"

Color directIllumination( const Intersection &inter, const Scene &scene );

void growIA( IntersectionArray &array );
void freeIntersectionArray( IntersectionArray &array );
void addToIA( IntersectionArray &in, const Intersection &intersection );
void shrinkIA( IntersectionArray &in );
IntersectionArray createIntersectionArray();

struct Surfel intersectionToSurfel( const Intersection &inter, const Scene &scene );
#endif
