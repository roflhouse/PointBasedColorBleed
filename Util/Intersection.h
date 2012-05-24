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
#include "Scene.h"
Color directIllumination( Intersection inter, Scene scene );
#endif
