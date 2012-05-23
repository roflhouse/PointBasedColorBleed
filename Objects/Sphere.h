/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef SPHERE_H
#define SPHERE_H
#include "../Util/vec3.h"
#include "ObjectInfo.h"

typedef struct Sphere {
   float radius;
   vec3 pos;
   ObjectInfo info;
} Sphere;
Sphere parseSphere(FILE *file);
#endif
