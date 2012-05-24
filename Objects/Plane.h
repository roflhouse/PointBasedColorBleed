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
#include "../Util/Ray.h"
#include "../Util/Intersection.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Plane {
   float distance;
   vec3 normal;
   vec3 point;
   ObjectInfo info;
} Plane;

Plane parsePlane( FILE *file );
float planeHitTest( const Plane &sphere, const Ray &ray );
Intersection planeIntersection( const Plane &sphere, const Ray &ray, float t );
#endif
