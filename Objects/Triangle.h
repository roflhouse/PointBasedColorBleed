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
#include <stdlib.h>

typedef struct Triangle {
   float distance;
   vec3 a;
   vec3 b;
   vec3 c;
   vec3 normal;
   ObjectInfo info;
} Triangle;

#include "../Util/Ray.h"
#include "../Util/Intersection.h"
Triangle parseTriangle( FILE *file );
float triangleHitTest( const Triangle &triangle, const Ray &ray ); 
Intersection triangleIntersection( const Triangle &triangle, const Ray &ray, float t );
#endif
