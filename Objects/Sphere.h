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
#include "../Util/Ray.h"
#include "../Util/Intersection.h"
#include "ObjectInfo.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct Sphere {
   float radius;
   vec3 pos;
   ObjectInfo info;
} Sphere;

Sphere parseSphere(FILE *file);
float sphereHitTest( const Sphere &sphere, const Ray &ray ); 
Intersection sphereIntersection( const Sphere &sphere, const Ray &ray, float t );
#endif
