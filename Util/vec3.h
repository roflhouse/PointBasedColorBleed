/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef VEC3_H 
#define VEC3_H

#include <math.h>

typedef struct vec3{
   float x;
   float y;
   float z;
} vec3;

float mag(const vec3 &in);
float dot(const vec3 &one, const vec3 &two);
vec3 cross(const vec3 &one, const vec3 &two);
float theta(const vec3 &one, const vec3 &two);
float distance(const vec3 &one, const vec3 &two );
float squareDistance(const vec3 &one, const vec3 &two );
vec3 newDirection(const vec3 &to, const vec3 &from );
vec3 unit(const vec3 &in);
void keepMin( vec3 &min, vec3 pos );
void keepMax( vec3 &max, vec3 pos );

#endif
