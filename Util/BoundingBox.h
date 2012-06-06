/**
 *  CPE 2011
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H
#include "vec3.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct BoundingBox {
   vec3 min;
   vec3 max;
} BoundingBox;

#include "RayType.h"
bool testForHit( const BoundingBox &box, const Ray &ray );
bool isIn( const BoundingBox &box, const vec3 &post );
BoundingBox *getSubBoxes( const BoundingBox &box );
BoundingBox createBoundingBox( const vec3 &min, const vec3 &max );

#endif
