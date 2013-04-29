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
#include "BoundingBoxType.h"
#include "RayType.h"

bool testForHit( const BoundingBox &box, const Ray &ray );
bool isIn( const BoundingBox &box, const vec3 &post );
int belowHorizon(const BoundingBox &box, vec3 &position, vec3 &normal );
BoundingBox *getSubBoxes( const BoundingBox &box );
BoundingBox createBoundingBox( const vec3 &min, const vec3 &max );
vec3 getCenter(const BoundingBox &box );
float distanceToBox( const BoundingBox &box, vec3 &pos );
bool equals( const BoundingBox &one, const BoundingBox &two );

#endif
