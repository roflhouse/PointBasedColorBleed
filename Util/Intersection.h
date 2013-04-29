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
#include "ColorType.h"
#include "../Objects/ObjectInfo.h"
#include "IntersectionType.h"
#include "../Objects/Surfel.h"
#include "Scene.h"

Color directIllumination( const Intersection &inter, const struct Scene &scene );

void growIA( IntersectionArray &array );
void freeIntersectionArray( IntersectionArray &array );
void addToIA( IntersectionArray &in, const Intersection &intersection );
void shrinkIA( IntersectionArray &in );
IntersectionArray createIntersectionArray( int num=1000 );

struct Surfel intersectionToSurfel( const Intersection &inter, const struct Scene &scene );
struct Sphere intersectionToSphere( const Intersection &inter, const struct Scene &scene );
#endif
