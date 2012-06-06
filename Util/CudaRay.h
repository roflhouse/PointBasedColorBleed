/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef CUDARAY_H
#define CUDARAY_H
#include "vec3.h"
#include "Color.h"
#include "../Objects/SurfelType.h"
#include "RayType.h"

void castRaysCuda( const SurfelArray &s, Ray *rays, int numRays, Color *buffer, int width, int height );
#endif
