/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef SURFEL_H
#define SURFEL_H
#include "../Util/vec3.h"
#include "../Util/ColorType.h"
#include "SurfelType.h"
#include "../Util/RayType.h"

float surfelHitTest( const Surfel &surfel, const struct Ray &ray );
bool equals( Surfel &one, Surfel &two );
#endif

