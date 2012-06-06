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
#include "../Util/Color.h"
#include "SurfelType.h"


#include "../Util/Ray.h"
float surfelHitTest( const Surfel &surfel, const struct Ray &ray );
SurfelArray createSurfelArray();
void growSA( SurfelArray &array );
void freeSurfelArray( SurfelArray &in );
void addToSA( SurfelArray &in, const Surfel &surfel );
void shrinkSA( SurfelArray &in );
#endif

