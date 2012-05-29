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
#include "../Util/Ray.h"
#include "../Util/vec3.h"

typedef struct Surfel {
   vec3 pos;
   vec3 normal;
   float distance;
   Color color;
   float radius;
} Surfel;

typedef struct SurfelArray {
   Surfel *array;
   int num;
   int max;
} SurfelArray;

float surfelHitTest( const Surfel &surfel, const struct Ray &ray );
SurfelArray createSurfelArray();
void growSA( SurfelArray &array );
void freeSurfelArray( SurfelArray &in );
void addToSA( SurfelArray &in, const Surfel &surfel );
void shrinkSA( SurfelArray &in );
#endif

