/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef SURFELTYPE_H
#define SURFELTYPE_H
#include "../Util/Color.h"
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
SurfelArray createSurfelArray();
void growSA( SurfelArray &array );
void freeSurfelArray( SurfelArray &in );
void addToSA( SurfelArray &in, const Surfel &surfel );
void shrinkSA( SurfelArray &in );
#endif
