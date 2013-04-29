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
#include "../Util/ColorType.h"
#include "ObjectInfo.h"
typedef struct Surfel {
   vec3 pos;
   vec3 normal;
   float distance;
   Color color;
   float radius;
   //ColorInfo info;
} Surfel;

typedef struct SurfelArray {
   Surfel *array;
   int num;
   int max;
} SurfelArray;
SurfelArray createSurfelArray( int num=1000 );
void growSA( SurfelArray &array );
void freeSurfelArray( SurfelArray &in );
void addToSA( SurfelArray &in, const Surfel &surfel );
void shrinkSA( SurfelArray &in );
#endif
