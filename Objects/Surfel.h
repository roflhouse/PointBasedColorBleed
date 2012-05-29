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

typedef struct Surfel {
   vec3 pos;
   vec3 normal;
   Color color;
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

