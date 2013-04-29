/**
 *  CPE 2013
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef BOUNDINGBOXTYPE_H
#define BOUNDINGBOXTYPE_H
#include "vec3.h"
typedef struct BoundingBox {
      vec3 min;
         vec3 max;
} BoundingBox;
#endif

