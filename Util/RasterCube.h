/**
 *  CPE 2013
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef RASTERCUBE_H
#define RASTERCUBE_H
#include "Util/ColorType.h"

typedef struct RasterCube {
   Color sides[6][8][8];
   float depth[6][8][8];
} RasterCube;
#endif
