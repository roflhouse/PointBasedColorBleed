/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "../Objects/Objects.h"
#include "vec3.h"

typedef struct Intersection {
   vec3 hitMark;
   vec3 normal;
   vec3 viewVector;
   float hit;
   
} Intersection;
