/**
 *  CPE 2010
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */
#ifndef LIGHTSOURCE_H
#define LIGHTSOURCE_H
#include <stdio.h>
#include <stdlib.h>
#include "../Util/ColorType.h"
#include "../Util/vec3.h"

typedef struct PointLight {
   Color color;
   vec3 pos;
   vec3 points[100];
} PointLight;

PointLight parsePointLight( FILE *file );
#endif
