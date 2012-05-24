/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#ifndef OBJECT_H
#define OBJECT_H
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../Util/vec3.h"
#include "../Util/Color.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

typedef struct ColorInfo {
   float finish_ambient;
   float finish_diffuse;
   float finish_specular;
   float finish_roughness;
   float finish_reflection;
   float finish_refraction;
   float finish_ior;
   float pigment_f;
   Color pigment;
} ColorInfo;

typedef struct ObjectInfo {
   ColorInfo colorInfo;
   glm::mat4 transforms;
   glm::mat4 transpose;
} ObjectInfo;

ObjectInfo createObjectInfo();
void parseObjectPigment( FILE *file, ObjectInfo &info );
void parseObjectFinish( FILE *file, ObjectInfo &info );
void parseObjectTransforms( FILE *file, ObjectInfo &info );
#endif
