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
#include "../Util/vec3.h"
#include "../Util/Color.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

typedef struct ObjectInfo {
   float finish_ambient;
   float finish_diffuse;
   float finish_specular;
   float finish_roughness;
   float finish_reflection;
   float finish_refraction;
   float finish_ior;
   float pigment_f;
   Color pigment;

   BoundingBox *boundingbox;
   glm::mat4 transforms;
   glm::mat4 transpose;
} ObjectInfo;
#endif
