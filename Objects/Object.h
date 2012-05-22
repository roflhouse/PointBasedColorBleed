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
#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "../Util/Vector.h"
#include <vector>
#include "../Cuda/CudaSwitch.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../Util/BoundingBox.h"

class Object{
   public:
      struct pixel {
         float r;
         float g;
         float b;
      } ;
      struct rayInfo {
         pixel color;
         float camDistance;
         bool hit;
         //point that a ray hit an object
         Vector hitMark;
         //normal to serface of object at hitMark
         Vector normal;
         //View vector
         Vector viewVector;
         //Object that ray hit
         Object *obj;
      };

      Object();
      virtual int parse( FILE *file ) = 0;
      //hitTest test to see if a ray will hit object if it does returns distance in t from cam
      // t > 0 if hit t < 0 if no hit
      virtual float hitTest( Vector direction, Vector position ) = 0;
      //rayIntersect gives point of intersect and data needed after hitTest returns true
      //float t is the float hitTest returns if t > 0
      virtual rayInfo rayIntersect( Vector direction, Vector position, float t ) = 0;
      pixel getColor( rayInfo ray, int monteCarlo );
      float finish_ambient;
      float finish_diffuse;
      float finish_specular;
      float finish_roughness;
      float finish_reflection;
      float finish_refraction;
      float finish_ior;
      float pigment_f;

      BoundingBox *boundingbox;
   protected:
      pixel pigment;
      glm::mat4 transforms;
      glm::mat4 transpose;

      int parsePigment( FILE *file );
      int parseFinish( FILE *file );
      int parseTransforms( FILE *file );
      void transformMinMax( Vector *min, Vector *max );
};
extern Object **spheres;
extern Object **planes;
extern Object **triangles;
extern int numTriangles;
extern int numLights;
extern int numSpheres;
extern int numPlanes;
extern int maxSpheres;
extern int maxPlanes;
extern int maxLights;
#endif
