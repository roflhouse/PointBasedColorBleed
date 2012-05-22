/**
 *  CPE 2011
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#ifndef BVH_H
#define BVH_H

#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "BoundingBox.h"
#include "Vector.h"
#include "../Objects/Object.h"

extern Object **spheres;
extern Object **triangles;
extern Object **planes;
extern int numSpheres;
extern int numTriangles;
extern int numPlanes;

class BVH
{
   public:
      class BVHNode
      {
         public:
            BVHNode(Object **obj, int numObj);
            void sort( Object **objects, int numObj, int axis );

            BoundingBox *box;
            BVHNode *left;
            BVHNode *right;
            Object *obj;
      };
      BVH( );
      void getIntersections( Vector dir, Vector pos, Object **retObjs[], int *retNumObjs );

   private:
      BVHNode *root;
      void getIntersections( Vector dir, Vector pos, Object **retObjs[], int *retNumObjs, BVHNode *n );
};
#endif
