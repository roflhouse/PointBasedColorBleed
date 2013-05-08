/**
 *  CPE 2013
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef OCTREETYPE_H
#define OCTREETYPE_H
#include "vec3.h"
#include "BoundingBox.h"
#include "../Objects/SurfelType.h"
typedef struct Hermonics {
   float red[9];
   float green[9];
   float blue[9];
   float area[9];
} Hermonics;

typedef struct TreeNode {
   int leaf;
   long int numInNode;
   struct BoundingBox box;
   struct TreeNode *children[8];
   struct SurfelArray SA;
   struct Hermonics hermonics;
} TreeNode;

typedef struct CudaNode {
   int leaf;
   struct BoundingBox box;
   int children[8];
   struct Hermonics hermonics;
} CudaNode;

#endif
