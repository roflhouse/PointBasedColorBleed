/**
 *  CPE 2012
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#ifndef OCTREE_H
#define OCTREE_H
#include "BoundingBox.h"
#include "../Objects/SurfelType.h"
#include "vec3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Hermonics {
   float red[9];
   float green[9];
   float blue[9];
   float area[9];
} Hermonics;
typedef struct TreeNode {
   int leaf;
   struct BoundingBox box;
   struct TreeNode *children[8];
   struct SurfelArray SA;
   struct Hermonics hermonics;
} TreeNode;
typedef struct ArrayNode {
   int leaf;
   struct BoundingBox box;
   int children[8];
} ArrayNode;

typedef struct Node {
   int leaf;
   int children[8];
   struct BoundingBox box;
} Node;

typedef struct Octree {
   struct BoundingBox box;
   int nodes[8];
   int numNodes;
} Octree;

TreeNode createOctree( struct SurfelArray &SA, vec3 min, vec3 max );
TreeNode createOctreeMark2( struct SurfelArray &SA, vec3 min, vec3 max );
TreeNode *createTreeNode( TreeNode *root, const BoundingBox &box, int depth );
ArrayNode *createOctreeForCuda( struct SurfelArray &SA, vec3 min, vec3 max, int &size );
Hermonics calculateSphericalHermonics( struct Surfel &surfel );
double *getYLM(double x, double y, double z);
void filloutHermonics( TreeNode *root, int total );
Hermonics createHermonics();
void clearHermonics( Hermonics &hermonics );
void addHermonics( Hermonics &save, Hermonics &gone );
void averageHermonics( Hermonics &save, float factor );
double SH(int l, int m, double theta, double phi);

#endif
