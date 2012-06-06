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
#include "../Objects/Surfel.h"
#include "vec3.h"

typedef struct TreeNode {
   int leaf;
   struct BoundingBox box;
   struct TreeNode *children[8];
   struct SurfelArray SA;
} TreeNode;

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
TreeNode *createTreeNode( TreeNode root, const BoundingBox &box, int depth );

#endif
