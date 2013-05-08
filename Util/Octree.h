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
//#include "CudaOctree.h"
#include "vec3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "OctreeType.h"

TreeNode createOctree( struct SurfelArray &SA, vec3 min, vec3 max );
TreeNode createOctreeMark2( struct SurfelArray &SA, vec3 min, vec3 max );
TreeNode *createTreeNode( TreeNode *root, const BoundingBox &box, int depth );
Hermonics calculateSphericalHermonics( struct Surfel &surfel );
double *getYLM(double x, double y, double z);
void filloutHermonics( TreeNode *root, int total );
Hermonics createHermonics();
void clearHermonics( Hermonics &hermonics );
void addHermonics( Hermonics &save, Hermonics &gone );
void averageHermonics( Hermonics &save, float factor );
double SH(int l, int m, double theta, double phi);
int octreeToCudaTree( TreeNode *cpu_root, CudaNode* gpu_root, int current_node, 
      SurfelArray &gpu_array );
void createCudaTree( SurfelArray cpu_array, vec3 min, vec3 max, CudaNode* &gpu_root, int &nodes,
            SurfelArray &gpu_array );
int countNodes( TreeNode *root );
int countLeafNodes( TreeNode *root );
int getLeafAddrs( CudaNode *gpu_root, int node, int *leaf_addrs, int current );

#endif
