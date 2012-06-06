/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Octree.h"

int glob;
int count( TreeNode *root );
int buildOctreeArray( TreeNode *tree, ArrayNode *octree, int &cur, SurfelArray &SA );
TreeNode createOctree( SurfelArray &SA, vec3 min, vec3 max )
{
   TreeNode root;

   root.box = createBoundingBox( min, max );
   root.SA = SA;
   printf("first %d\n", SA.num );
   if( root.SA.num > 32 )
   {
      root.leaf = false;
      BoundingBox *boxes = getSubBoxes( root.box );
      for( int i = 0; i < 8; i++ )
         root.children[i] = createTreeNode( root, boxes[i], 1 );
      freeSurfelArray( SA );
   }
   else
      root.leaf = true;
   //printf("Octree finished %d\n", sizeof(Surfel) * SA.num);
   int numberNodes = count( &root );

   printf("Octree finished %d\n", numberNodes);
   return root;
}
int count( TreeNode *root )
{
   if( root == NULL )
   {
      printf("NULL\n");
      return 0;
   }
   if( root->leaf )
   {
      glob += root->SA.num;
      if( root->SA.num )
         return 1;
      else
         return 0;
   }
   else
   {
      int ret = 0;
      for( int i = 0; i < 8; i++ )
      {
         ret += count( root->children[i] );
      }
      return ret + 1;
   }
}
ArrayNode *createOctreeForCuda( SurfelArray &SA, vec3 min, vec3 max, int &size )
{
   glob = 0;
   TreeNode root;

   root.box = createBoundingBox( min, max );
   root.SA = SA;
   printf("first %d\n", SA.num );
   if( root.SA.num > 32 )
   {
      root.leaf = false;
      BoundingBox *boxes = getSubBoxes( root.box );
      for( int i = 0; i < 8; i++ )
         root.children[i] = createTreeNode( root, boxes[i], 1 );
   }
   else
      root.leaf = true;

   int numberNodes = count( &root );
   shrinkSA(SA);

   printf("Octree finished %d, %d, %d\n", glob, SA.num, numberNodes);

   ArrayNode *octree = (ArrayNode *) malloc ( sizeof(ArrayNode) * numberNodes );
   int cur = 0;
   freeSurfelArray( SA );
   SA = createSurfelArray();
   buildOctreeArray( &root, octree, cur, SA );
   shrinkSA( SA );
   size = numberNodes;
   /*for( int i = 0; i < size; i++ )
   {
      //printf("Octree %d\n", i);
      //printf("Leaf %d\n", octree[i].leaf );
      for( int j = 0; j< 8; j++ )
         //printf("\tChild: %d\n", octree[i].children[j] );
   }
   */
   return octree;
}
int buildOctreeArray( TreeNode *tree, ArrayNode *octree, int &cur, SurfelArray &SA )
{
   int mySpot = cur;
   octree[mySpot].leaf = tree->leaf;
   octree[mySpot].box = tree->box;
   if( octree[mySpot].leaf )
   {
      if( tree->SA.num )
      {
         int adding = tree->SA.num;
         octree[mySpot].children[0] = SA.num;
         octree[mySpot].children[1] = SA.num+adding;
         if( SA.num +adding > SA.max )
         {
            growSA( SA );
         }
         memcpy( &(SA.array[SA.num]), tree->SA.array, adding * sizeof(Surfel) );
         SA.num += adding;
         //freeSurfelArray( tree->SA );
         cur++;
         return mySpot;
      }
      else
         return -1;
   }
   else
   {
      cur++;
      for( int i = 0; i < 8; i++ )
      {
         octree[mySpot].children[i] = buildOctreeArray( tree->children[i], octree, cur, SA );
      }
      return mySpot;
   }
}
TreeNode *createTreeNode( TreeNode root, const BoundingBox &box, int depth )
{
   TreeNode *ret = (TreeNode *) malloc ( sizeof( TreeNode ) );
   ret->box = box;
   ret->SA = createSurfelArray();
   for( int i = 0; i < root.SA.num; i++ )
   {
      if( isIn( ret->box, root.SA.array[i].pos ) )
         addToSA( ret->SA, root.SA.array[i] );
   }
   shrinkSA( ret->SA );
   if( ret->SA.num > 32 && depth < 15 )
   {
      ret->leaf = false;
      BoundingBox *boxes = getSubBoxes( ret->box );
      for( int i = 0; i < 8; i++ )
         ret->children[i] = createTreeNode( *ret, boxes[i], depth+1 );
      freeSurfelArray( ret->SA );
   }
   else
      ret->leaf = true;

   return ret;
}
