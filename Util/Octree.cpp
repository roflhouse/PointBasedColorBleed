/**
 *  CPE 2012
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Octree.h"

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
   printf("Octree finished\n");
   return root;
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
   if( ret->SA.num > 32 && depth < 20 )
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
