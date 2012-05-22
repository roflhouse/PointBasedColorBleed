/**
 *  CPE 2011
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "BVH.h"

BVH::BVH()
{
    if( numSpheres + numTriangles == 0 )
    {
        root = NULL;
        return;
    }
    int i = 0;
    int j = 0;

    Object **objects = (Object **) malloc( (numSpheres + numTriangles + numPlanes)*sizeof(Object *) );

    for( i = 0; i < numSpheres; i++)
    {
        objects[i] = spheres[i];
    }
    for( j = 0; j < numTriangles; j++ )
    {
        objects[i+j] = triangles[j];
    }
    /*for( k = 0; k < numPlanes; k++ )
      {
      objects[i+j+k] = planes[k];
      }*/

    root = new BVHNode( objects, numSpheres + numTriangles ); // + numPlanes );
    free( objects );
}
BVH::BVHNode::BVHNode( Object **objects, int numObj )
{
    Vector min;
    Vector max;
    if( numObj == 1 )
    {
        left = NULL;
        right = NULL;
        obj = *objects;
        box = new BoundingBox(*(obj->boundingbox));
    }
    else if( numObj == 2 )
    {
        left = new BVHNode( objects, 1 );
        right = new BVHNode( objects+1, 1 );
        box = new BoundingBox( *(objects[0]->boundingbox), *(objects[1]->boundingbox) );
    }
    else if( numObj > 2 )
    {
        min = objects[0]->boundingbox->min;
        max = objects[0]->boundingbox->max;
        //sort by greatest length
        for ( int i = 1; i < numObj; i++ )
        {
            if( min.x > objects[i]->boundingbox->min.x )
                min.x = objects[i]->boundingbox->min.x;
            if( min.y > objects[i]->boundingbox->min.y )
                min.y = objects[i]->boundingbox->min.y;
            if( min.z > objects[i]->boundingbox->min.z )
                min.z = objects[i]->boundingbox->min.z;

            if( max.x < objects[i]->boundingbox->max.x )
                max.x = objects[i]->boundingbox->max.x;
            if( max.y < objects[i]->boundingbox->max.y )
                max.y = objects[i]->boundingbox->max.y;
            if( max.z < objects[i]->boundingbox->max.z )
                max.z = objects[i]->boundingbox->max.z;
        }
        float dx = max.x - min.x;
        float dy = max.y - min.y;
        float dz = max.z - min.z;
        if( dx > dy && dx > dz )
        {
            //sort by x
            sort( objects, numObj, 1 );
        }
        else if( dy > dz )
        {
            //sort by y
            sort( objects, numObj, 2 );
        }
        else
        {
            //sort by z
            sort( objects, numObj, 3 );
        }
        left = new BVHNode( objects, numObj/2 );
        right = new BVHNode( objects + numObj/2, numObj - numObj/2 );
        box = new BoundingBox( min, max );
    }
}
void BVH::BVHNode::sort( Object **objects, int numObj, int axis )
{
    float centers[numObj];
    for( int i = 0; i < numObj; i++ )
    {
        if( axis == 1 )
            centers[i] = objects[i]->boundingbox->min.x + (objects[i]->boundingbox->max.x
                    - objects[i]->boundingbox->min.x )/2;
        else if( axis == 2 )
            centers[i] = objects[i]->boundingbox->min.y + (objects[i]->boundingbox->max.y
                    - objects[i]->boundingbox->min.y )/2;
        else if( axis == 3 )
            centers[i] = objects[i]->boundingbox->min.z + (objects[i]->boundingbox->max.z
                    - objects[i]->boundingbox->min.z )/2;
    }
    for( int i = 0; i < numObj; i++ )
    {
        for( int j = i; j < numObj; j++ )
        {
            if( centers[i] > centers[j] )
            {
                int temp = centers[i];
                Object *tempO = objects[i];
                objects[i] = objects[j];
                objects[j] = tempO;
                centers[i] = centers[j];
                centers[j] = temp;
            }
        }
    }
}
void BVH::getIntersections( Vector dir, Vector pos, Object **retObjs[], int *retNumObjs )
{
    if( root == NULL )
    {
        *retNumObjs = 0;
        return;
    }
    getIntersections( dir, pos, retObjs, retNumObjs, root );
}
void BVH::getIntersections( Vector dir, Vector pos, Object **retObjs[], int *retNumObjs, BVHNode *n )
{
    bool testHit = n->box->testForHit( dir, pos );
    //Is this a leaf Node?
    if( n->obj != NULL )
    {
        if( testHit )
        {
            *retObjs = (Object **) malloc( sizeof(Object *) );
            **retObjs = n->obj;
            *retNumObjs = 1;
            return;
        }
    }
    //Not a Leaf continue
    else if( testHit )
    {
        //get lists right and left
        int leftNum = 0;
        int rightNum = 0;
        Object **leftList = NULL;
        Object **rightList = NULL;
        if( n->left != NULL )
        {
            getIntersections( dir, pos, &leftList, &leftNum, n->left );
        }
        if( n->right != NULL )
        {
            getIntersections( dir, pos, &rightList, &rightNum, n->right );
        }

        *retNumObjs = leftNum + rightNum;
        if( *retNumObjs > 0 )
            *retObjs = (Object **) malloc( *retNumObjs * sizeof( Object*));

        int i;
        for( i = 0; i < leftNum ; i++ )
        {
            (*retObjs)[i] = leftList[i];
        }
        for( int j = 0; j < rightNum; j++)
        {
            (*retObjs)[i+j] = rightList[j];
        }
        if( leftNum > 0 )
            free( leftList );
        if( rightNum > 0 )
            free( rightList );
    }
    else
    {
        //No hits
        *retNumObjs = 0;
    }
}
