/**
 *  CPE 2011
 *  -------------------
 *  Program 
 *
 *  Last Modified: 
 *  @author Nick Feeney
 */

#include "BoundingBox.h"

BoundingBox::BoundingBox( Vector minIn, Vector maxIn )
{
    min = minIn;
    max = maxIn;
    if( minIn.x > maxIn.x )
    {
        printf("Warning BoundingBox min.x %f > max.xi %f\n", min.x, max.x); 
    }
    if( minIn.y > maxIn.y )
    {
        printf("Warning BoundingBox min.y %f > max.y %f\n", min.y, max.y); 
    }
    if( minIn.z > maxIn.z )
    {
        printf("Warning BoundingBox min.z %f > max.z\n %f", min.z, max.z); 
    }
}
BoundingBox::BoundingBox( BoundingBox one, BoundingBox two )
{
    min.x = one.min.x;
    min.y = one.min.y;
    min.z = one.min.z;
    max.x = one.max.x;
    max.y = one.max.y;
    max.z = one.max.z;
    if( min.x > two.min.x )
        min.x = two.min.x;
    if( min.y > two.min.y )
        min.y = two.min.y;
    if( min.z > two.min.z )
        min.z = two.min.z;
    if( max.x < two.max.x )
        max.x = two.max.x;
    if( max.y < two.max.y )
        max.y = two.max.y;
    if( max.z < two.max.z )
        max.z = two.max.z;
}
bool BoundingBox::testForHit( Vector dir, Vector pos )
{
    if( dir.x > -0.0001 && dir.x < 0.0001 )
    {
        if( pos.x < min.x || pos.x > max.x )
            return false;
    }
    if( dir.y > -0.0001 && dir.y < 0.0001 )
    {
        if( pos.y < min.y || pos.y > max.y )
            return false;
    }
    if( dir.z > -0.0001 && dir.z < 0.0001 )
    {
        if( pos.z < min.z || pos.z > max.z )
            return false;
    }
    float txmin = (min.x - pos.x) / dir.x;
    float tymin = (min.y - pos.y) / dir.y;
    float tzmin = (min.z - pos.z) / dir.z;
    float txmax = (max.x - pos.x) / dir.x;
    float tymax = (max.y - pos.y) / dir.y;
    float tzmax = (max.z - pos.z) / dir.z;

    if( txmin > txmax )
    {
        float temp = txmax;
        txmax = txmin;
        txmin = temp;
    }
    if( tymin > tymax )
    {
        float temp = tymax;
        tymax = tymin;
        tymin = temp;
    }
    if( tzmin > tzmax )
    {
        float temp = tzmax;
        tzmax = tzmin;
        tzmin = temp;
    }

    float tgmin = txmin;
    float tgmax = txmax;
    //find largest min
    if( tgmin < tymin )
        tgmin = tymin;
    if( tgmin < tzmin )
        tgmin = tzmin;

    //find smallest max
    if( tgmax > tymax )
        tgmax = tymax;
    if( tgmax > tzmax )
        tgmax = tzmax;

    if( tgmin > tgmax )
        return false;
    return true;
}
