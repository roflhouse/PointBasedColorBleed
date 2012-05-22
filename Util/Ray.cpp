/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Ray.h"

#define PI 3.141592

Ray::Ray( Vector startDir, Vector eyePos, int pixelW, int pixelH )
{
    w= pixelW;
    h= pixelH;
    position = eyePos;
    direction = startDir.unit();
    hit = false;
    curDistance = -1;
    mod = 1;
    depth = 1;
    refractionCur = 1;
}
//Construction for reflection rays
Ray::Ray( Vector startDir, Vector eyePos, int pixelW, int pixelH, float modifier, int d )
{
    w= pixelW;
    h= pixelH;
    depth = d;
    mod = modifier;
    position = eyePos;
    direction = startDir.unit();
    hit = false;
    curDistance = -1;
    refractionCur = 1;
}
//Constructor for refraction rays
Ray::Ray( Vector startDir, Vector eyePos, int pixelW, int pixelH, float modifier, int d,
        float refraction )
{
    w= pixelW;
    h= pixelH;
    depth = d;
    mod = modifier;
    position = eyePos;
    direction = startDir.unit();
    hit = false;
    curDistance = -1;
    refractionCur = refraction;
}
Object::pixel Ray::castRay()
{
    Object::pixel color;
    color.r = 0;
    color.b = 0;
    color.g = 0;
    Object::rayInfo ret;
    ret.obj = NULL;
    ret.color.r = 0;
    ret.color.b = 0;
    ret.color.g = 0;
    ret.camDistance = -1;
    ret.hit = false;
    Object::rayInfo intersect;
    intersect.obj = NULL;
    intersect.color.r = 0;
    intersect.color.g = 0;
    intersect.color.b = 0;
    intersect.camDistance = -1;
    intersect.hit = false;
    float t;

    //BVH tests
    Object **possible;
    int numPos = 0;

    numTriangles = 0;
    bvh->getIntersections( direction, position, &possible, &numPos );

    //Test all possible objects according to bvh
    for( int i = 0; i < numPos; i++ )
    {
        t = possible[i]->hitTest( direction, position );
        if( t > 0 )
        {
            ret = possible[i]->rayIntersect( direction, position, t );
            if(ret.hit)
            {
                if(ret.camDistance < curDistance || hit == false)
                {
                    intersect = ret;
                    curDistance = ret.camDistance;
                    hit = true;
                }
            }
        }
    }

    if( numPos > 0 )
        free( possible );
    /*
       for( int i = 0; i < numSpheres; i++ )
       {
       t = spheres[i]->hitTest( direction, position );
       if( t > 0 )
       {
       ret = spheres[i]->rayIntersect( direction, position, t );
       if(ret.hit)
       {
       if(ret.camDistance < curDistance || hit == false)
       {
       intersect = ret;
       curDistance = ret.camDistance;
       hit = true;
       }
       }
       }
       }*/
    //boundingbox for plane is infinite so we dont put them in bvh
    for ( int i = 0; i < numPlanes; i++ )
    {
        t = planes[i]->hitTest( direction, position );
        if( t > 0 )
        {
            ret = planes[i]->rayIntersect( direction, position, t );
            if(ret.hit)
            {
                if(ret.camDistance < curDistance || hit == false)
                {
                    intersect = ret;
                    curDistance = ret.camDistance;
                    hit = true;
                }
            }
        }
    }
    /*for ( int i = 0; i < numTriangles; i++ )
      {
      t = triangles[i]->hitTest( direction, position );
      if( t > 0 )
      {
      ret = triangles[i]->rayIntersect( direction, position, t );
      if(ret.hit)
      {
      if(ret.camDistance < curDistance || hit == false)
      {
      intersect = ret;
      curDistance = ret.camDistance;
      hit = true;
      }
      }
      }
      }*/
    if (hit && intersect.obj != NULL)
        color = (intersect.obj)->getColor( intersect );

    color.r = color.r * mod;
    color.g = color.g * mod;
    color.b = color.b * mod;
    return color;
}

Object::pixel Ray::castRay(float it, int index, float tri_it, int triIndex)
{
    Object::pixel color;
    color.r = 0;
    color.b = 0;
    color.g = 0;
    Object::rayInfo ret;
    Object::rayInfo intersect;
    intersect.obj = NULL;
    intersect.color.r = 0;
    intersect.color.g = 0;
    intersect.color.b = 0;
    intersect.camDistance = -1;
    intersect.hit = false;
    float t;
    //Cuda version
    t = it;

    //Hit Sphere
    if( t > 0 )
    {
        intersect = spheres[index]->rayIntersect( direction, position, t );
        curDistance = intersect.camDistance;
        hit = true;
    }
    //Hit Triangle
    else if (tri_it > 0 ) {
        ret = triangles[triIndex]->rayIntersect(direction, position, tri_it );
        if( curDistance < ret.camDistance )
        {
            intersect = ret;
            curDistance = ret.camDistance;
        }
        hit = true;
    }



    /*for( int i = 0; i < numSpheres; i++ )
      {
      t = spheres[i]->hitTest( direction, position );
      if( t > 0 )
      {
      ret = spheres[i]->rayIntersect( direction, position, t );
      if(ret.hit)
      {
      if(ret.camDistance < curDistance || hit == false)
      {
      intersect = ret;
      curDistance = ret.camDistance;
      hit = true;
      }
      }
      }
      }*/
    for ( int i = 0; i < numPlanes; i++ )
    {
        t = planes[i]->hitTest( direction, position );
        if( t > 0 )
        {
            ret = planes[i]->rayIntersect( direction, position, t );
            if(ret.hit)
            {
                if(ret.camDistance < curDistance || hit == false)
                {
                    intersect = ret;
                    curDistance = ret.camDistance;
                    hit = true;
                }
            }
        }
    }
    /*for ( int i = 0; i < numTriangles; i++ )
      {
      t = triangles[i]->hitTest( direction, position );
      if( t > 0 )
      {
      ret = triangles[i]->rayIntersect( direction, position, t );
      if(ret.hit)
      {
      if(ret.camDistance < curDistance || hit == false)
      {
      intersect = ret;
      curDistance = ret.camDistance;
      hit = true;
      }
      }
      }
      }*/
    if (hit && intersect.obj != NULL)
    {
        color = (intersect.obj)->getColor(intersect, 0);
    }
    return color;
}
