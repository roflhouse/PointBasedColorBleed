/**
 *  CPE 2011
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "BoundingBox.h"
#define RADIUS 0.002 

BoundingBox createBoundingBox( const vec3 &min, const vec3 &max )
{
   BoundingBox ret;
   ret.min = min;
   ret.max = max;
   if( min.x > max.x )
   {
      printf("Warning BoundingBox min.x %f > max.xi %f\n", min.x, max.x);
   }
   if( min.y > max.y )
   {
      printf("Warning BoundingBox min.y %f > max.y %f\n", min.y, max.y);
   }
   if( min.z > max.z )
   {
      printf("Warning BoundingBox min.z %f > max.z\n %f", min.z, max.z);
   }
   return ret;
}
bool isIn( const BoundingBox &box, const vec3 &pos )
{
   if (pos.x >= box.max.x || pos.x < box.min.x )
      return false;
   if (pos.y >= box.max.y || pos.y < box.min.y )
      return false;
   if (pos.z >= box.max.z || pos.z < box.min.z )
      return false;
   return true;
}
bool testForHit( const BoundingBox &boxIn, const Ray &ray )
{
   vec3 min = boxIn.min;
   min.x -= RADIUS;
   min.y -= RADIUS;
   min.z -= RADIUS;

   vec3 max = boxIn.max;
   max.x += RADIUS;
   max.y += RADIUS;
   max.z += RADIUS;
   BoundingBox box = createBoundingBox(min, max);
   if( ray.dir.x > -0.0001 && ray.dir.x < 0.0001 )
   {
      if( ray.pos.x < box.min.x || ray.pos.x > box.max.x )
         return false;
   }
   if( ray.dir.y > -0.0001 && ray.dir.y < 0.0001 )
   {
      if( ray.pos.y < box.min.y || ray.pos.y > box.max.y )
         return false;
   }
   if( ray.dir.z > -0.0001 && ray.dir.z < 0.0001 )
   {
      if( ray.pos.z < box.min.z || ray.pos.z > box.max.z )
         return false;
   }
   float txmin = (box.min.x - ray.pos.x) / ray.dir.x;
   float tymin = (box.min.y - ray.pos.y) / ray.dir.y;
   float tzmin = (box.min.z - ray.pos.z) / ray.dir.z;
   float txmax = (box.max.x - ray.pos.x) / ray.dir.x;
   float tymax = (box.max.y - ray.pos.y) / ray.dir.y;
   float tzmax = (box.max.z - ray.pos.z) / ray.dir.z;

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
BoundingBox *getSubBoxes( const BoundingBox &box )
{
   BoundingBox *boxes = (BoundingBox *) malloc( sizeof(BoundingBox) * 8 );

   for( int i = 0; i < 8; i++ )
   {
      boxes[i].min = box.min;
      boxes[i].max = box.max;
   }

   vec3 half;
   half.x = box.min.x + (box.max.x - box.min.x)/2;
   half.y = box.min.y + (box.max.y - box.min.y)/2;
   half.z = box.min.z + (box.max.z - box.min.z)/2;

   //first box
   boxes[0].max.x = half.x;
   boxes[0].max.y = half.y;
   boxes[0].max.z = half.z;

   //second box (x change)
   boxes[1].min.x = half.x;
   boxes[1].max.y = half.y;
   boxes[1].max.z = half.z;

   //third box (y change)
   boxes[2].max.x = half.x;
   boxes[2].min.y = half.y;
   boxes[2].max.z = half.z;

   //fourth box ( x and y change )
   boxes[3].min.x = half.x;
   boxes[3].min.y = half.y;
   boxes[3].max.z = half.z;

   //fifth box (z change)
   boxes[4].max.x = half.x;
   boxes[4].max.y = half.y;
   boxes[4].min.z = half.z;

   //sixth (z and x change)
   boxes[5].min.x = half.x;
   boxes[5].max.y = half.y;
   boxes[5].min.z = half.z;

   //seventh (z and y)
   boxes[6].max.x = half.x;
   boxes[6].min.y = half.y;
   boxes[6].min.z = half.z;

   //8th (z,y,x )
   boxes[7].min.x = half.x;
   boxes[7].min.y = half.y;
   boxes[7].min.z = half.z;

   return boxes;
}
int belowHorizon( const BoundingBox &box, vec3 &position, vec3 &normal )
{
   //eight points off cube
   vec3 points[8];
   points[0] = box.min;
   points[1] = box.min;
   points[1].z = box.max.z;
   points[2] = box.min;
   points[2].y = box.max.y;
   points[3] = box.min;
   points[3].y = box.max.y;
   points[3].z = box.max.z;
   points[4] = box.min;
   points[4].x = box.max.x;
   points[5] = box.min;
   points[5].x = box.max.x;
   points[5].z = box.max.z;
   points[6] = box.min;
   points[6].x = box.max.x;
   points[6].y = box.max.y;
   points[7] = box.max;
   int below = 0;
   for( int i = 0; i < 8; i++ )
   {
      vec3 temp = unit( newDirection( points[i], position ) );
      if( dot( normal, temp ) <= 0 )
         below++;
   }
   return below;
}
vec3 getCenter( const BoundingBox &box )
{
   vec3 c;
   c.x = (box.max.x -box.min.x)/2 + box.min.x;
   c.y = (box.max.y -box.min.y)/2 + box.min.y;
   c.x = (box.max.z -box.min.z)/2 + box.min.z;
   return c;
}
float distanceToBox( const BoundingBox &box, vec3 &pos )
{
   vec3 close;
   close.x = fmin( pos.x, box.max.x );
   close.y = fmin( pos.y, box.max.y );
   close.z = fmin( pos.z, box.max.z );
   close.x = fmax( close.x, box.min.x );
   close.y = fmax( close.y, box.min.y );
   close.z = fmax( close.z, box.min.z );
   return distance( close, pos );
}
