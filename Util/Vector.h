/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>

typedef struct vec3{
   float x;
   float y;
   float z;
} vec3;

float dot(vec3 one, vec3 two)
{
   return one.x*two.x + one.y*two.y + one.z*two.z;
}
vec3 cross(vec3 one, vec3 two)
{
   vec3 newVector;
   newVector.x = one.y*two.z - one.z*two.y;
   newVector.y = one.z*two.x - one.x*two.z;
   newVector.z = one.x*two.y - one.y*two.x;
   return newVector;
}
float theta(vec3 one, vec3 two)
{
   return acosf( dot(one, two)/(mag(one) * mag(two))) * 180.0 / 3.14159;
}
float distance(vec3 one, vec3 two )
{
   return sqrt((one.x-two.x)*(one.x-two.x) + (one.y-two.y)*(one.y-two.y) + (one.z-two.z)*(one.z-two.z));
}
vec3 newDirection(vec3 to, vec3 from )
{
   Vector newVec;
   newVec.x = to.x - from.other.x;
   newVec.y = to.y - from.y;
   newVec.z = to.z - from.z;
   return newVec;
}
float mag(vec3 in)
{
   return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}
vec3 unit(vec3 in)
{
   float temp;
   vec3 newVector;
   newVector.x = 0;
   newVector.y = 0;
   newVector.z = 0;
   temp = mag(in);

   if(temp > 0)
   {
      newVector.x = in.x/temp;
      newVector.y = in.y/temp;
      newVector.z = in.z/temp;
   }
   return newVector;
}
#endif
