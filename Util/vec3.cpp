/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "vec3.h"

float mag(const vec3 &in)
{
   return sqrt(in.x*in.x + in.y*in.y + in.z*in.z);
}
float dot(const vec3 &one, const vec3 &two)
{
   return one.x*two.x + one.y*two.y + one.z*two.z;
}
vec3 cross(const vec3 &one,const vec3 &two)
{
   vec3 newVector;
   newVector.x = one.y*two.z - one.z*two.y;
   newVector.y = one.z*two.x - one.x*two.z;
   newVector.z = one.x*two.y - one.y*two.x;
   return newVector;
}
float theta(const vec3 &one, const vec3 &two)
{
   return acosf( dot(one, two)/(mag(one) * mag(two))) * 180.0 / 3.14159;
}
float distance(const vec3 &one, const vec3 &two )
{
   return sqrt((one.x-two.x)*(one.x-two.x) + (one.y-two.y)*(one.y-two.y) + (one.z-two.z)*(one.z-two.z));
}
vec3 newDirection(const vec3 &to, const vec3 &from )
{
   vec3 newVec;
   newVec.x = to.x - from.x;
   newVec.y = to.y - from.y;
   newVec.z = to.z - from.z;
   return newVec;
}
vec3 unit(const vec3 &in)
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
