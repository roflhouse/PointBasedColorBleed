/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Surfel.h"

float surfelHitTest( const Surfel &surfel, const Ray &ray )
{
   vec3 direction = unit(ray.dir);
   vec3 position;
   vec3 normal = unit(surfel.normal);

   position.x = ray.pos.x;
   position.y = ray.pos.y;
   position.z = ray.pos.z;

   float vd = dot(normal, direction);
   if( vd > 0.001 )// && vd > -0.001 )
      return -1;
   float v0 = -(dot(position, normal) + surfel.distance );
   float t = v0/vd;
   if( t < 0.01)
      return -1;

   vec3 hitMark;
   hitMark.x = ray.pos.x + direction.x*t;
   hitMark.y = ray.pos.y + direction.y*t;
   hitMark.z = ray.pos.z + direction.z*t;
   float d = squareDistance( hitMark, surfel.pos );

   if( d < surfel.radius*surfel.radius )
      return t;
   else
      return -1;
}
SurfelArray createSurfelArray( int num )
{
   SurfelArray IA;
   IA.array = (Surfel *) malloc( sizeof(Surfel) * num );
   IA.num = 0;
   IA.max = num;
   return IA;
}
void growSA( SurfelArray &in )
{
   in.max *= 3;
   in.array = (Surfel *)realloc( in.array, sizeof(Surfel) * in.max );
   if( in.array == NULL )
   {
      printf("You have run out of memory realloc %d\n", in.max);
      exit(1);
   }
}
void shrinkSA( SurfelArray &in )
{
   in.max = in.num;
   in.array = (Surfel *)realloc( in.array, sizeof(Surfel) * in.max );
   if( in.array == NULL && in.num != 0 )
   {
      printf("Shrinking Failed %d\n", in.max);
      exit(1);
   }
}
void addToSA( SurfelArray &in, const Surfel &surfel )
{
   if( in.num +1 >=in.max )
   {
      growSA( in );
   }
   in.array[in.num] = surfel;
   in.num++;
}
void freeSurfelArray( SurfelArray &array )
{
   if( array.array != NULL )
      free( array.array );
   array.array = NULL;
}
