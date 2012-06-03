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
   direction.x = direction.x;
   direction.y = direction.y;
   direction.z = direction.z;
   position.x = ray.pos.x;
   position.y = ray.pos.y;
   position.z = ray.pos.z;

   /*normal.x *= -1;
   normal.y *= -1;
   normal.z *= -1;
   */
   float vd = dot(normal, direction);
   if( vd < 0.0001 && vd > -0.0001 )
      return -1;
   float v0 = -(dot(position, normal) - surfel.distance );
   float t = v0/vd;
   vec3 hitMark;
   hitMark.x = ray.pos.x + direction.x*t;
   hitMark.y = ray.pos.y + direction.y*t;
   hitMark.z = ray.pos.z + direction.z*t;
   float d = squareDistance( hitMark, surfel.pos );
   if( ray.i == 10 && ray.j == 10 && surfel.pos.x > -4.001 && surfel.pos.x < -3.999 && surfel.pos.z > 3.99 && surfel.pos.z < 4.01 )
      printf("surfel ray dir:%f %f %f pos: %f %f %f  surf n: %f %f %f pos: %f %f %f,  t: %f dis: %f vd: %f v0: %f    %f hitMark: %f %f %f\n", direction.x, direction.y, direction.z, ray.pos.x, ray.pos.y, ray.pos.z, surfel.normal.x, surfel.normal.y, surfel.normal.z, surfel.pos.x, surfel.pos.y, surfel.pos.z, t, surfel.distance, vd, v0 , d, hitMark.x, hitMark.y, hitMark.z);
   if( t < 0.001)
   {
      return -1;
   }


   if( d < surfel.radius*surfel.radius )
   {
      return t;
   }
   else
   {
      return -1;
   }
}

SurfelArray createSurfelArray()
{
   SurfelArray IA;
   IA.array = (Surfel *) malloc( sizeof(Surfel) * 1000 );
   IA.num = 0;
   IA.max = 1000;
   return IA;
}
void growSA( SurfelArray &in )
{
   in.max *= 5;
   in.array = (Surfel *)realloc( in.array, sizeof(Surfel) * in.max );
   if( in.array == NULL )
   {
      printf("You have run out of memory\n");
      exit(1);
   }
}
void shrinkSA( SurfelArray &in )
{
   in.max = in.num;
   in.array = (Surfel *)realloc( in.array, sizeof(Surfel) * in.max );
   if( in.array == NULL )
   {
      printf("You have run out of memory\n");
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
   free( array.array );
}
