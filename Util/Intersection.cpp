#include "Intersection.h"
#define AREA_LIGHT_POINTS 50

Color directIllumination( const Intersection &inter, const Scene &scene )
{
   Color ret;
   ret.r = 0;//inter.hitMark.x;
   ret.b = 0;//inter.hitMark.y;
   ret.g = 0;

   for( int i = 0; i < scene.numPointLights; i++ )
   {
      for( int k = 0; k < AREA_LIGHT_POINTS; k++ )
      {
         PointLight light = scene.pointLights[i];
         vec3 lvec = unit(newDirection(light.points[k], inter.hitMark ));

         float nlDot = dot(lvec, inter.normal );
         bool inShadow = false;
         float lightDistance = distance( light.points[k], inter.hitMark );

         //contruct possible hits for shadow ray using bvh
         for( int j = 0; j < scene.numSpheres; j++ )
         {
            Ray shadowRay;
            shadowRay.pos = inter.hitMark;
            shadowRay.dir = lvec;
            float_2 temp = sphereHitTest(scene.spheres[j], shadowRay );
            float t = temp.t0;
            if( t < 0 )
               t = temp.t1;
            if( t > 0 )
            {
               Intersection info = sphereIntersection( scene.spheres[j], shadowRay, t );
               if(info.hit)
               {
                  if( distance( info.hitMark, inter.hitMark ) < lightDistance )
                  {
                     inShadow = true;
                     break;
                  }
               }
            }
         }

         ObjectInfo shadowInter;
         if( !inShadow )
         {
            vec3 r;
            r.x = -lvec.x + 2 * nlDot * inter.normal.x;
            r.y = -lvec.y + 2 * nlDot * inter.normal.y;
            r.z = -lvec.z + 2 * nlDot * inter.normal.z;
            r = unit(r);
            float rvDot = dot(r, unit(inter.viewVector));
            if(nlDot < 0)
               nlDot = 0;
            if(rvDot < 0)
               rvDot = 0;
            float n = 1.0/inter.colorInfo.finish_roughness;
            float powRV = pow( rvDot, n );
            float atten = 1 / (lightDistance*AREA_LIGHT_POINTS);

            ret.r += light.color.r * powRV*inter.colorInfo.finish_specular * atten;
            ret.g += light.color.g* powRV*inter.colorInfo.finish_specular * atten;
            ret.b += light.color.b * powRV*inter.colorInfo.finish_specular*atten;
            ret.r += inter.colorInfo.pigment.r * light.color.r * nlDot*inter.colorInfo.finish_diffuse *atten;
            ret.g += inter.colorInfo.pigment.g * light.color.g * nlDot*inter.colorInfo.finish_diffuse*atten;
            ret.b += inter.colorInfo.pigment.b * light.color.b * nlDot*inter.colorInfo.finish_diffuse*atten;
         }
      }
      ret.r = fmin( 1.0, ret.r );
      ret.g = fmin( 1.0, ret.g );
      ret.b = fmin( 1.0, ret.b );
   }
   return ret;
}
Surfel intersectionToSurfel( const Intersection &inter, const Scene &scene )
{
   Surfel surfel;
   vec3 normal = unit(inter.normal);
   surfel.pos = inter.hitMark;
   surfel.distance = -dot( normal, inter.hitMark );
   surfel.normal = normal;
   surfel.color = directIllumination( inter, scene );
   //surfel.radius = 0.30;
   surfel.radius = inter.radius;
   //surfel.radius = 0.08;
   //surfel.info = inter.colorInfo;
   return surfel;
}
Sphere intersectionToSphere( const Intersection &inter, const Scene &scene )
{
   Sphere sphere;
   sphere.pos = inter.hitMark;
   sphere.info.colorInfo.pigment = directIllumination( inter, scene );
   sphere.radius = .1;
   return sphere;
}

/////Intersection Array//////////

IntersectionArray createIntersectionArray( int num )
{
   IntersectionArray IA;
   IA.array = (Intersection *) malloc( sizeof(Intersection) * num );
   IA.num = 0;
   IA.max = num;
   return IA;
}
void growIA( IntersectionArray &in )
{
   in.max = in.max * 5 +1;
   in.array = (Intersection *)realloc( in.array, sizeof(Intersection) * in.max );
   if( in.array == NULL )
   {
      printf("You have run out of memory\n");
      exit(1);
   }
}
void shrinkIA( IntersectionArray &in )
{
   in.max = in.num+1;
   in.array = (Intersection *)realloc( in.array, sizeof(Intersection) * in.max );
   if( in.array == NULL )
   {
      printf("You have run out of memory\n");
      exit(1);
   }
}
void addToIA( IntersectionArray &in, const Intersection &intersection )
{
   if( in.num +1 >in.max )
   {
      growIA( in );
   }
   in.array[in.num] = intersection;
   in.num++;
}
void freeIntersectionArray( IntersectionArray &array )
{
   free( array.array );
}
