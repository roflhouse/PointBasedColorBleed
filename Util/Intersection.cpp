#include "Intersection.h"

Color directIllumination( const Intersection &inter, const Scene &scene )
{
   Color ret;
   ret.r = 0;
   ret.b = 0;
   ret.g = 0;

   for( int i = 0; i < scene.numPointLights; i++ )
   {
      PointLight temp = scene.pointLights[i];
      vec3 lvec = unit(newDirection(temp.pos, inter.hitMark ));

      float nlDot = dot(lvec, inter.normal );
      bool inShadow = false;
      float lightDistance = distance( temp.pos, inter.hitMark );

      //contruct possible hits for shadow ray using bvh
      for( int j = 0; j < scene.numSpheres; j++ )
      {
         Ray shadowRay;
         shadowRay.pos = inter.hitMark;
         shadowRay.dir = lvec;
         float t = sphereHitTest(scene.spheres[j], shadowRay );
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
         float powRV = pow( rvDot, 1.0/inter.colorInfo.finish_roughness );

         ret.r  =ret.r + temp.color.r * powRV*inter.colorInfo.finish_specular;
         ret.g = ret.g + temp.color.g* powRV*inter.colorInfo.finish_specular;
         ret.b = ret.b + temp.color.b * powRV*inter.colorInfo.finish_specular;
         ret.r+= inter.colorInfo.pigment.r * temp.color.r * nlDot*inter.colorInfo.finish_diffuse;
         ret.g += inter.colorInfo.pigment.g * temp.color.g * nlDot*inter.colorInfo.finish_diffuse;
         ret.b+= inter.colorInfo.pigment.b * temp.color.b * nlDot*inter.colorInfo.finish_diffuse;
         ret = limitColor( ret );
      }
   }
   //1.5 not 1 to increase the directlight which will be balanced in Util/tga.cpp during gamma correction
   float mod = 1;//.2 - finish_reflection - pigment_f*finish_refraction;
   ret.r = ret.r * mod;
   ret.g = ret.g * mod;
   ret.b = ret.b * mod;
   return ret;
}
Surfel intersectionToSurfel( const Intersection &inter, const Scene &scene )
{
   Surfel surfel;
   surfel.pos = inter.hitMark;
   surfel.distance = -dot( inter.normal, inter.hitMark );
   surfel.normal = inter.normal;
   surfel.color = directIllumination( inter, scene );
   surfel.radius = 1;
   return surfel;
}

/////Intersection Array//////////

IntersectionArray createIntersectionArray()
{
   IntersectionArray IA;
   IA.array = (Intersection *) malloc( sizeof(Intersection) * 1000 );
   IA.num = 0;
   IA.max = 1000;
   return IA;
}
void growIA( IntersectionArray &in )
{
   in.max = in.max * 5;
   in.array = (Intersection *)realloc( in.array, sizeof(Intersection) * in.max );
   if( in.array == NULL )
   {
      printf("You have run out of memory\n");
      exit(1);
   }
}
void shrinkIA( IntersectionArray &in )
{
   in.max = in.num;
   in.array = (Intersection *)realloc( in.array, sizeof(Intersection) * in.max );
   if( in.array == NULL )
   {
      printf("You have run out of memory\n");
      exit(1);
   }
}
void addToIA( IntersectionArray &in, const Intersection &intersection )
{
   if( in.num +1 >=in.max )
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
