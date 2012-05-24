#include "Intersection.h"

Color directIllumination( Intersection inter, Scene scene )
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

      //contruct possible hits for shadow ray using bvh

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
