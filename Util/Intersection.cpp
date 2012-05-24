#include "Intersection.h"

Color calcColor( Intersection inter, Scene scene )
{
   Color ret;
   ret.r = 0;
   ret.b = 0;
   ret.g = 0;

   for( int i = 0; i < scene.numPointLights; i++ )
   {
      PointLight temp = scene.lights[i];
      vec3 lvec = unit(newDirection(temp->pos, inter.hitMark ));

      float nlDot = dot(lvec, inter.normal );
      bool inShadow = false;

      //contruct possible hits for shadow ray using bvh

      ObjectInfo shadowInter;
      float lightDistance = distance(temp->pos, inter.hitMark );
      if( !inShadow )
      {
         vec3 r;
         r.x = -lvec.x + 2 * nlDot * inter.normal.x;
         r.y = -lvec.y + 2 * nlDot * inter.normal.y;
         r.z = -lvec.z + 2 * nlDot * inter.normal.z;
         r = r.unit();
         float rvDot = dot(r, inter.viewVector.unit());
         if(nlDot < 0)
            nlDot = 0;
         if(rvDot < 0)
            rvDot = 0;
         float powRV = pow( rvDot, 1.0/finish_roughness );

         int red =ret.r + temp->red * powRV*finish_specular;
         int green = ret.g + temp->green* powRV*finish_specular;
         int blue = ret.b + temp->blue * powRV*finish_specular;
         if( red < 1.0)
            ret.r = red;
         else
            ret.r = 1.0;
         if( green < 1.0)
            ret.g = green;
         else
            ret.g = 1.0;
         if( blue < 1.0)
            ret.b= blue;
         else
            ret.b = 1.0;

         if(ret.r + pigment.r * temp->red * nlDot*finish_diffuse < 1.0)
            ret.r+= pigment.r * temp->red * nlDot*finish_diffuse;
         else
            ret.r = 1.0;
         if(ret.g + pigment.g * temp->green* nlDot*finish_diffuse < 1.0)
            ret.g += pigment.g * temp->green * nlDot*finish_diffuse;
         else
            ret.g = 1.0;
         if(ret.b + pigment.b * temp->blue * nlDot*finish_diffuse < 1.0)
            ret.b+= pigment.b * temp->blue * nlDot*finish_diffuse;
         else
            ret.b = 1.0;
      }
   }
   //1.5 not 1 to increase the directlight which will be balanced in Util/tga.cpp during gamma correction
   float mod = 1.2 - finish_reflection - pigment_f*finish_refraction;
   ret.r = ret.r * mod;
   ret.g = ret.g * mod;
   ret.b = ret.b * mod;
   return ret;
}
