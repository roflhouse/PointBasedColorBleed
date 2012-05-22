/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#include "Object.h"
#include "LightSource.h"
#include "../Util/BVH.h"

#define PI 3.141592

//Extern Things that use Object.h
extern BVH *bvh;
extern LightSource **lights;

Object::Object()
{
   pigment.r = 0;
   pigment.g = 0;
   pigment.b = 0;
   pigment_f = 0;

   finish_ambient = 0;
   finish_diffuse = 0;
   finish_specular = 0;
   finish_roughness = 0;
   finish_reflection = 0;
   finish_refraction = 0;
   finish_ior = 0;
   transforms = glm::mat4(1.0f);
   transpose = glm::mat4(1.0f);
}
Object::pixel Object::getColor( Object::rayInfo ray, int monteCarlo )
{
   pixel ret;
   ret.r = 0;
   ret.b = 0;
   ret.g = 0;

   for( int i = 0; i < numLights; i++ )
   {
      LightSource *temp = lights[i];
      Vector lvec = temp->location.newDirection( ray.hitMark ).unit();

      float nlDot = lvec.dot( ray.normal );
      bool inShadow = false;

      //contruct possible hits for shadow ray using bvh

      Object **possible;
      Object::rayInfo info;
      float lightDistance = temp->location.distance( ray.hitMark );
      int numPos = 0;

      bvh->getIntersections( lvec, ray.hitMark, &possible, &numPos );

      for( int j = 0; j < numPos; j++)
      {
         float t = possible[j]->hitTest( lvec, ray.hitMark );
         if( t > 0 )
         {
            info = planes[i]->rayIntersect( lvec, ray.hitMark, t );
            if(info.hit)
            {
               if( info.camDistance < lightDistance )
               {
                  inShadow = true;
                  free( possible );
                  break;
               }
            }
         }
      }
      if( !inShadow )
      {
         for( int j = 0; j < numPlanes; j++)
         {
            float t = planes[j]->hitTest( lvec, ray.hitMark );
            if( t > 0 )
            {
               info = planes[i]->rayIntersect( lvec, ray.hitMark, t );
               if(info.hit)
               {
                  if( info.camDistance < lightDistance )
                  {
                     inShadow = true;
                     break;
                  }
               }
            }
         }
      }

      if( !inShadow )
      {
         Vector r;
         r.x = -lvec.x + 2 * nlDot * ray.normal.x;
         r.y = -lvec.y + 2 * nlDot * ray.normal.y;
         r.z = -lvec.z + 2 * nlDot * ray.normal.z;
         r = r.unit();
         float rvDot = r.dot(ray.viewVector.unit());
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
      if( monteCarlo == 0 )
      {
         if(ret.r + pigment.r * temp->red * finish_ambient < 1.0)
            ret.r+= pigment.r * temp->red * finish_ambient;
         else
            ret.r = 1.0;
         if(ret.g + pigment.g * temp->green * finish_ambient< 1.0)
            ret.g += pigment.g * temp->green * finish_ambient ;
         else
            ret.g = 1.0;
         if(ret.b + pigment.b * temp->blue * finish_ambient < 1.0)
            ret.b+= pigment.b * temp->blue * finish_ambient;
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
int Object::parsePigment( FILE *file )
{
   char cur = '\0';

   while(cur != '<')
   {
      if( fscanf(file, "%c", &cur) == EOF)
         return 1;
   }
   float r = 0, g = 0, b = 0;
   if( fscanf(file, " %f , %f , %f ", &(r), &(g), &(b) ) == EOF )
      return 1;
   pigment.r = r;
   pigment.g = g;
   pigment.b = b;
   cur = ' ';
   while( isspace(cur) )
   {
      if( fscanf(file, "%c", &cur) == EOF)
         return 1;
   }
   if( cur == ',' )
   {
      if( fscanf(file, " %f ", &pigment_f ) == EOF )
         return 1;
   }
   else if ( cur == '>')
   {
   }
   else
      return 1;

   while(cur != '}')
   {
      if( fscanf(file, "%c", &cur) == EOF)
         return 1;
   }
   //printf( " pigment: %d, %d, %d\n",  pigment.r, pigment.g, pigment.b );
   return 0;
}
int Object::parseFinish( FILE *file )
{
   char cur = ' ';
   while( cur != '{' )
   {
      if( fscanf(file, "%c", &cur) == EOF)
      {
         return 1;
      }
   }
   cur = ' ';

   while(cur != '}')
   {
      //Optional things
      while ( isspace( cur ) )
      {
         if( fscanf(file, "%c", &cur) == EOF)
            return 1;
      }
      //ambient
      if( cur == 'a' )
      {
         while(cur != 't')
         {
            if( fscanf(file, "%c", &cur) == EOF)
               return 1;
         }
         if( fscanf(file, " %f ", &(finish_ambient) ) == EOF )
            return 1;
         cur =  ' ';
      }
      //defuse
      else if( cur == 'd' )
      {
         //read in tell next number
         while(cur != 'e')
         {
            if( fscanf(file, "%c", &cur) == EOF)
               return 1;
         }
         if( fscanf(file, " %f ", &(finish_diffuse) ) == EOF )
            return 1;
         cur =  ' ';
      }
      //Specular
      else if( cur == 's' )
      {
         while( cur != 'r' )
         {
            if( fscanf(file, "%c", &cur) == EOF)
               return 1;
         }
         if( fscanf( file, " %f ", &(finish_specular)) == EOF)
            return 1;
         cur =  ' ';
      }
      //roughness or reflection, or refaction
      else if( cur == 'r' )
      {
         if( fscanf(file, "%c", &cur) == EOF)
            return 1;
         //roughness
         if( cur == 'o' )
         {
            while( cur != 's' )
            {
               if( fscanf(file, "%c", &cur) == EOF)
                  return 1;
            }
            if( fscanf(file, "%c", &cur) == EOF)
               return 1;
            if( cur != 's' )
               return 1;
            if( fscanf( file, " %f ", &(finish_roughness)) == EOF)
               return 1;
            cur = ' ';
         }
         else if( cur == 'e' )
         {
            //reflection or refraction
            if( fscanf(file, "%c", &cur) == EOF)
               return 1;

            if( cur != 'f' )
               return 1;

            if( fscanf(file, "%c", &cur) == EOF)
               return 1;
            if(cur == 'l')
            {
               //reflection
               while( cur != 'n' )
               {
                  if( fscanf(file, "%c", &cur) == EOF)
                     return 1;
               }
               if( fscanf( file, " %f ", &(finish_reflection)) == EOF)
                  return 1;
            }
            else if( cur == 'r' )
            {
               //refraction
               while( cur != 'n' )
               {
                  if( fscanf(file, "%c", &cur) == EOF)
                     return 1;
               }
               if( fscanf( file, " %f ", &(finish_refraction)) == EOF)
                  return 1;
            }
            else
               return 1;
         }
         cur = ' ';
      }
      //ior See: "My Life for Aiur"
      else if( cur == 'i')
      {
         while( cur != 'r' )
         {
            if( fscanf(file, "%c", &cur) == EOF)
               return 1;
         }
         if( fscanf( file, " %f ", &(finish_ior)) == EOF)
            return 1;
         //printf( "%f\n", finish_ior);
         cur = ' ';
      }
      else if( cur == '}' )
      {
         //do nothing
      }
      else
      {
         //printf( "this %c\n", cur );
         return 1;
      }
   }
   //printf( "Finish: %f, %f, %f, %f, %f, %f, %f\n", finish_ambient, finish_diffuse, finish_specular, finish_roughness,
   //        finish_reflection, finish_refraction, finish_ior);
   return 0;

}
int Object::parseTransforms( FILE *file )
{
   char cur = '\0';

   //read in until the end of the object
   while( cur != '}' )
   {
      if( fscanf( file, "%c", &cur ) == EOF)
         return 1;

      //Translate found
      if(cur == 't' || cur == 'T')
      {
         while( cur != '<' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
               return 1;
         }
         Vector translate;
         if( fscanf( file, " %f , %f , %f ", &(translate.x), &(translate.y), &(translate.z) ) == EOF )
            return 1;

         while( cur != '>' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
               return 1;
         }
         //printf(" translate: %f , %f , %f\n", translate.x, translate.y, translate.z );
         //add translate to transform matrix
         transforms = glm::translate( transforms, glm::vec3( translate.x, translate.y, translate.z ));
      }
      //Scale found
      else if( cur == 's' || cur == 'S' )
      {
         while( cur != '<' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
               return 1;
         }
         Vector scale;
         if( fscanf( file, " %f , %f , %f ", &(scale.x), &(scale.y), &(scale.z) ) == EOF )
            return 1;

         while( cur != '>' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
               return 1;
         }
         //printf(" Scale: %f , %f , %f\n", scale.x, scale.y, scale.z );
         transforms = glm::scale( transforms, glm::vec3( scale.x, scale.y, scale.z ) );
      }
      //Rotate found
      else if( cur == 'r' || cur == 'R' )
      {
         while( cur != '<' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
               return 1;
         }

         Vector rotate;
         if( fscanf( file, " %f , %f , %f ", &(rotate.x), &(rotate.y), &(rotate.z) ) ==EOF )
            return 1;

         while( cur != '>' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
               return 1;
         }
         //printf(" Rotate: %f , %f , %f\n", rotate.x, rotate.y, rotate.z );
         glm::mat4 rotz = glm::rotate( glm::mat4(1.0f), rotate.z, glm::vec3( 0.0, 0.0, 1.0 ) );
         glm::mat4 roty = glm::rotate( glm::mat4(1.0f), rotate.y, glm::vec3( 0.0, 1.0, 0.0 ) );
         glm::mat4 rotx = glm::rotate( glm::mat4(1.0f), rotate.x, glm::vec3( 1.0, 0.0, 0.0 ) );
         transforms = rotz * roty * rotx * transforms;
      }
   }
   transforms = glm::inverse( transforms );
   transpose = glm::transpose( transforms );
   return 0;
}
void Object::transformMinMax( Vector *m, Vector *x )
{
   //transform all four corners
   Vector one( m->x, m->y, m->z );
   Vector two( m->x, m->y, x->z );
   Vector three( m->x, x->y, m->z );
   Vector four( m->x, x->y, x->z );
   Vector five( x->x, m->y, m->z );
   Vector six( x->x, m->y, x->z );
   Vector seven( x->x, x->y, m->z );
   Vector eight( x->x, x->y, x->z );

   glm::vec4 oneg = glm::vec4( one.x, one.y, one.z, 1.0f );
   glm::vec4 twog = glm::vec4( two.x, two.y, two.z, 1.0f );
   glm::vec4 threeg = glm::vec4( three.x, three.y, three.z, 1.0f );
   glm::vec4 fourg = glm::vec4( four.x, four.y, four.z, 1.0f );
   glm::vec4 fiveg = glm::vec4( five.x, five.y, five.z, 1.0f );
   glm::vec4 sixg = glm::vec4( six.x, six.y, six.z, 1.0f );
   glm::vec4 seveng = glm::vec4( seven.x, seven.y, seven.z, 1.0f );
   glm::vec4 eightg = glm::vec4( eight.x, eight.y, eight.z, 1.0f );

   glm::mat4 trans = glm::inverse( transforms );

   oneg = trans * oneg;
   twog = trans * twog;
   threeg = trans * threeg;
   fourg = trans * fourg;
   fiveg = trans * fiveg;
   sixg = trans * sixg;
   seveng = trans * seveng;
   eightg = trans * eightg;

   Vector mint = one;
   Vector maxt = one;

   mint.x = std::min( mint.x, twog[0] );
   maxt.x = std::max( maxt.x, twog[0] );
   mint.y = std::min( mint.y, twog[1] );
   maxt.y = std::max( maxt.y, twog[1] );
   mint.z = std::min( mint.z, twog[2] );
   maxt.z = std::max( maxt.z, twog[2] );

   mint.x = std::min( mint.x, threeg[0] );
   maxt.x = std::max( maxt.x, threeg[0] );
   mint.y = std::min( mint.y, threeg[1] );
   maxt.y = std::max( maxt.y, threeg[1] );
   mint.z = std::min( mint.z, threeg[2] );
   maxt.z = std::max( maxt.z, threeg[2] );

   mint.x = std::min( mint.x, fourg[0] );
   maxt.x = std::max( maxt.x, fourg[0] );
   mint.y = std::min( mint.y, fourg[1] );
   maxt.y = std::max( maxt.y, fourg[1] );
   mint.z = std::min( mint.z, fourg[2] );
   maxt.z = std::max( maxt.z, fourg[2] );

   mint.x = std::min( mint.x, fiveg[0] );
   maxt.x = std::max( maxt.x, fiveg[0] );
   mint.y = std::min( mint.y, fiveg[1] );
   maxt.y = std::max( maxt.y, fiveg[1] );
   mint.z = std::min( mint.z, fiveg[2] );
   maxt.z = std::max( maxt.z, fiveg[2] );

   mint.x = std::min( mint.x, sixg[0] );
   maxt.x = std::max( maxt.x, sixg[0] );
   mint.y = std::min( mint.y, sixg[1] );
   maxt.y = std::max( maxt.y, sixg[1] );
   mint.z = std::min( mint.z, sixg[2] );
   maxt.z = std::max( maxt.z, sixg[2] );

   mint.x = std::min( mint.x, seveng[0] );
   maxt.x = std::max( maxt.x, seveng[0] );
   mint.y = std::min( mint.y, seveng[1] );
   maxt.y = std::max( maxt.y, seveng[1] );
   mint.z = std::min( mint.z, seveng[2] );
   maxt.z = std::max( maxt.z, seveng[2] );

   mint.x = std::min( mint.x, eightg[0] );
   maxt.x = std::max( maxt.x, eightg[0] );
   mint.y = std::min( mint.y, eightg[1] );
   maxt.y = std::max( maxt.y, eightg[1] );
   mint.z = std::min( mint.z, eightg[2] );
   maxt.z = std::max( maxt.z, eightg[2] );
   *m = mint;
   *x = maxt;
}
