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

ObjectInfo createObjectInfo()
{
   ObjectInfo obj;
   obj.pigment.r = 0;
   obj.pigment.g = 0;
   obj.pigment.b = 0;
   obj.pigment_f = 0;

   obj.finish_ambient = 0;
   obj.finish_diffuse = 0;
   obj.finish_specular = 0;
   obj.finish_roughness = 0;
   obj.finish_reflection = 0;
   obj.finish_refraction = 0;
   obj.finish_ior = 0;
   obj.transforms = glm::mat4(1.0f);
   obj.transpose = glm::mat4(1.0f);
   return obj;
}
Color calcColor( ObjectInfo inter, Scene scene )
{
   Color ret;
   ret.r = 0;
   ret.b = 0;
   ret.g = 0;

   for( int i = 0; i < numLights; i++ )
   {
      PointLight *temp = lights[i];
      vec3 lvec = unit(newDirection(temp->pos, inter.hitMark ));

      float nlDot = dot(lvec, inter.normal );
      bool inShadow = false;

      //contruct possible hits for shadow ray using bvh

      ObjectInfo shadowInter;
      float lightDistance = distance(temp->pos, inter.hitMark );

      /*
         for( int j = 0; j < scene.numSpheres; j++)
         {
         float t = hitTest(scene.sphers[j], createRay(inter.hitMark, lvec) );
         if( t > 0 )
         {
         info = sphereIntersection(sphere[i], shadowRay, t ); lvec, inter.hitMark, t );
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
         float t = planes[j]->hitTest( lvec, inter.hitMark );
         if( t > 0 )
         {
         info = planeIntersection( planes[i], lvec, inter.hitMark, t );
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
         */

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
void parseObjectPigment( FILE *file, ObjectInfo &info )
{
   char cur = '\0';

   while(cur != '<')
   {
      if( fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error Parsing pigment\n");
         exit(1);
      }
   }
   float r = 0, g = 0, b = 0;
   if( fscanf(file, " %f , %f , %f ", &(r), &(g), &(b) ) == EOF )
   {
      printf("Error Parsing pigment\n");
      exit(1);
   }
   info.colorInfo.pigment.r = r;
   info.colorInfo.pigment.g = g;
   info.colorInfo.pigment.b = b;
   cur = ' ';
   while( isspace(cur) )
   {
      if( fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error Parsing pigment\n");
         exit(1);
      }
   }
   if( cur == ',' )
   {
      if( fscanf(file, " %f ", &info.colorInfo.pigment_f ) == EOF )
      {
         printf("Error Parsing pigment\n");
         exit(1);
      }
   }
   else if ( cur == '>')
   {
   }
   else
   {
      printf("Error Parsing pigment\n");
      exit(1);
   }

   while(cur != '}')
   {
      if( fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error Parsing pigment\n");
         exit(1);
      }
   }
   printf( " pigment: %d, %d, %d\n",  info.colorInfo.pigment.r, info.colorInfo.pigment.g, info.colorInfo.pigment.b );
}
int parseObjectFinish( FILE *file, ObjectInfo &info )
{
   char cur = ' ';
   while( cur != '{' )
   {
      if( fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error parsing finish\n");
         exit(1);
      }
   }
   cur = ' ';

   while(cur != '}')
   {
      //Optional things
      while ( isspace( cur ) )
      {
         if( fscanf(file, "%c", &cur) == EOF)
         {
            printf("Error parsing finish\n");
            exit(1);
         }
      }
      //ambient
      if( cur == 'a' )
      {
         while(cur != 't')
         {
            if( fscanf(file, "%c", &cur) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }
         }
         if( fscanf(file, " %f ", &(info.colorInfo.finish_ambient) ) == EOF )
         {
            printf("Error parsing finish\n");
            exit(1);
         }
         cur =  ' ';
      }
      //defuse
      else if( cur == 'd' )
      {
         //read in tell next number
         while(cur != 'e')
         {
            if( fscanf(file, "%c", &cur) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }
         }
         if( fscanf(file, " %f ", &(info.colorInfo.finish_diffuse) ) == EOF )
         {
            printf("Error parsing finish\n");
            exit(1);
         }
         cur =  ' ';
      }
      //Specular
      else if( cur == 's' )
      {
         while( cur != 'r' )
         {
            if( fscanf(file, "%c", &cur) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }
         }
         if( fscanf( file, " %f ", &(info.colorInfo.finish_specular)) == EOF)
         {
            printf("Error parsing finish\n");
            exit(1);
         }
         cur =  ' ';
      }
      //roughness or reflection, or refaction
      else if( cur == 'r' )
      {
         if( fscanf(file, "%c", &cur) == EOF)
         {
            printf("Error parsing finish\n");
            exit(1);
         }
         //roughness
         if( cur == 'o' )
         {
            while( cur != 's' )
            {
               if( fscanf(file, "%c", &cur) == EOF)
               {
                  printf("Error parsing finish\n");
                  exit(1);
               }
            }
            if( fscanf(file, "%c", &cur) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }
            if( cur != 's' )
            {
               printf("Error parsing finish\n");
               exit(1);
            }
            if( fscanf( file, " %f ", &(info.colorInfo.finish_roughness)) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }
            cur = ' ';
         }
         else if( cur == 'e' )
         {
            //reflection or refraction
            if( fscanf(file, "%c", &cur) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }

            if( cur != 'f' )
            {
               printf("Error parsing finish\n");
               exit(1);
            }

            if( fscanf(file, "%c", &cur) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }
            if(cur == 'l')
            {
               //reflection
               while( cur != 'n' )
               {
                  if( fscanf(file, "%c", &cur) == EOF)
                  {
                     printf("Error parsing finish\n");
                     exit(1);
                  }
               }
               if( fscanf( file, " %f ", &(info.colorInfo.finish_reflection)) == EOF)
               {
                  printf("Error parsing finish\n");
                  exit(1);
               }
            }
            else if( cur == 'r' )
            {
               //refraction
               while( cur != 'n' )
               {
                  if( fscanf(file, "%c", &cur) == EOF)
                  {
                     printf("Error parsing finish\n");
                     exit(1);
                  }
               }
               if( fscanf( file, " %f ", &(info.colorInfo.finish_refraction)) == EOF)
               {
                  printf("Error parsing finish\n");
                  exit(1);
               }
            }
            else
            {
               printf("Error parsing finish\n");
               exit(1);
            }
         }
         cur = ' ';
      }
      //ior See: "My Life for Aiur"
      else if( cur == 'i')
      {
         while( cur != 'r' )
         {
            if( fscanf(file, "%c", &cur) == EOF)
            {
               printf("Error parsing finish\n");
               exit(1);
            }
         }
         if( fscanf( file, " %f ", &(info.colorInfo.finish_ior)) == EOF)
         {
            printf("Error parsing finish\n");
            exit(1);
         }
         printf( "%f\n", info.color.info.finish_ior);
         cur = ' ';
      }
      else if( cur == '}' )
      {
         //do nothing
      }
      else
      {
         printf("Error parsing finish\n");
         exit(1);
      }
   }
   printf( "Finish: %f, %f, %f, %f, %f, %f, %f\n", info.colorInfo.finish_ambient, info.colorInfo.finish_diffuse,
         info.colorInfo.finish_specular, info.colorInfo.finish_roughness,
         infor.colorInfo.finish_reflection, info.colorInfo.finish_refraction, info.colorInfo.finish_ior);

}
void parseObjectTransforms( FILE *file, ObjectInfo &info )
{
   char cur = '\0';

   //read in until the end of the object
   while( cur != '}' )
   {
      if( fscanf( file, "%c", &cur ) == EOF)
      {
         printf("Error parsing Transforms\n");
         exit(1);
      }

      //Translate found
      if(cur == 't' || cur == 'T')
      {
         while( cur != '<' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
            {
               printf("Error parsing Transforms\n");
               exit(1);
            }
         }
         vec3 translate;
         if( fscanf( file, " %f , %f , %f ", &(translate.x), &(translate.y), &(translate.z) ) == EOF )
         {
            printf("Error parsing Transforms\n");
            exit(1);
         }

         while( cur != '>' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
            {
               printf("Error parsing Transforms\n");
               exit(1);
            }
         }
         printf(" translate: %f , %f , %f\n", translate.x, translate.y, translate.z );
         //add translate to transform matrix
         info.transforms = glm::translate( info.transforms,
               glm::vec3( translate.x, translate.y, translate.z ));
      }
      //Scale found
      else if( cur == 's' || cur == 'S' )
      {
         while( cur != '<' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
            {
               printf("Error parsing Transforms\n");
               exit(1);
            }
         }
         vec3 scale;
         if( fscanf( file, " %f , %f , %f ", &(scale.x), &(scale.y), &(scale.z) ) == EOF )
         {
            printf("Error parsing Transforms\n");
            exit(1);
         }

         while( cur != '>' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
            {
               printf("Error parsing Transforms\n");
               exit(1);
            }
         }
         printf(" Scale: %f , %f , %f\n", scale.x, scale.y, scale.z );
         info.transforms = glm::scale( info.transforms, glm::vec3( scale.x, scale.y, scale.z ) );
      }
      //Rotate found
      else if( cur == 'r' || cur == 'R' )
      {
         while( cur != '<' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
            {
               printf("Error parsing Transforms\n");
               exit(1);
            }
         }

         vec3 rotate;
         if( fscanf( file, " %f , %f , %f ", &(rotate.x), &(rotate.y), &(rotate.z) ) ==EOF )
         {
            printf("Error parsing Transforms\n");
            exit(1);
         }

         while( cur != '>' )
         {
            if( fscanf( file, "%c", &cur) == EOF )
            {
               printf("Error parsing Transforms\n");
               exit(1);
            }
         }
         printf(" Rotate: %f , %f , %f\n", rotate.x, rotate.y, rotate.z );
         glm::mat4 rotz = glm::rotate( glm::mat4(1.0f), rotate.z, glm::vec3( 0.0, 0.0, 1.0 ) );
         glm::mat4 roty = glm::rotate( glm::mat4(1.0f), rotate.y, glm::vec3( 0.0, 1.0, 0.0 ) );
         glm::mat4 rotx = glm::rotate( glm::mat4(1.0f), rotate.x, glm::vec3( 1.0, 0.0, 0.0 ) );
         info.transforms = rotz * roty * rotx * info.transforms;
      }
   }
   info.transforms = glm::inverse( info.transforms );
   info.transpose = glm::transpose( info.transforms );
}
