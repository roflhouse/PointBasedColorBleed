/**
 *  CPE 2010
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */
#include "Triangle.h"

float triangleHitTest( const Triangle &triangle, const Ray &ray )
{
   vec3 direction = unit( ray.dir );
   vec3 position;
   glm::vec4 dir = glm::vec4(direction.x, direction.y, direction.z, 0.0f);
   glm::vec4 pos = glm::vec4(ray.pos.x, ray.pos.y, ray.pos.z, 1.0f);
   dir = triangle.info.transforms*dir;
   pos = triangle.info.transforms*pos;
   direction.x = dir[0];
   direction.y = dir[1];
   direction.z = dir[2];
   position.x = pos[0];
   position.y = pos[1];
   position.z = pos[2];

   float aa = triangle.a.x - triangle.b.x;
   float bb = triangle.a.y - triangle.b.y;
   float cc = triangle.a.z - triangle.b.z;
   float d = triangle.a.x - triangle.c.x;
   float e = triangle.a.y - triangle.c.y;
   float f = triangle.a.z - triangle.c.z;
   float g = direction.x;
   float h = direction.y;
   float i = direction.z;
   float j = triangle.a.x - position.x;
   float k = triangle.a.y - position.y;
   float l = triangle.a.z - position.z;

   float t = -1;
   float beta = 0;
   float gamma = 0;

   float ei_m_hf = e*i - h*f;
   float gf_m_di = g*f - d*i;
   float dh_m_eg = d*h - e*g;
   float ak_m_jb = aa*k - j*bb;
   float jc_m_al = j*cc - aa*l;
   float bl_m_kc = bb*l - k*cc;
   float M = aa*ei_m_hf + bb*gf_m_di + cc*dh_m_eg;
   if( M  < 0.0001 && M > -0.0001 )
   {
      return -1;
   }
   t = -(f*ak_m_jb + e*jc_m_al + d* bl_m_kc)/M;

   if(t < 0.001)
   {
      return -1;
   }

   gamma = (i*ak_m_jb + h*jc_m_al + g* bl_m_kc)/M;
   if(gamma < 0 || gamma > 1)
   {
      return -1;
   }

   beta = (j*ei_m_hf + k*gf_m_di + l*dh_m_eg)/M;
   if(beta < 0 || beta > (1 - gamma))
   {
      return -1;
   }

   return t;
}
Intersection triangleIntersection( const Triangle &triangle, const Ray &ray, float t )
{
   Intersection ret;

   ret.hit = true;
   vec3 direction = unit( ray.dir );
   ret.viewVector.x = -direction.x;
   ret.viewVector.y = -direction.y;
   ret.viewVector.z = -direction.z;
   ret.hitMark.x = ray.pos.x + direction.x*t;
   ret.hitMark.y = ray.pos.y + direction.y*t;
   ret.hitMark.z = ray.pos.z + direction.z*t;
   ret.normal = triangle.normal;
   ret.colorInfo = triangle.info.colorInfo;
   return ret;
}

Triangle parseTriangle( FILE *file )
{
   Triangle tri;
   char cur = '\0';
   //Point a
   while(cur != '<')
   {
      if(fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error Parsing Triangle\n");
         exit(1);
      }
   }
   if( fscanf(file, " %f , %f , %f ", &(tri.a.x), &(tri.a.y), &(tri.a.z) ) == EOF )
   {
      printf("Error Parsing Triangle\n");
      exit(1);
   }
   printf( "Triangle a: %f %f %f\n", tri.a.x, tri.a.y, tri.a.z );

   cur = '\0';

   //Point b
   while(cur != '<')
   {
      if(fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error Parsing Triangle\n");
         exit(1);
      }
   }
   if( fscanf(file, " %f , %f , %f ", &(tri.b.x), &(tri.b.y), &(tri.b.z) ) == EOF )
   {
      printf("Error Parsing Triangle\n");
      exit(1);
   }
   printf( "Triangle b: %f %f %f\n", tri.b.x, tri.b.y, tri.b.z );

   cur = '\0';

   //Point c
   while(cur != '<')
   {
      if(fscanf(file, "%c", &cur) == EOF)
      {
         printf("Error Parsing Triangle\n");
         exit(1);
      }
   }
   if( fscanf(file, " %f , %f , %f ", &(tri.c.x), &(tri.c.y), &(tri.c.z) ) == EOF )
   {
      printf("Error Parsing Triangle\n");
      exit(1);
   }
   printf( "Triangle c: %f %f %f\n", tri.c.x, tri.c.y, tri.c.z );

   vec3 atob = newDirection(tri.b, tri.a);
   vec3 atoc = newDirection( tri.c, tri.a );
   tri.normal = unit(cross( atob, atoc ));

   tri.info = createObjectInfo();
   parseObjectPigment( file, tri.info );
   parseObjectFinish( file, tri.info );
   parseObjectTransforms( file, tri.info );
   glm::vec4 n = glm::vec4( tri.normal.x, tri.normal.y, tri.normal.z, 1 );

   n = tri.info.transpose * n ;
   tri.normal.x = n[0];
   tri.normal.y = n[1];
   tri.normal.z = n[2];
   tri.normal = unit(tri.normal);

   //Parsing transforms uses up the ending bracket so no need to read to it
   return tri;
}
