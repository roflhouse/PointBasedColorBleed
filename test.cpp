/**
 *  CPE 2013
 *  -------------------
 *  Program
 *
 *  Last Modified:
 *  @author Nick Feeney
 */

#define _POSIX_SOURCE
#define _POSIX_C_SOURCE 199506L
#define _XOPEN_SOURCE 600
#define PI 3.14159
#define SAMPLES 50000

#include <sys/types.h>
#include <unistd.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "Util/RasterCube.h"
#include "Util/Ray.h"
#include "Util/ColorType.h"
#include "Util/Tga.h"
#include "Util/Octree.h"
#include <iostream>
#include <sstream>
#include <string>
#include "Util/Parser.h"
extern "C" int getTime( );
extern "C" float getDiffTime( int start, int end );
extern "C" void testME();
extern "C" void gpuFilloutSphericalHermonics( CudaNode *root, int nodes, SurfelArray &SA,
      int *leaf_addrs, int leaf_nodes );
struct SHSample {
   double sph[3];
   double vec[3];
   double *coeff;
};


int factorial( int x)
{
   int ret = 1;
   for( int i = 1; i <= x; i++ )
      ret *= i;
   return ret;
}
double K(int l, int m)
{
   // renormalisation constant for SH function
   double temp = ((2.0*l+1.0)*factorial(l-m)) / (4.0*PI*factorial(l+m));
   return sqrt(temp);
}
double P(int l,int m,double x)
{
   // evaluate an Associated Legendre Polynomial P(l,m,x) at x
   double pmm = 1.0;
   if(m>0) {
      double somx2 = sqrt((1.0-x)*(1.0+x));
      double fact = 1.0;
      for(int i=1; i<=m; i++) {
         pmm *= (-fact) * somx2;
         fact += 2.0;
      }
   }
   if(l==m) return pmm;
   double pmmp1 = x * (2.0*m+1.0) * pmm;
   if(l==m+1) return pmmp1;
   double pll = 0.0;
   for(int ll=m+2; ll<=l; ++ll) {
      pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m);
      pmm = pmmp1;
      pmmp1 = pll;
   }
   return pll;
}
double SH(int l, int m, double theta, double phi)
{
   // return a point sample of a Spherical Harmonic basis function
   // l is the band, range [0..N]
   // m in the range [-l..l]
   // theta in the range [0..Pi]
   // phi in the range [0..2*Pi]
   const double sqrt2 = sqrt(2.0);
   if(m==0) return K(l,0)*P(l,m,cos(theta));
   else if(m>0) return sqrt2*K(l,m)*cos(m*phi)*P(l,m,cos(theta));
   else return sqrt2*K(l,-m)*sin(-m*phi)*P(l,-m,cos(theta));
}
void SH_project_polar_function(const SHSample samples[],
      double result[], Surfel s)
{
   double area = s.radius * s.radius *PI;
   const double weight = 4.0*PI;
   // for each sample
   for(int i=0; i<SAMPLES; ++i) {
      double theta = samples[i].sph[0];
      double phi = samples[i].sph[1];
      double dx = sin(theta)*cos(phi);
      double dy = sin(theta)*sin(phi);
      double dz = cos(theta);
      double d_dot_n = dx * s.normal.x;
      d_dot_n += dy * s.normal.y;
      d_dot_n += dz * s.normal.z;
      if( d_dot_n > 0.4 )
         for(int n=0; n<16; ++n) {
            result[n] += area * d_dot_n * samples[i].coeff[n];
            result[n+16] += 0*area * d_dot_n * samples[i].coeff[n];
            result[n+16*2] += 0*area * d_dot_n * samples[i].coeff[n];
            result[n+16*3] += area * d_dot_n * samples[i].coeff[n];
         }
   }
   // divide the result by weight and number of samples
   double factor = weight / SAMPLES;
   for(int j=0; j<16; ++j) {
      result[j] = result[j] * factor;
   }
}
void SH_setup_spherical_samples(SHSample samples[], int sqrt_n_samples)
{
   // fill an N*N*2 array with uniformly distributed
   // samples across the sphere using jittered stratification
   int i=0; // array index
   double oneoverN = 1.0/sqrt_n_samples;
   for(int a=0; a<sqrt_n_samples; a++) {
      for(int b=0; b<sqrt_n_samples; b++) {
         // generate unbiased distribution of spherical coords
         double x = (a + random()/(double)RAND_MAX) * oneoverN; // do not reuse results
         double y = (b + random()/(double)RAND_MAX) * oneoverN; // each sample must be random
         double theta = 2.0 * acos(sqrt(1.0 - x));
         double phi = 2.0 * PI * y;
         samples[i].sph[0] = theta;
         samples[i].sph[1] = phi;
         samples[i].sph[2] = 1.0;
         // convert spherical coords to unit vector

         samples[i].vec[0] = sin(theta)*cos(phi);
         samples[i].vec[1] = sin(theta)*sin(phi);
         samples[i].vec[1] = cos(theta);
         samples[i].coeff = (double *) malloc( sizeof(double) * 16 );
         // precompute all SH coefficients for this sample
         for(int l=0; l<4; ++l) {
            for(int m=-l; m<=l; ++m) {
               int index = l*(l+1)+m;
               samples[i].coeff[index] = SH(l,m,theta,phi);
            }
         }
         ++i;
      }
   }
}
void tester( Surfel s, double theta, double phi )
{
   SHSample *samples = (SHSample *)malloc( sizeof(SHSample) * SAMPLES);
   SH_setup_spherical_samples( samples, sqrt(SAMPLES) );
   double result[16];
   for( int i = 0; i < 16; i++ )
      result[i] = 0;
   SH_project_polar_function( samples, result, s );
   double area = 0;
   Color c;
   c.r = 0;
   c.g = 0;
   c.b = 0;
   for(int l=0; l<4; ++l) {
      for(int m=-l; m<=l; ++m) {
         int index = l*(l+1)+m;
         area+=result[index] *SH(l,m,theta,phi);
      }
   }
   printf("Area: %lf\n", area);
}
double *hermonicsTest( Surfel s )
{
   SHSample *samples = (SHSample *)malloc( sizeof(SHSample) * SAMPLES);
   SH_setup_spherical_samples( samples, sqrt(SAMPLES) );
   double *result = (double *) malloc(sizeof(double) *16*4);
   for( int i = 0; i < 16*4; i++ )
      result[i] = 0;
   SH_project_polar_function( samples, result, s );
   return result;
}

void displayRasterCube( RasterCube &cube )
{
   for( int i = 0; i < 6; i++ )
   {
      std::stringstream s;
      s << "Output/side-" << i << ".tga";

      printf("%s\n",s.str().c_str());
      Tga outfile( 8, 8 );
      Color *buffer = outfile.getBuffer();
      for( int j = 0; j < 8; j++ )
      {
         for( int k =0; k < 8; k++ )
         {
            buffer[(7-j)*8 + k] = cube.sides[i][j][k];
         }
      }
      outfile.writeTga( s.str().c_str() );
   }
}
void testRaster()
{
   glm::mat4 vp = getViewPixelMatrix();
   glm::mat4 orth = getOrthMatrix();
   glm::mat4 prot = getProjectMatrix();
   glm::vec4 p = glm::vec4( -1+1, -1+1, 1+1, 1 );
   glm::mat4 eyeTrans = glm::mat4(1.0);
   glm::mat4 *cubetrans;
   initCubeTransforms( &cubetrans );
   eyeTrans[0][3] = -1;
   eyeTrans[1][3] = -1;
   eyeTrans[2][3] = -1;
   glm::mat4 t = glm::mat4(1.0);

   glm::mat4 M = vp * orth * prot * cubetrans[0] * glm::transpose(eyeTrans);

   for( int i = 0; i < 4; i++ )
   {
      for( int j = 0; j < 4; j++ )
         printf("%f ", M[i][j]);
      printf("\n");
   }
   glm::vec4 kc = M * p;
   kc /= kc[3];
   printf("\n\n%f %f %f %f\n", kc[0], kc[1], kc[2], kc[3] );
   printf("%f %f %f %f\n", p[0], p[1], p[2], p[3] );
}
float testEval( Hermonics hermonics, vec3 cen )
{
   double * TYlm = getYLM( cen.x, cen.y, cen.z );
   float area = 0;

   for( int i =0; i < 9; i++ )
   {
      area += hermonics.area[i] * TYlm[i];
   }
   area = fmax( area, 0 );

   return area;
}
Color testColorEval( Hermonics hermonics, vec3 cen )
{
   double * TYlm = getYLM( cen.x, cen.y, cen.z );
   Color color;
   color.r= 0;
   color.g= 0;
   color.b= 0;

   for( int i =0; i < 9; i++ )
   {
      color.r += hermonics.red[i] * TYlm[i];
      color.g += hermonics.green[i] * TYlm[i];
      color.b += hermonics.blue[i] * TYlm[i];
   }
   //if( color.r < 0 )
   //   color.r = -color.r;

   color.r = fmax( color.r, 0 );
   color.g = fmax( color.g, 0 );
   color.b = fmax( color.b, 0 );
   color.r = fmin( color.r, 1.0);
   color.g = fmin( color.g, 1.0);
   color.b = fmin( color.b, 1.0);
   return color;
}
float testEval( double *hermonics, vec3 cen )
{
   double theta = acosf(cen.z);
   double phi = atanf(cen.y/cen.x);
   float area = 0;

   for(int l=0; l<4; ++l) {
      for(int m=-l; m<=l; ++m) {
         int i = l*(l+1)+m;
         area += SH(l,m,theta,phi) * hermonics[i + 16*3];
      }
   }
   area = fmax( area, 0 );

   return area;
}
Color testColorEval( double * hermonics, vec3 cen )
{
   double s = sqrt(cen.x * cen.x + cen.y*cen.y);
   double theta = acosf(cen.z);
   double phi;
   if( cen.x >= 0 )
      phi = asin(cen.y/s);
   else
      phi = PI - asin(cen.y/s);
   Color color;
   color.r= 0;
   color.g= 0;
   color.b= 0;
   for(int l=0; l<4; ++l) {
      for(int m=-l; m<=l; ++m) {
         int i = l*(l+1)+m;
         color.r += SH(l,m,theta,phi) * hermonics[i];
         color.g += SH(l,m,theta,phi) * hermonics[i + 16];
         color.b += SH(l,m,theta,phi) * hermonics[i + 16*2];
      }
   }

   //if( color.r < 0 )
   //   color.r = -color.r;

   color.r = fmax( color.r, 0 );
   color.g = fmax( color.g, 0 );
   color.b = fmax( color.b, 0 );
   color.r = fmin( color.r, 1.0);
   color.g = fmin( color.g, 1.0);
   color.b = fmin( color.b, 1.0);
   return color;
}
int createDrawingRaysTest( Ray **rays, int width, int height )
{
   float rightUnitX = 1;
   float rightUnitY = 0;
   float rightUnitZ = 0;
   float upUnitX = 0;
   float upUnitY = 1;
   float upUnitZ = 0;
   vec3 lookat;
   lookat.x = 0;
   lookat.y = 0;
   lookat.z = 0;
   vec3 pos;
   pos.x =0;
   pos.y =0;
   pos.z = 3;
   vec3 uv = unit( newDirection(lookat, pos) );

   float l = -.5;
   float r = .5;
   float t = .5;
   float b = -0.5;
   *rays = (Ray *) malloc( sizeof(Ray) *height*width );
   for( int i = 0; i < height; i++)
   {
      for( int j = 0; j < width ; j ++ )
      {
         float u = l + (r-l)*( (float)j)/(float)width;
         float v = b + (t-b)*( (float)i)/(float)height;
         float w = 1;
         int c = i*width + j;

         (*rays)[c].pos = pos;
         (*rays)[c].dir.x = u * rightUnitX + v * upUnitX + w * uv.x;
         (*rays)[c].dir.y = u * rightUnitY + v * upUnitY + w * uv.y;
         (*rays)[c].dir.z = u * rightUnitZ + v * upUnitZ + w * uv.z;
         (*rays)[c].dir = unit((*rays)[c].dir);
         (*rays)[c].i = i;
         (*rays)[c].j = j;
      }
   }
   return width * height;
}
void sphereLightingTest16( Surfel *surfel, int num )
{
   Ray *rays;
   double hermonics[16*4];
   for( int j = 0; j < 16*4; j++ )
      hermonics[j] = 0;
   for( int i = 0; i < num; i++ )
   {
      double *temp = hermonicsTest( surfel[i] );
      for( int j = 0; j < 16*4; j++ )
         hermonics[j] += temp[j];
   }

   int width = 400;
   int height = 400;
   int number = createDrawingRaysTest( &rays, width, height );

   Sphere sphere;
   sphere.radius = 1;
   sphere.pos.x = 0;
   sphere.pos.y = 0;
   sphere.pos.z = 0;
   sphere.info.transforms = glm::mat4(1.0);
   float actualArea = sphere.radius * sphere.radius * PI;

   Tga outfile( width, height );
   Color *buffer = outfile.getBuffer();
   Tga outfileArea( width, height );
   Color *bufferArea = outfileArea.getBuffer();

   for( int i = 0; i < number; i++ )
   {
      float_2 hits = sphereHitTest( sphere, rays[i] );
      float use = -1;

      if ( hits.t0 > 0 )
         use = hits.t0;
      else if( hits.t1 > 0 )
         use = hits.t1;
      if( use > 0 )
      {
         vec3 hitmark;
         hitmark.x = rays[i].pos.x + rays[i].dir.x * use;
         hitmark.y = rays[i].pos.y + rays[i].dir.y * use;
         hitmark.z = rays[i].pos.z + rays[i].dir.z * use;

         buffer[rays[i].i*width + rays[i].j] = testColorEval( hermonics, hitmark );
         Color color = testColorEval(hermonics, hitmark);
         color.r = 1;
         color.g = 0;
         color.b = 0;
         //buffer[rays[i].i*width + rays[i].j] = color;
         float area = fmax(testEval( hermonics, hitmark ), 0 );
         bufferArea[rays[i].i*width + rays[i].j].r = fmax(fmin(area/actualArea, 1.0), 0.0);
         bufferArea[rays[i].i*width + rays[i].j].g = fmax(fmin(area/actualArea, 1.0), 0.0);
         bufferArea[rays[i].i*width + rays[i].j].b = fmax(fmin(area/actualArea, 1.0), 0.0);
      }
   }
   outfile.writeTga( "mark16.tga" );
   outfileArea.writeTga( "mark16area.tga" );
}
void sphereLightingTest( Surfel *surfel, int num )
{
   Ray *rays;
   Hermonics hermonics = createHermonics();
   for( int i = 0; i < num; i++ )
   {
      printf("Normal: %f %f %f\n", surfel[i].normal.x, surfel[i].normal.y, surfel[i].normal.z );
      Hermonics temp = calculateSphericalHermonics( surfel[i] ) ;
      addHermonics( hermonics, temp );
   }


   int width = 200;
   int height = 200;
   int number = createDrawingRaysTest( &rays, width, height );

   Sphere sphere;
   sphere.radius = 1;
   sphere.pos.x = 0;
   sphere.pos.y = 0;
   sphere.pos.z = 0;
   sphere.info.transforms = glm::mat4(1.0);
   float actualArea = sphere.radius * sphere.radius * PI;

   Tga outfile( width, height );
   Color *buffer = outfile.getBuffer();
   Tga outfileArea( width, height );
   Color *bufferArea = outfileArea.getBuffer();

   for( int i = 0; i < number; i++ )
   {
      float_2 hits = sphereHitTest( sphere, rays[i] );
      float use = -1;

      if ( hits.t0 > 0 )
         use = hits.t0;
      else if( hits.t1 > 0 )
         use = hits.t1;
      if( use > 0 )
      {
         vec3 hitmark;
         hitmark.x = rays[i].pos.x + rays[i].dir.x * use;
         hitmark.y = rays[i].pos.y + rays[i].dir.y * use;
         hitmark.z = rays[i].pos.z + rays[i].dir.z * use;

         buffer[rays[i].i*width + rays[i].j] = testColorEval( hermonics, hitmark );
         Color color = testColorEval(hermonics, hitmark);
         color.r = 1;
         color.g = 0;
         color.b = 0;
         //buffer[rays[i].i*width + rays[i].j] = color;
         float area = fmax(testEval( hermonics, hitmark ), 0 );
         bufferArea[rays[i].i*width + rays[i].j].r = fmax(fmin(area/actualArea, 1.0), 0.0);
         bufferArea[rays[i].i*width + rays[i].j].g = fmax(fmin(area/actualArea, 1.0), 0.0);
         bufferArea[rays[i].i*width + rays[i].j].b = fmax(fmin(area/actualArea, 1.0), 0.0);
      }
   }
   outfile.writeTga( "lightingtestArray.tga" );
   outfileArea.writeTga( "areatestArray.tga" );
}
void sphereLightingTest( Hermonics hermonics )
{
   Ray *rays;

   int width = 200;
   int height = 200;
   int number = createDrawingRaysTest( &rays, width, height );

   Sphere sphere;
   sphere.radius = 1;
   sphere.pos.x = 0;
   sphere.pos.y = 0;
   sphere.pos.z = 0;
   sphere.info.transforms = glm::mat4(1.0);
   float actualArea = sphere.radius * sphere.radius * PI;

   Tga outfile( width, height );
   Color *buffer = outfile.getBuffer();
   Tga outfileArea( width, height );
   Color *bufferArea = outfileArea.getBuffer();

   for( int i = 0; i < number; i++ )
   {
      float_2 hits = sphereHitTest( sphere, rays[i] );
      float use = -1;

      if ( hits.t0 > 0 )
         use = hits.t0;
      else if( hits.t1 > 0 )
         use = hits.t1;
      if( use > 0 )
      {
         vec3 hitmark;
         hitmark.x = rays[i].pos.x + rays[i].dir.x * use;
         hitmark.y = rays[i].pos.y + rays[i].dir.y * use;
         hitmark.z = rays[i].pos.z + rays[i].dir.z * use;

         buffer[rays[i].i*width + rays[i].j] = testColorEval( hermonics, hitmark );
         //buffer[rays[i].i*width + rays[i].j] = color;
         float area = fmax(testEval( hermonics, hitmark ), 0 );
         bufferArea[rays[i].i*width + rays[i].j].r = fmax(fmin(area/actualArea, 1.0), 0.0);
         bufferArea[rays[i].i*width + rays[i].j].g = fmax(fmin(area/actualArea, 1.0), 0.0);
         bufferArea[rays[i].i*width + rays[i].j].b = fmax(fmin(area/actualArea, 1.0), 0.0);
      }
   }
   outfile.writeTga( "hermonicslightingtest.tga" );
   outfileArea.writeTga( "hermonicsareatest.tga" );
}
void sphereLightingTest( )
{
   Surfel s[2];
   s[0].normal.x = 0;
   s[0].normal.y = 1;
   s[0].normal.z = 0;
   s[0].normal = unit(s[0].normal);
   s[0].radius = .28;
   s[0].color.r = 1.0;
   s[0].color.g = 0.0;
   s[0].color.b = 0.0;
   s[1].normal.x = 0;
   s[1].normal.y = -1;
   s[1].normal.z = 0;
   s[1].normal = unit(s[1].normal);
   s[1].radius = .28;
   s[1].color.r = 1.0;
   s[1].color.g = 0.0;
   s[1].color.b = 0.0;
   sphereLightingTest16( s, 2 );
}
void octreeLightingTest( )
{
   int width = 200;
   int height = 200;
   Ray *rays;
   Sphere sphere;
   sphere.radius = 1;
   sphere.pos.x = 0;
   sphere.pos.y = 0;
   sphere.pos.z = 0;
   sphere.info = createObjectInfo();
   sphere.info.colorInfo.finish_diffuse = .8;
   sphere.info.colorInfo.finish_specular = .2;
   sphere.info.colorInfo.pigment.r = 1;

   PointLight p;
   p.color.r = 1;
   p.color.g = 1;
   p.color.b = 1;
   p.pos.x = 100;
   p.pos.y = 0;
   p.pos.z = -25;

   Scene scene;
   scene.spheres = &sphere;
   scene.pointLights = &p;
   scene.numSpheres = 1;
   scene.numPointLights = 1;

   int number = createDrawingRaysTest( &rays, width, height );
   SurfelArray SA = createSurfelArray();

   for( int i = 0; i < number; i++ )
   {
      float_2 hits = sphereHitTest( sphere, rays[i] );
      float use = -1;
      Intersection inter;

      if ( hits.t0 > 0 )
      {
         vec3 hitmark;
         hitmark.x = rays[i].pos.x + rays[i].dir.x * hits.t0;
         hitmark.y = rays[i].pos.y + rays[i].dir.y * hits.t0;
         hitmark.z = rays[i].pos.z + rays[i].dir.z * hits.t0;
         inter.hitMark = hitmark;
         inter.normal = hitmark;
         inter.viewVector = rays[i].dir;
         inter.hit = hits.t0;
         inter.colorInfo = sphere.info.colorInfo;
         Surfel s = intersectionToSurfel( inter, scene );
         addToSA( SA, s );
      }
      if( hits.t1 > 0 )
      {
         vec3 hitmark;
         hitmark.x = rays[i].pos.x + rays[i].dir.x * hits.t1;
         hitmark.y = rays[i].pos.y + rays[i].dir.y * hits.t1;
         hitmark.z = rays[i].pos.z + rays[i].dir.z * hits.t1;
         inter.hitMark = hitmark;
         inter.normal = hitmark;
         inter.viewVector = rays[i].dir;
         inter.hit = hits.t0;
         inter.colorInfo = sphere.info.colorInfo;
         Surfel s = intersectionToSurfel( inter, scene );
         addToSA( SA, s );
      }
   }
   vec3 min;
   min.x = -1;
   min.y = -1;
   min.z = -1;
   vec3 max;
   max.x = 1;
   max.y = 1;
   max.z = 1;

   //printf("Lighting Test Array\n");
   //sphereLightingTest( SA.array, SA.num );

   printf("Octree\n");
   TreeNode root = createOctreeMark2( SA, min, max );
   Hermonics her = root.children[0]->hermonics;
   addHermonics( her, root.children[7]->hermonics );
   addHermonics( her, root.children[2]->hermonics );
   addHermonics( her, root.children[4]->hermonics );
   addHermonics( her, root.children[6]->hermonics );

   printf("Lighting Test\n");
   sphereLightingTest( her );
}
int width_of_image;
int height_of_image;
char *parseCommandLine(int argc, char *argv[])
{
   if (argc >= 3 )
   {
      if( argv[1][0] == '+' && (argv[1][1] == 'W' || argv[1][1] == 'w') )
      {
         char *temp = &(argv[1][2]);
         std::string tempstring( temp );
         std::stringstream s( tempstring );
         s >> width_of_image;
         if (!s )
         {
            printf("Input Error width unknown\n");
            exit(1);
         }
      }
      if( argv[2][0] == '+' && (argv[2][1] == 'H' || argv[2][1] == 'h') )
      {
         char *temp = &(argv[2][2]);
         std::string tempstring( temp );
         std::stringstream s( tempstring );
         s >> height_of_image;
         if (!s)
         {
            printf("Input Error height unknown\n");
            exit(1);
         }
      }
      if (width_of_image <= 0 || height_of_image <= 0)
      {
         printf("Input Error invalid demenstions, width: %d, height: %d\n", width_of_image, height_of_image);
         exit(1);
      }
      if( argc > 3 )
      {
         return argv[3];
      }
   }
   printf("Error miss use of PBC: PBC +w#### +h#### filename.pov\n");
   exit(EXIT_FAILURE);
}

void sceneLightingTest(int argc, char *argv[])
{
   char *filename = parseCommandLine(argc, argv);
   std::string str(filename);

   Scene scene = parseFile( str );

   Ray *rays;

   int number = createInitRays( &rays, width_of_image, height_of_image, 1.0, scene.camera );
   int size = 0;

   SurfelArray SA = createSurfelArray();

   TreeNode surfels = createSurfelTree( scene, rays, number );
   free( rays );

   number = createDrawingRays( &rays, width_of_image, height_of_image, scene.camera );
   vec3 ***cuberays = initCuberays();
   glm::mat4 *cubetrans;
   initCubeTransforms( &cubetrans );
   Color color;
   color.r = 0;
   color.g = 0;
   color.b = 0;
   RasterCube cube;
   vec3 normal;
   normal.x = 1;
   normal.y = 0;
   normal.z = 0;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            float ndotr = dot(normal, cuberays[i][j][k]);
            if( ndotr < 0.001 )
            {
               cube.sides[i][j][k] = color;
               cube.depth[i][j][k] = -1;
            }
            else {
               cube.sides[i][j][k] = color;
               cube.depth[i][j][k] = 100+1;
            }

         }
   vec3 pos;
   pos.x = -3;
   pos.y = -3;
   pos.z = 0;
   vec3 center;
   center = newDirection(surfels.box.max, surfels.box.min);
   center.x /= 2.0;
   center.y /= 2.0;
   center.z /= 2.0;
   vec3 centerToEye = newDirection( pos, center );
   centerToEye = unit(centerToEye);

   float area = testEval( surfels.hermonics, centerToEye );
   Color c = testColorEval( surfels.hermonics, centerToEye );
   printf("Color: %f %f %f\n", c.r, c.g, c.b );
   printf("Area: %f\n", area );

   rasterizeClusterToCube( cube, c, area, getCenter(surfels.box), cubetrans, cuberays, pos, normal );
   displayRasterCube( cube );
   int num = 0;
   color.r = 0;
   color.g = 0;
   color.b = 0;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            if( cube.depth[i][j][k] < 0 )
               continue;
            num++;
            if( cube.depth[i][j][k] < 100 +1 )
            {
               float dotProd = dot( cuberays[i][j][k], normal );
               if(cube.sides[i][j][k].r > 0 )
                  color.r += cube.sides[i][j][k].r*dotProd;
               if(cube.sides[i][j][k].g > 0 )
                  color.g += cube.sides[i][j][k].g*dotProd;
               if(cube.sides[i][j][k].b > 0 )
                  color.b += cube.sides[i][j][k].b*dotProd;
            }
         }

   if( num > 0 )
   {
      color.r /= (float)num;
      color.g /= (float)num;
      color.b /= (float)num;
   }
   printf("Calculated Point:( %f, %f, %f ): Color: (%f, %f, %f)\n", pos.x, pos.y, pos.z, color.r,
         color.g, color.b );


   sphereLightingTest( surfels.hermonics );
}
void checkTrees( CudaTree *gpu_root, TreeNode *cpu_root, int cur_array, SurfelArray &gpu_array )
{
   if( !equals(gpu_root[cur_array].box, cpu_root->box) )
   {
      vec3 min = gpu_root[cur_array].box.min;
      vec3 max = gpu_root[cur_array].box.max;
      printf("Problem %f %f %f, %f %f %f  ", min.x, min.y, min.z, max.x, max.y, max.z );
      min = cpu_root->box.min;
      max = cpu_root->box.max;
      printf("Problem %f %f %f, %f %f %f\n", min.x, min.y, min.z, max.x, max.y, max.z );
      printf("Current: %d\n", cur_array );
      exit(1);
   }
   if( !gpu_root[cur_array].leaf )
   {
      for( int i = 0; i < 8; i++ )
      {
         checkTrees( gpu_root, cpu_root->children[i], gpu_root[cur_array].children[i], gpu_array );
      }
   }
   else
   {
      for( int i = 0; i < cpu_root->SA.num; i++ )
      {

         if( !equals( cpu_root->SA.array[i], gpu_array.array[gpu_root[cur_array].children[0] + i ] ))
         {
            printf("Surfel\n");
         }
      }
      for( int i = 0; i < 9; i++ )
         //if( fabs(cpu_root->hermonics.area[i] - gpu_root[cur_array].hermonics.area[i]) > 0.001 )
         if( cpu_root->hermonics.area[i] > 0.1 )
            printf("Hermonics %f %f\n", cpu_root->hermonics.area[i], gpu_root[cur_array].hermonics.area[i] );
   }
}
void testOctree(int argc, char *argv[] )
{
   char *filename = parseCommandLine(argc, argv);
   std::string str(filename);

   Scene scene = parseFile( str );

   Ray *rays;

   int number = createInitRays( &rays, width_of_image, height_of_image, 1.0, scene.camera );
   int size = 0;

   vec3 min;
   vec3 max;
   IntersectionArray IA = createIntersectionArray();

   for( int i = 0; i < number; i++ )
   {
      collectIntersections( scene, rays[i], IA );
   }
   shrinkIA( IA );
   SurfelArray SA = createSurfelArray( IA.num );
   for( int i = 0; i < IA.num; i++ )
   {
      if( i == 0 )
      {
         min = IA.array[i].hitMark;
         max = min;
      }
      addToSA( SA, intersectionToSurfel( IA.array[i], scene ) );
      keepMin( min, IA.array[i].hitMark );
      keepMax( max, IA.array[i].hitMark );
   }
   shrinkSA( SA );

   CudaTree *gpu_root = NULL;
   SurfelArray gpu_array;

   int start1 = getTime();
   createCudaTree( SA, min, max, gpu_root, gpu_array );
   int stop1 = getTime();

   int start = getTime();
   TreeNode tree_root =createOctreeMark2( SA, min, max );
   int stop = getTime();


   printf("CPU Time: %f\n", getDiffTime( start, stop ));

   printf("Rasterize Time: %f\n", getDiffTime( start1, stop1) );

   TreeNode *cur_tree = &tree_root;
   int cur_array = 0;

   checkTrees( gpu_root, &tree_root, 0, gpu_array );

   free( rays );
}
int main(int argc, char *argv[])
{
   //octreeLightingTest( );
   testOctree( argc, argv );
   return 0;
   sceneLightingTest(argc, argv);
   return 0;
   vec3 ***cuberays = initCuberays();
   glm::mat4 *cubetrans;
   initCubeTransforms( &cubetrans );
   RasterCube cube;
   Color color;
   color.r = 0;
   color.g = 0;
   color.b = 0;
   color.r = 1;

   //front
   vec3 pos;
   pos.x = 0;
   pos.y = 0;
   pos.z = 0;

   vec3 normal;
   normal.x = 0;
   normal.y = 0;
   normal.z = -1;
   normal = unit(normal);

   Surfel surfel;
   surfel.pos.x = 4;
   surfel.pos.y = 0;
   surfel.pos.z = 0;
   surfel.normal.x = 1;
   surfel.normal.y = 0;
   surfel.normal.z = 0;
   surfel.normal = unit(surfel.normal);
   surfel.color = color;
   surfel.radius = .56419;
   surfel.distance = -dot( surfel.normal, surfel.pos );

   sphereLightingTest( &surfel, 1 );
   return 0;

   color.r = 0;
   for( int i = 0; i <6; i++)
      for( int j = 0; j<8; j++)
         for( int k =0; k<8;k++)
         {
            float ndotr = dot(normal, cuberays[i][j][k]);
            if( ndotr < 0.001 )
            {
               cube.sides[i][j][k] = color;
               cube.depth[i][j][k] = -1;
            }
            else {
               cube.sides[i][j][k] = color;
               cube.depth[i][j][k] = 100+1;
            }

         }
   color.r = 1;
   Hermonics her = calculateSphericalHermonics( surfel );
   vec3 cen = newDirection( pos,surfel.pos );
   cen = unit( cen );
   double theta = acosf(cen.z);
   double phi = atanf(cen.y/cen.x);

   return 0;

   printf("cen:%f %f %f\n", cen.x, cen.y, cen.z );
   float area = testEval( her, cen );
   Color c = testColorEval( her, cen );
   printf("area: %f, %f %f %f, %f %f\n", area, c.r, c.g, c.b, theta, phi );
   tester( surfel, theta, phi );
   printf("act: %f\n", surfel.radius * surfel.radius *PI);
   //rasterizeSurfelToCube( cube, surfel, cubetrans, cuberays, pos, normal );
   rasterizeClusterToCube( cube, c, area, surfel.pos, cubetrans, cuberays, pos, normal );
   displayRasterCube( cube );

   return EXIT_SUCCESS;
}

